#!/usr/bin/env python3
"""Nightshift Grid: one command to run a whole local grid.

    python nightshift.py up --workers 3 --mock      # coordinator + 3 workers + live dashboard
    python nightshift.py submit --wait              # (second terminal) 24-chunk demo job
    python nightshift.py status                     # one-shot table
    python nightshift.py chaos-test --mock          # 4-phase fault benchmark -> Markdown table
    python nightshift.py token                      # print (or create) this grid's tokens
    python nightshift.py down                       # stop a grid whose launcher was killed

Everything an `up` session needs lives in one *state dir* (default `.nightshift/` next to this
file, override with `--state-dir` or $NIGHTSHIFT_STATE_DIR):

    worker.token, client.token   0600 bearer tokens, reused across sessions (separate roles, so a
                                 compromised worker cannot submit jobs or read results)
    session.json                 coordinator URL + child PIDs of the running session
    grid_state.db                coordinator SQLite state (unfinished jobs survive a restart)
    output_graph.jsonl           the GraphRAG sink
    logs/*.log                   one log per process

`submit` / `status` / `token` read the URL and client token from that directory, so `up` in
one terminal and `submit` in another works with zero flags. Tokens are handed to children as
*files*, never on argv (argv is world-readable through `ps`).

Process lifecycle. Each child runs in its own session/process group, so a terminal Ctrl-C reaches
only this launcher, which then stops things in the order the protocol needs: workers first
(SIGTERM makes them hand their leases back while the coordinator is still up), then the
coordinator. Every stop has a timeout and a SIGKILL fallback, an atexit hook repeats it, and
`session.json` records the PIDs so `down` can clean up even if the launcher itself was SIGKILLed.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

HERE = Path(__file__).resolve().parent
# A source checkout keeps its state next to the code (as before); an installed copy (uv tool / pipx)
# lives in site-packages, which is no place for tokens and databases, so it uses ~/.nightshift.
IS_CHECKOUT = (HERE / "pyproject.toml").exists()
DEFAULT_STATE_DIR = Path(os.environ.get("NIGHTSHIFT_STATE_DIR",
                                        HERE / ".nightshift" if IS_CHECKOUT else Path.home() / ".nightshift"))


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("nightshift-grid")
    except Exception:
        return "0.1.0+source"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_PORT = 8000
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
ROLES = ("worker", "client")
TIER_CYCLE = (1, 2, 3)  # node-1 = Tier 1, node-2 = Tier 2, node-3 = Tier 3, node-4 = Tier 1, ...
OUR_SCRIPTS = ("coordinator.py", "worker_daemon.py")


class CliError(RuntimeError):
    """A user-facing failure: printed without a traceback, exit code 2."""


# --------------------------------------------------------------------------- state dir & tokens
def write_secret(path: Path, value: str) -> None:
    """Atomically write `value` to `path` with mode 0600 from the first byte.

    Creating with os.open(..., 0o600) (rather than write-then-chmod) means there is no window
    where another local user could read a freshly generated token."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o600)


@dataclass
class StateDir:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()

    def ensure(self) -> "StateDir":
        self.path.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(self.path, 0o700)  # holds tokens and the job database
        self.logs.mkdir(exist_ok=True)
        return self

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    @property
    def session_file(self) -> Path:
        return self.path / "session.json"

    @property
    def db(self) -> Path:
        return self.path / "grid_state.db"

    @property
    def output(self) -> Path:
        return self.path / "output_graph.jsonl"

    def token_path(self, role: str) -> Path:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        return self.path / f"{role}.token"  # same names coordinator.py --auto-token uses

    def read_token(self, role: str) -> Optional[str]:
        try:
            return self.token_path(role).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def ensure_token(self, role: str, rotate: bool = False) -> str:
        """Reuse the role's token if present (tightening loose permissions), else generate one.

        Reuse matters: remote workers and scripts configured with a token keep working across
        `up` restarts. `rotate=True` is the explicit way to invalidate it."""
        self.ensure()
        path = self.token_path(role)
        token = None if rotate else self.read_token(role)
        if token is None:
            token = secrets.token_urlsafe(32)
            write_secret(path, token)
        elif os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            os.chmod(path, 0o600)
            print(f"[nightshift] tightened permissions on {path} to 0600", file=sys.stderr)
        return token

    def read_session(self) -> Optional[dict]:
        try:
            return json.loads(self.session_file.read_text())
        except (OSError, ValueError):
            return None

    def write_session(self, session: dict) -> None:
        tmp = self.session_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(session, indent=2))
        os.replace(tmp, self.session_file)

    def clear_session(self) -> None:
        self.session_file.unlink(missing_ok=True)


def _loopback_host(host: str) -> bool:
    import ipaddress

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_endpoint(state: StateDir, coordinator: Optional[str], token_file: Optional[str]) -> tuple[str, Optional[str]]:
    """(url, client token) for client commands. Precedence: explicit flags, then the state dir
    written by `up` (the matching pair for the grid it started), then the environment."""
    from grid_auth import CLIENT_TOKEN_ENV, GridAuthError, resolve_token

    session = state.read_session() or {}
    url = coordinator or session.get("url") or f"http://127.0.0.1:{DEFAULT_PORT}"
    if session.get("ca_file") and url == session.get("url"):
        from grid_tls import CA_FILE_ENV, FINGERPRINT_ENV
        if not (os.environ.get(CA_FILE_ENV) or os.environ.get(FINGERPRINT_ENV)):
            os.environ[CA_FILE_ENV] = session["ca_file"]  # the self-signed cert `up` minted; children inherit it
    if token_file:
        try:
            return url, resolve_token(token_file, CLIENT_TOKEN_ENV)[0]
        except GridAuthError as e:
            raise CliError(str(e))
    token = state.read_token("client") or resolve_token(None, CLIENT_TOKEN_ENV)[0]
    return url, token


# --------------------------------------------------------------------------- process teardown
def _is_ours(pid: int, scripts: Sequence[str] = OUR_SCRIPTS) -> bool:
    """True iff `pid` is alive and is one of our scripts (guards `down` against PID reuse)."""
    import psutil

    try:
        cmd = " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        return False
    return any(s in cmd for s in scripts)


def terminate_tree(pids: Iterable[int], timeout: float, sig: int = signal.SIGTERM) -> list[int]:
    """Send `sig` to each PID's process group (and every descendant), wait up to `timeout`
    seconds, then SIGKILL survivors. Returns the PIDs that had to be force-killed.

    Process groups are signalled because a child started with `start_new_session=True` is its
    own group leader, so one killpg also reaches anything it forked (e.g. a uvicorn reloader).
    Descendants are collected *before* signalling so re-parented grandchildren are not missed.
    psutil.wait_procs reaps our own children, so no zombies are left behind."""
    import psutil

    procs: dict[int, psutil.Process] = {}
    leaders = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
        except psutil.Error:
            continue
        leaders.append(p)
        procs[p.pid] = p
        try:
            for c in p.children(recursive=True):
                procs[c.pid] = c
        except psutil.Error:
            pass
    for p in leaders:
        _signal(p, sig)
    for p in procs.values():  # descendants outside the leader's group
        try:
            p.send_signal(sig)
        except psutil.Error:
            pass
    alive = _wait_gone(list(procs.values()), timeout)
    for p in alive:
        _signal(p, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    _wait_gone(alive, 5)
    return [p.pid for p in alive]


def _wait_gone(procs: list, timeout: float) -> list:
    """Wait until every process has exited; return the ones still running at `timeout`.

    A process that has exited but not been reaped yet (a zombie) counts as gone: when `down` runs
    in a separate process from `up`, `up` is the parent and only it can reap them, so waiting for
    a zombie to disappear would wait the whole timeout and then "force-kill" a process that had
    already shut down cleanly. psutil.wait_procs still reaps the ones that are our own children."""
    import psutil

    def running(p) -> bool:
        try:
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    deadline = time.monotonic() + timeout
    pending = [p for p in procs if running(p)]
    while pending and time.monotonic() < deadline:
        _, still = psutil.wait_procs(pending, timeout=min(0.2, max(0.0, deadline - time.monotonic())))
        pending = [p for p in still if running(p)]
    return pending


def _signal(p, sig: int) -> None:
    import psutil

    try:
        if hasattr(os, "killpg") and os.getpgid(p.pid) == p.pid:
            os.killpg(p.pid, sig)
        else:
            p.send_signal(sig)
    except (ProcessLookupError, PermissionError, psutil.Error, OSError):
        pass


@dataclass
class Child:
    name: str
    proc: subprocess.Popen
    log: Path


@dataclass
class Supervisor:
    """Owns the coordinator + worker processes of one `up` session."""
    state: StateDir
    workers: list[Child] = field(default_factory=list)
    coordinator: Optional[Child] = None
    worker_timeout: float = 10.0
    coordinator_timeout: float = 5.0
    _stopped: bool = False

    def spawn(self, name: str, argv: list[str], env: Optional[dict] = None) -> Child:
        log = self.state.logs / f"{name}.log"
        fh = open(log, "a", buffering=1)
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True  # terminal Ctrl-C reaches only us; we order the shutdown
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})}, **kwargs)
        fh.close()  # the child holds its own descriptor
        return Child(name, proc, log)

    def children(self) -> list[Child]:
        return self.workers + ([self.coordinator] if self.coordinator else [])

    def record_session(self, **extra) -> None:
        self.state.write_session({"pid": os.getpid(), "started_at": time.time(),
                                  "children": {c.name: c.proc.pid for c in self.children()}, **extra})

    def teardown(self) -> None:
        """Workers first (graceful lease hand-back needs a live coordinator), then coordinator.
        Idempotent: called from `finally` and again from atexit."""
        if self._stopped:
            return
        self._stopped = True
        old = {s: signal.getsignal(s) for s in _stop_signals()}
        for s in old:  # a second Ctrl-C must not abort teardown halfway and orphan children
            signal.signal(s, signal.SIG_IGN)
        try:
            live = [c for c in self.workers if c.proc.poll() is None]
            if live:
                print(f"[nightshift] stopping {len(live)} worker(s) (leases are handed back)...", file=sys.stderr)
                forced = terminate_tree([c.proc.pid for c in live], self.worker_timeout)
                if forced:
                    print(f"[nightshift] force-killed worker pid(s) {forced}", file=sys.stderr)
            if self.coordinator and self.coordinator.proc.poll() is None:
                print("[nightshift] stopping coordinator...", file=sys.stderr)
                terminate_tree([self.coordinator.proc.pid], self.coordinator_timeout)
            for c in self.children():
                c.proc.poll()  # sync Popen's returncode after psutil reaped it
            self.state.clear_session()
        finally:
            for s, h in old.items():
                try:
                    signal.signal(s, h)
                except (ValueError, TypeError):
                    pass


def _stop_signals() -> list[int]:
    return [s for s in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP")) if s is not None]


def live_session_pids(state: StateDir) -> dict[str, int]:
    session = state.read_session() or {}
    return {name: pid for name, pid in (session.get("children") or {}).items() if _is_ours(pid)}


# --------------------------------------------------------------------------- preflight helpers
def help_text(script: str) -> str:
    try:
        return subprocess.run([sys.executable, str(HERE / script), "--help"], capture_output=True, text=True,
                              timeout=30, cwd=HERE).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def has_flag(help_out: str, flag: str) -> bool:
    return any(tok.strip("[],") == flag for tok in help_out.split())


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def ollama_preflight(ollama_url: str, model: str) -> None:
    """Fail fast with an actionable message instead of N workers each dying on their own."""
    import httpx

    try:
        tags = httpx.get(f"{ollama_url}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError) as e:
        raise CliError(f"Ollama is not reachable at {ollama_url} ({type(e).__name__}). Start it with "
                       f"`ollama serve`, or try the grid without a model: nightshift.py up --mock")
    names = {m.get("name") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise CliError(f"Model '{model}' is not pulled in Ollama (have: {', '.join(sorted(n for n in names if n)) or 'none'}). "
                       f"Run `ollama pull {model}`, pick one with --model, or use --mock.")


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def worker_specs(n: int) -> list[tuple[str, int]]:
    return [(f"node-{i}", TIER_CYCLE[(i - 1) % len(TIER_CYCLE)]) for i in range(1, n + 1)]


# --------------------------------------------------------------------------- commands
def cmd_up(a: argparse.Namespace) -> int:
    import httpx

    state = StateDir(a.state_dir).ensure()
    running = live_session_pids(state)
    if running:
        raise CliError(f"a grid from {state.path} is already running ({', '.join(f'{k}={v}' for k, v in running.items())}). "
                       f"Stop it with `nightshift.py down --state-dir {state.path}` or use another --state-dir.")
    if not port_free(a.host, a.port):
        raise CliError(f"port {a.port} on {a.host} is in use; pick another with --port")
    if not a.mock:
        ollama_preflight(a.ollama, a.model)
    if a.fresh:
        for f in (state.db, Path(f"{state.db}-wal"), Path(f"{state.db}-shm"), state.output):
            f.unlink(missing_ok=True)
    state.ensure_token("worker")
    state.ensure_token("client")
    client_token = state.read_token("client")

    # A LAN-reachable grid serves TLS (the coordinator refuses plaintext there); local clients verify
    # it against the self-signed cert in the state dir, which also carries a 127.0.0.1 SAN.
    tls = not _loopback_host(a.host)
    scheme = "https" if tls else "http"
    url = f"{scheme}://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    cert_file = state.path / "coordinator-cert.pem"
    session_extra = {"ca_file": str(cert_file)} if tls else {}
    sup = Supervisor(state)
    atexit.register(sup.teardown)
    # SIGTERM/SIGHUP behave like Ctrl-C so `finally` runs teardown. SIGINT is (re)installed too:
    # a shell starts background jobs with SIGINT ignored, and Python keeps an inherited SIG_IGN.
    for s in _stop_signals():
        signal.signal(s, signal.default_int_handler)

    coord_help = help_text("coordinator.py")
    try:
        argv = ["coordinator.py", "--host", a.host, "--port", str(a.port), "--db", str(state.db),
                "--output", str(state.output), "--worker-token-file", str(state.token_path("worker")),
                "--client-token-file", str(state.token_path("client"))]
        if has_flag(coord_help, "--no-advertise") and not a.advertise:
            argv.append("--no-advertise")
        if tls:
            argv.append("--auto-tls")
        sup.coordinator = sup.spawn("coordinator", argv)
        sup.record_session(url=url, host=a.host, port=a.port, mode="mock" if a.mock else a.model, **session_extra)
        if tls:
            from grid_tls import CA_FILE_ENV
            _wait(cert_file.exists, 20, f"TLS certificate {cert_file}", sup)
            os.environ[CA_FILE_ENV] = str(cert_file)  # inherited by workers and used by our own requests
        from grid_tls import client_verify
        _wait(lambda: httpx.get(f"{url}/healthz", timeout=1, verify=client_verify(url)).status_code == 200,
              20, "coordinator /healthz", sup)

        from grid_auth import derive_worker_credential, write_secret_file
        cred_dir = state.path / "credentials"
        cred_dir.mkdir(mode=0o700, exist_ok=True)
        master = state.read_token("worker")

        for wid, tier in worker_specs(a.workers):
            # Each local worker gets a credential bound to its own id, never the master worker token.
            cred = cred_dir / f"{wid}.credential"
            write_secret_file(str(cred), derive_worker_credential(master, wid), force=True)
            wargv = ["worker_daemon.py", "--id", wid, "--tier", str(tier), "--coordinator", url,
                     "--token-file", str(cred)]
            wargv += ["--mock"] if a.mock else ["--model", a.model, "--ollama", a.ollama]
            if a.allow_battery:
                wargv.append("--allow-battery")
            for extra in a.worker_arg:
                wargv += shlex.split(extra)
            sup.workers.append(sup.spawn(wid, wargv))
        sup.record_session(url=url, host=a.host, port=a.port, mode="mock" if a.mock else a.model, **session_extra)

        ids = {wid for wid, _ in worker_specs(a.workers)}
        headers = {"Authorization": f"Bearer {client_token}"}
        _wait(lambda: ids <= {w["worker_id"] for w in httpx.get(f"{url}/v1/grid/status", headers=headers, timeout=2,
                                                                   verify=client_verify(url)).json().get("workers", [])},
              a.ready_timeout, f"{a.workers} worker registration(s)", sup)

        sd = "" if state.path == DEFAULT_STATE_DIR.resolve() else f" --state-dir {_display(state.path)}"
        tips = (f"submit: python nightshift.py submit --wait{sd}    status: python nightshift.py status{sd}\n"
                f"chaos:  touch /tmp/preempt_node-2  (node-2's 'user' returns; rm it to release)    stop: Ctrl-C")
        print(f"\nNightshift grid up: coordinator {url}, {a.workers} worker(s) "
              f"({'mock' if a.mock else a.model}); logs in {_display(state.logs)}/\n{tips}\n", flush=True)

        def coordinator_dead() -> bool:
            for c in sup.workers:  # reap workers that exited (e.g. stopped by `down`): no zombies
                c.proc.poll()
            return sup.coordinator.proc.poll() is not None

        if a.no_dashboard:
            while not coordinator_dead():
                time.sleep(0.5)
        else:
            from dashboard import run_dashboard
            run_dashboard(url, client_token, should_stop=coordinator_dead, footer=tips)
        if coordinator_dead():
            print(f"[nightshift] coordinator exited (code {sup.coordinator.proc.returncode}); log tail:\n"
                  f"{tail(sup.coordinator.log)}", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("\n[nightshift] shutting down...", file=sys.stderr)
    finally:
        sup.teardown()
    return 0


def _display(p: Path) -> str:
    """Path relative to the cwd when it is underneath it (shorter, copy-pasteable tips)."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _wait(pred, timeout: float, what: str, sup: Supervisor) -> None:
    """Poll `pred` until true; fail early (with the log tail) if a supervised process dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for c in sup.children():
            if c.proc.poll() is not None:
                raise CliError(f"{c.name} exited with code {c.proc.returncode} while waiting for {what}. "
                               f"Last lines of {c.log}:\n{tail(c.log)}")
        try:
            if pred():
                return
        except Exception:  # not up yet
            pass
        time.sleep(0.2)
    raise CliError(f"timed out after {timeout:.0f}s waiting for {what}; see logs in {sup.state.logs}")


def cmd_down(a: argparse.Namespace) -> int:
    state = StateDir(a.state_dir)
    running = live_session_pids(state)
    if not running:
        print("no running grid recorded in", state.path)
        state.clear_session()
        return 0
    workers = [pid for name, pid in running.items() if name != "coordinator"]
    forced = terminate_tree(workers, 10.0)
    if "coordinator" in running:
        forced += terminate_tree([running["coordinator"]], 5.0)
    state.clear_session()
    print(f"stopped {len(running)} process(es)" + (f" (force-killed {forced})" if forced else ""))
    return 0


def _client(a: argparse.Namespace):
    import httpx

    from grid_tls import TLSConfigError, client_verify

    url, token = resolve_endpoint(StateDir(a.state_dir), a.coordinator, a.token_file)
    try:
        verify = client_verify(url)
    except TLSConfigError as e:
        raise CliError(str(e))
    except OSError as e:
        raise CliError(f"no coordinator reachable at {url} ({type(e).__name__}). Start one with `python nightshift.py up`.")
    return url, token, httpx.Client(base_url=url, timeout=10, verify=verify,
                                    headers={"Authorization": f"Bearer {token}"} if token else {})


def _get(client, path: str) -> dict:
    import httpx

    try:
        r = client.get(path)
    except httpx.HTTPError as e:
        raise CliError(f"no coordinator reachable at {client.base_url} ({type(e).__name__}). "
                       f"Start one with `python nightshift.py up`.")
    if r.status_code in (401, 403):
        raise CliError(f"coordinator rejected the client token (HTTP {r.status_code}). Is --state-dir the one "
                       f"`up` used, or pass --token-file?")
    r.raise_for_status()
    return r.json()


def wait_job(client, job_id: str, timeout: float, quiet: bool = False) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        st = _get(client, f"/v1/grid/jobs/{job_id}")
        line = f"  {st['counts']['COMPLETED']}/{st['total']} completed, {st['counts']['LEASED']} in flight"
        if not quiet and line != last and sys.stdout.isatty():
            print("\r" + line, end="", flush=True)
            last = line
        if st["done"]:
            if last:
                print()
            return st
        time.sleep(0.5)
    raise CliError(f"job {job_id} not done after {timeout:.0f}s")


def cmd_submit(a: argparse.Namespace) -> int:
    url, token, client = _client(a)
    state = StateDir(a.state_dir)
    if a.dir:
        argv = [sys.executable, str(HERE / "ingest.py"), *a.dir, "--coordinator", url,
                "--manifest-dir", str(state.path / "manifests")]
        if a.tier is not None:
            argv += ["--tier", str(a.tier)]
        if a.job_name:
            argv += ["--job-name", a.job_name]
        if a.wait:
            argv.append("--wait")
        env = {**os.environ, "GRID_CLIENT_TOKEN": token or ""}  # env, not argv: tokens stay out of `ps`
        return subprocess.call(argv, cwd=os.getcwd(), env=env)

    if a.file:
        try:
            chunks = [l.strip() for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
        except OSError as e:
            raise CliError(f"cannot read {a.file}: {e.strerror}")
        if not chunks:
            raise CliError(f"{a.file} has no non-empty lines")
        name = a.job_name or f"{Path(a.file).name} ({len(chunks)} chunks)"
    else:
        from demo_submit import SAMPLE_CHUNKS
        chunks, name = SAMPLE_CHUNKS, a.job_name or f"Industrial Maintenance GraphRAG demo ({len(SAMPLE_CHUNKS)} chunks)"

    import httpx
    try:
        r = client.post("/v1/grid/mapreduce", json={"job_name": name, "chunks": chunks,
                                                     "required_tier": a.tier if a.tier is not None else 1})
    except httpx.HTTPError as e:
        raise CliError(f"no coordinator reachable at {url} ({type(e).__name__}). Start one with `python nightshift.py up`.")
    if r.status_code in (401, 403):
        raise CliError(f"coordinator rejected the client token (HTTP {r.status_code}); check --state-dir / --token-file")
    r.raise_for_status()
    job = r.json()
    print(f"submitted job {job['job_id']}: {len(chunks)} chunks -> {url}")
    if a.wait:
        st = wait_job(client, job["job_id"], a.timeout)
        print(f"done in {st['elapsed_s']:.1f}s: {st['counts']['COMPLETED']}/{st['total']} completed, "
              f"attempts={st['total_attempts']} preemptions={st['total_preemptions']} failures={st['total_failures']}")
        return 0 if st["counts"]["FAILED"] == 0 else 1
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    from rich.console import Console

    from dashboard import header_text, jobs_table, workers_table

    url, _, client = _client(a)
    data = _get(client, "/v1/grid/status")
    if a.json:
        print(json.dumps(data, indent=2))
        return 0
    con = Console()
    con.print(f"{header_text(data)} | {url}")
    con.print(workers_table(data))
    if data.get("jobs"):
        con.print(jobs_table(data))
    return 0


def chaos_argv(a: argparse.Namespace) -> list[str]:
    argv = [sys.executable, str(HERE / "chaos_test.py"), "--markdown", "--port", str(a.port)]
    argv += ["--mock"] if a.mock else ["--model", a.model]
    if a.markdown_out:
        argv += ["--markdown-out", a.markdown_out]
    if a.phases:
        argv += ["--phases", a.phases]
    if a.report:
        argv += ["--report", a.report]
    return argv


def cmd_chaos(a: argparse.Namespace) -> int:
    if not a.mock:
        ollama_preflight(a.ollama, a.model)
    return subprocess.call(chaos_argv(a), cwd=HERE)


def cmd_token(a: argparse.Namespace) -> int:
    state = StateDir(a.state_dir)
    roles = ROLES if a.role == "both" else (a.role,)
    if a.rotate and live_session_pids(state):
        print("[nightshift] warning: a grid is running with the old tokens; restart it (`down` + `up`)",
              file=sys.stderr)
    tokens = {r: state.ensure_token(r, rotate=a.rotate) for r in roles}
    if a.role != "both":
        print(tokens[a.role])
        return 0
    for r, t in tokens.items():
        print(f"{r:<7} {t}    ({state.token_path(r)})")
    print(f"\nJoin a remote worker (coordinator started with --host 0.0.0.0, which serves TLS):\n"
          f"  python grid_auth.py issue --id node-x --worker-token-file {state.token_path('worker')} --out node-x.credential\n"
          f"  # copy node-x.credential to the worker machine, then on it:\n"
          f"  python worker_daemon.py --token-file node-x.credential --discover")
    return 0


# --------------------------------------------------------------------------- argument parsing
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nightshift", description="Nightshift Grid single-command CLI",
                                 allow_abbrev=False)
    ap.add_argument("--version", action="version", version=f"nightshift-grid {_version()}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="{up,submit,status,chaos-test,token,down}")

    def state_arg(p):
        p.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR,
                       help=f"tokens, session, db, logs (default {DEFAULT_STATE_DIR}; env NIGHTSHIFT_STATE_DIR)")

    def client_args(p):
        state_arg(p)
        p.add_argument("--coordinator", default=None, help="URL (default: from the state dir's session)")
        p.add_argument("--token-file", default=None, metavar="PATH", help="client token file (default: state dir)")

    up = sub.add_parser("up", help="start coordinator + N local workers + live dashboard", allow_abbrev=False)
    up.add_argument("--workers", type=int, default=3)
    mode = up.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="simulated inference, no Ollama needed")
    mode.add_argument("--model", default=DEFAULT_MODEL)
    up.add_argument("--ollama", default=DEFAULT_OLLAMA)
    up.add_argument("--host", default="127.0.0.1", help="coordinator bind address (0.0.0.0 lets LAN workers join)")
    up.add_argument("--port", type=int, default=DEFAULT_PORT)
    up.add_argument("--no-dashboard", action="store_true", help="run headless (block until Ctrl-C / SIGTERM)")
    up.add_argument("--fresh", action="store_true", help="discard the previous session's job database and output")
    up.add_argument("--advertise", action="store_true", help="advertise the coordinator via mDNS (if supported)")
    up.add_argument("--allow-battery", action="store_true", help="let local workers harvest on battery power")
    up.add_argument("--worker-arg", action="append", default=[], metavar="ARGS",
                    help="extra worker_daemon.py args, e.g. --worker-arg='--max-host-cpu-pct 80' (repeatable)")
    up.add_argument("--ready-timeout", type=float, default=60.0)
    state_arg(up)
    up.set_defaults(func=cmd_up)

    s = sub.add_parser("submit", help="submit the 24-chunk demo job, a file of lines, or a directory",
                       allow_abbrev=False)
    src = s.add_mutually_exclusive_group()
    src.add_argument("--file", help="one chunk per non-empty line")
    src.add_argument("--dir", nargs="+", metavar="PATH", help="files/directories to chunk via ingest.py")
    s.add_argument("--tier", type=int, default=None, help="required tier (demo/file default 1, --dir default 2)")
    s.add_argument("--job-name", default=None)
    s.add_argument("--wait", action="store_true", help="block until the job finishes")
    s.add_argument("--timeout", type=float, default=900.0)
    client_args(s)
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status", help="one-shot grid status table", allow_abbrev=False)
    st.add_argument("--json", action="store_true", help="raw /v1/grid/status JSON")
    client_args(st)
    st.set_defaults(func=cmd_status)

    c = sub.add_parser("chaos-test", help="run the 4-phase chaos benchmark and print a Markdown table",
                       allow_abbrev=False)
    cmode = c.add_mutually_exclusive_group()
    cmode.add_argument("--mock", action="store_true")
    cmode.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--ollama", default=DEFAULT_OLLAMA)
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--phases", default=None, help="e.g. 1,4 (default all)")
    c.add_argument("--markdown-out", default=None, metavar="FILE")
    c.add_argument("--report", default=None, help="JSON report path (chaos_test.py default chaos_report.json)")
    c.set_defaults(func=cmd_chaos)

    t = sub.add_parser("token", help="print this grid's tokens (generating them if missing)", allow_abbrev=False)
    t.add_argument("--role", choices=(*ROLES, "both"), default="both", help="a single role prints just the token")
    t.add_argument("--rotate", action="store_true", help="replace with a fresh token")
    state_arg(t)
    t.set_defaults(func=cmd_token)

    d = sub.add_parser("down", help="stop a grid recorded in the state dir (e.g. after the launcher was killed)",
                       allow_abbrev=False)
    state_arg(d)
    d.set_defaults(func=cmd_down)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "workers", 1) < 1:
        print("nightshift: --workers must be >= 1", file=sys.stderr)
        return 2
    try:
        return a.func(a)
    except CliError as e:
        print(f"nightshift: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
