"""nightshift.py: argument parsing, state-dir tokens, process-group teardown, chaos passthrough.

No coordinator, worker or Ollama is started here; teardown is exercised on throwaway `sleep`
processes so the orphan-prevention logic is tested for real, not mocked."""
import os
import signal
import stat
import subprocess
import sys
import time

import psutil
import pytest

import nightshift as ns


# ---------------------------------------------------------------- parsing
def test_up_defaults():
    a = ns.build_parser().parse_args(["up"])
    assert (a.workers, a.mock, a.model, a.port, a.host, a.no_dashboard) == (3, False, ns.DEFAULT_MODEL, 8000,
                                                                            "127.0.0.1", False)
    assert a.func is ns.cmd_up


def test_up_mock_and_model_are_exclusive(capsys):
    with pytest.raises(SystemExit):
        ns.build_parser().parse_args(["up", "--mock", "--model", "x"])


def test_up_no_abbreviations():
    # `--work 3` silently meaning `--workers 3` (or `--worker-arg`) would be a footgun.
    with pytest.raises(SystemExit):
        ns.build_parser().parse_args(["up", "--work", "3"])


def test_workers_must_be_positive(capsys):
    assert ns.main(["up", "--workers", "0"]) == 2


def test_submit_sources_are_exclusive():
    p = ns.build_parser()
    assert p.parse_args(["submit", "--wait"]).wait
    assert p.parse_args(["submit", "--dir", "a", "b"]).dir == ["a", "b"]
    with pytest.raises(SystemExit):
        p.parse_args(["submit", "--file", "x", "--dir", "y"])


def test_worker_specs_cycle_tiers():
    assert ns.worker_specs(4) == [("node-1", 1), ("node-2", 2), ("node-3", 3), ("node-4", 1)]


def test_has_flag_matches_whole_tokens():
    out = "usage: x [--advertise | --no-advertise]\n  --advertise, --no-advertise\n  --max-host-cpu-pct P"
    assert ns.has_flag(out, "--no-advertise")
    assert ns.has_flag(out, "--max-host-cpu-pct")
    assert not ns.has_flag(out, "--max-host")


def test_chaos_argv_always_requests_markdown():
    p = ns.build_parser()
    argv = ns.chaos_argv(p.parse_args(["chaos-test", "--mock", "--markdown-out", "b.md", "--phases", "1,4"]))
    assert argv[1].endswith("chaos_test.py")
    assert "--markdown" in argv and "--mock" in argv and "--model" not in argv
    assert argv[argv.index("--markdown-out") + 1] == "b.md"
    assert argv[argv.index("--phases") + 1] == "1,4"
    live = ns.chaos_argv(p.parse_args(["chaos-test", "--model", "qwen3:4b"]))
    assert live[live.index("--model") + 1] == "qwen3:4b" and "--mock" not in live


# ---------------------------------------------------------------- state dir / tokens
def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_tokens_created_0600_and_reused(tmp_path):
    sd = ns.StateDir(tmp_path / "st")
    w = sd.ensure_token("worker")
    c = sd.ensure_token("client")
    assert w != c and len(w) >= 32
    assert _mode(sd.token_path("worker")) == 0o600 and _mode(sd.token_path("client")) == 0o600
    assert _mode(sd.path) == 0o700
    assert sd.ensure_token("worker") == w  # reuse across sessions
    assert ns.StateDir(tmp_path / "st").read_token("client") == c


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_loose_token_permissions_are_tightened(tmp_path, capsys):
    sd = ns.StateDir(tmp_path).ensure()
    sd.token_path("client").write_text("keepme\n")
    os.chmod(sd.token_path("client"), 0o644)
    assert sd.ensure_token("client") == "keepme"
    assert _mode(sd.token_path("client")) == 0o600
    assert "tightened" in capsys.readouterr().err


def test_rotate_and_empty_file_regenerate(tmp_path):
    sd = ns.StateDir(tmp_path).ensure()
    first = sd.ensure_token("worker")
    assert sd.ensure_token("worker", rotate=True) != first
    sd.token_path("client").write_text("  \n")
    assert sd.ensure_token("client").strip()


def test_token_role_validation(tmp_path):
    with pytest.raises(ValueError):
        ns.StateDir(tmp_path).token_path("admin")


def test_token_command_prints_single_role(tmp_path, capsys):
    assert ns.main(["token", "--role", "client", "--state-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == ns.StateDir(tmp_path).read_token("client")


def test_resolve_endpoint_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_CLIENT_TOKEN", "from-env")
    monkeypatch.delenv("GRID_TOKEN", raising=False)
    sd = ns.StateDir(tmp_path).ensure()
    # Nothing in the state dir: default URL, env token.
    assert ns.resolve_endpoint(sd, None, None) == ("http://127.0.0.1:8000", "from-env")
    # `up` wrote a session and tokens: those win over env, so two-terminal use is zero-config.
    tok = sd.ensure_token("client")
    sd.write_session({"url": "http://127.0.0.1:8850", "children": {}})
    assert ns.resolve_endpoint(sd, None, None) == ("http://127.0.0.1:8850", tok)
    # Explicit flags win over everything.
    f = tmp_path / "other.token"
    f.write_text("explicit\n")
    os.chmod(f, 0o600)
    assert ns.resolve_endpoint(sd, "http://h:1", str(f)) == ("http://h:1", "explicit")
    with pytest.raises(ns.CliError):
        ns.resolve_endpoint(sd, None, str(tmp_path / "missing"))


def test_live_session_ignores_foreign_pids(tmp_path):
    sd = ns.StateDir(tmp_path).ensure()
    # Our own pid is alive but is not coordinator.py/worker_daemon.py (simulates PID reuse).
    sd.write_session({"children": {"coordinator": os.getpid(), "node-1": 999_999_9}})
    assert ns.live_session_pids(sd) == {}
    assert ns.main(["down", "--state-dir", str(tmp_path)]) == 0
    assert psutil.pid_exists(os.getpid())


# ---------------------------------------------------------------- teardown
SLEEPER = "import time\nwhile True: time.sleep(0.1)\n"
STUBBORN = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(0.1)\n"
PARENT = ("import subprocess, sys, time\n"
          "c = subprocess.Popen([sys.executable, '-c', 'import time\\nwhile True: time.sleep(0.1)'])\n"
          "print(c.pid, flush=True)\nwhile True: time.sleep(0.1)\n")


def _spawn(code, **kw):
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True, **kw)


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_terminate_tree_kills_group_including_grandchild():
    parent = _spawn(PARENT, stdout=subprocess.PIPE, text=True)
    grandchild = int(parent.stdout.readline())
    assert psutil.pid_exists(grandchild)
    forced = ns.terminate_tree([parent.pid], timeout=5)
    assert forced == []
    parent.wait(5)
    deadline = time.time() + 5
    while psutil.pid_exists(grandchild) and time.time() < deadline:
        try:
            if psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE


@pytest.mark.skipif(os.name != "posix", reason="signals")
def test_terminate_tree_falls_back_to_sigkill():
    p = _spawn(STUBBORN)
    time.sleep(0.3)  # let it install the SIGTERM handler
    t0 = time.time()
    assert ns.terminate_tree([p.pid], timeout=0.5) == [p.pid]
    p.wait(5)  # psutil already reaped it, so Popen only learns that it is gone
    assert not psutil.pid_exists(p.pid) or psutil.Process(p.pid).status() == psutil.STATUS_ZOMBIE
    assert time.time() - t0 < 5


def test_terminate_tree_tolerates_dead_pids():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert ns.terminate_tree([p.pid, 999_999_9], timeout=0.2) == []


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_supervisor_stops_workers_before_coordinator(tmp_path, monkeypatch):
    sd = ns.StateDir(tmp_path).ensure()
    sup = ns.Supervisor(sd)
    sup.coordinator = ns.Child("coordinator", _spawn(SLEEPER), tmp_path / "c.log")
    sup.workers = [ns.Child(f"node-{i}", _spawn(SLEEPER), tmp_path / f"{i}.log") for i in (1, 2)]
    sup.record_session(url="http://x")
    order = []
    real = ns.terminate_tree

    def spy(pids, timeout, sig=signal.SIGTERM):
        pids = list(pids)
        order.append(pids)
        # When the coordinator is signalled, every worker must already be gone.
        if pids == [sup.coordinator.proc.pid]:
            assert all(w.proc.poll() is not None for w in sup.workers)
        return real(pids, timeout, sig)

    monkeypatch.setattr(ns, "terminate_tree", spy)
    sup.teardown()
    sup.teardown()  # idempotent (atexit calls it again)
    assert order == [[w.proc.pid for w in sup.workers], [sup.coordinator.proc.pid]]
    assert all(c.proc.poll() is not None for c in sup.children())
    assert not sd.session_file.exists()
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN  # handlers restored


# ---------------------------------------------------------------- preflight
def test_ollama_preflight_unreachable_is_actionable():
    with pytest.raises(ns.CliError, match="--mock"):
        ns.ollama_preflight("http://127.0.0.1:9", "qwen3:4b")


def test_ollama_preflight_missing_model(monkeypatch):
    import httpx

    class R:
        def json(self):
            return {"models": [{"name": "llama3.2:3b"}, {"name": "qwen3:latest"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    with pytest.raises(ns.CliError, match="ollama pull qwen3:4b"):
        ns.ollama_preflight("http://o", "qwen3:4b")
    ns.ollama_preflight("http://o", "llama3.2:3b")
    ns.ollama_preflight("http://o", "qwen3")  # ':latest' tag implied


def test_port_free_detects_listener():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    try:
        assert not ns.port_free("127.0.0.1", s.getsockname()[1])
    finally:
        s.close()


def test_status_without_coordinator_is_clean_error(tmp_path, capsys):
    assert ns.main(["status", "--state-dir", str(tmp_path), "--coordinator", "http://127.0.0.1:9"]) == 2
    assert "nightshift.py up" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="zombies are a POSIX concept")
def test_wait_gone_treats_unreaped_zombies_as_exited():
    """`down` runs in a different process than `up`, so a worker that exited cleanly stays a zombie
    until `up` reaps it. That must count as gone, not trigger a 10 s wait and a bogus force-kill."""
    code = ("import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','pass']);"
            "print(p.pid,flush=True);time.sleep(30)")  # never waits on p -> p becomes a zombie
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        zpid = int(parent.stdout.readline())
        z = psutil.Process(zpid)
        for _ in range(50):
            if z.status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        start = time.monotonic()
        assert ns._wait_gone([z], timeout=5) == []
        assert time.monotonic() - start < 1.0
    finally:
        parent.kill()
        parent.wait()
