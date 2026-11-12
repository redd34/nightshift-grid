"""End-to-end chaos test: Coordinator + 3 heterogeneous workers, real processes, real HTTP, auth on.

Phase 1 - Good-citizen preemption: submit 24 chunks, wait until node-legacy-laptop holds a
          lease mid-stream, `touch /tmp/preempt_node-legacy-laptop`, and measure
          detect -> reclaim -> re-lease -> recovered-chunk-complete latencies.
Phase 2 - Worker crash: SIGKILL node-office-pc mid-lease (no goodbye) and measure lease-expiry
          recovery.
Phase 3 - Coordinator crash: SIGKILL the coordinator mid-job while chunks are in flight,
          restart it on the same SQLite state file, and verify the job finishes - in-flight
          leases should be completed by the workers that held them (no wasted work).
Phase 4 - Host CPU spike ("the user started a compile"): once every worker is leasing, spawn
          one busy-loop process per core. Every worker shares this host, so every worker's
          foreign-CPU guard should trip: measure hog-start -> `cpu_busy` preemption per worker,
          assert no lease is handed to a paused worker while the hog runs, kill the hog and
          measure the time until each worker leases again.
Verify  - every chunk COMPLETED exactly once, nothing dead-lettered, output_graph.jsonl holds
          exactly one record per (job, chunk), unauthenticated requests are rejected, and no
          worker was preempted for a reason the test did not inject (e.g. the CPU guard firing
          on our own load during phases 1-3).

The CPU guard is enabled with the same threshold for the whole run rather than only for
phase 4: the worker processes are shared by all phases, and running phases 1-3 with the guard
armed is exactly what proves it does not false-positive on the grid's own load.

    python chaos_test.py --mock                    # ~3 min, no model needed
    python chaos_test.py --model qwen3:4b          # live Ollama
    python chaos_test.py --mock --phases 1,4 --markdown --markdown-out bench.md
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from demo_submit import SAMPLE_CHUNKS

HERE = Path(__file__).resolve().parent
# Reports and logs go next to the code in a checkout, else to the current directory (installed copy).
OUT = HERE if (HERE / "pyproject.toml").exists() else Path.cwd()
WORKERS = [("node-office-pc", 2), ("node-legacy-laptop", 1), ("node-mac-studio", 3)]
PREEMPT_NODE, CRASH_NODE = "node-legacy-laptop", "node-office-pc"
N = len(SAMPLE_CHUNKS)
ALL_PHASES = (1, 2, 3, 4)
PREEMPT_KINDS = ("PREEMPT_RECLAIM", "PREEMPT_IDLE")
# Reasons the harness itself injects. Anything else seen before phase 4 is a false preemption.
INJECTED_REASONS = {"trigger_file", "shutdown"}

# The hog bounds its own lifetime so that even if this harness is SIGKILLed before its
# `finally` runs, no busy-loop outlives the test by more than a few seconds. It is still a
# pure-Python spin (no sleeps, no syscalls in the hot loop) so it reads as 100% user CPU.
HOG_CODE = "import time, sys\nend = time.monotonic() + float(sys.argv[1])\nwhile time.monotonic() < end: pass\n"


@functools.lru_cache(maxsize=1)
def worker_help() -> str:
    """`worker_daemon.py --help`, used to feature-detect guard flags instead of hard-coding them.

    The worker's CLI is evolving (e.g. `--cpu-busy-percent` -> `--max-host-cpu-pct`); probing
    keeps this harness runnable against either version and lets it say which one it used."""
    try:
        return subprocess.run([sys.executable, str(HERE / "worker_daemon.py"), "--help"], capture_output=True,
                              text=True, timeout=30, cwd=HERE).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def worker_supports(flag: str) -> bool:
    return f"{flag} " in worker_help() or f"{flag}\n" in worker_help() or f"{flag}," in worker_help()


def guard_args(cpu_pct: float, cpu_samples: Optional[int]) -> tuple[list[str], str]:
    """Worker argv that arms the foreign-CPU guard at `cpu_pct`, plus a description for the report."""
    if worker_supports("--max-host-cpu-pct"):
        argv, desc = ["--max-host-cpu-pct", f"{cpu_pct:g}"], f"--max-host-cpu-pct {cpu_pct:g}"
    elif worker_supports("--cpu-busy-percent"):
        argv, desc = ["--cpu-busy-percent", f"{cpu_pct:g}"], f"--cpu-busy-percent {cpu_pct:g} (legacy flag)"
    else:
        return [], "none (worker has no CPU guard flag)"
    if cpu_samples and worker_supports("--cpu-busy-samples"):
        argv += ["--cpu-busy-samples", str(cpu_samples)]
        desc += f" --cpu-busy-samples {cpu_samples}"
    return argv, desc


class Grid:
    def __init__(self, args):
        self.args = args
        self.url = f"http://127.0.0.1:{args.port}"
        self.logdir = OUT / "chaos_logs"
        self.logdir.mkdir(exist_ok=True)
        self.output = OUT / args.output
        self.db = self.logdir / "grid_state.db"
        self.worker_token = secrets.token_urlsafe(32)
        self.client_token = secrets.token_urlsafe(32)
        self.http = httpx.Client(base_url=self.url, timeout=5,
                                 headers={"Authorization": f"Bearer {self.client_token}"})
        self.procs: dict[str, subprocess.Popen] = {}
        self.events: list[dict] = []
        self.since = 0
        self.coordinator_boots = 0
        self.spawns: dict[str, int] = {}
        self.guard_argv, self.guard_desc = guard_args(args.cpu_pct, args.cpu_samples)

    # ---------------------------------------------------------------- process management
    def spawn(self, name: str, argv: list[str], env: dict[str, str]) -> None:
        self.spawns[name] = self.spawns.get(name, 0) + 1
        log_name = name if self.spawns[name] == 1 else f"{name}-{self.spawns[name]}"
        log = open(self.logdir / f"{log_name}.log", "w")
        self.procs[name] = subprocess.Popen([sys.executable, "-u", *argv], cwd=HERE, stdout=log,
                                            stderr=subprocess.STDOUT, env={**os.environ, **env})

    def start_coordinator(self) -> None:
        self.coordinator_boots += 1
        argv = ["coordinator.py", "--port", str(self.args.port), "--output", str(self.output), "--db", str(self.db)]
        if "--no-advertise" in coordinator_help():
            argv.append("--no-advertise")  # a throwaway test grid must not show up in LAN discovery
        if getattr(self.args, "relay", False):
            argv.append("--relay")
        self.spawn("coordinator", argv,
                   {"GRID_WORKER_TOKEN": self.worker_token, "GRID_CLIENT_TOKEN": self.client_token, "GRID_TOKEN": ""})
        self.wait_for(lambda: httpx.get(f"{self.url}/healthz", timeout=1).status_code == 200, 15, "coordinator up")

    def start_worker(self, wid: str, tier: int) -> None:
        argv = ["worker_daemon.py", "--id", wid, "--tier", str(tier), "--coordinator", self.url, "--allow-battery"]
        argv += ["--mock"] if self.args.mock else ["--model", self.args.model]
        argv += self.guard_argv
        if worker_supports("--allow-no-idle-sensor"):
            argv.append("--allow-no-idle-sensor")  # the harness injects activity via trigger files, not HID
        if self.args.min_free_ram_gb is not None and worker_supports("--min-free-ram-gb"):
            argv += ["--min-free-ram-gb", f"{self.args.min_free_ram_gb:g}"]
        self.spawn(wid, argv, {"GRID_WORKER_TOKEN": self.worker_token, "GRID_TOKEN": "", "NIGHTSHIFT_TOKEN": ""})

    def start(self) -> None:
        for f in (self.output, self.db, Path(f"{self.db}-wal"), Path(f"{self.db}-shm")):
            f.unlink(missing_ok=True)
        for wid, _ in WORKERS:
            Path(f"/tmp/preempt_{wid}").unlink(missing_ok=True)
        self.start_coordinator()
        for wid, tier in WORKERS:
            self.start_worker(wid, tier)
        self.wait_for(lambda: len(self.status()["workers"]) == len(WORKERS), 60, "all workers registered")

    def live_workers(self) -> list[str]:
        return [wid for wid, _ in WORKERS if wid in self.procs and self.procs[wid].poll() is None]

    def respawn_dead_workers(self) -> list[str]:
        """Bring back workers killed by earlier phases so later phases measure the full grid."""
        revived = [(wid, tier) for wid, tier in WORKERS if wid not in self.live_workers()]
        for wid, tier in revived:
            self.start_worker(wid, tier)
        if revived:
            ids = {w for w, _ in revived}
            self.wait_for(lambda: ids <= {w["worker_id"] for w in self.status()["workers"] if w["state"] != "OFFLINE"},
                          60, "revived workers registered")
        return [w for w, _ in revived]

    def stop(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
        for p in self.procs.values():
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        for wid, _ in WORKERS:
            Path(f"/tmp/preempt_{wid}").unlink(missing_ok=True)

    # ---------------------------------------------------------------- API helpers
    def status(self) -> dict:
        return self.http.get("/v1/grid/status").raise_for_status().json()

    def job(self, job_id: str) -> dict:
        return self.http.get(f"/v1/grid/jobs/{job_id}").raise_for_status().json()

    def submit(self, name: str) -> str:
        r = self.http.post("/v1/grid/mapreduce", json={"job_name": name, "chunks": SAMPLE_CHUNKS, "required_tier": 1})
        return r.raise_for_status().json()["job_id"]

    def poll_events(self) -> None:
        try:
            new = self.http.get("/v1/grid/events", params={"since_seq": self.since}).raise_for_status().json()
        except httpx.HTTPError:
            return  # coordinator restarting
        if new:
            self.since = new[-1]["seq"]
            self.events.extend(new)

    def wait_event(self, pred, timeout: float, what: str) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for e in self.events:
                if pred(e):
                    return e
            self.poll_events()
            time.sleep(0.02)
        raise TimeoutError(f"timed out waiting for {what}")

    @staticmethod
    def wait_for(pred, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if pred():
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"timed out waiting for {what}")

    def wait_job(self, job_id: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.poll_events()
            try:
                st = self.job(job_id)
                if st["done"]:
                    return st
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"job {job_id} did not finish")

    def ev(self, kind: str, **match) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())]


@functools.lru_cache(maxsize=1)
def coordinator_help() -> str:
    try:
        return subprocess.run([sys.executable, str(HERE / "coordinator.py"), "--help"], capture_output=True,
                              text=True, timeout=30, cwd=HERE).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _r(x: Optional[float]) -> Optional[float]:
    return round(x, 3) if x is not None else None


# -------------------------------------------------------------------- phases
def phase_preempt(g: Grid) -> dict:
    print("\n=== Phase 1: live preemption of", PREEMPT_NODE, "===")
    job_id = g.submit("phase1-preemption")
    lease = g.wait_event(lambda e: e["kind"] == "LEASED" and e["worker_id"] == PREEMPT_NODE and e["job_id"] == job_id,
                         60, f"lease on {PREEMPT_NODE}")
    time.sleep(g.args.preempt_after)  # let it get mid-stream
    trigger = Path(f"/tmp/preempt_{PREEMPT_NODE}")
    t_touch = time.time()
    trigger.touch()
    print(f"  touched {trigger} while {PREEMPT_NODE} held chunk #{lease['chunk_index']}")
    task_id = lease["task_id"]
    reclaim = g.wait_event(lambda e: e["kind"] == "PREEMPT_RECLAIM" and e["task_id"] == task_id and e["ts"] >= t_touch, 10, "reclaim")
    relet = g.wait_event(lambda e: e["kind"] == "LEASED" and e["task_id"] == task_id and e["ts"] > reclaim["ts"], 60, "re-lease")
    done = g.wait_event(lambda e: e["kind"] == "COMPLETED" and e["task_id"] == task_id, 300, "recovered chunk completion")
    st = g.wait_job(job_id, 900)
    leaked = [e for e in g.ev("LEASED", worker_id=PREEMPT_NODE) if e["ts"] > t_touch]
    trigger.unlink()
    det = reclaim["data"].get("detected_at")
    return {
        "job": st, "preempted_chunk": lease["chunk_index"], "recovered_by": relet["worker_id"],
        "leases_issued_to_preempted_node_while_user_active": len(leaked),
        "latency_s": {
            "touch_to_detect": round(det - t_touch, 3) if det else None,
            "detect_to_coordinator_reclaim": round(reclaim["ts"] - det, 3) if det else None,
            "touch_to_reclaim": round(reclaim["ts"] - t_touch, 3),
            "reclaim_to_relet": round(relet["ts"] - reclaim["ts"], 3),
            "touch_to_recovered_chunk_completed": round(done["ts"] - t_touch, 3),
        },
    }


def phase_worker_crash(g: Grid) -> dict:
    print("\n=== Phase 2: SIGKILL", CRASH_NODE, "mid-lease (lease-expiry path) ===")
    job_id = g.submit("phase2-worker-crash")
    lease = g.wait_event(lambda e: e["kind"] == "LEASED" and e["worker_id"] == CRASH_NODE and e["job_id"] == job_id,
                         60, f"lease on {CRASH_NODE}")
    time.sleep(0.5)
    t_kill = time.time()
    g.procs[CRASH_NODE].send_signal(signal.SIGKILL)
    g.procs[CRASH_NODE].wait()
    print(f"  SIGKILLed {CRASH_NODE} while holding chunk #{lease['chunk_index']}")
    task_id = lease["task_id"]
    expired = g.wait_event(lambda e: e["kind"] == "LEASE_EXPIRED" and e["task_id"] == task_id, 30, "lease expiry")
    relet = g.wait_event(lambda e: e["kind"] == "LEASED" and e["task_id"] == task_id and e["ts"] > expired["ts"], 60, "re-lease")
    done = g.wait_event(lambda e: e["kind"] == "COMPLETED" and e["task_id"] == task_id, 300, "recovered chunk completion")
    offline = g.wait_event(lambda e: e["kind"] == "NODE_OFFLINE" and e["worker_id"] == CRASH_NODE, 30, "offline detection")
    st = g.wait_job(job_id, 900)
    resumed = len(g.ev("COMPLETED", job_id=job_id, worker_id=PREEMPT_NODE))
    return {
        "job": st, "killed_chunk": lease["chunk_index"], "recovered_by": relet["worker_id"],
        "previously_preempted_node_resumed_chunks": resumed,
        "latency_s": {
            "kill_to_lease_expired": round(expired["ts"] - t_kill, 3),
            "expired_to_relet": round(relet["ts"] - expired["ts"], 3),
            "kill_to_recovered_chunk_completed": round(done["ts"] - t_kill, 3),
            "kill_to_node_offline": round(offline["ts"] - t_kill, 3),
        },
    }


def phase_coordinator_crash(g: Grid) -> dict:
    print("\n=== Phase 3: SIGKILL the coordinator mid-job, restart on the same state file ===")
    job_id = g.submit("phase3-coordinator-crash")
    g.wait_event(lambda e: e["kind"] == "COMPLETED" and e["job_id"] == job_id
                 and len(g.ev("COMPLETED", job_id=job_id)) >= 4, 300, "a few completions")
    # Crash while at least one chunk is in flight.
    g.wait_for(lambda: g.status()["task_counts"]["LEASED"] > 0, 30, "an in-flight lease")
    g.poll_events()
    completed_before = {e["task_id"] for e in g.ev("COMPLETED", job_id=job_id)}
    leased = {e["task_id"]: e for e in g.ev("LEASED", job_id=job_id)}
    inflight = {t for t in leased if t not in completed_before}
    t_kill = time.time()
    g.procs["coordinator"].send_signal(signal.SIGKILL)
    g.procs["coordinator"].wait()
    print(f"  SIGKILLed coordinator: {len(completed_before)}/{N} done, {len(inflight)} chunk(s) in flight")
    time.sleep(g.args.outage)
    t_restart = time.time()
    g.start_coordinator()
    t_ready = time.time()
    recovered = g.wait_event(lambda e: e["kind"] == "RECOVERED" and e["ts"] >= t_restart, 10, "recovery event")
    st = g.wait_job(job_id, 900)
    recs = {r["task_id"]: r for r in map(json.loads, g.output.read_text().splitlines()) if r["job_id"] == job_id}
    by_original = sum(1 for t in inflight if t in recs and recs[t]["worker_id"] == leased[t]["worker_id"]
                      and recs[t]["attempts"] == leased[t]["data"]["attempt"])
    first_after = min((e["ts"] for e in g.ev("COMPLETED", job_id=job_id) if e["ts"] > t_restart), default=None)
    return {
        "job": st, "completed_before_crash": len(completed_before), "inflight_at_crash": len(inflight),
        "inflight_completed_by_original_holder": by_original,
        "recovered": recovered["data"],
        "latency_s": {
            "outage_injected": g.args.outage,
            "restart_to_ready": round(t_ready - t_restart, 3),
            "kill_to_first_completion_after_restart": round(first_after - t_kill, 3) if first_after else None,
        },
    }


def spawn_hogs(n: int, max_s: float) -> list[subprocess.Popen]:
    """Start `n` single-core busy loops, each in its own session so a killpg cannot miss one."""
    return [subprocess.Popen([sys.executable, "-c", HOG_CODE, str(max_s + 2)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            for _ in range(n)]


def kill_hogs(hogs: list[subprocess.Popen]) -> None:
    for h in hogs:
        try:
            os.killpg(h.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                h.kill()
            except OSError:
                pass
    for h in hogs:
        try:
            h.wait(5)
        except subprocess.TimeoutExpired:
            pass


def phase_cpu_spike(g: Grid) -> dict:
    print("\n=== Phase 4: host CPU spike (busy-loop on every core) ===")
    revived = g.respawn_dead_workers()
    if revived:
        print(f"  respawned {', '.join(revived)} (killed in an earlier phase) so every node is measured")
    job_id = g.submit("phase4-cpu-spike")
    workers = set(g.live_workers())

    def all_leasing() -> bool:
        g.poll_events()
        return {e["worker_id"] for e in g.ev("LEASED", job_id=job_id)} >= workers

    g.wait_for(all_leasing, 90, "every worker leasing")
    n_hogs = g.args.hog_procs or os.cpu_count() or 4
    hogs: list[subprocess.Popen] = []
    preempt: dict[str, dict] = {}
    t_hog = time.time()
    try:
        hogs = spawn_hogs(n_hogs, g.args.hog_max)
        print(f"  started {n_hogs} busy-loop processes (cap {g.args.hog_max:g}s); guard: {g.guard_desc}")
        deadline = t_hog + g.args.hog_max
        while time.time() < deadline and set(preempt) < workers:
            g.poll_events()
            for e in g.events:
                if (e["kind"] in PREEMPT_KINDS and e["ts"] >= t_hog and e.get("worker_id") in workers
                        and e["data"].get("reason") == "cpu_busy"):
                    preempt.setdefault(e["worker_id"], e)
            time.sleep(0.05)
        # Keep the load on a little longer so "no leases while paused" is observed, not assumed.
        hold_until = min(deadline, time.time() + g.args.hog_hold)
        while time.time() < hold_until:
            g.poll_events()
            time.sleep(0.1)
    finally:
        kill_hogs(hogs)
    t_kill = time.time()
    hog_s = t_kill - t_hog
    print(f"  killed hog after {hog_s:.1f}s; {len(preempt)}/{len(workers)} worker(s) yielded with reason cpu_busy")

    def resumed(w: str) -> Optional[dict]:
        return next((e for e in g.ev("LEASED", worker_id=w) if e["ts"] > t_kill), None)

    resume_deadline = time.time() + g.args.resume_timeout
    while time.time() < resume_deadline and not all(resumed(w) for w in preempt):
        g.poll_events()
        if g.job(job_id)["done"]:
            break
        time.sleep(0.1)
    st = g.wait_job(job_id, 900)

    per_worker = {}
    for w in sorted(workers):
        p = preempt.get(w)
        r = resumed(w)
        det = p["data"].get("detected_at") if p else None
        leaked = [e for e in g.ev("LEASED", worker_id=w) if p and p["ts"] < e["ts"] <= t_kill]
        per_worker[w] = {
            "preempted": p is not None, "event": p["kind"] if p else None,
            "chunk_reclaimed": p.get("chunk_index") if p and p["kind"] == "PREEMPT_RECLAIM" else None,
            "leases_while_hogged": len(leaked),
            "latency_s": {
                "hog_to_worker_detect": _r(det - t_hog) if det else None,
                "hog_to_coordinator_preempt": _r(p["ts"] - t_hog) if p else None,
                "kill_to_resume_lease": _r(r["ts"] - t_kill) if r else None,
            },
        }

    def worst(key: str) -> Optional[float]:
        vals = [v["latency_s"][key] for v in per_worker.values() if v["latency_s"][key] is not None]
        return max(vals) if vals else None

    return {
        "job": st, "hog_processes": n_hogs, "hog_seconds": round(hog_s, 3), "guard": g.guard_desc,
        "workers": per_worker, "revived_workers": revived,
        "latency_s": {
            "max_hog_to_worker_detect": worst("hog_to_worker_detect"),
            "max_hog_to_coordinator_preempt": worst("hog_to_coordinator_preempt"),
            "max_kill_to_resume_lease": worst("kill_to_resume_lease"),
        },
        "_t_hog": t_hog,
    }


def unintended_preemptions(g: Grid, before_ts: float) -> list[dict]:
    """Preemptions the harness did not cause: any reason except trigger-file (phase 1, on
    PREEMPT_NODE only) and shutdown, up to `before_ts` (the start of phase 4's hog)."""
    out = []
    for e in g.events:
        if e["kind"] not in PREEMPT_KINDS or e["ts"] >= before_ts:
            continue
        reason = e["data"].get("reason")
        if reason in INJECTED_REASONS and (reason != "trigger_file" or e.get("worker_id") == PREEMPT_NODE):
            continue
        out.append({"worker_id": e.get("worker_id"), "reason": reason, "kind": e["kind"], "ts": e["ts"]})
    return out


def check_auth(g: Grid) -> dict:
    anon = httpx.post(f"{g.url}/v1/grid/mapreduce", json={"job_name": "x", "chunks": ["x"]}, timeout=5).status_code
    worker_as_client = httpx.get(f"{g.url}/v1/grid/status", timeout=5,
                                 headers={"Authorization": f"Bearer {g.worker_token}"}).status_code
    forged = httpx.post(f"{g.url}/worker/complete", timeout=5, headers={"Authorization": "Bearer guess"},
                        json={"worker_id": "x", "task_id": "x", "lease_id": "x", "result": {}}).status_code
    return {"anonymous_submit": anon, "worker_token_on_client_api": worker_as_client, "forged_completion": forged}


def verify_output(g: Grid, job_ids: list[str]) -> dict:
    recs = [json.loads(l) for l in g.output.read_text().splitlines() if l.strip()] if g.output.exists() else []
    keys = [(r["job_id"], r["chunk_index"]) for r in recs]
    expected = {(j, i) for j in job_ids for i in range(N)}
    return {
        "records": len(recs), "expected": len(expected),
        "missing": sorted(expected - set(keys)), "duplicates": len(keys) - len(set(keys)),
        "total_triples": sum(len(r["triples"]) for r in recs),
        "total_entities": sum(len(r["entities"]) for r in recs),
    }


def throughput(g: Grid, job_ids: list[str]) -> dict:
    per: dict[str, list[float]] = {}
    for e in g.events:
        if e["kind"] == "COMPLETED" and e["job_id"] in job_ids:
            per.setdefault(e["worker_id"], []).append(e["data"].get("tokens_per_sec", 0.0))
    return {w: {"chunks": len(v), "avg_tok_s": round(sum(v) / len(v), 1)} for w, v in sorted(per.items())}


# -------------------------------------------------------------------- report
PHASE_KEYS = {1: "phase1_preemption", 2: "phase2_worker_crash", 3: "phase3_coordinator_crash", 4: "phase4_cpu_spike"}
PHASE_FAULTS = {
    1: f"trigger file on {PREEMPT_NODE} mid-stream",
    2: f"SIGKILL {CRASH_NODE} mid-lease",
    3: "SIGKILL coordinator mid-job, restart on SQLite",
    4: "busy-loop on every core (host CPU spike)",
}


def job_complete(p: dict) -> bool:
    j = p["job"]
    return j["counts"]["COMPLETED"] == j["total"] and j["counts"]["FAILED"] == 0


def build_checks(report: dict) -> dict[str, bool]:
    phases = {n: report[k] for n, k in PHASE_KEYS.items() if k in report}
    out, auth = report["output_verification"], report["auth"]
    checks = {
        "all chunks completed, none dead-lettered": all(job_complete(p) for p in phases.values()),
        "output: no missing / duplicate records": out["records"] == out["expected"] and not out["missing"]
                                                  and out["duplicates"] == 0,
        "auth rejects anonymous / wrong-role / forged calls": all(v == 401 for v in auth.values()),
        "no unintended preemptions before phase 4 (CPU/RAM guards quiet on grid's own load)":
            not report["unintended_preemptions"],
    }
    if 1 in phases:
        checks["no work sent to a node while its user was active"] = \
            phases[1]["leases_issued_to_preempted_node_while_user_active"] == 0
    if 1 in phases and 2 in phases:
        checks["preempted node resumed harvesting"] = phases[2]["previously_preempted_node_resumed_chunks"] > 0
    relay = report.get("relay") or {}
    if relay.get("enabled"):
        checks["relay: interrupted chunks resumed from their checkpoint"] = relay.get("resumes", 0) > 0
        checks["relay: no resumed attempt failed (resumed output identical to uninterrupted)"] = \
            all(p["job"]["total_failures"] <= (1 if n == 2 else 0) for n, p in phases.items())
    if 4 in phases:
        ws = phases[4]["workers"].values()
        checks["phase 4: every worker yielded with reason cpu_busy"] = all(w["preempted"] for w in ws)
        checks["phase 4: no leases to a CPU-preempted worker while the hog ran"] = \
            all(w["leases_while_hogged"] == 0 for w in ws)
        checks["phase 4: every worker resumed leasing after the hog"] = \
            all(w["latency_s"]["kill_to_resume_lease"] is not None for w in ws)
    return checks


def _s(v) -> str:
    return "–" if v is None else f"{v:.2f} s"


def _recovery_summary(n: int, p: dict) -> str:
    lat = p["latency_s"]
    if n == 1:
        return f"reclaim {_s(lat['touch_to_reclaim'])}, re-lease +{_s(lat['reclaim_to_relet'])}"
    if n == 2:
        return f"lease expired {_s(lat['kill_to_lease_expired'])}, chunk done {_s(lat['kill_to_recovered_chunk_completed'])}"
    if n == 3:
        return (f"ready {_s(lat['restart_to_ready'])} after restart, {p['inflight_completed_by_original_holder']}/"
                f"{p['inflight_at_crash']} in-flight kept")
    return (f"all yielded ≤ {_s(lat['max_hog_to_coordinator_preempt'])}, "
            f"resumed ≤ {_s(lat['max_kill_to_resume_lease'])} after hog")


def render_markdown(report: dict) -> str:
    """Render a chaos report as README-ready Markdown. Pure function of the JSON report so the
    table can be regenerated from any saved `chaos_report*.json` without re-running the grid."""
    checks = report.get("checks", {})
    lines = [f"### Nightshift chaos benchmark ({report.get('mode', '?')}, {len(WORKERS)} workers, "
             f"{N} chunks/phase, CPU guard: {report.get('cpu_guard', '?')})", "",
             "| Phase | Fault injected | Chunks | Wall time | Attempts | Preemptions | Recovery | Result |",
             "|---|---|---|---|---|---|---|---|"]
    lat_rows = []
    for n, key in PHASE_KEYS.items():
        p = report.get(key)
        if not p:
            continue
        j = p["job"]
        lines.append(f"| {n} | {PHASE_FAULTS[n]} | {j['counts']['COMPLETED']}/{j['total']} | {j['elapsed_s']:.1f} s | "
                     f"{j['total_attempts']} | {j['total_preemptions']} | {_recovery_summary(n, p)} | "
                     f"{'PASS' if job_complete(p) else 'FAIL'} |")
        lat_rows += [f"| {n} | `{k}` | {_s(v)} |" for k, v in p["latency_s"].items()]
    lines += ["", "| Phase | Latency | Value |", "|---|---|---|", *lat_rows]

    p4 = report.get(PHASE_KEYS[4])
    if p4:
        lines += ["", f"Phase 4: {p4['hog_processes']} busy-loop processes for {p4['hog_seconds']:.1f} s.", "",
                  "| Worker | Yield event | Hog → worker detect | Hog → coordinator | Leases while hogged | "
                  "Hog killed → next lease |", "|---|---|---|---|---|---|"]
        for w, v in p4["workers"].items():
            lat = v["latency_s"]
            lines.append(f"| {w} | {v['event'] or 'none'} | {_s(lat['hog_to_worker_detect'])} | "
                         f"{_s(lat['hog_to_coordinator_preempt'])} | {v['leases_while_hogged']} | "
                         f"{_s(lat['kill_to_resume_lease'])} |")

    out = report.get("output_verification", {})
    lines += ["", f"Output: {out.get('records')}/{out.get('expected')} records, {len(out.get('missing', []))} missing, "
                  f"{out.get('duplicates')} duplicates, {out.get('total_triples')} triples. "
                  f"Unintended preemptions: {len(report.get('unintended_preemptions', []))}.", "",
              "| Check | Result |", "|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else '**FAIL**'} |" for name, ok in checks.items()]
    return "\n".join(lines) + "\n"


def parse_phases(text: str) -> list[int]:
    try:
        phases = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--phases expects e.g. 1,2,3,4 (got {text!r})")
    if not phases or any(p not in ALL_PHASES for p in phases):
        raise argparse.ArgumentTypeError(f"--phases must be a subset of {','.join(map(str, ALL_PHASES))}")
    return phases


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Nightshift end-to-end chaos benchmark", allow_abbrev=False)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--phases", type=parse_phases, default=list(ALL_PHASES), help="comma list, default 1,2,3,4")
    ap.add_argument("--output", default="chaos_logs/output_graph.jsonl", help="test sink (kept out of the real output_graph.jsonl)")
    ap.add_argument("--preempt-after", type=float, default=1.0, help="seconds after lease before touching trigger")
    ap.add_argument("--outage", type=float, default=3.0, help="seconds the coordinator stays down in phase 3")
    ap.add_argument("--cpu-pct", type=float, default=65.0, help="worker foreign-CPU guard threshold for the whole run")
    ap.add_argument("--cpu-samples", type=int, default=2, help="consecutive ~1s samples to pause/resume (if supported)")
    ap.add_argument("--min-free-ram-gb", type=float, default=None, help="pass to workers (default: worker default)")
    ap.add_argument("--hog-procs", type=int, default=0, help="phase 4 busy loops (default os.cpu_count())")
    ap.add_argument("--hog-max", type=float, default=20.0, help="hard cap on phase 4 hog duration, seconds")
    ap.add_argument("--hog-hold", type=float, default=3.0, help="keep the hog on this long after all workers yield")
    ap.add_argument("--resume-timeout", type=float, default=30.0, help="phase 4: wait this long for workers to resume")
    ap.add_argument("--relay", action="store_true", help="run the grid with Baton token relay enabled")
    ap.add_argument("--report", default="chaos_report.json")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown benchmark table at the end")
    ap.add_argument("--markdown-out", default=None, metavar="FILE", help="also write the Markdown table to FILE")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    g = Grid(args)
    print(f"CPU guard for all phases: {g.guard_desc}")
    runners = {1: phase_preempt, 2: phase_worker_crash, 3: phase_coordinator_crash, 4: phase_cpu_spike}
    results: dict[int, dict] = {}
    try:
        g.start()
        auth = check_auth(g)
        for n in args.phases:
            results[n] = runners[n](g)
        g.poll_events()
        relay = g.status().get("relay")
        hog_start = results[4].pop("_t_hog") if 4 in results else float("inf")
        jobs = [p["job"]["job_id"] for p in results.values()]
        report = {"mode": "mock" if args.mock else f"live:{args.model}", "phases": args.phases,
                  "cpu_guard": g.guard_desc, "auth": auth,
                  **{PHASE_KEYS[n]: p for n, p in results.items()},
                  "unintended_preemptions": unintended_preemptions(g, hog_start),
                  "output_verification": verify_output(g, jobs), "throughput": throughput(g, jobs)}
        if relay is not None:
            report["relay"] = relay
    finally:
        g.stop()
    report["checks"] = checks = build_checks(report)
    (OUT / args.report).write_text(json.dumps(report, indent=2))

    names = {1: "Phase 1 preemption", 2: "Phase 2 worker crash", 3: "Phase 3 coordinator crash", 4: "Phase 4 CPU spike"}
    for n, p in results.items():
        j = p["job"]
        print(f"\n{names[n]}: {j['counts']['COMPLETED']}/{j['total']} completed in {j['elapsed_s']:.1f}s "
              f"(attempts={j['total_attempts']}, preemptions={j['total_preemptions']}, failures={j['total_failures']})")
        for k, v in p["latency_s"].items():
            print(f"    {k:<42} {v}s")
    if 3 in results:
        p3 = results[3]
        print(f"\nPhase 3: {p3['inflight_at_crash']} in flight at crash, {p3['inflight_completed_by_original_holder']} "
              f"completed by their original holder after restart; recovered {p3['recovered']}")
    if 4 in results:
        for w, v in results[4]["workers"].items():
            print(f"    {w:<20} {v['event'] or 'NOT PREEMPTED':<16} {v['latency_s']} leases_while_hogged={v['leases_while_hogged']}")
    out = report["output_verification"]
    print(f"\nOutput: {out['records']}/{out['expected']} records, missing={len(out['missing'])}, "
          f"duplicates={out['duplicates']}, triples={out['total_triples']}")
    print("Auth:", auth)
    print("Throughput:", report["throughput"])
    if report.get("relay"):
        r = report["relay"]
        print(f"Relay: enabled={r['enabled']} resumes={r['resumes']} relayed_chunks={r['relayed_chunks']} "
              f"tokens generated={r['tokens_generated']} in results={r['tokens_in_results']} "
              f"salvaged={r['tokens_salvaged']} goodput={r['goodput_pct']}%")
    if report["unintended_preemptions"]:
        print("Unintended preemptions:", report["unintended_preemptions"])
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(checks.values())
    print("\nRESULT:", "PASS - zero dropped chunks" if ok else "FAIL")
    md = render_markdown(report)
    if args.markdown_out:
        Path(args.markdown_out).write_text(md)
    if args.markdown:
        print("\n" + md)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
