"""chaos_test.py report logic: checks, false-preemption filter, Markdown rendering, phase 4 hog.

The full harness needs real processes and minutes of wall time; everything here runs on
synthetic reports/events so the pass/fail logic and the README table are tested in isolation."""
import argparse
import subprocess
import sys
import time

import pytest

import chaos_test as ct


def _job(total=24, done=24, failed=0, elapsed=30.0, attempts=25, preemptions=1):
    return {"job_id": "j", "total": total, "counts": {"COMPLETED": done, "FAILED": failed, "LEASED": 0, "PENDING": 0},
            "elapsed_s": elapsed, "total_attempts": attempts, "total_preemptions": preemptions,
            "total_failures": 0, "done": True}


def _report(**over):
    w = {"preempted": True, "event": "PREEMPT_RECLAIM", "chunk_reclaimed": 3, "leases_while_hogged": 0,
         "latency_s": {"hog_to_worker_detect": 2.1, "hog_to_coordinator_preempt": 2.12, "kill_to_resume_lease": 3.4}}
    r = {
        "mode": "mock", "cpu_guard": "--max-host-cpu-pct 65", "auth": {"a": 401, "b": 401, "c": 401},
        "phase1_preemption": {"job": _job(), "leases_issued_to_preempted_node_while_user_active": 0,
                              "latency_s": {"touch_to_reclaim": 0.1, "reclaim_to_relet": 1.6}},
        "phase2_worker_crash": {"job": _job(), "previously_preempted_node_resumed_chunks": 5,
                                "latency_s": {"kill_to_lease_expired": 12.1, "kill_to_recovered_chunk_completed": 16.0}},
        "phase3_coordinator_crash": {"job": _job(), "inflight_at_crash": 2, "inflight_completed_by_original_holder": 2,
                                     "latency_s": {"restart_to_ready": 0.23, "outage_injected": 3.0}},
        "phase4_cpu_spike": {"job": _job(attempts=27, preemptions=3), "hog_processes": 10, "hog_seconds": 6.0,
                             "workers": {"node-a": dict(w), "node-b": {**w, "event": "PREEMPT_IDLE"}},
                             "latency_s": {"max_hog_to_worker_detect": 2.1, "max_hog_to_coordinator_preempt": 2.12,
                                           "max_kill_to_resume_lease": 3.4}},
        "unintended_preemptions": [],
        "output_verification": {"records": 96, "expected": 96, "missing": [], "duplicates": 0, "total_triples": 300},
    }
    r.update(over)
    return r


def test_all_checks_pass_on_clean_report():
    checks = ct.build_checks(_report())
    assert checks and all(checks.values()), checks
    assert any("phase 4" in k for k in checks)


def test_unintended_preemption_fails_run():
    r = _report(unintended_preemptions=[{"worker_id": "node-a", "reason": "cpu_busy"}])
    checks = ct.build_checks(r)
    assert not checks["no unintended preemptions before phase 4 (CPU/RAM guards quiet on grid's own load)"]


def test_phase4_failures_are_detected():
    r = _report()
    r["phase4_cpu_spike"]["workers"]["node-b"] = {
        "preempted": False, "event": None, "chunk_reclaimed": None, "leases_while_hogged": 1,
        "latency_s": {"hog_to_worker_detect": None, "hog_to_coordinator_preempt": None, "kill_to_resume_lease": None}}
    checks = ct.build_checks(r)
    assert not checks["phase 4: every worker yielded with reason cpu_busy"]
    assert not checks["phase 4: no leases to a CPU-preempted worker while the hog ran"]
    assert not checks["phase 4: every worker resumed leasing after the hog"]


def test_phase_specific_checks_only_for_phases_run():
    r = _report()
    for k in ("phase1_preemption", "phase2_worker_crash", "phase4_cpu_spike"):
        del r[k]
    checks = ct.build_checks(r)
    assert not any(k.startswith("phase 4:") or "user was active" in k or "resumed harvesting" in k for k in checks)


def test_markdown_tables():
    md = ct.render_markdown({**_report(), "checks": ct.build_checks(_report())})
    lines = md.splitlines()
    assert lines[0].startswith("### Nightshift chaos benchmark (mock")
    summary = [l for l in lines if l.startswith("| ") and l[2].isdigit() and "Fault" not in l and "`" not in l]
    assert [l.split("|")[1].strip() for l in summary] == ["1", "2", "3", "4"]
    assert all(l.rstrip().endswith("PASS |") for l in summary)
    assert "| node-a | PREEMPT_RECLAIM | 2.10 s | 2.12 s | 0 | 3.40 s |" in md
    assert "| node-b | PREEMPT_IDLE |" in md
    assert "`restart_to_ready` | 0.23 s" in md
    assert "**FAIL**" not in md
    # Every table row has the same column count as its header (valid GitHub Markdown).
    for block in md.split("\n\n"):
        rows = [l for l in block.splitlines() if l.startswith("|")]
        if rows:
            assert len({r.count("|") for r in rows}) == 1, block


def test_markdown_marks_failures_and_missing_values():
    r = _report()
    r["phase1_preemption"]["job"] = _job(done=23, failed=1)
    r["phase4_cpu_spike"]["workers"]["node-a"]["latency_s"]["kill_to_resume_lease"] = None
    md = ct.render_markdown({**r, "checks": ct.build_checks(r)})
    assert "| 23/24 |" in md and "FAIL |" in md and "**FAIL**" in md
    assert "| – |" in md


class FakeGrid:
    def __init__(self, events):
        self.events = events


def _ev(kind, wid, reason, ts):
    return {"kind": kind, "worker_id": wid, "ts": ts, "data": {"reason": reason}}


def test_unintended_preemptions_filter():
    events = [
        _ev("PREEMPT_RECLAIM", ct.PREEMPT_NODE, "trigger_file", 1),   # phase 1, intended
        _ev("PREEMPT_IDLE", "node-mac-studio", "shutdown", 2),        # harness teardown
        _ev("PREEMPT_IDLE", "node-mac-studio", "trigger_file", 3),    # trigger on the wrong node
        _ev("PREEMPT_RECLAIM", "node-office-pc", "cpu_busy", 4),      # false CPU-guard positive
        {"kind": "LEASED", "worker_id": "x", "ts": 5, "data": {}},
        _ev("PREEMPT_RECLAIM", "node-office-pc", "cpu_busy", 10),     # phase 4: intended
    ]
    bad = ct.unintended_preemptions(FakeGrid(events), before_ts=9)
    assert [(b["worker_id"], b["reason"]) for b in bad] == [("node-mac-studio", "trigger_file"),
                                                            ("node-office-pc", "cpu_busy")]


def test_parse_phases():
    assert ct.parse_phases("4,1, 2") == [1, 2, 4]
    for bad in ("", "5", "a"):
        with pytest.raises(argparse.ArgumentTypeError):
            ct.parse_phases(bad)
    assert ct.build_parser().parse_args([]).phases == [1, 2, 3, 4]


def test_guard_args_prefers_new_flag(monkeypatch):
    monkeypatch.setattr(ct, "worker_help", lambda: "  --max-host-cpu-pct P  x\n  --cpu-busy-samples N  y\n")
    argv, desc = ct.guard_args(65, 2)
    assert argv == ["--max-host-cpu-pct", "65", "--cpu-busy-samples", "2"] and "65" in desc
    monkeypatch.setattr(ct, "worker_help", lambda: "  --cpu-busy-percent P  x\n")
    assert ct.guard_args(65, 2)[0] == ["--cpu-busy-percent", "65"]
    monkeypatch.setattr(ct, "worker_help", lambda: "")
    assert ct.guard_args(65, 2) == ([], "none (worker has no CPU guard flag)")


def test_hog_self_terminates_even_if_never_killed():
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", ct.HOG_CODE, "0.3"])
    assert p.wait(10) == 0
    assert time.time() - t0 < 10


def test_kill_hogs_is_immediate():
    hogs = ct.spawn_hogs(2, max_s=30)
    time.sleep(0.2)
    t0 = time.time()
    ct.kill_hogs(hogs)
    assert all(h.poll() is not None for h in hogs)
    assert time.time() - t0 < 5
