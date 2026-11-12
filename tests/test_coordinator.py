import json

import pytest

from coordinator import (BatchSubmitRequest, GridState, HeartbeatRequest, LeaseError, PreemptRequest, TaskCompleteRequest,
                         TaskFailRequest, TaskState, WorkerInfo, WorkerState)
from graphrag import JsonlGraphSink
from grid_store import GridStore

GOOD = {"entities": ["A", "B"], "triples": [{"subject": "A", "relation": "LINKS", "object": "B"}]}


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


@pytest.fixture
def clock(): return Clock()


@pytest.fixture
def grid(clock, tmp_path):
    g = GridState(sink=JsonlGraphSink(tmp_path / "out.jsonl"), store=GridStore(tmp_path / "grid.db"), clock=clock)
    for wid, tier in [("w1", 1), ("w2", 2), ("w3", 3)]:
        g.register(WorkerInfo(worker_id=wid, tier=tier))
    yield g
    g.check_invariants()


def submit(g, n=3, tier=1): return g.submit(BatchSubmitRequest(job_name="j", chunks=[f"c{i}" for i in range(n)], required_tier=tier))
def complete(g, wid, t, result=GOOD): return g.complete(TaskCompleteRequest(worker_id=wid, task_id=t.task_id, lease_id=t.lease_id, result=result))
def lines(tmp_path): return [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]


def test_happy_path_writes_each_chunk_once(grid, tmp_path):
    job = submit(grid, 3)
    for _ in range(3):
        t = grid.pull("w1"); complete(grid, "w1", t)
    st = grid.job_status(job.job_id)
    assert st["done"] and st["counts"]["COMPLETED"] == 3
    recs = lines(tmp_path)
    assert sorted(r["chunk_index"] for r in recs) == [0, 1, 2]
    assert recs[0]["triples"] == [["A", "LINKS", "B"]]


def test_preempt_requeues_to_front_without_counting_failure(grid):
    submit(grid, 3)
    t = grid.pull("w2")
    assert grid.preempt(PreemptRequest(worker_id="w2")) == 1
    assert t.state == TaskState.PENDING and t.failures == 0 and t.preemptions == 1
    assert grid.workers["w2"].state == WorkerState.PREEMPTED_USER_ACTIVE
    assert grid.pull("w1").task_id == t.task_id  # recovered chunk is served first


def test_stale_lease_cannot_complete_or_heartbeat(grid, clock):
    submit(grid, 1)
    t = grid.pull("w1"); stale = t.lease_id
    clock.advance(13); grid.reap()                       # lease expires
    t2 = grid.pull("w2")
    assert t2.task_id == t.task_id and t2.lease_id != stale
    with pytest.raises(LeaseError):
        grid.complete(TaskCompleteRequest(worker_id="w1", task_id=t.task_id, lease_id=stale, result=GOOD))
    hb = grid.heartbeat(HeartbeatRequest(worker_id="w1", task_id=t.task_id, lease_id=stale))
    assert hb["lease_valid"] is False
    # Same worker re-leasing the same chunk must still be fenced by lease_id
    clock.advance(13); grid.reap(); t3 = grid.pull("w1")
    with pytest.raises(LeaseError):
        grid.complete(TaskCompleteRequest(worker_id="w1", task_id=t.task_id, lease_id=stale, result=GOOD))
    assert complete(grid, "w1", t3)["status"] == "accepted"


def test_heartbeat_extends_lease(grid, clock):
    submit(grid, 1); t = grid.pull("w1")
    for _ in range(10):
        clock.advance(3)
        assert grid.heartbeat(HeartbeatRequest(worker_id="w1", task_id=t.task_id, lease_id=t.lease_id,
                                               state=WorkerState.BUSY_INFERENCING))["lease_valid"]
        grid.reap()
    assert t.state == TaskState.LEASED


def test_duplicate_completion_is_idempotent(grid, tmp_path):
    submit(grid, 1); t = grid.pull("w1"); lease = t.lease_id
    complete(grid, "w1", t)
    r = grid.complete(TaskCompleteRequest(worker_id="w1", task_id=t.task_id, lease_id=lease, result=GOOD))
    assert r["status"] == "duplicate" and len(lines(tmp_path)) == 1
    assert grid.workers["w1"].completed_count == 1


def test_poison_chunk_dead_lettered(grid):
    job = submit(grid, 1)
    for _ in range(3):
        t = grid.pull("w1")
        grid.fail(TaskFailRequest(worker_id="w1", task_id=t.task_id, lease_id=t.lease_id, error="boom"))
    st = grid.job_status(job.job_id)
    assert st["counts"]["FAILED"] == 1 and st["done"] and grid.pull("w1") is None


def test_invalid_extraction_is_rejected_and_retried(grid, tmp_path):
    submit(grid, 1); t = grid.pull("w1")
    assert complete(grid, "w1", t, result={"raw": "sorry I cannot"})["status"] == "rejected"
    assert t.state == TaskState.PENDING and t.failures == 1
    assert not (tmp_path / "out.jsonl").read_text()


def test_tier_scheduling(grid):
    submit(grid, 1, tier=1); submit(grid, 1, tier=3)
    assert grid.pull("w2").required_tier == 1  # tier-3 work invisible to tier 2
    assert grid.pull("w1") is None
    assert grid.pull("w3").required_tier == 3


def test_tier3_prefers_tier3_work(grid):
    submit(grid, 1, tier=1); submit(grid, 1, tier=3)
    assert grid.pull("w3").required_tier == 3


def test_offline_detection_and_reregister_reclaims(grid, clock):
    submit(grid, 1); t = grid.pull("w1")
    grid.register(WorkerInfo(worker_id="w1", tier=1))      # crashed & restarted
    assert t.state == TaskState.PENDING and t.failures == 0
    clock.advance(13); grid.reap()
    assert all(w.state == WorkerState.OFFLINE for w in grid.workers.values())




def test_late_preempt_with_old_lease_does_not_reclaim_new_lease(grid):
    submit(grid, 2)
    a = grid.pull("w1"); old = a.lease_id
    grid.preempt(PreemptRequest(worker_id="w1", lease_id=old))
    grid.heartbeat(HeartbeatRequest(worker_id="w1", state=WorkerState.IDLE_HARVESTING))
    b = grid.pull("w1")
    assert grid.preempt(PreemptRequest(worker_id="w1", lease_id=old)) == 0  # stale, delayed preempt
    assert b.state == TaskState.LEASED


def test_preempted_worker_gets_no_work_until_idle_heartbeat(grid):
    submit(grid, 1); grid.pull("w1"); grid.preempt(PreemptRequest(worker_id="w1"))
    assert grid.pull("w1") is None
    grid.heartbeat(HeartbeatRequest(worker_id="w1", state=WorkerState.IDLE_HARVESTING))
    assert grid.pull("w1") is not None


def test_one_lease_per_worker(grid):
    submit(grid, 2); a = grid.pull("w1"); old = a.lease_id; b = grid.pull("w1")
    # abandoned lease is re-queued to the front, so the same chunk comes back under a new lease
    assert b.task_id == a.task_id and b.lease_id != old and b.failures == 0
    assert sum(t.state == TaskState.LEASED for t in grid.tasks.values()) == 1


def test_sink_dedupes_when_fsync_fails(tmp_path, monkeypatch):
    import os as _os
    sink = JsonlGraphSink(tmp_path / "s.jsonl")
    monkeypatch.setattr(_os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        sink.write({"task_id": "t1", "x": 1})
    monkeypatch.undo()
    sink.write({"task_id": "t1", "x": 1}); sink.close()
    assert len((tmp_path / "s.jsonl").read_text().splitlines()) == 1
    assert len(JsonlGraphSink(tmp_path / "s.jsonl")._written) == 1  # survives reopen
