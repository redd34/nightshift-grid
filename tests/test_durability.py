"""Crash recovery, index invariants under random operation sequences, and HTTP auth."""
import json
import random

import pytest
from fastapi.testclient import TestClient

from coordinator import (BatchSubmitRequest, GridState, HeartbeatRequest, LeaseError, PreemptRequest, TaskCompleteRequest,
                         TaskFailRequest, TaskState, WorkerInfo, WorkerState, create_app)
from graphrag import JsonlGraphSink
from grid_store import GridStore, StoreVersionError

GOOD = {"entities": ["A", "B"], "triples": [["A", "LINKS", "B"]]}


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def boot(tmp_path, clock):
    return GridState(sink=JsonlGraphSink(tmp_path / "out.jsonl"), store=GridStore(tmp_path / "grid.db"), clock=clock)


def test_restart_recovers_queue_leases_and_results(tmp_path):
    clock = Clock()
    g = boot(tmp_path, clock)
    g.register(WorkerInfo(worker_id="w1", tier=1)); g.register(WorkerInfo(worker_id="w2", tier=1))
    job = g.submit(BatchSubmitRequest(job_name="j", chunks=[f"c{i}" for i in range(5)]))
    done = g.pull("w1"); g.complete(TaskCompleteRequest(worker_id="w1", task_id=done.task_id, lease_id=done.lease_id, result=GOOD))
    inflight = g.pull("w2")
    preempted = g.pull("w1"); g.preempt(PreemptRequest(worker_id="w1"))   # back to the FRONT of the queue
    g.store.close()                                                        # -- coordinator crashes --

    clock.advance(5)
    g2 = boot(tmp_path, clock)
    g2.check_invariants()
    st = g2.job_status(job.job_id)
    assert st["counts"] == {"PENDING": 3, "LEASED": 1, "COMPLETED": 1, "FAILED": 0}
    assert g2.tasks[done.task_id].result["triples"] == [["A", "LINKS", "B"]]
    assert g2.workers["w1"].completed_count == 1 and g2.workers["w2"].state == WorkerState.OFFLINE
    # The worker that kept running through the outage can still heartbeat and complete its lease.
    assert g2.heartbeat(HeartbeatRequest(worker_id="w2", task_id=inflight.task_id, lease_id=inflight.lease_id,
                                         state=WorkerState.BUSY_INFERENCING))["lease_valid"]
    assert g2.complete(TaskCompleteRequest(worker_id="w2", task_id=inflight.task_id, lease_id=inflight.lease_id,
                                           result=GOOD))["status"] == "accepted"
    # Queue order survives: the preempted chunk is served first.
    g2.heartbeat(HeartbeatRequest(worker_id="w1", state=WorkerState.IDLE_HARVESTING))
    assert g2.pull("w1").task_id == preempted.task_id
    g2.check_invariants()


def test_recovered_lease_of_dead_worker_expires(tmp_path):
    clock = Clock()
    g = boot(tmp_path, clock)
    g.register(WorkerInfo(worker_id="w1", tier=1))
    g.submit(BatchSubmitRequest(job_name="j", chunks=["x"])); t = g.pull("w1")
    g.store.close()
    clock.advance(100)                       # long outage; old lease_expires_at is far in the past
    g2 = boot(tmp_path, clock)
    g2.reap()
    assert g2.tasks[t.task_id].state == TaskState.LEASED   # grace period, not instant expiry
    clock.advance(13); g2.reap()
    assert g2.tasks[t.task_id].state == TaskState.PENDING and g2.tasks[t.task_id].failures == 1


def test_finished_job_and_event_seq_survive_restart(tmp_path):
    clock = Clock()
    g = boot(tmp_path, clock)
    g.register(WorkerInfo(worker_id="w1", tier=1))
    job = g.submit(BatchSubmitRequest(job_name="j", chunks=["x"])); t = g.pull("w1")
    g.complete(TaskCompleteRequest(worker_id="w1", task_id=t.task_id, lease_id=t.lease_id, result=GOOD))
    last_seq = g.events[-1].seq
    g.store.close(); clock.advance(1)
    g2 = boot(tmp_path, clock)
    assert g2.job_status(job.job_id)["done"]
    assert g2.events[-1].seq > last_seq


def test_schema_version_guard(tmp_path):
    s = GridStore(tmp_path / "g.db"); s.db.execute("PRAGMA user_version=99"); s.close()
    with pytest.raises(StoreVersionError):
        GridStore(tmp_path / "g.db")


@pytest.mark.parametrize("seed", range(25))
def test_random_operation_sequences_keep_invariants(tmp_path, seed):
    """Fuzz: random interleavings of every operation (incl. restarts) never lose, duplicate or
    double-lease a chunk, and indexes always match the task table."""
    rng, clock = random.Random(seed), Clock()
    g = boot(tmp_path, clock)
    workers = [("w1", 1), ("w2", 2), ("w3", 3)]
    sessions = {wid: g.register(WorkerInfo(worker_id=wid, tier=tier)) for wid, tier in workers}
    jobs = [g.submit(BatchSubmitRequest(job_name=f"j{i}", chunks=[f"c{i}-{k}" for k in range(6)], required_tier=rng.randint(1, 3)))
            for i in range(3)]
    held: dict[str, tuple[str, str]] = {}
    for _ in range(400):
        wid, _tier = rng.choice(workers)
        op = rng.random()
        try:
            if op < 0.35:
                t = g.pull(wid)
                held.pop(wid, None)
                if t:
                    held[wid] = (t.task_id, t.lease_id)
            elif op < 0.55 and wid in held:
                tid, lid = held.pop(wid)
                g.complete(TaskCompleteRequest(worker_id=wid, task_id=tid, lease_id=lid,
                                               result=GOOD if rng.random() > 0.1 else {"raw": "junk"}))
            elif op < 0.62 and wid in held:
                tid, lid = held.pop(wid)
                g.fail(TaskFailRequest(worker_id=wid, task_id=tid, lease_id=lid, error="boom"))
            elif op < 0.70:
                g.preempt(PreemptRequest(worker_id=wid, lease_id=held.get(wid, (None, None))[1] if rng.random() < .5 else None))
            elif op < 0.80:
                tid, lid = held.get(wid, (None, None))
                g.heartbeat(HeartbeatRequest(worker_id=wid, task_id=tid, lease_id=lid,
                                             state=rng.choice([WorkerState.IDLE_HARVESTING, WorkerState.BUSY_INFERENCING])))
            elif op < 0.90:
                clock.advance(rng.choice([1, 5, 13])); g.reap()
            elif op < 0.93:
                sessions[wid] = g.register(WorkerInfo(worker_id=wid, tier=_tier), session=sessions[wid])
                held.pop(wid, None)
            elif op < 0.96:
                g.store.close(); g = boot(tmp_path, clock)          # crash + recover
        except LeaseError:
            pass  # stale lease in our local view - exactly what fencing is for
        g.check_invariants()
    # Drain everything to completion and verify the sink holds each completed chunk exactly once.
    for _ in range(1000):
        clock.advance(13); g.reap()
        for wid, _tier in workers:
            g.heartbeat(HeartbeatRequest(worker_id=wid, state=WorkerState.IDLE_HARVESTING))
            t = g.pull(wid)
            if t:
                g.complete(TaskCompleteRequest(worker_id=wid, task_id=t.task_id, lease_id=t.lease_id, result=GOOD))
        if all(g.job_status(j.job_id)["done"] for j in jobs):
            break
    g.check_invariants()
    assert all(g.job_status(j.job_id)["done"] for j in jobs)
    recs = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    completed = {t.task_id for t in g.tasks.values() if t.state == TaskState.COMPLETED}
    assert len(recs) == len({r["task_id"] for r in recs}) and {r["task_id"] for r in recs} == completed


# ---------------------------------------------------------------- HTTP auth
@pytest.fixture
def client(tmp_path):
    app = create_app(GridState(store=GridStore(tmp_path / "g.db")), worker_token="wtok", client_token="ctok")
    with TestClient(app) as c:
        yield c


def test_auth_separates_worker_and_client_roles(client):
    W, C = {"Authorization": "Bearer wtok"}, {"Authorization": "Bearer ctok"}
    assert client.get("/healthz").status_code == 200
    assert client.post("/worker/register", json={"worker_id": "w"}).status_code == 401
    assert client.post("/worker/register", json={"worker_id": "w"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    reg = client.post("/worker/register", json={"worker_id": "w"}, headers=W)
    assert reg.status_code == 200
    WS = {**W, "X-Nightshift-Session": reg.json()["session"]}
    job = {"job_name": "j", "chunks": ["a"]}
    assert client.post("/v1/grid/mapreduce", json=job, headers=W).status_code == 401  # worker can't submit
    assert client.get("/v1/grid/status", headers=W).status_code == 401                # ...or read results
    assert client.post("/v1/grid/mapreduce", json=job, headers=C).status_code == 200
    assert client.post("/worker/pull-task", params={"worker_id": "w"}, headers=C).status_code == 401
    # The shared worker token alone can't act as, or take over, a live worker: identity is session-bound.
    assert client.post("/worker/pull-task", params={"worker_id": "w"}, headers=W).status_code == 404
    assert client.post("/worker/pull-task", params={"worker_id": "w"},
                       headers={**W, "X-Nightshift-Session": "forged"}).status_code == 404
    assert client.post("/worker/register", json={"worker_id": "w"}, headers=W).status_code == 409
    r = client.post("/worker/pull-task", params={"worker_id": "w"}, headers=WS)
    assert r.status_code == 200 and r.json()["task"]["prompt"] == "a"


def test_browser_origin_and_dns_rebinding_are_refused(tmp_path):
    app = create_app(GridState())  # auth disabled: loopback dev mode
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/grid/status", headers={"Origin": "http://evil.example"}).status_code == 403
        assert c.get("/v1/grid/status", headers={"Host": "evil.example"}).status_code == 403   # rebinding
        assert c.get("/v1/grid/status", headers={"Host": "localhost:8000"}).status_code == 200
    with TestClient(create_app(GridState(), worker_token="w", client_token="c"), base_url="http://10.0.0.5") as c:
        # With auth on, a LAN Host is fine (the token is the protection), but browsers still can't cross origins.
        assert c.get("/healthz").status_code == 200
        assert c.get("/healthz", headers={"Origin": "http://evil.example"}).status_code == 403


def test_reused_token_file_must_be_private_and_not_a_symlink(tmp_path):
    import os
    from coordinator import ensure_auto_tokens
    from grid_auth import GridAuthError
    d = tmp_path / "state"; d.mkdir(mode=0o700)
    ensure_auto_tokens(d, None, None)
    (d / "worker.token").chmod(0o644)                    # world-readable -> refuse
    with pytest.raises(GridAuthError, match="mode 0600"):
        ensure_auto_tokens(d, None, None)
    (d / "worker.token").unlink()
    (tmp_path / "planted").write_text("attacker-known\n"); (tmp_path / "planted").chmod(0o600)
    os.symlink(tmp_path / "planted", d / "worker.token")  # planted symlink -> refuse
    with pytest.raises(GridAuthError, match="symlink"):
        ensure_auto_tokens(d, None, None)
    os.unlink(d / "worker.token"); d.chmod(0o777)          # world-writable dir -> refuse
    with pytest.raises(GridAuthError, match="not group/world-writable"):
        ensure_auto_tokens(d, None, None)
    d.chmod(0o700)


def test_request_limits(client):
    C = {"Authorization": "Bearer ctok"}
    assert client.post("/v1/grid/mapreduce", json={"job_name": "j", "chunks": []}, headers=C).status_code == 422
    assert client.post("/v1/grid/mapreduce", json={"job_name": "j", "chunks": ["x" * 100_001]}, headers=C).status_code == 422


def test_refuses_insecure_lan_bind(monkeypatch):
    import coordinator
    for k in ("GRID_TOKEN", "GRID_WORKER_TOKEN", "GRID_CLIENT_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="Refusing to bind"):
        coordinator.main(["--host", "0.0.0.0", "--db", "none", "--output", "none"])


def test_recovery_does_not_open_a_window_to_hijack_live_worker_ids(tmp_path):
    from coordinator import WorkerIdInUse
    clock = Clock()
    g = boot(tmp_path, clock)
    session = g.register(WorkerInfo(worker_id="w1", tier=1))
    g.store.close(); clock.advance(2)
    g2 = boot(tmp_path, clock)                       # every recovered worker is OFFLINE now...
    assert g2.workers["w1"].state == WorkerState.OFFLINE
    with pytest.raises(WorkerIdInUse):               # ...but its id is still protected
        g2.register(WorkerInfo(worker_id="w1", tier=1))
    assert g2.verify_session("w1", session)          # the real worker carries on with its session
    clock.advance(13)                                # no contact for worker_timeout -> id released
    assert g2.register(WorkerInfo(worker_id="w1", tier=1))
