"""Live Qdrant sink: in-process Qdrant (:memory:) + the hash embedder, so these tests are hermetic."""
import json
import threading
import time

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient, models  # noqa: E402

from coordinator import BatchSubmitRequest, GridState, TaskCompleteRequest, WorkerInfo, create_app  # noqa: E402
from graphrag import JsonlGraphSink  # noqa: E402
from qdrant_sink import (FanoutSink, HashingEmbedder, QdrantConfigError, QdrantGraphSink, backfill,  # noqa: E402
                         build_qdrant_sink, chunk_point_id)

COLL = "nightshift_test_graph"


def rec(i, **kw):
    return {"job_id": "j1", "task_id": f"task-{i}", "chunk_index": i, "worker_id": "w1", "model": "mock",
            "entities": [f"Pump-{i}", "Bearing"], "triples": [[f"Pump-{i}", "HAS_PART", "Bearing"]],
            "source_text": f"Pump-{i} vibration exceeded limits; bearing replaced.", "completed_at": 1.0, **kw}


class FlakyClient:
    """Wraps a real client; `upsert` raises while `down` is set (simulates a Qdrant outage)."""

    def __init__(self, inner):
        self.inner, self.down, self.upsert_calls = inner, False, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def upsert(self, *a, **k):
        self.upsert_calls += 1
        if self.down:
            raise ConnectionError("qdrant down")
        return self.inner.upsert(*a, **k)


@pytest.fixture
def client():
    return QdrantClient(location=":memory:")


def make(client, tmp_path, **kw):
    kw.setdefault("backoff_initial", 0.05)
    kw.setdefault("backoff_max", 0.2)
    return QdrantGraphSink(client, kw.pop("embedder", HashingEmbedder(64)), kw.pop("collection", COLL),
                           spool_path=tmp_path / "spool.jsonl", **kw)


def count(client, coll=COLL):
    return client.count(coll, exact=True).count


# ---------------------------------------------------------------- embedder
def test_hash_embedder_deterministic_normalized_and_lexical():
    e = HashingEmbedder(128)
    a, b, c = e(["pump bearing vibration", "pump bearing vibration", "quarterly revenue forecast"])
    assert a == b and len(a) == 128
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    assert dot(a, e.query("bearing vibration on the pump")) > dot(c, e.query("bearing vibration on the pump"))
    assert e([""])[0][0] == 1.0  # never a zero vector (cosine undefined)
    assert HashingEmbedder(768).name == "hash-v1:768"


# ---------------------------------------------------------------- happy path
def test_write_is_async_upserts_payload_and_is_searchable(client, tmp_path):
    s = make(client, tmp_path)
    for i in range(10):
        s.write(rec(i))
    s.write(rec(3))  # duplicate (retried completion): same point id, no duplicate
    assert s.flush(5)
    assert count(client) == 10
    p = client.retrieve(COLL, [chunk_point_id("task-3")], with_payload=True)[0].payload
    assert {k: p[k] for k in ("job_id", "chunk_index", "task_id", "worker_id", "model", "embedder")} == \
        {"job_id": "j1", "chunk_index": 3, "task_id": "task-3", "worker_id": "w1", "model": "mock", "embedder": "hash-v1:64"}
    assert p["triples"] == [["Pump-3", "HAS_PART", "Bearing"]] and p["source_text"].startswith("Pump-3")
    assert s.search("Pump-7 vibration", k=1)[0]["task_id"] == "task-7"
    st = s.stats()
    assert st["written"] in (10, 11)  # the duplicate is a no-op overwrite (same id)
    assert st["queued"] == 0 and st["spooled"] == 0 and st["failed"] == 0 and st["healthy"]
    s.close()


def test_source_text_truncated_in_payload(client, tmp_path):
    s = make(client, tmp_path)
    s.write(rec(1, source_text="x" * 10_000))
    assert s.flush(5)
    p = client.retrieve(COLL, [chunk_point_id("task-1")], with_payload=True)[0].payload
    assert len(p["source_text"]) == 4000 and p["source_truncated"] is True
    s.close()


# ---------------------------------------------------------------- safety guards
@pytest.mark.parametrize("name", ["graph", "venues", "events", "nightshiftgraph", "Nightshift_graph"])
def test_prefix_guard(client, tmp_path, name):
    with pytest.raises(QdrantConfigError, match="nightshift_"):
        make(client, tmp_path, collection=name)
    with pytest.raises(QdrantConfigError):
        build_qdrant_sink(":memory:", name, embedder_mode="hash", spool_path=tmp_path / "s")
    assert not client.collection_exists(name)


def test_refuses_mismatched_vector_size(client, tmp_path):
    client.create_collection(COLL, vectors_config=models.VectorParams(size=32, distance=models.Distance.COSINE))
    with pytest.raises(QdrantConfigError, match="size=32"):
        make(client, tmp_path, embedder=HashingEmbedder(64))
    assert client.get_collection(COLL).config.params.vectors.size == 32  # untouched, never recreated


def test_refuses_mixing_embedders_with_same_dimension(client, tmp_path):
    """Same dim (e.g. nomic-768 vs hash-768) must still be refused: different vector spaces."""
    s = make(client, tmp_path, embedder=HashingEmbedder(64))
    s.write(rec(1)); assert s.flush(5); s.close()

    class OtherEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__(64); self.name = "ollama:other:latest"

    with pytest.raises(QdrantConfigError, match="not written by embedder"):
        make(client, tmp_path, embedder=OtherEmbedder())
    # A collection created by someone else (no embedder field) is refused too.
    client.create_collection("nightshift_foreign", vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE))
    client.upsert("nightshift_foreign", points=[models.PointStruct(id=1, vector=[1.0] + [0.0] * 63, payload={})])
    with pytest.raises(QdrantConfigError):
        make(client, tmp_path, collection="nightshift_foreign")
    # Same embedder again is fine (restart).
    make(client, tmp_path, embedder=HashingEmbedder(64)).close()


# ---------------------------------------------------------------- failure injection
def test_outage_spools_then_replays_without_loss(client, tmp_path):
    flaky = FlakyClient(client)
    s = make(flaky, tmp_path)
    flaky.down = True
    for i in range(20):
        s.write(rec(i))
    deadline = time.time() + 5
    while s.stats()["spooled"] < 20 and time.time() < deadline:
        time.sleep(0.01)
    st = s.stats()
    assert st["spooled"] == 20 and st["failed"] >= 1 and not st["healthy"] and "qdrant down" in st["last_error"]
    assert (tmp_path / "spool.jsonl").exists() or (tmp_path / "spool.jsonl.replaying").exists()
    flaky.down = False
    assert s.flush(5), s.stats()
    assert count(client) == 20 and s.stats()["spooled"] == 0 and s.stats()["replayed"] == 20
    s.close()


def test_completions_succeed_while_qdrant_raises(client, tmp_path):
    """The real contract: a broken Qdrant never fails or blocks a coordinator completion."""
    flaky = FlakyClient(client)
    q = make(flaky, tmp_path)
    flaky.down = True
    g = GridState(sink=FanoutSink(JsonlGraphSink(tmp_path / "out.jsonl"), q))
    g.register(WorkerInfo(worker_id="w1"))
    g.submit(BatchSubmitRequest(job_name="j", chunks=[f"Pump-{i} uses Bearing B-{i}" for i in range(5)]))
    t0 = time.perf_counter()
    for _ in range(5):
        t = g.pull("w1")
        r = g.complete(TaskCompleteRequest(worker_id="w1", task_id=t.task_id, lease_id=t.lease_id,
                                           result={"entities": ["Pump", "Bearing"], "triples": [["Pump", "USES", "Bearing"]]}))
        assert r["status"] == "accepted"
    assert time.perf_counter() - t0 < 1.0
    assert g.counts["COMPLETED"] == 5
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 5      # system of record complete
    assert g.status()["sink"]["qdrant"]["written"] == 0
    flaky.down = False
    assert q.flush(5)
    assert count(client) == 5
    st = g.status()["sink"]["qdrant"]
    assert st["written"] == 5 and st["spooled"] == 0
    g.sink.close()


def test_secondary_write_exception_is_contained(tmp_path):
    class Boom:
        def write(self, r): raise RuntimeError("boom")
        def close(self): raise RuntimeError("boom")

    f = FanoutSink(JsonlGraphSink(tmp_path / "o.jsonl"), Boom())
    f.write(rec(1)); f.close()
    assert f.secondary_errors == 1 and len((tmp_path / "o.jsonl").read_text().splitlines()) == 1


def test_primary_failure_propagates(tmp_path, client):
    class Bad:
        def write(self, r): raise OSError("disk full")
        def close(self): pass

    q = make(client, tmp_path)
    with pytest.raises(OSError):
        FanoutSink(Bad(), q).write(rec(1))
    assert q.stats_counters["enqueued"] == 0  # nothing indexed that isn't in the system of record
    q.close()


def test_queue_full_spills_to_disk_never_blocks(client, tmp_path):
    flaky = FlakyClient(client)
    flaky.down = True
    s = make(flaky, tmp_path, max_queue=2, backoff_initial=5, backoff_max=5)
    t0 = time.perf_counter()
    for i in range(200):
        s.write(rec(i))
    assert time.perf_counter() - t0 < 2.0
    assert s.stats()["spooled"] + s.stats()["queued"] >= 198 and s.stats()["dropped"] == 0
    flaky.down = False
    s._next_attempt = 0  # skip the 5s breaker for the test
    assert s.flush(10), s.stats()
    assert count(client) == 200
    s.close()


def test_qdrant_down_at_startup_then_recovers(client, tmp_path):
    class DownAtStart(FlakyClient):
        def collection_exists(self, name):
            if self.down:
                raise ConnectionError("refused")
            return self.inner.collection_exists(name)

    c = DownAtStart(client); c.down = True
    s = make(c, tmp_path)                 # must not raise: completions spool meanwhile
    s.write(rec(1))
    time.sleep(0.1)
    c.down = False
    assert s.flush(5) and count(client) == 1
    s.close()


# ---------------------------------------------------------------- shutdown / restart
def test_close_flushes_queue(client, tmp_path):
    s = make(client, tmp_path)
    for i in range(50):
        s.write(rec(i))
    s.close()                              # no explicit flush
    assert count(client) == 50 and s.spool.count == 0


def test_spool_survives_restart_and_is_replayed(client, tmp_path):
    flaky = FlakyClient(client); flaky.down = True
    s = make(flaky, tmp_path)
    for i in range(7):
        s.write(rec(i))
    s.close()                              # final attempt fails -> spooled to disk
    assert s.spool.count == 7 and count(client) == 0
    with open(tmp_path / "spool.jsonl", "a") as f:
        f.write('{"torn": ')               # a crash mid-append must not wedge replay
    s2 = make(client, tmp_path)            # next coordinator run
    assert s2.spool.count == 8
    assert s2.flush(5), s2.stats()
    assert count(client) == 7
    s2.close()


def test_close_with_hung_upsert_spools_inflight(client, tmp_path):
    release = threading.Event()

    class Hung(FlakyClient):
        def upsert(self, *a, **k):
            release.wait(5)
            return self.inner.upsert(*a, **k)

    s = make(Hung(client), tmp_path, close_timeout=0.2)
    s.write(rec(1)); s.write(rec(2))
    time.sleep(0.1)
    s.close()
    assert s.spool.count >= 2              # nothing unacknowledged is lost
    release.set()


def test_backfill_from_jsonl(client, tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("".join(json.dumps(rec(i)) + "\n" for i in range(5)) + "garbage\n")
    s = make(client, tmp_path)
    assert backfill(s, p) == 5 and s.flush(5)
    assert count(client) == 5
    s.close()


def test_lifespan_closes_sink_and_status_exposes_stats(client, tmp_path):
    from fastapi.testclient import TestClient

    q = make(client, tmp_path)
    grid = GridState(sink=FanoutSink(JsonlGraphSink(tmp_path / "o.jsonl"), q))
    with TestClient(create_app(grid)) as c:
        c.post("/worker/register", json={"worker_id": "w"})
        c.post("/v1/grid/mapreduce", json={"job_name": "j", "chunks": ["Alpha links Beta"]})
        t = c.post("/worker/pull-task", params={"worker_id": "w"}).json()["task"]
        assert c.post("/worker/complete", json={"worker_id": "w", "task_id": t["task_id"], "lease_id": t["lease_id"],
                                                "result": {"entities": ["Alpha", "Beta"],
                                                           "triples": [["Alpha", "LINKS", "Beta"]]}}).status_code == 200
        st = c.get("/v1/grid/status").json()["sink"]["qdrant"]
        assert set(st) >= {"queued", "written", "failed", "spooled", "embedder", "collection"}
    assert q._closed and count(client) == 1
