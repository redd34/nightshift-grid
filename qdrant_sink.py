"""Live Qdrant sink: completed chunks become searchable vectors seconds after a worker finishes.

How it fits with the JSONL sink
-------------------------------
`output_graph.jsonl` stays the system of record. `FanoutSink` writes it first, and that write
must succeed for a completion to be accepted. Qdrant is a derived index fed through
`QdrantGraphSink`, whose `write()` only enqueues, so a slow, down or misconfigured Qdrant
or Ollama can never fail, block or slow a completion. Anything missing from Qdrant can be
rebuilt from the JSONL file (`python qdrant_sink.py backfill ...`; upserts are idempotent).

Threading and backpressure
--------------------------
* `write()` runs on the coordinator's event loop. It builds a small dict and calls
  `put_nowait` on a bounded in-memory queue, which is O(1) with no I/O on the happy path.
* One daemon thread drains the queue in batches (no linger: under load batches form
  naturally, when idle a single record goes out immediately), embeds them, and upserts.
* Queue full -> the record is appended to an on-disk **spool** (JSONL) instead. We never
  block the loop and never drop. The thread also spools a batch whose upsert failed.
* Failure trips a circuit breaker with exponential backoff (0.5s -> 30s). While it's open,
  new batches go straight to the spool. When it closes, the spool is replayed *before* new
  work. The spool survives restarts and is replayed on the next run.
* `close()` (coordinator lifespan shutdown) drains the queue with one final attempt, spools
  whatever didn't make it, and joins the thread with a deadline. A hung HTTP call can't wedge
  shutdown: whatever is still unacknowledged is spooled by `close()` itself (dupes are harmless).
* Crash window: records in the in-memory queue when the *process* dies (SIGKILL) are not in
  the spool, but they are in the JSONL. `backfill` restores them.

Embedder consistency
--------------------
Vectors from different embedders live in different spaces. Mixing them in one collection
silently ruins search, even when the dimensions match (nomic-embed-text and the default hash
embedder are both 768-d). So every point carries `payload.embedder` (e.g.
`ollama:nomic-embed-text:latest`, `hash-v1:768`), and at startup we refuse a collection that
holds any point with a different or missing embedder. The embedder is chosen **once at
startup**. If Ollama dies mid-run, that's a transient failure (spool and retry), never a
silent switch to the hash embedder.

Shared-server safety (same contract as reduce_graph.py)
-------------------------------------------------------
We only ever touch collections named `nightshift_*`. We never delete, recreate or alias a
collection. An existing collection with the wrong vector config is an error, not something
to fix. The only writes are create (if absent), a payload index on our own new collection,
and upserts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

from graphrag import GraphSink
from reduce_graph import (DEFAULT_OLLAMA, OllamaEmbedder, QdrantSink, _check_collection_name,
                          _nomic_prefix, chunk_point_id, open_qdrant)

log = logging.getLogger("nightshift.qdrant_sink")

DEFAULT_COLLECTION = "nightshift_graph"
DEFAULT_HASH_DIM = 768
SOURCE_TEXT_PAYLOAD_CHARS = 4000   # payload copy (display/RAG context); the full text stays in the JSONL
EMBED_TEXT_CHARS = 6000            # ~1.5k tokens: inside nomic-embed-text's 2048-token context
EMBEDDER_FIELD = "embedder"


class QdrantConfigError(RuntimeError):
    """A permanent misconfiguration (wrong prefix, vector size or embedder). Retrying can't fix it."""


# --------------------------------------------------------------------------- embedders
class HashingEmbedder:
    """Deterministic feature-hashing embedder: unigrams + bigrams -> signed buckets -> L2 norm.

    It captures lexical overlap, not semantics. It exists so the live sink works with zero
    models installed and so tests are hermetic. It uses blake2b, not `hash()`, because
    Python's string hash is salted per process, and vectors must be identical across restarts
    and machines or the collection becomes garbage."""

    def __init__(self, dim: int = DEFAULT_HASH_DIM):
        if dim < 8:
            raise ValueError("hash embedder dim must be >= 8")
        self.dim = dim
        self.name = f"hash-v1:{dim}"

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[0-9a-z]+(?:[.\-/][0-9a-z]+)*", text.casefold())

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = self.tokens(text)
        feats = [(t, 1.0) for t in toks] + [(f"{a} {b}", 0.5) for a, b in zip(toks, toks[1:])]
        for feat, w in feats:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += w if (h >> 63) & 1 else -w
        n = math.sqrt(sum(x * x for x in v))
        if n == 0.0:  # cosine is undefined on a zero vector; give empty text a fixed unit vector
            v[0] = 1.0
            return v
        return [x / n for x in v]

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def query(self, text: str) -> list[float]:
        return self._vec(text)


class OllamaDocEmbedder:
    """`reduce_graph.OllamaEmbedder` plus nomic's asymmetric task prefixes and a stable name.

    The dimension is probed once at construction, which is also our startup availability check."""

    def __init__(self, model: str = "nomic-embed-text", url: str = DEFAULT_OLLAMA):
        self.model = model if ":" in model else f"{model}:latest"
        self.name = f"ollama:{self.model}"
        self._doc = OllamaEmbedder(model, url, prefix=_nomic_prefix(model, "search_document"))
        self._query = OllamaEmbedder(model, url, prefix=_nomic_prefix(model, "search_query"))
        self.dim = len(self._doc(["dimension probe"])[0])

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        return self._doc(texts)

    def query(self, text: str) -> list[float]:
        return self._query([text])[0]


def make_embedder(mode: str, *, model: str = "nomic-embed-text", ollama_url: str = DEFAULT_OLLAMA,
                  hash_dim: int = DEFAULT_HASH_DIM):
    """`ollama`: fail fast if the model is unusable now. `hash`: always works. `auto`: Ollama
    if it answers now, else hash, with a loud warning because that choice then sticks to the
    collection (see "Embedder consistency")."""
    if mode == "hash":
        return HashingEmbedder(hash_dim)
    try:
        return OllamaDocEmbedder(model, ollama_url)
    except Exception as e:
        if mode == "ollama":
            raise QdrantConfigError(f"embedding model {model!r} unavailable at {ollama_url}: {e} "
                                    f"(try `ollama pull {model}`, or --qdrant-embedder hash)") from e
        print(f"[qdrant] WARNING: {model!r} unavailable at {ollama_url} ({e}); falling back to the hash embedder. "
              f"A collection is locked to one embedder, so use a different --qdrant-collection for "
              f"nomic vectors later.", file=sys.stderr)
        return HashingEmbedder(hash_dim)


# --------------------------------------------------------------------------- record shaping
def slim_record(record: dict) -> dict:
    """The subset of a completion that Qdrant needs, kept JSON-serialisable for the spool."""
    return {k: record.get(k) for k in ("job_id", "chunk_index", "task_id", "worker_id", "model", "entities",
                                       "triples", "source_text", "completed_at")}


def embed_text(rec: dict) -> str:
    """Chunk text plus its extracted facts. The triples pull the vector toward the entities
    that GraphRAG queries ask about, even when the prose around them is noisy."""
    facts = "; ".join(" ".join(map(str, t)) for t in (rec.get("triples") or []))
    return f"{rec.get('source_text') or ''}\n{facts}"[:EMBED_TEXT_CHARS]


def payload(rec: dict, embedder_name: str) -> dict:
    text = rec.get("source_text") or ""
    return {"job_id": rec.get("job_id"), "chunk_index": rec.get("chunk_index"), "task_id": rec["task_id"],
            "worker_id": rec.get("worker_id"), "model": rec.get("model"), "entities": rec.get("entities") or [],
            "triples": rec.get("triples") or [], "source_text": text[:SOURCE_TEXT_PAYLOAD_CHARS],
            "source_truncated": len(text) > SOURCE_TEXT_PAYLOAD_CHARS, "completed_at": rec.get("completed_at"),
            EMBEDDER_FIELD: embedder_name}


# --------------------------------------------------------------------------- spool
class Spool:
    """Append-only JSONL retry spool. Thread-safe. Replay goes through a rename so new appends
    never race with the file being replayed."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.replaying = self.path.with_name(self.path.name + ".replaying")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.count = sum(self._lines(p) for p in (self.path, self.replaying))

    @staticmethod
    def _lines(p: Path) -> int:
        try:
            with p.open("rb") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0

    def append(self, recs: Sequence[dict]) -> None:
        if not recs:
            return
        data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            self.count += len(recs)

    def take(self) -> Optional[Path]:
        """Claim the spool for replay (an earlier unfinished replay file is resumed first)."""
        with self._lock:
            if self.replaying.exists():
                return self.replaying
            if not self.path.exists():
                return None
            os.replace(self.path, self.replaying)
            return self.replaying

    def read(self, p: Path) -> list[dict]:
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # torn tail line from a crash mid-append
        return out

    def done(self, p: Path) -> None:
        with self._lock:
            p.unlink(missing_ok=True)
            # Recount rather than subtract: torn lines were counted but never replayed.
            self.count = self._lines(self.path) + self._lines(self.replaying)


# --------------------------------------------------------------------------- the sink
class QdrantGraphSink:
    """GraphSink that asynchronously embeds and upserts completed chunks into Qdrant."""

    def __init__(self, client, embedder, collection: str = DEFAULT_COLLECTION, *, spool_path: str | os.PathLike,
                 max_queue: int = 1000, batch_size: int = 32, backoff_initial: float = 0.5, backoff_max: float = 30.0,
                 close_timeout: float = 10.0, clock: Callable[[], float] = time.monotonic):
        try:
            _check_collection_name(collection)
        except ValueError as e:
            raise QdrantConfigError(str(e)) from e
        self.client, self.embedder, self.collection = client, embedder, collection
        self.batch_size, self.backoff_initial, self.backoff_max = batch_size, backoff_initial, backoff_max
        self.close_timeout, self.clock = close_timeout, clock
        self.spool = Spool(spool_path)
        self._q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._inflight: list[dict] = []
        self._inflight_lock = threading.Lock()
        self._ready = False
        self._config_error: Optional[str] = None
        self._backoff = 0.0
        self._next_attempt = 0.0
        self._closed = False
        self.stats_counters = {"enqueued": 0, "written": 0, "failed": 0, "spooled_total": 0, "replayed": 0, "dropped": 0}
        self.last_error: Optional[str] = None
        self.last_write_latency_s: Optional[float] = None
        # Fail fast on permanent misconfiguration when Qdrant is reachable now. If it's merely
        # down, start anyway: the thread retries, and completions spool in the meantime.
        try:
            self._ensure_ready()
        except QdrantConfigError:
            raise
        except Exception as e:
            self._note_failure(e)
            print(f"[qdrant] WARNING: Qdrant not reachable at startup ({e!r}); spooling to {self.spool.path} "
                  f"and retrying in the background", file=sys.stderr)
        self._thread = threading.Thread(target=self._run, name="qdrant-sink", daemon=True)
        self._thread.start()

    # ---- GraphSink protocol
    def write(self, record: dict) -> None:
        """Never raises, never blocks: enqueue, or spool to disk if the queue is full."""
        rec = slim_record(record)
        rec["_enqueued_at"] = time.time()
        self.stats_counters["enqueued"] += 1
        try:
            if self._closed:
                raise queue.Full
            self._q.put_nowait(rec)
        except queue.Full:
            self._spool([rec])

    def close(self) -> None:
        """Flush on shutdown: one last attempt for queued records, then spool the rest."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join(self.close_timeout)
        if self._thread.is_alive():  # stuck in an HTTP call: secure everything unacknowledged
            with self._inflight_lock:
                pending = list(self._inflight)
            pending += self._drain_nowait(10**9)
            self._spool(pending)
            print(f"[qdrant] close timed out; spooled {len(pending)} records to {self.spool.path}", file=sys.stderr)
        left = self.spool.count
        if left:
            print(f"[qdrant] {left} records remain in {self.spool.path}; they will be replayed on the next start",
                  file=sys.stderr)

    # ---- introspection
    def stats(self) -> dict:
        return {"collection": self.collection, "embedder": self.embedder.name, "queued": self._q.qsize(),
                "written": self.stats_counters["written"], "failed": self.stats_counters["failed"],
                "spooled": self.spool.count, "spooled_total": self.stats_counters["spooled_total"],
                "replayed": self.stats_counters["replayed"], "dropped": self.stats_counters["dropped"],
                "healthy": self._ready and self._backoff == 0.0 and not self._config_error,
                "config_error": self._config_error, "last_error": self.last_error,
                "last_write_latency_s": self.last_write_latency_s}

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until queue and spool are empty (tests/tools). True if fully flushed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._inflight_lock:
                busy = bool(self._inflight)
            if self._q.unfinished_tasks == 0 and not busy and self.spool.count == 0:
                return True
            time.sleep(0.01)
        return False

    def search(self, text: str, k: int = 5) -> list[dict]:
        """kNN over chunk vectors (with the same embedder that wrote them)."""
        qv = self.embedder.query(text) if hasattr(self.embedder, "query") else self.embedder([text])[0]
        hits = self.client.query_points(self.collection, query=qv, limit=k, with_payload=True).points
        return [{"score": h.score, **(h.payload or {})} for h in hits]

    # ---- internals
    def _ensure_ready(self) -> None:
        if self._ready:
            return
        from qdrant_client import models

        name = self.collection
        existed = self.client.collection_exists(name)
        try:  # reuse reduce_graph's guarded create-or-validate (prefix + size/distance, never recreate)
            QdrantSink(self.client).ensure_collection(name, self.embedder.dim)
        except (RuntimeError, ValueError) as e:
            raise QdrantConfigError(str(e)) from e
        if existed:
            foreign = self.client.count(name, exact=True, count_filter=models.Filter(must_not=[
                models.FieldCondition(key=EMBEDDER_FIELD, match=models.MatchValue(value=self.embedder.name))])).count
            if foreign:
                raise QdrantConfigError(
                    f"Qdrant collection {name!r} already holds {foreign} points not written by embedder "
                    f"{self.embedder.name!r} (different embedder, or not created by the live sink). Mixing vector "
                    f"spaces breaks search; use another --qdrant-collection or the matching --qdrant-embedder.")
        else:
            try:  # cheap filtered count for the check above; ours, freshly created
                with warnings.catch_warnings():  # local mode warns that indexes are no-ops there
                    warnings.simplefilter("ignore")
                    self.client.create_payload_index(name, EMBEDDER_FIELD, field_schema=models.PayloadSchemaType.KEYWORD)
            except Exception as e:  # local/:memory: mode ignores indexes; never fatal
                log.debug("payload index not created: %r", e)
        self._ready = True

    def _drain_nowait(self, limit: int) -> list[dict]:
        out: list[dict] = []
        while len(out) < limit:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def _take_batch(self) -> list[dict]:
        try:
            first = self._q.get(timeout=0.2)
        except queue.Empty:
            return []
        return [first] + self._drain_nowait(self.batch_size - 1)

    def _done(self, batch: list[dict]) -> None:
        # task_done only after the batch is upserted or spooled, so flush() can't see a gap.
        for _ in batch:
            self._q.task_done()

    def _spool(self, recs: list[dict]) -> None:
        if not recs:
            return
        try:
            self.spool.append(recs)
            self.stats_counters["spooled_total"] += len(recs)
        except OSError as e:  # disk full too: the JSONL still has them; count loudly for backfill
            self.stats_counters["dropped"] += len(recs)
            self.last_error = f"spool write failed: {e!r}"
            print(f"[qdrant] ERROR: could not spool {len(recs)} records ({e}); run `qdrant_sink.py backfill`",
                  file=sys.stderr)

    def _note_failure(self, e: BaseException) -> None:
        self.stats_counters["failed"] += 1
        self.last_error = repr(e)[:300]
        self._backoff = min(self.backoff_max, max(self.backoff_initial, self._backoff * 2))
        self._next_attempt = self.clock() + self._backoff

    def _upsert(self, recs: list[dict]) -> None:
        from qdrant_client import models

        self._ensure_ready()
        # Dedupe within a batch (a retried completion can be enqueued twice): last one wins.
        uniq = list({r["task_id"]: r for r in recs if r.get("task_id")}.values())
        for i in range(0, len(uniq), self.batch_size):
            part = uniq[i:i + self.batch_size]
            vectors = self.embedder([embed_text(r) for r in part])
            self.client.upsert(self.collection, wait=True, points=[
                models.PointStruct(id=chunk_point_id(r["task_id"]), vector=v, payload=payload(r, self.embedder.name))
                for r, v in zip(part, vectors)])
            self.stats_counters["written"] += len(part)
        now = time.time()
        lat = [now - r["_enqueued_at"] for r in recs if isinstance(r.get("_enqueued_at"), (int, float))]
        if lat:
            self.last_write_latency_s = round(max(lat), 4)

    def _attempt(self, batch: list[dict], *, force: bool = False) -> None:
        """Replay the spool first (older work), then this batch. Any failure spools the batch."""
        if self._config_error or (not force and self._backoff and self.clock() < self._next_attempt):
            self._spool(batch)
            return
        with self._inflight_lock:
            self._inflight = list(batch)
        try:
            p = self.spool.take()
            if p is not None:
                old = self.spool.read(p)
                self._upsert(old)
                self.spool.done(p)
                self.stats_counters["replayed"] += len(old)
            if batch:
                self._upsert(batch)
            self._backoff = 0.0
            self.last_error = None
        except QdrantConfigError as e:
            self._config_error = str(e)
            self.last_error = str(e)
            print(f"[qdrant] ERROR (sink disabled, records spooled for later): {e}", file=sys.stderr)
            self._spool(batch)
        except Exception as e:
            self._note_failure(e)
            log.warning("qdrant upsert failed (%r); retry in %.1fs", e, self._backoff)
            self._spool(batch)
        finally:
            with self._inflight_lock:
                self._inflight = []

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._take_batch()
            pending_retry = self.spool.count and self._backoff and self.clock() >= self._next_attempt
            if batch or pending_retry or (self.spool.count and not self._backoff and not self._config_error):
                self._attempt(batch)
            self._done(batch)
        # Shutdown: one final forced attempt, ignoring the breaker, so a clean stop flushes.
        rest = self._drain_nowait(10**9)
        if rest or (self.spool.count and not self._config_error):
            self._attempt(rest, force=True)
        self._done(rest)


class FanoutSink:
    """Primary sink (JSONL, must succeed: its exception fails the completion so the worker
    retries) followed by best-effort secondaries whose errors are logged and counted, never raised."""

    def __init__(self, primary: GraphSink, *secondaries: GraphSink):
        self.primary, self.secondaries = primary, list(secondaries)
        self.secondary_errors = 0

    def write(self, record: dict) -> None:
        self.primary.write(record)
        for s in self.secondaries:
            try:
                s.write(record)
            except Exception as e:
                self.secondary_errors += 1
                print(f"[sink] secondary {type(s).__name__} failed: {e!r}", file=sys.stderr)

    def close(self) -> None:
        for s in [*self.secondaries, self.primary]:  # flush derived indexes before the primary
            try:
                s.close()
            except Exception as e:
                print(f"[sink] close of {type(s).__name__} failed: {e!r}", file=sys.stderr)

    def stats(self) -> dict:
        out: dict = {"primary": type(self.primary).__name__, "secondary_errors": self.secondary_errors}
        for s in self.secondaries:
            if hasattr(s, "stats"):
                out["qdrant" if isinstance(s, QdrantGraphSink) else type(s).__name__] = s.stats()
        return out


def build_qdrant_sink(location: str, collection: str = DEFAULT_COLLECTION, *, embedder_mode: str = "auto",
                      embed_model: str = "nomic-embed-text", ollama_url: str = DEFAULT_OLLAMA,
                      hash_dim: int = DEFAULT_HASH_DIM, spool_path: str | os.PathLike) -> QdrantGraphSink:
    """Coordinator wiring. `location`: http(s) URL, ':memory:' or a local path."""
    try:
        _check_collection_name(collection)
    except ValueError as e:  # before any network I/O
        raise QdrantConfigError(str(e)) from e
    embedder = make_embedder(embedder_mode, model=embed_model, ollama_url=ollama_url, hash_dim=hash_dim)
    return QdrantGraphSink(open_qdrant(location), embedder, collection, spool_path=spool_path)


# --------------------------------------------------------------------------- CLI: backfill
def backfill(sink: QdrantGraphSink, jsonl_path: str | os.PathLike, job: Optional[str] = None) -> int:
    n = 0
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("task_id") and (job is None or rec.get("job_id") == job):
            sink.write(rec)
            n += 1
    return n


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild/repair the live Qdrant index from the JSONL system of record "
                                             "(idempotent: same uuid5 point ids).")
    ap.add_argument("command", choices=["backfill"])
    ap.add_argument("--jsonl", default="output_graph.jsonl")
    ap.add_argument("--job", default=None, help="only this job_id")
    ap.add_argument("--qdrant-url", default="http://localhost:6333", help="URL, local path or :memory:")
    ap.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--qdrant-embedder", choices=["auto", "ollama", "hash"], default="auto")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--hash-dim", type=int, default=DEFAULT_HASH_DIM)
    ap.add_argument("--spool", default=None, help="retry spool path (default: <collection>.spool.jsonl next to --jsonl)")
    a = ap.parse_args(argv)
    spool = a.spool or Path(a.jsonl).resolve().parent / f"{a.qdrant_collection}.spool.jsonl"
    try:
        sink = build_qdrant_sink(a.qdrant_url, a.qdrant_collection, embedder_mode=a.qdrant_embedder,
                                 embed_model=a.embed_model, ollama_url=a.ollama, hash_dim=a.hash_dim, spool_path=spool)
    except QdrantConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    n = backfill(sink, a.jsonl, a.job)
    ok = sink.flush(timeout=600)
    sink.close()
    print(f"backfilled {n} records into {a.qdrant_collection} ({sink.stats()['written']} upserted, "
          f"{sink.spool.count} still spooled){'' if ok else ' -- NOT fully flushed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
