"""Edge AI Compute Grid - Coordinator (control plane).

Design notes
------------
* All grid state lives in `GridState`, a plain synchronous class with an injectable clock.
  FastAPI handlers are thin adapters. Because handlers never `await` while mutating state,
  every state transition is atomic on the single asyncio event loop - no locks needed.
* Every task state change goes through `_set_state()` / `_lease()` / `_unlease()`, which keep
  the secondary indexes (per-tier pending queues, live leases, leases-by-worker, per-job
  counters) consistent. Hot paths (pull, preempt, reap, status) are O(leases), not O(tasks).
* Durability: with a `GridStore`, each public mutation is journaled to SQLite in one
  transaction (see grid_store.py). On restart, jobs/tasks/workers are recovered, queue order is
  preserved, and in-flight leases get a fresh TTL so workers that kept running can still
  heartbeat and complete them - a coordinator restart costs zero completed work.
* Leases are *fenced*: every lease gets a fresh `lease_id`. Heartbeats, completions and
  failures must present the current lease_id, so a zombie worker (partitioned, then back)
  can never extend or complete a lease that was already re-issued to someone else.
* Preemption (human came back) is not the chunk's fault: it re-queues immediately and does
  not count toward the poison-chunk budget. Errors and lease expiries do; after
  `MAX_FAILURES` the chunk is dead-lettered (FAILED) instead of looping forever.
* Re-queued chunks go to the *front* of the queue so recovery cuts tail latency.
* Scheduling is capability-aware: a worker serves the highest-tier work it is capable of
  first, so Tier 3 silicon is not wasted on embeddings while Tier 3 work is waiting.
* Auth: separate bearer tokens for workers (`/worker/*`) and clients (`/v1/*`), so a
  compromised office laptop can lease work but cannot submit jobs or read results.
  `--auto-token` mints and persists them (0600 files next to the db) for zero-config setups.
* Discovery (`discovery.py`): with `--advertise` a UDP responder lets `worker --discover` find
  this coordinator. Replies carry an HMAC proof keyed by the worker token, so a rogue host can't
  impersonate the coordinator and harvest worker tokens.
* Sinks: the JSONL file is the system of record; `--qdrant-url`/`--qdrant-memory` adds a live,
  non-blocking Qdrant index (`qdrant_sink.py`) whose failures never fail a completion.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hmac
import ipaddress
import os
import secrets
import sys
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, TextIO

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from discovery import DISCOVERY_PORT, DiscoveryResponder, default_advertise_url
from graphrag import ExtractionError, GraphSink, JsonlGraphSink, NullSink, normalize_extraction
from grid_auth import CLIENT_TOKEN_ENV, WORKER_TOKEN_ENV, GridAuthError, resolve_token
from grid_store import GridStore

LEASE_TTL_SECONDS = 12.0
HEARTBEAT_INTERVAL_SECONDS = 3.0
WORKER_TIMEOUT_SECONDS = 12.0
REAPER_INTERVAL_SECONDS = 0.5
MAX_FAILURES = 3
MAX_EVENTS = 2000
MAX_CHUNKS_PER_JOB = 50_000
MAX_CHUNK_CHARS = 100_000


class TaskState(str, Enum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # dead-lettered after MAX_FAILURES


TERMINAL = (TaskState.COMPLETED, TaskState.FAILED)


class WorkerState(str, Enum):
    IDLE_HARVESTING = "IDLE_HARVESTING"
    BUSY_INFERENCING = "BUSY_INFERENCING"
    PREEMPTED_USER_ACTIVE = "PREEMPTED_USER_ACTIVE"
    OFFLINE = "OFFLINE"


# ---------------------------------------------------------------------------- models
class WorkerInfo(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    hostname: str = ""
    tier: int = Field(1, ge=1, le=3)
    ram_free_gb: float = 0.0
    cpu_percent: float = 0.0
    model_name: str = ""
    state: WorkerState = WorkerState.IDLE_HARVESTING
    current_task_id: Optional[str] = None
    tokens_per_sec: float = 0.0
    completed_count: int = 0
    preempted_count: int = 0
    failed_count: int = 0
    last_seen: float = 0.0


class GridTask(BaseModel):
    task_id: str
    job_id: str
    chunk_index: int
    prompt: str
    required_tier: int = 1
    state: TaskState = TaskState.PENDING
    queue_seq: int = 0         # position in its tier queue; negative = re-queued to the front
    assigned_worker: Optional[str] = None
    lease_id: Optional[str] = None
    lease_expires_at: float = 0.0
    attempts: int = 0          # total leases issued
    failures: int = 0          # errors + lease expiries (preemptions excluded)
    preemptions: int = 0
    last_error: Optional[str] = None
    result: Optional[dict] = None
    completed_by: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


class Job(BaseModel):
    job_id: str
    job_name: str
    task_ids: list[str]
    required_tier: int
    submitted_at: float
    finished_at: Optional[float] = None


class GridEvent(BaseModel):
    seq: int
    ts: float
    kind: str
    msg: str
    worker_id: Optional[str] = None
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    chunk_index: Optional[int] = None
    data: dict = Field(default_factory=dict)


class BatchSubmitRequest(BaseModel):
    job_name: str = Field(min_length=1, max_length=256)
    chunks: list[str] = Field(min_length=1, max_length=MAX_CHUNKS_PER_JOB)
    required_tier: int = Field(1, ge=1, le=3)


class HeartbeatRequest(BaseModel):
    worker_id: str
    task_id: Optional[str] = None
    lease_id: Optional[str] = None
    tokens_per_sec: float = 0.0
    cpu_percent: float = 0.0
    ram_free_gb: float = 0.0
    state: WorkerState = WorkerState.IDLE_HARVESTING


class PreemptRequest(BaseModel):
    worker_id: str
    reason: str = "user_activity"
    detected_at: Optional[float] = None  # worker wall clock when activity was seen
    lease_id: Optional[str] = None       # if set, only this lease is handed back (fences late preempts)


class TaskCompleteRequest(BaseModel):
    worker_id: str
    task_id: str
    lease_id: str
    result: dict
    tokens_per_sec: float = 0.0


class TaskFailRequest(BaseModel):
    worker_id: str
    task_id: str
    lease_id: str
    error: str


class LeaseError(Exception):
    """Caller does not hold the current lease on the task."""


def mutation(fn):
    """Journal every row the operation dirtied in one transaction, even if it raised midway
    (in-memory state already changed, so disk must follow)."""
    @functools.wraps(fn)
    def wrapper(self: "GridState", *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._flush()
    return wrapper


# ---------------------------------------------------------------------------- core state machine
class GridState:
    def __init__(
        self,
        sink: GraphSink | None = None,
        store: GridStore | None = None,
        clock: Callable[[], float] = time.time,
        lease_ttl: float = LEASE_TTL_SECONDS,
        worker_timeout: float = WORKER_TIMEOUT_SECONDS,
        max_failures: int = MAX_FAILURES,
    ):
        self.sink = sink or NullSink()
        self.store = store
        self.clock = clock
        self.lease_ttl = lease_ttl
        self.worker_timeout = worker_timeout
        self.max_failures = max_failures
        self.workers: dict[str, WorkerInfo] = {}
        self.tasks: dict[str, GridTask] = {}
        self.jobs: dict[str, Job] = {}
        # Secondary indexes - only ever mutated through _set_state/_lease/_unlease/_enqueue.
        self.pending: dict[int, deque[str]] = {1: deque(), 2: deque(), 3: deque()}
        self.leased: set[str] = set()
        self.leases_by_worker: dict[str, set[str]] = {}
        self.counts: Counter[str] = Counter()
        self.job_counts: dict[str, Counter[str]] = {}
        self._front_seq = 0
        self._back_seq = 0
        self.events: deque[GridEvent] = deque(maxlen=MAX_EVENTS)
        # Event seqs start at boot-time ms so they keep increasing across coordinator restarts
        # (clients page with ?since_seq= and must not miss post-restart events).
        self._seq = int(self.clock() * 1000)
        self._dirty_tasks: set[str] = set()
        self._dirty_jobs: set[str] = set()
        self._dirty_workers: set[str] = set()
        if store:
            self._recover()

    # ---- persistence
    def _flush(self) -> None:
        if not self.store:
            self._dirty_tasks.clear(); self._dirty_jobs.clear(); self._dirty_workers.clear()
            return
        tasks = [(t.task_id, t.job_id, t.state.value, t.model_dump_json())
                 for t in (self.tasks[i] for i in self._dirty_tasks)]
        jobs = [(j, self.jobs[j].model_dump_json()) for j in self._dirty_jobs]
        workers = [(w, self.workers[w].model_dump_json()) for w in self._dirty_workers if w in self.workers]
        self._dirty_tasks.clear(); self._dirty_jobs.clear(); self._dirty_workers.clear()
        self.store.commit(jobs, tasks, workers)

    def _recover(self) -> None:
        assert self.store is not None
        now = self.clock()
        for j in sorted(self.store.load_jobs(), key=lambda j: j["submitted_at"]):
            job = Job(**j)
            self.jobs[job.job_id] = job
            self.job_counts[job.job_id] = Counter()
        pending: list[GridTask] = []
        live_leases = 0
        for raw in self.store.load_tasks():
            t = GridTask(**raw)
            self.tasks[t.task_id] = t
            self.counts[t.state.value] += 1
            self.job_counts.setdefault(t.job_id, Counter())[t.state.value] += 1
            if t.state == TaskState.PENDING:
                pending.append(t)
            elif t.state == TaskState.LEASED:
                # Grace period: the holder may still be running and will heartbeat/complete.
                t.lease_expires_at = now + self.lease_ttl
                self.leased.add(t.task_id)
                self.leases_by_worker.setdefault(t.assigned_worker or "", set()).add(t.task_id)
                live_leases += 1
        for t in sorted(pending, key=lambda t: t.queue_seq):
            self.pending[t.required_tier].append(t.task_id)
        seqs = [t.queue_seq for t in self.tasks.values()]
        self._front_seq, self._back_seq = (min(seqs, default=0), max(seqs, default=0))
        for raw in self.store.load_workers():
            w = WorkerInfo(**raw)
            # Unknown liveness until it heartbeats; last_seen=now gives it a full timeout window.
            w.state, w.last_seen, w.tokens_per_sec = WorkerState.OFFLINE, now, 0.0
            held = self.leases_by_worker.get(w.worker_id)
            w.current_task_id = next(iter(held)) if held else None
            self.workers[w.worker_id] = w
        if self.tasks or self.workers:
            self.log("RECOVERED", f"Recovered {len(self.jobs)} jobs, {len(self.tasks)} tasks "
                     f"({self.counts['PENDING']} pending, {live_leases} in-flight leases), {len(self.workers)} workers",
                     jobs=len(self.jobs), tasks=len(self.tasks), live_leases=live_leases)

    # ---- index-maintaining primitives
    def _set_state(self, task: GridTask, new: TaskState) -> None:
        old = task.state
        if old == new:
            return
        assert old not in TERMINAL, f"task {task.task_id} cannot leave terminal state {old}"
        self.counts[old.value] -= 1
        self.counts[new.value] += 1
        jc = self.job_counts[task.job_id]
        jc[old.value] -= 1
        jc[new.value] += 1
        task.state = new
        self._dirty_tasks.add(task.task_id)
        if new in TERMINAL:
            self._maybe_finish_job(task.job_id)

    def _lease(self, task: GridTask, worker_id: str, now: float) -> None:
        self._set_state(task, TaskState.LEASED)
        task.assigned_worker = worker_id
        task.lease_id = uuid.uuid4().hex
        task.lease_expires_at = now + self.lease_ttl
        task.attempts += 1
        self.leased.add(task.task_id)
        self.leases_by_worker.setdefault(worker_id, set()).add(task.task_id)

    def _unlease(self, task: GridTask) -> None:
        self.leased.discard(task.task_id)
        holder = self.leases_by_worker.get(task.assigned_worker or "")
        if holder:
            holder.discard(task.task_id)
        self._release_worker(task.assigned_worker, task.task_id)
        task.assigned_worker = None
        task.lease_id = None
        task.lease_expires_at = 0.0
        self._dirty_tasks.add(task.task_id)

    def _enqueue(self, task: GridTask, *, front: bool) -> None:
        if front:
            self._front_seq -= 1
            task.queue_seq = self._front_seq
            self.pending[task.required_tier].appendleft(task.task_id)
        else:
            self._back_seq += 1
            task.queue_seq = self._back_seq
            self.pending[task.required_tier].append(task.task_id)
        self._dirty_tasks.add(task.task_id)

    def _requeue(self, task: GridTask) -> None:
        self._unlease(task)
        self._set_state(task, TaskState.PENDING)
        self._enqueue(task, front=True)

    def _worker_leases(self, worker_id: str) -> list[GridTask]:
        return [self.tasks[t] for t in list(self.leases_by_worker.get(worker_id, ()))]

    # ---- helpers
    def log(self, kind: str, msg: str, task: GridTask | None = None, worker_id: str | None = None,
            job_id: str | None = None, **data) -> None:
        self._seq += 1
        self.events.append(GridEvent(
            seq=self._seq, ts=self.clock(), kind=kind, msg=msg, worker_id=worker_id,
            task_id=task.task_id if task else None, job_id=task.job_id if task else job_id,
            chunk_index=task.chunk_index if task else None, data=data,
        ))

    def _release_worker(self, worker_id: str | None, task_id: str) -> None:
        w = self.workers.get(worker_id) if worker_id else None
        if w and w.current_task_id == task_id:
            w.current_task_id = None
            if w.state == WorkerState.BUSY_INFERENCING:
                w.state = WorkerState.IDLE_HARVESTING

    def _record_failure(self, task: GridTask, error: str) -> None:
        task.failures += 1
        task.last_error = error
        if task.failures >= self.max_failures:
            self._unlease(task)
            self.log("DEAD_LETTER", f"Chunk #{task.chunk_index} dead-lettered after {task.failures} failures: {error}", task)
            self._set_state(task, TaskState.FAILED)
        else:
            self._requeue(task)

    def _maybe_finish_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        jc = self.job_counts[job_id]
        if job and job.finished_at is None and jc["COMPLETED"] + jc["FAILED"] == len(job.task_ids):
            job.finished_at = self.clock()
            self._dirty_jobs.add(job_id)
            self.log("JOB_FINISHED", f"Job '{job.job_name}' finished in {job.finished_at - job.submitted_at:.1f}s "
                     f"({jc['COMPLETED']} completed, {jc['FAILED']} dead-lettered)", job_id=job_id)

    def _get_lease(self, worker_id: str, task_id: str, lease_id: str) -> GridTask:
        task = self.tasks.get(task_id)
        if not task or task.state != TaskState.LEASED or task.lease_id != lease_id or task.assigned_worker != worker_id:
            raise LeaseError(f"{worker_id} does not hold lease {lease_id[:8] if lease_id else None} on {task_id[:8]}")
        return task

    def _worker(self, worker_id: str) -> WorkerInfo:
        w = self.workers.get(worker_id)
        if not w:
            raise KeyError(worker_id)
        return w

    # ---- worker API
    @mutation
    def register(self, info: WorkerInfo) -> None:
        prev = self.workers.get(info.worker_id)
        if prev:  # re-registration (worker restart / lost contact): keep counters
            info.completed_count, info.preempted_count, info.failed_count = (
                prev.completed_count, prev.preempted_count, prev.failed_count)
            # A freshly (re)started worker process can't be holding its old leases any more.
            for t in self._worker_leases(info.worker_id):
                self.log("LEASE_RECLAIMED", f"{info.worker_id} re-registered; re-queued chunk #{t.chunk_index}", t, info.worker_id)
                self._requeue(t)
        info.last_seen = self.clock()
        info.current_task_id = None
        self.workers[info.worker_id] = info
        self._dirty_workers.add(info.worker_id)
        self.log("NODE_JOINED", f"{info.worker_id} joined (Tier {info.tier}, {info.ram_free_gb:.1f}GB free, {info.model_name})",
                 worker_id=info.worker_id, tier=info.tier)

    @mutation
    def pull(self, worker_id: str) -> GridTask | None:
        w = self._worker(worker_id)
        now = self.clock()
        w.last_seen = now
        if w.state == WorkerState.PREEMPTED_USER_ACTIVE:
            return None  # worker clears this with an IDLE heartbeat when the human leaves
        # One lease per worker: if it's pulling, it has abandoned any lease it still holds
        # (e.g. a cancelled inference). Hand it back now instead of waiting for TTL expiry.
        for t in self._worker_leases(worker_id):
            self.log("LEASE_RECLAIMED", f"{worker_id} pulled again; re-queued abandoned chunk #{t.chunk_index}", t, worker_id)
            self._requeue(t)
        for tier in range(w.tier, 0, -1):
            q = self.pending[tier]
            while q:
                task = self.tasks[q.popleft()]
                if task.state != TaskState.PENDING:  # defensive: never double-lease
                    continue
                self._lease(task, worker_id, now)
                w.state = WorkerState.BUSY_INFERENCING
                w.current_task_id = task.task_id
                self.log("LEASED", f"Chunk #{task.chunk_index} -> {worker_id} (attempt {task.attempts})", task, worker_id,
                         attempt=task.attempts)
                return task
        w.state = WorkerState.IDLE_HARVESTING
        w.current_task_id = None
        return None

    def heartbeat(self, hb: HeartbeatRequest) -> dict:
        """Not journaled: telemetry and lease extensions are re-established after recovery."""
        w = self._worker(hb.worker_id)
        now = self.clock()
        w.last_seen = now
        w.tokens_per_sec, w.cpu_percent, w.ram_free_gb = hb.tokens_per_sec, hb.cpu_percent, hb.ram_free_gb
        if w.state == WorkerState.OFFLINE:
            self.log("NODE_BACK", f"{w.worker_id} is back online", worker_id=w.worker_id)
        w.state = hb.state
        lease_valid = None
        if hb.task_id:
            task = self.tasks.get(hb.task_id)
            lease_valid = bool(task and task.state == TaskState.LEASED and task.lease_id == hb.lease_id
                               and task.assigned_worker == hb.worker_id)
            if lease_valid:
                task.lease_expires_at = now + self.lease_ttl
                w.current_task_id = task.task_id
        # Tell the worker if its lease is gone so it can stop burning compute on zombie work.
        return {"ok": True, "lease_valid": lease_valid}

    @mutation
    def preempt(self, req: PreemptRequest) -> int:
        w = self._worker(req.worker_id)
        now = self.clock()
        w.state = WorkerState.PREEMPTED_USER_ACTIVE
        w.current_task_id = None
        w.tokens_per_sec = 0.0
        w.preempted_count += 1
        w.last_seen = now
        self._dirty_workers.add(w.worker_id)
        reclaimed = 0
        for task in self._worker_leases(req.worker_id):
            if req.lease_id is not None and task.lease_id != req.lease_id:
                continue
            task.preemptions += 1
            self._requeue(task)
            reclaimed += 1
            latency = (now - req.detected_at) if req.detected_at else None
            self.log("PREEMPT_RECLAIM",
                     f"{req.worker_id} yielded ({req.reason}); re-queued chunk #{task.chunk_index}",
                     task, req.worker_id, reason=req.reason, detected_at=req.detected_at, notify_latency_s=latency)
        if not reclaimed:
            self.log("PREEMPT_IDLE", f"{req.worker_id} paused ({req.reason}), no lease held", worker_id=req.worker_id,
                     reason=req.reason, detected_at=req.detected_at)
        return reclaimed

    @mutation
    def complete(self, req: TaskCompleteRequest) -> dict:
        task = self.tasks.get(req.task_id)
        # Idempotent retry of a completion that already landed (e.g. response lost on Wi-Fi).
        if task and task.state == TaskState.COMPLETED and task.completed_by == req.worker_id and task.result is not None \
                and task.result.get("_lease_id") == req.lease_id:
            return {"status": "duplicate"}
        task = self._get_lease(req.worker_id, req.task_id, req.lease_id)
        try:
            extraction = normalize_extraction(req.result, source_text=task.prompt)
        except ExtractionError as e:
            self._fail(task, req.worker_id, f"invalid extraction: {e}")
            return {"status": "rejected", "error": str(e)}
        now = self.clock()
        w = self.workers.get(req.worker_id)
        record = {
            "job_id": task.job_id, "task_id": task.task_id, "chunk_index": task.chunk_index,
            "worker_id": req.worker_id, "model": w.model_name if w else None,
            "attempts": task.attempts, "preemptions": task.preemptions,
            "completed_at": now, "tokens_per_sec": req.tokens_per_sec,
            "entities": extraction["entities"], "triples": extraction["triples"],
            "repaired_values": extraction["repaired_values"],
            "source_text": task.prompt,
        }
        # Sink first: if it raises, nothing changed -> 500 -> the worker retries (sink dedupes by task_id).
        self.sink.write(record)
        task.result = {**extraction, "_lease_id": req.lease_id}
        task.completed_by = req.worker_id
        task.completed_at = now
        if w:
            w.completed_count += 1
            w.tokens_per_sec = req.tokens_per_sec
            w.last_seen = now
            self._dirty_workers.add(w.worker_id)
        self._unlease(task)
        self.log("COMPLETED", f"Chunk #{task.chunk_index} by {req.worker_id} ({req.tokens_per_sec:.1f} tok/s, "
                 f"{len(extraction['triples'])} triples)", task, req.worker_id, tokens_per_sec=req.tokens_per_sec)
        self._set_state(task, TaskState.COMPLETED)
        return {"status": "accepted"}

    @mutation
    def fail(self, req: TaskFailRequest) -> None:
        self._fail(self._get_lease(req.worker_id, req.task_id, req.lease_id), req.worker_id, req.error)

    def _fail(self, task: GridTask, worker_id: str, error: str) -> None:
        w = self.workers.get(worker_id)
        if w:
            w.failed_count += 1
            self._dirty_workers.add(worker_id)
        self.log("TASK_FAILED", f"Chunk #{task.chunk_index} failed on {worker_id}: {error[:120]}", task, worker_id)
        self._record_failure(task, error)

    # ---- maintenance
    @mutation
    def reap(self) -> None:
        now = self.clock()
        for task_id in list(self.leased):
            task = self.tasks[task_id]
            if now > task.lease_expires_at:
                old = task.assigned_worker
                self.log("LEASE_EXPIRED", f"Lease on chunk #{task.chunk_index} expired on {old}; reclaiming", task, old)
                self._record_failure(task, f"lease expired on {old}")
        for w in self.workers.values():
            if w.state != WorkerState.OFFLINE and now - w.last_seen > self.worker_timeout:
                w.state = WorkerState.OFFLINE
                w.current_task_id = None
                w.tokens_per_sec = 0.0
                self._dirty_workers.add(w.worker_id)
                self.log("NODE_OFFLINE", f"{w.worker_id} missed heartbeats", worker_id=w.worker_id)

    # ---- client API
    @mutation
    def submit(self, req: BatchSubmitRequest) -> Job:
        too_big = [i for i, c in enumerate(req.chunks) if len(c) > MAX_CHUNK_CHARS]
        if too_big:
            raise ValueError(f"chunks {too_big[:5]} exceed {MAX_CHUNK_CHARS} chars; split them before submitting")
        now = self.clock()
        job = Job(job_id=uuid.uuid4().hex[:12], job_name=req.job_name, task_ids=[], required_tier=req.required_tier,
                  submitted_at=now)
        self.jobs[job.job_id] = job
        self.job_counts[job.job_id] = Counter()
        for idx, chunk in enumerate(req.chunks):
            task = GridTask(task_id=uuid.uuid4().hex, job_id=job.job_id, chunk_index=idx, prompt=chunk,
                            required_tier=req.required_tier, created_at=now)
            self.tasks[task.task_id] = task
            self.counts[TaskState.PENDING.value] += 1
            self.job_counts[job.job_id][TaskState.PENDING.value] += 1
            self._enqueue(task, front=False)
            job.task_ids.append(task.task_id)
        self._dirty_jobs.add(job.job_id)
        capable = [w.worker_id for w in self.workers.values() if w.tier >= req.required_tier and w.state != WorkerState.OFFLINE]
        self.log("JOB_SUBMITTED", f"Job '{req.job_name}' ({len(job.task_ids)} chunks, Tier>={req.required_tier}, "
                 f"{len(capable)} capable nodes online)", job_id=job.job_id)
        return job

    def _job_summary(self, job: Job) -> dict:
        jc = self.job_counts[job.job_id]
        return {
            "job_id": job.job_id, "job_name": job.job_name, "total": len(job.task_ids),
            "counts": {s.value: jc[s.value] for s in TaskState},
            "done": job.finished_at is not None, "submitted_at": job.submitted_at, "finished_at": job.finished_at,
            "elapsed_s": (job.finished_at or self.clock()) - job.submitted_at,
        }

    def job_status(self, job_id: str, include_results: bool = False) -> dict:
        job = self.jobs[job_id]
        tasks = [self.tasks[t] for t in job.task_ids]
        out = {
            **self._job_summary(job),
            "total_attempts": sum(t.attempts for t in tasks), "total_preemptions": sum(t.preemptions for t in tasks),
            "total_failures": sum(t.failures for t in tasks),
            "failed_chunks": [{"chunk_index": t.chunk_index, "error": t.last_error} for t in tasks if t.state == TaskState.FAILED],
        }
        if include_results:
            out["results"] = [{"chunk_index": t.chunk_index, "worker_id": t.completed_by,
                               **{k: v for k, v in (t.result or {}).items() if not k.startswith("_")}} for t in tasks]
        return out

    def status(self) -> dict:
        out = {
            "workers": [w.model_dump() for w in self.workers.values()],
            "task_counts": {s.value: self.counts[s.value] for s in TaskState},
            "total_tasks": len(self.tasks),
            "jobs": [self._job_summary(self.jobs[j]) for j in list(self.jobs)[-5:]],
            "events": [f"[{time.strftime('%H:%M:%S', time.localtime(e.ts))}] {e.kind}: {e.msg}" for e in list(self.events)[-14:]],
        }
        stats = getattr(self.sink, "stats", None)
        if callable(stats):  # e.g. Qdrant queue/spool depth: lets operators see a lagging index
            out["sink"] = stats()
        return out

    def check_invariants(self) -> None:
        """Expensive consistency check of every index against the task table (tests / debugging)."""
        assert Counter(t.state.value for t in self.tasks.values()) == +self.counts
        assert self.leased == {t.task_id for t in self.tasks.values() if t.state == TaskState.LEASED}
        queued = [tid for q in self.pending.values() for tid in q if self.tasks[tid].state == TaskState.PENDING]
        assert sorted(queued) == sorted(t.task_id for t in self.tasks.values() if t.state == TaskState.PENDING), \
            "every PENDING task must be queued exactly once"
        by_worker = {(w, tid) for w, s in self.leases_by_worker.items() for tid in s}
        assert by_worker == {(t.assigned_worker, t.task_id) for t in self.tasks.values() if t.state == TaskState.LEASED}


# ---------------------------------------------------------------------------- HTTP layer
def _bearer_guard(expected: Optional[str]):
    async def guard(authorization: Optional[str] = Header(default=None)) -> None:
        if expected is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip().encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                                headers={"WWW-Authenticate": "Bearer"})
    return guard


def create_app(state: GridState | None = None, worker_token: str | None = None,
               client_token: str | None = None, discovery: DiscoveryResponder | None = None) -> FastAPI:
    grid = state or GridState(sink=_sink_from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def reaper():
            while True:
                try:
                    grid.reap()
                except Exception as e:  # never let the reaper die: it is what guarantees no lost chunks
                    print(f"[coordinator] reaper error: {e!r}", file=sys.stderr)
                await asyncio.sleep(REAPER_INTERVAL_SECONDS)
        task = asyncio.create_task(reaper())
        grid.log("COORDINATOR_UP", "Grid Coordinator started")
        if discovery is not None:
            try:
                await discovery.start()
                grid.log("DISCOVERY_UP", f"Advertising {discovery.advertise_url} on UDP {discovery.port}",
                         port=discovery.port, authenticated=discovery.worker_token is not None)
            except OSError as e:  # discovery is a convenience: the HTTP API must still come up
                print(f"[coordinator] WARNING: discovery responder failed on UDP {discovery.port}: {e}; "
                      f"workers need --coordinator URL", file=sys.stderr)
        try:
            yield
        finally:
            task.cancel()
            if discovery is not None:
                discovery.close()
            # Off the loop: the Qdrant sink's close() joins its flush thread (bounded deadline).
            await asyncio.to_thread(grid.sink.close)
            if grid.store:
                grid.store.close()

    app = FastAPI(title="Nightshift Grid Coordinator", lifespan=lifespan)
    app.state.grid = grid
    worker_api = APIRouter(prefix="/worker", dependencies=[Depends(_bearer_guard(worker_token))])
    client_api = APIRouter(prefix="/v1", dependencies=[Depends(_bearer_guard(client_token))])

    def _unknown(worker_id: str):
        # 404 tells the worker to re-register.
        return HTTPException(status_code=404, detail=f"worker {worker_id} not registered")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}  # deliberately unauthenticated and information-free

    @worker_api.post("/register")
    async def register_worker(info: WorkerInfo):
        grid.register(info)
        return {"status": "registered", "lease_ttl": grid.lease_ttl, "heartbeat_interval": HEARTBEAT_INTERVAL_SECONDS}

    @worker_api.post("/pull-task")
    async def pull_task(worker_id: str):
        try:
            task = grid.pull(worker_id)
        except KeyError:
            raise _unknown(worker_id)
        if not task:
            return {"task": None}
        return {"task": {"task_id": task.task_id, "lease_id": task.lease_id, "job_id": task.job_id,
                         "chunk_index": task.chunk_index, "prompt": task.prompt, "lease_ttl": grid.lease_ttl}}

    @worker_api.post("/heartbeat")
    async def worker_heartbeat(hb: HeartbeatRequest):
        try:
            return grid.heartbeat(hb)
        except KeyError:
            raise _unknown(hb.worker_id)

    @worker_api.post("/preempt")
    async def preempt_worker(req: PreemptRequest):
        try:
            return {"reclaimed": grid.preempt(req)}
        except KeyError:
            raise _unknown(req.worker_id)

    @worker_api.post("/complete")
    async def complete_task(req: TaskCompleteRequest):
        try:
            return grid.complete(req)
        except LeaseError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @worker_api.post("/fail")
    async def fail_task(req: TaskFailRequest):
        try:
            grid.fail(req)
            return {"status": "requeued"}
        except LeaseError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @client_api.post("/grid/mapreduce")
    async def submit_mapreduce_job(req: BatchSubmitRequest):
        try:
            job = grid.submit(req)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"job_id": job.job_id, "task_count": len(job.task_ids)}

    @client_api.get("/grid/jobs/{job_id}")
    async def get_job(job_id: str, results: bool = False):
        if job_id not in grid.jobs:
            raise HTTPException(status_code=404, detail="unknown job")
        return grid.job_status(job_id, include_results=results)

    @client_api.get("/grid/status")
    async def grid_status():
        return grid.status()

    @client_api.get("/grid/events")
    async def grid_events(since_seq: int = 0, limit: int = 1000):
        return [e for e in grid.events if e.seq > since_seq][:limit]

    app.include_router(worker_api)
    app.include_router(client_api)
    return app


def _sink_from_env() -> GraphSink:
    path = os.environ.get("GRID_OUTPUT_JSONL", "output_graph.jsonl")
    return NullSink() if path.lower() in ("", "none", "off") else JsonlGraphSink(path)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


AUTO_TOKEN_FILES = {"worker": "worker.token", "client": "client.token"}


def _state_dir(db: str) -> Path:
    """Where auto-generated secrets and the Qdrant spool live: next to the SQLite db, so one
    directory holds everything that must survive a restart (cwd when running db-less)."""
    return Path(db).resolve().parent if db.lower() != "none" else Path.cwd()


def ensure_auto_tokens(token_dir: str | os.PathLike, worker_token: str | None, client_token: str | None, *,
                       stdout: TextIO | None = None, stderr: TextIO | None = None) -> tuple[str, str, str, str]:
    """Fill in missing role tokens from `<token_dir>/{worker,client}.token`, minting them on first use.

    Returns (worker_token, client_token, worker_src, client_src); the sources are labels only, safe
    to log. Reusing the file on later runs is the point: workers provisioned with the token keep
    working across coordinator restarts. Files are created with O_EXCL and mode 0600 in one
    syscall, so there's no window where a secret is world-readable and two coordinators racing
    on first start can't clobber each other's token. The token values are printed once, at
    creation, and only to a TTY, so they never end up in a log file or journald capture."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    d = Path(token_dir)
    d.mkdir(parents=True, exist_ok=True)
    tokens = {"worker": worker_token, "client": client_token}
    sources = {"worker": "configured", "client": "configured"}
    minted: dict[str, str] = {}
    for role, fname in AUTO_TOKEN_FILES.items():
        if tokens[role]:
            continue
        path = d / fname
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            tokens[role], _ = resolve_token(str(path), ())  # warns if the file is group/world-readable
            sources[role] = f"auto file {path} (reused)"
            continue
        tok = secrets.token_urlsafe(32)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(tok + "\n")
            f.flush()
            os.fsync(f.fileno())
        tokens[role], sources[role], minted[role] = tok, f"auto file {path} (new)", tok
    for role in AUTO_TOKEN_FILES:
        if sources[role].startswith("auto"):
            print(f"[coordinator] {role} token: {sources[role]}", file=stderr)
    if minted:
        hint = (f"copy {d / AUTO_TOKEN_FILES['worker']} to each worker host (chmod 600) and run "
                f"`worker_daemon.py --token-file <it>`")
        print(f"[coordinator] generated {', '.join(minted)} token(s); {hint}", file=stderr)
        if getattr(stdout, "isatty", lambda: False)():
            for role, tok in minted.items():
                print(f"  {role} token (shown once): {tok}", file=stdout)
    assert tokens["worker"] and tokens["client"]
    return tokens["worker"], tokens["client"], sources["worker"], sources["client"]


def _build_sink(a: argparse.Namespace) -> GraphSink:
    jsonl = NullSink() if a.output.lower() == "none" else JsonlGraphSink(a.output)
    location = ":memory:" if a.qdrant_memory else a.qdrant_url
    if not location:
        return jsonl
    try:
        from qdrant_sink import FanoutSink, QdrantConfigError, build_qdrant_sink
    except ImportError as e:
        raise SystemExit(f"--qdrant-* needs qdrant-client: pip install qdrant-client ({e})")
    if isinstance(jsonl, NullSink):
        print("[coordinator] WARNING: --output none with Qdrant: the vector index becomes the only copy of results",
              file=sys.stderr)
    spool = a.qdrant_spool or _state_dir(a.db) / f"{a.qdrant_collection}.spool.jsonl"
    try:
        q = build_qdrant_sink(location, a.qdrant_collection, embedder_mode=a.qdrant_embedder,
                              embed_model=a.qdrant_embed_model, ollama_url=a.ollama, hash_dim=a.qdrant_hash_dim,
                              spool_path=spool)
    except QdrantConfigError as e:
        raise SystemExit(f"Qdrant sink: {e}")
    print(f"[coordinator] qdrant sink: {location} collection={q.collection} embedder={q.embedder.name} "
          f"spool={q.spool.path}", file=sys.stderr)
    return FanoutSink(jsonl, q)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Nightshift Grid Coordinator")
    p.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 for LAN workers; requires tokens)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--output", default=os.environ.get("GRID_OUTPUT_JSONL", "output_graph.jsonl"),
                   help="JSONL file for completed GraphRAG triples ('none' to disable)")
    p.add_argument("--db", default=os.environ.get("GRID_DB", "grid_state.db"),
                   help="SQLite state file for crash recovery ('none' = in-memory only)")
    p.add_argument("--worker-token-file", help="file with the worker bearer token (else $GRID_WORKER_TOKEN / $GRID_TOKEN)")
    p.add_argument("--client-token-file", help="file with the client bearer token (else $GRID_CLIENT_TOKEN / $GRID_TOKEN)")
    p.add_argument("--auto-token", action="store_true",
                   help="generate any missing worker/client token into 0600 files next to --db and reuse them later")
    p.add_argument("--allow-insecure", action="store_true", help="allow a non-loopback bind without tokens (not recommended)")
    g = p.add_argument_group("LAN discovery")
    g.add_argument("--advertise", action=argparse.BooleanOptionalAction, default=None,
                   help="answer UDP discovery probes (default: on for a non-loopback --host, off on loopback)")
    g.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    g.add_argument("--advertise-url", default=None, help="URL workers should use (default http://<primary LAN IP>:<port>)")
    q = p.add_argument_group("live Qdrant sink (optional; JSONL stays the system of record)")
    qm = q.add_mutually_exclusive_group()
    qm.add_argument("--qdrant-url", default=None, help="e.g. http://localhost:6333 (or a local path)")
    qm.add_argument("--qdrant-memory", action="store_true", help="in-process Qdrant (ephemeral; demos/tests)")
    q.add_argument("--qdrant-collection", default="nightshift_graph", help="must start with nightshift_")
    q.add_argument("--qdrant-embedder", choices=("auto", "ollama", "hash"), default="auto",
                   help="auto = Ollama if reachable at startup, else hash; fixed for the collection's lifetime")
    q.add_argument("--qdrant-embed-model", default="nomic-embed-text")
    q.add_argument("--qdrant-hash-dim", type=int, default=768)
    q.add_argument("--qdrant-spool", default=None, help="retry spool file (default <db dir>/<collection>.spool.jsonl)")
    q.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama URL for embeddings")
    a = p.parse_args(argv)

    try:  # tokens come from files or env, never argv (argv is world-readable via `ps`)
        worker_token, worker_src = resolve_token(a.worker_token_file, WORKER_TOKEN_ENV)
        client_token, client_src = resolve_token(a.client_token_file, CLIENT_TOKEN_ENV)
        if a.auto_token:
            worker_token, client_token, ws, cs = ensure_auto_tokens(_state_dir(a.db), worker_token, client_token)
            worker_src = ws if ws != "configured" else worker_src
            client_src = cs if cs != "configured" else client_src
    except (GridAuthError, OSError) as e:
        raise SystemExit(str(e))
    if not _is_loopback(a.host) and not (worker_token and client_token) and not a.allow_insecure:
        raise SystemExit(f"Refusing to bind {a.host} without worker AND client tokens (anyone on the network could "
                         f"submit jobs or forge results). Pass --auto-token to generate them (saved next to the db), "
                         f"set GRID_TOKEN, or pass --allow-insecure.")
    if not (worker_token or client_token):
        print("[coordinator] WARNING: auth disabled (no tokens configured)", file=sys.stderr)
    elif worker_token == client_token:
        print("[coordinator] note: worker and client share one token; set GRID_WORKER_TOKEN and GRID_CLIENT_TOKEN "
              "separately so a compromised worker cannot submit jobs or read results", file=sys.stderr)
    print(f"[coordinator] worker token: {worker_src} | client token: {client_src} | db: {a.db} | output: {a.output}",
          file=sys.stderr)

    advertise = (not _is_loopback(a.host)) if a.advertise is None else a.advertise
    responder = None
    if advertise:
        url = a.advertise_url or default_advertise_url(a.host, a.port)
        # On a loopback bind, also keep the UDP responder on loopback: never advertise to the LAN
        # a URL that only works on this machine.
        bind = "127.0.0.1" if _is_loopback(a.host) else ""
        try:
            responder = DiscoveryResponder(url, worker_token, bind_host=bind, port=a.discovery_port)
        except ValueError as e:
            raise SystemExit(str(e))
        if not worker_token:
            print("[coordinator] WARNING: advertising without a worker token: discovery replies are unauthenticated",
                  file=sys.stderr)
        print(f"[coordinator] discovery: advertising {url} on UDP {a.discovery_port}", file=sys.stderr)

    sink = _build_sink(a)
    store = None if a.db.lower() == "none" else GridStore(a.db)
    grid = GridState(sink=sink, store=store)
    uvicorn.run(create_app(grid, worker_token=worker_token, client_token=client_token, discovery=responder),
                host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
