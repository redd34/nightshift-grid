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
import hashlib
import hmac
import ipaddress
import os
import secrets
import stat
import sys
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, TextIO

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from discovery import DISCOVERY_PORT, DiscoveryResponder, default_advertise_url
from graphrag import ExtractionError, GraphSink, JsonlGraphSink, NullSink, normalize_extraction
from relay import MAX_CKPT_CHARS
from grid_auth import (CLIENT_TOKEN_ENV, WORKER_TOKEN_ENV, GridAuthError, credential_worker_id, resolve_token,
                       verify_worker_credential)
from grid_tls import TLSConfigError, cert_sans, default_sans, ensure_self_signed, fingerprint_file
from grid_store import GridStore

LEASE_TTL_SECONDS = 12.0
HEARTBEAT_INTERVAL_SECONDS = 3.0
WORKER_TIMEOUT_SECONDS = 12.0
REAPER_INTERVAL_SECONDS = 0.5
MAX_FAILURES = 3
MAX_HOPS = 16            # relay-ledger entries kept per task
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
    # sha256 of the per-registration session secret; server-managed, never returned by status APIs.
    session_digest: Optional[str] = None
    # Decode accounting (Baton): tokens this node generated, and how many of them were thrown away
    # (lease ended without a result and without a checkpoint a successor could continue from).
    tokens_generated: int = 0
    tokens_wasted: int = 0


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
    # Baton token relay. The checkpoint is the generation-so-far; it is journaled with the task, so it
    # survives a coordinator crash as well as a worker one.
    ckpt_text: str = ""
    ckpt_tokens: int = 0           # decode tokens that text represents
    ckpt_model: str = ""           # model that produced it: only the same model may continue it
    lease_start_tokens: int = 0    # tokens the current holder was handed (0 = from scratch)
    lease_reported_tokens: int = 0  # holder's latest cumulative token count
    tokens_generated: int = 0      # decode tokens burned on this task across all attempts
    tokens_salvaged: int = 0       # tokens successors did not have to decode again
    output_tokens: int = 0         # tokens in the accepted result
    resumes: int = 0
    hops: list[dict] = Field(default_factory=list)  # relay ledger (last MAX_HOPS leases)


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


class CheckpointFields(BaseModel):
    """Baton: the holder's progress. `gen_tokens` is cumulative for this generation (a resumed
    lease starts at the tokens it was handed). The text travels as a delta: `ckpt_delta` replaces
    the stored checkpoint from character `ckpt_from` on, so each heartbeat only carries new text."""
    gen_tokens: int = Field(0, ge=0)
    ckpt_from: Optional[int] = Field(None, ge=0)
    ckpt_delta: Optional[str] = Field(None, max_length=MAX_CKPT_CHARS)


class HeartbeatRequest(CheckpointFields):
    worker_id: str
    task_id: Optional[str] = None
    lease_id: Optional[str] = None
    tokens_per_sec: float = 0.0
    cpu_percent: float = 0.0
    ram_free_gb: float = 0.0
    state: WorkerState = WorkerState.IDLE_HARVESTING


class PreemptRequest(CheckpointFields):
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
    gen_tokens: int = Field(0, ge=0)


class TaskFailRequest(BaseModel):
    worker_id: str
    task_id: str
    lease_id: str
    error: str


SESSION_HEADER = "X-Nightshift-Session"


def _digest(session: str) -> str:
    return hashlib.sha256(session.encode()).hexdigest()


class WorkerIdInUse(Exception):
    """A live worker already holds this worker_id and the caller didn't prove it is that worker."""


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
        relay: bool = False,
    ):
        self.sink = sink or NullSink()
        self.store = store
        self.clock = clock
        self.lease_ttl = lease_ttl
        self.worker_timeout = worker_timeout
        self.max_failures = max_failures
        # Baton: hand a preempted generation's partial output to the next same-model worker.
        # Decode accounting (generated / wasted / goodput) runs either way.
        self.relay = relay
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
        w = self.workers.get(worker_id)
        usable = (self.relay and task.ckpt_tokens > 0 and bool(task.ckpt_text) and w is not None
                  and w.model_name == task.ckpt_model)
        task.lease_start_tokens = task.ckpt_tokens if usable else 0
        task.lease_reported_tokens = task.lease_start_tokens
        if usable:
            task.resumes += 1
            task.tokens_salvaged += task.ckpt_tokens
            prev = list(dict.fromkeys(h["worker"] for h in task.hops)) or ["?"]
            self.log("RELAY_RESUME", f"Chunk #{task.chunk_index} -> {worker_id} resumes at token {task.ckpt_tokens} "
                     f"(decoded so far by {', '.join(prev)})", task, worker_id, resume_tokens=task.ckpt_tokens)

    def _unlease(self, task: GridTask) -> None:
        # Every way a lease ends (complete / preempt / expiry / fail / re-register) funnels through here.
        if task.assigned_worker:
            new = max(0, task.lease_reported_tokens - task.lease_start_tokens)
            done = task.completed_at is not None and task.completed_by == task.assigned_worker
            # Tokens a successor can continue from are not wasted (if the checkpoint is later dropped,
            # they are counted then - see _drop_checkpoint).
            saved = new if done else max(0, min(task.lease_reported_tokens,
                                                 task.ckpt_tokens if self.relay else 0) - task.lease_start_tokens)
            task.tokens_generated += new
            task.hops = (task.hops + [{"worker": task.assigned_worker, "start": task.lease_start_tokens,
                                       "end": task.lease_reported_tokens, "completed": done}])[-MAX_HOPS:]
            w = self.workers.get(task.assigned_worker)
            if w:
                w.tokens_generated += new
                w.tokens_wasted += new - saved
                self._dirty_workers.add(w.worker_id)
            task.lease_start_tokens = task.lease_reported_tokens = 0
        self.leased.discard(task.task_id)
        holder = self.leases_by_worker.get(task.assigned_worker or "")
        if holder:
            holder.discard(task.task_id)
        self._release_worker(task.assigned_worker, task.task_id)
        task.assigned_worker = None
        task.lease_id = None
        task.lease_expires_at = 0.0
        self._dirty_tasks.add(task.task_id)

    def _absorb(self, task: GridTask, cp: CheckpointFields) -> bool:
        """Take the current holder's progress report. Returns whether the checkpoint delta was
        stored. Callers have already checked the lease, so a zombie can never write here."""
        task.lease_reported_tokens = max(task.lease_reported_tokens, cp.gen_tokens)
        if not self.relay or cp.ckpt_delta is None or cp.ckpt_from is None:
            return False
        if cp.ckpt_from > len(task.ckpt_text):
            return False  # a gap: the worker resends from its last acknowledged offset
        text = task.ckpt_text[:cp.ckpt_from] + cp.ckpt_delta
        if text == task.ckpt_text:
            return False  # nothing new (already acknowledged)
        if len(text) > MAX_CKPT_CHARS or len(text) < len(task.ckpt_text):
            return False  # only progress replaces a checkpoint (a rolled-back resume catches up first)
        w = self.workers.get(task.assigned_worker or "")
        task.ckpt_text, task.ckpt_tokens = text, max(cp.gen_tokens, task.lease_start_tokens)
        task.ckpt_model = w.model_name if w else ""
        self._dirty_tasks.add(task.task_id)
        return True

    def _drop_checkpoint(self, task: GridTask) -> None:
        """An attempt failed validation and may have been steered wrong by its prefix: never reuse it.
        Tokens that were only 'saved' by this checkpoint become waste of the workers who decoded them."""
        if task.ckpt_tokens:
            for h in task.hops:
                if not h.get("completed") and h["worker"] in self.workers:
                    kept = max(0, min(h["end"], task.ckpt_tokens) - h["start"])
                    self.workers[h["worker"]].tokens_wasted += kept
                    self._dirty_workers.add(h["worker"])
        task.ckpt_text, task.ckpt_tokens, task.ckpt_model = "", 0, ""
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
    def verify_session(self, worker_id: str, session: Optional[str]) -> WorkerInfo:
        """Bind every worker call to the registration that created it. The shared worker token only
        proves "some worker"; without this, any token holder could heartbeat, preempt or steal leases
        as another live worker_id. Raises KeyError (-> 404, worker re-registers) on unknown/mismatch."""
        w = self._worker(worker_id)
        if not session or not w.session_digest or not hmac.compare_digest(_digest(session), w.session_digest):
            raise KeyError(worker_id)
        return w

    @mutation
    def register(self, info: WorkerInfo, session: Optional[str] = None) -> str:
        """Register (or re-register) a worker and return its new session secret.

        Re-registering a worker_id that holds a session requires that session, so another token
        holder can't hijack it. The id is released only by (a) a graceful shutdown, which revokes
        the session, or (b) `worker_timeout` without any contact, measured from last_seen. It is
        deliberately NOT released by the OFFLINE *state*: after coordinator recovery every worker
        is OFFLINE until it heartbeats, and recovery resets last_seen to now precisely so that
        window can't be used to steal a live worker's id."""
        prev = self.workers.get(info.worker_id)
        if prev and prev.session_digest and self.clock() - prev.last_seen <= self.worker_timeout and not (
                session and hmac.compare_digest(_digest(session), prev.session_digest)):
            raise WorkerIdInUse(info.worker_id)
        new_session = secrets.token_urlsafe(24)
        info.session_digest = _digest(new_session)
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
        return new_session

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
                if self._absorb(task, hb):
                    self._flush()  # journal the checkpoint: it must survive a coordinator crash too
        # Tell the worker if its lease is gone so it can stop burning compute on zombie work.
        return {"ok": True, "lease_valid": lease_valid,
                "ckpt_len": len(task.ckpt_text) if lease_valid and self.relay else None}

    @mutation
    def preempt(self, req: PreemptRequest) -> int:
        w = self._worker(req.worker_id)
        now = self.clock()
        # A graceful shutdown revokes the session, freeing the worker_id for the restarted process.
        w.state = WorkerState.OFFLINE if req.reason == "shutdown" else WorkerState.PREEMPTED_USER_ACTIVE
        if req.reason == "shutdown":
            w.session_digest = None
        w.current_task_id = None
        w.tokens_per_sec = 0.0
        w.preempted_count += 1
        w.last_seen = now
        self._dirty_workers.add(w.worker_id)
        reclaimed = 0
        for task in self._worker_leases(req.worker_id):
            if req.lease_id is not None and task.lease_id != req.lease_id:
                continue
            self._absorb(task, req)  # the exact text at the moment of yield (rides the POST we send anyway)
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
        task.lease_reported_tokens = max(task.lease_reported_tokens, req.gen_tokens)
        task.output_tokens = task.lease_reported_tokens
        task.ckpt_text = ""  # the result is the record now
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
        self._drop_checkpoint(task)
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
        out["relay"] = {"resumes": sum(t.resumes for t in tasks), "tokens_generated": sum(t.tokens_generated for t in tasks),
                        "tokens_in_results": sum(t.output_tokens for t in tasks if t.state == TaskState.COMPLETED)}
        if include_results:
            out["results"] = [{"chunk_index": t.chunk_index, "worker_id": t.completed_by,
                               **{k: v for k, v in (t.result or {}).items() if not k.startswith("_")}} for t in tasks]
        return out

    def status(self) -> dict:
        out = {
            "workers": [w.model_dump(exclude={"session_digest"}) for w in self.workers.values()],
            "task_counts": {s.value: self.counts[s.value] for s in TaskState},
            "total_tasks": len(self.tasks),
            "jobs": [self._job_summary(self.jobs[j]) for j in list(self.jobs)[-5:]],
            "events": [f"[{time.strftime('%H:%M:%S', time.localtime(e.ts))}] {e.kind}: {e.msg}" for e in list(self.events)[-14:]],
        }
        out["relay"] = self.relay_stats()
        stats = getattr(self.sink, "stats", None)
        if callable(stats):  # e.g. Qdrant queue/spool depth: lets operators see a lagging index
            out["sink"] = stats()
        return out

    def relay_stats(self) -> dict:
        """Grid-wide decode accounting. goodput = tokens in accepted results / tokens decoded."""
        generated = output = salvaged = resumes = inflight = relayed = 0
        for t in self.tasks.values():
            generated += t.tokens_generated
            salvaged += t.tokens_salvaged
            resumes += t.resumes
            if t.state == TaskState.COMPLETED:
                output += t.output_tokens
                relayed += t.resumes > 0
            elif t.state == TaskState.LEASED:
                inflight += max(0, t.lease_reported_tokens - t.lease_start_tokens)
        return {
            "enabled": self.relay, "tokens_generated": generated, "tokens_in_results": output,
            "tokens_salvaged": salvaged, "tokens_inflight": inflight, "resumes": resumes, "relayed_chunks": relayed,
            "goodput_pct": round(100.0 * output / generated, 1) if generated else None,
            "wasted_by_worker": {w.worker_id: w.tokens_wasted for w in self.workers.values() if w.tokens_wasted},
        }

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


def _worker_guard(master: Optional[str], allow_shared: bool):
    """Authenticate /worker/* and pin the caller to an identity (`request.state.worker_identity`).

    A per-worker credential (see grid_auth.derive_worker_credential) authorizes exactly one worker
    id. The master worker token authorizes any id, and is accepted only when `allow_shared` (the
    loopback default, where every process runs on one machine anyway)."""
    async def guard(request: Request, authorization: Optional[str] = Header(default=None)) -> None:
        request.state.worker_identity = None
        if master is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        token = token.strip()
        if scheme.lower() == "bearer":
            wid = verify_worker_credential(master, token)
            if wid is not None:
                request.state.worker_identity = wid
                return
            if hmac.compare_digest(token.encode(), master.encode()):
                if allow_shared:
                    return
                raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Bearer"},
                                    detail="this coordinator requires per-worker credentials; the shared worker token "
                                           "is not accepted from workers (issue one with `python grid_auth.py issue "
                                           "--id <worker-id> --out <file>` on the coordinator host)")
        raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    return guard


def _check_identity(request: Request, worker_id: str) -> None:
    identity = getattr(request.state, "worker_identity", None)
    if identity is not None and identity != worker_id:
        raise HTTPException(status_code=403, detail=f"credential is for worker {identity!r}, not {worker_id!r}")


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _hostname(value: str) -> str:
    """Host part of a Host header or Origin URL: 'Example.com:8000' -> 'example.com', '[::1]:80' -> '::1'."""
    v = value.strip().lower()
    if "://" in v:
        v = v.split("://", 1)[1]
    v = v.split("/", 1)[0]
    if v.startswith("["):
        return v[1:v.index("]")] if "]" in v else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def create_app(state: GridState | None = None, worker_token: str | None = None,
               client_token: str | None = None, discovery: DiscoveryResponder | None = None,
               trusted_hosts: frozenset[str] | set[str] | None = None,
               allow_shared_worker_token: bool = True) -> FastAPI:
    """`trusted_hosts`: hostnames accepted in Host/Origin headers (loopback names are always allowed).
    `allow_shared_worker_token=False` makes workers authenticate with per-worker credentials only."""
    grid = state or GridState(sink=_sink_from_env())
    allowed_hosts = LOOPBACK_HOSTS | {h.lower() for h in (trusted_hosts or ())}
    auth_enabled = bool(worker_token or client_token)

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

    @app.middleware("http")
    async def host_and_origin_guard(request: Request, call_next):
        """Browser-borne attacks on the API. Workers, the CLI and MCP never send an Origin header,
        so any cross-origin browser request (CSRF via a malicious page) is refused outright. When
        auth is disabled (loopback dev mode), DNS rebinding could still let a web page reach
        127.0.0.1 under its own hostname - so the Host header must then be a trusted name."""
        origin = request.headers.get("origin")
        if origin is not None and _hostname(origin) not in allowed_hosts:
            return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
        if not auth_enabled and _hostname(request.headers.get("host", "")) not in allowed_hosts:
            return JSONResponse({"detail": "untrusted Host header (DNS rebinding protection)"}, status_code=403)
        return await call_next(request)
    worker_api = APIRouter(prefix="/worker", dependencies=[Depends(_worker_guard(worker_token, allow_shared_worker_token))])
    client_api = APIRouter(prefix="/v1", dependencies=[Depends(_bearer_guard(client_token))])

    def _unknown(worker_id: str):
        # 404 tells the worker to re-register (unknown id, or a session that isn't the current one).
        return HTTPException(status_code=404, detail=f"worker {worker_id} not registered (or stale session)")

    def _session(request: Request, worker_id: str, session: Optional[str]) -> None:
        _check_identity(request, worker_id)
        try:
            grid.verify_session(worker_id, session)
        except KeyError:
            raise _unknown(worker_id)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}  # deliberately unauthenticated and information-free

    @worker_api.post("/register")
    async def register_worker(request: Request, info: WorkerInfo,
                              x_nightshift_session: Optional[str] = Header(default=None)):
        _check_identity(request, info.worker_id)
        try:
            session = grid.register(info, session=x_nightshift_session)
        except WorkerIdInUse:
            raise HTTPException(status_code=409, detail=f"worker_id {info.worker_id} is held by a live worker; "
                                f"it frees up when that worker shuts down or misses heartbeats")
        return {"status": "registered", "session": session, "lease_ttl": grid.lease_ttl,
                "heartbeat_interval": HEARTBEAT_INTERVAL_SECONDS, "relay": grid.relay}

    @worker_api.post("/pull-task")
    async def pull_task(request: Request, worker_id: str, x_nightshift_session: Optional[str] = Header(default=None)):
        _session(request, worker_id, x_nightshift_session)
        try:
            task = grid.pull(worker_id)
        except KeyError:
            raise _unknown(worker_id)
        if not task:
            return {"task": None}
        out = {"task_id": task.task_id, "lease_id": task.lease_id, "job_id": task.job_id,
               "chunk_index": task.chunk_index, "prompt": task.prompt, "lease_ttl": grid.lease_ttl}
        if task.lease_start_tokens:  # Baton: continue this generation instead of starting over
            out["resume"] = {"text": task.ckpt_text, "tokens": task.lease_start_tokens}
        return {"task": out}

    @worker_api.post("/heartbeat")
    async def worker_heartbeat(request: Request, hb: HeartbeatRequest, x_nightshift_session: Optional[str] = Header(default=None)):
        _session(request, hb.worker_id, x_nightshift_session)
        try:
            return grid.heartbeat(hb)
        except KeyError:
            raise _unknown(hb.worker_id)

    @worker_api.post("/preempt")
    async def preempt_worker(request: Request, req: PreemptRequest, x_nightshift_session: Optional[str] = Header(default=None)):
        _session(request, req.worker_id, x_nightshift_session)
        try:
            return {"reclaimed": grid.preempt(req)}
        except KeyError:
            raise _unknown(req.worker_id)

    @worker_api.post("/complete")
    async def complete_task(request: Request, req: TaskCompleteRequest, x_nightshift_session: Optional[str] = Header(default=None)):
        _session(request, req.worker_id, x_nightshift_session)
        try:
            return grid.complete(req)
        except LeaseError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @worker_api.post("/fail")
    async def fail_task(request: Request, req: TaskFailRequest, x_nightshift_session: Optional[str] = Header(default=None)):
        _session(request, req.worker_id, x_nightshift_session)
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
    if not d.exists():
        d.mkdir(parents=True, mode=0o700)
    _check_private_dir(d)
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
            tokens[role] = _read_trusted_secret(path)
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
        hint = (f"keep {d / AUTO_TOKEN_FILES['worker']} on this host (it is the master worker secret) and give each "
                f"worker its own credential: `python grid_auth.py issue --id <worker-id> --worker-token-file "
                f"{d / AUTO_TOKEN_FILES['worker']} --out <worker-id>.credential`")
        print(f"[coordinator] generated {', '.join(minted)} token(s); {hint}", file=stderr)
        if getattr(stdout, "isatty", lambda: False)():
            for role, tok in minted.items():
                print(f"  {role} token (shown once): {tok}", file=stdout)
    assert tokens["worker"] and tokens["client"]
    return tokens["worker"], tokens["client"], sources["worker"], sources["client"]


def _check_private_dir(d: Path) -> None:
    """The token dir must not be writable by anyone else: otherwise another local user could plant
    a token file (or a symlink) before our first run and know - or redirect - the grid's secret."""
    if os.name != "posix":
        return
    st = d.stat()
    if st.st_uid != os.getuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GridAuthError(f"token directory {d} must be owned by you and not group/world-writable "
                            f"(chmod 700 {d})")


def _read_trusted_secret(path: Path) -> str:
    """Read a reused auto-token, refusing anything we didn't plausibly create ourselves: symlinks
    (O_NOFOLLOW), non-regular files, files owned by another user, or files with any group/other
    permission bits. A mismatch is a hard error, never a warning - silently trusting a planted
    token would hand the grid to whoever planted it."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as e:
        raise GridAuthError(f"cannot open token file {path} safely ({e.strerror}); is it a symlink?") from e
    with os.fdopen(fd, encoding="utf-8") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise GridAuthError(f"token file {path} is not a regular file")
        if os.name == "posix" and (st.st_uid != os.getuid() or st.st_mode & 0o077):
            raise GridAuthError(f"token file {path} must be owned by you with mode 0600 "
                                f"(found uid {st.st_uid}, mode {stat.S_IMODE(st.st_mode):o}); "
                                f"fix it or delete it to mint a new token")
        token = f.read().strip()
    if not token:
        raise GridAuthError(f"token file {path} is empty")
    return token


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
    p.add_argument("--allow-insecure", action="store_true",
                   help="allow a non-loopback bind without tokens or TLS (not recommended)")
    p.add_argument("--worker-auth", choices=("auto", "per-worker", "shared"), default="auto",
                   help="per-worker = workers must present a credential bound to their id (grid_auth.py issue); "
                        "shared = the master worker token is also accepted. auto = per-worker on a non-loopback "
                        "bind, shared on loopback")
    p.add_argument("--trusted-host", action="append", default=[], metavar="NAME",
                   help="extra hostname accepted in Host/Origin headers (repeatable; loopback names always allowed)")
    p.add_argument("--relay", action=argparse.BooleanOptionalAction, default=False,
                   help="Baton token relay: a preempted chunk continues from its partial output on the next "
                        "same-model worker instead of restarting (decode accounting is always on)")
    t = p.add_argument_group("TLS (required for a non-loopback bind unless --allow-plaintext)")
    t.add_argument("--tls-cert", default=None, help="PEM certificate (chain) file")
    t.add_argument("--tls-key", default=None, help="PEM private key file (mode 0600)")
    t.add_argument("--auto-tls", action="store_true",
                   help="use (minting on first run) a self-signed certificate next to --db; workers pin its "
                        "fingerprint, which discovery delivers authenticated")
    t.add_argument("--tls-san", action="append", default=[], metavar="NAME",
                   help="extra DNS name / IP for a newly minted --auto-tls certificate (repeatable)")
    t.add_argument("--allow-plaintext", action="store_true",
                   help="serve plain HTTP on a non-loopback bind (only if the network is already encrypted, "
                        "e.g. Tailscale/WireGuard)")
    g = p.add_argument_group("LAN discovery")
    g.add_argument("--advertise", action=argparse.BooleanOptionalAction, default=None,
                   help="answer UDP discovery probes (default: on for a non-loopback --host, off on loopback)")
    g.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    g.add_argument("--advertise-url", default=None,
                   help="URL workers should use (default http(s)://<primary LAN IP>:<port>)")
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
    if credential_worker_id(worker_token):
        raise SystemExit("the configured worker token is a per-worker credential; the coordinator needs the master "
                         "worker token it was derived from")

    # ---- TLS
    if bool(a.tls_cert) != bool(a.tls_key):
        raise SystemExit("--tls-cert and --tls-key must be given together")
    if a.auto_tls and a.tls_cert:
        raise SystemExit("--auto-tls and --tls-cert/--tls-key are mutually exclusive")
    cert = key = tls_fp = None
    try:
        if a.auto_tls:
            extra = list(a.tls_san) + ([_hostname(a.advertise_url)] if a.advertise_url else []) + \
                ([a.host] if a.host not in ("0.0.0.0", "::") else [])
            cert, key, tls_fp, created = ensure_self_signed(_state_dir(a.db), default_sans(extra))
            print(f"[coordinator] TLS: {'minted' if created else 'reusing'} self-signed certificate {cert}",
                  file=sys.stderr)
            missing = [n for n in extra if n and n not in cert_sans(cert)]
            if missing and not created:
                print(f"[coordinator] note: certificate has no SAN for {', '.join(missing)}; fingerprint-pinning "
                      f"clients are unaffected, CA-file clients need a new cert (delete {cert} and {key})",
                      file=sys.stderr)
        elif a.tls_cert:
            cert, key = Path(a.tls_cert), Path(a.tls_key)
            tls_fp = fingerprint_file(cert)
            if os.name == "posix" and os.stat(key).st_mode & 0o077:
                print(f"[coordinator] WARNING: TLS key {key} is readable by other users; chmod 600 it", file=sys.stderr)
    except (TLSConfigError, OSError, ValueError) as e:
        raise SystemExit(f"TLS setup failed: {e}")
    if not _is_loopback(a.host) and cert is None and not (a.allow_plaintext or a.allow_insecure):
        raise SystemExit(f"Refusing to serve plain HTTP on {a.host}: bearer tokens and results would cross the network "
                         f"unencrypted. Pass --auto-tls (self-signed; workers pin the fingerprint, which --discover "
                         f"delivers automatically), --tls-cert/--tls-key, or --allow-plaintext if the network is "
                         f"already encrypted (Tailscale/WireGuard).")
    if tls_fp:
        print(f"[coordinator] TLS fingerprint: {tls_fp}  (workers: --tls-fingerprint {tls_fp})", file=sys.stderr)
    scheme = "https" if cert else "http"

    # ---- worker identity
    per_worker = {"per-worker": True, "shared": False, "auto": not _is_loopback(a.host)}[a.worker_auth]
    if per_worker and worker_token:
        print("[coordinator] workers must use per-worker credentials: "
              "python grid_auth.py issue --id <worker-id> --worker-token-file <master> --out <file>", file=sys.stderr)

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
        url = a.advertise_url or default_advertise_url(a.host, a.port, scheme)
        if cert and not url.startswith("https://"):
            raise SystemExit(f"--advertise-url {url} must be https:// when TLS is enabled")
        # On a loopback bind, also keep the UDP responder on loopback: never advertise to the LAN
        # a URL that only works on this machine.
        bind = "127.0.0.1" if _is_loopback(a.host) else ""
        try:
            responder = DiscoveryResponder(url, worker_token, bind_host=bind, port=a.discovery_port,
                                           tls_fingerprint=tls_fp)
        except ValueError as e:
            raise SystemExit(str(e))
        if not worker_token:
            print("[coordinator] WARNING: advertising without a worker token: discovery replies are unauthenticated",
                  file=sys.stderr)
        print(f"[coordinator] discovery: advertising {url} on UDP {a.discovery_port}", file=sys.stderr)

    sink = _build_sink(a)
    store = None if a.db.lower() == "none" else GridStore(a.db)
    grid = GridState(sink=sink, store=store, relay=a.relay)
    if a.relay:
        print("[coordinator] Baton token relay: ON (preempted chunks resume from their partial output)", file=sys.stderr)
    trusted = set(a.trusted_host)
    if not _is_loopback(a.host) and a.host not in ("0.0.0.0", "::"):
        trusted.add(a.host)
    if responder is not None:
        trusted.add(_hostname(responder.advertise_url))
    tls_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if cert else {}
    uvicorn.run(create_app(grid, worker_token=worker_token, client_token=client_token, discovery=responder,
                           trusted_hosts=trusted, allow_shared_worker_token=not per_worker),
                host=a.host, port=a.port, log_level="warning", **tls_kwargs)


if __name__ == "__main__":
    main()
