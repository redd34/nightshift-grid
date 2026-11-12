"""SQLite persistence for the coordinator's grid state.

Design
------
* `GridState` stays the in-memory source of truth for reads (fast scheduling); this store is a
  write-through journal. Every public GridState mutation marks the rows it touched dirty and
  `commit()`s them in ONE transaction, so a crash can never persist half a state transition.
* Rows are stored as the model's JSON plus a few indexed columns (state, job_id) so the schema
  can evolve without migrations for every new field; `PRAGMA user_version` guards real breaks.
* WAL + synchronous=NORMAL: commits are atomic and crash-consistent; on sudden *power loss* the
  last few ms of transitions may roll back, which the lease protocol tolerates (a rolled-back
  completion is simply re-leased, and the JSONL sink dedupes by task_id).
* Heartbeat lease extensions are deliberately NOT persisted - recovery grants every live lease
  a fresh TTL instead - which keeps the hot 3s heartbeat path write-free.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id   TEXT PRIMARY KEY,
    body     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id  TEXT PRIMARY KEY,
    job_id   TEXT NOT NULL,
    state    TEXT NOT NULL,
    body     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks(job_id);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state);
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    body      TEXT NOT NULL
);
"""


class StoreVersionError(RuntimeError):
    pass


class GridStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI may run sync code on different threads at startup /
        # shutdown; all mutations still happen on the single event-loop thread.
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise StoreVersionError(f"{self.path} has schema v{version}; this coordinator expects v{SCHEMA_VERSION}")
        self.db.executescript(_SCHEMA)
        self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def commit(self, jobs: Iterable[tuple[str, str]], tasks: Iterable[tuple[str, str, str, str]],
               workers: Iterable[tuple[str, str]]) -> None:
        """Atomically upsert rows. jobs: (job_id, json); tasks: (task_id, job_id, state, json);
        workers: (worker_id, json)."""
        jobs, tasks, workers = list(jobs), list(tasks), list(workers)
        if not (jobs or tasks or workers):
            return
        cur = self.db.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            if jobs:
                cur.executemany("INSERT INTO jobs(job_id, body) VALUES (?, ?) "
                                "ON CONFLICT(job_id) DO UPDATE SET body=excluded.body", jobs)
            if tasks:
                cur.executemany("INSERT INTO tasks(task_id, job_id, state, body) VALUES (?, ?, ?, ?) "
                                "ON CONFLICT(task_id) DO UPDATE SET state=excluded.state, body=excluded.body", tasks)
            if workers:
                cur.executemany("INSERT INTO workers(worker_id, body) VALUES (?, ?) "
                                "ON CONFLICT(worker_id) DO UPDATE SET body=excluded.body", workers)
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise

    def load_jobs(self) -> Iterator[dict]:
        for (body,) in self.db.execute("SELECT body FROM jobs"):
            yield json.loads(body)

    def load_tasks(self) -> Iterator[dict]:
        for (body,) in self.db.execute("SELECT body FROM tasks"):
            yield json.loads(body)

    def load_workers(self) -> Iterator[dict]:
        for (body,) in self.db.execute("SELECT body FROM workers"):
            yield json.loads(body)

    def close(self) -> None:
        self.db.close()
