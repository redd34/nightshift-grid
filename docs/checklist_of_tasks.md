# Phase 2 Production Hardening & MCP Integration (`PHASE2_CHECKLIST.md`)

## Objective
Take the Nightshift Grid PoC from an in-memory demo to a crash-resilient, zero-config local AI engine that can be triggered directly from AI IDEs (Claude Code, Kiro, Cursor) via MCP. Close all remaining gaps identified in the Phase 1 review.

> **Status (2026-11-12): all six sections are complete.**
> - 361 tests pass.
> - The live 4-phase chaos benchmark on `qwen3:4b` passes all 9 checks with 96/96 records, 0 missing and 0 duplicates.
>
> Legend: `[x]` done · `[~]` done differently from the spec (see note).

---

## 1. SQLite WAL State Store (Crash-Resilient Coordinator) ✅
- [x] **Create `store.py` using SQLite in `WAL` mode** (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`).
  - *Implemented as `grid_store.py`. Each state transition is committed atomically in one transaction, with a schema-version guard.*
- [x] **Persist Jobs & Tasks** (`task_id`, `job_id`, `chunk_index`, `prompt`, `required_tier`, `state`, `assigned_worker`, `lease_id`, `lease_expires_at`, `attempts`, `result_json`, `completed_at`).
  - *All of these fields are persisted, plus queue position, failure/preemption counts and a `workers` table. Rows are stored as JSON with indexed `state`/`job_id` columns.*
- [x] **Startup Recovery**, losing zero work.
  - *Recovered `LEASED` chunks get a fresh TTL, so the worker that held one still completes it.*
  - *Live: 3s outage, ready 0.23s after restart, 24 attempts for 24 chunks, and 2/2 in-flight chunks kept by their original holders.*
- [x] **Unit Test**: restarting the coordinator mid-job keeps `COMPLETED` chunks and recovers `LEASED` chunks once their TTL expires.
  - *In `tests/test_durability.py`, plus 25 fuzz runs of random operations with crash-restarts mixed in.*

---

## 2. Fail-Closed Idle Detection & Host CPU/Memory Pressure Guard ✅
- [x] **Fail-Closed Default:** with `--native-idle-seconds`, if no HID reader works, the worker exits at startup unless `--allow-no-idle-sensor` is passed.
  - *Readers: `ioreg`, `GetLastInputInfo`, GNOME/Wayland Mutter `GetIdletime` and freedesktop `GetSessionIdleTime` over D-Bus, and `xprintidle`.*
  - *Also fails closed at runtime: 3 failed reads in a row pause the worker with reason `idle_sensor_lost`.*
  - *Verified live: an unreadable sensor exits with code 1 and an actionable message.*
- [x] **Host CPU & RAM Spike Preemption:** `--max-host-cpu-pct` (default **65.0**) and `--min-free-ram-gb` (default **2.0**).
  - [x] Measures **non-worker** CPU. It excludes this worker, `ollama*`/`llama-*` processes and co-located `worker_daemon.py` processes, and takes a baseline on the first poll so startup can't trip it. *Live: 0 unintended preemptions in phases 1–3.*
  - [x] Uses the same `task.cancel()` + `/worker/preempt` flow, with reasons `cpu_busy` / `low_memory`.
  - *Live CPU spike (busy loop on all 16 cores): all 3 workers yielded in **1.9–2.7s**, 0 leases went out while the load ran, and all resumed **2.1s** after it stopped.*
  - *RAM (mock smoke test): paused with `low_memory` after **2.0s** when the floor was set above free memory.*
  - *The RAM guard resumes only at floor + model size + 1 GB and enforces a minimum pause that grows if it keeps re-triggering, so unloading the model can't cause a pause/reload loop. This is covered by a closed-loop simulation test.*
  - *`--cpu-busy-percent` remains as a deprecated alias.*
- [x] **VRAM / Memory Eviction on Preempt:** a fire-and-forget `keep_alive: 0` is always sent on `low_memory`. It's skipped if Ollama reports the model isn't loaded, and `--unload-on-preempt never` turns it off.

---

## 3. Model Context Protocol (MCP) Server (`mcp_server.py`) ✅
- [x] **Add `mcp` to `requirements.txt`.**
  - *`mcp>=2.3`. In the 2.x SDK `FastMCP` is renamed `MCPServer`.*
- [x] **Implement 4 MCP Tools:** `grid_status()`, `dispatch_mapreduce_text(...)` (with progress notifications and truncation notes), `ingest_directory_to_graph(...)` (writes `<out>/<job_id>/{manifest, output_graph.jsonl, graph.json}`) and `query_local_graph(...)` (1/2-hop triples with evidence, plus optional read-only Qdrant).
  - *Security: file access only inside `NIGHTSHIFT_MCP_ROOTS` (symlinks resolved, `..` and absolute globs rejected), secret files refused, an egress warning for non-loopback coordinators, and tokens never logged.*
- [x] **Smoke Test MCP** over a real stdio MCP session:
  - *Ingesting the repo's own `**/*.md`: 6 files, 31 chunks, 31/31 done, dispatch → done 32.4s on mock workers, reduce 2ms, query 1.9ms.*
  - *Live `qwen3:4b`: 1 sample doc took 9.1s and gave 24 entities, with cited query facts.*
  - *39 tests in `tests/test_mcp.py`.*

---

## 4. Zero-Config LAN Auto-Discovery & Shared Secret Auth ✅
- [~] **Shared Token Auth (`--grid-token` / `NIGHTSHIFT_TOKEN`).**
  - *Bearer tokens on `/worker/*` **and** `/v1/*`, with separate worker and client roles.*
  - *`NIGHTSHIFT_TOKEN` and `NIGHTSHIFT_TOKEN_FILE` are supported, alongside `GRID_*`.*
  - *`--auto-token` generates 0600 token files on the first run and reuses them; the value is printed once, and only to a TTY.*
  - *Deliberate difference: there's no `--grid-token VALUE` flag, because values on the command line show up in `ps`. Use `--token-file`.*
- [x] **UDP Broadcast Auto-Discovery (`--discover`).**
  - *Zero-dependency UDP; `zeroconf` isn't needed.*
  - *The coordinator replies with an **HMAC proof keyed by the worker token**, so a rogue host can't impersonate it and harvest tokens. Replays fail, two valid coordinators are an error, and replies are size-capped and rate-limited.*
  - *Live: a worker found the coordinator via `--discover`, and a wrong token was rejected.*

---

## 5. Optional Local Qdrant Vector + Graph Sink (`qdrant_sink.py`) ✅
- [x] **Create `qdrant_sink.py`:** `--qdrant-url` / `--qdrant-memory`, nomic-embed-text or a deterministic hash embedder, idempotent upserts into `nightshift_graph`, with payload `job_id, chunk_index, task_id, worker_id, model, entities, triples, source_text, completed_at`.
  - *Writes are queued off the event loop and batched. Failures go to an on-disk retry file and are replayed. One embedder per collection is enforced, and only `nightshift_*` names are accepted.*
  - *`python qdrant_sink.py backfill` rebuilds the index from the JSONL.*
  - *Live: completion → searchable p50 **0.085s**, p95 0.10s. Qdrant outage: 6/6 completions still succeeded, and 6/6 were replayed after restart.*
  - *Pre-existing collections on the shared Qdrant were verified unchanged; only `nightshift_dev_*` collections were created.*
- [x] **Graceful Fallback:** the JSONL is the always-on primary sink, and Qdrant errors never fail a completion.

---

## 6. Single-Command CLI & Demo Harness (`nightshift.py`) ✅
- [x] **Create a unified CLI entrypoint:** `up`, `down`, `submit`, `status`, `chaos-test` and `token`.
  - [x] `nightshift.py up --workers 3` starts the coordinator, N workers and the Rich dashboard in one terminal.
    - *Tokens and state live in `.nightshift/`.*
    - *Each process runs in its own process group; workers shut down first and hand their leases back.*
    - *Smoke-tested with no orphaned processes left behind.*
  - [x] `nightshift.py chaos-test` runs a **4-phase** benchmark: preemption, worker SIGKILL, coordinator SIGKILL + SQLite recovery, and a host CPU spike. It prints a Markdown table.
- [x] **Save README** with the final Phase 2 benchmark numbers.

---

## Acceptance Criteria & Final Report
1. [x] **Unit tests:** 338 passed, 0 failed (`.venv/bin/python -m pytest -q tests`); pyflakes clean.
2. [x] **Live timings** (`qwen3:4b`, 3 workers, 24 chunks per phase):

   | Phase | Fault | Result |
   |---|---|---|
   | 1 | trigger file mid-stream | 24/24 in 47.3s; reclaim 0.011s, re-lease +0.82s |
   | 2 | SIGKILL worker | 24/24 in 45.8s; lease expired 11.9s, chunk done 16.8s |
   | 3 | **SIGKILL coordinator + SQLite restart** | 24/24 in 46.4s; ready 0.23s, **0 wasted attempts**, 2/2 in-flight kept |
   | 4 | **host CPU spike** (16 cores) | 24/24 in 54.4s; all workers yielded in ≤2.65s and resumed in ≤2.13s |
   | — | **MCP directory ingestion** | 31 chunks → graph in 32.4s (mock); 1 doc in 9.1s (live) |
   | — | Output | 96/96 records, 0 missing, 0 duplicates, 434 triples |

3. [x] **Edge cases found and fixed or documented:**
   - Ollama JSON-schema decoding injects `:` into values (`B-120` → `B-12:0`). These are repaired against the source text: 64 repaired, 0 left.
   - The RAM guard could loop (unloading the model frees RAM, the worker resumes, reloads and pauses again). Fixed by the resume threshold and the growing minimum pause.
   - Sending `keep_alive: 0` for an unloaded model could make Ollama load it first. That case is now skipped.
   - Co-located workers would have counted each other's CPU. They're now excluded.
   - Without proof of the token, discovery could leak tokens to a rogue host. The HMAC proof prevents this.

### Still open (outside this checklist)
- ~~**No TLS.**~~ *Fixed:* a LAN bind serves HTTPS (`--auto-tls` or `--tls-cert/--tls-key`) and refuses plaintext unless `--allow-plaintext` is passed. Clients pin the fingerprint, which `--discover` delivers inside the HMAC proof.
- ~~**A worker token can still register new worker ids.**~~ *Fixed:* LAN workers use per-worker credentials, `HMAC(master, worker_id)`, which are valid for one id only (`python grid_auth.py issue`). The master token is no longer accepted from workers (`--worker-auth`).
- **Results aren't cross-checked.** A compromised worker can still return bad extractions for chunks it leases.
- **Single coordinator.** It's durable but not highly available.
- **Qdrant backfill after a SIGKILL.** Records still in the sink queue when the coordinator is SIGKILLed (under a second's worth) need `backfill`.
- **Not run on real hardware:** the Linux, Windows and Wayland sensors and installers, battery and thermal guards (tested with fakes only), live `keep_alive: 0` under real memory pressure, and multi-NIC discovery.
- **Multi-machine pilot** has not been run yet.
- **Extraction-quality evaluation** has not been done yet.
