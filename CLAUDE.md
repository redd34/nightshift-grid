# Brownfield Edge AI Compute Grid (PoC Specification)

## Context & Objective
We are building a zero-CapEx, fault-tolerant distributed Edge AI compute grid ("Datadog + MapReduce for idle local silicon").
Instead of tensor-splitting a large model across slow LAN links, each node runs an independent small model (`qwen2.5:3b`, `phi4-mini`, or `nomic-embed-text` via Ollama/llama.cpp) and pulls data-parallel tasks from a central Coordinator.

## Core Architectural Invariants
1. **Good-Citizen Preemption:** Worker daemons harvest idle cycles on heterogeneous machines (office PCs, legacy servers, laptops). The millisecond human activity (HID input / CPU spike) is detected, the worker MUST close its active inference HTTP stream (sending TCP FIN/RST so Ollama/llama.cpp immediately stops token generation), set `keep_alive: 0` if memory pressure is high, and fire `POST /worker/preempt` to the Coordinator.
2. **Lease-Based Fault Tolerance:** Every task chunk is leased for `LEASE_TTL_SECONDS` (12s). Workers send heartbeats every 3s while streaming tokens. If a worker is preempted, loses Wi-Fi, or throttles, the lease expires and the Coordinator automatically re-queues the chunk to another idle node.
3. **Capability Tiers:**
   - `Tier 1` (<8GB free RAM): Embeddings, chunking, lightweight classification.
   - `Tier 2` (8-24GB free RAM): 3B-8B INT4 structured JSON entity/relationship extraction (GraphRAG ingestion).
   - `Tier 3` (>24GB free RAM / Apple Silicon / GPU): Heavy verification, consensus judging, synthesis.

## Current File Structure
- `coordinator.py`: FastAPI Control Plane, task state machine, lease reaper, and WebSocket/REST telemetry endpoints.
- `worker_daemon.py`: Async worker agent with hardware fingerprinting (`psutil`), streaming Ollama cancellation, and human-activity preemption simulation.
- `dashboard.py`: Live `rich` TUI showing node status (`IDLE_HARVESTING`, `PREEMPTED_USER_ACTIVE`, `OFFLINE`), live tok/s, and MapReduce progress.
- `demo_submit.py`: Submits a 24-chunk GraphRAG entity extraction batch job (`--wait` to block on completion).
- `idle_detection.py`: `ActivityMonitor` (trigger file + OS-native HID idle: macOS ioreg, Windows GetLastInputInfo, Linux xprintidle).
- `graphrag.py`: extraction JSON schema, `normalize_extraction()`, idempotent `JsonlGraphSink` (`output_graph.jsonl`).
- `chaos_test.py`: E2E chaos harness (preemption + SIGKILL phases, zero-loss verification) → `chaos_report*.json`.
- `grid_store.py`: SQLite (WAL) journal for coordinator crash recovery; `grid_auth.py`: worker/client bearer-token resolution + per-worker credentials (`python grid_auth.py issue`); `grid_tls.py`: self-signed cert (`--auto-tls`) + client fingerprint pinning / CA verification.
- `ingest.py` (files → chunks → jobs + provenance manifest), `reduce_graph.py` (entity resolution, graph.json/GraphML, Qdrant sink, `query`).
- `nightshift.py` (one-command CLI: up/down/submit/status/chaos-test/token), `mcp_server.py` (MCP tools for AI IDEs, see docs/MCP.md), `discovery.py` (HMAC-proven UDP LAN discovery), `qdrant_sink.py` (live Qdrant sink with disk spool).
- `deploy/`: per-user worker service installers (launchd / systemd --user / Windows Task Scheduler), coordinator systemd unit, token generator.
- Qdrant at localhost:6333 holds OTHER projects' collections: only ever write `nightshift_*` collections (enforced in code); never delete/recreate.
- `tests/`: pytest unit tests for the coordinator state machine (fake clock), normalizer, and idle detection.

Dev: `.venv/bin/python -m pytest -q tests`, `.venv/bin/python chaos_test.py --mock` (or `--model qwen3:4b` for live Ollama).

## Tasks for Claude Code
1. **Verify Environment & Dependencies:** Create a `.venv`, install `requirements.txt`, and verify whether local `ollama` is running on port `11434` (if not, test using `--mock` mode first, then wire up live Ollama with a small model like `qwen2.5:3b` or `llama3.2:3b`).
2. **OS-Native Idle Detection:** Enhance `worker_daemon.py` `check_user_activity()` to read real macOS IOKit `HIDIdleTime` (`ioreg -c IOHIDSystem`) and Linux/Windows equivalents alongside the `/tmp/preempt_<worker_id>` chaos trigger file.
3. **Qdrant / GraphRAG Sink:** Add an optional sink in `coordinator.py` so completed JSON entity-relationship triples and embeddings are upserted into a local Qdrant instance (or persisted to `output_graph.jsonl`).
4. **End-to-End Chaos Test:** Spin up `coordinator.py`, launch 3 `worker_daemon.py` processes (with different tiers/IDs), submit a batch via `demo_submit.py`, trigger preemption mid-flight on Worker 2 (`touch /tmp/preempt_node-2`), and verify zero dropped chunks. Report back with benchmark metrics and recovery latency.
