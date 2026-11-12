"""Edge AI Compute Grid - preemptible worker daemon.

Preemption model
----------------
Inference runs in its own asyncio task. A separate `ActivityMonitor` watchdog polls for reasons not to
run (trigger file every 100ms, OS-native HID idle every 500ms, battery / thermal / foreign CPU
load at their own slower cadence) and, on any of them, *cancels* the inference task. Cancellation unwinds `async with client.stream(...)`, which
closes the HTTP connection; Ollama/llama.cpp detects the disconnect and stops generating.

This is deliberately NOT a per-token check inside the stream loop: during model load and
prompt prefill no tokens arrive for seconds, so a per-token check cannot fire. Measured on
an M-series Mac with qwen3:4b: GPU utilisation drops from ~90% to 0% within one 250ms sample
after cancellation, whereas merely *stopping iteration* without closing the response keeps
the GPU at ~90% until the generation finishes.

Good-citizen guards
-------------------
Reasons to pause (sent verbatim as the preempt `reason`): trigger_file, hid_input,
idle_sensor_lost, on_battery, battery_low, thermal_throttle, cpu_busy, low_memory. HID idle
fails *closed* (no working sensor -> refuse to start; sensor lost at runtime -> pause); the
other guards fail open with a warning. See `idle_detection` for the policy and hysteresis.
On low_memory the model is always evicted (`keep_alive: 0`, fire-and-forget) unless
`--unload-on-preempt never`, because freeing RAM is the point of yielding under pressure.

Lease protocol
--------------
pull-task -> (task_id, lease_id). Heartbeats every 3s extend the lease and report whether
it is still ours (`lease_valid`). If the coordinator re-issued the lease (we were
partitioned) the worker cancels its now-zombie inference immediately.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import platform
import random
import re
import signal
import sys
import tempfile
import time
from typing import Optional

import httpx
import psutil

from graphrag import EXTRACTION_SCHEMA, SYSTEM_PROMPT, ExtractionError, normalize_extraction
from relay import MAX_RESUME_TOKENS, ChatML, closable, first_json, json_done, scan_checkpoint, structured_resume
from grid_auth import (TOKEN_FILE_ENV, WORKER_TOKEN_ENV, GridAuthError, bearer_headers, check_auth,
                       credential_worker_id, resolve_token)
from grid_tls import CA_FILE_ENV, FINGERPRINT_ENV, TLSConfigError, client_verify, normalize_fingerprint
from idle_detection import (
    DEFAULT_MAX_HOST_CPU_PCT, DEFAULT_MIN_FREE_RAM_GB, ActivityMonitor, ActivitySample, IdleSensorUnavailable,
    build_guard_sources,
)

GB = 1024 ** 3
TRIGGER_DIR = "/tmp" if os.name == "posix" else tempfile.gettempdir()
DEFAULT_COORDINATOR = "http://127.0.0.1:8000"
# RAM a not-yet-measured model is assumed to need when it reloads (memory guard resume level).
# A 3-4B INT4 model with a 4k context is ~2.5-3.5 GB resident; erring high only delays resume.
FALLBACK_MODEL_BYTES = 4 * GB
MODEL_SIZE_REFRESH_S = 60.0


def body(r: httpx.Response) -> dict:
    """Response JSON, or {} for non-JSON bodies (e.g. a plain-text 500). Never raises."""
    try:
        data = r.json()
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def fingerprint_tier(free_gb: float, is_apple_silicon: bool) -> int:
    if free_gb >= 24 or (is_apple_silicon and free_gb >= 16):
        return 3
    if free_gb >= 8:
        return 2
    return 1


class CancelReason:
    PREEMPTED = "preempted"
    LEASE_LOST = "lease_lost"
    SHUTDOWN = "shutdown"


class _Slip(Exception):
    """Naive resume produced invalid JSON; switch to the grammar-constrained rung."""


def mock_output(prompt: str) -> tuple[str, list[str]]:
    """Deterministic mock 'model output' for a prompt, split into 50-90 stream pieces."""
    ids = re.findall(r"\b(?:PLC-Node-\d+|B-\d+|register \d+|Equipment Log #\d+)", prompt)
    subject = ids[0] if ids else prompt[:24]
    triples = [{"subject": subject, "relation": "MENTIONS", "object": o} for o in ids[1:]] or \
              [{"subject": subject, "relation": "MENTIONS", "object": "unknown"}]
    text = json.dumps({"entities": ids, "triples": triples})
    n = 50 + hashlib.sha256(prompt.encode()).digest()[0] % 41
    step = max(1, -(-len(text) // n))
    return text, [text[i:i + step] for i in range(0, len(text), step)]


class PreemptibleEdgeWorker:
    def __init__(self, args: argparse.Namespace):
        self.token, self.token_source = resolve_token(args.token_file, WORKER_TOKEN_ENV)
        cred_id = credential_worker_id(self.token)  # a per-worker credential names the one id it is valid for
        if args.id is None:
            if cred_id is None:
                raise GridAuthError("--id is required (or use a per-worker credential, which carries its id)")
            args.id = cred_id
        elif cred_id is not None and cred_id != args.id:
            raise GridAuthError(f"--id {args.id!r} does not match the worker credential, which is for {cred_id!r}")
        self.worker_id: str = args.id
        self.ca_file: Optional[str] = args.ca_file
        self.tls_fingerprint: Optional[str] = args.tls_fingerprint
        self.coordinator_url = (args.coordinator or DEFAULT_COORDINATOR).rstrip("/")
        self.ollama_url = args.ollama.rstrip("/")
        self.model: str = args.model
        self.mock: bool = args.mock
        self.unload_policy: str = args.unload_on_preempt
        self.hostname = platform.node()
        self.is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        free_gb = psutil.virtual_memory().available / GB
        self.tier: int = args.tier or fingerprint_tier(free_gb, self.is_apple_silicon)
        # Resident size of our model as last reported by Ollama /api/ps (None until observed).
        self.model_bytes: Optional[float] = None
        self._model_size_at = 0.0
        guard_log = lambda msg: self.log("~", msg)  # noqa: E731
        self.monitor = ActivityMonitor(  # raises IdleSensorUnavailable (fail-closed HID); see _amain
            trigger_path=os.path.join(TRIGGER_DIR, f"preempt_{self.worker_id}"),
            native_idle_threshold_s=args.native_idle_seconds,
            allow_no_idle_sensor=args.allow_no_idle_sensor,
            guards=build_guard_sources(
                allow_battery=args.allow_battery, min_battery_percent=args.min_battery_percent,
                ignore_thermal=args.ignore_thermal, max_temp_c=args.max_temp_c,
                max_host_cpu_pct=args.max_host_cpu_pct, cpu_busy_samples=args.cpu_busy_samples,
                min_free_ram_gb=args.min_free_ram_gb, model_bytes=self.model_reload_bytes, log=guard_log),
            log=guard_log,
        )
        self.fatal: Optional[str] = None  # set on unrecoverable errors (bad credentials) -> exit non-zero
        self.heartbeat_interval = 3.0
        self.supports_thinking = False
        self.rng = random.Random(self.worker_id)

        # Current lease + the inference task working on it.
        self.task_id: Optional[str] = None
        self.lease_id: Optional[str] = None
        self.inference: Optional[asyncio.Task] = None
        self.cancel_reason: Optional[str] = None
        self._unloaded_this_pause = False
        self.tokens_per_sec = 0.0
        self._idle = asyncio.Event()
        self._idle.set()
        self._register_lock = asyncio.Lock()
        self._registrations = 0
        self._background: set[asyncio.Task] = set()  # fire-and-forget work (model unloads)

        # Baton token relay: the generation-so-far of the current lease, shipped to the coordinator
        # as heartbeat deltas so another worker can continue it if this lease ends early.
        self.relay_on = False            # coordinator said so at registration
        self.can_resume = self.mock      # live: verified per model in check_ollama()
        self.gen_lease: Optional[str] = None
        self.gen_pieces: list[str] = []
        self.gen_tokens = 0              # cumulative, including tokens handed to us on resume
        self.ckpt_acked = 0              # chars of our text the coordinator has stored
        self.ckpt_epoch = 0              # bumped when our text is rewritten (structured resume)
        self.resume: Optional[dict] = None

        self.coord: httpx.AsyncClient  # set in run()
        self.ollama: httpx.AsyncClient

    def log(self, sym: str, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} [{sym}] {self.worker_id}: {msg}", flush=True)

    # ------------------------------------------------------------------ coordinator I/O
    async def register(self) -> None:
        """Single-flight: if another coroutine re-registered while we waited, don't do it again
        (a second register would make the coordinator re-queue the lease we just pulled)."""
        seen = self._registrations
        async with self._register_lock:
            if self._registrations != seen:
                return
            await self._register()
            self._registrations += 1

    async def _register(self) -> None:
        mem = psutil.virtual_memory()
        payload = {
            "worker_id": self.worker_id, "hostname": self.hostname, "tier": self.tier,
            "ram_free_gb": round(mem.available / GB, 2), "cpu_percent": psutil.cpu_percent(interval=None),
            "model_name": f"{self.model}{' (mock)' if self.mock else ''}",
            "state": "PREEMPTED_USER_ACTIVE" if self.monitor.active else "IDLE_HARVESTING",
        }
        delay = 0.5
        while True:
            try:
                # The client already carries our current session header (if any), so re-registering
                # our own still-live id succeeds; a different process using this id gets 409.
                r = await self.coord.post("/worker/register", json=payload)
                if r.status_code == 409:
                    self.log("~", f"worker id {self.worker_id} is still held by a live worker (a previous run "
                                  f"that hasn't timed out yet?); retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
                    continue
                r.raise_for_status()
                reply = body(r)
                self.heartbeat_interval = reply.get("heartbeat_interval", 3.0)
                self.relay_on = bool(reply.get("relay"))
                if reply.get("session"):
                    # Binds every later call to this registration; never logged.
                    self.coord.headers["X-Nightshift-Session"] = reply["session"]
                break
            except httpx.HTTPError as e:
                self.log("~", f"coordinator unreachable ({type(e).__name__}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
        self.log("*", f"registered Tier {self.tier} | guards: {self.monitor.describe()} | auth: {self.token_source} | "
                      f"chaos trigger: touch {self.monitor.trigger_path}")

    async def _check_coordinator_auth(self, r: httpx.Response) -> None:
        """Response hook: a 401/403 means our token is missing or wrong, which no retry can fix.
        Raising GridAuthError (not an httpx.HTTPError) escapes every transient-error retry loop."""
        try:
            check_auth(r, f"Set --token-file or {' / '.join(WORKER_TOKEN_ENV)} (worker token).")
        except GridAuthError as e:
            if self.fatal is None:
                self.fatal = str(e)
                self.log("x", f"FATAL: {e}")
            raise

    # ------------------------------------------------------------------ Baton checkpoints
    @property
    def ckpt_enabled(self) -> bool:
        return self.relay_on and self.can_resume

    def progress(self, lease_id: Optional[str]) -> dict:
        """Progress fields for heartbeat/preempt: cumulative tokens, plus the not-yet-stored tail
        of our text when relay is on. Empty unless `lease_id` is the lease we are generating for."""
        if lease_id is None or lease_id != self.gen_lease:
            return {}
        out: dict = {"gen_tokens": self.gen_tokens}
        if self.ckpt_enabled:
            text = "".join(self.gen_pieces)
            if len(text) > self.ckpt_acked:
                out["ckpt_from"], out["ckpt_delta"] = self.ckpt_acked, text[self.ckpt_acked:]
        return out

    def ack_checkpoint(self, sent: dict, reply: dict, epoch: int) -> None:
        if "ckpt_delta" in sent and epoch == self.ckpt_epoch and \
                reply.get("ckpt_len") == sent["ckpt_from"] + len(sent["ckpt_delta"]):
            self.ckpt_acked = reply["ckpt_len"]

    async def notify_preempt(self, reason: str, detected_at: float, lease_id: Optional[str] = None) -> None:
        try:
            r = await self.coord.post("/worker/preempt", json={
                "worker_id": self.worker_id, "reason": reason, "detected_at": detected_at, "lease_id": lease_id,
                **self.progress(lease_id)})  # the exact text at the moment of yield: zero tokens lost
            if r.status_code == 404:
                await self.register()
        except httpx.HTTPError as e:
            # Lease will expire on its own within LEASE_TTL; correctness is preserved.
            self.log("x", f"preempt notify failed ({type(e).__name__}); relying on lease expiry")

    async def heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            state = ("PREEMPTED_USER_ACTIVE" if self.monitor.active
                     else "BUSY_INFERENCING" if self.task_id else "IDLE_HARVESTING")
            task_id, lease_id = self.task_id, self.lease_id
            prog, epoch = self.progress(lease_id), self.ckpt_epoch
            try:
                r = await self.coord.post("/worker/heartbeat", json={
                    "worker_id": self.worker_id, "task_id": task_id, "lease_id": lease_id,
                    "tokens_per_sec": round(self.tokens_per_sec, 1), "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_free_gb": round(psutil.virtual_memory().available / GB, 2), "state": state, **prog,
                }, timeout=3.0)
                if r.status_code == 200:
                    self.ack_checkpoint(prog, body(r), epoch)
                if r.status_code == 404:
                    self.log("~", "coordinator forgot us (restart?); re-registering")
                    self.cancel_inference(CancelReason.LEASE_LOST)
                    await self.register()
                elif body(r).get("lease_valid") is False and lease_id is not None and lease_id == self.lease_id:
                    self.log("!", "lease was re-issued elsewhere; cancelling zombie inference")
                    self.cancel_inference(CancelReason.LEASE_LOST)
            except httpx.HTTPError:
                pass  # transient; lease TTL covers us
            except GridAuthError:
                raise  # fatal; crashing this task shuts the worker down
            except Exception as e:  # never let the heartbeat die silently
                self.log("x", f"heartbeat error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ preemption
    def cancel_inference(self, reason: str) -> None:
        if self.inference and not self.inference.done():
            self.cancel_reason = reason
            self.inference.cancel()  # unwinds client.stream() -> socket closed -> Ollama stops

    def spawn(self, coro) -> asyncio.Task:
        """Fire-and-forget: keep a strong reference (the loop only holds weak ones) until done."""
        t = asyncio.create_task(coro)
        self._background.add(t)
        t.add_done_callback(self._background.discard)
        return t

    async def on_user_active(self, sample: ActivitySample, detected_at: float) -> None:
        self._idle.clear()
        had_task = self.inference is not None and not self.inference.done()
        lease_id = self.lease_id
        self.cancel_inference(CancelReason.PREEMPTED)
        self.tokens_per_sec = 0.0
        self.log("!", f"pausing ({sample.reason}); "
                 f"{'aborted in-flight inference, ' if had_task else ''}yielding CPU/GPU/RAM")
        # Unload in the background: it must never delay the preempt notify (the coordinator
        # re-queues the chunk on it) nor stall the activity monitor that awaits this callback.
        self._unloaded_this_pause = "low_memory" in sample.reasons
        self.spawn(self.maybe_unload_model(force=self._unloaded_this_pause, reason=sample.reason))
        await self.notify_preempt(sample.reason or "user_activity", detected_at, lease_id)

    async def on_reasons_change(self, sample: ActivitySample) -> None:
        """Memory pressure that starts during an existing pause (say, a trigger-file pause) is
        masked by the higher-priority reason, but the model must still be evicted."""
        if "low_memory" in sample.reasons and not self._unloaded_this_pause:
            self._unloaded_this_pause = True
            self.spawn(self.maybe_unload_model(force=True, reason="low_memory"))

    async def on_user_idle(self) -> None:
        self.log("+", "machine available again; resuming harvesting")
        try:  # tell the coordinator now rather than at the next 3s heartbeat
            await self.coord.post("/worker/heartbeat", json={"worker_id": self.worker_id, "state": "IDLE_HARVESTING"},
                                  timeout=3.0)
        except httpx.HTTPError:
            pass
        self._idle.set()

    async def maybe_unload_model(self, force: bool = False, reason: Optional[str] = None) -> None:
        """Evict the model with `keep_alive: 0`. `force` (low_memory) skips the `auto` pressure
        heuristic; `--unload-on-preempt never` still wins, as an explicit operator override."""
        if self.unload_policy == "never":
            return
        mem = psutil.virtual_memory()
        if not force and self.unload_policy == "auto" and mem.percent < 85 and mem.available > 4 * GB:
            return
        if self.mock:
            if force:
                self.log("!", f"{reason}: mock mode, no model to unload (RAM {mem.available / GB:.1f} GB free)")
            return
        # Measure before evicting (feeds the resume level). If Ollama says the model is not
        # resident, skip: a keep_alive=0 generate for an unloaded model can make Ollama load it
        # first - allocating GBs at exactly the moment we are yielding RAM.
        if await self.refresh_model_size(force=True) is False:
            self.log("!", f"{reason}: {self.model} not resident in Ollama; nothing to unload "
                          f"(RAM {mem.available / GB:.1f} GB free)")
            return
        try:
            await self.ollama.post("/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=5.0)
            self.log("!", f"unloaded {self.model} from memory (keep_alive=0, reason {reason}, "
                          f"RAM {mem.percent:.0f}% used, {mem.available / GB:.1f} GB free)")
        except httpx.HTTPError as e:
            self.log("x", f"model unload failed ({type(e).__name__})")

    def model_reload_bytes(self) -> float:
        """RAM a resume would cost us again (memory guard hysteresis): 0 in mock mode, the size
        Ollama last reported, or a conservative constant before we have ever seen it loaded."""
        if self.mock:
            return 0.0
        return self.model_bytes if self.model_bytes is not None else FALLBACK_MODEL_BYTES

    async def refresh_model_size(self, force: bool = False) -> Optional[bool]:
        """Learn our model's resident size from Ollama `/api/ps` (`size`: weights + KV cache).
        Returns whether the model is loaded (None: unknown / not checked).

        We count the whole `size`, even the VRAM share on discrete GPUs: overestimating only
        delays a low_memory resume, underestimating causes the reload-flap it exists to stop."""
        if self.mock or (not force and time.monotonic() - self._model_size_at < MODEL_SIZE_REFRESH_S):
            return None
        self._model_size_at = time.monotonic()
        try:
            r = await self.ollama.get("/api/ps", timeout=2.0)
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        for m in body(r).get("models") or []:
            if m.get("name") in (self.model, f"{self.model}:latest", f"{self.model.removesuffix(':latest')}"):
                if m.get("size"):
                    self.model_bytes = float(m["size"])
                return True
        return False

    # ------------------------------------------------------------------ inference
    async def check_ollama(self) -> None:
        try:
            tags = (await self.ollama.get("/api/tags", timeout=5.0)).json()
        except httpx.HTTPError as e:
            raise SystemExit(f"Ollama not reachable at {self.ollama_url} ({e}). Start `ollama serve` or use --mock.")
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise SystemExit(f"Model '{self.model}' not pulled. Available: {sorted(names) or 'none'}. "
                             f"Run `ollama pull {self.model}` or pass --model.")
        show = (await self.ollama.post("/api/show", json={"model": self.model})).json()
        self.supports_thinking = "thinking" in show.get("capabilities", [])
        self.can_resume = await ChatML.verify(self.ollama, self.model, SYSTEM_PROMPT, self.supports_thinking)

    async def infer(self, prompt: str) -> tuple[dict, float]:
        return await (self._infer_mock(prompt) if self.mock else self._infer_ollama(prompt))

    def _tick(self, piece: str, start: float, append: bool = True) -> None:
        """One decoded token: record it (append=False for grammar-resume items, which arrive whole)."""
        if append:
            self.gen_pieces.append(piece)
        self.gen_tokens += 1
        self._gen_new += 1
        self.tokens_per_sec = self._gen_new / max(0.05, time.monotonic() - start)

    async def _stream_raw(self, prompt: str, fmt: Optional[dict], limit: Optional[int], start: float,
                          append: bool, stop_at_close: bool = False) -> str:
        """Raw /api/generate stream (resume path). Leaving the loop closes the socket -> Ollama stops."""
        payload = {"model": self.model, "raw": True, "stream": True, "keep_alive": "10m", "prompt": prompt,
                   "options": {"num_ctx": 4096, "temperature": 0.1, "num_predict": limit or MAX_RESUME_TOKENS}}
        if fmt is not None:
            payload["format"] = fmt
        out: list[str] = []
        async with self.ollama.stream("POST", "/api/generate", json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"ollama HTTP {resp.status_code}: {(await resp.aread())[:200]!r}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"ollama error: {data['error']}")
                piece = data.get("response") or ""
                if piece:
                    out.append(piece)
                    self._tick(piece, start, append)
                    if stop_at_close:
                        text = "".join(self.gen_pieces)
                        if json_done(text):
                            break
                        if piece.rstrip()[-1:] in ('"', "}", "]") and closable(text) is None:
                            raise _Slip()  # syntax slip: fall back to the grammar-constrained rung
                if data.get("done"):
                    break
        return "".join(out)

    async def _resume_ollama(self, prompt: str, start: float) -> tuple[dict, float]:
        """Baton resume ladder (see relay.py): naive continuation -> grammar-per-item -> fail/restart."""
        prefix = "".join(self.gen_pieces)
        render = lambda p: ChatML.render(SYSTEM_PROMPT, prompt, p, self.supports_thinking)  # noqa: E731
        if closable(prefix) is not None:
            try:
                await self._stream_raw(render(prefix), None, None, start, append=True, stop_at_close=True)
                return normalize_extraction(first_json("".join(self.gen_pieces)), source_text=prompt), self.tokens_per_sec
            except (_Slip, ExtractionError):
                pass
        good = closable("".join(self.gen_pieces)) or closable(prefix) or ""
        if not good or scan_checkpoint(good)["arrays_opened"] == 0:
            raise ExtractionError("resume checkpoint unusable")  # -> fail -> coordinator drops it -> restart
        self.log("»", f"naive resume slipped; grammar-constrained resume from a validated boundary ({len(good)} chars)")
        self.ckpt_epoch += 1
        self.gen_pieces = [good]
        self.ckpt_acked = min(self.ckpt_acked, len(good))

        def on_text(t: str) -> None:
            self.gen_pieces = [t]

        async def raw(p: str, fmt: Optional[dict], limit: Optional[int]) -> str:
            return await self._stream_raw(p, fmt, limit, start, append=False)
        out = await structured_resume(raw, render, good, on_text=on_text)
        return normalize_extraction(out, source_text=prompt), self.tokens_per_sec

    async def _infer_ollama(self, prompt: str) -> tuple[dict, float]:
        start = time.monotonic()
        self._gen_new = 0
        if self.resume:
            return await self._resume_ollama(prompt, start)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "format": EXTRACTION_SCHEMA,
            "stream": True,
            "keep_alive": "10m",
            "options": {"num_ctx": 4096, "temperature": 0.1, "num_predict": 4 * MAX_RESUME_TOKENS},
        }
        if self.supports_thinking:
            payload["think"] = False  # reasoning tokens are wasted work for extraction
        final = {}
        async with self.ollama.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"ollama HTTP {resp.status_code}: {(await resp.aread())[:200]!r}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    self._tick(piece, start)
                if data.get("done"):
                    final = data
                    break
        if final.get("eval_count") and final.get("eval_duration"):
            self.tokens_per_sec = final["eval_count"] / (final["eval_duration"] / 1e9)  # exact decode speed
            self.gen_tokens = final["eval_count"]  # exact (stream chunks can carry >1 token)
        return normalize_extraction("".join(self.gen_pieces)), self.tokens_per_sec

    async def _infer_mock(self, prompt: str) -> tuple[dict, float]:
        """Simulated token stream (~25 tok/s) with a plausible extraction, cancellable at any await.
        The output is a pure function of the prompt, streamed as 50-90 'tokens', so a relayed
        generation can be checked byte-for-byte against an uninterrupted one."""
        target, pieces = mock_output(prompt)
        start_at = 0
        if self.resume:
            if not target.startswith(self.resume["text"]) or self.resume["tokens"] > len(pieces):
                raise ExtractionError("resume prefix does not match this model's output")
            start_at = self.resume["tokens"]
        start = time.monotonic()
        for i in range(start_at, len(pieces)):
            await asyncio.sleep(0.04)
            self.gen_pieces.append(pieces[i])
            self.gen_tokens = i + 1
            self.tokens_per_sec = (i + 1 - start_at) / (time.monotonic() - start)
        out = "".join(self.gen_pieces)
        if out != target:
            raise ExtractionError("relayed output diverged from the uninterrupted output")
        return normalize_extraction(out), self.tokens_per_sec

    # ------------------------------------------------------------------ main loop
    async def report(self, path: str, payload: dict, deadline_s: float = 30.0) -> Optional[httpx.Response]:
        """POST with retries on transport errors and 5xx until `deadline_s` elapses.

        Fenced by lease_id, so retries are safe. The window is deliberately longer than a
        coordinator restart: a recovered coordinator grants in-flight leases a fresh TTL, so a
        result computed during the outage is still accepted instead of being thrown away."""
        r, i, start = None, 0, time.monotonic()
        while True:
            try:
                r = await self.coord.post(path, json=payload)
                if r.status_code < 500:
                    return r
                why = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                why = type(e).__name__
            delay = min(0.5 * 2 ** i, 4.0)
            if time.monotonic() - start + delay > deadline_s:
                self.log("x", f"{path} failed ({why}); giving up after {deadline_s:.0f}s (lease TTL will recover the chunk)")
                return r
            self.log("~", f"{path} failed ({why}), retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            i += 1

    async def work_one(self, task: dict) -> None:
        self.task_id, self.lease_id, self.cancel_reason = task["task_id"], task["lease_id"], None
        self.resume = task.get("resume") if self.can_resume else None
        self.gen_lease, self.ckpt_epoch = self.lease_id, self.ckpt_epoch + 1
        self.gen_pieces = [self.resume["text"]] if self.resume else []
        self.gen_tokens = self.resume["tokens"] if self.resume else 0
        self.ckpt_acked = len(self.resume["text"]) if self.resume else 0
        if self.resume:
            self.log("»", f"chunk #{task['chunk_index']} resumes at token {self.gen_tokens} (Baton relay)")
        base = {"worker_id": self.worker_id, "task_id": self.task_id, "lease_id": self.lease_id}
        self.inference = asyncio.create_task(self.infer(task["prompt"]))
        try:
            await asyncio.wait({self.inference})  # don't propagate the child's cancellation to us
            if self.inference.cancelled():
                if self.cancel_reason == CancelReason.PREEMPTED:
                    self.log("!", f"chunk #{task['chunk_index']} aborted mid-stream; coordinator re-queues it")
                return
            exc = self.inference.exception()
            if exc is not None:
                kind = "invalid extraction" if isinstance(exc, ExtractionError) else type(exc).__name__
                self.log("x", f"chunk #{task['chunk_index']} failed: {kind}: {exc}")
                await self.report("/worker/fail", {**base, "error": f"{kind}: {exc}"[:500]})
                return
            result, tps = self.inference.result()
            await self.refresh_model_size()  # the model is certainly loaded right now
            r = await self.report("/worker/complete", {**base, "result": result, "tokens_per_sec": round(tps, 1),
                                                       "gen_tokens": self.gen_tokens})
            if r is not None and r.status_code == 409:
                self.log("~", f"chunk #{task['chunk_index']} result discarded: lease lost (already re-issued)")
            elif r is not None and r.status_code >= 500:
                self.log("x", f"chunk #{task['chunk_index']} completion failed (HTTP {r.status_code}); lease will expire")
            elif r is not None and body(r).get("status") == "rejected":
                self.log("x", f"chunk #{task['chunk_index']} rejected by coordinator: {body(r).get('error')}")
        finally:
            if self.inference and not self.inference.done():  # we were cancelled (shutdown)
                self.inference.cancel()
            self.task_id = self.lease_id = self.inference = None
            self.tokens_per_sec = 0.0

    async def main_loop(self) -> None:
        while True:
            await self._idle.wait()
            try:
                r = await self.coord.post("/worker/pull-task", params={"worker_id": self.worker_id})
                if r.status_code == 404:
                    await self.register()
                    continue
                task = body(r).get("task") if r.status_code == 200 else None
            except httpx.HTTPError as e:
                self.log("~", f"pull failed ({type(e).__name__}); backing off")
                await asyncio.sleep(2.0)
                continue
            if not task:
                await asyncio.sleep(0.5)
                continue
            if self.monitor.active:
                # Human returned while pull was in flight; our earlier preempt may have reached the
                # coordinator *before* this lease was issued. Hand it back now (not a failure).
                await self.notify_preempt(self.monitor.reason or "user_activity", time.time(), task["lease_id"])
                continue
            try:
                await self.work_one(task)
            except GridAuthError:
                raise
            except Exception as e:  # keep harvesting; the lease TTL covers whatever we dropped
                self.log("x", f"unexpected error on chunk #{task.get('chunk_index')}: {type(e).__name__}: {e}")
                await asyncio.sleep(1.0)

    async def tls_verify(self):
        """Authenticate the coordinator's certificate *before* the first request carries our token.
        With a pinned fingerprint that needs the server up, so wait for it like register() does."""
        delay = 0.5
        while True:
            try:
                return await asyncio.to_thread(client_verify, self.coordinator_url, self.ca_file, self.tls_fingerprint)
            except TLSConfigError as e:
                self.fatal = str(e)
                self.log("x", f"FATAL: {e}")
                raise GridAuthError(str(e)) from e
            except OSError as e:
                self.log("~", f"coordinator unreachable for TLS check ({type(e).__name__}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)

    async def run(self) -> None:
        verify = await self.tls_verify()
        async with httpx.AsyncClient(base_url=self.coordinator_url, timeout=5.0, headers=bearer_headers(self.token),
                                     verify=verify, event_hooks={"response": [self._check_coordinator_auth]}) as coord, \
                httpx.AsyncClient(base_url=self.ollama_url, timeout=httpx.Timeout(120.0, connect=5.0)) as ollama:
            self.coord, self.ollama = coord, ollama
            if not self.mock:
                await self.check_ollama()
            await self.register()
            bg = [asyncio.create_task(self.heartbeat_loop(), name="heartbeat"),
                  asyncio.create_task(self.monitor.run(self.on_user_active, self.on_user_idle, self.on_reasons_change),
                                      name="activity-monitor")]
            main = asyncio.current_task()

            def _crashed(t: asyncio.Task) -> None:  # a dead watchdog means no preemption: fail loudly
                if not t.cancelled() and t.exception() is not None:
                    if not self.fatal:  # a fatal error was already logged by whoever set it
                        self.log("x", f"background task {t.get_name()} crashed: {t.exception()!r}; shutting down")
                    main.cancel()
            for t in bg:
                t.add_done_callback(_crashed)
            try:
                await self.main_loop()
            finally:
                for t in bg:
                    t.cancel()
                # Graceful drain: hand back any lease immediately instead of waiting for TTL expiry.
                self.cancel_inference(CancelReason.SHUTDOWN)
                if not self.fatal:  # with bad credentials the hand-back is rejected too; lease TTL reclaims it
                    try:
                        r = await self.coord.post("/worker/preempt", json={
                            "worker_id": self.worker_id, "reason": "shutdown", "detected_at": time.time()},
                            timeout=2.0)
                        if r.status_code == 200 and body(r).get("reclaimed"):
                            self.log("*", "shutdown: in-flight lease handed back to coordinator")
                    except httpx.HTTPError:
                        pass


async def resolve_coordinator(args: argparse.Namespace, token: Optional[str]) -> str:
    """Coordinator base URL: explicit `--coordinator` > `--discover` > the loopback default.

    `discovery` is imported lazily so a missing optional dependency (zeroconf) only matters to
    workers that ask for discovery. Failure is fatal rather than a fallback to loopback: a
    worker that asked to find the grid and silently polls 127.0.0.1 forever looks healthy but
    does nothing.
    """
    if args.coordinator:
        if args.discover:
            print(f"[warn] --discover ignored: --coordinator {args.coordinator} given explicitly", file=sys.stderr)
        return args.coordinator
    if not args.discover:
        return DEFAULT_COORDINATOR
    try:
        discovery = importlib.import_module("discovery")
    except ImportError as e:
        raise SystemExit(f"FATAL: --discover needs discovery.py and its dependencies ({e}). "
                         f"Install them or pass --coordinator URL.")
    errors = (getattr(discovery, "DiscoveryError", RuntimeError), OSError)
    try:
        found = await asyncio.to_thread(discovery.discover_coordinator_info, token, timeout=3.0,
                                        **({"port": args.discovery_port} if args.discovery_port else {}))
        url = found.url
        fp = found.tls_fingerprint
        if fp and url.startswith("https://"):
            if args.tls_fingerprint and normalize_fingerprint(args.tls_fingerprint) != fp:
                raise SystemExit(f"FATAL: discovered coordinator {url} presents TLS fingerprint {fp}, not the "
                                 f"--tls-fingerprint you configured")
            if not args.tls_fingerprint and not args.ca_file:
                args.tls_fingerprint = fp  # authenticated by the discovery HMAC (when we hold a token)
    except errors as e:
        raise SystemExit(f"FATAL: --discover found no coordinator on the LAN ({e}). Check that the coordinator "
                         f"is running on this subnet and that multicast/UDP is not blocked, or pass --coordinator URL.")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise SystemExit(f"FATAL: --discover returned an invalid coordinator URL {url!r}")
    return url


async def _amain(args: argparse.Namespace) -> None:
    try:
        worker = PreemptibleEdgeWorker(args)
    except (GridAuthError, IdleSensorUnavailable) as e:
        raise SystemExit(f"FATAL: {e}")
    if args.discover:
        worker.coordinator_url = (await resolve_coordinator(args, worker.token)).rstrip("/")
        worker.tls_fingerprint = args.tls_fingerprint
        if not args.coordinator:
            worker.log("*", f"discovered coordinator at {worker.coordinator_url}")
    main = asyncio.create_task(worker.run())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, main.cancel)
        except NotImplementedError:  # Windows
            pass
    try:
        await main
    except (asyncio.CancelledError, GridAuthError):
        pass
    if worker.fatal:
        raise SystemExit(1)  # already logged where it was detected


def _percent(value: str) -> float:
    v = float(value)
    if not 0 < v <= 100:
        raise argparse.ArgumentTypeError(f"{value} is not in (0, 100]")
    return v


def _positive_int(value: str) -> int:
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError(f"{value} must be >= 1")
    return v


def _percent_or_off(value: str) -> float:
    """CPU threshold: (0, 100], or 0 / "off" to disable the guard."""
    if value.strip().lower() in ("off", "none", "0", "0.0"):
        return 0.0
    return _percent(value)


def _nonneg_gb(value: str) -> float:
    v = float(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"{value} must be >= 0 (0 disables)")
    return v


_warned_deprecated: set[str] = set()


def _resolve_cpu_threshold(a: argparse.Namespace) -> None:
    """Fold the deprecated `--cpu-busy-percent` into canonical `--max-host-cpu-pct`."""
    if a.cpu_busy_percent is not None:
        if "--cpu-busy-percent" not in _warned_deprecated:
            _warned_deprecated.add("--cpu-busy-percent")
            print("[warn] --cpu-busy-percent is deprecated; use --max-host-cpu-pct "
                  f"({'ignored: --max-host-cpu-pct also given' if a.max_host_cpu_pct is not None else 'same meaning'})",
                  file=sys.stderr)
        if a.max_host_cpu_pct is None:
            a.max_host_cpu_pct = a.cpu_busy_percent
    if a.max_host_cpu_pct is None:
        a.max_host_cpu_pct = DEFAULT_MAX_HOST_CPU_PCT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # allow_abbrev=False: otherwise `--token SECRET` would silently parse as `--token-file SECRET`.
    p = argparse.ArgumentParser(description="Preemptible Edge AI Worker Daemon", allow_abbrev=False)
    p.add_argument("--id", default=None,
                   help="Unique worker ID (e.g. node-office-pc); optional with a per-worker credential, which names it")
    p.add_argument("--coordinator", default=None, metavar="URL", help=f"Coordinator base URL (default {DEFAULT_COORDINATOR})")
    p.add_argument("--discover", action="store_true",
                   help="Find the coordinator on the LAN at startup (ignored if --coordinator is given)")
    p.add_argument("--discovery-port", type=int, default=None, metavar="PORT",
                   help="UDP port for --discover (default 47474; must match the coordinator's)")
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    p.add_argument("--model", default=os.environ.get("GRID_MODEL", "qwen2.5:3b"))
    p.add_argument("--tier", type=int, choices=(1, 2, 3), default=None, help="Override capability tier")
    p.add_argument("--mock", action="store_true", help="Simulated token stream (no Ollama needed)")
    p.add_argument("--native-idle-seconds", type=float, default=None, metavar="N",
                   help="Preempt while OS HID idle time < N seconds (macOS ioreg / Windows GetLastInputInfo / "
                        "Linux Wayland D-Bus or X11 xprintidle). Fails closed: exits if no sensor works")
    p.add_argument("--allow-no-idle-sensor", action="store_true",
                   help="With --native-idle-seconds: keep running (fail-open, warn) if no HID sensor works")
    p.add_argument("--allow-battery", action="store_true",
                   help="Keep harvesting on battery power (default: pause while unplugged)")
    p.add_argument("--min-battery-percent", type=_percent, default=None, metavar="P",
                   help="Also pause while plugged in but charged below P%% (default: off)")
    p.add_argument("--ignore-thermal", action="store_true",
                   help="Keep harvesting while the OS reports thermal throttling")
    p.add_argument("--max-temp-c", type=float, default=90.0, metavar="C",
                   help="Linux: pause while the hottest sensor is >= C (resume 5C below; default 90)")
    p.add_argument("--max-host-cpu-pct", type=_percent_or_off, default=None, metavar="P",
                   help=f"Pause while non-worker CPU (excluding this and other workers, Ollama/llama.cpp) stays "
                        f"above P%% of host capacity; resume below 0.7*P (default {DEFAULT_MAX_HOST_CPU_PCT:g}; 0/off disables)")
    p.add_argument("--cpu-busy-percent", type=_percent, default=None, metavar="P",
                   help="Deprecated alias for --max-host-cpu-pct")
    p.add_argument("--cpu-busy-samples", type=_positive_int, default=3, metavar="N",
                   help="Consecutive ~1s samples needed to pause/resume on CPU load (default 3)")
    p.add_argument("--min-free-ram-gb", type=_nonneg_gb, default=DEFAULT_MIN_FREE_RAM_GB, metavar="GB",
                   help=f"Pause (and unload the model) while available RAM stays below GB; resume once it can "
                        f"reload the model with 1 GB to spare (default {DEFAULT_MIN_FREE_RAM_GB:g}; 0 disables)")
    p.add_argument("--token-file", default=None, metavar="PATH",
                   help=f"Coordinator worker token file (else ${TOKEN_FILE_ENV} / env {' / '.join(WORKER_TOKEN_ENV)})")
    p.add_argument("--tls-fingerprint", default=os.environ.get(FINGERPRINT_ENV) or None, metavar="SHA256",
                   help=f"Pin the coordinator's TLS certificate (sha256:<hex>, printed by the coordinator; "
                        f"--discover fills it in) [env {FINGERPRINT_ENV}]")
    p.add_argument("--ca-file", default=os.environ.get(CA_FILE_ENV) or None, metavar="PEM",
                   help=f"Verify the coordinator against this CA / certificate file [env {CA_FILE_ENV}]")
    p.add_argument("--unload-on-preempt", choices=("auto", "always", "never"), default="auto",
                   help="Send keep_alive=0 on preempt: auto = only under memory pressure (always on low_memory); "
                        "never = also not on low_memory")
    a = p.parse_args(argv)
    _resolve_cpu_threshold(a)
    return a


if __name__ == "__main__":
    asyncio.run(_amain(parse_args()))
