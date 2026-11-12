"""Worker wiring for the guards: CLI flags/defaults, fail-closed startup, low_memory eviction, --discover."""
import asyncio
import sys
import time
import types

import httpx
import pytest

import idle_detection
import worker_daemon
from idle_detection import ActivitySample, CpuBusySource, IdleReader, LowMemorySource

BASE = ["--id", "node-guard-unit", "--mock", "--allow-battery", "--ignore-thermal"]


def make_worker(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(worker_daemon, "TRIGGER_DIR", str(tmp_path))
    return worker_daemon.PreemptibleEdgeWorker(worker_daemon.parse_args([*BASE, *extra]))


def guard_types(worker):
    return [type(s) for s in worker.monitor.sources[1:]]


# ---------------------------------------------------------------- flags
def test_defaults_enable_cpu_and_ram_guards(monkeypatch, tmp_path):
    a = worker_daemon.parse_args(BASE)
    assert a.max_host_cpu_pct == 65.0 and a.min_free_ram_gb == 2.0
    assert a.coordinator is None and not a.discover and not a.allow_no_idle_sensor
    w = make_worker(monkeypatch, tmp_path)
    assert guard_types(w) == [CpuBusySource, LowMemorySource]
    assert w.coordinator_url == "http://127.0.0.1:8000"
    d = w.monitor.describe()
    assert "non-worker CPU > 65% x3" in d and "free RAM < 2.0 GB" in d


@pytest.mark.parametrize("value", ["0", "off", "OFF"])
def test_cpu_guard_can_be_disabled(monkeypatch, tmp_path, value):
    w = make_worker(monkeypatch, tmp_path, "--max-host-cpu-pct", value, "--min-free-ram-gb", "0")
    assert guard_types(w) == []


@pytest.mark.parametrize("bad", [["--max-host-cpu-pct", "150"], ["--max-host-cpu-pct", "-1"],
                                 ["--min-free-ram-gb", "-2"], ["--cpu-busy-percent", "0"]])
def test_bad_guard_values_rejected(bad):
    with pytest.raises(SystemExit):
        worker_daemon.parse_args([*BASE, *bad])


def test_deprecated_cpu_alias_maps_and_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(worker_daemon, "_warned_deprecated", set())
    assert worker_daemon.parse_args([*BASE, "--cpu-busy-percent", "50"]).max_host_cpu_pct == 50
    assert worker_daemon.parse_args([*BASE, "--cpu-busy-percent", "50"]).max_host_cpu_pct == 50
    assert capsys.readouterr().err.count("deprecated") == 1
    both = worker_daemon.parse_args([*BASE, "--cpu-busy-percent", "50", "--max-host-cpu-pct", "80"])
    assert both.max_host_cpu_pct == 80                     # canonical flag wins


# ---------------------------------------------------------------- fail-closed startup
def test_worker_exits_nonzero_when_no_idle_sensor(monkeypatch, tmp_path):
    monkeypatch.setattr(idle_detection, "native_idle_candidates", lambda: [IdleReader("xprintidle (X11)", lambda: None)])
    monkeypatch.setattr(worker_daemon, "TRIGGER_DIR", str(tmp_path))
    args = worker_daemon.parse_args([*BASE, "--native-idle-seconds", "30"])
    with pytest.raises(SystemExit) as e:
        asyncio.run(worker_daemon._amain(args))
    assert e.value.code not in (0, None)
    assert "FATAL" in str(e.value.code) and "--allow-no-idle-sensor" in str(e.value.code)


def test_worker_allow_no_idle_sensor_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(idle_detection, "native_idle_candidates", lambda: [])
    w = make_worker(monkeypatch, tmp_path, "--native-idle-seconds", "30", "--allow-no-idle-sensor")
    assert not w.monitor.native_enabled


# ---------------------------------------------------------------- low_memory eviction
class FakeOllama:
    """Async MockTransport for the Ollama client; /api/generate (the unload) can be made slow."""

    def __init__(self, unload_delay=0.0, ps_size=3 * 1024 ** 3):
        self.calls, self.unload_delay, self.ps_size = [], unload_delay, ps_size

    async def __call__(self, request):
        self.calls.append((time.monotonic(), request.url.path))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b", "size": self.ps_size}]})
        if request.url.path == "/api/generate":
            await asyncio.sleep(self.unload_delay)
        return httpx.Response(200, json={})


def wire(worker, ollama, coord_log):
    async def coord(request):
        coord_log.append((time.monotonic(), request.url.path))
        return httpx.Response(200, json={})
    worker.ollama = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(ollama))
    worker.coord = httpx.AsyncClient(base_url="http://coord.test", transport=httpx.MockTransport(coord))


def sample(*reasons):
    return ActivitySample(True, reasons[0], None, tuple(reasons))


def run_preempt(worker, s, settle=0.3):
    async def go():
        await worker.on_user_active(s, time.time())
        notified = time.monotonic()
        await asyncio.sleep(settle)
        return notified
    return asyncio.run(go())


def live_worker(monkeypatch, tmp_path, *extra):
    w = make_worker(monkeypatch, tmp_path, *extra)
    w.mock = False                                          # exercise the real unload path, fake transports
    return w


def test_low_memory_always_unloads_even_when_auto_says_no(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path)                  # this host has plenty of RAM: auto would skip
    ollama, coord = FakeOllama(), []
    wire(w, ollama, coord)
    run_preempt(w, sample("low_memory"))
    assert [p for _, p in ollama.calls] == ["/api/ps", "/api/generate"]
    assert w.model_bytes == 3 * 1024 ** 3                   # learned before evicting, feeds resume level
    assert w.model_reload_bytes() == 3 * 1024 ** 3


def test_low_memory_skips_unload_when_model_not_resident(monkeypatch, tmp_path, capsys):
    w = live_worker(monkeypatch, tmp_path)
    ollama = FakeOllama()

    async def nothing_loaded(request):
        ollama.calls.append((time.monotonic(), request.url.path))
        return httpx.Response(200, json={"models": [{"name": "other:7b", "size": 5}]})
    wire(w, nothing_loaded, [])
    run_preempt(w, sample("low_memory"))
    assert [p for _, p in ollama.calls] == ["/api/ps"]      # no generate: it could trigger a load
    assert "not resident" in capsys.readouterr().out


def test_low_memory_unloads_when_residency_unknown(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path)
    calls = []

    async def old_ollama(request):
        calls.append(request.url.path)
        return httpx.Response(404 if request.url.path == "/api/ps" else 200, json={})
    wire(w, old_ollama, [])
    run_preempt(w, sample("low_memory"))
    assert calls == ["/api/ps", "/api/generate"]


def test_other_reasons_respect_auto_policy(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path)
    ollama = FakeOllama()
    wire(w, ollama, [])
    run_preempt(w, sample("trigger_file"))
    assert ollama.calls == []


def test_unload_never_overrides_low_memory(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path, "--unload-on-preempt", "never")
    ollama = FakeOllama()
    wire(w, ollama, [])
    run_preempt(w, sample("low_memory"))
    assert ollama.calls == []


def test_unload_is_fire_and_forget(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path)
    ollama, coord = FakeOllama(unload_delay=1.0), []
    wire(w, ollama, coord)
    start = time.monotonic()
    returned = run_preempt(w, sample("low_memory"), settle=0.0)
    assert returned - start < 0.5                           # did not wait for the 1s unload
    assert coord and coord[0][1] == "/worker/preempt"


def test_low_memory_during_other_pause_unloads_once(monkeypatch, tmp_path):
    w = live_worker(monkeypatch, tmp_path)
    ollama = FakeOllama()
    wire(w, ollama, [])

    async def go():
        await w.on_user_active(sample("trigger_file"), time.time())
        await w.on_reasons_change(sample("trigger_file", "low_memory"))
        await w.on_reasons_change(sample("trigger_file", "cpu_busy", "low_memory"))
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert [p for _, p in ollama.calls].count("/api/generate") == 1


def test_model_reload_bytes_fallback_and_mock(monkeypatch, tmp_path):
    w = make_worker(monkeypatch, tmp_path)
    assert w.model_reload_bytes() == 0.0                    # mock: nothing to reload
    w.mock = False
    assert w.model_reload_bytes() == worker_daemon.FALLBACK_MODEL_BYTES


def test_mock_low_memory_logs_instead_of_unloading(monkeypatch, tmp_path, capsys):
    w = make_worker(monkeypatch, tmp_path)
    ollama = FakeOllama()
    wire(w, ollama, [])
    run_preempt(w, sample("low_memory"))
    assert ollama.calls == [] and "mock mode, no model to unload" in capsys.readouterr().out


# ---------------------------------------------------------------- --discover hook
class DiscoveryError(Exception):
    pass


def fake_discovery(monkeypatch, behaviour):
    mod = types.ModuleType("discovery")
    mod.DiscoveryError = DiscoveryError
    mod.calls = []

    def discover_coordinator(worker_token, timeout=3.0):
        mod.calls.append((worker_token, timeout))
        return behaviour()
    mod.discover_coordinator = discover_coordinator
    monkeypatch.setitem(sys.modules, "discovery", mod)
    return mod


def resolve(*extra, token=None):
    return asyncio.run(worker_daemon.resolve_coordinator(worker_daemon.parse_args([*BASE, *extra]), token))


def test_discover_returns_url_and_passes_token(monkeypatch):
    mod = fake_discovery(monkeypatch, lambda: "http://192.168.1.20:8000")
    assert resolve("--discover", token="tok") == "http://192.168.1.20:8000"
    assert mod.calls == [("tok", 3.0)]


def test_explicit_coordinator_skips_discovery(monkeypatch, capsys):
    mod = fake_discovery(monkeypatch, lambda: pytest.fail("must not discover"))
    assert resolve("--discover", "--coordinator", "http://c:1") == "http://c:1" and mod.calls == []
    assert "ignored" in capsys.readouterr().err
    assert resolve() == worker_daemon.DEFAULT_COORDINATOR and mod.calls == []


def test_discovery_error_is_fatal(monkeypatch):
    def fail():
        raise DiscoveryError("no beacon within 3s")
    fake_discovery(monkeypatch, fail)
    with pytest.raises(SystemExit) as e:
        resolve("--discover")
    assert "no beacon within 3s" in str(e.value.code) and "--coordinator" in str(e.value.code)


def test_missing_discovery_module_is_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "discovery", None)    # makes `import discovery` raise ImportError
    with pytest.raises(SystemExit, match="discovery"):
        resolve("--discover")


def test_discovery_bad_url_is_fatal(monkeypatch):
    fake_discovery(monkeypatch, lambda: "192.168.1.20:8000")
    with pytest.raises(SystemExit, match="invalid coordinator URL"):
        resolve("--discover")


def test_amain_uses_discovered_url(monkeypatch, tmp_path):
    fake_discovery(monkeypatch, lambda: "http://10.0.0.5:8000/")
    monkeypatch.setattr(worker_daemon, "TRIGGER_DIR", str(tmp_path))
    seen = {}

    async def fake_run(self):
        seen["url"] = self.coordinator_url
    monkeypatch.setattr(worker_daemon.PreemptibleEdgeWorker, "run", fake_run)
    asyncio.run(worker_daemon._amain(worker_daemon.parse_args([*BASE, "--discover"])))
    assert seen["url"] == "http://10.0.0.5:8000"
