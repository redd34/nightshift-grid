"""Host CPU and RAM pressure guards: peer-worker exclusion, startup priming, memory hysteresis/anti-flap."""
import asyncio
import os
import subprocess
import sys
import time

import pytest

from idle_detection import (
    GB, REASONS, ActivityMonitor, CpuBusySource, ForeignCpuSampler, GridProcessScanner, LowMemorySource,
    is_peer_worker_cmdline,
)


def check(src):
    return asyncio.run(src.check())


# ---------------------------------------------------------------- CPU: who counts as "ours"
@pytest.mark.parametrize("cmdline,expected", [
    (["/usr/bin/python3", "worker_daemon.py", "--id", "n1"], True),
    (["python", "/opt/nightshift/worker_daemon.py"], True),
    (["python", "-m", "worker_daemon", "--id", "n1"], True),
    (["python", "coordinator.py"], False),
    (["python", "-c", "print('worker_daemon.py')"], False),   # mentioned in code, not the script being run
    (None, False),
    ([], False),
])
def test_peer_worker_cmdline(cmdline, expected):
    assert is_peer_worker_cmdline(cmdline) is expected


def test_scanner_excludes_self_and_peer_workers_but_not_other_python(tmp_path):
    peer_script = tmp_path / "worker_daemon.py"
    other_script = tmp_path / "build_job.py"
    for f in (peer_script, other_script):
        f.write_text("import time\ntime.sleep(30)\n")
    procs = [subprocess.Popen([sys.executable, str(f)]) for f in (peer_script, other_script)]
    try:
        time.sleep(0.3)
        scanner = GridProcessScanner()
        got = scanner()
        assert os.getpid() in got
        assert procs[0].pid in got and procs[1].pid not in got
        assert procs[1].pid not in GridProcessScanner(exclude_peers=True)()
        assert procs[0].pid not in GridProcessScanner(exclude_peers=False)()
        scanner()                                            # cached verdicts survive a second scan
        assert any(k[0] == procs[0].pid for k in scanner._peer_cache)
    finally:
        for p in procs:
            p.kill()
            p.wait()


# ---------------------------------------------------------------- CPU: priming / startup
class PrimeableSampler:
    def __init__(self, values):
        self.values, self.primes = iter(values), 0

    def prime(self):
        self.primes += 1

    def __call__(self):
        return next(self.values)


def test_cpu_first_poll_only_primes_so_startup_cannot_trigger():
    sampler = PrimeableSampler([100.0, 0.0])
    src = CpuBusySource(65, sustain=1, poll_s=0, sampler=sampler)
    assert check(src) is None and sampler.primes == 1      # priming poll decides nothing
    assert check(src) == "cpu_busy"                        # first real sample may (sustain=1)
    assert check(src) is None                              # resumes below 0.7*65


def test_foreign_sampler_prime_rebaselines():
    system = iter([(0, 0), (100, 100), (101, 104)])        # construct, re-prime after a busy startup, sample
    s = ForeignCpuSampler(system_times=lambda: next(system), own_times=lambda: {})
    s.prime()
    assert s() == pytest.approx(25.0)                      # not the 100% of the startup window


def test_cpu_busy_reason_propagates_through_monitor(tmp_path):
    src = CpuBusySource(65, sustain=2, poll_s=0, sampler=PrimeableSampler([90.0] * 50))
    mon = ActivityMonitor(str(tmp_path / "t"), guards=[src], file_poll_s=0.005)
    seen = []

    async def scenario():
        async def on_active(s, ts): seen.append(s.reason)
        async def on_idle(): seen.append("idle")
        task = asyncio.create_task(mon.run(on_active, on_idle))
        await asyncio.sleep(0.1)
        task.cancel()

    asyncio.run(scenario())
    assert seen == ["cpu_busy"]


# ---------------------------------------------------------------- RAM guard
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def mem_source(avail, clock, **kw):
    kw.setdefault("sustain", 3)
    return LowMemorySource(2.0, poll_s=0, reader=lambda: avail["v"] * GB, clock=clock, log=lambda m: None, **kw)


def test_low_memory_needs_sustained_pressure():
    avail, clock = {"v": 1.0}, Clock()
    src = mem_source(avail, clock)
    assert [check(src) for _ in range(3)] == [None, None, "low_memory"]
    src2 = mem_source({"v": 1.0}, clock)
    seq = iter([1.0, 1.0, 3.0, 1.0, 1.0])
    src2.reader = lambda: next(seq) * GB
    assert [check(src2) for _ in range(5)] == [None] * 5   # one healthy sample resets the streak


def test_low_memory_resume_requires_room_to_reload_model_plus_dwell():
    avail, clock = {"v": 1.0}, Clock()
    src = mem_source(avail, clock, sustain=1, model_bytes=lambda: 3 * GB, margin_gb=1.0, min_pause_s=30)
    assert check(src) == "low_memory"
    assert src.resume_level() == pytest.approx(6 * GB)     # floor 2 + model 3 + margin 1
    avail["v"] = 4.5                                       # what unloading the 3 GB model gives back
    clock.t = 100
    assert check(src) == "low_memory"                      # reload would land at 1.5 GB: stay paused
    avail["v"] = 6.5
    clock.t = 10
    src._paused_at = 0.0
    assert check(src) == "low_memory"                      # enough RAM, but inside the 30s dwell
    clock.t = 31
    assert check(src) is None


def test_model_estimate_is_read_live():
    size = {"b": 0.0}
    src = LowMemorySource(2.0, model_bytes=lambda: size["b"], margin_gb=1.0)
    assert src.resume_level() == pytest.approx(3 * GB)
    size["b"] = 5 * GB
    assert src.resume_level() == pytest.approx(8 * GB)


def simulate(src, clock, base_gb, model_gb, ticks=2000, tick_s=1.0):
    """Closed loop: the worker holds the model while harvesting and unloads it while paused."""
    transitions, paused = 0, False
    for _ in range(ticks):
        avail = base_gb - (0 if paused else model_gb)
        src.reader = lambda a=avail: a * GB
        now_paused = check(src) == "low_memory"
        transitions += now_paused != paused
        paused = now_paused
        clock.t += tick_s
    return transitions


def test_no_flapping_when_model_estimate_is_right():
    # 4.5 GB free with the model unloaded, 1.5 GB with it loaded (3 GB model), floor 2 GB.
    clock = Clock()
    src = LowMemorySource(2.0, poll_s=0, model_bytes=lambda: 3 * GB, clock=clock, log=lambda m: None)
    assert simulate(src, clock, base_gb=4.5, model_gb=3.0) == 1   # pauses once and stays paused


def test_naive_resume_would_flap_and_backoff_bounds_it():
    clock = Clock()
    naive = LowMemorySource(2.0, poll_s=0, margin_gb=0.0, min_pause_s=0, max_pause_s=0, clock=clock,
                            log=lambda m: None)
    flaps_naive = simulate(naive, clock, base_gb=4.5, model_gb=3.0)
    assert flaps_naive > 300                                # the failure mode this guard exists to stop
    # Same wrong (zero) model estimate, default dwell + exponential backoff: oscillation slows down.
    clock = Clock()
    wrong = LowMemorySource(2.0, poll_s=0, model_bytes=lambda: 0.0, clock=clock, log=lambda m: None)
    flaps = simulate(wrong, clock, base_gb=4.5, model_gb=3.0)
    assert flaps <= 16 and wrong.dwell_s == 600.0           # dwell 30,60,...,600s over ~33 min


def test_backoff_resets_after_a_stable_period():
    avail, clock = {"v": 1.0}, Clock()
    src = mem_source(avail, clock, sustain=1, min_pause_s=30, flap_window_s=120)
    check(src)
    avail["v"] = 10.0
    clock.t = 31
    assert check(src) is None
    avail["v"] = 1.0
    clock.t = 40                                           # re-paused 9s after resuming: flapping
    check(src)
    assert src.dwell_s == 60
    avail["v"] = 10.0
    clock.t = 101
    check(src)
    avail["v"] = 1.0
    clock.t = 1000                                         # long calm spell: back to the minimum
    check(src)
    assert src.dwell_s == 30


def test_low_memory_unknown_reading_fails_open():
    logs = []
    vals = iter([1.0, None, 1.0])
    src = LowMemorySource(2.0, sustain=1, poll_s=0, reader=lambda: (lambda v: v and v * GB)(next(vals)),
                          log=logs.append)
    assert check(src) == "low_memory"
    assert check(src) is None and any("unreadable" in m for m in logs)


def test_monitor_reports_all_reasons_and_reason_change_callback(tmp_path):
    trigger = tmp_path / "t"
    avail = {"v": 10.0}
    mem = LowMemorySource(2.0, sustain=1, poll_s=0, reader=lambda: avail["v"] * GB, log=lambda m: None)
    mon = ActivityMonitor(str(trigger), guards=[mem], file_poll_s=0.005, log=lambda m: None)
    events = []

    async def scenario():
        async def on_active(s, ts): events.append(("active", s.reason, s.reasons))
        async def on_idle(): events.append(("idle",))
        async def on_change(s): events.append(("change", s.reason, s.reasons))
        trigger.touch()
        task = asyncio.create_task(mon.run(on_active, on_idle, on_change))
        await asyncio.sleep(0.03)
        avail["v"] = 1.0                                   # memory pressure under a trigger-file pause
        await asyncio.sleep(0.03)
        task.cancel()

    asyncio.run(scenario())
    assert events[0] == ("active", "trigger_file", ("trigger_file",))
    assert events[1] == ("change", "trigger_file", ("trigger_file", "low_memory"))


def test_reason_catalogue():
    assert set(REASONS) == {"trigger_file", "hid_input", "idle_sensor_lost", "on_battery", "battery_low",
                            "thermal_throttle", "cpu_busy", "low_memory"}
