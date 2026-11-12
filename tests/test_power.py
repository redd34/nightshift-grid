"""Good-citizen guard sources: battery, thermal, foreign CPU load, and the monitor that ORs them."""
import asyncio
import sys
from types import SimpleNamespace as NS

import pytest

from idle_detection import (
    ActivityMonitor, BatterySource, CpuBusySource, ForeignCpuSampler, LinuxThermalReader, ThermalSource,
    build_guard_sources, is_inference_process, macos_thermal_throttled, own_process_cpu_seconds,
    parse_pmset_therm,
)


def check(src):
    return asyncio.run(src.check())


def batt(plugged, percent=50.0):
    return NS(power_plugged=plugged, percent=percent, secsleft=None)


# ---------------------------------------------------------------- battery
def test_battery_pauses_unplugged_and_resumes_when_plugged():
    state = {"b": batt(False)}
    src = BatterySource(poll_s=0, reader=lambda: state["b"])
    assert check(src) == "on_battery"
    state["b"] = batt(True)
    assert check(src) is None


def test_battery_desktop_is_inactive_without_warning():
    logs = []
    src = BatterySource(poll_s=0, reader=lambda: None, log=logs.append)
    assert check(src) is None and logs == []
    assert src.describe() == "battery (none detected)"


def test_battery_unknown_ac_state_fails_open_and_warns_once():
    logs = []
    src = BatterySource(poll_s=0, reader=lambda: batt(None), log=logs.append)
    assert check(src) is None and check(src) is None
    assert len(logs) == 1 and "unknown" in logs[0]


def test_battery_low_floor_with_resume_margin():
    state = {"b": batt(True, 19)}
    src = BatterySource(min_percent=20, poll_s=0, reader=lambda: state["b"])
    assert check(src) == "battery_low"
    state["b"] = batt(True, 22)          # above the floor but inside the 5-point margin
    assert check(src) == "battery_low"
    state["b"] = batt(True, 25)
    assert check(src) is None
    state["b"] = batt(True, 22)          # re-arms at the plain floor, not the margin
    assert check(src) is None


def test_battery_low_off_by_default():
    assert check(BatterySource(poll_s=0, reader=lambda: batt(True, 3))) is None


def test_reader_exception_fails_open_loudly():
    logs = []

    def boom():
        raise RuntimeError("ioctl failed")
    src = BatterySource(poll_s=0, reader=boom, log=logs.append)
    assert check(src) is None and check(src) is None
    assert len(logs) == 1 and "ioctl failed" in logs[0]


def test_polled_source_caches_between_polls():
    calls = []
    src = BatterySource(poll_s=60, reader=lambda: calls.append(1) or batt(False))
    assert check(src) == "on_battery" and check(src) == "on_battery"
    assert len(calls) == 1


# ---------------------------------------------------------------- thermal
PMSET_NOMINAL = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
"""
PMSET_THROTTLED = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
2024-05-01 10:00:00 +0000 CPU Power notify
\tCPU_Scheduler_Limit \t= 100
\tCPU_Available_CPUs \t= 8
\tCPU_Speed_Limit \t= 63
"""
PMSET_FULL_SPEED = PMSET_THROTTLED.replace("= 63", "= 100")


@pytest.mark.parametrize("out,expected", [
    (PMSET_NOMINAL, False),
    (PMSET_THROTTLED, True),
    (PMSET_FULL_SPEED, False),
    ("Thermal warning level set to 2.\n", True),
    ("Thermal warning level set to 0.\n", False),
    ("", None),
    ("something unexpected\n", None),
])
def test_parse_pmset_therm(out, expected):
    assert parse_pmset_therm(out) is expected


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_thermal_live():
    assert macos_thermal_throttled() in (True, False)


def temps(*values):
    return lambda: {"coretemp": [NS(label="", current=v, high=None, critical=None) for v in values]}


def test_linux_thermal_threshold_and_hysteresis():
    state = {"t": 80.0}
    reader = LinuxThermalReader(90, sensors=lambda: temps(50, state["t"])())
    assert reader() is False
    state["t"] = 91
    assert reader() is True and reader.last_temp_c == 91
    state["t"] = 87                      # cooled, but not by the 5C hysteresis
    assert reader() is True
    state["t"] = 84
    assert reader() is False


def test_linux_thermal_no_sensors_is_unknown():
    assert LinuxThermalReader(90, sensors=lambda: {})() is None


def test_thermal_source_reasons_and_unknown_warning():
    state = {"v": True}
    logs = []
    src = ThermalSource(lambda: state["v"], poll_s=0, log=logs.append)
    assert check(src) == "thermal_throttle"
    state["v"] = None
    assert check(src) is None and len(logs) == 1


# ---------------------------------------------------------------- foreign CPU
def make_sampler(system, own):
    """system: list of (busy, total) CPU-seconds; own: list of {pid: cpu_seconds}. Index 0 primes."""
    it_sys, it_own = iter(system), iter(own)
    return ForeignCpuSampler(system_times=lambda: next(it_sys), own_times=lambda: next(it_own))


def test_foreign_cpu_subtracts_our_processes():
    # 4 cores x 1s = 4 CPU-s per interval. System busy +3s, of which ollama (pid 7) used 2s.
    s = make_sampler([(0, 0), (3, 4)], [{7: 10.0}, {7: 12.0}])
    assert s() == pytest.approx(25.0)


def test_foreign_cpu_vanished_and_new_processes_count_as_foreign_once():
    s = make_sampler([(0, 0), (4, 4), (6, 8)],
                     [{7: 1.0}, {8: 5.0}, {8: 7.0}])     # pid 7 vanished, pid 8 appeared
    assert s() == pytest.approx(100.0)                   # no baseline for 8: whole delta foreign
    assert s() == pytest.approx(0.0)                     # now 8 is primed and fully subtracted


def test_foreign_cpu_clamped_and_zero_interval_repeats_last():
    s = make_sampler([(0, 0), (1, 4), (1, 4)], [{}, {1: 9.0}, {1: 9.0}])
    assert s() == pytest.approx(25.0)
    assert s() == pytest.approx(25.0)                    # no time elapsed: repeat, don't divide by 0
    s2 = make_sampler([(0, 0), (1, 4)], [{1: 0.0}, {1: 3.0}])   # own > system (clock skew)
    assert s2() == 0.0


def test_inference_process_names():
    assert all(map(is_inference_process, ["ollama", "Ollama", "ollama_llama_server", "llama-server"]))
    assert not any(map(is_inference_process, ["python", "Chrome", "llamafile-ish"]))


def test_own_process_scan_includes_self():
    import os
    assert os.getpid() in own_process_cpu_seconds()


def cpu_source(values, **kw):
    it = iter(values)
    return CpuBusySource(80, sustain=3, poll_s=0, sampler=lambda: next(it), **kw)


def run_all(src, n):
    return [check(src) for _ in range(n)]


def test_cpu_busy_requires_sustained_load():
    src = cpu_source([95, 95, 10, 95, 95, 95])
    assert run_all(src, 6) == [None, None, None, None, None, "cpu_busy"]


def test_cpu_busy_hysteresis_on_resume():
    # Pause after 3 hot samples; 70% is below 80 but not below 80*0.7=56, so still paused.
    src = cpu_source([90, 90, 90, 70, 70, 70, 50, 50, 60, 50, 50, 50])
    got = run_all(src, 12)
    assert got[:2] == [None, None]
    assert got[2:11] == ["cpu_busy"] * 9                 # the 60 reading resets the calm streak
    assert got[11] is None


def test_cpu_busy_unknown_resets_and_warns():
    logs = []
    src = cpu_source([90, 90, 90, None, 90], log=logs.append)
    assert run_all(src, 5) == [None, None, "cpu_busy", None, None]
    assert len(logs) == 1


# ---------------------------------------------------------------- monitor integration
def test_monitor_priority_and_reason_change(tmp_path):
    trigger = tmp_path / "preempt"
    on_batt = {"v": True}
    logs = []
    mon = ActivityMonitor(str(trigger), file_poll_s=0.01, log=logs.append,
                          guards=[BatterySource(poll_s=0, reader=lambda: batt(not on_batt["v"]))])
    seen = []

    async def scenario():
        async def on_active(s, ts): seen.append(("active", s.reason))
        async def on_idle(): seen.append(("idle", None))
        trigger.touch()
        task = asyncio.create_task(mon.run(on_active, on_idle))
        await asyncio.sleep(0.05)
        assert mon.reason == "trigger_file"              # trigger outranks battery
        trigger.unlink()
        await asyncio.sleep(0.05)
        assert mon.active and mon.reason == "on_battery"
        on_batt["v"] = False
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(scenario())
    assert seen == [("active", "trigger_file"), ("idle", None)]
    assert any("trigger_file -> on_battery" in m for m in logs)


def test_build_guard_sources_flags():
    names = [s.name for s in build_guard_sources(max_host_cpu_pct=80, log=lambda m: None)]
    assert "battery" in names and "cpu_busy" in names and "memory" in names   # CPU + RAM on by default
    assert ("thermal" in names) == (sys.platform == "darwin" or sys.platform.startswith("linux"))
    assert build_guard_sources(allow_battery=True, ignore_thermal=True, max_host_cpu_pct=0, min_free_ram_gb=0) == []
    assert [s.name for s in build_guard_sources(allow_battery=True, ignore_thermal=True)] == ["cpu_busy", "memory"]


def test_monitor_describe_lists_guards(tmp_path):
    mon = ActivityMonitor(str(tmp_path / "x"), native_idle_threshold_s=60, native_reader=lambda: 1.0,
                          guards=[CpuBusySource(80, sampler=lambda: 0.0)])
    d = mon.describe()
    assert "HID idle < 60s" in d and "non-worker CPU > 80%" in d
    assert ActivityMonitor(str(tmp_path / "x")).describe() == "none"
