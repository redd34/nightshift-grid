"""Fail-closed HID idle detection: reader parsing, candidate selection, startup probe, runtime loss."""
import asyncio

import pytest

import idle_detection as idle
from idle_detection import (
    ActivityMonitor, HIDIdleSource, IdleReader, IdleSensorUnavailable, native_idle_candidates, parse_dbus_uint,
    probe_idle_reader,
)


def check(src):
    return asyncio.run(src.check())


def seq_reader(values):
    """Reader yielding `values` in order; an Exception instance is raised instead of returned."""
    it = iter(values)

    def read():
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v
    return read


# ---------------------------------------------------------------- D-Bus parsing / readers
@pytest.mark.parametrize("out,expected", [
    ("(uint64 12345,)\n", 12345),                                         # gdbus (Mutter)
    ("(uint32 7,)\n", 7),                                                 # gdbus (ScreenSaver)
    ("method return time=1700000000.12 sender=:1.2 -> destination=:1.9 serial=55 reply_serial=2\n"
     "   uint32 4200\n", 4200),                                           # dbus-send: header numbers ignored
    ("4200\n", 4200),                                                     # qdbus
    ("Error: GDBus.Error:org.freedesktop.DBus.Error.ServiceUnknown\n", None),
    ("", None),
])
def test_parse_dbus_uint(out, expected):
    assert parse_dbus_uint(out) == expected


def test_dbus_reader_converts_ms_and_handles_failure(monkeypatch):
    outputs = {"ok": "(uint64 2500,)", "bad": None}
    monkeypatch.setattr(idle, "_run", lambda argv, timeout=2.0: outputs[argv[0]])
    assert idle.dbus_ms_reader(["ok"])() == pytest.approx(2.5)
    assert idle.dbus_ms_reader(["bad"])() is None


def names(readers):
    return [r.name for r in readers]


def which_of(*present):
    return lambda b: f"/usr/bin/{b}" if b in present else None


def test_candidates_per_platform():
    assert names(native_idle_candidates("darwin")) == ["ioreg HIDIdleTime"]
    assert names(native_idle_candidates("win32")) == ["GetLastInputInfo"]
    assert native_idle_candidates("freebsd13", env={}, which=which_of()) == []
    assert native_idle_candidates("linux", env={}, which=which_of()) == []


def test_linux_wayland_prefers_dbus_over_xwayland():
    all_bins = which_of("gdbus", "dbus-send", "qdbus6", "xprintidle")
    wayland = names(native_idle_candidates("linux", env={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"},
                                           which=all_bins))
    assert wayland[0] == "gnome mutter IdleMonitor (gdbus)" and wayland[-1] == "xprintidle (X11)"
    assert "freedesktop ScreenSaver (dbus-send)" in wayland and "freedesktop ScreenSaver (qdbus6)" in wayland
    x11 = names(native_idle_candidates("linux", env={"DISPLAY": ":0"}, which=all_bins))
    assert x11[0] == "xprintidle (X11)"


def test_xprintidle_needs_display():
    got = names(native_idle_candidates("linux", env={}, which=which_of("xprintidle", "gdbus")))
    assert "xprintidle (X11)" not in got and got


def test_probe_skips_broken_readers():
    def boom():
        raise OSError("no bus")
    a, b, c = IdleReader("a", lambda: None), IdleReader("b", boom), IdleReader("c", lambda: 12.0)
    assert probe_idle_reader([a, b, c]) is c
    assert probe_idle_reader([a, b, IdleReader("neg", lambda: -1.0)]) is None


# ---------------------------------------------------------------- startup: fail closed
def test_monitor_refuses_to_start_without_working_sensor(tmp_path):
    cands = [IdleReader("mutter", lambda: None), IdleReader("screensaver", lambda: None)]
    with pytest.raises(IdleSensorUnavailable) as e:
        ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30, idle_candidates=cands)
    msg = str(e.value)
    assert "tried: mutter, screensaver" in msg and "--allow-no-idle-sensor" in msg


def test_monitor_allow_no_idle_sensor_warns_and_continues(tmp_path):
    logs = []
    mon = ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30, idle_candidates=[],
                          allow_no_idle_sensor=True, log=logs.append)
    assert not mon.native_enabled and any("WITHOUT HID" in m for m in logs)


def test_monitor_uses_first_working_candidate(tmp_path):
    mon = ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30, idle_candidates=[
        IdleReader("dead", lambda: None), IdleReader("kde", lambda: 99.0)])
    assert mon.hid is not None and mon.hid.reader.name == "kde"
    assert "HID idle < 30s (kde, fail-closed)" in mon.describe()


def test_no_threshold_means_no_probe(tmp_path):
    def never():
        raise AssertionError("must not probe")
    assert not ActivityMonitor(str(tmp_path / "t"), idle_candidates=[IdleReader("x", never)]).native_enabled


def test_monitor_uses_platform_candidates_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(idle, "native_idle_candidates", lambda: [IdleReader("fake", lambda: None)])
    with pytest.raises(IdleSensorUnavailable, match="tried: fake"):
        ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30)


# ---------------------------------------------------------------- runtime: sensor lost
def test_runtime_sensor_loss_pauses_after_n_failures_and_recovers():
    logs = []
    src = HIDIdleSource(30, seq_reader([100.0, None, RuntimeError("ioreg hung"), None, None, 100.0]),
                        poll_s=0, max_failures=3, log=logs.append)
    got = [check(src) for _ in range(6)]
    assert got == [None, None, None, "idle_sensor_lost", "idle_sensor_lost", None]
    assert sum("failed 3 reads in a row" in m for m in logs) == 1     # warned once, not per read
    assert any("recovered" in m for m in logs)


def test_transient_failure_holds_previous_answer():
    # User typing (idle 2s < 30s) then a blip: must stay paused, not resume on "unknown".
    src = HIDIdleSource(30, seq_reader([2.0, None, None, 2.0]), poll_s=0, max_failures=3, log=lambda m: None)
    assert [check(src) for _ in range(4)] == ["hid_input", "hid_input", "hid_input", "hid_input"]


def test_fail_open_mode_never_reports_sensor_lost():
    logs = []
    src = HIDIdleSource(30, lambda: None, poll_s=0, fail_closed=False, log=logs.append)
    assert [check(src) for _ in range(6)] == [None] * 6 and len(logs) == 1
    assert "fail-open" in src.describe()


def test_sensor_lost_reason_reaches_on_active(tmp_path):
    state = {"v": 100.0}
    mon = ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30, native_reader=lambda: state["v"],
                          native_poll_s=0, file_poll_s=0.005, log=lambda m: None)
    seen = []

    async def scenario():
        async def on_active(s, ts): seen.append(s.reason)
        async def on_idle(): seen.append("idle")
        task = asyncio.create_task(mon.run(on_active, on_idle))
        await asyncio.sleep(0.03)
        state["v"] = None
        await asyncio.sleep(0.1)
        state["v"] = 100.0
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(scenario())
    assert seen == ["idle_sensor_lost", "idle"]


def test_allow_no_idle_sensor_makes_runtime_loss_fail_open(tmp_path):
    state = {"v": 100.0}
    mon = ActivityMonitor(str(tmp_path / "t"), native_idle_threshold_s=30, native_reader=lambda: state["v"],
                          native_poll_s=0, allow_no_idle_sensor=True, log=lambda m: None)
    state["v"] = None
    assert all(not asyncio.run(mon.sample()).active for _ in range(5))
