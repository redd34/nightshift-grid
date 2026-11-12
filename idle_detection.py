"""Good-citizen activity detection: decides when this machine must NOT be harvested.

The `ActivityMonitor` ORs a set of pluggable *sources*. Each source answers one question -
"is there a reason to pause right now?" - and returns a reason string (sent to the
coordinator verbatim via `/worker/preempt`) or None. Sources, in priority order:

1. `TriggerFileSource`  "trigger_file"      `/tmp/preempt_<worker_id>` exists (chaos tests/demos).
2. `HIDIdleSource`      "hid_input"         OS HID idle time < N s (opt-in `--native-idle-seconds`).
                        "idle_sensor_lost"  the HID reader that worked at startup keeps failing.
     * macOS:   IOKit `HIDIdleTime` (nanoseconds) via `ioreg -c IOHIDSystem`
     * Windows: user32 `GetLastInputInfo`
     * Linux:   Wayland/GNOME `org.gnome.Mutter.IdleMonitor.GetIdletime` (gdbus), KDE/freedesktop
                `org.freedesktop.ScreenSaver.GetSessionIdleTime` (gdbus / dbus-send / qdbus),
                X11 `xprintidle`. The first one that answers at startup is used.
3. `BatterySource`      "on_battery"        unplugged laptop: never drain someone's battery.
                        "battery_low"       plugged in but below `--min-battery-percent`.
4. `ThermalSource`      "thermal_throttle"  macOS `pmset -g therm` speed limit / warning level,
                                            Linux hottest sensor >= `--max-temp-c`.
5. `CpuBusySource`      "cpu_busy"          *non-worker* CPU sustained above `--max-host-cpu-pct`
                                            (default 65% of total host capacity).
6. `LowMemorySource`    "low_memory"        available RAM sustained below `--min-free-ram-gb`
                                            (default 2 GB); resume needs room to reload the model.

Failure policy - deliberately NOT uniform, because the guards protect different things:

* **HID idle: fail closed** (when `--native-idle-seconds` is requested). It is the only guard
  standing between the grid and a human at the keyboard, so "cannot tell whether someone is
  typing" must not mean "assume nobody is". At startup every candidate reader is probed; if none
  answers, `ActivityMonitor` raises `IdleSensorUnavailable` and the worker exits non-zero with
  an actionable message. At runtime, `SENSOR_LOSS_FAILURES` (3) consecutive failed reads pause
  the node with "idle_sensor_lost" until the sensor answers again; fewer failures hold the last
  answer, so one slow `ioreg` spawn neither pauses nor resumes anything. `--allow-no-idle-sensor`
  restores the old fail-open-loudly behaviour for headless hosts where nobody types.
* **Battery, thermal, CPU, memory: fail open, loudly.** If a sensor cannot be read the source
  reports "no reason to pause" and logs a warning once. These protect the machine's comfort and
  battery rather than a person's interactive session; a broken thermal sensor on one node must
  not silently remove that node from the grid forever. The trigger file and lease TTL still
  guarantee correctness, and the warning tells the operator the guard is not protecting them.
  "No battery present" (a desktop) is a definite answer, not an unknown, so the battery guard is
  simply inactive without a warning.

Every source owns its poll interval and caches its last answer, so the monitor can tick every
100ms for the cheap trigger file while `pmset`/`ioreg`/process scans run at their own pace,
off the event loop via `asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import psutil

Log = Callable[[str], None]
GB = 1024 ** 3

# Every reason string a worker can send to `/worker/preempt` (plus "shutdown" from the worker itself).
REASONS = ("trigger_file", "hid_input", "idle_sensor_lost", "on_battery", "battery_low",
           "thermal_throttle", "cpu_busy", "low_memory")

DEFAULT_MAX_HOST_CPU_PCT = 65.0
DEFAULT_MIN_FREE_RAM_GB = 2.0
SENSOR_LOSS_FAILURES = 3


class IdleSensorUnavailable(RuntimeError):
    """HID idle detection was requested but no reader works on this host (fail-closed startup)."""


# ============================================================================ HID idle readers
class IdleReader:
    """A named HID idle reader: `reader()` -> seconds since last input, or None if unreadable.

    The name exists for the startup log and for the actionable error that lists what was tried.
    """

    def __init__(self, name: str, fn: Callable[[], Optional[float]]):
        self.name = name
        self.fn = fn

    def __call__(self) -> Optional[float]:
        return self.fn()

    def __repr__(self) -> str:
        return f"IdleReader({self.name})"


def _run(argv: Sequence[str], timeout: float = 2.0) -> Optional[str]:
    """stdout of a successful command, else None (missing binary, non-zero exit, timeout)."""
    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


_HID_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')


def parse_ioreg_hid_idle(output: str) -> Optional[float]:
    """Return seconds since last HID event from `ioreg -c IOHIDSystem` output."""
    m = _HID_RE.search(output)
    return int(m.group(1)) / 1e9 if m else None


def macos_idle_seconds() -> Optional[float]:
    out = _run(["ioreg", "-c", "IOHIDSystem", "-r", "-d", "1", "-k", "HIDIdleTime"])
    return parse_ioreg_hid_idle(out) if out is not None else None


def windows_idle_seconds() -> Optional[float]:
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
            return None
        # Both values are in ms; GetTickCount wraps at 2^32 like dwTime, so mask the delta.
        now = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
        return ((now - info.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:
        return None


def xprintidle_seconds() -> Optional[float]:
    if not os.environ.get("DISPLAY"):
        return None
    out = _run(["xprintidle"])
    try:
        return int(out.strip()) / 1000.0 if out is not None else None
    except ValueError:
        return None


linux_idle_seconds = xprintidle_seconds  # backwards-compatible name


# gdbus prints "(uint64 1234,)", dbus-send "   uint32 1234" after a header full of other numbers
# (time=, serial=), qdbus a bare "1234". Prefer the typed form so header numbers never match.
_DBUS_TYPED_UINT_RE = re.compile(r"\buint(?:16|32|64)\s+(\d+)")


def parse_dbus_uint(output: str) -> Optional[int]:
    """The unsigned integer a D-Bus method returned, from gdbus / dbus-send / qdbus output."""
    m = _DBUS_TYPED_UINT_RE.search(output)
    if m:
        return int(m.group(1))
    bare = output.strip()
    return int(bare) if bare.isdigit() else None


def dbus_ms_reader(argv: Sequence[str]) -> Callable[[], Optional[float]]:
    """Reader for a D-Bus idle method that returns milliseconds.

    Mutter's `GetIdletime` is documented as ms. freedesktop `GetSessionIdleTime` is ambiguous in
    the wild (the spec text says seconds; KDE, its main implementer on Wayland, returns ms). We
    read it as ms because that errs in the safe direction: a seconds value misread as ms looks
    1000x *less* idle (worker pauses too eagerly), whereas ms misread as seconds would look 1000x
    more idle and harvest over a busy human.
    """
    def read() -> Optional[float]:
        out = _run(argv)
        v = parse_dbus_uint(out) if out is not None else None
        return v / 1000.0 if v is not None else None
    return read


MUTTER_IDLE_ARGV = ("gdbus", "call", "--session", "--dest", "org.gnome.Mutter.IdleMonitor",
                    "--object-path", "/org/gnome/Mutter/IdleMonitor/Core",
                    "--method", "org.gnome.Mutter.IdleMonitor.GetIdletime")
SCREENSAVER_GDBUS_ARGV = ("gdbus", "call", "--session", "--dest", "org.freedesktop.ScreenSaver",
                          "--object-path", "/org/freedesktop/ScreenSaver",
                          "--method", "org.freedesktop.ScreenSaver.GetSessionIdleTime")
SCREENSAVER_DBUS_SEND_ARGV = ("dbus-send", "--session", "--print-reply", "--dest=org.freedesktop.ScreenSaver",
                              "/org/freedesktop/ScreenSaver", "org.freedesktop.ScreenSaver.GetSessionIdleTime")
QDBUS_BINARIES = ("qdbus", "qdbus6", "qdbus-qt5")


def native_idle_candidates(platform: str = sys.platform, env: Optional[Mapping[str, str]] = None,
                           which: Callable[[str], Optional[str]] = shutil.which) -> list[IdleReader]:
    """Every HID idle reader worth probing on this host, best first. Nothing is executed here.

    On Linux a Wayland session puts the D-Bus readers ahead of `xprintidle`: under XWayland,
    `xprintidle` only sees input delivered to X clients, so a user typing into a native Wayland
    window would look idle. Binaries that are not installed are skipped up front.
    """
    env = os.environ if env is None else env
    if platform == "darwin":
        return [IdleReader("ioreg HIDIdleTime", macos_idle_seconds)]
    if platform.startswith("win"):
        return [IdleReader("GetLastInputInfo", windows_idle_seconds)]
    if not platform.startswith("linux"):
        return []
    dbus: list[IdleReader] = []
    if which("gdbus"):
        dbus.append(IdleReader("gnome mutter IdleMonitor (gdbus)", dbus_ms_reader(MUTTER_IDLE_ARGV)))
        dbus.append(IdleReader("freedesktop ScreenSaver (gdbus)", dbus_ms_reader(SCREENSAVER_GDBUS_ARGV)))
    if which("dbus-send"):
        dbus.append(IdleReader("freedesktop ScreenSaver (dbus-send)", dbus_ms_reader(SCREENSAVER_DBUS_SEND_ARGV)))
    for qdbus in QDBUS_BINARIES:
        if which(qdbus):
            dbus.append(IdleReader(f"freedesktop ScreenSaver ({qdbus})", dbus_ms_reader(
                (qdbus, "org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver",
                 "org.freedesktop.ScreenSaver.GetSessionIdleTime"))))
            break
    x11 = [IdleReader("xprintidle (X11)", xprintidle_seconds)] if which("xprintidle") and env.get("DISPLAY") else []
    return dbus + x11 if env.get("WAYLAND_DISPLAY") else x11 + dbus


def probe_idle_reader(candidates: Iterable[IdleReader]) -> Optional[IdleReader]:
    """First candidate that returns a real reading right now. "Installed" is not "working": the
    D-Bus service may be absent, or the worker may run outside the user's session."""
    for c in candidates:
        try:
            v = c()
        except Exception:
            continue
        if v is not None and v >= 0:
            return c
    return None


def idle_sensor_help(platform: str = sys.platform) -> str:
    """How to get a working HID idle reader on `platform` (appended to the fail-closed error)."""
    if platform == "darwin":
        return "`ioreg -c IOHIDSystem` returned no HIDIdleTime; run the worker inside the logged-in user's session."
    if platform.startswith("win"):
        return ("GetLastInputInfo only sees the interactive desktop; run the worker in the user's session "
                "(a scheduled task at logon), not as a session-0 service.")
    if platform.startswith("linux"):
        return ("Run the worker as the desktop user with DBUS_SESSION_BUS_ADDRESS set (GNOME/Wayland: Mutter "
                "IdleMonitor via gdbus; KDE: org.freedesktop.ScreenSaver via gdbus/dbus-send/qdbus), or on X11 "
                "install `xprintidle` and export DISPLAY.")
    return f"no HID idle reader is implemented for {platform}."


# ============================================================================ thermal readers
_SPEED_LIMIT_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
# e.g. "Thermal warning level set to 1." The "No thermal warning level has been recorded" note
# has no digit on its line, so `[^\d\n]*` cannot match across into another line's number.
_THERMAL_LEVEL_RE = re.compile(r"thermal warning level[^\d\n]*(\d+)", re.IGNORECASE)
_NO_THERMAL_RE = re.compile(r"no thermal warning level has been recorded", re.IGNORECASE)


def parse_pmset_therm(output: str) -> Optional[bool]:
    """True if `pmset -g therm` reports throttling, False if it reports none, None if unparseable.

    Two signals: `CPU_Speed_Limit` below 100 (the kernel is capping clocks) or a non-zero
    recorded thermal warning level. Unrecognised output is None rather than False so a future
    macOS format change surfaces as a warning instead of a silently disabled guard.
    """
    limits = [int(v) for v in _SPEED_LIMIT_RE.findall(output)]
    levels = [int(v) for v in _THERMAL_LEVEL_RE.findall(output)]
    if any(v < 100 for v in limits) or any(v > 0 for v in levels):
        return True
    if limits or levels or _NO_THERMAL_RE.search(output):
        return False
    return None


def macos_thermal_throttled() -> Optional[bool]:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3.0).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_pmset_therm(out)


class LinuxThermalReader:
    """Hottest `psutil.sensors_temperatures()` reading vs a limit, with cool-down hysteresis.

    Without hysteresis a node pausing at exactly the limit would cool 1C within one poll and
    resume, oscillating every poll; requiring `hysteresis_c` of cooling lets the chassis recover.
    """

    def __init__(self, max_temp_c: float, hysteresis_c: float = 5.0,
                 sensors: Optional[Callable[[], dict[str, list[Any]]]] = None):
        self.max_temp_c = max_temp_c
        self.hysteresis_c = hysteresis_c
        self.sensors = sensors or getattr(psutil, "sensors_temperatures", lambda: {})
        self.last_temp_c: Optional[float] = None
        self._hot = False

    def __call__(self) -> Optional[bool]:
        temps = [t.current for entries in (self.sensors() or {}).values() for t in entries
                 if t.current is not None and t.current > 0]
        if not temps:
            self._hot = False
            return None
        self.last_temp_c = max(temps)
        limit = self.max_temp_c - self.hysteresis_c if self._hot else self.max_temp_c
        self._hot = self.last_temp_c >= limit
        return self._hot


def thermal_reader(max_temp_c: float) -> Optional[Callable[[], Optional[bool]]]:
    """Platform thermal-throttle reader, or None where unsupported (Windows has no portable API)."""
    if sys.platform == "darwin":
        return macos_thermal_throttled
    if sys.platform.startswith("linux"):
        return LinuxThermalReader(max_temp_c)
    return None


# ============================================================================ CPU readers
# Processes whose CPU load is *ours* (the inference server this worker drives). Ollama spawns
# runner subprocesses ("ollama", "ollama_llama_server"); llama.cpp ships llama-server/llama-cli.
OWN_PROCESS_PREFIXES = ("ollama", "llama-", "llama_", "llama.cpp")
# Other Nightshift workers on this host: grid load, not the owner's. Without this, co-located
# workers (chaos tests, a multi-GPU box) would see each other as "foreign" and preempt each other.
PEER_WORKER_MARKER = "worker_daemon"
_PYTHON_NAME_PREFIXES = ("python", "pypy")


def is_inference_process(name: str) -> bool:
    return name.lower().startswith(OWN_PROCESS_PREFIXES)


def is_peer_worker_cmdline(cmdline: Optional[Sequence[str]]) -> bool:
    """True for `python worker_daemon.py ...` and `python -m worker_daemon ...` (any path)."""
    return any(os.path.basename(arg).startswith(PEER_WORKER_MARKER) for arg in cmdline or ())


def system_cpu_seconds() -> tuple[float, float]:
    """(busy, total) CPU-seconds summed over all cores since boot, mirroring psutil's accounting.

    We difference raw cpu_times ourselves instead of calling `psutil.cpu_percent(interval=None)`
    because that function keeps a single module-global baseline: the worker's heartbeat also
    calls it, which would silently shrink our measurement window to "since the last heartbeat".
    """
    t = psutil.cpu_times()
    # Linux counts guest time inside user time already; psutil subtracts it to avoid double counting.
    total = sum(t) - getattr(t, "guest", 0.0) - getattr(t, "guest_nice", 0.0)
    idle = t.idle + getattr(t, "iowait", 0.0)
    return total - idle, total


class GridProcessScanner:
    """CPU-seconds (user+system) of every process whose load is the grid's, not the owner's:
    this worker, local inference servers, and peer Nightshift workers.

    Peer detection needs the command line, which is expensive on macOS (one sysctl per process)
    and usually AccessDenied for other users' processes; so only python-named processes are
    inspected, and each verdict is cached per (pid, create_time) - a command line never changes,
    and keying on create_time stops a recycled pid inheriting a stale verdict.

    Processes may vanish or be unreadable between listing and reading; they are skipped. An
    unreadable inference server (e.g. running as another user on macOS) is therefore counted
    as foreign load - the safe direction for a good citizen.
    """

    def __init__(self, own_pid: Optional[int] = None, exclude_peers: bool = True):
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self.exclude_peers = exclude_peers
        self._peer_cache: dict[tuple[int, float], bool] = {}

    def _is_peer(self, p: psutil.Process) -> bool:
        key = (p.pid, p.info.get("create_time") or 0.0)
        if key not in self._peer_cache:
            try:
                self._peer_cache[key] = is_peer_worker_cmdline(p.cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                self._peer_cache[key] = False
        return self._peer_cache[key]

    def __call__(self) -> dict[int, float]:
        out: dict[int, float] = {}
        live: set[tuple[int, float]] = set()
        for p in psutil.process_iter(["name", "create_time"]):
            name = p.info.get("name") or ""
            live.add((p.pid, p.info.get("create_time") or 0.0))
            ours = (p.pid == self.own_pid or is_inference_process(name)
                    or (self.exclude_peers and name.lower().startswith(_PYTHON_NAME_PREFIXES) and self._is_peer(p)))
            if not ours:
                continue
            try:
                t = p.cpu_times()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            out[p.pid] = t.user + t.system
        self._peer_cache = {k: v for k, v in self._peer_cache.items() if k in live}
        return out


def own_process_cpu_seconds(own_pid: Optional[int] = None) -> dict[int, float]:
    """One-shot `GridProcessScanner` (no verdict cache); kept for callers of the old function."""
    return GridProcessScanner(own_pid)()


class ForeignCpuSampler:
    """System CPU% attributable to processes other than the grid's (see `GridProcessScanner`).

    Computed from CPU-second deltas between calls: (system busy - grid processes' busy) / total,
    as a percentage of *total host capacity* (all cores), so 65% on an 8-core box means ~5.2
    cores of someone else's work. This needs a baseline ("priming"): taken in __init__ and again
    via `prime()` when monitoring actually starts, so the first sample never averages over the
    worker's own startup. A process first seen in the current sample has no baseline and one that
    vanished has no final reading; both are treated as foreign for that single interval, which
    the caller's sustain window (N consecutive samples) absorbs.
    """

    def __init__(self, system_times: Callable[[], tuple[float, float]] = system_cpu_seconds,
                 own_times: Optional[Callable[[], dict[int, float]]] = None):
        self.system_times = system_times
        self.own_times = own_times if own_times is not None else GridProcessScanner()
        self._last = 0.0
        self.prime()

    def prime(self) -> None:
        self._prev_sys = self.system_times()
        self._prev_own = self.own_times()

    def __call__(self) -> float:
        (busy, total), own = self.system_times(), self.own_times()
        (prev_busy, prev_total), prev_own = self._prev_sys, self._prev_own
        self._prev_sys, self._prev_own = (busy, total), own
        d_total = total - prev_total
        if d_total <= 0:  # called twice within the kernel's accounting tick; nothing new to say
            return self._last
        d_own = sum(max(0.0, own[pid] - prev_own[pid]) for pid in own.keys() & prev_own.keys())
        self._last = min(100.0, max(0.0, 100.0 * (busy - prev_busy - d_own) / d_total))
        return self._last


def available_ram_bytes() -> float:
    return float(psutil.virtual_memory().available)


# ============================================================================ sources
class ActivitySource(Protocol):
    """One reason the machine may be off-limits. `check()` is awaited every monitor tick
    (~100ms); implementations rate-limit their own expensive reads."""

    name: str

    async def check(self) -> Optional[str]:
        """Reason string when the machine must NOT be harvested, else None."""
        ...

    def describe(self) -> str:
        """Short human-readable summary for the worker's startup log."""
        ...


class TriggerFileSource:
    """Deterministic chaos trigger: a `stat` per tick is cheap enough to never cache."""

    name = "trigger_file"

    def __init__(self, path: str):
        self.path = path

    async def check(self) -> Optional[str]:
        return "trigger_file" if os.path.exists(self.path) else None

    def describe(self) -> str:
        return f"trigger file {self.path}"


class PolledSource:
    """Base for sources with a costly read: rate-limits, runs the read in a thread, caches the
    answer, and routes unknown readings through `on_unknown` (default: fail open, loudly).

    Subclasses implement `read()` (blocking I/O, runs off-loop; return None for "unknown") and
    `evaluate(reading)` (pure decision + hysteresis state, runs on the loop).
    """

    name = "polled"
    unavailable_msg: Optional[str] = "sensor unavailable"  # None: "unknown" is a definite "no"

    def __init__(self, poll_s: float, log: Log = print):
        self.poll_s = poll_s
        self.log = log
        self._polled_at: Optional[float] = None
        self._reason: Optional[str] = None
        self._warned = False

    def read(self) -> Any:
        raise NotImplementedError

    def evaluate(self, reading: Any) -> Optional[str]:
        raise NotImplementedError

    def reset(self) -> None:
        """Forget hysteresis state after an unknown reading (we are failing open)."""

    def on_unknown(self, why: Optional[str]) -> Optional[str]:
        """Reason to report when the reading is unknown. Default policy: fail open, loudly."""
        if why is not None:
            self.warn_once(why)
        self.reset()
        return None

    def warn_once(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            self.log(f"[warn] {self.name} guard: {msg}; not pausing for it")

    async def check(self) -> Optional[str]:
        now = time.monotonic()
        if self._polled_at is not None and now - self._polled_at < self.poll_s:
            return self._reason
        self._polled_at = now
        why = self.unavailable_msg
        try:
            reading = await asyncio.to_thread(self.read)
        except Exception as e:  # a broken sensor must not kill the preemption watchdog
            reading, why = None, f"read failed ({type(e).__name__}: {e})"
        self._reason = self.on_unknown(why) if reading is None else self.evaluate(reading)
        return self._reason

    def describe(self) -> str:
        return self.name


class HIDIdleSource(PolledSource):
    """Paused while the last keyboard/mouse event is younger than `threshold` seconds. The
    threshold itself is the hysteresis: one keypress pauses, N quiet seconds resume.

    With `fail_closed` (the default), a reader that stops answering is treated as "user possibly
    active": `max_failures` consecutive unknown reads report "idle_sensor_lost" until a read
    succeeds. Fewer failures hold the previous answer - a single slow subprocess must neither
    resume a node paused for typing nor pause an idle one.
    """

    name = "hid_input"
    unavailable_msg = "native idle detection unavailable on this host"

    def __init__(self, threshold: float, reader: Callable[[], Optional[float]], poll_s: float = 0.5,
                 fail_closed: bool = True, max_failures: int = SENSOR_LOSS_FAILURES, log: Log = print):
        super().__init__(poll_s, log)
        self.threshold = threshold
        self.reader = reader
        self.fail_closed = fail_closed
        self.max_failures = max(1, max_failures)
        self.last_idle_seconds: Optional[float] = None
        self.failures = 0
        self.lost = False

    def read(self) -> Optional[float]:
        return self.reader()

    def evaluate(self, idle: float) -> Optional[str]:
        if self.lost:
            self.log(f"{self.name} guard: idle sensor recovered after {self.failures} failed reads")
        self.failures, self.lost = 0, False
        self.last_idle_seconds = idle
        return "hid_input" if idle < self.threshold else None

    def on_unknown(self, why: Optional[str]) -> Optional[str]:
        if not self.fail_closed:
            return super().on_unknown(why)
        self.failures += 1
        if self.failures < self.max_failures:
            return self._reason
        if not self.lost:
            self.lost = True
            self.log(f"[warn] {self.name} guard: idle sensor failed {self.failures} reads in a row ({why}); "
                     f"pausing until it answers (fail-closed; --allow-no-idle-sensor to fail open)")
        self.last_idle_seconds = None
        return "idle_sensor_lost"

    def describe(self) -> str:
        via = getattr(self.reader, "name", None)
        policy = "fail-closed" if self.fail_closed else "fail-open"
        return f"HID idle < {self.threshold:.0f}s ({f'{via}, ' if via else ''}{policy})"


class BatterySource(PolledSource):
    """Never drain a laptop: pause while unplugged, and optionally while charging below a floor.

    `min_percent` uses a 5-point resume margin so a battery hovering at the floor does not
    flap the node every poll.
    """

    name = "battery"
    unavailable_msg = None  # sensors_battery() is None on desktops: no battery, nothing to protect
    RESUME_MARGIN = 5.0

    def __init__(self, min_percent: Optional[float] = None, poll_s: float = 5.0,
                 reader: Callable[[], Any] = psutil.sensors_battery, log: Log = print):
        super().__init__(poll_s, log)
        self.min_percent = min_percent
        self.reader = reader
        self._low = False

    def read(self) -> Any:
        return self.reader()

    def evaluate(self, battery: Any) -> Optional[str]:
        if battery.power_plugged is None:
            self.warn_once("battery present but AC state unknown")
            self._low = False
            return None
        if not battery.power_plugged:
            return "on_battery"
        if self.min_percent is None or battery.percent is None:
            self._low = False
            return None
        floor = self.min_percent + self.RESUME_MARGIN if self._low else self.min_percent
        self._low = battery.percent < floor
        return "battery_low" if self._low else None

    def reset(self) -> None:
        self._low = False

    def describe(self) -> str:
        try:
            present = self.reader() is not None
        except Exception:
            present = False
        floor = f", min {self.min_percent:.0f}%" if self.min_percent is not None else ""
        return f"battery{floor}" if present else "battery (none detected)"


class ThermalSource(PolledSource):
    """Pause while the OS reports thermal throttling: inference on a throttled chassis is slow
    for us and makes the owner's machine hot and loud."""

    name = "thermal"
    unavailable_msg = "thermal state unreadable on this host"

    def __init__(self, reader: Callable[[], Optional[bool]], poll_s: float = 10.0, log: Log = print):
        super().__init__(poll_s, log)
        self.reader = reader

    def read(self) -> Optional[bool]:
        return self.reader()

    def evaluate(self, throttled: bool) -> Optional[str]:
        return "thermal_throttle" if throttled else None

    def describe(self) -> str:
        limit = getattr(self.reader, "max_temp_c", None)
        return f"thermal (>= {limit:.0f}C)" if limit is not None else "thermal (pmset)"


_PRIMING = object()  # CpuBusySource's first read: re-baseline the sampler, decide nothing


class CpuBusySource(PolledSource):
    """Pause when someone else's workload needs the CPU (the CLAUDE.md "CPU spike" invariant).

    Sustain + hysteresis: pause only after `sustain` consecutive samples above `threshold`, and
    resume only after `sustain` consecutive samples below `threshold * resume_ratio`. A single
    compile or browser tab spike must not flap the node, and "just under the threshold" must not
    count as calm.

    Because this guard is on by default, startup must not false-trigger: the first poll only
    re-primes the sampler (if it has `prime()`), so the first decision-bearing sample covers
    one clean poll interval after monitoring started, not "since the worker was constructed".
    """

    name = "cpu_busy"
    unavailable_msg = "foreign CPU load unreadable"

    def __init__(self, threshold: float, sustain: int = 3, resume_ratio: float = 0.7, poll_s: float = 1.0,
                 sampler: Optional[Callable[[], Optional[float]]] = None, log: Log = print):
        super().__init__(poll_s, log)
        self.threshold = threshold
        self.resume_below = threshold * resume_ratio
        self.sustain = max(1, sustain)
        self.sampler = sampler if sampler is not None else ForeignCpuSampler()
        self.last_percent: Optional[float] = None
        self._busy = False
        self._streak = 0
        self._primed = not callable(getattr(self.sampler, "prime", None))

    def read(self) -> Any:
        if not self._primed:
            self._primed = True
            self.sampler.prime()  # type: ignore[union-attr]
            return _PRIMING
        return self.sampler()

    def evaluate(self, pct: Any) -> Optional[str]:
        if pct is _PRIMING:
            return None
        self.last_percent = pct
        crossing = pct < self.resume_below if self._busy else pct > self.threshold
        self._streak = self._streak + 1 if crossing else 0
        if self._streak >= self.sustain:
            self._busy, self._streak = not self._busy, 0
        return "cpu_busy" if self._busy else None

    def reset(self) -> None:
        self._busy, self._streak = False, 0

    def describe(self) -> str:
        return f"non-worker CPU > {self.threshold:.0f}% x{self.sustain} (resume < {self.resume_below:.0f}%)"


class LowMemorySource(PolledSource):
    """Pause when the owner's workload is squeezing RAM, without flapping on our own model.

    The trap: pausing under pressure unloads the model (keep_alive=0), which frees its weights;
    a naive "resume when available >= threshold" then resumes at once, the reload eats the same
    GB again, and the node oscillates pause -> unload -> resume -> reload -> pause.

    Design:
    * pause after `sustain` consecutive samples with available < `min_free`;
    * resume only after `sustain` consecutive samples with
      available >= `min_free` + model size + `margin`  - i.e. the model can be reloaded and still
      leave `margin` above the floor. (Deliberately additive rather than `min_free + max(margin,
      model)`: with the max form, a reload lands exactly on the floor and the next allocation
      re-pauses.) The model size is a live callable (the worker feeds it from Ollama `/api/ps`,
      with a conservative fallback when unknown, and 0 in mock mode where nothing loads);
    * minimum pause dwell `min_pause_s`, doubled (capped at `max_pause_s`) each time a pause
      follows the previous resume within `flap_window_s` - if the estimate is still wrong, the
      oscillation slows down exponentially instead of hammering the disk with model reloads.
    Pausing is never delayed by the dwell: safety first, only resuming is slowed.
    """

    name = "memory"
    unavailable_msg = "available RAM unreadable"

    def __init__(self, min_free_gb: float, *, sustain: int = 3, margin_gb: float = 1.0,
                 model_bytes: Callable[[], float] = lambda: 0.0, min_pause_s: float = 30.0,
                 max_pause_s: float = 600.0, flap_window_s: float = 120.0, poll_s: float = 1.0,
                 reader: Callable[[], Optional[float]] = available_ram_bytes,
                 clock: Callable[[], float] = time.monotonic, log: Log = print):
        super().__init__(poll_s, log)
        self.min_free = min_free_gb * GB
        self.margin = margin_gb * GB
        self.sustain = max(1, sustain)
        self.model_bytes = model_bytes
        self.min_pause_s, self.max_pause_s, self.flap_window_s = min_pause_s, max_pause_s, flap_window_s
        self.reader = reader
        self.clock = clock
        self.last_available: Optional[float] = None
        self.dwell_s = min_pause_s
        self._low = False
        self._streak = 0
        self._paused_at: Optional[float] = None
        self._resumed_at: Optional[float] = None

    def resume_level(self) -> float:
        return self.min_free + max(0.0, float(self.model_bytes() or 0.0)) + self.margin

    def read(self) -> Optional[float]:
        return self.reader()

    def evaluate(self, avail: float) -> Optional[str]:
        now = self.clock()
        self.last_available = avail
        if not self._low:
            self._streak = self._streak + 1 if avail < self.min_free else 0
            if self._streak >= self.sustain:
                flapping = self._resumed_at is not None and now - self._resumed_at < self.flap_window_s
                self.dwell_s = min(self.dwell_s * 2, self.max_pause_s) if flapping else self.min_pause_s
                self._low, self._streak, self._paused_at = True, 0, now
                self.log(f"{self.name} guard: {avail / GB:.1f} GB available < {self.min_free / GB:.1f} GB; "
                         f"pausing >= {self.dwell_s:.0f}s, resume at >= {self.resume_level() / GB:.1f} GB")
        else:
            dwelled = self._paused_at is None or now - self._paused_at >= self.dwell_s
            self._streak = self._streak + 1 if dwelled and avail >= self.resume_level() else 0
            if self._streak >= self.sustain:
                self._low, self._streak, self._resumed_at = False, 0, now
        return "low_memory" if self._low else None

    def reset(self) -> None:
        self._low, self._streak = False, 0

    def describe(self) -> str:
        return (f"free RAM < {self.min_free / GB:.1f} GB x{self.sustain} "
                f"(resume >= floor + model + {self.margin / GB:.1f} GB, min pause {self.min_pause_s:.0f}s)")


def build_guard_sources(
    *, allow_battery: bool = False, min_battery_percent: Optional[float] = None,
    ignore_thermal: bool = False, max_temp_c: float = 90.0,
    max_host_cpu_pct: Optional[float] = DEFAULT_MAX_HOST_CPU_PCT, cpu_busy_samples: int = 3,
    min_free_ram_gb: Optional[float] = DEFAULT_MIN_FREE_RAM_GB,
    model_bytes: Callable[[], float] = lambda: 0.0, log: Log = print,
) -> list[ActivitySource]:
    """Default good-citizen guards from worker CLI flags; defaults match the worker CLI.

    `max_host_cpu_pct` / `min_free_ram_gb` of None or 0 disable those guards. `model_bytes`
    feeds the memory guard's resume hysteresis (see `LowMemorySource`).
    """
    sources: list[ActivitySource] = []
    if not allow_battery:
        sources.append(BatterySource(min_percent=min_battery_percent, log=log))
    if not ignore_thermal:
        reader = thermal_reader(max_temp_c)
        if reader is None:
            log(f"[warn] thermal guard unsupported on {sys.platform}; not pausing for it")
        else:
            sources.append(ThermalSource(reader, log=log))
    if max_host_cpu_pct:
        sources.append(CpuBusySource(max_host_cpu_pct, sustain=cpu_busy_samples, log=log))
    if min_free_ram_gb:
        sources.append(LowMemorySource(min_free_ram_gb, model_bytes=model_bytes, log=log))
    return sources


# ============================================================================ monitor
@dataclass
class ActivitySample:
    active: bool
    reason: Optional[str] = None             # highest-priority reason (what the coordinator is told)
    idle_seconds: Optional[float] = None
    reasons: tuple[str, ...] = field(default_factory=tuple)  # every source currently objecting


class ActivityMonitor:
    """ORs activity sources and invokes callbacks on active/idle transitions.

    Sources are checked concurrently each tick (so a slow `pmset` spawn does not delay the
    trigger-file check) and the first non-None reason in priority order wins:
    trigger file, HID input, then `guards` in the order given. `ActivitySample.reasons` keeps
    all of them, because some reactions (unloading the model under memory pressure) must happen
    even when a higher-priority reason is the one reported.

    HID idle (when `native_idle_threshold_s` is set) is probed here, at construction: the
    reader - injected, or the first working one of `idle_candidates` / the platform defaults -
    must return a reading now, else `IdleSensorUnavailable` is raised (fail closed) unless
    `allow_no_idle_sensor`, in which case HID detection is disabled with a warning.
    """

    def __init__(
        self,
        trigger_path: str,
        native_idle_threshold_s: Optional[float] = None,
        native_reader: Optional[Callable[[], Optional[float]]] = None,
        file_poll_s: float = 0.1,
        native_poll_s: float = 0.5,
        guards: Iterable[ActivitySource] = (),
        allow_no_idle_sensor: bool = False,
        idle_candidates: Optional[Sequence[IdleReader]] = None,
        sensor_loss_failures: int = SENSOR_LOSS_FAILURES,
        log: Log = print,
    ):
        self.trigger_path = trigger_path
        self.threshold = native_idle_threshold_s
        self.file_poll_s = file_poll_s
        self.log = log
        self.hid: Optional[HIDIdleSource] = None
        if native_idle_threshold_s is not None:
            if native_reader is not None:
                candidates = [native_reader if isinstance(native_reader, IdleReader)
                              else IdleReader(getattr(native_reader, "name", "injected"), native_reader)]
            else:
                candidates = list(idle_candidates) if idle_candidates is not None else native_idle_candidates()
            reader = probe_idle_reader(candidates)
            if reader is None:
                tried = ", ".join(c.name for c in candidates) or "none available"
                msg = (f"--native-idle-seconds is set but no HID idle reader works on this host "
                       f"(tried: {tried}). {idle_sensor_help()}")
                if not allow_no_idle_sensor:
                    raise IdleSensorUnavailable(
                        f"{msg} Refusing to harvest blind; pass --allow-no-idle-sensor to run without "
                        f"keyboard/mouse detection (fail-open), or drop --native-idle-seconds.")
                log(f"[warn] hid_input guard: {msg} Continuing WITHOUT HID detection (--allow-no-idle-sensor).")
            else:
                self.hid = HIDIdleSource(native_idle_threshold_s, reader, poll_s=native_poll_s,
                                         fail_closed=not allow_no_idle_sensor,
                                         max_failures=sensor_loss_failures, log=log)
        self.sources: list[ActivitySource] = [TriggerFileSource(trigger_path)]
        if self.hid is not None:
            self.sources.append(self.hid)
        self.sources.extend(guards)
        self.active = False
        self.reason: Optional[str] = None
        self.reasons: tuple[str, ...] = ()

    @property
    def native_enabled(self) -> bool:
        return self.hid is not None

    @property
    def last_idle_seconds(self) -> Optional[float]:
        return self.hid.last_idle_seconds if self.hid is not None else None

    def describe(self) -> str:
        """Which guards are on, for the worker's startup log (trigger file excluded; it is always on)."""
        return ", ".join(s.describe() for s in self.sources[1:]) or "none"

    async def sample(self) -> ActivitySample:
        got: Sequence[Optional[str]] = await asyncio.gather(*(s.check() for s in self.sources))
        reasons = tuple(r for r in got if r is not None)
        reason = reasons[0] if reasons else None
        return ActivitySample(reason is not None, reason, self.last_idle_seconds, reasons)

    async def run(
        self,
        on_active: Callable[[ActivitySample, float], Awaitable[None]],
        on_idle: Callable[[], Awaitable[None]],
        on_reasons_change: Optional[Callable[[ActivitySample], Awaitable[None]]] = None,
    ) -> None:
        """Poll forever. `on_reasons_change` fires while already paused when the *set* of
        objecting sources changes (e.g. memory got low during a trigger-file pause)."""
        while True:
            s = await self.sample()
            if s.active and not self.active:
                self.active, self.reason, self.reasons = True, s.reason, s.reasons
                await on_active(s, time.time())
            elif s.active:
                if s.reason != self.reason:
                    # Still paused, for a different reason (e.g. trigger removed but now on battery).
                    self.log(f"still paused: reason {self.reason} -> {s.reason}")
                    self.reason = s.reason
                if set(s.reasons) != set(self.reasons):
                    self.reasons = s.reasons
                    if on_reasons_change is not None:
                        await on_reasons_change(s)
            elif self.active:
                self.active, self.reason, self.reasons = False, None, ()
                await on_idle()
            await asyncio.sleep(self.file_poll_s)
