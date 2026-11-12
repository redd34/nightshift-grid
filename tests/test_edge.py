import asyncio, sys

import pytest

from graphrag import ExtractionError, normalize_extraction
from idle_detection import ActivityMonitor, IdleSensorUnavailable, macos_idle_seconds, parse_ioreg_hid_idle


def test_normalize_variants():
    out = normalize_extraction('{"entities":[{"name":"PLC-1"}],"relationships":[{"source":"PLC-1","relation":"HAS","target":"B-101"},["x","y"],["a","",""]]}')
    assert out["triples"] == [["PLC-1", "HAS", "B-101"]]
    assert out["entities"] == ["PLC-1", "B-101"] and out["dropped_triples"] == 2


def test_relations_canonicalized():
    out = normalize_extraction({"triples": [["a", "exceeded during", "b"], ["a", "exceededDuring", "b"], ["a", "!!", "c"]]})
    assert out["triples"] == [["a", "EXCEEDED_DURING", "b"]] and out["dropped_triples"] == 1


@pytest.mark.parametrize("bad", ["not json", "[1,2]", '{"entities":[],"triples":[]}'])
def test_normalize_rejects(bad):
    with pytest.raises(ExtractionError):
        normalize_extraction(bad)


def test_parse_ioreg():
    assert parse_ioreg_hid_idle('  | |   "HIDIdleTime" = 1032886875\n') == pytest.approx(1.032886875)
    assert parse_ioreg_hid_idle("nothing") is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_reader_live():
    v = macos_idle_seconds()
    assert v is not None and v >= 0


def test_monitor_transitions(tmp_path):
    trigger = tmp_path / "preempt"
    idle = {"v": 100.0}
    mon = ActivityMonitor(str(trigger), native_idle_threshold_s=30, native_reader=lambda: idle["v"],
                          file_poll_s=0.01, native_poll_s=0.0)
    seen = []

    async def scenario():
        async def on_active(s, ts): seen.append(("active", s.reason))
        async def on_idle(): seen.append(("idle", None))
        task = asyncio.create_task(mon.run(on_active, on_idle))
        await asyncio.sleep(0.05); trigger.touch()
        await asyncio.sleep(0.05); trigger.unlink()
        await asyncio.sleep(0.05); idle["v"] = 2.0          # user moved mouse
        await asyncio.sleep(0.05); idle["v"] = 31.0         # idle long enough
        await asyncio.sleep(0.05); task.cancel()

    asyncio.run(scenario())
    assert seen == [("active", "trigger_file"), ("idle", None), ("active", "hid_input"), ("idle", None)]


def test_monitor_unavailable_reader_fails_closed_unless_allowed(tmp_path):
    with pytest.raises(IdleSensorUnavailable, match="--allow-no-idle-sensor"):
        ActivityMonitor(str(tmp_path / "x"), native_idle_threshold_s=30, native_reader=lambda: None)
    mon = ActivityMonitor(str(tmp_path / "x"), native_idle_threshold_s=30, native_reader=lambda: None, native_poll_s=0,
                          allow_no_idle_sensor=True, log=lambda m: None)
    assert not mon.native_enabled and asyncio.run(mon.sample()).active is False


SRC = ("Equipment Log #16: Compressor turbine vibration exceeded 4.2 mm/s on PLC-Node-2 during shift handover. "
       "Technician replaced bearing assembly B-117; later bearing assembly B-120 was fitted.")


@pytest.mark.parametrize("corrupt, fixed", [   # real corruptions observed from qwen3:4b + JSON-schema decoding
    ("4.2 mm: s", "4.2 mm/s"), ("4.2 mm: mm/s", "4.2 mm/s"), ("Compressor: turbine", "Compressor turbine"),
    ("PLC-Node-:2", "PLC-Node-2"), ("bearing assembly B-12:0", "bearing assembly B-120"),
    ("bearing assembly B-1:117", "bearing assembly B-117"),
])
def test_repair_value_regrounds_in_source(corrupt, fixed):
    from graphrag import repair_value
    assert repair_value(corrupt, SRC) == (fixed, True)


def test_repair_never_invents_text():
    from graphrag import repair_value
    assert repair_value("B-9:99", SRC) == ("B-9:99", False)          # not in source: left alone
    assert repair_value("Equipment Log #16:", SRC)[1] is False       # colon really is in the source
    assert repair_value("no colon", SRC) == ("no colon", False)


def test_normalize_with_source_repairs_and_dedupes():
    out = normalize_extraction({"entities": ["B-12:0", "B-120"], "triples": [["Technician", "replaced", "B-12:0"]]},
                               source_text=SRC)
    assert out["entities"] == ["B-120", "Technician"] and out["triples"] == [["Technician", "REPLACED", "B-120"]]
    assert out["repaired_values"] == 2
