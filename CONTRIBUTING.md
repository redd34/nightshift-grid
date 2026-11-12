# Contributing to Nightshift Grid

Thanks for helping! Nightshift Grid harvests idle cycles on other people's machines, so two rules
outrank everything else: **never slow down the human at the keyboard**, and **never lose or
duplicate a chunk**. Changes are judged against those first.

## Dev setup

```bash
git clone https://github.com/redd34/nightshift-grid && cd nightshift-grid
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q tests                  # fast, offline, no Ollama needed
python -m pyflakes *.py tests/*.py scripts/*.py
python chaos_test.py --mock --phases 1,2,3 # end-to-end fault injection (~5 min, no Ollama)
```

Live inference needs [Ollama](https://ollama.com) and a small model (`ollama pull qwen3:4b`).
Always pass a bounded `num_ctx` (the worker uses 4096): an unset context makes Ollama load
qwen3 at its 262k default, which costs about 40 GB.

## Where things live

`coordinator.py` holds the whole lease state machine in `GridState` (injectable clock, so tests
run on a fake one). `worker_daemon.py` is the preemptible worker, `idle_detection.py` the
activity sensors, `relay.py` the Baton resume helpers. The README has a full file map.

## Pull requests

- Add or update tests for behaviour changes. State-machine changes should keep
  `GridState.check_invariants()` passing and, where it matters, survive the durability fuzz in
  `tests/test_durability.py`.
- Run the mock chaos test for anything touching leases, preemption, recovery or the worker loop.
- Report numbers honestly: say whether a result is mock or live, which model, and on what
  hardware. Please don't present projections next to measurements without labelling them.
- Keep secrets off argv and out of logs (tokens come from files or env).

## Good first issues

- Run the Linux (Wayland/X11) and Windows idle sensors and installers on real hardware and report.
- Battery and thermal guards on real laptops (today they're tested with fakes only).
- A Raspberry Pi 5 or 8 GB CPU-only worker run with `chaos_test.py`, with numbers.

By contributing you agree your contributions are licensed under the Apache License 2.0.
