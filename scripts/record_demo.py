"""Record docs/demo.svg: the real dashboard of a real (mock-inference) grid run, as an animated SVG.

Starts a coordinator and three mock workers on loopback, submits the 24-chunk demo job, and has
"the human" sit down at node-legacy-laptop mid-job (trigger file) and leave again. Every second
the actual dashboard renderable is captured with rich and the frames are stacked into one SVG
film strip animated with CSS (GitHub renders these in READMEs; no GIF tooling needed).

    python scripts/record_demo.py            # writes docs/demo.svg (~40 s)
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dashboard import render_dashboard  # noqa: E402
from demo_submit import SAMPLE_CHUNKS  # noqa: E402
from worker_daemon import TRIGGER_DIR  # noqa: E402

PORT = 18990
URL = f"http://127.0.0.1:{PORT}"
WORKERS = [("node-office-pc", 2), ("node-legacy-laptop", 1), ("node-mac-studio", 3)]
HUMAN = "node-legacy-laptop"
WIDTH, HEIGHT = 128, 34
OUT = ROOT / "docs" / "demo.svg"


def frame(data: dict, caption: str) -> str:
    c = Console(record=True, width=WIDTH, height=HEIGHT, force_terminal=True, color_system="truecolor",
                file=open(os.devnull, "w"))
    c.print(render_dashboard(data, footer=caption), height=HEIGHT)
    return c.export_svg(title="nightshift up")


def main() -> int:
    trigger = Path(TRIGGER_DIR) / f"preempt_{HUMAN}"
    trigger.unlink(missing_ok=True)
    procs = [subprocess.Popen([sys.executable, "coordinator.py", "--port", str(PORT), "--db", "none",
                               "--output", "none", "--relay"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    frames: list[str] = []
    try:
        for _ in range(50):
            try:
                if httpx.get(f"{URL}/healthz", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        for wid, tier in WORKERS:
            procs.append(subprocess.Popen([sys.executable, "worker_daemon.py", "--id", wid, "--tier", str(tier),
                                           "--coordinator", URL, "--mock", "--model", "qwen3:4b", "--allow-battery",
                                           "--max-host-cpu-pct", "off"], cwd=ROOT,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2.5)
        job = httpx.post(f"{URL}/v1/grid/mapreduce", json={"job_name": "GraphRAG demo (24 chunks)",
                                                           "chunks": SAMPLE_CHUNKS, "required_tier": 1}).json()
        t0, done_at = time.time(), None
        while True:
            t = time.time() - t0
            if 5 <= t < 13:
                trigger.touch()
                caption = f"t={t:4.1f}s  a human sat down at {HUMAN}: it yielded instantly; its chunk moved on"
            else:
                trigger.unlink(missing_ok=True)
                caption = (f"t={t:4.1f}s  every node runs its own small model; leases mean no chunk is ever lost"
                           if t < 5 else f"t={t:4.1f}s  {HUMAN} is idle again and back to harvesting")
            data = httpx.get(f"{URL}/v1/grid/status", timeout=2).json()
            st = httpx.get(f"{URL}/v1/grid/jobs/{job['job_id']}", timeout=2).json()
            if st["done"] and done_at is None:
                done_at = t
            if done_at is not None:
                caption = (f"done: {st['counts']['COMPLETED']}/{st['total']} chunks in {st['elapsed_s']:.1f}s, "
                           f"{st['total_preemptions']} preemption(s), {st['counts']['FAILED']} dead-lettered")
            frames.append(frame(data, caption))
            if done_at is not None and t - done_at > 3:
                break
            time.sleep(1.0)
    finally:
        trigger.unlink(missing_ok=True)
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    write_filmstrip(frames)
    print(f"wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB)")
    write_video(frames)
    return 0


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def write_video(frames: list[str]) -> None:
    """docs/demo.gif (README) and docs/demo.mp4 (social posts): each frame rasterized by headless
    Chrome, encoded by ffmpeg. Skipped with a note if either tool is missing."""
    import shutil
    chrome = os.environ.get("CHROME", CHROME)
    if not (Path(chrome).exists() and shutil.which("ffmpeg")):
        print("skipping GIF/MP4: needs Google Chrome and ffmpeg")
        return
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', frames[0])
    w, h = int(float(m.group(1))) + 1, int(float(m.group(2))) + 1
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            svg = Path(d, f"f{i:03d}.svg")
            svg.write_text(f, encoding="utf-8")
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                            f"--screenshot={d}/f{i:03d}.png", svg.as_uri()], capture_output=True, timeout=60)
        pal = f"{d}/palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", f"{d}/f%03d.png",
                        "-vf", "scale=1000:-1:flags=lanczos,palettegen=max_colors=64", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", f"{d}/f%03d.png", "-i", pal,
                        "-lavfi", "scale=1000:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none", "-loop", "0",
                        str(OUT.with_suffix(".gif"))], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i", f"{d}/f%03d.png",
                        "-vf", "scale=1280:-2,fps=10,format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart",
                        str(OUT.with_suffix(".mp4"))], check=True)
    for ext in (".gif", ".mp4"):
        print(f"wrote {OUT.with_suffix(ext)} ({OUT.with_suffix(ext).stat().st_size / 1e6:.2f} MB)")


def write_filmstrip(frames: list[str], seconds_per_frame: float = 1.0) -> None:
    """Stack the frame SVGs vertically and step a viewport through them with one CSS animation."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', frames[0])
    w, h = float(m.group(1)), float(m.group(2))
    n = len(frames)
    parts = []
    for i, f in enumerate(frames):
        inner = f[f.index("<svg"):]
        inner = inner.replace("<svg ", f'<svg y="{i * h:.1f}" width="{w}" height="{h}" ', 1)
        parts.append(inner)
    total = n * seconds_per_frame
    keys = "".join(f"{100 * i / n:.4f}%{{transform:translateY({-i * h:.1f}px)}}" for i in range(n))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
           f"<style>@keyframes strip{{{keys}100%{{transform:translateY({-(n - 1) * h:.1f}px)}}}}"
           f".strip{{animation:strip {total:.0f}s steps(1,end) infinite}}</style>"
           f'<g class="strip">{"".join(parts)}</g></svg>')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
