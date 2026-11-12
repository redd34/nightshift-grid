"""Live `rich` TUI for the Nightshift grid: node states, tok/s and MapReduce progress.

    python dashboard.py --coordinator http://127.0.0.1:8000 --token-file .nightshift/client.token

The rendering is split into pure functions of the `/v1/grid/status` payload (`header_text`,
`workers_table`, `render_dashboard`) so that `nightshift.py status` can print the same table
once, and tests can render without a terminal. `run_dashboard()` is the blocking entry point
`nightshift.py up` calls in its own terminal; it takes an optional `should_stop` callback so the
launcher can end the TUI when the coordinator it supervises dies, instead of showing
"waiting for coordinator" forever.
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Callable, Optional

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from grid_auth import CLIENT_TOKEN_ENV, GridAuthError, bearer_headers, check_auth, resolve_token
from grid_tls import client_verify

COORDINATOR_URL = "http://127.0.0.1:8000"
console = Console()

STATE_COLORS = {
    "IDLE_HARVESTING": "green",
    "BUSY_INFERENCING": "bold cyan",
    "PREEMPTED_USER_ACTIVE": "bold yellow",
    "OFFLINE": "bold red",
}


def header_text(data: dict) -> str:
    counts = data.get("task_counts", {})
    text = (
        f"[bold cyan]Nightshift Edge AI Grid[/bold cyan] | "
        f"Total Chunks: [bold]{data.get('total_tasks', 0)}[/bold] | "
        f"[green]Completed: {counts.get('COMPLETED', 0)}[/green] | "
        f"[yellow]In-Flight (Leased): {counts.get('LEASED', 0)}[/yellow] | "
        f"[white]Pending: {counts.get('PENDING', 0)}[/white]"
    )
    if counts.get("FAILED"):
        text += f" | [red]Dead-lettered: {counts['FAILED']}[/red]"
    relay = data.get("relay") or {}
    if relay.get("goodput_pct") is not None:  # tokens in results / tokens decoded (Baton accounting)
        text += (f" | Goodput: [bold]{relay['goodput_pct']:.0f}%[/bold]"
                 + (f" (relay: {relay.get('resumes', 0)} resumes)" if relay.get("enabled") else ""))
    return text


def workers_table(data: dict, title: str = "Connected Heterogeneous Edge Nodes") -> Table:
    table = Table(expand=True, title=title)
    for col in ("Node ID", "Tier", "Model", "State", "Active Task", "Throughput", "RAM Free", "Done / Preempted"):
        table.add_column(col, style="bold" if col == "Node ID" else None)
    for w in data.get("workers", []):
        st = w.get("state", "?")
        color = STATE_COLORS.get(st, "white")
        table.add_row(
            w.get("worker_id", "?"),
            f"Tier {w.get('tier', '?')}",
            str(w.get("model_name", "?")),
            f"[{color}]{st}[/{color}]",
            (w.get("current_task_id") or "-")[:8],
            f"{w.get('tokens_per_sec', 0.0):.1f} tok/s",
            f"{w.get('ram_free_gb', 0.0):.1f} GB",
            f"{w.get('completed_count', 0)} / [yellow]{w.get('preempted_count', 0)}[/yellow]",
        )
    return table


def jobs_table(data: dict) -> Table:
    """Recent jobs from the status payload (which carries counts, not per-job attempt totals)."""
    table = Table(expand=True, title="Recent Jobs")
    for col in ("Job", "Name", "Completed", "In flight", "Dead-lettered", "Elapsed"):
        table.add_column(col)
    for j in data.get("jobs", []):
        counts = j.get("counts", {})
        elapsed = j.get("elapsed_s")
        table.add_row(
            str(j.get("job_id", "?")), str(j.get("job_name", ""))[:48],
            f"{counts.get('COMPLETED', 0)}/{j.get('total', 0)}" + (" [green]done[/green]" if j.get("done") else ""),
            str(counts.get("LEASED", 0)), str(counts.get("FAILED", 0)),
            f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "-",
        )
    return table


def render_dashboard(data: dict, footer: str = "") -> Layout:
    layout = Layout()
    parts = [Layout(name="header", size=4), Layout(name="main", ratio=1), Layout(name="events", size=14)]
    if footer:
        parts.append(Layout(Panel(footer), name="footer", size=footer.count("\n") + 3))
    layout.split_column(*parts)
    layout["header"].update(Panel(header_text(data), title="Grid MapReduce Telemetry"))
    layout["main"].update(workers_table(data))
    events_str = "\n".join(data.get("events", [])) or "Waiting for grid events..."
    layout["events"].update(Panel(events_str, title="Live Fault-Tolerance & Preemption Log"))
    return layout


async def dashboard_loop(url: str, token: Optional[str], *, should_stop: Optional[Callable[[], bool]] = None,
                         interval: float = 0.3, footer: str = "") -> None:
    """Poll `/v1/grid/status` and redraw until cancelled or `should_stop()` returns True."""
    hint = f"Set --token-file or {' / '.join(CLIENT_TOKEN_ENV)} (client token)."
    verify = await asyncio.to_thread(client_verify, url)
    async with httpx.AsyncClient(base_url=url, headers=bearer_headers(token), verify=verify) as client:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while not (should_stop and should_stop()):
                try:
                    resp = await client.get("/v1/grid/status", timeout=2.0)
                    check_auth(resp, hint)
                    live.update(render_dashboard(resp.json(), footer))
                except GridAuthError as e:
                    # Keep polling (the coordinator may be restarted with a matching token), but don't
                    # disguise a credentials problem as "waiting for the coordinator".
                    live.update(Panel(f"[bold red]Unauthorized[/bold red]\n{e}", title="Coordinator auth"))
                except Exception as e:
                    live.update(Panel(f"[red]Waiting for Coordinator at {url}... ({e})[/red]"))
                await asyncio.sleep(interval)


def run_dashboard(url: str = COORDINATOR_URL, token: Optional[str] = None, *,
                  should_stop: Optional[Callable[[], bool]] = None, footer: str = "") -> None:
    """Blocking TUI. Ctrl-C raises KeyboardInterrupt to the caller (who owns any cleanup)."""
    asyncio.run(dashboard_loop(url, token, should_stop=should_stop, footer=footer))


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)  # `--token X` must not mean --token-file
    ap.add_argument("--coordinator", default=COORDINATOR_URL)
    ap.add_argument("--token-file", default=None, metavar="PATH",
                    help=f"Client token file (else env {' / '.join(CLIENT_TOKEN_ENV)})")
    args = ap.parse_args(argv)
    try:
        token, _ = resolve_token(args.token_file, CLIENT_TOKEN_ENV)
    except GridAuthError as e:
        raise SystemExit(f"FATAL: {e}")
    try:
        run_dashboard(args.coordinator, token)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
