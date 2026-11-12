"""Submit a 24-chunk GraphRAG entity-extraction batch and (optionally) wait for completion."""
import argparse
import time
from typing import Optional

import httpx

from grid_auth import CLIENT_TOKEN_ENV, GridAuthError, bearer_headers, check_auth, resolve_token
from grid_tls import client_verify

SAMPLE_CHUNKS = [
    f"Equipment Log #{i}: Compressor turbine vibration exceeded 4.2 mm/s on PLC-Node-{i % 4} during shift handover. "
    f"Technician replaced bearing assembly B-{100 + i} and recalibrated Modbus register 40012."
    for i in range(1, 25)
]
AUTH_HINT = f"Set --token-file or {' / '.join(CLIENT_TOKEN_ENV)} (client token)."


def _headers(token: Optional[str]) -> dict[str, str]:
    """Explicit token wins; None falls back to the environment so library callers need no plumbing."""
    return bearer_headers(token if token is not None else resolve_token(None, CLIENT_TOKEN_ENV)[0])


def _tls(coordinator: str) -> dict:
    """httpx `verify=` for an https coordinator (pinned fingerprint / CA file from the environment)."""
    return {"verify": client_verify(coordinator)} if coordinator.lower().startswith("https://") else {}


def submit(coordinator: str, chunks=SAMPLE_CHUNKS, required_tier: int = 1,
           job_name: str = f"Industrial Maintenance GraphRAG Ingestion ({len(SAMPLE_CHUNKS)} Chunks)",
           token: Optional[str] = None) -> dict:
    r = httpx.post(f"{coordinator}/v1/grid/mapreduce", timeout=10, headers=_headers(token), **_tls(coordinator),
                   json={"job_name": job_name, "chunks": list(chunks), "required_tier": required_tier})
    check_auth(r, AUTH_HINT)
    r.raise_for_status()
    return r.json()


def wait(coordinator: str, job_id: str, timeout: float = 900, token: Optional[str] = None) -> dict:
    headers = _headers(token)
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{coordinator}/v1/grid/jobs/{job_id}", timeout=5, headers=headers, **_tls(coordinator))
        check_auth(r, AUTH_HINT)
        r.raise_for_status()
        st = r.json()
        if st["done"]:
            return st
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} not done after {timeout}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)  # `--token X` must not mean --token-file
    ap.add_argument("--coordinator", default="http://127.0.0.1:8000")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--wait", action="store_true", help="block until the job finishes and print a summary")
    ap.add_argument("--token-file", default=None, metavar="PATH",
                    help=f"Client token file (else env {' / '.join(CLIENT_TOKEN_ENV)})")
    a = ap.parse_args()
    try:
        tok, _ = resolve_token(a.token_file, CLIENT_TOKEN_ENV)
        job = submit(a.coordinator, required_tier=a.tier, token=tok)
        print("Submitted batch job:", job)
        if a.wait:
            st = wait(a.coordinator, job["job_id"], token=tok)
            print(f"Done in {st['elapsed_s']:.1f}s: {st['counts']} | attempts={st['total_attempts']} "
                  f"preemptions={st['total_preemptions']} failures={st['total_failures']}")
    except GridAuthError as e:
        raise SystemExit(f"FATAL: {e}")
