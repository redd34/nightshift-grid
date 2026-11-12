"""Client-side bearer-token resolution shared by the worker, dashboard and submit tool.

The coordinator (when configured) requires `Authorization: Bearer <token>`: a *worker* token on
`/worker/*` and a *client* token on `/v1/*`; `/healthz` is open. Tokens come from a file or the
environment, never from argv, because command lines are world-readable via `ps`.

Resolution precedence (first non-empty wins), for the worker token (client: `GRID_CLIENT_TOKEN`):

1. an explicit token file (`--token-file` / the coordinator's `--*-token-file`)
2. `$GRID_WORKER_TOKEN`        role-specific: lets one host hold different worker/client secrets
3. `$GRID_TOKEN`               shared by both roles (the original name)
4. `$NIGHTSHIFT_TOKEN_FILE`    shared, delivered as a file path (systemd/docker-secrets style)
5. `$NIGHTSHIFT_TOKEN`         shared, the product-name alias

Role-specific beats shared so that adding the alias to a host can never silently collapse a
deliberate worker/client split into one token. `GRID_TOKEN` beats `NIGHTSHIFT_TOKEN` so existing
deployments (whose installers write `GRID_*`) keep their behaviour if both happen to be set. The
`_FILE` form sits just above its value form, matching the usual container convention, and like
an explicit file it is a hard error when unreadable rather than a silent fallback.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import stat
import sys
from typing import Optional, Sequence

import httpx

SHARED_TOKEN_ENV = "NIGHTSHIFT_TOKEN"
TOKEN_FILE_ENV = "NIGHTSHIFT_TOKEN_FILE"  # consulted just before SHARED_TOKEN_ENV (see module docstring)
WORKER_TOKEN_ENV = ("GRID_WORKER_TOKEN", "GRID_TOKEN", SHARED_TOKEN_ENV)
CLIENT_TOKEN_ENV = ("GRID_CLIENT_TOKEN", "GRID_TOKEN", SHARED_TOKEN_ENV)


# Per-worker credentials --------------------------------------------------------------------
# A credential is `nsw1.<b64url(worker_id)>.<b64url(HMAC-SHA256(worker_token, label || worker_id))>`.
# The coordinator verifies it statelessly from the master worker token, and it authorizes exactly
# one worker_id: a credential copied off one office PC can't register, heartbeat or complete work
# as any other node, and it can't mint new ids. Only the coordinator host holds the master token.
CREDENTIAL_PREFIX = "nsw1"
_CREDENTIAL_LABEL = b"nightshift-worker-credential-v1\n"
MAX_WORKER_ID_CHARS = 128


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _check_worker_id(worker_id: str) -> None:
    if not worker_id or len(worker_id) > MAX_WORKER_ID_CHARS or any(ord(c) < 0x21 or ord(c) == 0x7F for c in worker_id):
        raise ValueError(f"invalid worker id {worker_id!r}: 1-{MAX_WORKER_ID_CHARS} printable non-space characters")


def derive_worker_credential(worker_token: str, worker_id: str) -> str:
    _check_worker_id(worker_id)
    mac = hmac.new(worker_token.encode(), _CREDENTIAL_LABEL + worker_id.encode(), hashlib.sha256).digest()
    return f"{CREDENTIAL_PREFIX}.{_b64e(worker_id.encode())}.{_b64e(mac)}"


def credential_worker_id(token: Optional[str]) -> Optional[str]:
    """The worker id a credential claims (unverified), or None if `token` isn't a credential."""
    if not token or not token.startswith(CREDENTIAL_PREFIX + "."):
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        wid = _b64d(parts[1]).decode("utf-8")
        _check_worker_id(wid)
    except (ValueError, UnicodeDecodeError):
        return None
    return wid


def verify_worker_credential(worker_token: str, token: str) -> Optional[str]:
    """The worker id `token` authorizes, or None if it isn't a valid credential for this grid."""
    wid = credential_worker_id(token)
    if wid is None:
        return None
    return wid if hmac.compare_digest(token.encode(), derive_worker_credential(worker_token, wid).encode()) else None


def write_secret_file(path: str, value: str, *, force: bool = False) -> None:
    """Write `value` to a new 0600 file (never through a symlink; refuses to overwrite unless forced)."""
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | (os.O_TRUNC if force else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value + "\n")
    os.chmod(path, 0o600)


class GridAuthError(RuntimeError):
    """The coordinator rejected our credentials (HTTP 401/403). Retrying cannot fix this."""


def _read_token_file(token_file: str) -> str:
    try:
        with open(token_file, encoding="utf-8") as f:
            token = f.read().strip()
        mode = os.stat(token_file).st_mode
    except OSError as e:
        raise GridAuthError(f"cannot read token file {token_file}: {e.strerror or e}") from e
    if not token:
        raise GridAuthError(f"token file {token_file} is empty")
    if os.name == "posix" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"[warn] token file {token_file} is readable by other users; chmod 600 it", file=sys.stderr)
    return token


def resolve_token(token_file: Optional[str], env_vars: Sequence[str]) -> tuple[Optional[str], str]:
    """Return (token, where it came from). Precedence: `token_file`, then `env_vars` in order, with
    `$NIGHTSHIFT_TOKEN_FILE` checked immediately before `NIGHTSHIFT_TOKEN` when that is in the chain.

    The source label is safe to log; the token never is. An explicitly given file (argument or
    `$NIGHTSHIFT_TOKEN_FILE`) that is missing, unreadable or empty is a hard error rather than a
    silent fallback to "no auth", which would otherwise surface much later as a confusing 401.
    """
    if token_file:
        return _read_token_file(token_file), f"file {token_file}"
    for var in env_vars:
        if var == SHARED_TOKEN_ENV:
            path = os.environ.get(TOKEN_FILE_ENV, "").strip()
            if path:
                return _read_token_file(path), f"file {path} (via ${TOKEN_FILE_ENV})"
        token = os.environ.get(var, "").strip()
        if token:
            return token, f"env {var}"
    return None, "none"


def bearer_headers(token: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def check_auth(response: httpx.Response, hint: str) -> None:
    """Raise GridAuthError for 401/403 with an actionable message (`hint` says how to set a token)."""
    if response.status_code in (401, 403):
        sent = "Authorization" in response.request.headers
        problem = "token rejected" if sent else "no token sent but the coordinator requires one"
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        if isinstance(detail, str) and detail and detail != "missing or invalid bearer token":
            problem += f" ({detail[:300]})"
        raise GridAuthError(f"coordinator returned HTTP {response.status_code} on "
                            f"{response.request.url.path}: {problem}. {hint}")


def main(argv: Optional[list[str]] = None) -> int:
    """`python grid_auth.py issue --id node-x --out node-x.credential` (run on the coordinator host)."""
    ap = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]) or "grid_auth.py", description="Issue per-worker credentials", allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    iss = sub.add_parser("issue", help="derive a credential valid for exactly one worker id")
    iss.add_argument("--id", required=True, help="worker id the credential is bound to")
    iss.add_argument("--worker-token-file", default=None,
                     help=f"master worker token (else env {' / '.join(WORKER_TOKEN_ENV)})")
    iss.add_argument("--out", required=True, help="credential file to create (mode 0600); '-' prints to stdout")
    iss.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    a = ap.parse_args(argv)
    try:
        master, _src = resolve_token(a.worker_token_file, WORKER_TOKEN_ENV)
        if not master:
            raise GridAuthError("no master worker token: pass --worker-token-file (e.g. .nightshift/worker.token)")
        if credential_worker_id(master):
            raise GridAuthError("that is a per-worker credential, not the coordinator's master worker token")
        cred = derive_worker_credential(master, a.id)
        if a.out == "-":
            print(cred)
        else:
            write_secret_file(a.out, cred, force=a.force)
            print(f"wrote credential for worker {a.id!r} to {a.out} (mode 0600). On that machine: "
                  f"worker_daemon.py --id {a.id} --token-file <it>", file=sys.stderr)
    except (GridAuthError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
