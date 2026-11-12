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

import os
import stat
import sys
from typing import Optional, Sequence

import httpx

SHARED_TOKEN_ENV = "NIGHTSHIFT_TOKEN"
TOKEN_FILE_ENV = "NIGHTSHIFT_TOKEN_FILE"  # consulted just before SHARED_TOKEN_ENV (see module docstring)
WORKER_TOKEN_ENV = ("GRID_WORKER_TOKEN", "GRID_TOKEN", SHARED_TOKEN_ENV)
CLIENT_TOKEN_ENV = ("GRID_CLIENT_TOKEN", "GRID_TOKEN", SHARED_TOKEN_ENV)


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
        raise GridAuthError(f"coordinator returned HTTP {response.status_code} on "
                            f"{response.request.url.path}: {problem}. {hint}")
