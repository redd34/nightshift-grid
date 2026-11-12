"""Client-side auth: token resolution precedence, headers on the right client, fatal 401s."""
import asyncio
import os

import httpx
import pytest

import demo_submit
import worker_daemon
from grid_auth import CLIENT_TOKEN_ENV, TOKEN_FILE_ENV, WORKER_TOKEN_ENV, GridAuthError, check_auth, resolve_token

ALL_ENV = set(WORKER_TOKEN_ENV) | set(CLIENT_TOKEN_ENV) | {TOKEN_FILE_ENV}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ALL_ENV:
        monkeypatch.delenv(var, raising=False)


def test_resolution_precedence(tmp_path, monkeypatch):
    f = tmp_path / "tok"
    f.write_text("  file-secret \n")
    f.chmod(0o600)
    monkeypatch.setenv("GRID_TOKEN", "shared")
    assert resolve_token(None, WORKER_TOKEN_ENV) == ("shared", "env GRID_TOKEN")
    monkeypatch.setenv("GRID_WORKER_TOKEN", "worker")
    assert resolve_token(None, WORKER_TOKEN_ENV) == ("worker", "env GRID_WORKER_TOKEN")
    assert resolve_token(None, CLIENT_TOKEN_ENV) == ("shared", "env GRID_TOKEN")  # worker token isn't a client token
    assert resolve_token(str(f), WORKER_TOKEN_ENV) == ("file-secret", f"file {f}")


def test_nightshift_token_alias_precedence(monkeypatch):
    assert WORKER_TOKEN_ENV[-1] == CLIENT_TOKEN_ENV[-1] == "NIGHTSHIFT_TOKEN"
    monkeypatch.setenv("NIGHTSHIFT_TOKEN", "ns")
    assert resolve_token(None, WORKER_TOKEN_ENV) == ("ns", "env NIGHTSHIFT_TOKEN")
    assert resolve_token(None, CLIENT_TOKEN_ENV) == ("ns", "env NIGHTSHIFT_TOKEN")
    monkeypatch.setenv("GRID_TOKEN", "grid")                 # existing name keeps priority
    assert resolve_token(None, WORKER_TOKEN_ENV) == ("grid", "env GRID_TOKEN")
    monkeypatch.setenv("GRID_CLIENT_TOKEN", "client")         # role-specific beats both shared names
    assert resolve_token(None, CLIENT_TOKEN_ENV) == ("client", "env GRID_CLIENT_TOKEN")


def test_nightshift_token_file_env(tmp_path, monkeypatch):
    f = tmp_path / "ns.token"
    f.write_text("from-file\n")
    f.chmod(0o600)
    monkeypatch.setenv("NIGHTSHIFT_TOKEN", "value")
    monkeypatch.setenv(TOKEN_FILE_ENV, str(f))
    assert resolve_token(None, WORKER_TOKEN_ENV) == ("from-file", f"file {f} (via $NIGHTSHIFT_TOKEN_FILE)")
    monkeypatch.setenv("GRID_WORKER_TOKEN", "w")              # role-specific value still wins
    assert resolve_token(None, WORKER_TOKEN_ENV)[0] == "w"
    other = tmp_path / "flag.token"
    other.write_text("flag")
    other.chmod(0o600)
    assert resolve_token(str(other), WORKER_TOKEN_ENV)[0] == "flag"   # explicit --token-file wins
    assert resolve_token(None, ("GRID_TOKEN",)) == (None, "none")      # file env only joins chains with the alias
    monkeypatch.setenv(TOKEN_FILE_ENV, str(tmp_path / "missing"))
    monkeypatch.delenv("GRID_WORKER_TOKEN")
    with pytest.raises(GridAuthError, match="cannot read"):            # explicit file: no silent fallback
        resolve_token(None, WORKER_TOKEN_ENV)


def test_resolution_none_and_blank_env(monkeypatch):
    monkeypatch.setenv("GRID_CLIENT_TOKEN", "   ")
    assert resolve_token(None, CLIENT_TOKEN_ENV) == (None, "none")


def test_bad_token_files_are_errors(tmp_path):
    with pytest.raises(GridAuthError, match="cannot read"):
        resolve_token(str(tmp_path / "missing"), WORKER_TOKEN_ENV)
    empty = tmp_path / "empty"
    empty.write_text("\n")
    with pytest.raises(GridAuthError, match="empty"):
        resolve_token(str(empty), WORKER_TOKEN_ENV)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_world_readable_token_file_warns(tmp_path, capsys):
    f = tmp_path / "tok"
    f.write_text("x")
    f.chmod(0o644)
    resolve_token(str(f), WORKER_TOKEN_ENV)
    assert "chmod 600" in capsys.readouterr().err


def test_check_auth_messages():
    req = httpx.Request("GET", "http://c/v1/grid/status")
    with pytest.raises(GridAuthError, match="no token sent"):
        check_auth(httpx.Response(401, request=req), "hint")
    req = httpx.Request("GET", "http://c/v1/grid/status", headers={"Authorization": "Bearer x"})
    with pytest.raises(GridAuthError, match="token rejected"):
        check_auth(httpx.Response(401, request=req), "hint")
    check_auth(httpx.Response(200, request=req), "hint")


def mock_module_http(monkeypatch, handler):
    """Route demo_submit's module-level httpx.post/get through a MockTransport."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(demo_submit.httpx, "post", client.post)
    monkeypatch.setattr(demo_submit.httpx, "get", client.get)


def test_submit_and_wait_send_client_token(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        if request.url.path.endswith("/mapreduce"):
            return httpx.Response(200, json={"job_id": "j1"})
        return httpx.Response(200, json={"done": True})

    mock_module_http(monkeypatch, handler)
    monkeypatch.setenv("GRID_CLIENT_TOKEN", "env-client")
    assert demo_submit.submit("http://c", job_name="t")["job_id"] == "j1"   # old signature still works
    demo_submit.wait("http://c", "j1", 5, token="explicit")
    assert seen == ["Bearer env-client", "Bearer explicit"]


def test_submit_401_raises_auth_error(monkeypatch):
    mock_module_http(monkeypatch, lambda r: httpx.Response(401, json={"detail": "unauthorized"}))
    with pytest.raises(GridAuthError, match="401"):
        demo_submit.submit("http://c")


def make_worker(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(worker_daemon, "TRIGGER_DIR", str(tmp_path))
    args = worker_daemon.parse_args(["--id", "node-auth-test", "--mock", "--allow-battery", "--ignore-thermal",
                                     "--coordinator", "http://coord.test", *extra])
    return worker_daemon.PreemptibleEdgeWorker(args)


def worker_daemon_fatal_logged(capsys) -> bool:
    out = capsys.readouterr().out
    return "FATAL" in out and "401" in out and "/worker/register" in out


def test_worker_401_is_fatal_and_header_only_on_coordinator(monkeypatch, tmp_path, capsys):
    f = tmp_path / "tok"
    f.write_text("worker-secret\n")
    f.chmod(0o600)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path, request.headers.get("Authorization")))
        return httpx.Response(401, json={"detail": "bad token"})

    real_client = httpx.AsyncClient

    def client(*a, **k):
        return real_client(*a, transport=httpx.MockTransport(handler), **k)
    monkeypatch.setattr(worker_daemon.httpx, "AsyncClient", client)

    monkeypatch.setattr(worker_daemon, "TRIGGER_DIR", str(tmp_path))
    args = worker_daemon.parse_args(["--id", "node-auth-test", "--mock", "--allow-battery", "--ignore-thermal",
                                     "--coordinator", "http://coord.test", "--token-file", str(f)])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(asyncio.wait_for(worker_daemon._amain(args), 5))
    assert exc.value.code == 1 and worker_daemon_fatal_logged(capsys)
    # Registered once, gave up immediately (no retry loop), never sent a shutdown hand-back.
    assert seen == [("coord.test", "/worker/register", "Bearer worker-secret")]


def test_worker_has_no_token_flag():
    with pytest.raises(SystemExit):
        worker_daemon.parse_args(["--id", "x", "--token", "secret"])


def test_ollama_client_never_gets_token(monkeypatch, tmp_path):
    monkeypatch.setenv("GRID_WORKER_TOKEN", "w")
    created = []
    real_client = httpx.AsyncClient

    def client(*a, **k):
        created.append((k.get("base_url"), dict(k.get("headers") or {})))
        return real_client(*a, transport=httpx.MockTransport(lambda r: httpx.Response(401)), **k)
    monkeypatch.setattr(worker_daemon.httpx, "AsyncClient", client)
    worker = make_worker(monkeypatch, tmp_path)
    with pytest.raises(GridAuthError):
        asyncio.run(worker.run())
    headers = dict(created)
    assert headers["http://coord.test"] == {"Authorization": "Bearer w"}
    assert "Authorization" not in headers[worker.ollama_url]
