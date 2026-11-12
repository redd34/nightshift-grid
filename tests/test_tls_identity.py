"""TLS (self-signed cert, fingerprint pinning, CA file) and per-worker credentials."""
import json
import os
import socket
import stat
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import coordinator
import grid_auth
import grid_tls
from coordinator import GridState, create_app
from discovery import DiscoveryResponder, _check_reply, discover_coordinator_info, make_probe
from graphrag import NullSink
from grid_auth import credential_worker_id, derive_worker_credential, verify_worker_credential

MASTER = "master-worker-token-0123456789"
CLIENT = "client-token-0123456789"


# --------------------------------------------------------------------------- credentials
def test_credential_roundtrip_and_tamper():
    cred = derive_worker_credential(MASTER, "node-a")
    assert cred.startswith("nsw1.") and credential_worker_id(cred) == "node-a"
    assert verify_worker_credential(MASTER, cred) == "node-a"
    assert verify_worker_credential("other-master", cred) is None
    # Swap in another id but keep node-a's MAC: must not verify.
    forged = derive_worker_credential(MASTER, "node-b").split(".")[0:2] + [cred.split(".")[2]]
    assert verify_worker_credential(MASTER, ".".join(forged)) is None
    assert credential_worker_id(MASTER) is None and verify_worker_credential(MASTER, MASTER) is None
    for bad in ("", "has space", "x" * 129):
        with pytest.raises(ValueError):
            derive_worker_credential(MASTER, bad)


def test_issue_cli_writes_0600_and_refuses_overwrite(tmp_path, monkeypatch):
    master = tmp_path / "worker.token"
    master.write_text(MASTER)
    os.chmod(master, 0o600)
    out = tmp_path / "n.credential"
    assert grid_auth.main(["issue", "--id", "node-a", "--worker-token-file", str(master), "--out", str(out)]) == 0
    assert verify_worker_credential(MASTER, out.read_text().strip()) == "node-a"
    if os.name == "posix":
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert grid_auth.main(["issue", "--id", "node-a", "--worker-token-file", str(master), "--out", str(out)]) == 1
    # A credential can't be used to mint further credentials.
    assert grid_auth.main(["issue", "--id", "node-b", "--worker-token-file", str(out), "--out", "-"]) == 1


def _app(allow_shared: bool):
    return TestClient(create_app(GridState(sink=NullSink()), worker_token=MASTER, client_token=CLIENT,
                                 allow_shared_worker_token=allow_shared))


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_per_worker_credential_is_bound_to_its_id():
    c = _app(allow_shared=False)
    a = derive_worker_credential(MASTER, "node-a")
    r = c.post("/worker/register", json={"worker_id": "node-a"}, headers=_auth(a))
    assert r.status_code == 200
    session = r.json()["session"]
    # node-a's credential can't register, pull, heartbeat or complete as anyone else.
    assert c.post("/worker/register", json={"worker_id": "node-new"}, headers=_auth(a)).status_code == 403
    hdr = {**_auth(a), "X-Nightshift-Session": session}
    assert c.post("/worker/pull-task", params={"worker_id": "node-b"}, headers=hdr).status_code == 403
    assert c.post("/worker/heartbeat", json={"worker_id": "node-b"}, headers=hdr).status_code == 403
    assert c.post("/worker/complete", headers=hdr, json={"worker_id": "node-b", "task_id": "t", "lease_id": "l",
                                                         "result": {}}).status_code in (403, 422)
    assert c.post("/worker/pull-task", params={"worker_id": "node-a"}, headers=hdr).status_code == 200


def test_strict_mode_rejects_master_token_with_actionable_message():
    c = _app(allow_shared=False)
    r = c.post("/worker/register", json={"worker_id": "node-a"}, headers=_auth(MASTER))
    assert r.status_code == 401 and "per-worker credentials" in r.json()["detail"]
    assert c.post("/worker/register", json={"worker_id": "node-a"},
                  headers=_auth(derive_worker_credential("wrong", "node-a"))).status_code == 401
    # The client token is still a separate role.
    assert c.post("/worker/register", json={"worker_id": "node-a"}, headers=_auth(CLIENT)).status_code == 401


def test_shared_mode_accepts_master_and_credentials():
    c = _app(allow_shared=True)
    assert c.post("/worker/register", json={"worker_id": "anything"}, headers=_auth(MASTER)).status_code == 200
    cred = derive_worker_credential(MASTER, "node-a")
    assert c.post("/worker/register", json={"worker_id": "node-a"}, headers=_auth(cred)).status_code == 200
    assert c.post("/worker/register", json={"worker_id": "node-z"}, headers=_auth(cred)).status_code == 403


def test_worker_takes_id_from_credential_and_rejects_mismatch(tmp_path, monkeypatch):
    import worker_daemon
    cred = tmp_path / "c"
    cred.write_text(derive_worker_credential(MASTER, "node-cred"))
    os.chmod(cred, 0o600)
    base = ["--mock", "--allow-battery", "--token-file", str(cred)]
    w = worker_daemon.PreemptibleEdgeWorker(worker_daemon.parse_args(base))
    assert w.worker_id == "node-cred"
    with pytest.raises(grid_auth.GridAuthError, match="does not match"):
        worker_daemon.PreemptibleEdgeWorker(worker_daemon.parse_args([*base, "--id", "node-other"]))


# --------------------------------------------------------------------------- coordinator CLI policy
@pytest.fixture
def run_main(monkeypatch):
    import uvicorn
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **kw: captured.update(app=app, host=host, **kw))
    for k in ("GRID_TOKEN", "GRID_WORKER_TOKEN", "GRID_CLIENT_TOKEN", "NIGHTSHIFT_TOKEN", "NIGHTSHIFT_TOKEN_FILE"):
        monkeypatch.delenv(k, raising=False)

    def run(argv):
        captured.clear()
        coordinator.main(argv)
        return captured
    return run


def test_lan_bind_refuses_plaintext(tmp_path, run_main):
    with pytest.raises(SystemExit, match="--auto-tls"):
        run_main(["--host", "0.0.0.0", "--db", str(tmp_path / "g.db"), "--output", "none", "--auto-token",
                  "--no-advertise"])


def test_lan_bind_defaults_to_per_worker_credentials(tmp_path, run_main):
    db = tmp_path / "g.db"
    cap = run_main(["--host", "0.0.0.0", "--db", str(db), "--output", "none", "--auto-token", "--auto-tls",
                    "--no-advertise"])
    master = (tmp_path / "worker.token").read_text().strip()
    c = TestClient(cap["app"])
    assert c.post("/worker/register", json={"worker_id": "n"}, headers=_auth(master)).status_code == 401
    assert c.post("/worker/register", json={"worker_id": "n"},
                  headers=_auth(derive_worker_credential(master, "n"))).status_code == 200
    cap["app"].state.grid.store.close()


def test_auto_tls_reuses_certificate(tmp_path, run_main, capsys):
    argv = ["--host", "0.0.0.0", "--db", str(tmp_path / "g.db"), "--output", "none", "--auto-token", "--auto-tls",
            "--no-advertise"]
    run_main(argv)["app"].state.grid.store.close()
    fp1 = grid_tls.fingerprint_file(tmp_path / "coordinator-cert.pem")
    run_main(argv)["app"].state.grid.store.close()
    assert grid_tls.fingerprint_file(tmp_path / "coordinator-cert.pem") == fp1
    err = capsys.readouterr().err
    assert "minted self-signed" in err and "reusing self-signed" in err and fp1 in err
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "coordinator-key.pem").stat().st_mode) == 0o600


def test_auto_tls_refuses_world_readable_key(tmp_path):
    grid_tls.ensure_self_signed(tmp_path, ["localhost"])
    os.chmod(tmp_path / "coordinator-key.pem", 0o644)
    with pytest.raises(grid_tls.TLSConfigError, match="0600"):
        grid_tls.ensure_self_signed(tmp_path, ["localhost"])


# --------------------------------------------------------------------------- live HTTPS server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def https_server(tmp_path_factory):
    import uvicorn
    d = tmp_path_factory.mktemp("tls")
    cert, key, fp, _ = grid_tls.ensure_self_signed(d, ["localhost", "127.0.0.1"])
    port = _free_port()
    app = create_app(GridState(sink=NullSink()), worker_token=MASTER, client_token=CLIENT,
                     allow_shared_worker_token=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error",
                                           ssl_certfile=str(cert), ssl_keyfile=str(key)))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"https://127.0.0.1:{port}", fp, cert
    server.should_exit = True
    t.join(5)


def test_pinned_fingerprint_connects(https_server):
    url, fp, _ = https_server
    grid_tls._CONTEXT_CACHE.clear()
    verify = grid_tls.client_verify(url, fingerprint=fp.upper().replace("SHA256:", "sha256:"))
    r = httpx.get(f"{url}/healthz", verify=verify)
    assert r.status_code == 200
    cred = derive_worker_credential(MASTER, "tls-node")
    r = httpx.post(f"{url}/worker/register", json={"worker_id": "tls-node"}, headers=_auth(cred), verify=verify)
    assert r.status_code == 200


def test_wrong_fingerprint_is_refused_before_any_request(https_server):
    url, _, _ = https_server
    grid_tls._CONTEXT_CACHE.clear()
    with pytest.raises(grid_tls.TLSConfigError, match="mismatch"):
        grid_tls.client_verify(url, fingerprint="sha256:" + "0" * 64)


def test_ca_file_mode_and_default_trust(https_server, monkeypatch):
    url, _, cert = https_server
    grid_tls._CONTEXT_CACHE.clear()
    assert httpx.get(f"{url}/healthz", verify=grid_tls.client_verify(url, ca_file=str(cert))).status_code == 200
    monkeypatch.setenv(grid_tls.CA_FILE_ENV, str(cert))
    assert httpx.get(f"{url}/healthz", verify=grid_tls.client_verify(url)).status_code == 200
    monkeypatch.delenv(grid_tls.CA_FILE_ENV)
    with pytest.raises(httpx.ConnectError):  # self-signed: the system trust store rejects it
        httpx.get(f"{url}/healthz", verify=grid_tls.client_verify(url))


def test_pin_on_plain_http_is_a_config_error():
    with pytest.raises(grid_tls.TLSConfigError):
        grid_tls.client_verify("http://127.0.0.1:1", fingerprint="sha256:" + "a" * 64)
    with pytest.raises(grid_tls.TLSConfigError):
        grid_tls.normalize_fingerprint("sha256:abc")


# --------------------------------------------------------------------------- discovery
FP = "sha256:" + "ab" * 32


def test_discovery_delivers_authenticated_fingerprint_to_credential_holder():
    import asyncio

    async def go():
        r = await DiscoveryResponder("https://127.0.0.1:8443", MASTER, bind_host="127.0.0.1", port=0,
                                     tls_fingerprint=FP).start()
        try:
            cred = derive_worker_credential(MASTER, "node-d")
            found = await asyncio.to_thread(discover_coordinator_info, cred, timeout=0.6, port=r.port,
                                            targets=["127.0.0.1"])
            assert found.verified and found.tls_fingerprint == FP and found.url == "https://127.0.0.1:8443"
            with pytest.raises(Exception, match="did not prove"):
                await asyncio.to_thread(discover_coordinator_info, derive_worker_credential("x", "node-d"),
                                        timeout=0.4, port=r.port, targets=["127.0.0.1"])
        finally:
            r.close()
    asyncio.run(go())


def test_tampered_fingerprint_fails_the_proof():
    r = DiscoveryResponder("https://10.0.0.5:8443", MASTER, tls_fingerprint=FP)
    nonce = "cd" * 16
    cred = derive_worker_credential(MASTER, "node-e")
    reply = json.loads(r.build_reply(make_probe(nonce, "node-e"), "10.0.0.9"))
    assert isinstance(_check_reply(json.dumps(reply).encode(), "10.0.0.5", nonce, cred), object)
    assert _check_reply(json.dumps(reply).encode(), "10.0.0.5", nonce, cred).tls_fingerprint == FP
    reply["tls"] = "sha256:" + "ef" * 32  # an attacker swaps in their own certificate's pin
    assert "did not prove" in _check_reply(json.dumps(reply).encode(), "10.0.0.5", nonce, cred)
