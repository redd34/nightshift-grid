"""--auto-token, discovery/qdrant CLI wiring in coordinator.main (uvicorn stubbed out)."""
import io
import os
import stat

import pytest

import coordinator
from coordinator import ensure_auto_tokens

TOKEN_ENVS = ("GRID_TOKEN", "GRID_WORKER_TOKEN", "GRID_CLIENT_TOKEN", "NIGHTSHIFT_TOKEN", "NIGHTSHIFT_TOKEN_FILE")


class TTY(io.StringIO):
    def isatty(self): return True


@pytest.fixture(autouse=True)
def no_env_tokens(monkeypatch):
    for k in TOKEN_ENVS:
        monkeypatch.delenv(k, raising=False)


def test_creates_both_tokens_0600_and_prints_once_on_tty(tmp_path):
    out, err = TTY(), io.StringIO()
    w, c, ws, cs = ensure_auto_tokens(tmp_path, None, None, stdout=out, stderr=err)
    assert w != c and len(w) >= 40 and len(c) >= 40
    for role, tok in (("worker", w), ("client", c)):
        p = tmp_path / f"{role}.token"
        assert p.read_text().strip() == tok
        if os.name == "posix":
            assert stat.S_IMODE(p.stat().st_mode) == 0o600
        assert str(p) in err.getvalue()
    assert w in out.getvalue() and c in out.getvalue()        # shown once on a TTY...
    assert w not in err.getvalue() and c not in err.getvalue()  # ...and never in the (loggable) stderr
    assert "(new)" in ws and "(new)" in cs


def test_reuses_existing_tokens_and_does_not_reprint(tmp_path):
    w1, c1, *_ = ensure_auto_tokens(tmp_path, None, None, stdout=io.StringIO(), stderr=io.StringIO())
    out = TTY()
    w2, c2, ws, cs = ensure_auto_tokens(tmp_path, None, None, stdout=out, stderr=io.StringIO())
    assert (w2, c2) == (w1, c1) and "(reused)" in ws and "(reused)" in cs
    assert out.getvalue() == ""


def test_no_tty_never_prints_tokens(tmp_path):
    out, err = io.StringIO(), io.StringIO()   # StringIO.isatty() is False: like a pipe / log file
    w, c, *_ = ensure_auto_tokens(tmp_path, None, None, stdout=out, stderr=err)
    assert out.getvalue() == ""
    assert w not in err.getvalue() and c not in err.getvalue()
    assert "worker.token" in err.getvalue()


def test_only_missing_role_is_generated(tmp_path):
    w, c, ws, cs = ensure_auto_tokens(tmp_path, "configured-worker", None, stdout=io.StringIO(), stderr=io.StringIO())
    assert w == "configured-worker" and ws == "configured"
    assert not (tmp_path / "worker.token").exists() and (tmp_path / "client.token").read_text().strip() == c


def test_empty_existing_file_is_an_error(tmp_path):
    (tmp_path / "worker.token").write_text("")
    with pytest.raises(coordinator.GridAuthError, match="empty"):
        ensure_auto_tokens(tmp_path, None, None, stdout=io.StringIO(), stderr=io.StringIO())


# ---------------------------------------------------------------- main() wiring
@pytest.fixture
def run_main(monkeypatch):
    """Call coordinator.main(argv) with uvicorn.run captured instead of serving."""
    import uvicorn
    captured = {}

    def fake_run(app, host, port, **kw):
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    def run(argv):
        captured.clear()
        coordinator.main(argv)
        return captured
    return run


def test_lan_bind_refusal_suggests_auto_token(tmp_path):
    with pytest.raises(SystemExit, match="--auto-token"):
        coordinator.main(["--host", "0.0.0.0", "--db", str(tmp_path / "g.db"), "--output", "none"])


def test_lan_bind_with_auto_token_starts_and_advertises(tmp_path, run_main, capsys):
    db = tmp_path / "state" / "g.db"
    db.parent.mkdir()
    cap = run_main(["--host", "0.0.0.0", "--port", "8845", "--db", str(db), "--output", "none", "--auto-token",
                    "--discovery-port", "0"])
    assert cap["host"] == "0.0.0.0"
    assert (db.parent / "worker.token").exists() and (db.parent / "client.token").exists()
    err = capsys.readouterr().err
    assert "discovery: advertising http://" in err and ":8845" in err
    cap["app"].state.grid.store.close()


def test_loopback_default_does_not_advertise(tmp_path, run_main, capsys):
    run_main(["--db", "none", "--output", "none"])
    assert "discovery" not in capsys.readouterr().err


def test_no_advertise_flag(tmp_path, run_main, capsys):
    run_main(["--host", "0.0.0.0", "--db", "none", "--output", "none", "--auto-token", "--no-advertise"])
    assert "discovery" not in capsys.readouterr().err


def test_bad_qdrant_collection_is_rejected_before_serving(tmp_path, run_main):
    with pytest.raises(SystemExit, match="nightshift_"):
        run_main(["--db", "none", "--output", "none", "--qdrant-memory", "--qdrant-collection", "venues"])


def test_qdrant_memory_wiring_exposes_sink_stats(tmp_path, run_main):
    pytest.importorskip("qdrant_client")
    from fastapi.testclient import TestClient

    cap = run_main(["--db", str(tmp_path / "g.db"), "--output", str(tmp_path / "o.jsonl"), "--qdrant-memory",
                    "--qdrant-embedder", "hash", "--qdrant-collection", "nightshift_cli_test"])
    with TestClient(cap["app"]) as c:
        st = c.get("/v1/grid/status").json()
    assert st["sink"]["qdrant"]["collection"] == "nightshift_cli_test"
    assert st["sink"]["qdrant"]["embedder"] == "hash-v1:768"
    assert st["sink"]["primary"] == "JsonlGraphSink"


def test_discovery_responder_runs_in_lifespan(tmp_path):
    """End to end in-process: the app's lifespan starts the responder and a worker finds it."""
    from fastapi.testclient import TestClient
    from discovery import DiscoveryResponder, discover_coordinator

    r = DiscoveryResponder("http://127.0.0.1:8846", "wtok", bind_host="127.0.0.1", port=0)
    app = coordinator.create_app(coordinator.GridState(), worker_token="wtok", client_token="ctok", discovery=r)
    with TestClient(app):
        assert discover_coordinator("wtok", timeout=0.5, port=r.port, targets=["127.0.0.1"]) == "http://127.0.0.1:8846"
    assert r.transport is None  # closed on shutdown
