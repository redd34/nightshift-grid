"""UDP discovery: real sockets over loopback on ephemeral ports, plus pure responder logic."""
import asyncio
import json
import socket
import threading
from contextlib import contextmanager

import pytest

import discovery as d
from discovery import DiscoveryError, DiscoveryResponder, RateLimiter, discover_all, discover_coordinator


@contextmanager
def responders(*specs):
    """Run DiscoveryResponders (url, token) on a private event loop thread; yield their ports."""
    loop = asyncio.new_event_loop()
    started = threading.Event()
    rs = [DiscoveryResponder(url, tok, bind_host="127.0.0.1", port=0) for url, tok in specs]

    def run():
        asyncio.set_event_loop(loop)
        for r in rs:
            loop.run_until_complete(r.start())
        started.set()
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert started.wait(5)
    try:
        yield rs
    finally:
        for r in rs:
            loop.call_soon_threadsafe(r.close)
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()


def test_roundtrip_with_matching_token():
    with responders(("http://10.0.0.5:8000", "tok-A")) as (r,):
        url = discover_coordinator("tok-A", timeout=0.6, port=r.port, targets=["127.0.0.1"])
    assert url == "http://10.0.0.5:8000"
    assert r.stats["replies"] >= 1


def test_wrong_token_rejected_and_reported():
    with responders(("http://10.0.0.5:8000", "real-token")) as (r,):
        with pytest.raises(DiscoveryError, match="did not prove knowledge"):
            discover_coordinator("other-token", timeout=0.6, port=r.port, targets=["127.0.0.1"])


def test_worker_with_token_rejects_unauthenticated_coordinator():
    """A rogue (tokenless) responder must not be accepted by a worker that has a token."""
    with responders(("http://6.6.6.6:8000", None)) as (r,):
        with pytest.raises(DiscoveryError):
            discover_coordinator("tok", timeout=0.6, port=r.port, targets=["127.0.0.1"])


def test_tokenless_worker_accepts_but_flags(caplog):
    with responders(("http://10.0.0.7:8000", "tok")) as (r,):
        with caplog.at_level("WARNING", logger="nightshift.discovery"):
            url = discover_coordinator(None, timeout=0.6, port=r.port, targets=["127.0.0.1"])
    assert url == "http://10.0.0.7:8000"
    assert "UNVERIFIED" in caplog.text and "requires a worker token" in caplog.text
    found, _ = discover_all(None, timeout=0.3, port=r.port, targets=["127.0.0.1"])  # responder gone
    assert found == []


def test_none_found():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    try:  # a bound-but-silent port: probe is swallowed, nothing answers
        with pytest.raises(DiscoveryError, match="no Nightshift coordinator"):
            discover_coordinator("tok", timeout=0.3, port=port, targets=["127.0.0.1"])
    finally:
        s.close()


def test_ambiguity_raises():
    """Two coordinators that both prove the token: refusing to guess is the only safe answer."""
    with responders(("http://10.0.0.1:8000", "tok"), ("http://10.0.0.2:8000", "tok")) as (r1, r2):
        targets = [("127.0.0.1", r1.port), ("127.0.0.1", r2.port)]
        found, _ = discover_all("tok", timeout=0.6, targets=targets, retransmits=0)
        assert sorted(c.url for c in found) == ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
        assert all(c.verified for c in found)
        with pytest.raises(DiscoveryError, match="2 coordinators answered.*10.0.0.1.*10.0.0.2|2 coordinators answered.*10.0.0.2.*10.0.0.1"):
            discover_coordinator("tok", timeout=0.6, targets=targets)


def test_one_valid_among_rogues_is_chosen():
    """A rogue responder without the token doesn't make discovery ambiguous; it's ignored."""
    with responders(("http://10.0.0.1:8000", "tok"), ("http://6.6.6.6:8000", None),
                    ("http://6.6.6.7:8000", "guess")) as rs:
        targets = [("127.0.0.1", r.port) for r in rs]
        assert discover_coordinator("tok", timeout=0.6, targets=targets) == "http://10.0.0.1:8000"


def test_duplicate_replies_from_one_coordinator_collapse():
    """Several probe copies (retransmits / several broadcast targets) -> one coordinator."""
    with responders(("http://10.0.0.5:8000", "tok")) as (r,):
        found, _ = discover_all("tok", timeout=0.6, port=r.port, targets=["127.0.0.1", "127.0.0.1"], retransmits=2)
    assert [c.url for c in found] == ["http://10.0.0.5:8000"]
    assert r.stats["replies"] >= 2


def test_forged_or_replayed_replies_rejected():
    nonce = "ab" * 16
    good = json.dumps({"v": 1, "type": d.REPLY_TYPE, "url": "http://1.2.3.4:8000", "nonce": nonce,
                       "proof": d.compute_proof("tok", nonce, "http://1.2.3.4:8000"), "auth": True}).encode()
    assert d._check_reply(good, "x", nonce, "tok").url == "http://1.2.3.4:8000"
    assert "nonce" in d._check_reply(good, "x", "cd" * 16, "tok")                     # replay to a new probe
    swapped = json.loads(good); swapped["url"] = "http://6.6.6.6:8000"                 # relay with URL swapped
    assert "did not prove" in d._check_reply(json.dumps(swapped).encode(), "x", nonce, "tok")
    nulled = json.loads(good); nulled["proof"] = None
    assert "did not prove" in d._check_reply(json.dumps(nulled).encode(), "x", nonce, "tok")
    bad_url = json.loads(good); bad_url["url"] = "file:///etc/passwd"
    assert "invalid url" in d._check_reply(json.dumps(bad_url).encode(), "x", nonce, "tok")
    assert d._check_reply(b"\xff\xfe garbage", "x", nonce, "tok") is None
    assert d._check_reply(d.make_probe(nonce), "x", nonce, "tok") is None              # our own broadcast echo


# ---------------------------------------------------------------- responder hardening (pure)
def responder(**kw):
    return DiscoveryResponder("http://192.168.1.10:8000", "tok", **kw)


@pytest.mark.parametrize("packet", [
    b"", b"not json", b"[1,2,3]", b"\xff" * 300,
    json.dumps({"v": 2, "type": d.PROBE_TYPE, "nonce": "ab" * 16, "pad": "0" * 400}).encode(),
    json.dumps({"v": 1, "type": "other", "nonce": "ab" * 16, "pad": "0" * 400}).encode(),
    json.dumps({"v": 1, "type": d.PROBE_TYPE, "nonce": "zz" * 16, "pad": "0" * 400}).encode(),
    json.dumps({"v": 1, "type": d.PROBE_TYPE, "nonce": "ab", "pad": "0" * 400}).encode(),
    json.dumps({"v": 1, "type": d.PROBE_TYPE, "nonce": 12345, "pad": "0" * 400}).encode(),
    json.dumps({"v": 1, "type": d.PROBE_TYPE, "nonce": "ab" * 16, "pad": "0" * 2000}).encode(),  # oversized
])
def test_malformed_probes_get_silence(packet):
    r = responder()
    assert r.build_reply(packet, "10.0.0.9") is None
    assert r.stats["replies"] == 0


def test_unpadded_probe_ignored_and_reply_within_amplification_cap():
    r = responder()
    small = json.dumps({"v": 1, "type": d.PROBE_TYPE, "nonce": "ab" * 16}).encode()
    assert r.build_reply(small, "10.0.0.9") is None and r.stats["dropped_small"] == 1
    probe = d.make_probe("ab" * 16)
    assert len(probe) == d.PROBE_PAD_TO
    reply = r.build_reply(probe, "10.0.0.9")
    assert reply and len(reply) <= d.MAX_AMPLIFICATION * len(probe)
    msg = json.loads(reply)
    assert msg["proof"] == d.compute_proof("tok", "ab" * 16, msg["url"]) and msg["auth"] is True


def test_reply_never_exceeds_cap_even_with_long_url():
    r = DiscoveryResponder("http://" + "h" * 230 + ":8000", "tok")
    minimal = d._encode({"v": 1, "type": d.PROBE_TYPE, "nonce": "ab" * 32, "pad": "0" * 160})
    assert len(minimal) >= d.MIN_PROBE_BYTES
    reply = r.build_reply(minimal, "10.0.0.9")
    assert reply is None or len(reply) <= 2 * len(minimal)


def test_rate_limit_per_source_and_global():
    t = [0.0]
    r = responder(limiter=RateLimiter(per_source_rate=1.0, per_source_burst=3, global_rate=100, global_burst=5,
                                      clock=lambda: t[0]))
    probe = d.make_probe("ab" * 16)
    assert [bool(r.build_reply(probe, "10.0.0.1")) for _ in range(5)] == [True, True, True, False, False]
    assert r.build_reply(probe, "10.0.0.2") and r.build_reply(probe, "10.0.0.3")   # other sources: own buckets
    assert r.build_reply(probe, "10.0.0.4") is None                                # global burst (5) exhausted
    assert r.stats["dropped_rate"] == 3
    t[0] += 1.0                                                                     # refill
    assert r.build_reply(probe, "10.0.0.1")


def test_rate_limiter_source_table_is_bounded():
    t = [0.0]
    rl = RateLimiter(per_source_rate=1.0, per_source_burst=2, global_rate=1e9, global_burst=1e9, max_sources=10,
                     clock=lambda: t[0])
    for i in range(10):
        assert rl.allow(f"10.0.0.{i}")
    assert not rl.allow("10.0.1.1")       # table full of recently active sources: fail closed
    t[0] += 10                            # all buckets refill -> prunable
    assert rl.allow("10.0.1.1") and len(rl._sources) == 1


def test_rate_limited_over_real_socket():
    with responders(("http://10.0.0.5:8000", "tok")) as (r,):
        r.limiter = RateLimiter(per_source_rate=0.001, per_source_burst=1)
        found, _ = discover_all("tok", timeout=0.5, port=r.port, targets=["127.0.0.1"] * 4, retransmits=1)
    assert len(found) == 1 and r.stats["replies"] == 1 and r.stats["dropped_rate"] >= 3


def test_invalid_advertise_url_rejected():
    for bad in ("ftp://x:1", "http://", "x" * 300, "http://h:99999"):
        with pytest.raises(ValueError):
            DiscoveryResponder(bad, "tok")


def test_default_advertise_url():
    assert d.default_advertise_url("10.1.2.3", 8000) == "http://10.1.2.3:8000"
    assert d.default_advertise_url("0.0.0.0", 8001).endswith(":8001")
    assert "0.0.0.0" not in d.default_advertise_url("0.0.0.0", 8001)
    assert d.default_advertise_url("::1", 8000) == "http://[::1]:8000"
    assert d.broadcast_targets()[0] == "255.255.255.255"
