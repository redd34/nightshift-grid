"""Zero-config LAN discovery of the Nightshift coordinator over UDP broadcast (stdlib only).

Why not just mDNS? `zeroconf` is an extra dependency, and multicast is often filtered on
office Wi-Fi. A single UDP broadcast on one port works on any flat LAN, and we control
the security properties, which matter here.

Protocol (version 1, UTF-8 JSON, one datagram each way)
--------------------------------------------------------
probe (worker -> broadcast:47474)::

    {"v":1,"type":"nightshift.discover","nonce":"<32 hex>","wid":"<worker id>"?,"pad":"...."}

reply (coordinator -> unicast to the probe's source address)::

    {"v":1,"type":"nightshift.coordinator","url":"https://10.0.0.5:8000","nonce":"<echo>",
     "tls":"sha256:<hex>" | null,
     "proof":"<hex HMAC-SHA256(key, LABEL || nonce || '\\n' || url [|| '\\n' || tls])>" | null,
     "auth": true|false}

`key` is the worker's bearer secret: the master worker token, or - when the probe names a
`wid` - that worker's per-worker credential, which the coordinator re-derives from the master.

Security properties
-------------------
* **Proof of token possession, not a token leak.** The worker sends its bearer token on its
  first HTTP request, so whatever answers discovery could harvest that token. A worker that
  has a token therefore accepts only replies whose `proof` is a valid HMAC over *its own fresh
  nonce* and the *advertised URL*, keyed by the worker token. A rogue host without the token
  can't forge a proof. It can't replay an old reply either (the nonce is fresh per probe), and
  it can't relay a real reply with its own URL substituted (the URL is inside the MAC). The
  HMAC reveals nothing about the key.
* **TLS pinning for free.** The certificate fingerprint is inside the MAC, so a worker that
  discovers the coordinator also learns, authenticated, which certificate to pin.
* **Tokenless workers** can't verify anything, so they accept unproven replies. That is
  logged as a warning; it's only safe on a trusted network.
* **Ambiguity is an error.** If two different coordinators prove themselves, we raise rather
  than guess, because guessing wrong sends work and results to the wrong grid.
* **Not an amplification reflector.** UDP sources are spoofable, so the responder (a) answers
  only well-formed probes, (b) requires probes of at least `MIN_PROBE_BYTES` (clients pad),
  (c) never sends a reply larger than `MAX_AMPLIFICATION` x the probe, and (d) rate-limits
  per source IP plus globally. Inbound packets over `MAX_PACKET_BYTES` are dropped unparsed.
  Same idea as QUIC's padded Initial packets.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence
from urllib.parse import urlsplit

log = logging.getLogger("nightshift.discovery")

DISCOVERY_PORT = 47474
PROTOCOL_VERSION = 1
PROBE_TYPE = "nightshift.discover"
REPLY_TYPE = "nightshift.coordinator"
MAC_LABEL = b"nightshift-discovery-v1\n"   # domain separation: this MAC can't be confused with any other use of the token
MAX_PACKET_BYTES = 1200                    # below any sane MTU; bigger datagrams are dropped unparsed
MIN_PROBE_BYTES = 256                      # padding requirement that makes the 2x reply cap workable
PROBE_PAD_TO = 512                         # what our own client sends
MAX_AMPLIFICATION = 2.0                    # reply bytes <= 2 x probe bytes
MAX_URL_CHARS = 256
NONCE_HEX_MIN, NONCE_HEX_MAX = 32, 64


class DiscoveryError(RuntimeError):
    """No usable coordinator was found, or more than one valid coordinator answered."""


# --------------------------------------------------------------------------- shared helpers
def compute_proof(token: str, nonce: str, url: str, tls_fingerprint: Optional[str] = None) -> str:
    msg = MAC_LABEL + nonce.encode() + b"\n" + url.encode()
    if tls_fingerprint:
        msg += b"\n" + tls_fingerprint.encode()
    return hmac.new(token.encode(), msg, hashlib.sha256).hexdigest()


def _valid_fingerprint(fp: object) -> bool:
    return isinstance(fp, str) and fp.startswith("sha256:") and _is_hex(fp[7:], 64, 64)


def _valid_wid(wid: object) -> bool:
    return isinstance(wid, str) and 0 < len(wid) <= 128 and all(0x21 <= ord(c) < 0x7F for c in wid)


def _is_hex(s: object, lo: int, hi: int) -> bool:
    if not isinstance(s, str) or not (lo <= len(s) <= hi):
        return False
    try:
        bytes.fromhex(s)
        return True
    except ValueError:
        return False


def _valid_url(url: object) -> bool:
    if not isinstance(url, str) or not (0 < len(url) <= MAX_URL_CHARS):
        return False
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises ValueError on a non-numeric/out-of-range port
        return (parts.scheme in ("http", "https") and bool(parts.hostname) and not (parts.query or parts.fragment)
                and not any(c.isspace() or ord(c) < 0x20 for c in url))
    except ValueError:
        return False


def _encode(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def _decode(data: bytes) -> Optional[dict]:
    """Parse a datagram, returning None for anything malformed. Oversized packets are rejected
    before parsing so a flood of big packets can't keep the JSON parser busy."""
    if len(data) > MAX_PACKET_BYTES:
        return None
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) and obj.get("v") == PROTOCOL_VERSION else None


def primary_lan_ip() -> str:
    """The source IP the OS would use to reach the internet (the "primary" interface).

    A UDP connect() sends no packets; it only selects a route. Falls back to loopback
    on a machine with no route (air-gapped lab), where advertising a LAN URL is moot anyway."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1: never routed anywhere, only used for route lookup
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def default_advertise_url(host: str, port: int, scheme: str = "http") -> str:
    """A URL that LAN workers can actually reach. A wildcard bind (0.0.0.0 / ::) isn't an
    address, so we use the primary LAN IP. A specific bind address is used as-is."""
    try:
        ip = ipaddress.ip_address(host)
        h = primary_lan_ip() if ip.is_unspecified else host
    except ValueError:
        h = host  # a hostname: trust the operator
    if ":" in h:
        h = f"[{h}]"
    return f"{scheme}://{h}:{port}"


def broadcast_targets() -> list[str]:
    """255.255.255.255 plus each IPv4 interface's directed broadcast address.

    On macOS the limited broadcast (255.255.255.255) only leaves the default-route
    interface, so a coordinator on a second NIC (Ethernet + Wi-Fi) is only reachable via its
    subnet's directed broadcast. psutil is already a core dependency of the worker; the import
    is guarded so this module stays stdlib-only."""
    targets = ["255.255.255.255"]
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family != socket.AF_INET or a.address.startswith("127."):
                    continue
                bcast = a.broadcast
                if not bcast and a.netmask:
                    try:
                        bcast = str(ipaddress.ip_network(f"{a.address}/{a.netmask}", strict=False).broadcast_address)
                    except ValueError:
                        bcast = None
                if bcast and bcast not in targets:
                    targets.append(bcast)
    except Exception:  # psutil missing or interface enumeration failed: limited broadcast still works
        pass
    return targets


# --------------------------------------------------------------------------- coordinator side
@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token buckets per source IP plus one global bucket.

    The global bucket caps total reply bandwidth even if an attacker spoofs many sources. The
    source table is bounded: when it's full, buckets that have refilled completely carry no
    information and are dropped first."""
    per_source_rate: float = 2.0
    per_source_burst: float = 8.0      # a worker probes several broadcast addrs + retransmits
    global_rate: float = 50.0
    global_burst: float = 100.0
    max_sources: int = 4096
    clock: Callable[[], float] = time.monotonic
    _sources: dict = field(default_factory=dict)
    _global: Optional[_Bucket] = None

    def _take(self, b: _Bucket, rate: float, burst: float, now: float) -> bool:
        b.tokens = min(burst, b.tokens + (now - b.updated) * rate)
        b.updated = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True
        return False

    def allow(self, source: str) -> bool:
        now = self.clock()
        if self._global is None:
            self._global = _Bucket(self.global_burst, now)
        b = self._sources.get(source)
        if b is None:
            if len(self._sources) >= self.max_sources:
                self._prune(now)
                if len(self._sources) >= self.max_sources:
                    return False  # table full of active sources: fail closed (no reply)
            b = self._sources[source] = _Bucket(self.per_source_burst, now)
        # Check per-source first so one noisy source can't drain the global budget.
        return self._take(b, self.per_source_rate, self.per_source_burst, now) and \
            self._take(self._global, self.global_rate, self.global_burst, now)

    def _prune(self, now: float) -> None:
        full = [k for k, b in self._sources.items()
                if b.tokens + (now - b.updated) * self.per_source_rate >= self.per_source_burst]
        for k in full:
            del self._sources[k]


class DiscoveryResponder(asyncio.DatagramProtocol):
    """Answers discovery probes on the coordinator. Runs on the coordinator's event loop; each
    datagram costs one bounded JSON parse plus one HMAC, so it never stalls the HTTP API.

    `worker_token=None` means the grid runs without worker auth: replies then carry
    `proof: null, auth: false` and only tokenless workers will accept them."""

    def __init__(self, advertise_url: str, worker_token: Optional[str], *, bind_host: str = "",
                 port: int = DISCOVERY_PORT, limiter: Optional[RateLimiter] = None,
                 tls_fingerprint: Optional[str] = None):
        if not _valid_url(advertise_url):
            raise ValueError(f"invalid advertise URL {advertise_url!r} (need http(s)://host:port, <= {MAX_URL_CHARS} chars)")
        if tls_fingerprint is not None and not _valid_fingerprint(tls_fingerprint):
            raise ValueError(f"invalid TLS fingerprint {tls_fingerprint!r}")
        self.advertise_url = advertise_url
        self.worker_token = worker_token
        self.tls_fingerprint = tls_fingerprint
        self.bind_host = bind_host
        self.port = port
        self.limiter = limiter or RateLimiter()
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.stats = {"probes": 0, "replies": 0, "dropped_malformed": 0, "dropped_small": 0, "dropped_rate": 0}

    async def start(self) -> "DiscoveryResponder":
        """Bind the UDP socket. We bind the wildcard address by default because on Linux a
        socket bound to a unicast IP never receives broadcasts."""
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.port))
        sock.setblocking(False)
        self.transport, _ = await loop.create_datagram_endpoint(lambda: self, sock=sock)
        self.port = sock.getsockname()[1]  # resolve an ephemeral port (tests)
        return self

    def close(self) -> None:
        if self.transport:
            self.transport.close()
            self.transport = None

    def build_reply(self, data: bytes, source_ip: str) -> Optional[bytes]:
        """Pure decision function: the reply bytes, or None to stay silent. Silence is the
        right answer to anything unexpected; an error reply would itself be reflectable."""
        probe = _decode(data)
        if probe is None or probe.get("type") != PROBE_TYPE or not _is_hex(probe.get("nonce"), NONCE_HEX_MIN, NONCE_HEX_MAX):
            self.stats["dropped_malformed"] += 1
            return None
        self.stats["probes"] += 1
        if len(data) < MIN_PROBE_BYTES:
            self.stats["dropped_small"] += 1
            return None
        if not self.limiter.allow(source_ip):
            self.stats["dropped_rate"] += 1
            return None
        nonce = probe["nonce"]
        key = self.worker_token
        wid = probe.get("wid")
        if key and wid is not None:
            if not _valid_wid(wid):
                self.stats["dropped_malformed"] += 1
                return None
            from grid_auth import derive_worker_credential  # lazy: keeps this module stdlib-only for workers
            key = derive_worker_credential(key, wid)
        reply = _encode({
            "v": PROTOCOL_VERSION, "type": REPLY_TYPE, "url": self.advertise_url, "nonce": nonce,
            "tls": self.tls_fingerprint,
            "proof": compute_proof(key, nonce, self.advertise_url, self.tls_fingerprint) if key else None,
            "auth": self.worker_token is not None,
        })
        if len(reply) > MAX_AMPLIFICATION * len(data):
            self.stats["dropped_small"] += 1
            return None
        return reply

    def datagram_received(self, data: bytes, addr) -> None:  # asyncio callback
        try:
            reply = self.build_reply(data, addr[0])
            if reply is not None and self.transport is not None:
                self.transport.sendto(reply, addr)
                self.stats["replies"] += 1
        except Exception as e:  # a bad packet must never take down the coordinator's loop
            log.warning("discovery: error handling packet from %s: %r", addr[0], e)

    def error_received(self, exc: Exception) -> None:  # e.g. ICMP port unreachable from a gone worker
        log.debug("discovery socket error: %r", exc)


# --------------------------------------------------------------------------- worker side
@dataclass(frozen=True)
class DiscoveredCoordinator:
    url: str
    source: str        # IP the reply came from
    verified: bool     # proof checked against our token
    auth: bool         # the coordinator says it requires tokens
    tls_fingerprint: Optional[str] = None  # certificate to pin (authenticated iff verified)


def _worker_id_of(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    from grid_auth import credential_worker_id
    return credential_worker_id(token)


def make_probe(nonce: str, worker_id: Optional[str] = None) -> bytes:
    """A padded probe. Padding is what lets the responder enforce its 2x amplification cap.
    `worker_id` (for per-worker credentials) tells the coordinator which key to prove with."""
    base = {"v": PROTOCOL_VERSION, "type": PROBE_TYPE, "nonce": nonce, "pad": ""}
    if worker_id is not None:
        base["wid"] = worker_id
    need = PROBE_PAD_TO - len(_encode(base))
    base["pad"] = "0" * max(0, need)
    return _encode(base)


def discover_all(worker_token: Optional[str], timeout: float = 3.0, port: int = DISCOVERY_PORT,
                 targets: Optional[Sequence[str]] = None, retransmits: int = 2) -> tuple[list[DiscoveredCoordinator], list[str]]:
    """Broadcast a probe and collect replies for `timeout` seconds.

    Returns (accepted coordinators deduplicated by URL, descriptions of rejected replies).
    `targets` (default: `broadcast_targets()`) entries are hosts probed on `port`, or explicit
    `(host, port)` tuples, which tests use to reach several loopback responders.
    The probe is re-sent `retransmits` times, evenly spaced, because broadcast UDP is lossy
    on Wi-Fi. All copies share one nonce, so duplicate replies collapse by URL."""
    nonce = os.urandom(16).hex()
    probe = make_probe(nonce, _worker_id_of(worker_token))
    targets = list(targets) if targets is not None else broadcast_targets()
    accepted: dict[str, DiscoveredCoordinator] = {}
    rejected: list[str] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", 0))
        start = time.monotonic()
        deadline = start + timeout
        sends = [start + i * timeout / (retransmits + 1) for i in range(retransmits + 1)]
        while True:
            now = time.monotonic()
            while sends and sends[0] <= now:
                sends.pop(0)
                for t in targets:
                    try:
                        sock.sendto(probe, t if isinstance(t, tuple) else (t, port))
                    except OSError as e:  # e.g. no route for a directed broadcast on a down iface
                        log.debug("discovery: send to %s failed: %s", t, e)
            wake = min([deadline] + sends[:1])
            if now >= deadline:
                break
            sock.settimeout(max(0.001, wake - now))
            try:
                data, addr = sock.recvfrom(MAX_PACKET_BYTES + 1)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError as e:  # ICMP errors surface here on some platforms
                log.debug("discovery: recv error %s", e)
                continue
            found = _check_reply(data, addr[0], nonce, worker_token)
            if isinstance(found, DiscoveredCoordinator):
                accepted.setdefault(found.url, found)
            elif found and found not in rejected:  # retransmits x targets repeat the same reply
                rejected.append(found)
    finally:
        sock.close()
    return list(accepted.values()), rejected


def _check_reply(data: bytes, source: str, nonce: str, worker_token: Optional[str]):
    """DiscoveredCoordinator if acceptable, a rejection reason string, or None for noise
    (e.g. our own probe echoed back by the broadcast, or garbage)."""
    msg = _decode(data)
    if msg is None or msg.get("type") != REPLY_TYPE:
        return None
    if not hmac.compare_digest(str(msg.get("nonce", "")), nonce):
        return f"{source}: stale/foreign nonce"
    url = msg.get("url")
    if not _valid_url(url):
        return f"{source}: invalid url"
    auth = msg.get("auth") is True
    proof = msg.get("proof")
    tls = msg.get("tls")
    if tls is not None and not _valid_fingerprint(tls):
        return f"{source}: invalid tls fingerprint"
    if worker_token:
        if not isinstance(proof, str) or not hmac.compare_digest(proof, compute_proof(worker_token, nonce, url, tls)):
            return f"{source}: {url} did not prove knowledge of our worker token"
        return DiscoveredCoordinator(url, source, verified=True, auth=auth, tls_fingerprint=tls)
    return DiscoveredCoordinator(url, source, verified=False, auth=auth, tls_fingerprint=tls)


def discover_coordinator(worker_token: Optional[str], timeout: float = 3.0, port: int = DISCOVERY_PORT,
                         targets: Optional[Iterable[str]] = None) -> str:
    """Base URL of the single coordinator on the LAN (see `discover_coordinator_info`)."""
    return discover_coordinator_info(worker_token, timeout=timeout, port=port, targets=targets).url


def discover_coordinator_info(worker_token: Optional[str], timeout: float = 3.0, port: int = DISCOVERY_PORT,
                              targets: Optional[Iterable[str]] = None) -> DiscoveredCoordinator:
    """Find exactly one coordinator on the LAN and return its base URL.

    With a token, only coordinators that prove they hold the same worker token are accepted.
    Without one, any reply is accepted but flagged (warning), because we're trusting the LAN.
    Raises DiscoveryError when there are none, or when there are several valid ones."""
    found, rejected = discover_all(worker_token, timeout=timeout, port=port,
                                   targets=list(targets) if targets is not None else None)
    if not found:
        hint = f" Rejected replies: {'; '.join(rejected[:5])}." if rejected else ""
        raise DiscoveryError(f"no Nightshift coordinator answered on UDP port {port} within {timeout:.1f}s.{hint} "
                             f"Is the coordinator running with --advertise on this subnet? Otherwise pass --coordinator URL.")
    if len(found) > 1:
        listing = ", ".join(f"{c.url} (from {c.source})" for c in found)
        raise DiscoveryError(f"{len(found)} coordinators answered: {listing}. Refusing to guess; pass --coordinator URL.")
    c = found[0]
    if not c.verified:
        log.warning("discovery: accepted UNVERIFIED coordinator %s from %s (worker has no token, so a rogue host on "
                    "this LAN could impersonate it). Configure a worker token for authenticated discovery.", c.url, c.source)
        if c.auth:
            log.warning("discovery: %s requires a worker token, but this worker has none; requests will get 401", c.url)
    for r in rejected:
        log.warning("discovery: ignored reply %s", r)
    return c
