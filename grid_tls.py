"""TLS for the coordinator API: a reusable self-signed certificate plus client-side pinning.

Bearer tokens and extraction results cross the LAN, so a non-loopback coordinator serves HTTPS.
Two ways for a client to trust it:

* **Fingerprint pinning** (zero-config): `--tls-fingerprint sha256:<hex>` / `$NIGHTSHIFT_TLS_FINGERPRINT`.
  Discovery replies carry the fingerprint *inside* the HMAC proof, so a `--discover` worker pins
  the right certificate with no setup and a rogue host can't substitute its own.
* **CA file**: `--ca-file` / `$NIGHTSHIFT_CA_FILE` (an internal CA, or the self-signed cert itself).
  Otherwise an https URL is verified against the system trust store (e.g. a reverse proxy with a
  public certificate).

Pinning is done *before any request is sent*: we fetch the server's certificate, compare its
SHA-256 against the pin, and then build an SSL context whose only trust anchor is that exact
certificate. A MITM can't present it without the private key, and the bearer token is never
written to a connection we haven't authenticated. Hostname checks are off in that mode because
the identity being checked is the certificate itself, not a DNS name.
"""
from __future__ import annotations

import datetime
import hashlib
import ipaddress
import os
import socket
import ssl
import stat
from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.parse import urlsplit

CA_FILE_ENV = "NIGHTSHIFT_CA_FILE"
FINGERPRINT_ENV = "NIGHTSHIFT_TLS_FINGERPRINT"
AUTO_CERT, AUTO_KEY = "coordinator-cert.pem", "coordinator-key.pem"
CERT_VALID_DAYS = 3 * 365

Verify = Union[bool, ssl.SSLContext]


class TLSConfigError(RuntimeError):
    """TLS is misconfigured or the server failed verification. Retrying won't help."""


# --------------------------------------------------------------------------- fingerprints
def normalize_fingerprint(fp: str) -> str:
    """'sha256:AB:CD..' / 'abcd..' -> 'sha256:abcd..' (64 lowercase hex chars)."""
    s = fp.strip().lower()
    if s.startswith("sha256:"):
        s = s[len("sha256:"):]
    s = s.replace(":", "").replace(" ", "")
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise TLSConfigError(f"invalid TLS fingerprint {fp!r}: expected sha256:<64 hex chars>")
    return "sha256:" + s


def fingerprint_der(der: bytes) -> str:
    return "sha256:" + hashlib.sha256(der).hexdigest()


def fingerprint_pem(pem: str) -> str:
    return fingerprint_der(ssl.PEM_cert_to_DER_cert(pem))


def fingerprint_file(path: Union[str, os.PathLike]) -> str:
    """Fingerprint of the first certificate in a PEM file (the leaf, for a chain file)."""
    text = Path(path).read_text(encoding="ascii")
    end = "-----END CERTIFICATE-----"
    if end not in text:
        raise TLSConfigError(f"{path} contains no PEM certificate")
    first = text[text.index("-----BEGIN CERTIFICATE-----"): text.index(end) + len(end)]
    return fingerprint_pem(first + "\n")


# --------------------------------------------------------------------------- server side
def default_sans(extra: Iterable[str] = ()) -> list[str]:
    from discovery import primary_lan_ip  # stdlib-only module; lazy to keep imports light

    names = ["localhost", "127.0.0.1", "::1", socket.gethostname(), primary_lan_ip(), *extra]
    out: list[str] = []
    for n in names:
        n = (n or "").strip().strip("[]")
        if n and n not in out:
            out.append(n)
    return out


def _check_private_key(path: Path) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise TLSConfigError(f"TLS key {path} must be a regular file, not a symlink")
    if os.name == "posix" and (st.st_uid != os.getuid() or st.st_mode & 0o077):
        raise TLSConfigError(f"TLS key {path} must be owned by you with mode 0600 (chmod 600 it)")


def ensure_self_signed(directory: Union[str, os.PathLike], sans: Iterable[str]) -> tuple[Path, Path, str, bool]:
    """Return (cert_path, key_path, fingerprint, created). Reuses an existing pair, so workers that
    pinned the fingerprint keep working across restarts; otherwise mints an ECDSA P-256 cert."""
    d = Path(directory)
    cert, key = d / AUTO_CERT, d / AUTO_KEY
    if cert.exists() and key.exists():
        _check_private_key(key)
        return cert, key, fingerprint_file(cert), False
    if cert.exists() or key.exists():
        raise TLSConfigError(f"found only one of {cert} / {key}; delete it to mint a new certificate")

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    names = list(sans)
    alt: list[x509.GeneralName] = []
    for n in names:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            alt.append(x509.DNSName(n))
    k = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nightshift-coordinator")])
    now = datetime.datetime.now(datetime.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(k.public_key())
         .serial_number(x509.random_serial_number())
         .not_valid_before(now - datetime.timedelta(minutes=5))
         .not_valid_after(now + datetime.timedelta(days=CERT_VALID_DAYS))
         .add_extension(x509.SubjectAlternativeName(alt), critical=False)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
         .sign(k, hashes.SHA256()))
    d.mkdir(parents=True, exist_ok=True)
    key_pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption())
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key_pem)
    cert_pem = c.public_bytes(serialization.Encoding.PEM)
    fd = os.open(cert, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(cert_pem)
    return cert, key, fingerprint_pem(cert_pem.decode()), True


def cert_sans(path: Union[str, os.PathLike]) -> list[str]:
    from cryptography import x509

    c = x509.load_pem_x509_certificate(Path(path).read_bytes())
    try:
        ext = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    return [str(v) for v in ext.get_values_for_type(x509.DNSName)] + \
           [str(v) for v in ext.get_values_for_type(x509.IPAddress)]


# --------------------------------------------------------------------------- client side
def _host_port(url: str) -> tuple[str, int]:
    u = urlsplit(url)
    if not u.hostname:
        raise TLSConfigError(f"invalid coordinator URL {url!r}")
    return u.hostname, u.port or 443


def pinned_context(url: str, fingerprint: str, timeout: float = 5.0) -> ssl.SSLContext:
    """SSL context that trusts only the certificate `url` presents *if* it matches `fingerprint`.

    Raises OSError if the server is unreachable (callers may retry) and TLSConfigError on a
    fingerprint mismatch (a different certificate: rotated, or someone in the middle)."""
    want = normalize_fingerprint(fingerprint)
    host, port = _host_port(url)
    probe = ssl.create_default_context()
    probe.check_hostname, probe.verify_mode = False, ssl.CERT_NONE  # we only read the cert here
    with socket.create_connection((host, port), timeout=timeout) as raw:
        with probe.wrap_socket(raw, server_hostname=host) as s:
            der = s.getpeercert(binary_form=True)
    if not der:
        raise TLSConfigError(f"{url} presented no certificate")
    got = fingerprint_der(der)
    if got != want:
        raise TLSConfigError(f"TLS fingerprint mismatch for {url}: expected {want}, server presented {got}. "
                             f"Either the coordinator's certificate changed or something is intercepting the "
                             f"connection; re-check the fingerprint the coordinator prints at startup.")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False          # identity == this exact certificate, not a DNS name
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN  # lets a CA-issued leaf be its own anchor too
    ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
    return ctx


_CONTEXT_CACHE: dict[tuple, ssl.SSLContext] = {}


def client_verify(url: str, ca_file: Optional[str] = None, fingerprint: Optional[str] = None) -> Verify:
    """The `verify=` argument for an httpx client talking to `url`.

    Precedence: explicit fingerprint > explicit CA file > $NIGHTSHIFT_TLS_FINGERPRINT >
    $NIGHTSHIFT_CA_FILE > system trust store. Plain http URLs return True (unused by httpx)."""
    if not url.lower().startswith("https://"):
        if fingerprint or ca_file:
            raise TLSConfigError(f"a TLS fingerprint/CA file was given but the coordinator URL {url} is not https")
        return True
    if not fingerprint and not ca_file:
        fingerprint = os.environ.get(FINGERPRINT_ENV, "").strip() or None
        ca_file = None if fingerprint else (os.environ.get(CA_FILE_ENV, "").strip() or None)
    key = (_host_port(url), fingerprint and normalize_fingerprint(fingerprint), ca_file)
    if key in _CONTEXT_CACHE:  # polling clients would otherwise re-run the pin handshake every call
        return _CONTEXT_CACHE[key]
    if fingerprint:
        ctx = pinned_context(url, fingerprint)
    elif ca_file:
        try:
            ctx = ssl.create_default_context(cafile=ca_file)
        except (OSError, ssl.SSLError) as e:
            raise TLSConfigError(f"cannot load CA file {ca_file}: {e}") from e
    else:
        return True
    _CONTEXT_CACHE[key] = ctx
    return ctx
