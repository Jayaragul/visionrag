"""Generate a self-signed TLS certificate for LAN phone testing.

`getUserMedia` only works in a secure context. `localhost` is exempt, but the
phone is not localhost, so a LAN test needs TLS. This writes a combined
cert+key PEM covering every local IPv4 address it can find, so the same file
works whichever interface the phone reaches.

    python scripts/make_cert.py
    python apps/api/server.py --cert certs/dev.pem

The phone will still warn that the certificate is untrusted -- it is
self-signed, and that warning is correct. Accept it once. iOS is stricter than
Android and may require trusting the certificate under
Settings > General > VPN & Device Management.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import socket
import sys
from pathlib import Path


def local_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        # Connecting a UDP socket sends no packets but makes the OS pick the
        # interface it would actually use -- more reliable than hostname
        # lookup on machines with several adapters.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return sorted(ips)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="certs/dev.pem")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("cryptography is required:  python -m pip install cryptography")
        return 1

    ips = local_ips()
    print("certificate will cover:")
    for ip in ips:
        print(f"  {ip}")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "visionrag-dev")])
    alt_names: list[x509.GeneralName] = [x509.DNSName("localhost")]
    alt_names += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]

    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=args.days))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        + cert.public_bytes(serialization.Encoding.PEM)
    )
    print(f"\nwrote {out}  (valid {args.days} days)")
    print("\nstart the server with:")
    print(f"  python apps/api/server.py --cert {out}")
    print("\nthen open on the phone (same Wi-Fi):")
    for ip in ips:
        if ip != "127.0.0.1":
            print(f"  https://{ip}:8443")
    return 0


if __name__ == "__main__":
    sys.exit(main())
