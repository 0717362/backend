"""Shared, standard-library-only validation for public card and deployment URLs."""
from __future__ import annotations

import ipaddress
import os
import re
from urllib.parse import urlsplit


def is_public_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
        return (isinstance(address, ipaddress.IPv4Address) and address.is_global
                and not address.is_multicast and not address.is_reserved)
    except ValueError:
        host = host.lower()
        return (len(host) <= 253
                and re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", host) is not None
                and not host.endswith((".localhost", ".local", ".internal", ".home.arpa", ".test", ".invalid", ".example"))
                and not any(host == domain or host.endswith("." + domain)
                            for domain in ("example.com", "example.net", "example.org")))


def public_card_url(value) -> str:
    """Validate configuration; DNS resolution and public reachability require live checks."""
    text = str(value or "").strip()
    if not text or len(text) > 2048 or re.search(r"[\s\x00-\x1f]", text):
        return ""
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower().rstrip(".")
        if (parts.scheme not in ("http", "https") or not is_public_host(host)
                or parts.username or parts.password or parts.fragment):
            return ""
        _ = parts.port
        return text
    except ValueError:
        return ""


def require_public_origin() -> str:
    host = os.getenv("MUSIC_DOMAIN", "")
    if not is_public_host(host):
        raise ValueError("MUSIC_DOMAIN must be a public IPv4 address or DNS hostname.")
    if os.getenv("MUSIC_PUBLIC_BASE_URL", "").rstrip("/") != "https://" + host:
        raise ValueError("MUSIC_PUBLIC_BASE_URL must match https://MUSIC_DOMAIN.")
    return host


if __name__ == "__main__":
    try:
        require_public_origin()
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)
    print("Public HTTPS origin configuration passed; reachability is unverified.")
