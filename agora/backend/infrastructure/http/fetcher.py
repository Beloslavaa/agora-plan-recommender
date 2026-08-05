"""SSRF-guarded page fetching. This is the only place in the app allowed to
make an outbound HTTP request for an ingestion-supplied URL — every caller
goes through fetch_page (or one of the wrappers below) so the public-address
check in _assert_public_url can never be bypassed."""

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse

import httpx

from agora.backend.domain.event_parsing import extract_item_list_urls
from agora.backend.domain.schemas import FixedSource
from agora.backend.infrastructure.config import settings

logger = logging.getLogger(__name__)

# Hostnames that must never be fetched even if DNS says otherwise
# (cloud metadata endpoints are the classic SSRF target).
_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}


async def _assert_public_url(url: str) -> None:
    """Raise ValueError if *url* is not a public http(s) address.

    Guards against SSRF: only http/https is allowed, and the host must resolve
    exclusively to public IP addresses (no loopback, private, link-local,
    reserved or multicast ranges — which covers 127.0.0.0/8, 10/8, 172.16/12,
    192.168/16, 169.254/16 incl. cloud metadata, ::1, fc00::/7, etc.).
    Every DNS answer is checked, and redirects are validated per hop by the
    caller, which limits (though cannot fully eliminate) DNS-rebinding tricks.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"blocked non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no host: {url!r}")
    if host.lower() in _BLOCKED_HOSTS:
        raise ValueError(f"blocked host: {host}")

    if settings.scraper_allow_private_hosts:
        return

    # If the host is already a literal IP, check it directly; otherwise resolve.
    try:
        literal = ipaddress.ip_address(host)
        candidates = [literal]
    except ValueError:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise ValueError(f"DNS resolution failed for {host}: {e}")
        candidates = [ipaddress.ip_address(info[4][0]) for info in infos]

    for ip in candidates:
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"blocked private/internal address {ip} for host {host}")


async def fetch_page(url: str, *, max_redirects: int | None = None) -> str:
    """Fetch a page, refusing internal targets and capping the response size.

    Redirects are followed manually so that each hop can be re-validated against
    the SSRF policy (httpx's built-in redirect following would skip that check).
    """
    if max_redirects is None:
        max_redirects = settings.scraper_max_redirects

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AgoraBot/1.0)"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        current = url
        for _ in range(max_redirects + 1):
            await _assert_public_url(current)
            async with client.stream("GET", current, headers=headers) as resp:
                if resp.is_redirect and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                resp.raise_for_status()

                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > settings.scraper_max_bytes:
                        raise RuntimeError(
                            f"response exceeded {settings.scraper_max_bytes} bytes: {current}"
                        )
                    chunks.append(chunk)

            raw = b"".join(chunks)
            encoding = resp.encoding or "utf-8"
            return raw.decode(encoding, errors="replace")

    raise RuntimeError(f"too many redirects (> {max_redirects}) fetching {url}")


async def scrape_fixed_source(source: FixedSource) -> str:
    return await fetch_page(source.url)


async def fetch_page_with_details(url: str) -> list[tuple[str, str]]:
    """Fetch *url*, following detail links from a JSON-LD ItemList if present.

    Returns a list of ``(html, page_url)`` tuples:
    - If the page has a JSON-LD ``ItemList`` → one entry per detail page
    - Otherwise → a single entry with the original page

    Detail URLs come from attacker-influenced page markup, so each one is fetched
    through ``fetch_page`` and is therefore subject to the same SSRF guard.
    """
    html = await fetch_page(url)
    urls = extract_item_list_urls(html, base_url=url)
    if not urls:
        return [(html, url)]

    logger.info("  → Found %d detail URLs via JSON-LD, fetching each concurrently", len(urls))

    async def _fetch(u: str) -> tuple[str, str] | None:
        try:
            detail_html = await fetch_page(u)
            return (detail_html, u)
        except Exception as e:
            logger.debug("  ✗ Failed to fetch %s: %s", u, e)
            return None

    results = await asyncio.gather(*[_fetch(u) for u in urls])
    return [r for r in results if r is not None]


async def fetch_fixed_source_with_details(
    source: FixedSource,
) -> list[tuple[str, str]]:
    """Fetch a fixed source, following detail links from JSON-LD ItemList if present.

    Thin wrapper over ``fetch_page_with_details`` kept for the fixed-source call
    site, which has a ``FixedSource`` rather than a bare URL.
    """
    return await fetch_page_with_details(source.url)
