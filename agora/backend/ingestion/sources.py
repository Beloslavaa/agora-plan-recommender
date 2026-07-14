import asyncio
import ipaddress
import json
import logging
import re
import socket
from datetime import datetime, time as _time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from agora.backend.config import settings
from agora.backend.ingestion.schemas import FixedSource

logger = logging.getLogger(__name__)
SOURCES_FILE = Path("data/fixed_sources.json")

# Events starting at/after this local time are treated as club/party listings
# rather than shows, and dropped during ingestion.
_LATE_NIGHT_CUTOFF = _time(23, 45)


def is_late_night(iso_datetime: str | None) -> bool:
    """True if *iso_datetime* has a time component at/after the late-night cutoff.

    Conservative by design: a date-only string (no time component) or anything
    that fails to parse is treated as "unknown", not late-night, so we never
    drop an event on data we can't actually confirm.
    """
    if not iso_datetime or len(iso_datetime) <= 10:
        return False
    try:
        dt = datetime.fromisoformat(iso_datetime)
    except ValueError:
        return False
    return dt.time() >= _LATE_NIGHT_CUTOFF


# ── URL safety ─────────────────────────────────────────────
# These patterns never appear in legitimate event URLs.
_SUSPICIOUS_URL_PATTERNS = re.compile(
    r"(data:|javascript:|vbscript:|file:|ftp:)",
    re.IGNORECASE,
)


def normalise_url(raw: str | None, base_url: str | None = None) -> str | None:
    """Validate and sanitise a URL extracted by the LLM.

    Returns a safe absolute ``http(s)`` URL or ``None`` if the input is
    missing, malformed, or suspicious. Relative URLs are resolved against
    *base_url* (the page the URL was extracted from).
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None

    # Resolve relative URLs
    if base_url:
        raw = urljoin(base_url, raw)

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        logger.debug("  ~ rejecting non-http(s) URL: %s", raw[:80])
        return None
    if not parsed.netloc:
        logger.debug("  ~ rejecting URL without host: %s", raw[:80])
        return None
    if _SUSPICIOUS_URL_PATTERNS.search(raw):
        logger.debug("  ~ rejecting suspicious URL: %s", raw[:80])
        return None

    return raw


# Hostnames that must never be fetched even if DNS says otherwise
# (cloud metadata endpoints are the classic SSRF target).
_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}


def load_fixed_sources() -> list[FixedSource]:
    if not SOURCES_FILE.exists():
        return []
    raw = json.loads(SOURCES_FILE.read_text())
    return [FixedSource(**item) for item in raw]


def save_fixed_sources(sources: list[FixedSource]) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_FILE.write_text(
        json.dumps([s.model_dump() for s in sources], indent=2)
    )


def promote_source(name: str, url: str, promoted_by: str | None = None) -> bool:
    sources = load_fixed_sources()
    if any(s.url == url for s in sources):
        return False
    sources.append(FixedSource(name=name, url=url, promoted_by=promoted_by))
    save_fixed_sources(sources)
    return True


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


def _parse_ld_event(obj: dict, base_url: str | None = None) -> dict | None:
    """Normalise a JSON-LD ``Event`` object into flat PlanData-shaped fields.

    Returns ``None`` if *obj* isn't an Event (or has no name). Fields absent from
    the source data are simply omitted from the result.
    """
    # schema.org has many Event subtypes (SocialEvent, MusicEvent, ScreeningEvent,
    # TheaterEvent, Festival, ...) that all carry the same fields we care about —
    # Eventbrite alone uses at least "Event" and "SocialEvent" across templates.
    ld_type = obj.get("@type") if isinstance(obj, dict) else None
    is_event = isinstance(ld_type, str) and ("Event" in ld_type or ld_type == "Festival")
    if not is_event:
        return None
    out: dict = {}

    name = obj.get("name")
    if name and str(name).strip():
        out["title"] = str(name).strip()
    else:
        return None

    desc = obj.get("description")
    if desc:
        out["description"] = str(desc).strip()

    url = normalise_url(obj.get("url"), base_url)
    if url:
        out["url"] = url

    image = obj.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    if isinstance(image, dict):
        image = image.get("url")
    if image:
        image_url = normalise_url(image, base_url)
        if image_url:
            out["image_url"] = image_url

    start = obj.get("startDate")
    if start:
        if is_late_night(str(start)):
            return None
        out["start_date"] = str(start)[:10]
    end = obj.get("endDate")
    if end:
        out["end_date"] = str(end)[:10]

    location = obj.get("location")
    if isinstance(location, dict):
        loc_name = location.get("name")
        address = location.get("address")
        addr_str = None
        if isinstance(address, dict):
            parts = [address.get("streetAddress"), address.get("addressLocality")]
            addr_str = ", ".join(p for p in parts if p) or None
        elif isinstance(address, str):
            addr_str = address
        location_str = ", ".join(p for p in (loc_name, addr_str) if p)
        if location_str:
            out["location"] = location_str
    elif isinstance(location, str) and location.strip():
        out["location"] = location.strip()

    offers = obj.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        ticket_url = normalise_url(offers.get("url"), base_url)
        if ticket_url:
            out["ticket_url"] = ticket_url
        price = offers.get("lowPrice") or offers.get("price")
        if price is not None:
            try:
                out["price"] = float(price)
            except (TypeError, ValueError):
                pass

    return out


def extract_ld_events(html: str, base_url: str | None = None) -> list[dict]:
    """Parse JSON-LD from *html* and return normalised event dicts.

    Handles the two shapes seen in the wild:
    - A single top-level ``Event`` (or a ``@graph`` array containing one) —
      typical on a dedicated event detail page.
    - An ``ItemList`` whose ``itemListElement[].item`` are ``Event`` objects —
      typical on listing/category pages (Eventbrite, Songkick, etc.).

    Each dict uses PlanData field names (title, url, image_url, ticket_url,
    start_date, end_date, location, price, description) so callers can build a
    PlanData directly from it without going through the LLM.
    """
    events: list[dict] = []
    # type= can be unquoted, double- or single-quoted — Yoast SEO (WordPress,
    # e.g. salaequis.es) emits type='application/ld+json' with single quotes,
    # which a "?-only pattern silently never matches.
    scripts = re.findall(
        r'<script[^>]*type\s*=\s*["\']?application/ld\+json["\']?[^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    for raw in scripts:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        candidates: list = data if isinstance(data, list) else [data]
        # Some sites wrap everything in a schema.org @graph array.
        expanded: list = []
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("@graph"), list):
                expanded.extend(c["@graph"])
            else:
                expanded.append(c)

        for obj in expanded:
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") == "ItemList":
                for item in obj.get("itemListElement") or []:
                    if not isinstance(item, dict):
                        continue
                    inner = item.get("item")
                    ev = _parse_ld_event(inner, base_url) if isinstance(inner, dict) else None
                    if ev:
                        events.append(ev)
                        continue
                    # Older/simpler shape: the ListItem carries a bare `url`
                    # directly, with no nested Event object (e.g. cinesrenoir,
                    # often used for a plain page/sitemap listing rather than a
                    # dedicated events feed). We only get a URL here, not full
                    # event data — good enough for detail-page discovery; the
                    # caller re-extracts once that page is fetched.
                    url = normalise_url(item.get("url"), base_url)
                    if url:
                        events.append({"url": url})
            else:
                # _parse_ld_event itself checks whether @type is an Event
                # subtype and returns None otherwise (e.g. the "WebPage" block
                # sites often emit alongside their Event block).
                ev = _parse_ld_event(obj, base_url)
                if ev:
                    # Some sites (cinesrenoir's showtime pages, for example)
                    # describe the event on the page itself but never state its
                    # own url in the JSON-LD. Since this Event was found directly
                    # on the page (not nested inside an ItemList of many events),
                    # that page IS the specific event page — use it.
                    if not ev.get("url") and base_url:
                        ev["url"] = base_url
                    events.append(ev)
    return events


def extract_dice_event_details(html: str) -> dict | None:
    """Pull an event's real start time, description and genre tags out of a
    dice.fm event page.

    dice.fm carries no schema.org Event markup, and neither the start time nor
    a real description/genre ever appear in the visible page text (or even in
    a meta description tag) — the browse/listing pages give an LLM nothing to
    extract them from. All of it is buried in Next.js's serialized page state
    (a JSON string nested inside __NEXT_DATA__). This is undocumented and
    specific to dice.fm's frontend, so parsing is defensive throughout; any
    structural mismatch just means "unknown" (returns None), not an error.

    Returns a dict with any of "start" (ISO datetime str), "description"
    (str), "tags" (list[str]) that were found — or None if nothing was.
    """
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL,
    )
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        init = json.loads(data["props"]["pageProps"]["initialState"])
        ev = init["event"]["event"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    out: dict = {}
    start = (ev.get("dates") or {}).get("event_start_date")
    if start:
        out["start"] = start
    description = (ev.get("about") or {}).get("description")
    if description and str(description).strip():
        out["description"] = str(description).strip()
    tags = [
        t.get("name") for t in (ev.get("tags_types") or [])
        if isinstance(t, dict) and t.get("name")
    ]
    if tags:
        out["tags"] = tags
    return out or None


def _extract_item_list_urls(html: str, base_url: str | None = None) -> list[str]:
    """Detail-page URLs from a JSON-LD ``ItemList`` (listing pages only)."""
    urls = [ev["url"] for ev in extract_ld_events(html, base_url) if ev.get("url")]
    # dedupe, keep order
    seen: set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def fetch_page_with_details(url: str) -> list[tuple[str, str]]:
    """Fetch *url*, following detail links from a JSON-LD ItemList if present.

    Returns a list of ``(html, page_url)`` tuples:
    - If the page has a JSON-LD ``ItemList`` → one entry per detail page
    - Otherwise → a single entry with the original page

    Detail URLs come from attacker-influenced page markup, so each one is fetched
    through ``fetch_page`` and is therefore subject to the same SSRF guard.
    """
    html = await fetch_page(url)
    urls = _extract_item_list_urls(html, base_url=url)
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