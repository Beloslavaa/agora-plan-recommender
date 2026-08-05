"""Turning already-fetched HTML into PlanData-shaped event dicts. Pure text/
JSON parsing — the caller (infrastructure/http/fetcher.py, application/
extraction.py) owns actually fetching the HTML."""

import json
import logging
import re

from agora.backend.domain.url_safety import SONGKICK_BLANK_IMAGE, is_late_night, normalise_url

logger = logging.getLogger(__name__)


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
        if image_url and not SONGKICK_BLANK_IMAGE.search(image_url):
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


def extract_item_list_urls(html: str, base_url: str | None = None) -> list[str]:
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
