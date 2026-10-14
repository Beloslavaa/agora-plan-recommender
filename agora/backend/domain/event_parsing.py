"""Turning already-fetched HTML into PlanData-shaped event dicts, page text
for the LLM, and an event page's meta image/description/ticket link. Pure text/
JSON parsing — the caller (infrastructure/http/fetcher.py, application/
extraction.py) owns actually fetching the HTML."""

import json
import logging
import re
from html import unescape

from agora.backend.domain.schemas import PlanData
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
            # Not shown in the location string, but they're what tell
            # "Madrid, ES" from "Madrid, US" (see domain/countries.py).
            country = address.get("addressCountry")
            if isinstance(country, dict):
                country = country.get("name") or country.get("@id")
            if isinstance(country, str) and country.strip():
                out["country"] = country.strip()
            postal = address.get("postalCode")
            if postal is not None and str(postal).strip():
                out["postal_code"] = str(postal).strip()
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


def extract_dice_listing_events(html: str) -> list[dict]:
    """Events from a dice.fm browse/listing page, without an LLM.

    The listing's visible text only shows dates, but Next.js's serialized page
    state (__NEXT_DATA__ → props.pageProps.events) carries every listed event
    in full: exact start time, venue address and country, price, image.
    Undocumented and specific to dice.fm's frontend, so parsing is defensive —
    any structural mismatch returns [] and the caller falls back to the LLM.

    Description and genre tags aren't in the listing; extraction's
    _apply_dice_details fetches each event's own page for those.

    Returns PlanData-shaped dicts, like extract_ld_events.
    """
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL,
    )
    if not m:
        return []
    try:
        events = json.loads(m.group(1))["props"]["pageProps"]["events"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return []
    if not isinstance(events, list):
        return []

    out: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict) or not ev.get("id") or not str(ev.get("name") or "").strip():
            continue
        dates = ev.get("dates") if isinstance(ev.get("dates"), dict) else {}
        start = dates.get("event_start_date")
        if start and is_late_night(str(start)):
            continue

        # The listing's own links use /event/{id}, which redirects to the
        # event's full slug URL — also the page tickets are bought on.
        url = f"https://dice.fm/event/{ev['id']}"
        item: dict = {"title": str(ev["name"]).strip(), "url": url, "ticket_url": url}
        if start:
            item["start_date"] = str(start)[:10]
        # A single night's end is the next morning — only a real multi-day
        # event gets an end_date.
        if dates.get("is_multi_days_event") and dates.get("event_end_date"):
            item["end_date"] = str(dates["event_end_date"])[:10]

        images = ev.get("images")
        if isinstance(images, dict):
            image = images.get("square") or next(iter(images.values()), None)
            image_url = normalise_url(image) if isinstance(image, str) else None
            if image_url:
                item["image_url"] = image_url

        venues = ev.get("venues")
        venue = venues[0] if isinstance(venues, list) and venues and isinstance(venues[0], dict) else {}
        location = ", ".join(str(p) for p in (venue.get("name"), venue.get("address")) if p)
        if location:
            item["location"] = location
        city = venue.get("city") if isinstance(venue.get("city"), dict) else {}
        if city.get("country_code"):
            item["country"] = str(city["country_code"])

        # In cents (2876 → 28.76 €, checked against event pages). Fixed-price
        # events set `amount`, tiered ones `amount_from` (the lowest tier).
        price = ev.get("price")
        amount = price.get("amount_from") if isinstance(price, dict) else None
        if amount is None and isinstance(price, dict):
            amount = price.get("amount")
        if isinstance(amount, (int, float)):
            item["price"] = amount / 100

        out.append(item)
    return out


def _wp_text(value) -> str:
    """Plain text from a WordPress HTML field: tags stripped, entities decoded."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def parse_wp_events(events: list, max_events: int) -> list[dict]:
    """PlanData-shaped dicts from The Events Calendar's REST API
    (/wp-json/tribe/events/v1/events — see fetcher.fetch_wp_events).

    Recurring shows (a comedy club's weekly open mic) come back as one entry
    per date; only the NEXT date of each is kept — the same "include it ONCE"
    rule the LLM extractor follows — soonest first, capped at *max_events*.
    """
    def _start(ev) -> str:
        return str(ev.get("start_date") or "") if isinstance(ev, dict) else ""

    out: list[dict] = []
    seen_titles: set[str] = set()
    for ev in sorted((e for e in events if isinstance(e, dict)), key=_start):
        title = _wp_text(ev.get("title"))
        if not title or ev.get("is_virtual") or ev.get("hide_from_listings"):
            continue
        start = _start(ev)
        if start and is_late_night(start):
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        item: dict = {"title": title}
        url = normalise_url(ev.get("url"))
        if url:
            item["url"] = url
        if start:
            item["start_date"] = start[:10]
        end = str(ev.get("end_date") or "")[:10]
        if end and end != item.get("start_date"):
            item["end_date"] = end

        description = _wp_text(ev.get("description"))
        if description:
            item["description"] = description

        image = ev.get("image")
        image_url = normalise_url(image.get("url")) if isinstance(image, dict) else None
        if image_url:
            item["image_url"] = image_url

        # Lowest listed price; "Free" comes through as ["0"].
        values = (ev.get("cost_details") or {}).get("values") if isinstance(ev.get("cost_details"), dict) else None
        prices = []
        for v in values or []:
            try:
                prices.append(float(str(v).replace(",", ".")))
            except ValueError:
                pass
        if prices:
            item["price"] = min(prices)

        venue = ev.get("venue") if isinstance(ev.get("venue"), dict) else {}
        location = ", ".join(
            _wp_text(p) for p in (venue.get("venue"), venue.get("address"), venue.get("city")) if p
        )
        if location:
            item["location"] = location
        if venue.get("country"):
            item["country"] = _wp_text(venue["country"])
        if venue.get("zip"):
            item["postal_code"] = str(venue["zip"]).strip()
        # The site's own categories aren't used as tags: on some sites they're
        # topics ("Stand-Up Comedy"), on others ticketing labels ("abono",
        # "carné joven") or month names.

        out.append(item)
        if len(out) >= max_events:
            break
    return out


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


# ── Page text and metadata (used by application/extraction.py, enrichment.py,
#    source_discovery.py) ──

# A bit larger than before because we now keep link/image URLs inline in the text.
_MAX_HTML_CHARS = 60_000


def _attr(tag: str, name: str) -> str | None:
    """Read an attribute value out of a raw start-tag string."""
    m = (re.search(rf'{name}\s*=\s*"([^"]*)"', tag, re.IGNORECASE)
         or re.search(rf"{name}\s*=\s*'([^']*)'", tag, re.IGNORECASE))
    return m.group(1) if m else None


def html_to_text(html: str, base_url: str | None = None) -> str:
    """Convert HTML to plain text WHILE preserving link and image URLs inline.

    Anchors become ``label [link: URL]`` and images become ``[image: URL]`` so the
    LLM can associate each event with its dedicated page, ticket link, and photo.
    Relative URLs are resolved against *base_url* and validated via ``normalise_url``.

    (The old version stripped every tag first, which destroyed all href/src — the
    model then had no URLs to extract, so url/ticket_url/image_url were mostly empty.)
    """
    # Drop non-content elements entirely (including their contents).
    html = re.sub(r'<(script|style|noscript|template|svg)\b[^>]*>.*?</\1>', ' ',
                  html, flags=re.DOTALL | re.IGNORECASE)

    # Images → " [image: URL] "
    def _img(m: re.Match) -> str:
        src = _attr(m.group(0), 'src') or _attr(m.group(0), 'data-src')
        u = normalise_url(src, base_url) if src else None
        return f" [image: {u}] " if u else " "
    html = re.sub(r'<img\b[^>]*>', _img, html, flags=re.IGNORECASE)

    # Anchors → " label [link: URL] "
    def _a(m: re.Match) -> str:
        href = _attr(m.group(1), 'href')
        label = re.sub(r'<[^>]+>', ' ', m.group(2))
        u = normalise_url(href, base_url) if href else None
        return f" {label} [link: {u}] " if u else f" {label} "
    html = re.sub(r'<a\b([^>]*)>(.*?)</a>', _a, html, flags=re.IGNORECASE | re.DOTALL)

    # Strip whatever tags remain, collapse whitespace, keep the START of the page.
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:_MAX_HTML_CHARS]


def plan_from_ld_event(
    ev: dict, source_url: str, source_type: str, category: str | None,
) -> PlanData | None:
    if not ev.get("title"):
        return None
    return PlanData(
        title=ev["title"],
        description=ev.get("description") or "",
        start_date=ev.get("start_date"),
        end_date=ev.get("end_date"),
        url=ev.get("url"),
        ticket_url=ev.get("ticket_url"),
        location=ev.get("location"),
        image_url=ev.get("image_url"),
        price=ev.get("price"),
        category=category,
        source_url=source_url,
        source_type=source_type,
        country=ev.get("country"),
        postal_code=ev.get("postal_code"),
    )


# ── Image (og:image / twitter:image / image_src) ─────────
_IMAGE_METAS = [
    re.compile(r'<meta\s[^>]*property=["\']og:image(?::url)?["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image(?::url)?["\']', re.IGNORECASE),
    re.compile(r'<meta\s[^>]*name=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<link\s[^>]*rel=["\']image_src["\'][^>]*href=["\']([^"\']+)["\']', re.IGNORECASE),
]

# ── Description (og:description / meta description) ──────
_DESCRIPTION_METAS = [
    re.compile(r'<meta\s[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\']', re.IGNORECASE),
    re.compile(r'<meta\s[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']', re.IGNORECASE),
]

# ── Ticket / purchase link detection ─────────────────────
_TICKET_HREF = re.compile(
    r'(entrad|ticket|comprar|checkout|/buy\b|boleto|taquilla|reserva|'
    r'pillalas|ticketmaster|entradas\.com|eventbrite|tiqets|wegow|seetickets|dice\.fm|fever)',
    re.IGNORECASE,
)
_TICKET_TEXT = re.compile(
    r'(entrad|comprar|tickets?|\bbuy\b|\bbook\b|reserva|adquirir|taquilla|sacar entradas)',
    re.IGNORECASE,
)


def extract_og_image(html: str, base_url: str | None = None) -> str | None:
    for rx in _IMAGE_METAS:
        m = rx.search(html)
        if m:
            u = normalise_url(m.group(1), base_url=base_url)
            if u:
                return u
    return None


def extract_meta_description(html: str) -> str | None:
    """Generic fallback: an event's own page often carries a real, per-event
    description in its og:description/meta description tag even when the
    listing page it was discovered on has none. Not every site has this (e.g.
    dice.fm's event pages carry neither tag at all — those need their own
    dedicated extraction), so this is best-effort and returns None rather than
    guessing when nothing is found.
    """
    for rx in _DESCRIPTION_METAS:
        m = rx.search(html)
        if m:
            text = unescape(m.group(1)).strip()
            if text:
                return text
    return None


def extract_ticket_url(html: str, base_url: str | None = None) -> str | None:
    """Find a purchase/ticket link on an event's own page."""
    for m in re.finditer(
        r'<a\b[^>]*?href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html, re.IGNORECASE | re.DOTALL,
    ):
        href = m.group(1)
        label = re.sub(r'<[^>]+>', ' ', m.group(2))
        if _TICKET_HREF.search(href) or _TICKET_TEXT.search(label):
            u = normalise_url(href, base_url=base_url)
            if u:
                return u
    return None
