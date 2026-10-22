import logging
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import unquote, urlparse

from agora.backend.domain.countries import country_code, postal_code_fits
from agora.backend.domain.schemas import PlanData
from agora.backend.domain.url_safety import is_placeholder_image

logger = logging.getLogger(__name__)

_MAX_FUTURE_YEARS = 3
_MIN_TITLE_LENGTH = 3

_BOILERPLATE_TITLES = {
    "event", "plan", "new event", "upcoming event", "tbd", "to be confirmed",
    "no title", "untitled", "coming soon",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"}

# One image shared by this many differently-titled plans from the same source
# page is site chrome (a logo, spinner, lazy-load placeholder), not an event
# photo. Real posters reused across a show's dates/branches top out around 5
# in live data; the junk ones were 10-41.
_SHARED_IMAGE_MIN_TITLES = 6


def _wrong_country(plan: PlanData, expected_country: str | None) -> str | None:
    """Why *plan* is clearly in another country than its city's, or None.

    Catches name collisions like "Madrid" (Spain) vs "Madrid, NM" — a search
    for a Madrid film festival once surfaced the New Mexico one. Only rejects
    on positive evidence: a country the source states, or a postal code that
    can't exist in the expected country. Missing data → keep.
    """
    expected = country_code(expected_country)
    if not expected:
        return None
    stated = country_code(plan.country)
    if stated and stated != expected:
        return f"country {plan.country!r}, expected {expected}"
    if not stated and postal_code_fits(plan.postal_code, expected) is False:
        return f"postal code {plan.postal_code!r} isn't a {expected} one"
    return None


def _is_valid_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _looks_like_image_url(url: str | None) -> bool:
    if not url:
        return False
    # Decode first: image CDNs (Eventbrite's img.evbuc.com, imgix, etc.) commonly
    # serve "https://cdn/.../original.jpg" percent-encoded as their own path,
    # e.g. "/https%3A%2F%2Fcdn.evbuc.com%2Fimages%2F.../original.20260619..." —
    # matching against the raw (still-encoded) path missed these entirely.
    path = unquote(urlparse(url).path.lower())
    if any(ext in path for ext in _IMAGE_EXTENSIONS):
        return True
    if "/image" in path or "/photo" in path or "/img" in path:
        return True
    return False


def _is_boilerplate(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    return text.strip().lower() in _BOILERPLATE_TITLES


def validate_plan(plan: PlanData, city_countries: dict[str, str] | None = None) -> tuple[bool, str | None]:
    # Computed per call so validation stays correct in a long-running process.
    today = date.today()
    issues: list[str] = []

    # ── Title ────────────────────────────────────
    if not plan.title or len(plan.title.strip()) < _MIN_TITLE_LENGTH:
        issues.append("title missing or too short")
    elif _is_boilerplate(plan.title):
        issues.append(f"title is boilerplate: '{plan.title}'")

    # ── Description ──────────────────────────────
    # Not required — many listing pages have only titles and dates.

    # ── Source URL ────────────────────────────────
    if not _is_valid_url(plan.source_url):
        issues.append("source_url invalid")

    # ── City ─────────────────────────────────────
    # Safety net: city is stamped by the ingestion orchestration layer
    # (ingestion.py / source_discovery.py), never asked of the LLM — an empty one here means
    # some call path forgot to stamp it, not a real data-quality issue.
    if not plan.city or not plan.city.strip():
        issues.append("city missing (ingestion pipeline bug, not scraped data)")
    elif wrong := _wrong_country(plan, (city_countries or {}).get(plan.city)):
        issues.append(f"wrong country: {wrong} (location '{plan.location}')")

    # ── Ticket URL ───────────────────────────────
    # Both fields are optional everywhere else (nullable on PlanData, and
    # enrichment backfills a missing one later) — a malformed value just
    # means the extractor pointed at the wrong thing, not that the rest of
    # the plan is untrustworthy. Clear the field instead of rejecting the
    # whole plan over it.
    if plan.ticket_url and not _is_valid_url(plan.ticket_url):
        logger.debug("  ~ dropping invalid ticket_url: %s", plan.ticket_url)
        plan.ticket_url = None

    # ── Image URL ────────────────────────────────
    if plan.image_url:
        if not _is_valid_url(plan.image_url):
            logger.debug("  ~ dropping invalid image_url: %s", plan.image_url)
            plan.image_url = None
        elif not _looks_like_image_url(plan.image_url):
            logger.debug("  ~ dropping non-image image_url: %s", plan.image_url)
            plan.image_url = None
        elif is_placeholder_image(plan.image_url):
            logger.debug("  ~ dropping placeholder image_url: %s", plan.image_url)
            plan.image_url = None

    # ── Price ────────────────────────────────────
    if plan.price is not None and (plan.price < 0 or plan.price > 100_000):
        issues.append(f"suspicious price: {plan.price}")

    # ── Date validation ──────────────────────────
    if plan.start_date and plan.end_date:
        if plan.end_date < today:
            issues.append(f"ended {plan.end_date} (before today {today})")
        elif plan.start_date > today + timedelta(days=365 * _MAX_FUTURE_YEARS):
            issues.append(f"start_date {plan.start_date} too far in future")
    elif plan.start_date:
        if plan.start_date < today:
            issues.append(f"start_date {plan.start_date} is in the past")
        elif plan.start_date > today + timedelta(days=365 * _MAX_FUTURE_YEARS):
            issues.append(f"start_date {plan.start_date} too far in future")
    elif plan.end_date:
        if plan.end_date < today:
            issues.append(f"ended {plan.end_date} (before today {today})")

    if issues:
        return False, "; ".join(issues)
    return True, None


def validate_and_filter(plans: list[PlanData], city_countries: dict[str, str] | None = None) -> list[PlanData]:
    valid: list[PlanData] = []
    seen: set[tuple[str, str, object]] = set()

    for plan in plans:
        # Matches store.py's (title, city, start_date) DB key — the same
        # event scraped from several different pages within one run (a
        # chain's many category pages, say) has a different source_url each
        # time, so keying dedup on that let every one of them through.
        key = (plan.title.lower().strip(), plan.city, plan.start_date)
        if key in seen:
            logger.debug("  ✗ Duplicate: %s", plan.title)
            continue
        seen.add(key)

        ok, reason = validate_plan(plan, city_countries)
        if ok:
            valid.append(plan)
        else:
            logger.debug("  ✗ Rejected: %s — %s", plan.title, reason)

    dropped = len(plans) - len(valid)
    if dropped:
        logger.info("  Validator dropped %d/%d plans", dropped, len(plans))
    _drop_shared_images(valid)
    return valid


def _drop_shared_images(plans: list[PlanData]) -> None:
    """Clear image_url on plans whose image is shared by many differently-
    titled plans from the same source page — catches site chrome that
    PLACEHOLDER_IMAGE doesn't know about yet. Enrichment can then backfill
    the plan's own og:image."""
    titles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for p in plans:
        if p.image_url:
            titles[(p.source_url, p.image_url)].add(p.title.lower().strip())
    shared = {k for k, t in titles.items() if len(t) >= _SHARED_IMAGE_MIN_TITLES}
    if not shared:
        return
    for p in plans:
        if (p.source_url, p.image_url) in shared:
            p.image_url = None
    for source_url, image_url in shared:
        logger.info("  ~ dropping image shared by %d plans from %s: %s",
                    len(titles[(source_url, image_url)]), source_url, image_url)


def correct_city_from_location(plans: list[PlanData], assumed_city: str, cities: list[str]) -> None:
    """Override plan.city in place when the scraped location text clearly
    names a DIFFERENT known city than the one this pipeline run assumed —
    e.g. a cinema chain with branches in more than one city, scraped under
    only one city's run, would otherwise blanket-stamp every branch with it.

    Also matches a bare 3-letter city code as the location's trailing
    comma-segment (e.g. "..., BAR", "..., MAD") — cinesrenoir.com's own
    JSON-LD address data uses these for some branches instead of full city
    names. Checked only as the LAST comma segment, not a substring search —
    "BAR" is also just the English word for a pub, and shows up in real
    Madrid venue names ("Ella Sky Bar") that aren't a city-code match.
    """
    others = [c for c in cities if c.lower() != assumed_city.lower()]
    for p in plans:
        if not p.location:
            continue
        loc = p.location.lower()
        if assumed_city.lower() in loc:
            continue
        last_segment = loc.rsplit(",", 1)[-1].strip()
        for other in others:
            code = other.lower()[:3]
            if other.lower() in loc or last_segment == code:
                p.city = other
                break
