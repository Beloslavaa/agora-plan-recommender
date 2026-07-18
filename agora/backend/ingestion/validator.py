import logging
from datetime import date, timedelta
from urllib.parse import unquote, urlparse

from agora.backend.ingestion.schemas import PlanData

logger = logging.getLogger(__name__)

_MAX_FUTURE_YEARS = 3
_MIN_TITLE_LENGTH = 3

_BOILERPLATE_TITLES = {
    "event", "plan", "new event", "upcoming event", "tbd", "to be confirmed",
    "no title", "untitled", "coming soon",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"}


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


def validate_plan(plan: PlanData) -> tuple[bool, str | None]:
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
    # (cli.py/explorer.py), never asked of the LLM — an empty one here means
    # some call path forgot to stamp it, not a real data-quality issue.
    if not plan.city or not plan.city.strip():
        issues.append("city missing (ingestion pipeline bug, not scraped data)")

    # ── Ticket URL ───────────────────────────────
    if plan.ticket_url and not _is_valid_url(plan.ticket_url):
        issues.append("ticket_url invalid")

    # ── Image URL ────────────────────────────────
    if plan.image_url:
        if not _is_valid_url(plan.image_url):
            issues.append("image_url invalid")
        elif not _looks_like_image_url(plan.image_url):
            issues.append("image_url doesn't look like an image")

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


def validate_and_filter(plans: list[PlanData]) -> list[PlanData]:
    valid: list[PlanData] = []
    seen: set[tuple[str, str]] = set()

    for plan in plans:
        # dedup by (title, source_url)
        key = (plan.title.lower().strip(), plan.source_url)
        if key in seen:
            logger.debug("  ✗ Duplicate: %s", plan.title)
            continue
        seen.add(key)

        ok, reason = validate_plan(plan)
        if ok:
            valid.append(plan)
        else:
            logger.debug("  ✗ Rejected: %s — %s", plan.title, reason)

    dropped = len(plans) - len(valid)
    if dropped:
        logger.info("  Validator dropped %d/%d plans", dropped, len(plans))
    return valid