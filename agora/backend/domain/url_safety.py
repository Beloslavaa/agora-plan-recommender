"""Content-safety rules for scraped data: what counts as a late-night event,
what counts as a usable URL. Pure — no network/filesystem access. See
infrastructure/http/fetcher.py for the SSRF-guarded *fetching* rules, which
are a separate (infra) concern from validating a URL string's shape."""

import logging
import re
from datetime import datetime, time as _time
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Events starting at/after this local time are treated as club/party listings
# rather than shows, and dropped during ingestion.
_LATE_NIGHT_CUTOFF = _time(23, 29)


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


# These patterns never appear in legitimate event URLs.
_SUSPICIOUS_URL_PATTERNS = re.compile(
    r"(data:|javascript:|vbscript:|file:|ftp:)",
    re.IGNORECASE,
)

# Songkick serves a shared blank placeholder avatar (105 bytes, verified) for
# festival/series listings with no uploaded photo, at
# .../profile_images/events/<id>/huge_avatar — a real, loadable image URL, so
# nothing else here catches it. .../profile_images/artists/<id>/huge_avatar is
# a real per-artist photo and is NOT matched by this.
SONGKICK_BLANK_IMAGE = re.compile(
    r"sk-static\.com/images/media/profile_images/events/", re.IGNORECASE,
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
