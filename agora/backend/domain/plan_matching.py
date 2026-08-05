"""Dedup/merge rules for incoming plans. Pure: plain dicts/PlanData in,
plain dicts out, no I/O — store.py owns turning these decisions into SQL."""

from agora.backend.domain.schemas import PlanData

# Fields backfilled when a plan matches an existing row by url (see
# compute_url_merge_updates) — deliberately excludes title/start_date/
# end_date/tags, which get their own handling below/in the caller.
URL_MERGE_FIELDS = (
    "short_title", "description", "ticket_url", "location",
    "image_url", "price", "category",
)


def _title_words(t: str) -> set[str]:
    return set("".join(c for c in t.lower() if c.isalnum() or c.isspace()).split())


def same_event_by_title(a: str, b: str, threshold: float = 0.6) -> bool:
    """True if the SHORTER title's words are mostly contained in the longer
    one — catches "Artist" vs "Artist at Venue" style pairs. Deliberately
    looser than an exact match (that's what the title/city/date key is for)
    but still guards against merging two different things that happen to
    share a url — see compute_url_merge_updates below."""
    wa, wb = _title_words(a), _title_words(b)
    smaller, larger = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if not smaller:
        return False
    return len(smaller & larger) / len(smaller) >= threshold


def compute_url_merge_updates(existing: dict, p: PlanData) -> dict:
    """Business rule for merging `p` into `existing` when they share a url +
    city (same canonical page) but arrived with different title text — e.g.
    a bare artist name from one source vs "Artist at Venue" from another,
    which the title/city/date key alone would treat as two different plans.

    Returns {column: new_value} to write — empty if nothing changed. Prefers
    the longer/more descriptive title.
    """
    updates: dict = {}
    if len(p.title) > len(existing["title"]):
        updates["title"] = p.title
        if p.short_title:
            updates["short_title"] = p.short_title
    for field in URL_MERGE_FIELDS:
        if not existing.get(field) and getattr(p, field, None):
            updates[field] = getattr(p, field)
    # start_date/end_date are REFRESHED here, not just backfilled-if-empty —
    # unlike the title/city/date match path (where a match means the date is
    # already identical by definition), a url match can be an evergreen
    # per-item page (a movie's own listing) rescraped with a NEW current
    # date each time. Backfill-only-if-empty froze it at whatever date was
    # first captured, so the row silently went stale forever even while the
    # real thing was still showing — confirmed on Cines Renoir movies still
    # playing but stuck on their original scrape date. Reset is_stale too:
    # a fresh scrape finding a current date means it isn't stale anymore,
    # and mark_stale_plans() never un-marks it on its own.
    new_start = p.start_date.isoformat() if p.start_date else None
    if new_start and new_start != existing.get("start_date"):
        updates["start_date"] = new_start
        if existing.get("is_stale"):
            updates["is_stale"] = False
    new_end = p.end_date.isoformat() if p.end_date else None
    if new_end and new_end != existing.get("end_date"):
        updates["end_date"] = new_end
    return updates
