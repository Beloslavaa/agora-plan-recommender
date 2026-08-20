"""Dedup/merge rules for incoming plans. Pure: plain dicts/PlanData in,
plain dicts out, no I/O — store.py owns turning these decisions into SQL."""

import re
import unicodedata
from difflib import SequenceMatcher

from agora.backend.domain.schemas import PlanData

# Fields backfilled when a plan matches an existing row (see
# compute_merge_updates) — deliberately excludes title/start_date/
# end_date/tags, which get their own handling below/in the caller.
URL_MERGE_FIELDS = (
    "short_title", "description", "ticket_url", "location",
    "image_url", "price", "category",
)

# Plus the dates, for collapsing two already-persisted rows — safe to
# backfill a missing date there since dates_compatible already ruled out
# a real conflicting date.
DEDUP_MERGE_FIELDS = URL_MERGE_FIELDS + ("start_date", "end_date")

_DIGIT_RE = re.compile(r"\d+")

# Skipped when picking a title's leading word, so articles like "the"/"la"
# aren't treated as the identifying word.
_LEADING_STOPWORDS = {"the", "a", "an", "el", "la", "los", "las"}


def _normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def _tokens(t: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in t).split()


def _leading_word(tokens: list[str]) -> str | None:
    return next((w for w in tokens if w not in _LEADING_STOPWORDS), None)


def fuzzy_same_event_by_title(a: str, b: str, threshold: float = 0.93) -> bool:
    """True if two titles are the same event written slightly differently
    (accents, punctuation, whitespace, a stray typo) — deliberately
    conservative, since callers use this to auto-merge/delete rows:

    - mismatched digit tokens veto the match outright ("Toy Story 4" vs "5").
    - the titles' leading (non-stopword) word must also match closely on its
      own. A long shared suffix can inflate the overall ratio even when the
      identifying word differs — real example: "Madrid International Film
      Festival" vs "FILMADRID International Film Festival" scores 0.958
      overall despite being two unrelated festivals, since "madrid" is a
      literal substring of "filmadrid".
    - the overall character-ratio bar defaults high enough that only
      near-identical strings pass.
    """
    na, nb = _normalize_title(a), _normalize_title(b)
    if set(_DIGIT_RE.findall(na)) != set(_DIGIT_RE.findall(nb)):
        return False
    lead_a, lead_b = _leading_word(_tokens(na)), _leading_word(_tokens(nb))
    if lead_a != lead_b and (not lead_a or not lead_b or SequenceMatcher(None, lead_a, lead_b).ratio() < 0.85):
        return False
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def dates_compatible(a: str | None, b: str | None) -> bool:
    """True if two plans' dates don't rule out being the same event: equal,
    or one/both not yet known (TBA). Two different real dates never are —
    that's the legitimate recurring-showtime case, which must stay as
    separate rows."""
    return a is None or b is None or a == b


def _title_words(t: str) -> set[str]:
    return set("".join(c for c in t.lower() if c.isalnum() or c.isspace()).split())


def same_event_by_title(a: str, b: str, threshold: float = 0.6) -> bool:
    """True if the SHORTER title's words are mostly contained in the longer
    one — catches "Artist" vs "Artist at Venue" style pairs. Deliberately
    looser than an exact match (that's what the title/city/date key is for)
    but still guards against merging two different things that happen to
    share a url — see compute_merge_updates below."""
    wa, wb = _title_words(a), _title_words(b)
    smaller, larger = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if not smaller:
        return False
    return len(smaller & larger) / len(smaller) >= threshold


def compute_merge_updates(existing: dict, p: PlanData) -> dict:
    """Business rule for merging incoming plan `p` into already-stored row
    `existing`, once the caller has decided they're the same event — either
    a url+city match (see same_event_by_title) or a fuzzy/exact title match
    with compatible dates (see fuzzy_same_event_by_title/dates_compatible).

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


def compute_dedup_merge_updates(keep: dict, drop: dict) -> dict:
    """Like compute_merge_updates, but for collapsing two ALREADY-PERSISTED
    rows a dedup pass matched, both plain DB rows (string dates). Returns
    {column: new_value} to write onto `keep`; un-stales it if `drop` was
    seen fresh.
    """
    updates: dict = {}
    if len(drop.get("title") or "") > len(keep.get("title") or ""):
        updates["title"] = drop["title"]
        if drop.get("short_title"):
            updates["short_title"] = drop["short_title"]
    for field in DEDUP_MERGE_FIELDS:
        if not keep.get(field) and drop.get(field):
            updates[field] = drop[field]
    if keep.get("is_stale") and not drop.get("is_stale"):
        updates["is_stale"] = False
    return updates
