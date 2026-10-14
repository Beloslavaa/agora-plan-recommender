"""Rules for deciding what is a fixed source — which URLs can be one, which
overlap a source we already have, and whether a sampled page passes. Pure:
no I/O. Driven by application/source_discovery.py."""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.parse import urlparse, urlunparse

from pydantic import ValidationError

from agora.backend.domain.schemas import PlanCategory, PlanData
from agora.backend.domain.validation import validate_plan

SAMPLE_SIZE = 3
MIN_GOOD_EVENTS = 2
REJECTION_DAYS = 90

# Sites that are never a fixed source on any page. Fever is added by hand
# (its city category pages), never through search — its search hits are
# flat /m/<id> event pages that don't lead back to a category page.
SKIP_DOMAINS = {
    "instagram.com", "facebook.com", "fb.com", "tiktok.com", "x.com", "twitter.com",
    "youtube.com", "youtu.be", "linkedin.com", "pinterest.com", "reddit.com", "threads.net",
    "wikipedia.org", "wikivoyage.org",
    "google.com", "maps.app.goo.gl", "goo.gl",
    "tripadvisor.com", "tripadvisor.es", "yelp.com", "yelp.es",
    "viagogo.com", "viagogo.es", "stubhub.com", "stubhub.es", "ticketswap.com", "ticketswap.es",
    "feverup.com",
}

# Paths that are files, search results, tag pages, news or blog posts —
# never an events listing worth saving.
_NON_LISTING_PATH = re.compile(
    r"\.(pdf|docx?|xlsx?|jpe?g|png|gif|webp)$"
    r"|/(search|buscar|busqueda|tag|tags|etiqueta|news|noticias|blog)(/|$)"
    r"|/20\d\d/\d{1,2}(/|$)",
    re.IGNORECASE,
)

_CATEGORY_VALUES = {c.value for c in PlanCategory}

UrlKey = tuple[str, str]


def url_key(url: str) -> UrlKey:
    """(host without www, path without trailing slash) — what "the same
    page" means for overlap and memory. Scheme, query and fragment ignored."""
    p = urlparse(url)
    host = (p.hostname or "").lower().removeprefix("www.")
    return host, p.path.rstrip("/")


def overlaps(url: str, keys: list[UrlKey]) -> bool:
    """Same page as, inside, or a parent of an existing source: e.g. with
    eventbrite.es/d/madrid/music saved, /d/madrid/music/jazz and /d/madrid
    overlap it, but sibling /d/madrid/comedy doesn't."""
    host, path = url_key(url)
    for h, p in keys:
        if h != host:
            continue
        if path == p or path.startswith(p + "/") or p.startswith(path + "/"):
            return True
    return False


def is_skip_listed(url: str) -> bool:
    host, _ = url_key(url)
    return any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS)


def unsavable_reason(url: str) -> str | None:
    """Why *url* could never be saved as a source, or None."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return "not a web URL"
    if p.query:
        return "has a query string"
    if _NON_LISTING_PATH.search(p.path):
        return "file, search, tag, news or blog URL"
    return None


def parent_candidates(url: str) -> list[str]:
    """Closest parent path, then the site's root — e.g. for
    site.com/agenda/conciertos/jazz-night: site.com/agenda/conciertos/,
    site.com/."""
    p = urlparse(url)
    segments = [s for s in p.path.split("/") if s]
    out = []
    if len(segments) > 1:
        out.append("/" + "/".join(segments[:-1]) + "/")
    if segments:
        out.append("/")
    return [urlunparse((p.scheme, p.netloc, path, "", "", "")) for path in out]


def recently_rejected(entry: dict | None, today: date) -> bool:
    """Whether a data/rejected_sources.json entry still applies."""
    return bool(entry) and entry["rejected_on"] > (today - timedelta(days=REJECTION_DAYS)).isoformat()


@dataclass
class Verdict:
    """What a sampled page is, and whether it's promoted."""
    page_type: str
    listing_link: str | None
    passed: bool
    reason: str = ""  # why it didn't pass
    name: str = ""
    good_events: list[PlanData] = field(default_factory=list)
    category: str | None = None  # most common among good_events
    is_general_listing: bool = False


def judge_sample(sample: dict, url: str, city: str, city_countries: dict[str, str]) -> Verdict:
    """Judge the LLM's sample of a page (see source_discovery.SAMPLE_SYSTEM).

    Passes when the page is a listing, at least MIN_GOOD_EVENTS of its
    sampled events are good (see _good_events), and the URL can be saved.
    """
    page_type = str(sample.get("page_type") or "other")
    listing_link = sample.get("listing_link") or None
    if page_type != "listing":
        return Verdict(page_type, listing_link, False, f"not a listing ({page_type})")

    good, problems = _good_events(sample.get("events"), url, city, city_countries)
    if len(good) < MIN_GOOD_EVENTS:
        detail = "; ".join(problems[:SAMPLE_SIZE]) or "no events found"
        return Verdict(page_type, listing_link, False,
                       f"{len(good)}/{SAMPLE_SIZE} sampled events good: {detail}")
    if why := unsavable_reason(url):
        return Verdict(page_type, listing_link, False, f"good listing, but can't be saved: {why}")

    cats = Counter(p.category for p in good)
    return Verdict(
        page_type, listing_link, True,
        name=str(sample.get("name") or "").strip()[:80] or url_key(url)[0],
        good_events=good,
        category=cats.most_common(1)[0][0],
        # Sampled events in different categories → a multi-category page,
        # whose plans mustn't all be stamped with one (see FixedSource).
        is_general_listing=len(cats) > 1,
    )


def _good_events(
    raw_events, url: str, city: str, city_countries: dict[str, str],
) -> tuple[list[PlanData], list[str]]:
    """The sampled events that are valid, in the city, fit one of the 12
    categories, and are distinct — plus why each of the others isn't."""
    good: list[PlanData] = []
    problems: list[str] = []
    titles: set[str] = set()
    for ev in (raw_events if isinstance(raw_events, list) else [])[:SAMPLE_SIZE]:
        if not isinstance(ev, dict):
            continue
        title = str(ev.get("title") or "").strip()
        try:
            plan = PlanData(
                title=title,
                description=str(ev.get("description") or ""),
                start_date=ev.get("start_date") or None,
                end_date=ev.get("end_date") or None,
                location=ev.get("location") or None,
                country=ev.get("country") or None,
                category=ev.get("category") or None,
                source_url=url,
                source_type="exploratory",
                city=city,
            )
        except ValidationError:
            problems.append(f"{title[:40]!r}: unreadable fields")
            continue
        ok, why = validate_plan(plan, city_countries)
        if not ok:
            problems.append(f"{title[:40]!r}: {why}")
        elif ev.get("in_city") is not True:
            problems.append(f"{title[:40]!r}: not in {city}")
        elif plan.category not in _CATEGORY_VALUES:
            problems.append(f"{title[:40]!r}: fits no category")
        elif title.lower() in titles:
            problems.append(f"{title[:40]!r}: repeats another sampled event")
        else:
            titles.add(title.lower())
            good.append(plan)
    return good, problems
