"""Use-cases for managing the fixed-source list: promoting a newly-found
site, and correcting a plan's city from its scraped location text. Both
orchestrate the JSON-file persistence in infrastructure/persistence/json_files.py."""

from agora.backend.domain.schemas import FixedSource
from agora.backend.infrastructure.persistence.json_files import load_cities, load_fixed_sources, save_fixed_sources


def correct_city_from_location(plans: list, assumed_city: str) -> None:
    """Override plan.city in place when the LLM-extracted location text
    clearly names a DIFFERENT known city than the one this pipeline run
    assumed.

    A single source URL is sometimes a multi-city chain (e.g. a cinema chain
    with branches in more than one city) scraped under only one city's fixed-
    source/search run — every plan from it would otherwise get blanket-
    stamped with that one assumed city regardless of which branch it's
    actually at, even though location almost always names the real city.

    Also matches a bare 3-letter city code as the location's trailing
    comma-segment (e.g. "..., BAR", "..., MAD") — cinesrenoir.com's own
    JSON-LD address data uses these instead of full city names for some
    branches, confirmed by fetching its pages directly (addressLocality:
    "BAR" / "MAD", never the full name). Deliberately checked only as the
    LAST comma-separated segment, not a bare substring/word search anywhere
    in the text — "BAR" is also just the English word for a pub, and shows
    up in plenty of real Madrid venue names ("Ella Sky Bar", "Hartem Bar"),
    which don't have it isolated as its own address segment the way a
    genuine city-code abbreviation does.
    """
    others = [c for c in load_cities() if c.lower() != assumed_city.lower()]
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


def promote_source(name: str, url: str, city: str, promoted_by: str | None = None) -> bool:
    sources = load_fixed_sources()
    if any(s.url == url for s in sources):
        return False
    sources.append(FixedSource(name=name, url=url, city=city, promoted_by=promoted_by))
    save_fixed_sources(sources)
    return True
