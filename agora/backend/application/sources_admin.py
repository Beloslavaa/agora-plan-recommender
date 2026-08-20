"""Use-cases for managing the fixed-source list: promoting a newly-found
site, and correcting a plan's city from its scraped location text. Both
orchestrate the JSON-file persistence in infrastructure/persistence/json_files.py."""

from agora.backend.domain.schemas import FixedSource
from agora.backend.infrastructure.persistence.json_files import load_cities, load_fixed_sources, save_fixed_sources


def correct_city_from_location(plans: list, assumed_city: str) -> None:
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
