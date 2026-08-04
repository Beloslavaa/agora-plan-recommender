from datetime import date
from enum import Enum

from pydantic import BaseModel


class PlanCategory(str, Enum):
    art_exhibitions = "art_exhibitions"
    photography = "photography"
    cinema = "cinema"
    music_concerts = "music_concerts"
    fashion = "fashion"
    cultural = "cultural"


CATEGORY_LABELS: dict[PlanCategory, str] = {
    PlanCategory.art_exhibitions: "Art exhibitions and galleries",
    PlanCategory.photography: "Photo exhibitions and art",
    PlanCategory.cinema: "Cinema and film screenings",
    PlanCategory.music_concerts: "Music concerts and live shows",
    PlanCategory.fashion: "Fashion pop-ups and clothing events",
    PlanCategory.cultural: "Cultural events and performances",
}

# These are style examples for the LLM (query phrasing), not target content —
# the actual city is injected separately via {city} in explorer.QUERY_SYSTEM.
# Deliberately no city name here, so they don't bias generation toward
# whichever city happened to be used when these were written.
CATEGORY_EXAMPLES: dict[PlanCategory, list[str]] = {
    PlanCategory.art_exhibitions: [
        "contemporary art gallery opening this month",
        "emerging artists exhibition this weekend",
    ],
    PlanCategory.photography: [
        "photography exhibition this month",
        "street photography show upcoming",
    ],
    PlanCategory.cinema: [
        "indie film screening this weekend",
        "film festival upcoming",
    ],
    PlanCategory.music_concerts: [
        "live concert this month tickets",
        "indie band playing this weekend",
        "jazz night upcoming",
    ],
    PlanCategory.fashion: [
        "clothing pop-up shop this week",
        "fashion market upcoming",
    ],
    PlanCategory.cultural: [
        "cultural event this month",
        "theatre performance upcoming",
    ],
}


class FixedSource(BaseModel):
    name: str
    url: str
    city: str
    promoted_by: str | None = None
    # True for a source whose page mixes multiple event types (a general city
    # "what's on" guide/calendar), as opposed to one that's inherently
    # single-category (a cinema chain, an Eventbrite category page, a
    # concerts-only platform like Songkick/Dice). promoted_by only records
    # which category SEARCH discovered the source, not what every event ON
    # it actually is — trusting it for a general listing would blanket-tag
    # e.g. concerts and exhibitions scraped from it as whatever category
    # happened to find it first. See cli.py's run_fixed_pipeline.
    is_general_listing: bool = False


class PlanData(BaseModel):
    title: str
    short_title: str = ""
    description: str
    start_date: date | None = None
    end_date: date | None = None
    url: str | None = None        # dedicated event page (vs listing page)
    ticket_url: str | None = None
    location: str | None = None
    image_url: str | None = None
    price: float | None = None
    tags: list[str] = []
    category: str | None = None
    source_url: str
    source_type: str  # "fixed" or "exploratory"
    # Stamped by the ingestion orchestration layer (cli.py/explorer.py), never
    # asked of the LLM extractor — always non-empty by the time a plan is
    # validated/stored. Defaults "" only so PlanData stays constructible
    # before that stamping happens.
    city: str = ""


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
