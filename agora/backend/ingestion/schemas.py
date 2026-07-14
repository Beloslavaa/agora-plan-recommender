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

CATEGORY_EXAMPLES: dict[PlanCategory, list[str]] = {
    PlanCategory.art_exhibitions: [
        "contemporary art gallery opening madrid this month",
        "emerging artists exhibition madrid this weekend",
    ],
    PlanCategory.photography: [
        "photography exhibition madrid this month",
        "street photography show madrid upcoming",
    ],
    PlanCategory.cinema: [
        "indie film screening madrid this weekend",
        "film festival madrid upcoming",
    ],
    PlanCategory.music_concerts: [
        "live concert madrid this month tickets",
        "indie band playing madrid this weekend",
        "jazz night madrid upcoming",
    ],
    PlanCategory.fashion: [
        "clothing pop-up shop madrid this week",
        "fashion market madrid upcoming",
    ],
    PlanCategory.cultural: [
        "cultural event madrid this month",
        "theatre performance madrid upcoming",
    ],
}


class FixedSource(BaseModel):
    name: str
    url: str
    promoted_by: str | None = None


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


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
