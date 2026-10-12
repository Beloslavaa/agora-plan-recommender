"""File-backed persistence for fixed sources and the city list — plain JSON
files under data/, no database involved."""

import json
from pathlib import Path

from agora.backend.domain.schemas import City, FixedSource

SOURCES_FILE = Path("data/fixed_sources.json")
CITIES_FILE = Path("data/cities.json")


def load_fixed_sources() -> list[FixedSource]:
    if not SOURCES_FILE.exists():
        return []
    raw = json.loads(SOURCES_FILE.read_text())
    return [FixedSource(**item) for item in raw]


def save_fixed_sources(sources: list[FixedSource]) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_FILE.write_text(
        json.dumps([s.model_dump() for s in sources], indent=2, ensure_ascii=False)
    )


def load_cities() -> list[str]:
    if not CITIES_FILE.exists():
        return []
    return [c["name"] for c in json.loads(CITIES_FILE.read_text())]


def load_city_settings() -> dict[str, City]:
    """City name -> its full data/cities.json entry (country, language)."""
    if not CITIES_FILE.exists():
        return {}
    return {c["name"]: City(**c) for c in json.loads(CITIES_FILE.read_text())}


def load_city_countries() -> dict[str, str]:
    """City name -> country, from data/cities.json. Used by domain/
    validation.py's wrong-region filter."""
    if not CITIES_FILE.exists():
        return {}
    return {c["name"]: c["country"] for c in json.loads(CITIES_FILE.read_text())}
