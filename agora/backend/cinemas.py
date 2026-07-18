"""Cinema chain classification.

A single ingestion source (e.g. cinesrenoir.com) publishes many individual
movie showtimes as separate plans. The "for you" feed shows one card per
cinema instead of one per movie; tapping it opens that cinema's own list.
This is the map from a plan's source domain (key) to its display name.
"""

CINEMA_SOURCES: dict[str, dict[str, str]] = {
    "cibelesdecine.com": {"name": "Cibeles de Cine", "city": "Madrid"},
    "cinesrenoir.com": {"name": "Cines Renoir", "city": "Madrid"},
    "salaequis.es": {"name": "Sala Equis", "city": "Madrid"},
    "yelmocines.es": {"name": "Yelmo Cines Ideal", "city": "Madrid"},
    "cinesa.es": {"name": "Cinesa", "city": "Madrid"},
}
