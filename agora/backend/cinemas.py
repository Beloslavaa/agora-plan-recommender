"""Cinema chain classification.

A single ingestion source (e.g. cinesrenoir.com) publishes many individual
movie showtimes as separate plans. The "for you" feed shows one card per
cinema instead of one per movie; tapping it opens that cinema's own list.
This is the map from a plan's source domain (key) to its display name.
"""

CINEMA_SOURCES: dict[str, str] = {
    "cibelesdecine.com": "Cibeles de Cine",
    "cinesrenoir.com": "Cines Renoir",
    "salaequis.es": "Sala Equis",
    "yelmocines.es": "Yelmo Cines Ideal",
    "cinesa.es": "Cinesa",
}
