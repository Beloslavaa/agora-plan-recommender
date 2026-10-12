"""Country and language lookups: normalisation for the wrong-country check in
domain/validation.py, plus per-language words used by discovery and enrichment.

Sources state a plan's country in different forms — JSON-LD uses "ES" on
Eventbrite/Ticketmaster, "Spain" on Songkick, "España" on smaller Spanish
sites; the LLM extractor is asked for a 2-letter code. Everything is mapped
to an ISO 3166-1 alpha-2 code before comparing.
"""

import re

# Names seen (or likely) on scraped pages, lowercased. Not exhaustive on
# purpose: an unrecognised name normalises to None, which validation treats
# as "unknown → keep" — never as a mismatch.
_COUNTRY_NAMES: dict[str, str] = {
    "spain": "ES", "españa": "ES", "espana": "ES", "espanya": "ES", "espagne": "ES",
    "united states": "US", "united states of america": "US", "usa": "US",
    "u.s.": "US", "u.s.a.": "US", "estados unidos": "US", "eeuu": "US", "ee.uu.": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB", "reino unido": "GB",
    "france": "FR", "francia": "FR",
    "portugal": "PT",
    "italy": "IT", "italia": "IT",
    "germany": "DE", "alemania": "DE", "deutschland": "DE",
    "mexico": "MX", "méxico": "MX",
    "argentina": "AR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE", "perú": "PE",
    "venezuela": "VE",
    "philippines": "PH", "filipinas": "PH",
    "bulgaria": "BG", "българия": "BG", "bulgarie": "BG",
}

# Postal code formats, keyed by the EXPECTED country. Only used to spot a code
# that can't belong to that country (Spanish codes are 5 digits starting with
# a province number 01–52, so the US ZIP of Madrid, NM — 87010 — can't be one;
# Bulgarian codes are 4 digits). No US entry: a 5-digit ZIP pattern would also
# accept every Spanish code, so it could never reject anything useful.
_POSTAL_PATTERNS: dict[str, re.Pattern] = {
    "ES": re.compile(r"(?<!\d)(0[1-9]|[1-4]\d|5[0-2])\d{3}(?!\d)"),
    "BG": re.compile(r"(?<!\d)\d{4}(?!\d)"),
}


def country_code(value) -> str | None:
    """ISO alpha-2 code for a country name or code, or None if unrecognised.

    Accepts the shapes JSON-LD uses for addressCountry: a plain string, or a
    Country object ({"@type": "Country", "name": "Spain"}).
    """
    if isinstance(value, dict):
        value = value.get("name") or value.get("@id")
    if not isinstance(value, str):
        return None
    v = value.strip()
    if len(v) == 2 and v.isalpha():
        return v.upper()
    return _COUNTRY_NAMES.get(v.lower())


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "es": "Spanish", "ca": "Catalan", "pt": "Portuguese",
    "fr": "French", "it": "Italian", "de": "German", "bg": "Bulgarian",
}

# The word people search for alongside an event title to find where to buy —
# enrichment's URL lookup ("<title>" <venue> entradas).
_TICKETS_WORD: dict[str, str] = {
    "en": "tickets", "es": "entradas", "ca": "entrades", "pt": "bilhetes",
    "fr": "billets", "it": "biglietti", "de": "tickets", "bg": "билети",
}


def search_country(country: str | None) -> str | None:
    """The lowercase country code search engines take as `gl` ("es")."""
    code = country_code(country)
    return code.lower() if code else None


def language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def tickets_word(language: str | None) -> str:
    return _TICKETS_WORD.get(language or "en", "tickets")


def postal_code_fits(postal_code: str | None, country: str) -> bool | None:
    """True/False if *postal_code* does/doesn't fit *country*'s format; None
    when there's nothing to judge (no code, no digits in it, or no known
    format for that country)."""
    pattern = _POSTAL_PATTERNS.get(country)
    if not pattern or not postal_code or not re.search(r"\d", postal_code):
        return None
    return bool(pattern.search(postal_code))
