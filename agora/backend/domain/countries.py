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
    # Bulgarian ticketing sites list events across the Balkans too.
    "cyprus": "CY", "кипър": "CY", "chipre": "CY",
    "greece": "GR", "гърция": "GR", "grecia": "GR",
    "romania": "RO", "румъния": "RO", "rumanía": "RO", "rumania": "RO",
    "serbia": "RS", "сърбия": "RS",
    "turkey": "TR", "türkiye": "TR", "турция": "TR", "turquía": "TR",
    "north macedonia": "MK", "северна македония": "MK", "македония": "MK",
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


def country_in_location(location: str | None) -> str | None:
    """The country a free-text location ends with ("…, Limassol, Cyprus" →
    "CY"), or None. Only the LAST comma segment is read, where addresses put
    the country — a country name earlier on is usually a venue's name
    ("Casa Argentina", "Café México"), not where it is."""
    if not location:
        return None
    last = location.rsplit(",", 1)[-1].strip(" .")
    return _COUNTRY_NAMES.get(last.lower())


# A venue's postal code inside free-text location, recognised only in
# address position: right before the town name ("9000 Варна", "28004
# Madrid") or, in the US, right after the state ("New York, NY 10001").
# That context keeps street numbers and years ("бул. Княз Борис I 115",
# "Culturama 2026") from being read as one.
_POSTAL_IN_TEXT: dict[str, re.Pattern] = {
    "ES": re.compile(r"(?<!\d)(\d{5})\s+[^\W\d_]"),
    "BG": re.compile(r"(?<!\d)(\d{4})\s+[^\W\d_]"),
    "US": re.compile(r"\b[A-Z]{2}\s+(\d{5})(?:-\d{4})?\b"),
}


def postal_code_in_text(text: str | None, country: str) -> str | None:
    """The last address-position postal code in *text* for *country*'s
    format, or None (no text, no known format, or no match)."""
    pattern = _POSTAL_IN_TEXT.get(country)
    if not pattern or not text:
        return None
    found = pattern.findall(text)
    return found[-1] if found else None


def postal_code_fits(postal_code: str | None, country: str) -> bool | None:
    """True/False if *postal_code* does/doesn't fit *country*'s format; None
    when there's nothing to judge (no code, no digits in it, or no known
    format for that country)."""
    pattern = _POSTAL_PATTERNS.get(country)
    if not pattern or not postal_code or not re.search(r"\d", postal_code):
        return None
    return bool(pattern.search(postal_code))
