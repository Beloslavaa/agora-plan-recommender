"""Read a listing page's event cards with CSS selectors instead of the LLM.

A source's selectors are learned once (application/selector_learning.py) and
then applied here on every run: `card` matches one element per event, and
every other selector is relative to that card. Pure functions — no I/O.
"""

import calendar
import re
import unicodedata
from collections import Counter
from datetime import date

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from agora.backend.domain.schemas import CardSelectors, PlanData
from agora.backend.domain.url_safety import (
    IMAGE_SRC_ATTRS,
    is_late_night,
    is_placeholder_image,
    normalise_url,
)


def _text(el: Tag | None) -> str:
    return " ".join(el.get_text(" ").split()) if el else ""


def _select_one(card: Tag, selector: str | None) -> Tag | None:
    if not selector:
        return None
    # ":scope" lets a selector point at the card itself (e.g. the card IS the <a>).
    if selector.strip() == ":scope":
        return card
    try:
        return card.select_one(selector)
    except Exception:
        return None


def _link(el: Tag | None) -> str | None:
    if el is None:
        return None
    a = el if el.name == "a" else el.find("a", href=True)
    return a.get("href") if a else None


_BG_URL = re.compile(r"url\(['\"]?([^'\")]+)")


def _image(el: Tag | None) -> str | None:
    if el is None:
        return None
    img = el if el.name in ("img", "source") else el.find(["img", "source"])
    if img is not None:
        for attr in IMAGE_SRC_ATTRS:
            v = img.get(attr)
            if v and v.strip() and not v.startswith("data:"):
                v = v.split(",")[0].split()[0]
                if not is_placeholder_image(v):
                    return v
    m = _BG_URL.search(el.get("style") or "")
    return m.group(1) if m else None


def apply_selectors(html: str, selectors: CardSelectors, base_url: str) -> list[dict]:
    """One dict per matched card: title, url, date_text, image_url, location,
    description — raw text, not yet dates. [] if the card selector is invalid."""
    soup = BeautifulSoup(html, "html.parser")
    try:
        cards = soup.select(selectors.card)
    except Exception:
        return []
    # A location selector pointing at the date or description element (seen
    # when one text holds all three) would store that whole text as the venue.
    location_sel = selectors.location if selectors.location not in (selectors.date, selectors.description) else None
    out = []
    for card in cards:
        location = _text(_select_one(card, location_sel))
        date_el = _select_one(card, selectors.date)
        # A <time datetime="2026-10-16"> is more reliable than its label.
        date_text = (date_el.get("datetime") or _text(date_el)) if date_el else ""
        out.append({
            "title": _text(_select_one(card, selectors.title)),
            "url": normalise_url(_link(_select_one(card, selectors.link) if selectors.link else card),
                                 base_url=base_url),
            "date_text": date_text,
            "image_url": normalise_url(_image(_select_one(card, selectors.image)), base_url=base_url),
            "location": location or None,
            "description": _text(_select_one(card, selectors.description)),
        })
    return out


# ── Dates ─────────────────────────────────────────────────────────────────
# Spanish, English, Catalan and Bulgarian month names, matched on their first 3 letters
# (accents stripped). Catalan "de gener", "d'octubre" etc. are covered too.
_MONTHS = {
    "ene": 1, "jan": 1, "gen": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4, "apr": 4,
    "may": 5, "mai": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8, "aug": 8,
    "sep": 9, "set": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12, "dec": 12, "des": 12,
    # Bulgarian, after _fold (which turns "май" into "маи").
    "яну": 1, "фев": 2, "мар": 3, "апр": 4, "маи": 5, "юни": 6,
    "юли": 7, "авг": 8, "сеп": 9, "окт": 10, "ное": 11, "дек": 12,
}
# Full words only — "mar" alone must not match "Mar Menor", hence the list of
# real month spellings rather than any word starting with a prefix.
_MONTH_WORDS = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
    "|january|february|march|april|may|june|july|august|september|october|november|december"
    "|gener|febrer|marc|maig|juny|juliol|agost|setembre|octubre|novembre|desembre"
    "|ene|feb|mar|abr|apr|jun|jul|ago|aug|sep|sept|set|oct|nov|dic|dec|des|jan|gen"
    "|януари|февруари|март|април|маи|юни|юли|август|септември|октомври|ноември|декември"
    "|яну|фев|мар|апр|авг|сеп|окт|ное|дек"
)
_WEEKDAYS = (
    "lunes|martes|miercoles|jueves|viernes|sabado|domingo"
    "|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    "|dilluns|dimarts|dimecres|dijous|divendres|dissabte|diumenge"
    "|понеделник|вторник|сряда|четвъртък|петък|събота|неделя"
)
_DAY = r"(?<![\d:.,])([0-3]?\d)(?:st|nd|rd|th)?(?![\d:.]|\s*h\b)"
# A run of days before a month: "15, viernes 16 y sábado 17 de octubre",
# "3 al 15 de noviembre", "27". Only weekday names and a short note in
# brackets ("21 (¡nueva fecha!) y 22") may sit between days, so
# "Edición 3 desde el 12 oct" reads as the 12th, not the 3rd.
_DAY_RUN = re.compile(
    rf"((?:{_DAY}\s*(?:\([^)]{{0,30}}\)\s*)?(?:(?:,|y|i|and|&|al|a|to|и|до|-|–|—)\s*)?(?:(?:{_WEEKDAYS})\s+)?)+?)"
    rf"(?:de\s+|d')?\b({_MONTH_WORDS})\b\.?"
    r"(?:\s*(?:de|,)?\s*((?:19|20)\d\d))?"
)
# Month first: "October 16, 2026", "Oct 16".
_MONTH_DAY = re.compile(rf"\b({_MONTH_WORDS})\b\.?\s+([0-3]?\d)(?![\d:])(?:st|nd|rd|th)?(?:,?\s*((?:19|20)\d\d))?")
_NUMERIC = re.compile(r"(?<!\d)([0-3]?\d)[/.]([01]?\d)[/.]((?:19|20)\d\d)(?!\d)")
# "06.10-25.10.2026": a day.month range sharing one year at the end.
_NUMERIC_RANGE = re.compile(
    r"(?<!\d)([0-3]?\d)[/.]([01]?\d)\s*[-–—]\s*([0-3]?\d)[/.]([01]?\d)[/.]((?:19|20)\d\d)(?!\d)"
)
_ISO = re.compile(r"(?<!\d)((?:19|20)\d\d)-([01]\d)-([0-3]\d)")


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _month(word: str) -> int | None:
    return _MONTHS.get(word[:3])


def _mentions(text: str) -> list[tuple[int, int, int, int | None]]:
    """(position, month, day, year or None) for every day+month in *text*."""
    found = [(m.start(), int(m.group(2)), int(m.group(3)), int(m.group(1))) for m in _ISO.finditer(text)]
    for m in _NUMERIC_RANGE.finditer(text):
        d1, m1, d2, m2, y = (int(g) for g in m.groups())
        if 1 <= m1 <= 12 and 1 <= m2 <= 12:
            found += [(m.start(), m1, d1, y if m1 <= m2 else y - 1), (m.start() + 1, m2, d2, y)]
    if found:
        return sorted(found)
    for m in _NUMERIC.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.append((m.start(), mo, d, y))
    if found:
        return found
    day_first = []
    for m in _DAY_RUN.finditer(text):
        run, _, month_word, year = m.groups()
        mo = _month(month_word)
        days = [int(x) for x in re.findall(_DAY, run) if 1 <= int(x) <= 31]
        if mo and days:
            for d in (days[0], days[-1]):
                day_first.append((m.start(), mo, d, int(year) if year else None))
    month_first = []
    for m in _MONTH_DAY.finditer(text):
        mo, d = _month(m.group(1)), int(m.group(2))
        if mo and 1 <= d <= 31:
            month_first.append((m.start(), mo, d, int(m.group(3)) if m.group(3) else None))
    # A card uses one style throughout; whichever appears first decides, so
    # "Oct 16 - Nov 2" isn't also read as "16 Nov".
    if month_first and (not day_first or month_first[0][0] < day_first[0][0]):
        return month_first
    return day_first


def _make(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, min(day, calendar.monthrange(year, month)[1]))
    except ValueError:
        return None


def parse_date_range(text: str, today: date) -> tuple[date, date | None] | None:
    """(start, end) from a card's date text, end None for a single day.

    Handles "16 oct 2026 - 10 ene 2027", "Jueves 15, viernes 16 y sábado 17
    de octubre 2026", "2 oct - 31 oct", "27 May-12 Oct", "09/10/2026 —
    28/02/2027" and "October 16, 2026". A missing year is taken as the one
    that makes the range end no earlier than about a month ago — listings
    only show running or upcoming events. None when no day+month is found.
    """
    found = _mentions(_fold(text))
    if not found:
        return None
    first, last = found[0], found[-1]
    end_year = last[3] or first[3]
    if end_year is None:
        end_year = today.year
        probe = _make(end_year, last[1], last[2])
        if probe and (today - probe).days > 31:
            end_year += 1
    end = _make(end_year, last[1], last[2])
    start_year = first[3] or (end_year if (first[1], first[2]) <= (last[1], last[2]) else end_year - 1)
    start = _make(start_year, first[1], first[2])
    if not start or not end:
        return None
    if end < start:
        return None
    return start, (end if end != start else None)


def plans_from_cards(
    html: str,
    selectors: CardSelectors,
    source_url: str,
    source_type: str,
    today: date,
    category: str | None = None,
) -> list[PlanData]:
    """Plans from the cards the selectors find. A card whose date can't be
    read keeps no date, like the LLM does for "Tuesday to Sunday" or "dates to
    be confirmed"; one without a title is skipped."""
    plans = []
    for card in apply_selectors(html, selectors, base_url=source_url):
        if not card["title"] or is_late_night(card["date_text"]):
            continue
        start, end = parse_date_range(card["date_text"], today) or (None, None)
        plans.append(PlanData(
            title=card["title"],
            description=card["description"],
            start_date=start,
            end_date=end,
            url=card["url"],
            location=card["location"],
            image_url=card["image_url"],
            category=category,
            source_url=source_url,
            source_type=source_type,
        ))
    return plans


# ── Learning support (used by application/selector_learning.py) ──

_DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "head", "form", "button", "select")
_KEEP_ATTRS = ("class", "id", "href", "src", "data-src", "datetime", "style")
_REPEATS_SHOWN = 3
_MAX_TEXT = 80
_MAX_SKELETON_CHARS = 60_000

# A source is only switched to selectors when they read most of its cards.
MIN_CARDS = 3
MIN_DATED_SHARE = 0.7


def _shape(el: Tag) -> tuple[str, str]:
    classes = el.get("class") or []
    return el.name, classes[0] if classes else ""


def page_skeleton(html: str) -> str:
    """The page's structure with its content trimmed: no scripts or styles,
    only the attributes a selector or a field needs, text cut to a few words,
    and runs of same-shaped siblings (the event cards) cut to 3 examples."""
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(_DROP_TAGS):
        el.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for el in soup.find_all(True):
        el.attrs = {k: (v if k != "style" else v[:120]) for k, v in el.attrs.items() if k in _KEEP_ATTRS}
    for el in soup.find_all(True):
        if el.decomposed:  # inside a run already cut below
            continue
        kids = [k for k in el.children if isinstance(k, Tag)]
        counts = Counter(_shape(k) for k in kids)
        seen: Counter = Counter()
        for k in kids:
            s = _shape(k)
            seen[s] += 1
            if counts[s] > _REPEATS_SHOWN and seen[s] > _REPEATS_SHOWN:
                if seen[s] == _REPEATS_SHOWN + 1:
                    k.replace_with(Comment(f" …{counts[s] - _REPEATS_SHOWN} more <{s[0]} class={s[1]}> "))
                else:
                    k.decompose()
    for s in soup.find_all(string=True):
        if isinstance(s, NavigableString) and not isinstance(s, Comment):
            t = " ".join(s.split())
            # Keep the start AND end: listings often put the date last
            # ("…(Reina Sofía Museum. 27 May-12 Oct).").
            if len(t) > _MAX_TEXT:
                t = t[:_MAX_TEXT // 2] + " … " + t[-_MAX_TEXT // 2:]
            s.replace_with(t)
    body = soup.body or soup
    return str(body)[:_MAX_SKELETON_CHARS]


def check_selectors(html: str, selectors: CardSelectors, source_url: str, today: date) -> tuple[bool, str]:
    """Whether the selectors read this page well enough to replace the LLM,
    and a one-line summary (also fed back to the LLM on a retry)."""
    cards = apply_selectors(html, selectors, base_url=source_url)
    if len(cards) < MIN_CARDS:
        return False, f"card selector matched {len(cards)} element(s), expected the event list"
    titled = [c for c in cards if c["title"]]
    dated = [c for c in titled if parse_date_range(c["date_text"], today)]
    share = len(dated) / len(cards)
    summary = (f"{len(cards)} cards, {len(titled)} with a title, {len(dated)} with a readable date"
               f" ({share:.0%}), {sum(1 for c in cards if c['url'])} with a link,"
               f" {sum(1 for c in cards if c['image_url'])} with an image")
    if share < MIN_DATED_SHARE:
        if not selectors.date:
            return False, (f"{summary}. No date selector was given, but every card needs its date — "
                           f"if the date is written inside another text (e.g. the description), "
                           f"use that element's selector as the date")
        samples = "; ".join(repr(c["date_text"][:60]) for c in cards[:5])
        return False, f"{summary}. Sample date texts: {samples}"
    return True, summary
