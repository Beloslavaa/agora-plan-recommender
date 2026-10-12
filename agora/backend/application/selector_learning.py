"""Learn a listing page's CSS selectors once, so later runs read its event
cards without the LLM (see domain/card_selectors.py).

The LLM sees a skeleton of the page — tags, classes and a little text, with
repeated cards collapsed to a few examples — and names the selectors. They
are then checked against the real page before anyone keeps them.
"""

import logging
from collections import Counter
from datetime import date

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from agora.backend.application.ports import LLMProvider
from agora.backend.domain.card_selectors import apply_selectors, parse_date_range
from agora.backend.domain.schemas import CardSelectors

logger = logging.getLogger(__name__)

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


SELECTOR_SYSTEM = """You write CSS selectors that read the event cards on a web page.

The page is given as a SKELETON: the real HTML structure, with scripts removed,
text shortened, and long runs of identical cards cut to a few examples
(marked with a comment like "…40 more <div class=card>").

Return ONE JSON object:
{{
  "card": "selector matching ONE element per event, on the whole page",
  "title": "selector inside the card for the event title",
  "link": "selector inside the card for the <a> to the event's own page, or null",
  "date": "selector inside the card for the element holding its date(s), or null",
  "image": "selector inside the card for the <img> (or element with a background-image), or null",
  "location": "selector inside the card for the venue, or null",
  "description": "selector inside the card for a short description, or null"
}}

Rules:
- "card" must match the repeated event items in the main listing — NOT
  navigation, menus, footers, newsletters, sponsors or "related" widgets.
- If the page has several event lists, pick the selector that covers the
  main/largest one; a comma-separated selector may join lists with the same
  card layout.
- All other selectors are relative to the card. Use ":scope" when the card
  element itself is the field (e.g. the card IS the <a>).
- The date element is where the event's date or date range is written
  ("16 oct 2026 - 10 ene 2027", "Jueves 15 de octubre"). If the date is only
  inside a longer text, select that text.
- Prefer stable class names over position (avoid :nth-child unless needed).
  Ignore auto-generated class names like "css-1x2y3z" or "_a1b2c".
- Use only standard CSS selectors, no XPath, no :contains().
- Today is {today}."""


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


async def learn_selectors(
    html: str, source_url: str, llm: LLMProvider, today: date,
) -> tuple[CardSelectors | None, str]:
    """Ask the LLM for selectors, check them on the page, retry once with
    what went wrong. Returns (selectors or None, summary of the last check)."""
    system = SELECTOR_SYSTEM.format(today=today.isoformat())
    prompt = f"Page URL: {source_url}\n\nSKELETON:\n{page_skeleton(html)}"
    summary = "no attempt"
    for attempt in range(2):
        try:
            data = await llm.parse_json(prompt, system=system, temperature=0.1, max_tokens=2048)
            selectors = CardSelectors(**data)
        except Exception as e:
            summary = f"LLM gave no usable selectors: {e}"
            continue
        ok, summary = check_selectors(html, selectors, source_url, today)
        logger.info("  attempt %d: %s → %s", attempt + 1, selectors.model_dump(exclude_none=True), summary)
        if ok:
            return selectors, summary
        prompt += (f"\n\nYour previous answer {selectors.model_dump_json(exclude_none=True)} "
                   f"did not work on the real page: {summary}. Fix it.")
    return None, summary
