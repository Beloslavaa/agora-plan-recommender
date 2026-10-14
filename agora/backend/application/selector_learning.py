"""Learn a listing page's CSS selectors once, so later runs read its event
cards without the LLM (see domain/card_selectors.py, which also holds the
page skeleton and the check below).

The LLM sees a skeleton of the page — tags, classes and a little text, with
repeated cards collapsed to a few examples — and names the selectors. They
are then checked against the real page before anyone keeps them.
"""

import logging
from datetime import date

from agora.backend.application.ports import LLMProvider
from agora.backend.domain.card_selectors import check_selectors, page_skeleton
from agora.backend.domain.schemas import CardSelectors

logger = logging.getLogger(__name__)

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
