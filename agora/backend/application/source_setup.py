"""Use-case for setting up how each fixed source is read, once, so
run_fixed_pipeline goes straight to the right method instead of trying
them all on every run.

For each source it records:
- `extractor`: "wp_api" when the site publishes its events through the
  WordPress events API, else "page".
- `selectors`: CSS selectors for the page's event cards, learned by the LLM
  (see selector_learning.py), only for pages that would otherwise go to the
  LLM extractor — JSON-LD and dice.fm pages are already read without it.

Results are written to data/fixed_sources.json, which ships inside the
image — so this runs before a deploy (the monthly source-discovery Action,
or locally), never from the deployed pipeline. Driven by
`main.py --mode setup-sources` (infrastructure/cli/ingest_cli.py).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal
from urllib.parse import urlparse

from agora.backend.application.extraction import extract_plans_from_wp_api
from agora.backend.application.ports import LLMProvider, PageFetcher, SourceStore
from agora.backend.application.selector_learning import learn_selectors
from agora.backend.domain.card_selectors import check_selectors, plans_from_cards
from agora.backend.domain.event_parsing import (
    extract_dice_listing_events,
    extract_item_list_urls,
    extract_ld_events,
)
from agora.backend.domain.schemas import CardSelectors, FixedSource, PlanData
from agora.backend.domain.url_safety import effective_base_url

logger = logging.getLogger(__name__)

_sem = asyncio.Semaphore(4)

# How run_fixed_pipeline ends up reading the source after setup.
Method = Literal["wp_api", "json_ld", "dice", "selectors", "llm", "fetch_failed"]


@dataclass
class SourceSetup:
    source: FixedSource
    method: Method
    # What happened to the selectors: kept, learned, or why there are none.
    note: str = ""
    sample: list[PlanData] = field(default_factory=list)


def _page_method(html: str, url: str) -> Method:
    """Which step of extract_plans_from_html would read this page."""
    base = effective_base_url(html, url)
    if extract_item_list_urls(html, base_url=base) or extract_ld_events(html, base_url=base):
        return "json_ld"
    if urlparse(url).hostname == "dice.fm" and extract_dice_listing_events(html):
        return "dice"
    return "llm"


async def _set_up(source: FixedSource, llm: LLMProvider, fetcher: PageFetcher, today: date) -> SourceSetup:
    """Detect the extractor and, for an LLM-read page, learn its selectors.
    Updates `source` in place."""
    async with _sem:
        if await extract_plans_from_wp_api(fetcher, source.url, "fixed"):
            source.extractor, source.selectors = "wp_api", None
            return SourceSetup(source, "wp_api")
        source.extractor = "page"
        try:
            html = await fetcher.fetch_page(source.url)
        except Exception as e:
            # Leave any selectors alone — a fetch hiccup says nothing about them.
            return SourceSetup(source, "fetch_failed", f"{type(e).__name__}")

        method = _page_method(html, source.url)
        if method != "llm":
            source.selectors = None
            return SourceSetup(source, method)

        if source.selectors:
            ok, summary = check_selectors(html, source.selectors, source.url, today)
            if ok:
                sample = plans_from_cards(html, source.selectors, source.url, "fixed", today)[:3]
                return SourceSetup(source, "selectors", f"kept: {summary}", sample)

        selectors, summary = await learn_selectors(html, source.url, llm, today)
        source.selectors = selectors
        if not selectors:
            return SourceSetup(source, "llm", f"no selectors: {summary}")
        sample = plans_from_cards(html, selectors, source.url, "fixed", today)[:3]
        return SourceSetup(source, "selectors", f"learned: {summary}", sample)


def sources_to_set_up(
    store: SourceStore,
    cities: list[str] | None = None,
    names: set[str] | None = None,
    redo: bool = False,
) -> list[FixedSource]:
    """Sources never set up (no `extractor` yet — e.g. just promoted by the
    explorer), or every source with `redo`; `names` picks sources by name
    whether or not they were set up before."""
    return [
        s for s in store.load_fixed_sources()
        if (not cities or s.city in cities)
        and (s.name in names if names else redo or s.extractor is None)
    ]


async def set_up_sources(
    llm: LLMProvider,
    fetcher: PageFetcher,
    store: SourceStore,
    cities: list[str] | None = None,
    names: set[str] | None = None,
    redo: bool = False,
    today: date | None = None,
) -> list[SourceSetup]:
    """Set up the selected sources (see sources_to_set_up) and save them to
    data/fixed_sources.json."""
    today = today or date.today()
    sources = store.load_fixed_sources()
    picked = {s.url for s in sources_to_set_up(store, cities, names, redo)}
    todo = [s for s in sources if s.url in picked]
    if not todo:
        return []
    logger.info("Setting up %d of %d fixed sources", len(todo), len(sources))
    results = list(await asyncio.gather(*[_set_up(s, llm, fetcher, today) for s in todo]))
    store.save_fixed_sources(sources)
    return results
