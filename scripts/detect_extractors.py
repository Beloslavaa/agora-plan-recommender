"""Record on each fixed source how the fixed pipeline should read it.

For every source in data/fixed_sources.json, checks once whether its site
publishes events through the WordPress events API and stores the answer in
the source's `extractor` field ("wp_api" or "page"), so run_fixed_pipeline
doesn't probe the ~60 sites without one on every run. Run locally and commit
the updated file — in production that file ships inside the image, so it
can't be written by the pipeline itself.

Also reports which method the page chain lands on for each "page" source
(JSON-LD, dice.fm page data, or the LLM), without calling the LLM.

Run from the project root (same folder as main.py / data/):

    python scripts/detect_extractors.py            # only sources without the field
    python scripts/detect_extractors.py --all      # re-check every source
    python scripts/detect_extractors.py --dry-run  # report, write nothing
"""

import argparse
import asyncio
import logging
from collections import Counter
from urllib.parse import urlparse

from agora.backend.application.extraction import extract_plans_from_wp_api
from agora.backend.domain.event_parsing import (
    extract_dice_listing_events,
    extract_item_list_urls,
    extract_ld_events,
)
from agora.backend.domain.schemas import FixedSource
from agora.backend.domain.url_safety import effective_base_url
from agora.backend.infrastructure.http.fetcher import fetch_page
from agora.backend.infrastructure.persistence.json_files import load_fixed_sources, save_fixed_sources

_sem = asyncio.Semaphore(8)


async def _page_method(url: str) -> str:
    """Which step of extract_plans_from_html would handle this page."""
    try:
        html = await fetch_page(url)
    except Exception as e:
        return f"fetch failed ({type(e).__name__})"
    base = effective_base_url(html, url)
    if extract_item_list_urls(html, base_url=base) or extract_ld_events(html, base_url=base):
        return "json_ld"
    if urlparse(url).hostname == "dice.fm" and extract_dice_listing_events(html):
        return "dice"
    return "llm"


async def _detect(source: FixedSource) -> tuple[str, str]:
    async with _sem:
        if await extract_plans_from_wp_api(source.url, "fixed"):
            return "wp_api", "wp_api"
        return "page", await _page_method(source.url)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Record how each fixed source should be read")
    ap.add_argument("--all", action="store_true", help="re-check sources that already have a value")
    ap.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    sources = load_fixed_sources()
    todo = [s for s in sources if args.all or s.extractor is None]
    print(f"Checking {len(todo)} of {len(sources)} fixed sources")

    results = await asyncio.gather(*[_detect(s) for s in todo])

    methods: Counter[str] = Counter()
    changed = 0
    for source, (extractor, method) in zip(todo, results):
        methods[method] += 1
        mark = "" if source.extractor in (None, extractor) else f"  (was {source.extractor})"
        print(f"  {extractor:7s} {method:22s} {source.city:10s} {source.name[:50]}{mark}")
        if source.extractor != extractor:
            source.extractor = extractor
            changed += 1

    print("\nMethods: " + ", ".join(f"{m} {n}" for m, n in methods.most_common()))
    if args.dry_run:
        print(f"Would update {changed} source(s)")
        return
    if changed:
        save_fixed_sources(sources)
    print(f"Updated {changed} source(s) in data/fixed_sources.json")


if __name__ == "__main__":
    asyncio.run(main())
