"""Learn CSS selectors for fixed sources that are still read by the LLM.

For each source the page chain hands to the LLM (no WordPress API, no
JSON-LD, not dice.fm), asks the LLM once for selectors of its event cards,
checks them against the live page (enough cards, most with a readable date)
and stores the ones that pass in data/fixed_sources.json. From then on
run_fixed_pipeline reads that page without the LLM. Most pages that aren't a
card list (articles, guides) fail the check and stay on the LLM — but not
all: a "related posts" box or a 3-film carousel can pass it too, so review
a dry run first and save only the sources whose sample plans look right.

Run locally and commit the updated file — in production that file ships
inside the image. Run from the project root (same folder as main.py / data/):

    python scripts/learn_selectors.py --dry-run          # try every source without selectors
    python scripts/learn_selectors.py --names "Condeduque" "La Casa Encendida"   # learn + save these
"""

import argparse
import asyncio
import logging
from datetime import date
from urllib.parse import urlparse

from agora.backend.application.selector_learning import learn_selectors
from agora.backend.domain.card_selectors import plans_from_cards
from agora.backend.domain.event_parsing import extract_item_list_urls, extract_ld_events
from agora.backend.domain.schemas import FixedSource
from agora.backend.domain.url_safety import effective_base_url
from agora.backend.infrastructure.http.fetcher import fetch_page
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence.json_files import load_fixed_sources, save_fixed_sources

_sem = asyncio.Semaphore(4)


async def _learn(source: FixedSource, llm, today: date) -> tuple[FixedSource, object, str, list]:
    async with _sem:
        try:
            html = await fetch_page(source.url)
        except Exception as e:
            return source, None, f"fetch failed ({type(e).__name__})", []
        base = effective_base_url(html, source.url)
        if extract_ld_events(html, base_url=base) or extract_item_list_urls(html, base_url=base):
            return source, None, "has JSON-LD, no LLM to replace", []
        if urlparse(source.url).hostname == "dice.fm":
            return source, None, "dice.fm page data, no LLM to replace", []
        selectors, summary = await learn_selectors(html, source.url, llm, today)
        sample = plans_from_cards(html, selectors, source.url, "fixed", today)[:3] if selectors else []
        return source, selectors, summary, sample


async def main() -> None:
    ap = argparse.ArgumentParser(description="Learn CSS selectors for LLM-read fixed sources")
    ap.add_argument("--names", nargs="+", help="only these sources (exact names)")
    ap.add_argument("--all", action="store_true", help="re-learn sources that already have selectors")
    ap.add_argument("--dry-run", action="store_true", help="print results, write nothing")
    args = ap.parse_args()
    if not args.dry_run and not args.names:
        ap.error("saving needs --names: review a --dry-run first, then name the sources to keep")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    sources = load_fixed_sources()
    todo = [
        s for s in sources
        if s.extractor != "wp_api"
        and (not args.names or s.name in args.names)
        and (args.all or args.names or s.selectors is None)
    ]
    print(f"Learning selectors for {len(todo)} of {len(sources)} fixed sources")

    llm = get_llm_provider()
    today = date.today()
    results = await asyncio.gather(*[_learn(s, llm, today) for s in todo])

    learned = 0
    for source, selectors, summary, sample in results:
        mark = "✓" if selectors else "–"
        print(f"\n{mark} {source.city} | {source.name}\n    {summary}")
        if selectors:
            print(f"    {selectors.model_dump(exclude_none=True)}")
            for p in sample:
                print(f"      · {p.title[:60]} | {p.start_date} → {p.end_date} | {p.url}")
            source.selectors = selectors
            learned += 1

    print(f"\nLearned selectors for {learned} of {len(todo)} source(s)")
    if args.dry_run:
        print("Dry run — nothing written")
        return
    if learned:
        save_fixed_sources(sources)
        print("Saved to data/fixed_sources.json")


if __name__ == "__main__":
    asyncio.run(main())
