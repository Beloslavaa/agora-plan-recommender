import argparse
import asyncio
import logging

from agora.backend.config import settings
from agora.backend.ingestion.explorer import explore_for_plans
from agora.backend.ingestion.extractor import extract_plans_from_html
from agora.backend.ingestion.llm import get_llm_provider
from agora.backend.ingestion.schemas import PlanCategory, PlanData
from agora.backend.ingestion.search import get_search_provider
from agora.backend.ingestion.sources import (
    fetch_fixed_source_with_details,
    load_fixed_sources,
)
from agora.backend.ingestion.enrich import enrich_plans
from agora.backend.ingestion.validator import validate_and_filter
from agora.backend.ingestion.store import get_plan_count, pool, upsert_plans

logger = logging.getLogger(__name__)


_sem = asyncio.Semaphore(5)


async def run_fixed_pipeline(llm, only_names: set[str] | None = None) -> list[PlanData]:
    sources = load_fixed_sources()
    if not sources:
        logger.info("No fixed sources yet — run explorer mode first")
        return []
    plans: list[PlanData] = []
    for source in sources:
        if only_names and source.name not in only_names:
            continue
        try:
            logger.info("Scraping fixed source: %s", source.name)
            page_htmls = await fetch_fixed_source_with_details(source)

            async def _extract(html: str, page_url: str) -> list[PlanData]:
                async with _sem:
                    try:
                        # Use page_url (the actual page the HTML came from), not
                        # source.url. For JSON-LD detail pages these differ, and
                        # tagging every plan with the listing URL was wrong — it
                        # broke the UI's "view source" link and the point of
                        # fetching detail pages at all.
                        extracted = await extract_plans_from_html(
                            html, page_url, "fixed", llm,
                        )
                        return validate_and_filter(extracted)
                    except Exception as e:
                        logger.warning("  ✗ Extraction failed for %s: %s", page_url, e)
                        return []

            results = await asyncio.gather(
                *[_extract(html, url) for html, url in page_htmls],
                return_exceptions=True,
            )
            for extracted in results:
                if isinstance(extracted, Exception):
                    continue
                plans.extend(extracted)

            # Guard against exceptions in the results (gather returns them as
            # values because return_exceptions=True); len(Exception) would throw.
            total = sum(len(r) for r in results if not isinstance(r, Exception))
            logger.info("  → %d total plans from %s (%d pages)",
                        total, source.name, len(page_htmls))
        except Exception as e:
            logger.warning("  ✗ Failed to scrape %s: %s", source.name, e)
    return plans


async def run_exploratory_pipeline(llm, only_categories: set[str] | None = None) -> list[PlanData]:
    search_provider = get_search_provider(settings.search_provider)
    cats = {PlanCategory(c) for c in only_categories} if only_categories else None
    plans = await explore_for_plans(
        llm=llm,
        search_provider=search_provider,
        min_per_category=3,
        max_per_category=10,
        only_categories=cats,
    )
    return validate_and_filter(plans)


async def run_full_pipeline() -> list[PlanData]:
    llm = get_llm_provider()
    fixed = await run_fixed_pipeline(llm)
    logger.info("Fixed pipeline: %d plans", len(fixed))

    exploratory = await run_exploratory_pipeline(llm)
    logger.info("Exploratory pipeline: %d plans", len(exploratory))

    return validate_and_filter(fixed + exploratory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agora ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["explorer", "fixed", "full"],
        default="explorer",
        help="explorer (default): search categories & promote good sources | "
             "fixed: scrape promoted sources | "
             "full: both",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[c.value for c in PlanCategory],
        help="Only run specific categories (e.g. --only music_concerts fashion)",
    )
    parser.add_argument(
        "--source",
        nargs="*",
        help="Only scrape specific fixed sources by name (e.g. --source 'Cinesa' 'Sala Equis')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without calling external services",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.dry_run:
        if args.mode in ("explorer", "full"):
            cats = [PlanCategory(c) for c in args.only] if args.only else list(PlanCategory)
            print(f"Would explore {len(cats)} categories:")
            for cat in cats:
                print(f"  · {cat.value}")
        if args.mode in ("fixed", "full"):
            sources = load_fixed_sources()
            if args.source:
                sources = [s for s in sources if s.name in args.source]
            print(f"Would scrape {len(sources)} fixed sources:")
            for s in sources:
                print(f"  · {s.name} — {s.url}")
        return

    if args.mode == "fixed":
        llm = get_llm_provider()
        plans = asyncio.run(run_fixed_pipeline(llm, only_names=set(args.source) if args.source else None))
    elif args.mode == "explorer":
        llm = get_llm_provider()
        plans = asyncio.run(run_exploratory_pipeline(llm, only_categories=args.only))
    else:
        llm = get_llm_provider()
        plans = asyncio.run(run_full_pipeline())

    search_provider = get_search_provider()
    plans = asyncio.run(enrich_plans(plans, llm=llm, search=search_provider))
    inserted = upsert_plans(plans)
    total = get_plan_count()
    print(f"\nNew: {inserted}  Total in DB: {total}")
    for p in plans:
        print(f"  · {p.title} [{p.source_type}] — {p.location or 'N/A'}")
    pool.close()


if __name__ == "__main__":
    main()