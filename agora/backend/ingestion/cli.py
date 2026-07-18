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
    load_cities,
    load_fixed_sources,
)
from agora.backend.ingestion.enrich import enrich_plans
from agora.backend.ingestion.validator import validate_and_filter
from agora.backend.ingestion.store import get_plan_count, pool, upsert_plans

logger = logging.getLogger(__name__)


_sem = asyncio.Semaphore(5)


async def run_fixed_pipeline(llm, city: str, only_names: set[str] | None = None) -> list[PlanData]:
    sources = [s for s in load_fixed_sources() if s.city == city]
    if not sources:
        logger.info("No fixed sources yet for %s — run explorer mode first", city)
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
    # Stamped here (not asked of the LLM extractor) — this is the one place
    # that knows which city this whole run targeted.
    for p in plans:
        p.city = city
    return plans


async def run_exploratory_pipeline(llm, city: str, only_categories: set[str] | None = None) -> list[PlanData]:
    search_provider = get_search_provider(settings.search_provider)
    cats = {PlanCategory(c) for c in only_categories} if only_categories else None
    plans = await explore_for_plans(
        llm=llm,
        city=city,
        search_provider=search_provider,
        min_per_category=3,
        max_per_category=10,
        only_categories=cats,
    )
    return validate_and_filter(plans)


async def run_full_pipeline(city: str) -> list[PlanData]:
    llm = get_llm_provider()
    fixed = await run_fixed_pipeline(llm, city)
    logger.info("[%s] Fixed pipeline: %d plans", city, len(fixed))

    exploratory = await run_exploratory_pipeline(llm, city)
    logger.info("[%s] Exploratory pipeline: %d plans", city, len(exploratory))

    return validate_and_filter(fixed + exploratory)


async def _run_one_city(
    mode: str,
    city: str,
    only_categories: list[str] | None,
    only_names: set[str] | None,
) -> list[PlanData]:
    """Run the selected mode for a single city, then enrich the results.

    Split out from main() so a multi-city (cron) run can call this once per
    configured city, in its own try/except — one city's failure must never
    abort the rest of the run.
    """
    llm = get_llm_provider()
    if mode == "fixed":
        plans = await run_fixed_pipeline(llm, city, only_names=only_names)
    elif mode == "explorer":
        plans = await run_exploratory_pipeline(llm, city, only_categories=only_categories)
    else:
        plans = await run_full_pipeline(city)

    search_provider = get_search_provider()
    return await enrich_plans(plans, llm=llm, search=search_provider)


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
        "--city",
        nargs="*",
        help="Only run for these cities (e.g. --city Madrid Barcelona). Omit "
             "to run every city listed in data/cities.json — that's what an "
             "unattended/cron run should use.",
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

    cities = args.city if args.city else load_cities()
    if not cities:
        print("No cities configured — add some to data/cities.json (or pass --city).")
        return

    if args.dry_run:
        print(f"Would run for {len(cities)} cities: {', '.join(cities)}")
        for city in cities:
            print(f"\n[{city}]")
            if args.mode in ("explorer", "full"):
                cats = [PlanCategory(c) for c in args.only] if args.only else list(PlanCategory)
                print(f"  Would explore {len(cats)} categories:")
                for cat in cats:
                    print(f"    · {cat.value}")
            if args.mode in ("fixed", "full"):
                sources = [s for s in load_fixed_sources() if s.city == city]
                if args.source:
                    sources = [s for s in sources if s.name in args.source]
                print(f"  Would scrape {len(sources)} fixed sources:")
                for s in sources:
                    print(f"    · {s.name} — {s.url}")
        return

    only_names = set(args.source) if args.source else None
    for city in cities:
        try:
            plans = asyncio.run(_run_one_city(args.mode, city, args.only, only_names))
        except Exception as e:
            # One city's bad source / LLM hiccup must not take down the rest
            # of an unattended multi-city run.
            logger.error("[%s] pipeline failed: %s", city, e)
            continue
        inserted = upsert_plans(plans)
        print(f"\n[{city}] New: {inserted}  Scraped this run: {len(plans)}")
        for p in plans:
            print(f"  · {p.title} [{p.source_type}] — {p.location or 'N/A'}")

    total = get_plan_count()
    print(f"\nTotal in DB (all cities): {total}")
    pool.close()


if __name__ == "__main__":
    main()
