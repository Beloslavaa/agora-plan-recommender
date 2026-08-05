import argparse
import asyncio
import logging

from agora.backend.application.ingestion import run_one_city
from agora.backend.application.recommendation import backfill_embeddings
from agora.backend.domain.schemas import PlanCategory
from agora.backend.infrastructure.persistence.json_files import load_cities, load_fixed_sources
from agora.backend.infrastructure.persistence.postgres_repository import (
    get_plan_count,
    mark_stale_plans,
    pool,
    upsert_plans,
)

logger = logging.getLogger(__name__)


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
            plans = asyncio.run(run_one_city(args.mode, city, args.only, only_names))
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

    embedded = backfill_embeddings()
    print(f"Embedded {embedded} plan(s) for the semantic recommender")

    staled = mark_stale_plans()
    print(f"Marked {staled} plan(s) stale (end date, or start date if no end date, in the past) — hidden from browsing, not deleted")

    pool.close()


if __name__ == "__main__":
    main()
