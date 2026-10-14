import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from agora.backend.application.ingestion import run_one_city
from agora.backend.application.source_discovery import (
    DISCOVERY_CATEGORIES,
    DiscoveryReport,
    discover_sources,
)
from agora.backend.application.source_setup import SourceSetup, set_up_sources, sources_to_set_up
from agora.backend.domain.schemas import PlanCategory
from agora.backend.infrastructure.persistence.json_files import load_cities, load_fixed_sources

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agora ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["explorer", "fixed", "setup-sources"],
        default="explorer",
        help="explorer (default): search categories & promote good sources | "
        "fixed: scrape promoted sources | "
        "setup-sources: record how each new fixed source is read (extractor, "
        "CSS selectors) in data/fixed_sources.json",
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
        help="explorer: only search these categories (e.g. --only music_concerts "
        "fashion) — default: " + ", ".join(c.value for c in DISCOVERY_CATEGORIES),
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
    parser.add_argument(
        "--redo",
        action="store_true",
        help="setup-sources: re-check every source, not only ones never set up",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="explorer / setup-sources: also write a Markdown summary here (the "
        "source-discovery Action puts it in its PR)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.mode == "setup-sources":
        _setup_sources(args)
        return

    cities = args.city if args.city else load_cities()
    if not cities:
        print("No cities configured — add some to data/cities.json (or pass --city).")
        return

    if args.mode == "explorer":
        _discover(args, cities)
        return

    if args.dry_run:
        print(f"Would run for {len(cities)} cities: {', '.join(cities)}")
        for city in cities:
            sources = [s for s in load_fixed_sources() if s.city == city]
            if args.source:
                sources = [s for s in sources if s.name in args.source]
            print(f"\n[{city}]\n  Would scrape {len(sources)} fixed sources:")
            for s in sources:
                print(f"    · {s.name} — {s.url}")
        return

    only_names = set(args.source) if args.source else None
    all_plans = []
    failed = 0
    for city in cities:
        try:
            plans = asyncio.run(run_one_city(city, only_names))
        except Exception as e:
            # One city's bad source / LLM hiccup must not take down the rest
            # of an unattended multi-city run.
            logger.error("[%s] pipeline failed: %s", city, e)
            failed += 1
            continue
        all_plans.extend(plans)
        print(f"\n[{city}] Scraped this run: {len(plans)}")
        for p in plans:
            print(f"  · {p.title} [{p.source_type}] — {p.location or 'N/A'}")

    # Partial failure stays tolerated above, but if *every* city blew up the
    # run accomplished nothing — exit non-zero so CI shows red instead of a
    # green check on a run that only produced stack traces.
    if failed == len(cities):
        logger.error("All %d city pipeline(s) failed — nothing was processed.", failed)
        sys.exit(1)

    from agora.backend.application.graph_recommendation import backfill_graph_embeddings
    from agora.backend.application.recommendation import backfill_embeddings
    from agora.backend.infrastructure.persistence.postgres_repository import (
        get_plan_count,
        mark_stale_plans,
        pool,
        upsert_plans,
    )

    inserted = upsert_plans(all_plans)
    total = get_plan_count()
    print(f"\nNew: {inserted}  Total in DB (all cities): {total}")

    embedded = backfill_embeddings()
    print(f"Embedded {embedded} plan(s) for the semantic recommender")

    staled = mark_stale_plans()
    print(
        f"Marked {staled} plan(s) stale (end date, or start date if no end date, in the past) — hidden from browsing, not deleted"
    )

    # After staling, so no proxy is spent on a plan that just went stale.
    # Cold-start proxies only, no retraining — see backfill_graph_embeddings.
    graphed = backfill_graph_embeddings()
    print(f"Gave {graphed} new plan(s) a cold-start graph embedding")

    pool.close()


def _discover(args: argparse.Namespace, cities: list[str]) -> None:
    categories = [PlanCategory(c) for c in args.only] if args.only else DISCOVERY_CATEGORIES
    if args.dry_run:
        print(f"Would discover sources for {len(cities)} cities: {', '.join(cities)}")
        print(f"  {len(categories)} categories × 2 queries = {2 * len(categories)} searches per city:")
        for cat in categories:
            print(f"    · {cat.value}")
        return

    from agora.backend.infrastructure.llm.providers import get_llm_provider
    from agora.backend.infrastructure.search.providers import get_search_provider

    async def _all() -> tuple[list[DiscoveryReport], int]:
        # One event loop for every city: LLM/HTTP clients bind to the loop
        # they were created on, so one asyncio.run per city breaks them.
        llm, search = get_llm_provider(), get_search_provider()
        reports, failed = [], 0
        for city in cities:
            try:
                reports.append(await discover_sources(llm, search, city, categories))
            except Exception as e:
                # One city's failure must not take down the rest of the run.
                logger.error("[%s] discovery failed: %s", city, e)
                failed += 1
        return reports, failed

    reports, failed = asyncio.run(_all())
    report = _discovery_report(reports)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    if failed == len(cities):
        logger.error("Discovery failed for all %d city(ies).", failed)
        sys.exit(1)


def _discovery_report(reports: list[DiscoveryReport], max_rejected: int = 150) -> str:
    """Summary for the PR body: every promotion with its sampled events,
    rejections folded away (capped, the PR body has a size limit)."""
    lines = []
    for r in reports:
        skipped = sum(r.skipped.values())
        lines.append(f"### {r.city}")
        lines.append(f"{r.searches} searches, {r.llm_calls} LLM calls — "
                     f"{len(r.promoted)} promoted, {len(r.rejected)} rejected, {skipped} skipped without a check.")
        lines.append("")
        for p in r.promoted:
            general = ", general listing" if p.is_general_listing else ""
            lines.append(f"- **{p.name}** ({p.url}) — `{p.category}`{general}")
            for e in p.events:
                lines.append(f"  - {e.title[:70]} · {e.start_date or 'no date'} · `{e.category}`")
        if r.rejected:
            lines.append("")
            lines.append(f"<details><summary>Rejected ({len(r.rejected)})</summary>")
            lines.append("")
            for url, reason in r.rejected[:max_rejected]:
                lines.append(f"- {url} — {reason}")
            if len(r.rejected) > max_rejected:
                lines.append(f"- … and {len(r.rejected) - max_rejected} more (see data/rejected_sources.json)")
            lines.append("")
            lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def _setup_sources(args: argparse.Namespace) -> None:
    names = set(args.source) if args.source else None
    if args.dry_run:
        todo = sources_to_set_up(args.city, names, args.redo)
        print(f"Would set up {len(todo)} fixed source(s):")
        for s in todo:
            print(f"  · {s.city} | {s.name} — {s.url}")
        return

    from agora.backend.infrastructure.llm.providers import get_llm_provider

    results = asyncio.run(set_up_sources(get_llm_provider(), args.city, names, args.redo))
    report = _setup_report(results)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")


def _setup_report(results: list[SourceSetup]) -> str:
    """Markdown summary of a setup-sources run — what each source will be read
    with, and for learned selectors a few sample plans to check them by."""
    if not results:
        return "No fixed sources needed setting up.\n"
    lines = [f"Set up {len(results)} fixed source(s):", ""]
    for r in sorted(results, key=lambda r: (r.source.city, r.source.name)):
        s = r.source
        lines.append(f"- **{s.city} — {s.name}** ({s.url}): `{r.method}`" + (f" — {r.note}" if r.note else ""))
        if r.method == "selectors" and s.selectors:
            lines.append(f"  - selectors: `{json.dumps(s.selectors.model_dump(exclude_none=True), ensure_ascii=False)}`")
            for p in r.sample:
                lines.append(f"  - {p.title[:70]} · {p.start_date or 'no date'} → {p.end_date or '–'} · {p.url or 'no link'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
