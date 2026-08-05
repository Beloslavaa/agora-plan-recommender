"""Use-cases for running the ingestion pipeline: scrape fixed sources,
explore for new ones, or both, then validate the results. Orchestrates the
LLM/search/http ports plus domain validation — see infrastructure/cli/
ingest_cli.py for the CLI adapter that drives this."""

import asyncio
import logging

from agora.backend.application.enrichment import enrich_plans
from agora.backend.application.explorer import explore_for_plans
from agora.backend.application.extraction import extract_plans_from_html
from agora.backend.application.sources_admin import correct_city_from_location
from agora.backend.domain.schemas import PlanCategory, PlanData
from agora.backend.domain.validation import validate_and_filter
from agora.backend.infrastructure.config import settings
from agora.backend.infrastructure.http.fetcher import fetch_fixed_source_with_details
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence.json_files import load_fixed_sources
from agora.backend.infrastructure.search.providers import get_search_provider

logger = logging.getLogger(__name__)


_sem = asyncio.Semaphore(5)


def _category_from_promoted_by(promoted_by: str | None) -> str | None:
    """Fixed sources record which category they were promoted under, e.g.
    "explorer/PlanCategory.music_concerts" or "manual/cinema" (see
    application.sources_admin.promote_source and data/fixed_sources.json).
    Recover it so run_fixed_pipeline can pass it to the extractor the same
    way the exploratory pipeline already does — otherwise every plan scraped
    from a fixed source comes in with category=None, since nothing else ever
    infers a category from page content."""
    if not promoted_by:
        return None
    token = promoted_by.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    try:
        return PlanCategory(token).value
    except ValueError:
        return None


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
            # Skip for a general listing — promoted_by only records which
            # category search discovered it under, not what every event on
            # a multi-category page actually is (see FixedSource docstring).
            category = None if source.is_general_listing else _category_from_promoted_by(source.promoted_by)

            async def _extract(html: str, page_url: str) -> list[PlanData]:
                async with _sem:
                    try:
                        # Use page_url (the actual page the HTML came from), not
                        # source.url. For JSON-LD detail pages these differ, and
                        # tagging every plan with the listing URL was wrong — it
                        # broke the UI's "view source" link and the point of
                        # fetching detail pages at all.
                        extracted = await extract_plans_from_html(
                            html, page_url, "fixed", llm, category=category,
                        )
                        return extracted
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
    # that knows which city this whole run targeted. Must happen BEFORE
    # validate_and_filter: domain/validation.py rejects any plan with an
    # empty city as a pipeline-bug safety net, so validating per-page inside
    # _extract() above (i.e. before this stamp existed) silently dropped
    # 100% of every fixed source's plans, every run — the "city missing"
    # issue just never surfaced because validate_and_filter only logs
    # individual reasons at DEBUG, not the INFO level this CLI runs at.
    for p in plans:
        p.city = city
    # Some fixed sources (e.g. a cinema chain with branches in more than one
    # city) span cities under one URL scraped for only one of them — correct
    # the ones whose actual location says otherwise before validating.
    correct_city_from_location(plans, city)
    return validate_and_filter(plans)


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


async def run_one_city(
    mode: str,
    city: str,
    only_categories: list[str] | None,
    only_names: set[str] | None,
) -> list[PlanData]:
    """Run the selected mode for a single city, then enrich the results.

    Split out so a multi-city (cron) run can call this once per configured
    city, in its own try/except — one city's failure must never abort the
    rest of the run.
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
