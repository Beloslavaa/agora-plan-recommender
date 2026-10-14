"""Use-cases for running the ingestion pipeline: scrape fixed sources, then
validate the results. (Finding new sources is source_discovery.py's job.) Orchestrates the
LLM/search/http ports plus domain validation — see infrastructure/cli/
ingest_cli.py for the CLI adapter that drives this."""

import asyncio
import logging
from datetime import date

from agora.backend.application.enrichment import classify_missing_categories, enrich_plans
from agora.backend.application.ports import LLMProvider, PageFetcher, SearchProvider, SourceStore
from agora.backend.application.extraction import extract_plans_from_html, extract_plans_from_wp_api
from agora.backend.domain.card_selectors import check_selectors, plans_from_cards
from agora.backend.domain.schemas import PlanData
from agora.backend.domain.validation import correct_city_from_location, validate_and_filter

logger = logging.getLogger(__name__)


_sem = asyncio.Semaphore(5)


async def run_fixed_pipeline(
    llm: LLMProvider, fetcher: PageFetcher, store: SourceStore, city: str, only_names: set[str] | None = None,
) -> list[PlanData]:
    sources = [s for s in store.load_fixed_sources() if s.city == city]
    if not sources:
        logger.info("No fixed sources yet for %s — run explorer mode first", city)
        return []
    plans: list[PlanData] = []
    for source in sources:
        if only_names and source.name not in only_names:
            continue
        try:
            logger.info("Scraping fixed source: %s", source.name)
            # Skip for a general listing — promoted_by only records which
            # category search discovered it under, not what every event on
            # a multi-category page actually is (see FixedSource docstring).
            category = None if source.is_general_listing else source.promoted_category()

            # A WordPress site with an events plugin publishes the same events
            # as JSON — when it does, that replaces scraping its pages.
            # source.extractor (see FixedSource) says whether this site has
            # one, so the other sources skip the probe; unchecked ones try it.
            if source.extractor != "page":
                wp_plans = await extract_plans_from_wp_api(fetcher, source.url, "fixed", category)
                if wp_plans:
                    plans.extend(wp_plans)
                    continue
                if source.extractor == "wp_api":
                    logger.warning("  ! %s: WordPress events API returned nothing, "
                                   "scraping its page instead", source.name)

            # Listing pages with learned CSS selectors (application/source_setup.py)
            # are read without the LLM — unless the selectors no longer fit the
            # page (usually a redesign), which falls through to the LLM below.
            if source.selectors:
                html = await fetcher.fetch_page(source.url)
                ok, summary = check_selectors(html, source.selectors, source.url, date.today())
                if ok:
                    card_plans = plans_from_cards(html, source.selectors, source.url, "fixed",
                                                  date.today(), category)
                    logger.info("  ✓ %d plan(s) from CSS selectors on %s", len(card_plans), source.url)
                    plans.extend(card_plans)
                    continue
                logger.warning("  ! %s: selectors no longer fit the page (%s), "
                               "using the LLM — re-run main.py --mode setup-sources --source %r",
                               source.name, summary, source.name)

            page_htmls = await fetcher.fetch_page_with_details(source.url)

            async def _extract(html: str, page_url: str) -> list[PlanData]:
                async with _sem:
                    try:
                        # Use page_url (the actual page the HTML came from), not
                        # source.url. For JSON-LD detail pages these differ, and
                        # tagging every plan with the listing URL was wrong — it
                        # broke the UI's "view source" link and the point of
                        # fetching detail pages at all.
                        extracted = await extract_plans_from_html(
                            html, page_url, "fixed", llm, fetcher, category=category,
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
    correct_city_from_location(plans, city, store.load_cities())
    return validate_and_filter(plans, store.load_city_countries())


async def run_one_city(
    llm: LLMProvider,
    search: SearchProvider,
    fetcher: PageFetcher,
    store: SourceStore,
    city: str,
    only_names: set[str] | None,
) -> list[PlanData]:
    """Scrape a single city's fixed sources, then classify/enrich the results.

    Split out so a multi-city (cron) run can call this once per configured
    city, in its own try/except — one city's failure must never abort the
    rest of the run. The providers come from the caller, created once for
    the whole run on its one event loop.
    """
    plans = await run_fixed_pipeline(llm, fetcher, store, city, only_names=only_names)

    # Not every plan gets a category while scraping (see
    # enrichment.classify_missing_categories) — general listings' plans
    # reach here with category=None. Classify from content before this batch is ever
    # written to the DB, instead of leaving it null for a separate manual
    # backfill run (scripts/backfill_categories.py remains for the
    # pre-existing backlog only).
    await classify_missing_categories(plans, llm)

    return await enrich_plans(plans, fetcher, store.load_city_settings(), llm=llm, search=search)
