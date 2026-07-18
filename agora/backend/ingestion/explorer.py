import asyncio
import logging
from datetime import date
from urllib.parse import urlparse

from agora.backend.ingestion.llm import LLMProvider
from agora.backend.ingestion.schemas import (
    CATEGORY_EXAMPLES,
    CATEGORY_LABELS,
    PlanCategory,
    PlanData,
)
from agora.backend.ingestion.search import SearchProvider, get_search_provider
from agora.backend.ingestion.sources import (
    fetch_page_with_details,
    load_fixed_sources,
    promote_source,
)
from agora.backend.ingestion.validator import validate_and_filter

logger = logging.getLogger(__name__)

QUERY_SYSTEM = """\
Today's date is {today}.

You are a creative curator. Generate 2 search queries to discover current \
{category} in {city}.

The queries MUST find events happening ON or AFTER {today}.
Use only relative time phrases like "this month", "this weekend", "upcoming" — \
never use a specific year or date in the query.

Here are examples of good queries for this category:
{examples}

Return a JSON array of query strings.
"""

PROMOTION_THRESHOLD = 2


async def generate_queries(
    llm: LLMProvider,
    category: PlanCategory,
    city: str,
) -> list[str]:
    examples = "\n".join(f"- {e}" for e in CATEGORY_EXAMPLES[category])
    system = QUERY_SYSTEM.format(
        today=date.today().isoformat(),  # per-call, not frozen at import time
        category=CATEGORY_LABELS[category],
        city=city,
        examples=examples,
    )
    return await llm.parse_json(
        prompt="Generate the queries now.",
        system=system,
        temperature=0.9,
        max_tokens=1024,
    )


async def _try_promote_base_domain(
    result_url: str,
    category: PlanCategory,
    llm: LLMProvider,
    checked_domains: set[str],
    existing_domains: set[str | None],
    city: str,
) -> list[PlanData]:
    """A search hit is often a single event's own page — that alone rarely
    clears PROMOTION_THRESHOLD, even when the site behind it runs an ongoing
    events calendar well worth scraping on every future run. Check the site's
    base URL (its root) for more culture-related events beyond this one hit;
    if there are enough, scrape + validate them all and promote the base URL
    as a fixed source.

    Skipped (returns []) when result_url IS already the site's root — that
    case is already covered by the per-URL promotion the caller does — or
    when the domain was already checked this run or is already fixed.
    """
    parsed = urlparse(result_url)
    domain = parsed.hostname
    if not domain or parsed.path in ("", "/"):
        return []
    if domain in checked_domains or domain in existing_domains:
        return []
    checked_domains.add(domain)
    base_url = f"{parsed.scheme}://{domain}/"

    from agora.backend.ingestion.extractor import extract_plans_from_html

    try:
        pages = await fetch_page_with_details(base_url)
        extracted: list[PlanData] = []
        for html, page_url in pages:
            extracted.extend(await extract_plans_from_html(
                html, page_url, "exploratory", llm, category=category.value,
            ))
    except Exception as e:
        logger.debug("  Base-domain check failed for %s: %s", base_url, e)
        return []

    valid = validate_and_filter(extracted)
    if len(valid) < PROMOTION_THRESHOLD:
        return []

    name = domain.removeprefix("www.").split(".")[0].title()
    if promote_source(name=name, url=base_url, city=city, promoted_by=f"explorer/{category}"):
        logger.info("  ★ Promoted base domain to fixed sources: %s (%d events found)",
                    base_url, len(valid))
    return valid


async def explore_for_plans(
    llm: LLMProvider,
    city: str,
    search_provider: SearchProvider | None = None,
    min_per_category: int = 3,
    max_per_category: int = 10,
    only_categories: set[PlanCategory] | None = None,
) -> list[PlanData]:
    if search_provider is None:
        search_provider = get_search_provider()

    plans: list[PlanData] = []
    # Base domains checked (or already fixed) this run — avoids re-fetching
    # the same site's root every time a different search hit lands on it.
    checked_domains: set[str] = set()
    existing_domains = {urlparse(s.url).hostname for s in load_fixed_sources()}

    categories = only_categories or set(PlanCategory)
    for category in PlanCategory:
        if category not in categories:
            continue
        logger.info("Exploring category: %s", CATEGORY_LABELS[category])
        cat_count = 0
        try:
            queries = await generate_queries(llm, category, city)
        except Exception as e:
            logger.warning("  Failed to generate queries for %s: %s", category, e)
            continue

        for query in queries:
            if cat_count >= max_per_category:
                break
            logger.info("  Query: %s", query)
            await asyncio.sleep(1)
            try:
                results = await search_provider.search(query, max_results=5)
            except Exception as e:
                logger.warning("  Search failed: %s", e or type(e).__name__)
                if cat_count >= min_per_category:
                    break
                continue

            from agora.backend.ingestion.extractor import extract_plans_from_html

            for result in results:
                if cat_count >= max_per_category:
                    break
                try:
                    # If result.url is a listing page (e.g. an Eventbrite category
                    # page) with a JSON-LD ItemList, this fetches each event's own
                    # detail page instead of just the listing — that's what gets us
                    # the specific event URL/image rather than falling back to it.
                    pages = await fetch_page_with_details(result.url)
                    extracted: list[PlanData] = []
                    for html, page_url in pages:
                        extracted.extend(await extract_plans_from_html(
                            html, page_url, "exploratory", llm,
                            category=category.value,
                        ))
                    if cat_count < max_per_category and extracted:
                        if len(extracted) >= PROMOTION_THRESHOLD:
                            name = result.title.split("—")[0].split("|")[0].strip()[:80]
                            if promote_source(
                                name=name,
                                url=result.url,
                                city=city,
                                promoted_by=f"explorer/{category}",
                            ):
                                logger.info(
                                    "  ★ Promoted to fixed sources: %s", name
                                )
                        plans.extend(extracted)
                        cat_count += len(extracted)
                        logger.info("  → %d plans from %s (cat total: %d)",
                                    len(extracted), result.url, cat_count)

                        # This hit might be one event on a site that runs an
                        # ongoing events calendar — worth a fixed source of
                        # its own even if this one page didn't have enough.
                        if cat_count < max_per_category:
                            base_plans = await _try_promote_base_domain(
                                result.url, category, llm,
                                checked_domains, existing_domains, city,
                            )
                            if base_plans:
                                plans.extend(base_plans)
                                cat_count += len(base_plans)
                                logger.info("  → %d plans from base domain of %s (cat total: %d)",
                                            len(base_plans), result.url, cat_count)
                except Exception as e:
                    logger.debug("  Failed to process %s: %s", result.url, e)
                    if cat_count >= min_per_category:
                        break
                    continue

        if cat_count < min_per_category:
            logger.warning("  Only got %d plans for %s (wanted %d)",
                           cat_count, category.value, min_per_category)

    # Stamped here (not asked of the LLM extractor) — this is the one place
    # that knows which city this whole run targeted.
    for p in plans:
        p.city = city
    return plans