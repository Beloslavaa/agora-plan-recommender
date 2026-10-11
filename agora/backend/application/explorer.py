import asyncio
import logging
from datetime import date
from urllib.parse import urlparse

from agora.backend.application.ports import LLMProvider, SearchProvider
from agora.backend.application.sources_admin import correct_city_from_location, promote_source
from agora.backend.domain.countries import language_name, search_country
from agora.backend.domain.schemas import (
    CATEGORY_EXAMPLES,
    CATEGORY_LABELS,
    City,
    PlanCategory,
    PlanData,
)
from agora.backend.domain.validation import validate_and_filter
from agora.backend.infrastructure.http.fetcher import fetch_page_with_details
from agora.backend.infrastructure.persistence.json_files import (
    load_city_countries,
    load_city_settings,
    load_fixed_sources,
)
from agora.backend.infrastructure.search.providers import get_search_provider

logger = logging.getLogger(__name__)

QUERY_SYSTEM = """\
Today's date is {today}.

You are a creative curator. Generate 2 search queries to discover current \
{category} in {place}.

{language_instructions}

The queries MUST find events happening ON or AFTER {today}.
Use only relative time phrases, in each query's own language (e.g. "este fin \
de semana", "this weekend", "próximamente", "upcoming") — never use a \
specific year or date in the query.

Here are examples of good queries for this category. They show the STYLE of a \
good query — short, specific, time-relative. They say nothing about language: \
do not translate them, and write each query in the language requested above.
{examples}

Return a JSON array of exactly 2 objects:
[{{"lang": "{local_code}", "q": "..."}}, {{"lang": "en", "q": "..."}}]
"""

# One query in the local language reaches venues that only publish in it;
# the English one keeps expat/visitor listings.
_BILINGUAL_INSTRUCTIONS = """\
- Query 1: in {language}, written the way someone who lives in {city} would \
type it into Google. Use the words locals actually use, not a translation of \
an English phrase.
- Query 2: in English, written the way a visitor or an English-speaking \
resident would type it."""

_ENGLISH_INSTRUCTIONS = """\
- Write both queries in English, taking two different angles (e.g. one \
specific, one broader), so they don't return the same results."""

PROMOTION_THRESHOLD = 2


async def generate_queries(
    llm: LLMProvider,
    category: PlanCategory,
    city: City,
) -> list[tuple[str | None, str]]:
    """(language, query) pairs. Language is None when the model didn't tag
    a query with one of the two languages asked for."""
    examples = "\n".join(f"- {e}" for e in CATEGORY_EXAMPLES[category])
    local = city.language
    if local == "en":
        language_instructions = _ENGLISH_INSTRUCTIONS
    else:
        language_instructions = _BILINGUAL_INSTRUCTIONS.format(
            language=language_name(local), city=city.name,
        )
    system = QUERY_SYSTEM.format(
        today=date.today().isoformat(),  # per-call, not frozen at import time
        category=CATEGORY_LABELS[category],
        # Country alongside the name: one more nudge away from same-named
        # cities elsewhere (Madrid, NM), on top of the search's gl.
        place=f"{city.name}, {city.country}" if city.country else city.name,
        language_instructions=language_instructions,
        local_code=local,
        examples=examples,
    )
    data = await llm.parse_json(
        prompt="Generate the queries now.",
        system=system,
        temperature=0.9,
        max_tokens=1024,
    )
    queries: list[tuple[str | None, str]] = []
    for item in data if isinstance(data, list) else []:
        # Plain strings (the old output shape) are still accepted, just
        # without a language — a small model slip shouldn't stop discovery.
        if isinstance(item, str) and item.strip():
            queries.append((None, item.strip()))
        elif isinstance(item, dict) and str(item.get("q") or "").strip():
            lang = item.get("lang")
            queries.append((lang if lang in (local, "en") else None, str(item["q"]).strip()))
    return queries


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

    from agora.backend.application.extraction import extract_plans_from_html

    try:
        pages = await fetch_page_with_details(base_url)
        extracted: list[PlanData] = []
        for html, page_url in pages:
            # category=None, not category.value — this is the site's ROOT,
            # a near-certain multi-category page (same reasoning as
            # FixedSource.is_general_listing). Stamping the category this
            # search happened to be looking for onto whatever it finds here
            # blanket-mistags anything off-category (see AGENTS.md/git
            # history: bcb487b fixed this for run_fixed_pipeline's general
            # listings, but never for the exploratory pipeline). Left None,
            # backfill_categories.py classifies it from actual content later.
            extracted.extend(await extract_plans_from_html(
                html, page_url, "exploratory", llm, category=None,
            ))
    except Exception as e:
        logger.debug("  Base-domain check failed for %s: %s", base_url, e)
        return []

    # PlanData.city defaults to "" and validate_plan rejects any plan with an
    # empty one — stamp it before validating, or every plan here fails that
    # check and this never promotes anything (city is otherwise only stamped
    # in bulk at the end of explore_for_plans, too late for this gate).
    for p in extracted:
        p.city = city
    # With city_countries, so a page about another country's same-named city
    # can't pass this gate and become a permanent fixed source.
    valid = validate_and_filter(extracted, load_city_countries())
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

    # A city missing from data/cities.json still gets explored, just in
    # English with no country hint — as before languages were configured.
    city_settings = load_city_settings().get(city) or City(name=city, country="")
    gl = search_country(city_settings.country)

    categories = only_categories or set(PlanCategory)
    for category in PlanCategory:
        if category not in categories:
            continue
        logger.info("Exploring category: %s", CATEGORY_LABELS[category])
        cat_count = 0
        try:
            queries = await generate_queries(llm, category, city_settings)
        except Exception as e:
            logger.warning("  Failed to generate queries for %s: %s", category, e)
            continue

        for lang, query in queries:
            if cat_count >= max_per_category:
                break
            logger.info("  Query [%s]: %s", lang or "?", query)
            await asyncio.sleep(1)
            try:
                results = await search_provider.search(query, max_results=5, gl=gl, hl=lang)
            except Exception as e:
                logger.warning("  Search failed: %s", e or type(e).__name__)
                if cat_count >= min_per_category:
                    break
                continue

            from agora.backend.application.extraction import extract_plans_from_html

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
                        # category=None, not category.value — a query built
                        # to find `category` content doesn't guarantee every
                        # page it surfaces actually IS that category (e.g. a
                        # "cinema" search once surfaced a live orchestral
                        # tribute concert page, mistagged "cinema" as a
                        # result). Same fix as bcb487b applied to
                        # run_fixed_pipeline's general listings; backfill_
                        # categories.py classifies these from real content.
                        extracted.extend(await extract_plans_from_html(
                            html, page_url, "exploratory", llm,
                            category=None,
                        ))
                    # Stamp city before validating — see the same fix in
                    # _try_promote_base_domain; validate_plan rejects any
                    # plan with an empty city, and it isn't stamped in bulk
                    # until the end of explore_for_plans.
                    for p in extracted:
                        p.city = city
                    valid = validate_and_filter(extracted, load_city_countries())
                    if cat_count < max_per_category and extracted:
                        if len(valid) >= PROMOTION_THRESHOLD:
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
    # A search hit can land on a page describing a DIFFERENT city than the
    # one this run searched for — correct city from location when it does.
    correct_city_from_location(plans, city)
    return plans