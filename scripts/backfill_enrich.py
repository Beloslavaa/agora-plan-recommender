"""One-off backfill: re-enrich the plans already in the DB and write results back.

This is the cheap way to fill missing purchase links / images on your EXISTING
rows without re-scraping listing pages or re-running the extractor LLM.

For every plan that has a `url` but is missing an image or ticket link, it fetches
that page once and pulls the og:image and the purchase link. Optionally it can also
search for missing URLs (SerpAPI) and generate missing short titles (LLM).

Run from the project root (same folder as main.py / data/).

    python scripts/backfill_enrich.py --no-llm --no-search   # cheapest: only scrape known event pages
                                                              # (no OpenAI/SerpAPI keys needed, just network)
    python scripts/backfill_enrich.py --no-llm                # also web-search for the 65 rows missing a url
    python scripts/backfill_enrich.py                         # everything (also LLM short titles)
"""

import argparse
import asyncio
import json
import logging
from datetime import date

from agora.backend.ingestion.enrich import enrich_plans
from agora.backend.ingestion.llm import get_llm_provider
from agora.backend.ingestion.schemas import PlanData
from agora.backend.ingestion.search import get_search_provider
from agora.backend.ingestion.store import get_all_plans, pool, upsert_plans


def _to_plan(row: dict) -> PlanData:
    def _d(s):
        return date.fromisoformat(s) if s else None

    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    return PlanData(
        title=row["title"],
        short_title=row.get("short_title") or "",
        description=row.get("description") or "",
        start_date=_d(row.get("start_date")),
        end_date=_d(row.get("end_date")),
        url=row.get("url"),
        ticket_url=row.get("ticket_url"),
        location=row.get("location"),
        image_url=row.get("image_url"),
        price=row.get("price"),
        tags=tags or [],
        category=row.get("category"),
        source_url=row["source_url"],
        source_type=row.get("source_type") or "fixed",
        city=row.get("city") or "",
    )


def _counts(rows: list[dict]) -> dict:
    keys = ("url", "ticket_url", "image_url", "short_title")
    return {k: sum(1 for r in rows if r.get(k)) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill enrichment for existing DB plans")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM (short-title generation)")
    ap.add_argument("--no-search", action="store_true", help="skip web search for missing URLs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    before = get_all_plans()
    print(f"Loaded {len(before)} plans.  Before: {_counts(before)}")

    llm = None if args.no_llm else get_llm_provider()
    search = None if args.no_search else get_search_provider()

    plans = [_to_plan(r) for r in before]
    asyncio.run(enrich_plans(plans, llm=llm, search=search))
    upsert_plans(plans)  # backfills empty columns; returns new-row count (≈0 here)

    after = get_all_plans()
    print(f"Done.           After:  {_counts(after)}")
    pool.close()


if __name__ == "__main__":
    main()