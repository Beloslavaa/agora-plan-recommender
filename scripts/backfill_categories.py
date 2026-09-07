"""One-off (and re-runnable) backfill: classify every plan missing a
`category`.

Needed only for the pre-existing backlog — run_one_city (application/
ingestion.py) now classifies every fresh scrape's missing categories
automatically at ingestion time (enrichment.classify_missing_categories),
for both fixed-source general listings and exploratory-sourced plans. This
script exists for rows already in the DB from before that was wired in.

Run from the project root (same folder as main.py / data/):

    PYTHONPATH=. python scripts/backfill_categories.py
    PYTHONPATH=. python scripts/backfill_categories.py --dry-run
"""

import argparse
import asyncio
import json
import logging

from agora.backend.application.enrichment import classify_category_from_text
from agora.backend.application.ports import LLMProvider
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence.postgres_repository import (
    get_plans_missing_category,
    init_db,
    pool,
    set_plan_category,
)

logger = logging.getLogger(__name__)

_sem = asyncio.Semaphore(5)


async def _classify(plan: dict, llm: LLMProvider) -> str | None:
    tags = plan.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []
    async with _sem:
        return await classify_category_from_text(plan["title"], plan.get("description"), tags, llm)


async def backfill(dry_run: bool) -> tuple[int, int]:
    init_db()
    missing = get_plans_missing_category()
    if not missing:
        return 0, 0

    llm = get_llm_provider()
    results = await asyncio.gather(*[_classify(p, llm) for p in missing])

    classified = 0
    for plan, category in zip(missing, results):
        if category is None:
            continue
        classified += 1
        print(f"  {plan['id']:>5}  {category:<18} {plan['title'][:60]}")
        if not dry_run:
            set_plan_category(plan["id"], category)

    return len(missing), classified


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Classify and print, but don't write")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    total, classified = asyncio.run(backfill(args.dry_run))
    verb = "Would classify" if args.dry_run else "Classified"
    print(f"\n{verb} {classified}/{total} plan(s) missing a category")
    pool.close()


if __name__ == "__main__":
    main()
