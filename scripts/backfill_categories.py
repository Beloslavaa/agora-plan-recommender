"""One-off (and re-runnable) backfill: classify every plan missing a
`category`.

Needed for plans scraped by the fixed-source pipeline before it threaded
each source's known category through to extraction (see cli.py's
_category_from_promoted_by) — those rows were never assigned one, since
nothing else ever infers category from page content. New scrapes no longer
need this; this is purely for the backlog.

Run from the project root (same folder as main.py / data/):

    PYTHONPATH=. python scripts/backfill_categories.py
    PYTHONPATH=. python scripts/backfill_categories.py --dry-run
"""

import argparse
import asyncio
import json
import logging

from agora.backend.ingestion.llm import LLMProvider, get_llm_provider
from agora.backend.ingestion.schemas import CATEGORY_LABELS, PlanCategory
from agora.backend.ingestion.store import get_plans_missing_category, init_db, pool, set_plan_category

logger = logging.getLogger(__name__)

_sem = asyncio.Semaphore(5)

_CATEGORY_OPTIONS = "\n".join(f"- {c.value}: {label}" for c, label in CATEGORY_LABELS.items())

_CLASSIFY_SYSTEM = f"""\
You classify a cultural/entertainment event into exactly one category, based \
only on the title/description/tags given.

Categories:
{_CATEGORY_OPTIONS}

Respond with JSON only: {{"category": "<one of the category keys above>"}}
If nothing fits well, pick the closest one — never invent a new category name.
"""


def _clean(s: str | None) -> str:
    return str(s or "").replace("<DATA>", " ").replace("</DATA>", " ")


async def _classify(plan: dict, llm: LLMProvider) -> str | None:
    tags = plan.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []

    prompt = (
        "<DATA>\n"
        f"title: {_clean(plan['title'])}\n"
        f"description: {_clean(plan.get('description'))[:500]}\n"
        f"tags: {_clean(', '.join(tags))}\n"
        "</DATA>"
    )
    async with _sem:
        try:
            data = await llm.parse_json(prompt, system=_CLASSIFY_SYSTEM, temperature=0.1, max_tokens=128)
            category = (data.get("category") or "").strip()
            return PlanCategory(category).value
        except Exception as e:
            logger.warning("  ~ classification failed for %r: %s", plan["title"][:50], e)
            return None


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
