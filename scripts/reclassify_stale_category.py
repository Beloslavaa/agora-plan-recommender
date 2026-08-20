"""One-off (and re-runnable) reclassification: re-run the LLM category
classifier against every plan CURRENTLY tagged a given category, instead of
only the ones missing one (see scripts/backfill_categories.py for that
case). Needed once for `cultural`'s split into `workshops`/`comedy_theatre`/
`food_drink`/`sports_wellness`/`day_trips`/`festivals_markets` (see
domain/schemas.py's PlanCategory) — those rows were classified back when
`cultural` was still the only catch-all bucket for all of them.

Run from the project root (same folder as main.py / data/):

    PYTHONPATH=. python scripts/reclassify_stale_category.py --from cultural
    PYTHONPATH=. python scripts/reclassify_stale_category.py --from cultural --dry-run
"""

import argparse
import asyncio
import json
import logging
from collections import Counter

from agora.backend.application.enrichment import classify_category_from_text
from agora.backend.application.ports import LLMProvider
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence.postgres_repository import (
    _conn,
    init_db,
    pool,
    set_plan_category,
)

logger = logging.getLogger(__name__)

_sem = asyncio.Semaphore(8)


def _get_plans_by_category(category: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, city, title, description, tags FROM plans WHERE category = %s",
            (category,),
        ).fetchall()
    return [dict(r) for r in rows]


async def _classify(plan: dict, llm: LLMProvider) -> tuple[dict, str | None]:
    tags = plan.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []
    async with _sem:
        new_category = await classify_category_from_text(plan["title"], plan.get("description"), tags, llm)
    return plan, new_category


async def reclassify(from_category: str, dry_run: bool) -> Counter:
    init_db()
    plans = _get_plans_by_category(from_category)
    if not plans:
        return Counter()

    llm = get_llm_provider()
    results = await asyncio.gather(*[_classify(p, llm) for p in plans])

    counts: Counter = Counter()
    for plan, new_category in results:
        if new_category is None:
            counts["<classification failed>"] += 1
            continue
        counts[new_category] += 1
        changed = " (unchanged)" if new_category == from_category else ""
        print(f"  {plan['id']:>5}  {new_category:<20}{changed}  {plan['title'][:55]}")
        if not dry_run and new_category != from_category:
            set_plan_category(plan["id"], new_category)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_category", required=True, help="Currently-stored category to re-classify")
    ap.add_argument("--dry-run", action="store_true", help="Classify and print, but don't write")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    counts = asyncio.run(reclassify(args.from_category, args.dry_run))
    verb = "Would reclassify" if args.dry_run else "Reclassified"
    total = sum(counts.values())
    print(f"\n{verb} {total} plan(s) previously tagged {args.from_category!r}:")
    for category, n in counts.most_common():
        print(f"  {category:<20} {n:>4}")
    if not args.dry_run:
        unchanged = counts.get(args.from_category, 0)
        moved = total - unchanged - counts.get("<classification failed>", 0)
        print(f"\n{moved} plan(s) moved to a new category (embeddings nulled — run backfill_embeddings.py next)")
    pool.close()


if __name__ == "__main__":
    main()
