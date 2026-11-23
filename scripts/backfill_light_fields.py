"""Fill the categories and short titles a run's batched LLM calls left empty.

Ingestion asks the light model for these in batches (see enrichment.
classify_missing_categories / generate_short_titles). When those calls fail —
e.g. the 2026-11 runs, where gemini-flash-lite-latest 400'd on every batch —
plans are saved without them. This fills exactly those, with the same
batched calls: no scraping, no page fetches, no SerpApi searches.

Only browsable plans (not stale): nobody sees a stale one, so it isn't worth
an LLM call. Categories go first, since the short-title prompt includes the
category. A newly set category clears that plan's embedding (the embedded
text includes it — see set_plan_category); the next ingestion run, or
scripts/backfill_embeddings.py, re-embeds it.

Run from the project root (in the container: cd /app):

    python scripts/backfill_light_fields.py --dry-run   # count only, no LLM calls
    python scripts/backfill_light_fields.py
"""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field

from agora.backend.application.enrichment import BATCH_SIZE, classify_missing_categories, generate_short_titles
from agora.backend.application.ports import QuotaExhaustedError
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence.postgres_repository import (
    get_live_plans_missing_category_or_short_title,
    init_db,
    pool,
    set_plan_category,
    set_plan_short_title,
)


@dataclass
class _Plan:
    """Just the fields the two batched calls read and write."""

    id: int
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    category: str | None = None
    short_title: str = ""


def _to_plan(row: dict) -> _Plan:
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []
    return _Plan(
        id=row["id"],
        title=row["title"],
        description=row.get("description") or "",
        tags=tags,
        category=row.get("category"),
        short_title=row.get("short_title") or "",
    )


def _calls(n: int) -> int:
    return -(-n // BATCH_SIZE)


async def _fill(plans: list[_Plan]) -> QuotaExhaustedError | None:
    """Mutates *plans* in place. Stops at the first out-of-credit error and
    returns it, keeping whatever was filled before it."""
    llm = get_llm_provider(light=True)
    try:
        await classify_missing_categories(plans, llm)
        await generate_short_titles(plans, llm)
    except QuotaExhaustedError as e:
        return e
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only count what's missing; no LLM calls, no writes")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()

    try:
        plans = [_to_plan(r) for r in get_live_plans_missing_category_or_short_title()]
        no_cat = sum(1 for p in plans if not p.category)
        no_st = sum(1 for p in plans if not p.short_title)
        print(
            f"Live plans missing a category: {no_cat} (~{_calls(no_cat)} call(s)), "
            f"missing a short title: {no_st} (~{_calls(no_st)} call(s))"
        )
        if args.dry_run or not plans:
            return

        before = {p.id: (p.category, p.short_title) for p in plans}
        quota_error = asyncio.run(_fill(plans))

        cats = sts = 0
        for p in plans:
            old_cat, old_st = before[p.id]
            if p.category and not old_cat:
                set_plan_category(p.id, p.category)
                cats += 1
            if p.short_title and not old_st:
                set_plan_short_title(p.id, p.short_title)
                sts += 1
        print(f"Saved {cats}/{no_cat} categories and {sts}/{no_st} short titles")
        if cats:
            print("Their embeddings were cleared; the next ingestion run (or scripts/backfill_embeddings.py) redoes them")
        if quota_error:
            print(f"Stopped early: LLM provider is out of credit ({quota_error}). Top up and re-run for the rest.")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
