"""Re-runnable: bring every stored plan's category in line with its fixed
source's CURRENT settings in data/fixed_sources.json. Run it after changing a
source's `is_general_listing` or `promoted_by` — ingestion only applies those
to newly scraped plans, and the merge never overwrites a stored category.

- Single-category source (is_general_listing false): its plans are stamped
  with the source's category, as ingestion does. No LLM — free.
- General listing: its plans that still carry the source's promoted
  category (the stamp from before it was marked general) are re-classified
  one by one with the LLM, like backfill_categories.py.

A plan belongs to a source when its source_url is the source's url. Plans
extracted from an event's own detail page carry that page's url instead;
those are matched by host, but only on a host where every source in that
city is a general listing (so a stamp is never guessed).

Changed categories null the plan's semantic embedding (see
set_plan_category), so run backfill_embeddings.py afterwards.

Run from the project root (/app in the container):

    PYTHONPATH=. python scripts/resync_source_categories.py --dry-run   # counts only, no LLM
    PYTHONPATH=. python scripts/resync_source_categories.py
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

from agora.backend.application.enrichment import classify_category_from_text
from agora.backend.application.ports import QuotaExhaustedError, gather_until_quota
from agora.backend.domain.schemas import FixedSource
from agora.backend.infrastructure.llm.providers import get_llm_provider
from agora.backend.infrastructure.persistence import json_files
from agora.backend.infrastructure.persistence.postgres_repository import (
    _conn,
    init_db,
    pool,
    set_plan_category,
)


def _key(url: str | None) -> str:
    return (url or "").strip().rstrip("/").lower()


def _host(url: str | None) -> str:
    return (urlparse(url or "").hostname or "").removeprefix("www.")


def _owner_index(sources: list[FixedSource]):
    """(exact url, city) -> source, and (host, city) -> source for hosts whose
    sources in that city are all general listings with the same stamp."""
    by_url = {(_key(s.url), s.city): s for s in sources}
    by_host: dict[tuple[str, str], list[FixedSource]] = defaultdict(list)
    for s in sources:
        by_host[(_host(s.url), s.city)].append(s)
    host_fallback = {
        k: group[0] for k, group in by_host.items()
        if all(s.is_general_listing for s in group)
        and len({s.promoted_category() for s in group}) == 1
    }
    return by_url, host_fallback


def _plans() -> list[dict]:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, city, title, description, tags, category, source_url FROM plans"
        ).fetchall()]


async def resync(dry_run: bool) -> None:
    init_db()
    sources = json_files.load_fixed_sources()
    by_url, host_fallback = _owner_index(sources)

    stamp: list[tuple[dict, str, FixedSource]] = []
    reclassify: list[tuple[dict, FixedSource]] = []
    for plan in _plans():
        src = by_url.get((_key(plan["source_url"]), plan["city"])) \
            or host_fallback.get((_host(plan["source_url"]), plan["city"]))
        if not src:
            continue
        stamped = src.promoted_category()
        if not src.is_general_listing:
            if stamped and plan["category"] != stamped:
                stamp.append((plan, stamped, src))
        elif plan["category"] is None or plan["category"] == stamped:
            reclassify.append((plan, src))

    per_source: Counter = Counter()
    for plan, category, src in stamp:
        per_source[f"stamp     {src.city:<9} {src.name[:45]} → {category}"] += 1
        if not dry_run:
            set_plan_category(plan["id"], category)
    for _, src in reclassify:
        per_source[f"classify  {src.city:<9} {src.name[:45]}"] += 1
    for line, n in sorted(per_source.items()):
        print(f"  {n:>5}  {line}")
    verb = "Would stamp" if dry_run else "Stamped"
    print(f"\n{verb} {len(stamp)} plan(s); {len(reclassify)} plan(s) from general listings to classify")
    if dry_run or not reclassify:
        return

    llm = get_llm_provider()
    sem = asyncio.Semaphore(5)

    async def _one(plan: dict) -> tuple[dict, str | None]:
        tags = plan.get("tags") or []
        if isinstance(tags, str):
            tags = json.loads(tags) if tags else []
        async with sem:
            return plan, await classify_category_from_text(plan["title"], plan.get("description"), tags, llm)

    results = await gather_until_quota([_one(p) for p, _ in reclassify])
    moved: Counter = Counter()
    for res in results:
        if isinstance(res, Exception):
            continue
        plan, category = res
        if category and category != plan["category"]:
            set_plan_category(plan["id"], category)
            moved[f"{plan['category']} → {category}"] += 1
    print(f"\nReclassified {sum(moved.values())} plan(s) into a different category:")
    for change, n in moved.most_common():
        print(f"  {n:>5}  {change}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print what would change; no writes, no LLM calls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        asyncio.run(resync(args.dry_run))
    except QuotaExhaustedError as e:
        print(f"\nOut of Gemini credit — stopped ({e}). Top up and re-run.")
        sys.exit(1)
    finally:
        pool.close()
    if not args.dry_run:
        print("\nNext: PYTHONPATH=. python scripts/backfill_embeddings.py")


if __name__ == "__main__":
    main()
