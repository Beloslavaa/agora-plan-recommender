"""Targeted backfill: patch existing DB rows using JSON-LD from their known source page.

Unlike backfill.py (which uses LLM/search to fill gaps), this makes zero LLM or
search calls. For every distinct source_url among plans missing a `url`, it
refetches that one page, reads its JSON-LD (Event / ItemList), matches events to
existing rows by title, and UPDATEs those exact rows in place by id — so it can
never create a duplicate row (unlike re-running the discovery pipeline, which
would insert new rows keyed on the event's own url as source_url).

Run from the project root (same folder as main.py / data/):

    python scripts/backfill_ld.py            # apply
    python scripts/backfill_ld.py --dry-run   # show what would change, no writes
"""

import argparse
import asyncio
import logging

from agora.backend.ingestion.sources import extract_ld_events, fetch_page_with_details
from agora.backend.ingestion.store import PATCHABLE_PLAN_COLUMNS, get_plans_missing_url, pool, update_plan_fields

logger = logging.getLogger(__name__)


def _norm(title: str) -> str:
    return " ".join(title.strip().lower().split())


async def main() -> None:
    ap = argparse.ArgumentParser(description="Patch existing plans from their source page's JSON-LD")
    ap.add_argument("--dry-run", action="store_true", help="print planned updates, write nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = get_plans_missing_url()
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_url"], []).append(r)

    print(f"{len(rows)} plans missing a url, across {len(by_source)} distinct source pages")

    total_patched = 0
    for source_url, plan_rows in by_source.items():
        try:
            # Follows a listing page's JSON-LD ItemList down to each detail
            # page (same expansion the ingestion pipeline does), so sites like
            # cinesrenoir — whose homepage ItemList only has bare urls, with
            # the real Event data one level down on each showtime page — get
            # picked up too. For a page with no ItemList this is just that
            # one page, same as a plain fetch.
            pages = await fetch_page_with_details(source_url)
        except Exception as e:
            print(f"  ✗ fetch failed for {source_url}: {e}")
            continue

        events = [
            ev
            for html, page_url in pages
            for ev in extract_ld_events(html, base_url=page_url)
        ]
        if not events:
            print(f"  – no JSON-LD on {source_url} ({len(plan_rows)} plans left as-is)")
            continue

        by_title = {_norm(ev["title"]): ev for ev in events if ev.get("title")}
        patched_here = 0
        for row in plan_rows:
            ev = by_title.get(_norm(row["title"]))
            if not ev:
                continue
            # `not row[field]` is deliberately falsy-inclusive (empty string/None
            # both count as "missing"), but on the ev side we must check `is not
            # None` rather than truthiness — price can legitimately be 0.0 for a
            # free event, which is falsy but a real value worth backfilling.
            updates = {
                field: ev[field]
                for field in PATCHABLE_PLAN_COLUMNS
                if ev.get(field) is not None and ev.get(field) != "" and not row[field]
            }
            if not updates:
                continue
            patched_here += 1
            total_patched += 1
            print(f"  ✓ {row['title'][:45]:45s} <- {list(updates.keys())}")
            if not args.dry_run:
                update_plan_fields(row["id"], updates)
        print(f"  {source_url}: {patched_here}/{len(plan_rows)} matched via JSON-LD")

    print(f"\n{'Would patch' if args.dry_run else 'Patched'} {total_patched}/{len(rows)} plans")
    pool.close()


if __name__ == "__main__":
    asyncio.run(main())
