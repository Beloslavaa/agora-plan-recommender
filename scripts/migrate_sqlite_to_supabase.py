"""One-off: copy every plan from the old local data/agora.db into Supabase Postgres.

Reads the legacy SQLite file directly (the one place outside store.py allowed to
touch it, since this script is the bridge off it) and upserts each row through
store.upsert_plans() — the same backfilling upsert the ingestion pipeline uses,
so re-running this is safe/idempotent.

Interactions are intentionally NOT migrated (assumed disposable demo/test data).
If that's wrong for your case, don't run this as-is — say so first.

Requires DATABASE_URL (Supabase) set in .env before running.

    python scripts/migrate_sqlite_to_supabase.py
    python scripts/migrate_sqlite_to_supabase.py --db-path data/agora.db
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

from agora.backend.ingestion.schemas import PlanData
from agora.backend.ingestion.store import get_plan_count, pool, upsert_plans


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
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate plans from local SQLite to Supabase Postgres")
    ap.add_argument("--db-path", default="data/agora.db", help="path to the legacy SQLite file")
    args = ap.parse_args()

    src_path = Path(args.db_path)
    if not src_path.exists():
        raise SystemExit(f"No SQLite file at {src_path}")

    conn = sqlite3.connect(str(src_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM plans").fetchall()]
    conn.close()

    before = get_plan_count()
    print(f"Read {len(rows)} plans from {src_path}. Supabase currently has {before}.")

    plans = [_to_plan(r) for r in rows]
    inserted = upsert_plans(plans)

    after = get_plan_count()
    not_new = len(rows) - inserted
    print(f"Inserted {inserted} new rows. Supabase now has {after} plans "
          f"(was {before}; {after - before} net new).")
    if not_new:
        print(f"({not_new} source row(s) did not produce a new row — expected for "
              f"duplicate title+source_url pairs in the source file; if that count "
              f"looks too high, a row may have failed silently inside upsert_plans.)")

    pool.close()


if __name__ == "__main__":
    main()
