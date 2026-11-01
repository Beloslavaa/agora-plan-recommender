"""Re-runnable: hide stored plans that are in another town or country than
their city — the same two checks ingestion's validation now applies to fresh
scrapes (domain/validation.py): a location ending in another country
("…, Cyprus"), or a venue postal code outside the city's postal_prefixes in
data/cities.json ("9000 Варна" under Sofia).

Hides by setting is_stale (like mark_stale_plans), not deleting, so any
interaction history stays. Stored plans have only their location text to go
on, so plans without a postal code or country in it are left alone.

Run from the project root (/app in the container):

    PYTHONPATH=. python scripts/hide_out_of_city_plans.py --dry-run   # list only
    PYTHONPATH=. python scripts/hide_out_of_city_plans.py
"""

import argparse
from collections import defaultdict

from agora.backend.domain.schemas import PlanData
from agora.backend.domain.validation import _wrong_country, outside_city
from agora.backend.infrastructure.persistence import json_files
from agora.backend.infrastructure.persistence.postgres_repository import _conn, init_db, pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="List what would be hidden; write nothing")
    args = ap.parse_args()

    try:
        init_db()
        cities = json_files.load_city_settings()
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, city, title, location FROM plans WHERE NOT is_stale AND location IS NOT NULL"
            ).fetchall()

        hide: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
        for r in rows:
            city = cities.get(r["city"])
            if not city:
                continue
            plan = PlanData.model_construct(title=r["title"], location=r["location"], city=r["city"],
                                            country=None, postal_code=None)
            why = _wrong_country(plan, city.country) or outside_city(
                r["location"], None, r["city"], city.country, city.postal_prefixes,
            )
            if why:
                hide[r["city"]].append((r["id"], why, r["title"], r["location"]))

        total = sum(len(v) for v in hide.values())
        for city, items in sorted(hide.items()):
            print(f"\n{city}: {len(items)}")
            for pid, why, title, loc in items:
                print(f"  {pid:>6}  {why:<40} {title[:45]:<45} | {loc[:60]}")

        if args.dry_run or not total:
            print(f"\n{'Would hide' if args.dry_run else 'Hid'} {total} plan(s) out of {len(rows)} checked")
            return
        ids = [pid for items in hide.values() for pid, *_ in items]
        with _conn() as conn:
            conn.execute("UPDATE plans SET is_stale = TRUE WHERE id = ANY(%s)", (ids,))
        print(f"\nHid {total} plan(s) out of {len(rows)} checked")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
