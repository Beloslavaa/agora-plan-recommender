"""Repeatable sanity check for the semantic recommender: prove that saving a
plan actually shifts /recommendations toward similar plans, then clean up
after itself. Uses a throwaway user id so it never touches real data.

Run from the project root (same folder as main.py / data/), with the API
server NOT required — this calls the recommender module directly, not over
HTTP:

    PYTHONPATH=. python scripts/smoke_test_recommender.py --city Madrid
    PYTHONPATH=. python scripts/smoke_test_recommender.py --city Madrid --title "fashion week"
    PYTHONPATH=. python scripts/smoke_test_recommender.py --city Madrid --plan-id 34

--title does a case-insensitive substring match; --plan-id picks an exact
plan. Omit both and it just uses whatever plan happens to be first.
"""

import argparse
import logging

from agora.backend.application.recommendation import rank_for_user
from agora.backend.infrastructure.persistence.postgres_repository import (
    get_plans_for_scoring,
    init_db,
    pool,
    record_interaction,
    remove_interaction,
)

TEST_USER = "smoke_test_recommender"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check the semantic recommender")
    ap.add_argument("--city", default="Madrid")
    ap.add_argument("--title", help="Case-insensitive substring match against plan titles")
    ap.add_argument("--plan-id", type=int, help="Save this exact plan id instead of searching by title")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    init_db()

    candidates = [p for p in get_plans_for_scoring(args.city) if p.get("embedding")]
    if not candidates:
        print(f"No embedded plans for {args.city} — run scripts/backfill_embeddings.py first.")
        pool.close()
        return

    seed = None
    if args.plan_id is not None:
        seed = next((p for p in candidates if p["id"] == args.plan_id), None)
        if seed is None:
            print(f"No embedded plan with id={args.plan_id} in {args.city}.")
            pool.close()
            return
    elif args.title:
        needle = args.title.lower()
        matches = [p for p in candidates if needle in p["title"].lower()]
        if not matches:
            print(f"No embedded plan title contains {args.title!r} in {args.city}.")
            pool.close()
            return
        if len(matches) > 1:
            print(f"{len(matches)} matches for {args.title!r} — using the first, or narrow it down:")
            for p in matches:
                print(f"  id={p['id']:<5} {p['title']}")
        seed = matches[0]
    else:
        seed = candidates[0]

    before = rank_for_user(TEST_USER, args.city, limit=5)
    print(f"\nBefore any interaction (popularity fallback), top 5 for {args.city}:")
    for r in before:
        print(f"  {r['score']:>8.2f}  [{r.get('category')}]  {r['title'][:60]}")

    print(f"\nSaving: {seed['title']!r} [{seed.get('category')}]\n")
    record_interaction(TEST_USER, seed["id"], "saved")

    try:
        after = rank_for_user(TEST_USER, args.city, limit=5)
        print(f"After saving it, top 5 for {args.city}:")
        for r in after:
            print(f"  {r['score']:>8.4f}  [{r.get('category')}]  {r['title'][:60]}")
        print(
            "\nIf this worked, the categories/topics above should visibly lean "
            "toward the saved plan compared to the popularity-only list."
        )
    finally:
        remove_interaction(TEST_USER, seed["id"], "saved")
        print(f"\n(cleaned up: removed the test 'saved' interaction for {TEST_USER!r})")
        pool.close()


if __name__ == "__main__":
    main()
