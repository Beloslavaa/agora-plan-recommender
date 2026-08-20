"""Find and collapse duplicate plans: same city, compatible dates (equal, or
one side TBA), near-exact/exact title match (fuzzy_same_event_by_title).

Re-runnable cleanup counterpart to the dedup upsert_plans() applies live at
ingestion time — for rows already in the DB, or a threshold change.

Dry-run by default: prints candidates, writes nothing. Pass --apply to
actually merge (see postgres_repository.merge_duplicate_plan).

Run from the project root (same folder as main.py / data/):

    PYTHONPATH=. python scripts/dedupe_plans.py
    PYTHONPATH=. python scripts/dedupe_plans.py --threshold 0.9
    PYTHONPATH=. python scripts/dedupe_plans.py --apply
"""

import argparse

from agora.backend.domain.plan_matching import dates_compatible, fuzzy_same_event_by_title
from agora.backend.infrastructure.persistence.json_files import load_cities
from agora.backend.infrastructure.persistence.postgres_repository import (
    get_plans_for_dedup,
    init_db,
    merge_duplicate_plan,
    pool,
)

# Tie-breaker for _pick_keep when neither row has a decisive reason to win.
_COMPLETENESS_FIELDS = ("description", "ticket_url", "location", "image_url", "price", "category")


def _completeness(p: dict) -> int:
    return sum(1 for f in _COMPLETENESS_FIELDS if p.get(f))


def _pick_keep(a: dict, b: dict) -> tuple[dict, dict]:
    """Which row survives a merge. A real start_date beats a TBA one (that's
    usually *why* the pair diverged in the first place); otherwise prefer
    the more complete row; otherwise the older (lower id) one."""
    a_dated, b_dated = bool(a.get("start_date")), bool(b.get("start_date"))
    if a_dated != b_dated:
        return (a, b) if a_dated else (b, a)
    a_score, b_score = _completeness(a), _completeness(b)
    if a_score != b_score:
        return (a, b) if a_score > b_score else (b, a)
    return (a, b) if a["id"] < b["id"] else (b, a)


def find_duplicate_pairs(plans: list[dict], threshold: float) -> list[tuple[dict, dict]]:
    """O(n^2) within one city's plans — fine at this catalog's scale for an
    occasional script."""
    pairs = []
    for i, a in enumerate(plans):
        for b in plans[i + 1 :]:
            if not dates_compatible(a.get("start_date"), b.get("start_date")):
                continue
            if fuzzy_same_event_by_title(a["title"], b["title"], threshold):
                pairs.append((a, b))
    return pairs


def run(threshold: float, apply: bool) -> int:
    init_db()
    total = 0
    for city in load_cities():
        plans = get_plans_for_dedup(city)
        pairs = find_duplicate_pairs(plans, threshold)
        if not pairs:
            continue
        print(f"\n{city}: {len(pairs)} candidate pair(s)")

        # A plan matching MORE THAN ONE other plan is the recurring-event
        # trap: a TBA row can be dates_compatible with several real,
        # distinct showtimes at once. Which one it actually belongs to is
        # ambiguous, so skip the whole cluster instead of guessing.
        match_counts: dict[int, int] = {}
        for a, b in pairs:
            match_counts[a["id"]] = match_counts.get(a["id"], 0) + 1
            match_counts[b["id"]] = match_counts.get(b["id"], 0) + 1
        ambiguous = {pid for pid, n in match_counts.items() if n > 1}
        if ambiguous:
            for a, b in pairs:
                if a["id"] in ambiguous or b["id"] in ambiguous:
                    print(
                        f"  Ambiguous, skipping — needs manual review: "
                        f"#{a['id']} ({a['start_date'] or 'TBA'}) {a['title']!r} <> "
                        f"#{b['id']} ({b['start_date'] or 'TBA'}) {b['title']!r}"
                    )
            pairs = [(a, b) for a, b in pairs if a["id"] not in ambiguous and b["id"] not in ambiguous]
            if not pairs:
                continue

        # A row can also appear in more than one surviving pair (a 3-way
        # cluster the ambiguity check above missed) — skip it once merged
        # away rather than crash; a second run picks up what's left.
        merged_away: set[int] = set()
        for a, b in pairs:
            if a["id"] in merged_away or b["id"] in merged_away:
                print(f"  Skipping #{a['id']}/#{b['id']} — one side already merged away this run")
                continue
            keep, drop = _pick_keep(a, b)
            total += 1
            verb = "Merging" if apply else "Would merge"
            print(
                f"  {verb} #{drop['id']} -> #{keep['id']}\n"
                f"    keep: [{keep['id']:>5}] {keep['start_date'] or 'TBA':<10} {keep['title']!r}\n"
                f"    drop: [{drop['id']:>5}] {drop['start_date'] or 'TBA':<10} {drop['title']!r}"
            )
            if apply:
                merge_duplicate_plan(keep["id"], drop["id"])
                merged_away.add(drop["id"])
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--threshold", type=float, default=0.93,
        help="Minimum title-similarity ratio to treat two plans as the same event (default: 0.93)",
    )
    ap.add_argument("--apply", action="store_true", help="Actually merge/delete, instead of just printing")
    args = ap.parse_args()

    total = run(args.threshold, args.apply)
    verb = "Merged" if args.apply else "Would merge"
    print(f"\n{verb} {total} duplicate pair(s)" + ("" if args.apply else " — re-run with --apply to do it"))
    pool.close()


if __name__ == "__main__":
    main()
