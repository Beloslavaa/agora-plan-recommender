"""Generate synthetic users + interactions against real plans, so the
LightGCN pipeline (offline notebooks — see AGENTS.md) has a graph with some
structure to train on before real multi-user traffic exists.

This is a stopgap, not a stand-in for real signal: archetypes below are
hand-picked category-preference vectors, so the resulting graph mostly
encodes those assumptions rather than genuine co-consumption. Treat it as a
way to exercise the pipeline (training loop, cold-start blending, embedding
dimensions) end to end — not as evidence the graph model beats Tier 1 on
anything real. Swap in real interaction data the moment there's enough of it.

Every synthetic user id is prefixed `synthetic_`, which is what makes
--clear (and keeping this out of anything user-facing) possible.

Run from the project root (same folder as main.py / data/):

    PYTHONPATH=. python scripts/generate_synthetic_interactions.py --city Madrid
    PYTHONPATH=. python scripts/generate_synthetic_interactions.py --city Madrid --users 300 --seed 42
    PYTHONPATH=. python scripts/generate_synthetic_interactions.py --city Madrid --dry-run
    PYTHONPATH=. python scripts/generate_synthetic_interactions.py --city Madrid --clear
"""

import argparse
import json
import logging
import random
from collections import Counter

import numpy as np  # already a project dependency (requirements.txt, via torch)

from agora.backend.infrastructure.persistence.postgres_repository import (
    get_all_city_plans,
    init_db,
    pool,
    record_interaction,
)

logger = logging.getLogger(__name__)

USER_PREFIX = "synthetic_"

# Category-preference vectors. Unlisted categories still get NOISE_FLOOR below,
# so no archetype is hard-locked to its categories — real people aren't either,
# and a few weak cross-cluster ties are what keeps the graph from splitting
# into disconnected islands (one per archetype), which LightGCN can't do
# anything with. `share` is how much of the synthetic population gets this
# archetype; they're relative weights, not required to sum to 1.
ARCHETYPES: dict[str, dict] = {
    "culture_vulture": {
        "share": 3,
        "weights": {"art_exhibitions": 5, "photography": 4, "cultural": 4},
    },
    "cinephile": {
        "share": 3,
        "weights": {"cinema": 6, "cultural": 2},
    },
    "music_head": {
        "share": 3,
        "weights": {"music_concerts": 6, "cultural": 2},
    },
    "fashion_forward": {
        "share": 2,
        "weights": {"fashion": 6, "art_exhibitions": 2, "photography": 2},
    },
    "eclectic_browser": {
        "share": 2,
        # Deliberately near-flat: a control group of users who don't cluster
        # cleanly into any one archetype, same as a real population would have.
        "weights": {
            "art_exhibitions": 1, "photography": 1, "cinema": 1,
            "music_concerts": 1, "fashion": 1, "cultural": 1,
        },
    },
    "workshop_learner": {
        "share": 2,
        # Fallback tilt only — actual picks come from the workshop keyword
        # pool below whenever it's non-empty; see POOL_BIAS.
        "weights": {"cultural": 3, "photography": 1},
        "pool": "workshop",
    },
    "night_owl": {
        "share": 2,
        # No plan carries a time-of-day field (start_date is date-only), so
        # this is a category tilt toward nightlife-associated categories,
        # not a real "only goes out late" filter.
        "weights": {"music_concerts": 5, "cinema": 3, "cultural": 1},
    },
    "trend_chaser": {
        "share": 2,
        # Flat baseline — actual picks are concentrated on the seeded
        # "popular" pool below whenever it's non-empty; see POOL_BIAS.
        "weights": {
            "art_exhibitions": 1, "photography": 1, "cinema": 1,
            "music_concerts": 1, "fashion": 1, "cultural": 1,
        },
        "pool": "popular",
    },
    # workshop_learner/trend_chaser above prove the pool mechanism already
    # solves cross-user clustering for keyword-matched content; these two
    # apply the same idea to "cultural" specifically — currently ~47% of
    # Madrid's catalog and, on its own, too broad and heterogeneous
    # (flamenco shows next to water parks next to escape rooms) for
    # category-weighted sampling alone to produce meaningful co-attendance.
    # Splitting it via ARCHETYPE rather than adding real DB categories keeps
    # this entirely inside the synthetic generator — no reclassification.
    "trip_lover": {
        "share": 2,
        "weights": {"cultural": 3},
        "pool": "trip",
    },
    "afterwork_guru": {
        "share": 2,
        "weights": {"cultural": 3, "music_concerts": 1},
        "pool": "afterwork",
    },
}

ALL_CATEGORIES = ["art_exhibitions", "photography", "cinema", "music_concerts", "fashion", "cultural"]

# Added to every category's weight for every archetype so no plan has zero
# chance of being picked — see the comment on ARCHETYPES above.
NOISE_FLOOR = 0.4

# Archetypes with a "pool" key (see ARCHETYPES) draw most of their picks from
# a curated plan list instead of category weights — the `weights` above are
# just the fallback used when that pool is empty. POOL_BIAS is the
# probability of drawing from the pool on any given interaction.
POOL_BIAS = 0.7

WORKSHOP_KEYWORDS = ("workshop", "taller", "clase", "curso", "class")

# Heuristic, same spirit/limitations as WORKSHOP_KEYWORDS — simple substring
# matching on title+description, not a real classifier. Both cut across
# `cultural` (and a bit beyond) to carve out two sub-themes that category
# alone can't separate.
TRIP_KEYWORDS = (
    "toledo", "segovia", "aquopolis", "aquarium", "acuario", "excursion",
    "excursión", "water park", "zoo", "ávila", "avila", "burgos", "globo",
    "hiking", "horseback", "a caballo", "en caballo", "barranquismo", "day trip",
)
AFTERWORK_KEYWORDS = (
    "escape room", "flamenco", "candlelight", "murder mystery", "comedy",
    "monólogo", "monologue", "pub crawl", "cocktail", "karaoke", "cabaret",
    "tablao", "impro", "stand-up", "quiz room", "trivia",
)

# No real popularity signal exists right after a fresh regenerate (every
# plan starts at 0 interactions), so trend_chaser draws from a seeded
# pseudo-random subset per category instead of an actual interaction count.
POPULAR_SHARE = 0.2

# saved > view_link > click, matching the strength ordering the Tier 1
# recommender already assumes (README) — saved is the strongest signal, so
# it should be the rarest one, not the most common.
INTERACTION_TYPES = ["click", "view_link", "saved"]
INTERACTION_WEIGHTS = [0.55, 0.25, 0.20]

MIN_INTERACTIONS_PER_USER = 5
MAX_INTERACTIONS_PER_USER = 40

# A user's first pick in a category is unconstrained (nothing established
# yet to compare against), but subsequent same-category picks are drawn
# from that user's own top-SAME_CATEGORY_TOP_K_FRAC most semantically
# similar remaining candidates in it, instead of uniform-random across the
# whole category. Without this, one synthetic user's "cultural" interactions
# (currently ~47% of Madrid's catalog, everything from flamenco shows to
# escape rooms to spa treatments) end up scattered across totally unrelated
# plans purely by chance. LightGCN then learns that random scatter as if it
# were real co-attendance, pulling unrelated plans' embeddings together on
# nothing but sampling noise.
#
# Rank-based, not raw-cosine-weighted: measured directly against this
# project's actual Gemini embeddings, within-category similarity is
# compressed into a narrow band (e.g. "cultural" plans: median pairwise
# cosine 0.65, barely above the CROSS-category cultural-vs-fashion mean of
# 0.61) — probably because `category` itself is one of the embedded text
# fields (gemini_embeddings.plan_text), which dominates over the more
# fine-grained title/description content. Weighting by raw cosine value
# barely moved anything (measured: 0.647 -> 0.652 mean intra-user
# coherence); picking from the top-K% by RANK instead exploits whatever
# fine-grained ordering survives that compression even when the absolute
# differences are tiny (measured: 0.647 -> 0.792).
SAME_CATEGORY_TOP_K_FRAC = 0.1
SAME_CATEGORY_TOP_K_MIN = 3  # floor for small categories (e.g. photography, ~20 plans)


def _pick_archetype(rng: random.Random) -> str:
    names = list(ARCHETYPES)
    shares = [ARCHETYPES[n]["share"] for n in names]
    return rng.choices(names, weights=shares, k=1)[0]


def _matches_keywords(plan: dict, keywords: tuple[str, ...]) -> bool:
    text = f"{plan.get('title', '')} {plan.get('description', '')}".lower()
    return any(kw in text for kw in keywords)


def _plan_vectors(plans: list[dict]) -> dict[int, np.ndarray]:
    """Normalized semantic embedding per plan (skips any plan without one —
    picking within a category just falls back to uniform for those)."""
    vecs = {}
    for p in plans:
        emb = p.get("embedding")
        if not emb:
            continue
        v = np.array(json.loads(emb), dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm > 0:
            vecs[p["id"]] = v / norm
    return vecs


def _pick_in_category(
    rng: random.Random, candidates: list[dict], taste_vec: np.ndarray | None, plan_vec: dict[int, np.ndarray],
) -> dict:
    """Uniform random for a user's first pick in this category (nothing
    established yet). Once taste_vec exists (the running-average embedding
    of this user's earlier picks in this SAME category), pick uniformly
    among the top-SAME_CATEGORY_TOP_K_FRAC candidates ranked by similarity
    to it — see the comment on that constant for why rank, not raw value."""
    if taste_vec is None:
        return rng.choice(candidates)
    ranked = sorted(
        candidates,
        key=lambda p: -float(plan_vec[p["id"]] @ taste_vec) if p["id"] in plan_vec else 1.0,
    )
    k = max(SAME_CATEGORY_TOP_K_MIN, int(len(ranked) * SAME_CATEGORY_TOP_K_FRAC))
    return rng.choice(ranked[:k])


def _user_category_weights(archetype: str, rng: random.Random) -> dict[str, float]:
    """Archetype base weights, jittered per-user so two users of the same
    archetype aren't identical draws — otherwise every 'cinephile' would be
    an indistinguishable duplicate node in the graph."""
    base = ARCHETYPES[archetype]["weights"]
    return {
        cat: base.get(cat, 0) + NOISE_FLOOR * rng.uniform(0.5, 1.5)
        for cat in ALL_CATEGORIES
    }


def generate(city: str, n_users: int, seed: int | None, dry_run: bool) -> Counter:
    rng = random.Random(seed)
    init_db()

    plans = get_all_city_plans(city)
    by_category: dict[str, list[dict]] = {cat: [] for cat in ALL_CATEGORIES}
    for p in plans:
        if p.get("category") in by_category:
            by_category[p["category"]].append(p)

    empty = [cat for cat, ps in by_category.items() if not ps]
    if empty:
        logger.warning("No plans for category(ies) %s in %s — those weights go unused.", empty, city)
    if not any(by_category.values()):
        raise SystemExit(f"No categorized plans found for {city!r}. Nothing to generate against.")

    plan_vec = _plan_vectors(plans)

    workshop_plans = [p for p in plans if _matches_keywords(p, WORKSHOP_KEYWORDS)]
    if not workshop_plans:
        logger.warning(
            "No plans match workshop keywords %s in %s — workshop_learner falls back to category weights.",
            WORKSHOP_KEYWORDS, city,
        )
    trip_plans = [p for p in plans if _matches_keywords(p, TRIP_KEYWORDS)]
    if not trip_plans:
        logger.warning(
            "No plans match trip keywords %s in %s — trip_lover falls back to category weights.",
            TRIP_KEYWORDS, city,
        )
    afterwork_plans = [p for p in plans if _matches_keywords(p, AFTERWORK_KEYWORDS)]
    if not afterwork_plans:
        logger.warning(
            "No plans match afterwork keywords %s in %s — afterwork_guru falls back to category weights.",
            AFTERWORK_KEYWORDS, city,
        )

    # Seeded pseudo-random "hyped" subset per category — see POPULAR_SHARE.
    popular_plans = [
        p
        for cat, ps in by_category.items() if ps
        for p in rng.sample(ps, max(1, round(len(ps) * POPULAR_SHARE)))
    ]
    pools = {
        "workshop": workshop_plans, "popular": popular_plans,
        "trip": trip_plans, "afterwork": afterwork_plans,
    }

    stats = Counter()
    for i in range(n_users):
        archetype = _pick_archetype(rng)
        user_id = f"{USER_PREFIX}{archetype}_{i:04d}"
        weights = _user_category_weights(archetype, rng)
        pool_plans = pools.get(ARCHETYPES[archetype].get("pool"), [])

        # Only weight categories that actually have plans, or random.choices
        # would be free to pick an empty bucket and crash on the empty sample.
        available_cats = [c for c in ALL_CATEGORIES if by_category[c]]
        cat_weights = [weights[c] for c in available_cats]

        n_interactions = rng.randint(MIN_INTERACTIONS_PER_USER, MAX_INTERACTIONS_PER_USER)
        seen: set[tuple[int, str]] = set()
        # Running (sum, count) per category, for this user only — lets each
        # subsequent same-category pick lean toward this user's own earlier
        # picks in it. Reset per user, never shared across users.
        cat_taste_sum: dict[str, np.ndarray] = {}
        cat_taste_count: dict[str, int] = {}
        for _ in range(n_interactions):
            if pool_plans and rng.random() < POOL_BIAS:
                plan = rng.choice(pool_plans)
                category = plan.get("category") or "uncategorized"
            else:
                category = rng.choices(available_cats, weights=cat_weights, k=1)[0]
                taste_vec = (
                    cat_taste_sum[category] / cat_taste_count[category]
                    if cat_taste_count.get(category) else None
                )
                plan = _pick_in_category(rng, by_category[category], taste_vec, plan_vec)
                if plan["id"] in plan_vec:
                    cat_taste_sum[category] = cat_taste_sum.get(category, 0) + plan_vec[plan["id"]]
                    cat_taste_count[category] = cat_taste_count.get(category, 0) + 1
            interaction_type = rng.choices(INTERACTION_TYPES, weights=INTERACTION_WEIGHTS, k=1)[0]
            key = (plan["id"], interaction_type)
            if key in seen:
                continue
            seen.add(key)

            stats[f"interaction:{interaction_type}"] += 1
            stats[f"category:{category}"] += 1
            if not dry_run:
                record_interaction(user_id, plan["id"], interaction_type)
        stats[f"archetype:{archetype}"] += 1

    stats["users"] = n_users
    return stats


def clear(city: str) -> int:
    """Delete every synthetic interaction for this city's plans (matched by
    the `synthetic_` user_id prefix), so reruns can start from a clean slate."""
    init_db()
    with pool.connection() as conn:
        result = conn.execute(
            """DELETE FROM interactions
               WHERE user_id LIKE %s
                 AND plan_id IN (SELECT id FROM plans WHERE city = %s)""",
            (f"{USER_PREFIX}%", city),
        )
        return result.rowcount


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="Madrid")
    ap.add_argument("--users", type=int, default=150, help="Number of synthetic users to generate")
    ap.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible runs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    ap.add_argument("--clear", action="store_true", help="Delete previously generated synthetic interactions and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.clear:
        n = clear(args.city)
        print(f"Deleted {n} synthetic interaction(s) for {args.city}")
        pool.close()
        return

    stats = generate(args.city, args.users, args.seed, args.dry_run)

    prefix = "[DRY RUN] Would generate" if args.dry_run else "Generated"
    print(f"\n{prefix} for {args.city}:")
    print(f"  users: {stats['users']}")
    print("  archetypes:", {k.split(':')[1]: v for k, v in stats.items() if k.startswith("archetype:")})
    print("  by interaction type:", {k.split(':')[1]: v for k, v in stats.items() if k.startswith("interaction:")})
    print("  by category:", {k.split(':')[1]: v for k, v in stats.items() if k.startswith("category:")})

    pool.close()


if __name__ == "__main__":
    main()
