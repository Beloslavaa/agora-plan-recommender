"""Scoring math for the semantic recommender. Pure: plain dicts/floats in,
plain dicts/floats out — no DB, no embedding API calls. See
application/recommendation.py for the use-case that fetches candidates and
calls this, and infrastructure/embeddings/ for the embedding provider."""

import json
from datetime import datetime

import numpy as np

from agora.backend.domain.cinemas import CINEMA_SOURCES

# Stronger implicit signals count for more when building a user's taste
# profile: saving a plan says more than clicking into it, which says more
# than nothing.
INTERACTION_WEIGHT = {"saved": 3.0, "view_link": 2.0, "click": 1.0}


def _row_normalise(m: np.ndarray) -> np.ndarray:
    """L2-normalise along the last axis; an all-zero row stays zero instead
    of dividing by zero (matches the old scalar cosine()'s "norm 0 → score
    0" behaviour)."""
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _weighted_profile(rows: list[dict], field: str) -> list[float] | None:
    """Weighted average of a JSON-list vector field across interacted plans,
    weighted the same way for any vector space (saved > view_link > click).
    None if no row has that field yet."""
    embedded = [r for r in rows if r.get(field)]
    if not embedded:
        return None

    dim = len(json.loads(embedded[0][field]))
    profile = [0.0] * dim
    total_weight = 0.0
    for r in embedded:
        vector = json.loads(r[field])
        weight = INTERACTION_WEIGHT.get(r["interaction_type"], 1.0)
        for d in range(dim):
            profile[d] += vector[d] * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return [v / total_weight for v in profile]


def user_profile(rows: list[dict]) -> list[float] | None:
    """Weighted average of the semantic embeddings of plans this user has
    interacted with (rows carrying an "embedding" JSON string and
    "interaction_type"). None if none of those plans have an embedding yet
    — the caller falls back to popularity."""
    return _weighted_profile(rows, "embedding")


def fold_in_user_embedding(rows: list[dict]) -> list[float] | None:
    """Weighted average of the GRAPH embeddings (not semantic) of plans this
    user has interacted with — a one-hop proxy for a user who wasn't in the
    bipartite graph at last training time (see
    notebooks/train_lightgcn.ipynb). Shallower than a properly trained
    embedding (no reciprocal gradient update, no multi-hop propagation of
    its own), but the plans' embeddings already carry real co-consumption
    structure from whoever trained alongside them, so this is still real
    graph signal, not just semantic. None if none of those plans have a
    graph embedding either — caller falls back further, to semantic-only."""
    return _weighted_profile(rows, "graph_embedding")


# How much post-training activity it takes for the live fold-in to count as
# much as the trained vector, in INTERACTION_WEIGHT units: 15 = five saves,
# or fifteen clicks. Lower = the trained vector fades faster.
PIN_FADE_WEIGHT = 15.0


def live_user_embedding(trained: tuple[list[float], datetime] | None, rows: list[dict]) -> list[float] | None:
    """Where this user sits in graph space RIGHT NOW: their trained vector
    (if any) blended with a fold-in of everything they've done since it was
    trained, so a save moves them immediately instead of at the next retrain.

    The trained vector already reflects every interaction up to its
    timestamp, so only LATER rows are folded in (counting earlier ones again
    would double-weight them). The mix leans on the fold-in as that later
    activity piles up — m = w / (w + PIN_FADE_WEIGHT), w = total weight of
    those later rows — so with nothing new it IS the trained vector, and
    after lots of new activity it's mostly the fold-in. Both sides are
    L2-normalised first: only direction matters for cosine scoring, and the
    trained vector and a plain average of plan vectors don't share a scale.

    No trained vector → plain fold-in over every row (a user who wasn't in
    the last training run). *rows* must already be limited to the city
    being ranked — graph spaces are per-city."""
    if trained is None:
        return fold_in_user_embedding(rows)
    vector, trained_at = trained
    recent = [r for r in rows if r.get("graph_embedding") and r["created_at"] > trained_at]
    folded = fold_in_user_embedding(recent)
    if folded is None:
        return vector
    w = sum(INTERACTION_WEIGHT.get(r["interaction_type"], 1.0) for r in recent)
    m = w / (w + PIN_FADE_WEIGHT)
    pin, live = _row_normalise(np.asarray([vector, folded], dtype=np.float64))
    return ((1 - m) * pin + m * live).tolist()


COLDSTART_K = 5         # trained neighbours averaged into one cold-start proxy
COLDSTART_PRIOR = 5     # interactions a neighbour needs to reach 50% trust
COLDSTART_SHORTLIST = 4 # similarity floor: only the top COLDSTART_K * this many are eligible at all


def cold_start_graph_embeddings(
    target_semantic: np.ndarray,
    anchor_semantic: np.ndarray,
    anchor_graph: np.ndarray,
    anchor_degree: np.ndarray,
) -> np.ndarray:
    """Graph-space proxy for plans that were never a node in the trained
    graph: a weighted average of the graph embeddings of their nearest
    trained neighbours ("anchors") in SEMANTIC space. No training — the
    anchors' vectors are only read, never changed. Shared by the training
    notebook's export and ingestion's backfill (graph_recommendation.
    backfill_graph_embeddings), so both produce the same proxies.

    Similarity FLOOR, then confidence breaks ties within it — a flat
    similarity*confidence score let confidence override a huge similarity
    gap: for one plan, a neighbour at only 0.55 similarity but very high
    confidence (31 interactions) beat one at 0.76 similarity with 3
    interactions, dragging the proxy toward a generic well-connected hub
    instead of anything thematically relevant. Two stages instead: (1) take
    the top COLDSTART_K * COLDSTART_SHORTLIST anchors by RAW similarity
    only, then (2) within THAT already-similar shortlist, prefer the more
    confidently-trained ones (degree / (degree + COLDSTART_PRIOR)).
    Confidence can only choose among options that already look alike.

    Shapes: targets (T, d_sem), anchors (A, d_sem) / (A, d_graph) / (A,).
    Returns (T, d_graph)."""
    sims = _row_normalise(np.asarray(target_semantic, dtype=np.float64)) @ \
        _row_normalise(np.asarray(anchor_semantic, dtype=np.float64)).T  # (T, A)
    degree = np.asarray(anchor_degree, dtype=np.float64)
    confidence = degree / (degree + COLDSTART_PRIOR)

    shortlist = np.argsort(-sims, axis=1)[:, :COLDSTART_K * COLDSTART_SHORTLIST]
    shortlist_score = np.clip(np.take_along_axis(sims, shortlist, axis=1), 1e-6, None) * confidence[shortlist]
    top_local = np.argsort(-shortlist_score, axis=1)[:, :COLDSTART_K]
    top = np.take_along_axis(shortlist, top_local, axis=1)                  # (T, k)
    weights = np.take_along_axis(shortlist_score, top_local, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("tk,tkd->td", weights, np.asarray(anchor_graph, dtype=np.float64)[top])


def prepare_scoring_items(rows: list[dict], field: str) -> list[tuple[float, list[float]]]:
    """Parse *field* out of every interacted-plan row ONCE, as (weight, vector)
    pairs, for score_candidates to score every candidate against. Call this
    once per request (in the use-case, before the candidate loop) — not once
    per candidate plan (hundreds per city), which turned a request that used
    to be sub-second into one that hung for minutes at real embedding sizes
    (3072-dim, ~800 candidates)."""
    items = []
    for r in rows:
        if not r.get(field):
            continue
        weight = INTERACTION_WEIGHT.get(r["interaction_type"], 1.0)
        items.append((weight, json.loads(r[field])))
    return items


def score_candidates(
    profile: list[float] | None,
    items: list[tuple[float, list[float]]],
    candidate_embeddings: list[list[float]],
) -> np.ndarray:
    """max(cosine-to-averaged-profile, cosine-to-single-closest-past-pick)
    for EVERY candidate at once — matrix ops instead of a per-candidate
    Python loop (the scalar version measured 15s of CPU per request at real
    data sizes). NaN for a candidate if neither *profile* nor *items* has
    anything to compare against.

    Why best-single-match, not just cosine-to-profile: a user with several
    distinct interests has a weighted-average *profile* that lands in the
    empty space between those clusters, resembling none of them — the same
    fix cinema_pseudo_plan's caller uses for scoring a cinema by its single
    best-matching movie. *items* are weighted like user_profile (saved >
    view_link > click), normalised so both scores stay comparable for max()."""
    n = len(candidate_embeddings)
    if n == 0:
        return np.zeros(0)
    if profile is None and not items:
        return np.full(n, np.nan)

    cand = _row_normalise(np.asarray(candidate_embeddings, dtype=np.float64))
    parts = []

    if profile is not None:
        p = _row_normalise(np.asarray(profile, dtype=np.float64)[None, :])[0]
        parts.append(cand @ p)

    if items:
        top_weight = INTERACTION_WEIGHT["saved"]
        weights = np.asarray([w / top_weight for w, _ in items], dtype=np.float64)
        item_vecs = _row_normalise(np.asarray([v for _, v in items], dtype=np.float64))
        item_sims = (item_vecs @ cand.T) * weights[:, None]  # (n_items, n_candidates)
        parts.append(item_sims.max(axis=0))

    return np.maximum.reduce(parts)


FRESHNESS_BOOST = 0.15         # a brand-new plan's score is multiplied by up to 1 + this
FRESHNESS_HALF_LIFE_DAYS = 15  # ingestion runs on the 1st and 15th — each older batch gets half the boost
FRESHNESS_MIN = 0.05           # below this (~65 days) a plan counts as not fresh at all
NEAR_DUPLICATE_SIM = 0.95      # semantic cosine at/above which a new plan is "the same thing again"


def plan_freshness(plans: list[dict], now: datetime) -> np.ndarray:
    """0-1 per plan: how "new" it is, for boosting recently ingested plans
    that already match the user (the caller MULTIPLIES the relevance score
    by 1 + FRESHNESS_BOOST * this, so freshness can lift a good match but
    never rescue an irrelevant one).

    Halves every FRESHNESS_HALF_LIFE_DAYS since the plan was first ingested
    (plans.created_at — a re-scrape of the same event merges into the
    existing row and keeps it, see upsert_plans). Fading instead of ever
    penalising: a plan the user scrolled past just drifts back to its plain
    relevance score as newer batches arrive.

    Zero for a new plan that's a near-copy (semantic cosine >=
    NEAR_DUPLICATE_SIM) of an OLDER plan in *plans* — the same event
    scraped from another source in different wording (which upsert's
    URL/fuzzy-title dedup can miss), or a recurring event on a new date.
    Neither is news, so neither should jump the feed."""
    age_days = np.array([max(0.0, (now - p["created_at"]).total_seconds() / 86400) for p in plans])
    fresh = 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS)
    fresh[fresh < FRESHNESS_MIN] = 0.0

    embedded = [i for i, p in enumerate(plans) if p.get("embedding")]
    recent = [k for k, i in enumerate(embedded) if fresh[i] > 0]
    if not recent:
        return fresh
    vecs = _row_normalise(np.asarray([json.loads(plans[i]["embedding"]) for i in embedded], dtype=np.float64))
    created = np.array([plans[i]["created_at"].timestamp() for i in embedded])
    sims = vecs[recent] @ vecs.T                                   # (n_recent, n_embedded)
    older = created[None, :] < created[recent][:, None]
    duplicate = ((sims >= NEAR_DUPLICATE_SIM) & older).any(axis=1)
    fresh[[embedded[recent[k]] for k in np.flatnonzero(duplicate)]] = 0.0
    return fresh


MMR_LAMBDA = 0.7  # relevance vs. diversity trade-off; 1.0 = pure relevance, no diversity
MMR_POOL_MULT = 3  # only rerank within the top (limit * this) already-relevant candidates


def mmr_rerank(scored: list[tuple[float, dict]], limit: int) -> list[tuple[float, dict]]:
    """Greedy Maximal Marginal Relevance over an already relevance-sorted
    (score, plan) list. At each step picks whichever unpicked candidate
    maximises MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * similarity to the
    closest already-picked item (cosine in semantic embedding space), instead
    of always taking the next-highest-relevance one.

    Stops a cluster of near-identical plans — dozens of variously-dated
    "Candlelight" concerts, or two scrapes of the same exhibition that
    fuzzy-title dedup (scripts/dedupe_plans.py) didn't catch because the
    wording differed across sources — from filling the whole top-N just
    because each one individually scores well.

    Only reranks within the top `limit * MMR_POOL_MULT` candidates by raw
    relevance — diversity is a tie-breaker among already-good matches, not a
    reason to promote something irrelevant. A plan with no `embedding` (e.g.
    a cinema pseudo-plan) is treated as similar to nothing, so it's never
    penalised or favoured by the diversity term."""
    pool = scored[: min(len(scored), limit * MMR_POOL_MULT)]

    vectors: list[np.ndarray | None] = []
    for _, plan in pool:
        emb = plan.get("embedding")
        vectors.append(_row_normalise(np.asarray(json.loads(emb), dtype=np.float64)[None, :])[0] if emb else None)

    selected: list[int] = []
    remaining = set(range(len(pool)))
    while remaining and len(selected) < limit:
        best_i, best_mmr = -1, -np.inf
        for i in remaining:
            sim = 0.0
            if vectors[i] is not None and selected:
                sims = [float(vectors[i] @ vectors[j]) for j in selected if vectors[j] is not None]
                sim = max(sims) if sims else 0.0
            mmr = MMR_LAMBDA * pool[i][0] - (1 - MMR_LAMBDA) * sim
            if mmr > best_mmr:
                best_i, best_mmr = i, mmr
        selected.append(best_i)
        remaining.discard(best_i)

    return [pool[i] for i in selected]


def cinema_pseudo_plan(domain: str, info: dict, movies: list[dict]) -> dict:
    """One card standing in for a whole cinema's catalogue (movies from a
    single source get grouped rather than shown individually — see
    domain/cinemas.py). Shaped like a plan row so it can be scored and merged
    into the same ranked list, instead of always being pinned to the top of
    the feed regardless of whether anything in it is actually a good match."""
    image_url = next((m["image_url"] for m in movies if m.get("image_url")), None)
    return {
        "id": f"cinema:{domain}", "is_cinema": True, "cinema_key": domain,
        "title": info["name"], "short_title": info["name"], "description": "",
        "start_date": None, "end_date": None, "url": None, "ticket_url": None,
        "location": None, "image_url": image_url, "price": None, "tags": ["cinema"],
        "category": None, "source_url": "", "source_type": "fixed", "city": info["city"],
    }


def cinema_domain(source_url: str | None) -> str | None:
    for domain in CINEMA_SOURCES:
        if domain in (source_url or ""):
            return domain
    return None
