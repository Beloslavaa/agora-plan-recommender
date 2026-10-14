"""Tier 2 recommender use-case: blends the LightGCN graph score (trained
offline — see notebooks/train_lightgcn.ipynb) with Tier 1's semantic score.

The backend never runs the model here. Every embedding is read straight
from Postgres; the only computation on this request path is cosine
similarity, a percentile-rank normalization (see domain/ranking.py's percentile_rank), and a
weighted sum — training is always offline, per AGENTS.md.

Falls through to pure Tier 1 (recommendation.rank_for_user) when there's no
graph signal to work with at all: a user with neither a trained embedding
(user_embeddings) nor any interacted plan carrying one to fold in from. The
user's position is recomputed live on every request — trained vector blended
with whatever they've done since training (see domain/ranking.py's
live_user_embedding) — so it never waits for a retrain. A single candidate plan missing
a graph_embedding (never trained, or too new for the last export's
cold-start pass) just falls back to its semantic score alone for that one
plan, rather than dropping the whole request to Tier 1.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np

from agora.backend.application import recommendation
from agora.backend.application.ports import PlanRepository
from agora.backend.application.recommendation import cached_city_plans
from agora.backend.domain.ranking import (
    FRESHNESS_BOOST,
    blend_scores,
    cinema_pseudo_plan,
    cold_start_graph_embeddings,
    field_scores,
    live_user_embedding,
    mmr_rerank,
    percentile_rank,
    plan_freshness,
    prepare_scoring_items,
    user_profile,
)

logger = logging.getLogger(__name__)

_freshness_cache: dict[str, tuple[list[dict], dict[int, float]]] = {}


def _city_freshness(city: str, candidates: list[dict], cinemas: dict) -> dict[int, float]:
    """plan id -> plan_freshness, for every plan in the city (cinema movies
    included). The same for every user, so it's computed once per
    cached_city_plans snapshot — that cache hands back the same list object
    until its TTL expires, which is what the identity check keys on —
    rather than re-parsing every embedding on every request."""
    hit = _freshness_cache.get(city)
    if hit and hit[0] is candidates:
        return hit[1]
    everything = candidates + [m for _, movies in cinemas.values() for m in movies]
    scores = plan_freshness(everything, datetime.now(timezone.utc))
    result = {p["id"]: float(s) for p, s in zip(everything, scores)}
    _freshness_cache[city] = (candidates, result)
    return result


def backfill_graph_embeddings(repository: PlanRepository) -> int:
    """Give every live plan that has a semantic embedding but no graph
    embedding a cold-start proxy (see domain/ranking.py's
    cold_start_graph_embeddings), so plans scraped since the last training
    run get a graph score right away instead of waiting for the next
    notebook run. Run after backfill_embeddings — the proxy is built from
    the semantic embedding.

    Per city: each city's graph is trained separately, so anchors from one
    city's graph space mean nothing in another's. A city with no trained
    anchors at all is skipped (its plans stay semantic-only, as before).
    Safe to re-run — only ever touches rows where graph_embedding IS NULL.
    Returns how many were written."""
    by_city: dict[str, list[dict]] = {}
    for plan in repository.get_plans_missing_graph_embedding():
        by_city.setdefault(plan["city"], []).append(plan)

    written = 0
    for city, targets in by_city.items():
        anchors = repository.get_graph_anchor_plans(city)
        if not anchors:
            logger.info("[%s] no trained graph yet — %d plan(s) stay semantic-only", city, len(targets))
            continue
        proxies = cold_start_graph_embeddings(
            np.asarray([json.loads(p["embedding"]) for p in targets]),
            np.asarray([json.loads(a["embedding"]) for a in anchors]),
            np.asarray([json.loads(a["graph_embedding"]) for a in anchors]),
            np.asarray([a["degree"] for a in anchors]),
        )
        repository.set_plan_graph_embeddings_bulk(
            [(p["id"], v.tolist()) for p, v in zip(targets, proxies)]
        )
        written += len(targets)
    return written


def rank_for_user(
    repository: PlanRepository, user_id: str, city: str, limit: int = 10,
) -> list[dict]:
    """Score every not-yet-interacted plan in `city` by the graph/semantic
    blend. Falls back to Tier 1 (and, through it, popularity) for users or
    cities with no usable graph signal yet."""
    rows = repository.get_user_interactions_with_embeddings(user_id)
    interacted_ids = {r["plan_id"] for r in rows}

    # Graph embeddings live in per-city spaces, so only this city's
    # interactions can place the user in it; the semantic space is shared,
    # so that side keeps every row.
    graph_rows = [r for r in rows if r.get("city") == city]
    user_graph = live_user_embedding(repository.get_user_embedding(user_id, city), graph_rows)
    semantic_profile = user_profile(rows)

    if user_graph is None and semantic_profile is None:
        return recommendation.rank_for_user(repository, user_id, city, limit)

    # Parsed ONCE per request — scoring runs once per candidate plan (hundreds
    # per city), so this must not re-parse the same handful of saved items'
    # embeddings from JSON on every single call (see prepare_scoring_items).
    graph_items = prepare_scoring_items(graph_rows, "graph_embedding")
    semantic_items = prepare_scoring_items(rows, "embedding")

    candidates, cinemas = cached_city_plans(city, repository)

    scoreable = [p for p in candidates if p["id"] not in interacted_ids]
    cinema_embedded = {
        domain: [m for m in movies if m["id"] not in interacted_ids]
        for domain, (info, movies) in cinemas.items()
    }

    # Main plans and cinema movies are rank-normalized TOGETHER (one pooled
    # call) rather than each cinema getting its own _percentile_rank call —
    # otherwise every cinema's single best movie would land near rank 1.0
    # just for being the top of its own small catalogue, regardless of how
    # it actually compares to the rest of the city.
    pooled = scoreable + [m for movies in cinema_embedded.values() for m in movies]
    graph_scores = field_scores(user_graph, graph_items, pooled, "graph_embedding")
    semantic_scores = field_scores(semantic_profile, semantic_items, pooled, "embedding")
    final_scores = blend_scores(percentile_rank(graph_scores), percentile_rank(semantic_scores))

    # Recently ingested plans get a multiplicative nudge (see
    # domain/ranking.py's plan_freshness) — applied to the blended score,
    # so it lifts new plans that already match and does nothing for ones
    # that don't. NaN (unscorable) stays NaN.
    freshness = _city_freshness(city, candidates, cinemas)
    final_scores = final_scores * (1 + FRESHNESS_BOOST * np.array([freshness.get(p["id"], 0.0) for p in pooled]))

    n_main = len(scoreable)
    scored: list[tuple[float, dict]] = [
        (float(final_scores[i]), scoreable[i])
        for i in range(n_main)
        if not np.isnan(final_scores[i])
    ]

    # Same "score a cinema by its single best-matching movie" trick Tier 1
    # uses — see recommendation.py's _rank_with_semantic for the rationale.
    offset = n_main
    for domain, embedded in cinema_embedded.items():
        n = len(embedded)
        movie_scores = final_scores[offset:offset + n]
        offset += n
        movie_scores = movie_scores[~np.isnan(movie_scores)]
        if movie_scores.size == 0:
            continue
        info, movies = cinemas[domain]
        scored.append((float(movie_scores.max()), cinema_pseudo_plan(domain, info, movies)))

    if not scored:
        return recommendation.rank_for_user(repository, user_id, city, limit)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in mmr_rerank(scored, limit):
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out
