"""Tier 2 recommender use-case: blends the LightGCN graph score (trained
offline — see notebooks/train_lightgcn.ipynb) with Tier 1's semantic score.

The backend never runs the model here. Every embedding is read straight
from Postgres; the only computation on this request path is cosine
similarity and a weighted sum — training is always offline, per AGENTS.md.

Falls through to pure Tier 1 (recommendation.rank_for_user) when there's no
graph signal to work with at all: a user with neither a trained embedding
(user_embeddings) nor any interacted plan carrying one to fold in from (see
domain/ranking.py's fold_in_user_embedding). A single candidate plan missing
a graph_embedding (never trained, or too new for the last export's
cold-start pass) just falls back to its semantic score alone for that one
plan, rather than dropping the whole request to Tier 1.
"""

import json
import logging

import numpy as np

from agora.backend.application import recommendation
from agora.backend.application.ports import PlanRepository
from agora.backend.application.recommendation import cached_city_plans
from agora.backend.domain.ranking import (
    cinema_pseudo_plan,
    fold_in_user_embedding,
    prepare_scoring_items,
    score_candidates,
    user_profile,
)
from agora.backend.infrastructure.persistence import postgres_repository as _default_repository

logger = logging.getLogger(__name__)

# The training notebook's own held-out evaluation (notebooks/train_lightgcn.ipynb)
# found a flat 50/50 blend UNDERPERFORMING pure graph on the current
# (synthetic-archetype-heavy) data — plausibly because graph embeddings
# already carry semantic information via their init + regularization, so
# blending raw semantic back in a second time dilutes signal rather than
# adding new information. Kept at 0.5 to match AGENTS.md's documented
# design until there's enough real interaction volume to retune it
# properly; not a settled number.
ALPHA = 0.5


def _field_scores(
    profile: list[float] | None, items: list[tuple[float, list[float]]], plans: list[dict], field: str,
) -> np.ndarray:
    """score_candidates (see its docstring) over *plans*, but NaN at any
    position where that plan doesn't carry *field* at all — rather than
    scoring it against a meaningless placeholder vector. NaN is also what
    score_candidates itself returns when *profile* is None and *items* is
    empty, i.e. this user has no signal in this embedding space at all; both
    cases get resolved together in _blend_scores below."""
    n = len(plans)
    out = np.full(n, np.nan)
    idx = [i for i, p in enumerate(plans) if p.get(field)]
    if not idx:
        return out
    embeddings = [json.loads(plans[i][field]) for i in idx]
    out[idx] = score_candidates(profile, items, embeddings)
    return out


def _blend_scores(graph_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the old scalar _blend's None-handling:
    ALPHA*graph + (1-ALPHA)*semantic where a plan has both, whichever one
    it has where it only has one, NaN (excluded downstream) where it has
    neither."""
    g_nan = np.isnan(graph_scores)
    s_nan = np.isnan(semantic_scores)
    blended = ALPHA * np.where(g_nan, 0.0, graph_scores) + (1 - ALPHA) * np.where(s_nan, 0.0, semantic_scores)
    result = np.where(g_nan & ~s_nan, semantic_scores, blended)
    result = np.where(s_nan & ~g_nan, graph_scores, result)
    result = np.where(g_nan & s_nan, np.nan, result)
    return result


def rank_for_user(
    user_id: str, city: str, limit: int = 10, repository: PlanRepository = _default_repository,
) -> list[dict]:
    """Score every not-yet-interacted plan in `city` by the graph/semantic
    blend. Falls back to Tier 1 (and, through it, popularity) for users or
    cities with no usable graph signal yet."""
    rows = repository.get_user_interactions_with_embeddings(user_id)
    interacted_ids = {r["plan_id"] for r in rows}

    user_graph = repository.get_user_embedding(user_id, city)
    if user_graph is None:
        user_graph = fold_in_user_embedding(rows)
    semantic_profile = user_profile(rows)

    if user_graph is None and semantic_profile is None:
        return recommendation.rank_for_user(user_id, city, limit, repository)

    # Parsed ONCE per request — scoring runs once per candidate plan (hundreds
    # per city), so this must not re-parse the same handful of saved items'
    # embeddings from JSON on every single call (see prepare_scoring_items).
    graph_items = prepare_scoring_items(rows, "graph_embedding")
    semantic_items = prepare_scoring_items(rows, "embedding")

    candidates, cinemas = cached_city_plans(city, repository)

    scoreable = [p for p in candidates if p["id"] not in interacted_ids]
    graph_scores = _field_scores(user_graph, graph_items, scoreable, "graph_embedding")
    semantic_scores = _field_scores(semantic_profile, semantic_items, scoreable, "embedding")
    final_scores = _blend_scores(graph_scores, semantic_scores)
    scored: list[tuple[float, dict]] = [
        (float(final_scores[i]), scoreable[i])
        for i in range(len(scoreable))
        if not np.isnan(final_scores[i])
    ]

    # Same "score a cinema by its single best-matching movie" trick Tier 1
    # uses — see recommendation.py's _rank_with_semantic for the rationale.
    for domain, (info, movies) in cinemas.items():
        embedded = [m for m in movies if m["id"] not in interacted_ids]
        if not embedded:
            continue
        g_scores = _field_scores(user_graph, graph_items, embedded, "graph_embedding")
        s_scores = _field_scores(semantic_profile, semantic_items, embedded, "embedding")
        movie_scores = _blend_scores(g_scores, s_scores)
        movie_scores = movie_scores[~np.isnan(movie_scores)]
        if movie_scores.size == 0:
            continue
        scored.append((float(movie_scores.max()), cinema_pseudo_plan(domain, info, movies)))

    if not scored:
        return recommendation.rank_for_user(user_id, city, limit, repository)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out
