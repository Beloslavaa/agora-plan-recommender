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

from agora.backend.application import recommendation
from agora.backend.application.ports import PlanRepository
from agora.backend.application.recommendation import cached_city_plans
from agora.backend.domain.ranking import cinema_pseudo_plan, cosine, fold_in_user_embedding, user_profile
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


def _blend(user_graph: list[float] | None, semantic_profile: list[float] | None, plan: dict) -> float | None:
    graph_score = None
    if user_graph is not None and plan.get("graph_embedding"):
        graph_score = cosine(user_graph, json.loads(plan["graph_embedding"]))

    semantic_score = None
    if semantic_profile is not None and plan.get("embedding"):
        semantic_score = cosine(semantic_profile, json.loads(plan["embedding"]))

    if graph_score is None and semantic_score is None:
        return None
    if graph_score is None:
        return semantic_score
    if semantic_score is None:
        return graph_score
    return ALPHA * graph_score + (1 - ALPHA) * semantic_score


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

    candidates, cinemas = cached_city_plans(city, repository)
    scored: list[tuple[float, dict]] = []
    for plan in candidates:
        if plan["id"] in interacted_ids:
            continue
        score = _blend(user_graph, semantic_profile, plan)
        if score is not None:
            scored.append((score, plan))

    # Same "score a cinema by its single best-matching movie" trick Tier 1
    # uses — see recommendation.py's _rank_with_semantic for the rationale.
    for domain, (info, movies) in cinemas.items():
        movie_scores = []
        for m in movies:
            if m["id"] in interacted_ids:
                continue
            score = _blend(user_graph, semantic_profile, m)
            if score is not None:
                movie_scores.append(score)
        if not movie_scores:
            continue
        scored.append((max(movie_scores), cinema_pseudo_plan(domain, info, movies)))

    if not scored:
        return recommendation.rank_for_user(user_id, city, limit, repository)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out
