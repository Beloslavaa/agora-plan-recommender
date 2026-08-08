"""Scoring math for the semantic recommender. Pure: plain dicts/floats in,
plain dicts/floats out — no DB, no embedding API calls. See
application/recommendation.py for the use-case that fetches candidates and
calls this, and infrastructure/embeddings/ for the embedding provider."""

import json
import math

from agora.backend.domain.cinemas import CINEMA_SOURCES

# Stronger implicit signals count for more when building a user's taste
# profile: saving a plan says more than clicking into it, which says more
# than nothing.
INTERACTION_WEIGHT = {"saved": 3.0, "view_link": 2.0, "click": 1.0}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
