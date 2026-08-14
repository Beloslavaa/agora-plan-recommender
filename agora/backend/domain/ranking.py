"""Scoring math for the semantic recommender. Pure: plain dicts/floats in,
plain dicts/floats out — no DB, no embedding API calls. See
application/recommendation.py for the use-case that fetches candidates and
calls this, and infrastructure/embeddings/ for the embedding provider."""

import json

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
    Python loop. NaN for every candidate if neither *profile* nor *items*
    has anything to compare against (the vectorized stand-in for the old
    scalar functions returning None); otherwise a real score per candidate.

    Why "best single match" at all, not just cosine-to-profile: a user with
    several genuinely distinct interests (photography AND arthouse cinema
    AND indie gigs, say) has embeddings that sit far apart from each other —
    their weighted AVERAGE (*profile*) lands in the empty space between
    those clusters, resembling none of them, and systematically loses to
    generically-worded content that (for the opposite reason) also sits in
    that same unspecific middle ground. Scoring by the single best match
    instead credits a candidate for resembling ANY one real past pick — the
    same trick cinema_pseudo_plan's caller already uses to score a cinema by
    its single best-matching movie rather than the catalogue average.
    *items* are weighted the same way user_profile is (saved > view_link >
    click), normalised against the top weight so both scores stay on the
    same ~cosine scale and can be safely combined with max().

    Why vectorized: the equivalent scalar per-candidate loop, at real data
    sizes (3072-dim embeddings, ~800 candidates, a dozen-ish saved items),
    measured at 15s of CPU PER REQUEST — this does the identical math as a
    couple of matrix multiplications, in milliseconds."""
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
