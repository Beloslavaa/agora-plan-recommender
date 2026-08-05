"""Tier 1 recommender use-case: content-based ranking from plan-text
embeddings.

No user-user or item-item interaction graph involved — a plan's embedding
comes purely from its own title/category/tags/description. A user's taste
profile is the weighted average of the embeddings of plans they've clicked,
saved, or followed the ticket link for. Ranking a city's plans for a user is
then just cosine similarity between that profile and each candidate's
embedding.

This works from day one with zero interaction history (falls back to
popularity) and cold-starts new plans immediately (embed the text, done) —
unlike a graph-based recommender, which needs real co-consumption volume
across many users to learn anything. See notebooks/ for that exploration.
"""

import json
import logging
import time

from agora.backend.application.ports import PlanRepository
from agora.backend.domain.cinemas import CINEMA_SOURCES
from agora.backend.domain.ranking import cinema_domain, cinema_pseudo_plan, cosine, user_profile
from agora.backend.infrastructure.embeddings.gemini_embeddings import embed_texts, plan_text
from agora.backend.infrastructure.persistence import postgres_repository as _default_repository

logger = logging.getLogger(__name__)


def backfill_embeddings(batch_size: int = 100, repository: PlanRepository = _default_repository) -> int:
    """Embed every plan that doesn't have one yet (new scrapes, or anything
    ingested before this column existed). Safe to re-run — only ever touches
    rows where embedding IS NULL. Returns how many were embedded."""
    missing = repository.get_plans_missing_embedding()
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        vectors = embed_texts([plan_text(p) for p in batch])
        for plan, vector in zip(batch, vectors):
            repository.set_plan_embedding(plan["id"], vector)
    return len(missing)


# Rebuilding this on every /recommendations call used to mean up to 6 network
# round-trips to Supabase per request (~1-2s each — the actual source of a
# slow reload, not the similarity math itself, per profiling). Plans only
# change via ingestion runs (a separate process), so a short TTL trades a
# little staleness for cutting that to ~1 query per city per TTL window.
_CACHE_TTL_SECONDS = 60.0
_city_plans_cache: dict[str, tuple[float, tuple[list[dict], dict[str, tuple[dict, list[dict]]]]]] = {}


def _cached_city_plans(
    city: str, repository: PlanRepository = _default_repository,
) -> tuple[list[dict], dict[str, tuple[dict, list[dict]]]]:
    """(scoring_candidates, {domain: (info, movies)}) for a city, from ONE
    query instead of one for the main list plus one ILIKE query per cinema."""
    now = time.monotonic()
    hit = _city_plans_cache.get(city)
    if hit and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]

    scoring: list[dict] = []
    cinema_movies: dict[str, list[dict]] = {}
    for row in repository.get_all_city_plans(city):
        domain = cinema_domain(row.get("source_url"))
        if domain:
            cinema_movies.setdefault(domain, []).append(row)
        else:
            scoring.append(row)

    cinemas = {
        domain: (CINEMA_SOURCES[domain], movies)
        for domain, movies in cinema_movies.items()
    }
    result = (scoring, cinemas)
    _city_plans_cache[city] = (now, result)
    return result


def _rank_with_semantic(
    profile: list[float], city: str, limit: int, interacted_ids: set[int],
    repository: PlanRepository = _default_repository,
) -> list[dict]:
    candidates, cinemas = _cached_city_plans(city, repository)
    scored: list[tuple[float, dict]] = []
    for plan in candidates:
        if plan["id"] in interacted_ids or not plan.get("embedding"):
            continue
        scored.append((cosine(profile, json.loads(plan["embedding"])), plan))

    # Score a cinema by its single best-matching movie — "if we'd recommend
    # one movie from here, show the cinema card" — rather than pinning every
    # cinema to the top of the feed regardless of relevance.
    for domain, (info, movies) in cinemas.items():
        # Exclude movies the user's already interacted with from the cinema's
        # own scoring pool too — otherwise saving a movie makes its cinema
        # match itself perfectly (cosine 1.0) rather than being scored on
        # whether the REST of its catalogue is a good match.
        embedded = [m for m in movies if m.get("embedding") and m["id"] not in interacted_ids]
        if not embedded:
            continue
        best = max(cosine(profile, json.loads(m["embedding"])) for m in embedded)
        scored.append((best, cinema_pseudo_plan(domain, info, movies)))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out


def _rank_with_popularity(
    user_id: str, city: str, limit: int, repository: PlanRepository = _default_repository,
) -> list[dict]:
    _, cinemas = _cached_city_plans(city, repository)
    # Headroom so merging in cinema candidates doesn't crowd out plans that
    # would otherwise have made the cut (mirrors get_recommendations' own
    # `limit + len(interacted)` overfetch for the same reason).
    base = repository.get_recommendations(user_id, city, limit + len(cinemas))
    scored: list[tuple[float, dict]] = [(row["score"], row) for row in base]

    movie_ids = [m["id"] for _, movies in cinemas.values() for m in movies]
    counts = repository.get_interaction_counts(movie_ids)
    for domain, (info, movies) in cinemas.items():
        best = max((counts.get(m["id"], 0) for m in movies), default=0)
        scored.append((float(best), cinema_pseudo_plan(domain, info, movies)))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out


def rank_for_user(
    user_id: str, city: str, limit: int = 10, repository: PlanRepository = _default_repository,
) -> list[dict]:
    """Score every not-yet-interacted plan in `city` (cinemas included, via
    their single best-matching movie) by cosine similarity to the user's
    weighted taste profile. Falls back to popularity for brand-new users
    with no usable history, or if nothing in the city has an embedding yet."""
    rows = repository.get_user_interactions_with_embeddings(user_id)
    interacted_ids = {r["plan_id"] for r in rows}

    profile = user_profile(rows)
    if profile is None:
        return _rank_with_popularity(user_id, city, limit, repository)

    ranked = _rank_with_semantic(profile, city, limit, interacted_ids, repository)
    if not ranked:
        return _rank_with_popularity(user_id, city, limit, repository)
    return ranked
