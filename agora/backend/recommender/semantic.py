"""Tier 1 recommender: content-based ranking from plan-text embeddings.

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
import math
import time

from openai import OpenAI

from agora.backend.cinemas import CINEMA_SOURCES
from agora.backend.config import settings
from agora.backend.ingestion import store

logger = logging.getLogger(__name__)

# Stronger implicit signals count for more when building a user's taste
# profile: saving a plan says more than clicking into it, which says more
# than nothing.
_INTERACTION_WEIGHT = {"saved": 3.0, "view_link": 2.0, "click": 1.0}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    # Gemini's OpenAI-compatible endpoint (same pattern GeminiProvider in
    # llm.py uses for chat) — sync client since this runs from a sync FastAPI
    # route and a plain script, neither of which need async here.
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _client


def _plan_text(row: dict) -> str:
    """The text a plan is embedded from. Title/category first (highest
    signal-to-noise), free-text description last."""
    tags = row.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags) if tags else []
    parts = [row.get("title") or "", row.get("category") or "", ", ".join(tags), row.get("description") or ""]
    return "\n".join(p for p in parts if p)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """One batched embeddings call for the whole list — cheaper and faster
    than one call per plan."""
    resp = _get_client().embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def backfill_embeddings(batch_size: int = 100) -> int:
    """Embed every plan that doesn't have one yet (new scrapes, or anything
    ingested before this column existed). Safe to re-run — only ever touches
    rows where embedding IS NULL. Returns how many were embedded."""
    missing = store.get_plans_missing_embedding()
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        vectors = embed_texts([_plan_text(p) for p in batch])
        for plan, vector in zip(batch, vectors):
            store.set_plan_embedding(plan["id"], vector)
    return len(missing)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _user_profile(rows: list[dict]) -> list[float] | None:
    """Weighted average of the embeddings of plans this user has interacted
    with (rows from get_user_interactions_with_embeddings). None if none of
    those plans have an embedding yet — the caller falls back to popularity."""
    embedded = [r for r in rows if r.get("embedding")]
    if not embedded:
        return None

    dim = len(json.loads(embedded[0]["embedding"]))
    profile = [0.0] * dim
    total_weight = 0.0
    for r in embedded:
        vector = json.loads(r["embedding"])
        weight = _INTERACTION_WEIGHT.get(r["interaction_type"], 1.0)
        for d in range(dim):
            profile[d] += vector[d] * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return [v / total_weight for v in profile]


def _cinema_pseudo_plan(domain: str, info: dict, movies: list[dict]) -> dict:
    """One card standing in for a whole cinema's catalogue (movies from a
    single source get grouped rather than shown individually — see
    cinemas.py). Shaped like a plan row so it can be scored and merged into
    the same ranked list, instead of always being pinned to the top of the
    feed regardless of whether anything in it is actually a good match."""
    image_url = next((m["image_url"] for m in movies if m.get("image_url")), None)
    return {
        "id": f"cinema:{domain}", "is_cinema": True, "cinema_key": domain,
        "title": info["name"], "short_title": info["name"], "description": "",
        "start_date": None, "end_date": None, "url": None, "ticket_url": None,
        "location": None, "image_url": image_url, "price": None, "tags": ["cinema"],
        "category": None, "source_url": "", "source_type": "fixed", "city": info["city"],
    }


def _cinema_domain(source_url: str | None) -> str | None:
    for domain in CINEMA_SOURCES:
        if domain in (source_url or ""):
            return domain
    return None


# Rebuilding this on every /recommendations call used to mean up to 6 network
# round-trips to Supabase per request (~1-2s each — the actual source of a
# slow reload, not the similarity math itself, per profiling). Plans only
# change via ingestion runs (a separate process), so a short TTL trades a
# little staleness for cutting that to ~1 query per city per TTL window.
_CACHE_TTL_SECONDS = 60.0
_city_plans_cache: dict[str, tuple[float, tuple[list[dict], dict[str, tuple[dict, list[dict]]]]]] = {}


def _cached_city_plans(city: str) -> tuple[list[dict], dict[str, tuple[dict, list[dict]]]]:
    """(scoring_candidates, {domain: (info, movies)}) for a city, from ONE
    query instead of one for the main list plus one ILIKE query per cinema."""
    now = time.monotonic()
    hit = _city_plans_cache.get(city)
    if hit and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]

    scoring: list[dict] = []
    cinema_movies: dict[str, list[dict]] = {}
    for row in store.get_all_city_plans(city):
        domain = _cinema_domain(row.get("source_url"))
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


def _rank_with_semantic(profile: list[float], city: str, limit: int, interacted_ids: set[int]) -> list[dict]:
    candidates, cinemas = _cached_city_plans(city)
    scored: list[tuple[float, dict]] = []
    for plan in candidates:
        if plan["id"] in interacted_ids or not plan.get("embedding"):
            continue
        scored.append((_cosine(profile, json.loads(plan["embedding"])), plan))

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
        best = max(_cosine(profile, json.loads(m["embedding"])) for m in embedded)
        scored.append((best, _cinema_pseudo_plan(domain, info, movies)))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out


def _rank_with_popularity(user_id: str, city: str, limit: int) -> list[dict]:
    _, cinemas = _cached_city_plans(city)
    # Headroom so merging in cinema candidates doesn't crowd out plans that
    # would otherwise have made the cut (mirrors get_recommendations' own
    # `limit + len(interacted)` overfetch for the same reason).
    base = store.get_recommendations(user_id, city, limit + len(cinemas))
    scored: list[tuple[float, dict]] = [(row["score"], row) for row in base]

    movie_ids = [m["id"] for _, movies in cinemas.values() for m in movies]
    counts = store.get_interaction_counts(movie_ids)
    for domain, (info, movies) in cinemas.items():
        best = max((counts.get(m["id"], 0) for m in movies), default=0)
        scored.append((float(best), _cinema_pseudo_plan(domain, info, movies)))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    out = []
    for score, plan in scored[:limit]:
        d = dict(plan)
        d["score"] = score
        out.append(d)
    return out


def rank_for_user(user_id: str, city: str, limit: int = 10) -> list[dict]:
    """Score every not-yet-interacted plan in `city` (cinemas included, via
    their single best-matching movie) by cosine similarity to the user's
    weighted taste profile. Falls back to popularity for brand-new users
    with no usable history, or if nothing in the city has an embedding yet."""
    rows = store.get_user_interactions_with_embeddings(user_id)
    interacted_ids = {r["plan_id"] for r in rows}

    profile = _user_profile(rows)
    if profile is None:
        return _rank_with_popularity(user_id, city, limit)

    ranked = _rank_with_semantic(profile, city, limit, interacted_ids)
    if not ranked:
        return _rank_with_popularity(user_id, city, limit)
    return ranked
