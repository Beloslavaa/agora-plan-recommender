"""Embedding provider adapter: text -> vector, via Gemini's OpenAI-compatible
endpoint. This is the only file that talks to the embeddings API — the
recommender use-case (application/recommendation.py) calls embed_texts and
never sees the client."""

import json

from openai import OpenAI

from agora.backend.infrastructure.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    # Same pattern GeminiProvider in infrastructure/llm/providers.py uses for
    # chat — sync client since this runs from a sync FastAPI route and a
    # plain script, neither of which need async here.
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _client


def plan_text(row: dict) -> str:
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
