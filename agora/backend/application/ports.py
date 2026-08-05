"""Port interfaces the application layer depends on. Infrastructure adapters
implement these; application code imports the abstraction here, never a
concrete adapter module — so swapping the LLM vendor, the search engine, or
the database never touches use-case code, only which adapter gets wired up
at the edges (infrastructure/web/api.py, infrastructure/cli/ingest_cli.py).
"""

import json
from abc import ABC, abstractmethod
from typing import Protocol

from agora.backend.domain.schemas import SearchResult


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        ...

    @abstractmethod
    async def parse_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        ...

    @staticmethod
    def _parse_json_safe(raw: str):
        raw = raw.strip()

        # Strip markdown fences
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
            if raw.endswith(fence):
                raw = raw[:-len(fence)]
        raw = raw.strip()

        # Try strict parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Use raw_decode to find the first valid JSON value and its boundary.
        # This handles trailing text after the JSON (e.g. venue descriptions).
        # Only decode from whichever bracket ("[" or "{") appears FIRST in the
        # response — trying the other bracket as a fallback after a failure is
        # wrong: on a truncated array response, raw.index("{") lands on the
        # opening brace of the array's first (complete) element, and raw_decode
        # happily parses just that nested object as if it were the whole
        # response. That silently drops every other item instead of surfacing
        # the truncation as an error the caller can retry on.
        decoder = json.JSONDecoder()
        candidates = [i for i in (raw.find("["), raw.find("{")) if i != -1]
        if candidates:
            idx = min(candidates)
            try:
                obj, _ = decoder.raw_decode(raw, idx)
                return obj
            except (ValueError, json.JSONDecodeError, IndexError):
                pass

        raise json.JSONDecodeError(
            f"Could not parse JSON from LLM response (possibly truncated). "
            f"First 300 chars: {raw[:300]}",
            raw,
            0,
        )


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        ...


class PlanRepository(Protocol):
    """The subset of persistence operations the recommender use-case needs.

    A structural (not nominal) interface: infrastructure/persistence/
    postgres_repository.py satisfies this just by exposing matching
    top-level functions — no wrapper class required, since it's a module of
    functions rather than a class. That's also why this only lists the
    handful of operations recommendation.py actually calls, not the full
    ~30-function surface store.py exposes for scripts/ and the web adapter —
    the port is the CONTRACT THE USE-CASE NEEDS, not the whole repository.
    """

    def get_plans_missing_embedding(self) -> list[dict]: ...
    def set_plan_embedding(self, plan_id: int, embedding: list[float]) -> None: ...
    def get_all_city_plans(self, city: str) -> list[dict]: ...
    def get_recommendations(self, user_id: str, city: str, limit: int = 10) -> list[dict]: ...
    def get_interaction_counts(self, plan_ids: list[int]) -> dict[int, int]: ...
    def get_user_interactions_with_embeddings(self, user_id: str) -> list[dict]: ...
