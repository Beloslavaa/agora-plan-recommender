from abc import ABC, abstractmethod

import logging

import httpx

from agora.backend.config import settings
from agora.backend.ingestion.schemas import SearchResult

logger = logging.getLogger(__name__)


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        ...


class DuckDuckGoSearch(SearchProvider):
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        import asyncio

        def _run() -> list[SearchResult]:
            from duckduckgo_search import DDGS

            out: list[SearchResult] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append(
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                        )
                    )
            return out

        try:
            # DDGS is synchronous/blocking — run it off the event loop.
            return await asyncio.to_thread(_run)
        except ImportError:
            raise RuntimeError(
                "duckduckgo_search not installed. Run: pip install duckduckgo_search"
            )


class SerpApiSearch(SearchProvider):
    BASE = "https://serpapi.com/search"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        params = {
            "q": query,
            "api_key": settings.serpapi_key,
            "engine": "google",
            "num": max_results,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.BASE, params=params)
            data = resp.json()
        if "error" in data:
            raise RuntimeError(f"SerpAPI: {data['error']}")
        if resp.status_code != 200:
            raise RuntimeError(f"SerpAPI HTTP {resp.status_code}: {data}")
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in data.get("organic_results", [])
        ]
        logger.debug("  SerpAPI: %d results for '%s'", len(results), query[:60])
        return results


class SerperSearch(SearchProvider):
    BASE = "https://google.serper.dev/search"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.BASE,
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": settings.serper_api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in data.get("organic", [])
        ]


class GoogleCustomSearch(SearchProvider):
    BASE = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        params = {
            "q": query,
            "key": settings.google_api_key,
            "cx": settings.google_cse_id,
            "num": min(max_results, 10),
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
            )
            for r in data.get("items", [])
        ]


def get_search_provider(kind: str | None = None) -> SearchProvider:
    kind = kind or settings.search_provider
    providers: dict[str, type[SearchProvider]] = {
        "duckduckgo": DuckDuckGoSearch,
        "serpapi": SerpApiSearch,
        "serper": SerperSearch,
        "google": GoogleCustomSearch,
    }
    cls = providers.get(kind)
    if not cls:
        msg = f"Unknown search provider: {kind}. Available: {list(providers)}"
        raise ValueError(msg)
    return cls()