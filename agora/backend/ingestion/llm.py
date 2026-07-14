import json
import logging
from abc import ABC, abstractmethod

from agora.backend.config import settings

logger = logging.getLogger(__name__)


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


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        kwargs = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = await self._client.messages.create(**kwargs)
        # Return the first text block rather than assuming content[0] is text —
        # a future model could emit a non-text block (e.g. tool use) first.
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return "".join(getattr(b, "text", "") for b in resp.content)

    async def parse_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        raw = await self.complete(prompt, system, temperature, max_tokens)
        return self._parse_json_safe(raw)


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        from openai import AsyncOpenAI
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = settings.openai_model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def parse_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        raw = await self.complete(prompt, system, temperature, max_tokens)
        return self._parse_json_safe(raw)


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._model = settings.gemini_model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # gemini-2.5-flash spends part of max_tokens on invisible "thinking"
            # tokens before writing the response, and the split is unpredictable
            # run to run (observed 11k-15k thinking tokens on the same prompt) —
            # that repeatedly ate almost the whole budget and truncated the
            # visible JSON mid-array. This task is plain extraction with no need
            # for reasoning, so turn thinking off entirely.
            reasoning_effort="none",
        )
        return resp.choices[0].message.content or ""

    async def parse_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        raw = await self.complete(prompt, system, temperature, max_tokens)
        return self._parse_json_safe(raw)


def get_llm_provider() -> LLMProvider:
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    cls = providers.get(settings.llm_provider)
    if not cls:
        msg = f"Unknown LLM provider: {settings.llm_provider}. Available: {list(providers)}"
        raise ValueError(msg)
    return cls()