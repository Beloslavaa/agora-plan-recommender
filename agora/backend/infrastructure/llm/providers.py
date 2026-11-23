import logging

from agora.backend.application.ports import LLMProvider
from agora.backend.infrastructure.config import settings
from agora.backend.infrastructure.llm.quota import raise_if_quota_exhausted

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.anthropic_model

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
        try:
            raw = await self.complete(prompt, system, temperature, max_tokens)
        except Exception as e:
            raise_if_quota_exhausted(e)
            raise
        return self._parse_json_safe(raw)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model or settings.openai_model

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
        try:
            raw = await self.complete(prompt, system, temperature, max_tokens)
        except Exception as e:
            raise_if_quota_exhausted(e)
            raise
        return self._parse_json_safe(raw)


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._model = model or settings.gemini_model
        # Flipped off the first time this model rejects reasoning_effort (see
        # complete()), so one 400 doesn't repeat on every later call.
        self._send_reasoning_effort = True

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
        kwargs = dict(model=self._model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if self._send_reasoning_effort:
            # Gemini flash models spend part of max_tokens on invisible "thinking"
            # tokens before writing the response, and the split is unpredictable
            # run to run (observed 11k-15k thinking tokens on the same prompt) —
            # that repeatedly ate almost the whole budget and truncated the
            # visible JSON mid-array. This task is plain extraction with no need
            # for reasoning, so keep thinking off via "none". (Historically
            # "none" 400d on gemini-flash-latest while "minimal" worked; as of
            # 2026-08 that flipped — gemini-flash-latest now rejects "minimal"
            # ("Thinking level MINIMAL is not supported for this model") while
            # "none" works. "none" is confirmed to work on both gemini-flash-
            # latest and the pinned gemini-2.5-flash, so it's the safe choice.)
            kwargs["reasoning_effort"] = "none"
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # A model that has no thinking to switch off (or takes a different
            # level, as above) 400s on the parameter itself. Retry once
            # without it — such models don't think by default anyway. Some
            # reject it with only a generic "Request contains an invalid
            # argument" (gemini-flash-lite-latest, 2026-11), so that counts too.
            msg = str(e).lower()
            rejected = (
                "reasoning_effort" in kwargs
                and getattr(e, "status_code", None) == 400
                and ("reasoning" in msg or "thinking" in msg or "invalid_argument" in msg)
            )
            if not rejected:
                raise
            logger.warning("%s rejected reasoning_effort=none — sending without it from now on", self._model)
            self._send_reasoning_effort = False
            kwargs.pop("reasoning_effort", None)
            resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def parse_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        try:
            raw = await self.complete(prompt, system, temperature, max_tokens)
        except Exception as e:
            raise_if_quota_exhausted(e)
            raise
        return self._parse_json_safe(raw)


def get_llm_provider(light: bool = False) -> LLMProvider:
    """The configured provider. *light*: its cheaper model (settings.*_light_model),
    for small batched tasks — see enrichment.generate_short_titles."""
    providers = {
        "anthropic": (AnthropicProvider, settings.anthropic_light_model),
        "openai": (OpenAIProvider, settings.openai_light_model),
        "gemini": (GeminiProvider, settings.gemini_light_model),
    }
    entry = providers.get(settings.llm_provider)
    if not entry:
        msg = f"Unknown LLM provider: {settings.llm_provider}. Available: {list(providers)}"
        raise ValueError(msg)
    cls, light_model = entry
    return cls(light_model if light else None)