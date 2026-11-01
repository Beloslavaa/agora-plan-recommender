"""Recognise a provider's "out of credit" error and turn it into the
application's QuotaExhaustedError. Shared by the chat providers
(providers.py) and the embeddings adapter."""

from agora.backend.application.ports import QuotaExhaustedError


def raise_if_quota_exhausted(e: Exception) -> None:
    """Raise QuotaExhaustedError if *e* means the account can't pay for calls;
    return otherwise so the caller re-raises *e* unchanged.

    - 402: Gemini's prepaid balance is empty ("Your prepayment credits are
      depleted").
    - 429 + code "insufficient_quota": OpenAI's no-credit error. A plain 429
      is a per-minute rate limit — transient, and the SDK already retries it —
      so it is deliberately NOT treated as fatal.
    """
    status = getattr(e, "status_code", None)
    if status == 402 or (status == 429 and getattr(e, "code", None) == "insufficient_quota"):
        raise QuotaExhaustedError(str(e)) from e
