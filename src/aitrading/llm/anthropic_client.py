"""Claude implementation of ``StructuredLLM`` (Anthropic Python SDK).

Choices
-------
* Model ``claude-opus-5-5`` by default (override with ``AITRADING_MODEL``).
* Adaptive thinking (Claude Opus 5.5 cannot disable thinking); depth controlled with
  ``output_config.effort`` - set explicitly because this model defaults to "medium".
* Structured outputs via ``client.beta.messages.parse(output_format=<pydantic model>)`` - the
  response is schema-constrained and validated, so no JSON scraping.
* Server-side refusal fallback (``fallbacks="default"``, beta ``server-side-fallback-2026-07-01``):
  if a safety classifier declines, the API re-runs the request on Anthropic's recommended fallback
  model inside the same call. A refusal on the final response means the whole chain declined.
* The system prompt is marked cacheable; it is identical across candidates in a run, so per-ticker
  explanation calls read it from cache.
"""

from __future__ import annotations

import os
import time
from typing import Any

from aitrading.core.models import LLMCallRecord
from aitrading.llm.base import LLMError, LLMRefusalError, T

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(
        self,
        model: str | None = None,
        *,
        effort: str = "high",
        use_fallbacks: bool = True,
        client: Any | None = None,
        max_retries: int = 3,
        timeout_s: float = 600.0,
    ):
        self.model = model or os.environ.get("AITRADING_MODEL", DEFAULT_MODEL)
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - dependency is declared in pyproject
                raise LLMError("the 'anthropic' package is required for AnthropicLLM") from e
            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)
        self._client = client
        self.calls: list[LLMCallRecord] = []

    @property
    def name(self) -> str:
        return self.model

    def structured(
        self,
        *,
        purpose: str,
        system: str,
        user: str | list[dict[str, Any]],
        output_model: type[T],
        effort: str | None = None,
        max_tokens: int = 16_000,
    ) -> T:
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or self.effort},
        )
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        record = LLMCallRecord(purpose=purpose, model=self.model)
        t0 = time.monotonic()
        try:
            resp = self._client.beta.messages.parse(**kwargs)
        except anthropic.BadRequestError as e:
            record.error = f"bad_request: {e.message}"
            raise LLMError(f"[{purpose}] request rejected: {e.message}") from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            record.error = f"auth: {e.message}"
            raise LLMError(f"[{purpose}] authentication/permission error: {e.message}") from e
        except anthropic.RateLimitError as e:
            record.error = "rate_limited"
            raise LLMError(f"[{purpose}] rate limited after SDK retries") from e
        except anthropic.APIStatusError as e:
            record.error = f"api_status_{e.status_code}"
            raise LLMError(f"[{purpose}] API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            record.error = "connection"
            raise LLMError(f"[{purpose}] connection error: {e}") from e
        finally:
            record.latency_s = round(time.monotonic() - t0, 3)
            self.calls.append(record)

        record.request_id = getattr(resp, "_request_id", None)
        record.stop_reason = resp.stop_reason
        record.model = getattr(resp, "model", None) or self.model
        usage = resp.usage
        record.input_tokens = usage.input_tokens or 0
        record.output_tokens = usage.output_tokens or 0
        record.cache_read_input_tokens = usage.cache_read_input_tokens or 0
        record.cache_creation_input_tokens = usage.cache_creation_input_tokens or 0
        record.served_by_fallback = any(getattr(it, "type", None) == "fallback_message" for it in (usage.iterations or []))

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            record.error = f"refusal:{category}"
            raise LLMRefusalError(f"[{purpose}] model declined (category={category})", category=category)
        if resp.stop_reason == "max_tokens":
            record.error = "max_tokens"
            raise LLMError(f"[{purpose}] output truncated at max_tokens={max_tokens}")
        parsed = resp.parsed_output
        if parsed is None:
            record.error = "unparsed"
            raise LLMError(f"[{purpose}] no structured output in response (stop_reason={resp.stop_reason})")
        return parsed
