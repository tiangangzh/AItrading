"""Claude implementation of ``StructuredLLM`` (Anthropic Python SDK).

Choices
-------
* Model ``claude-opus-5-5`` by default (override with ``AITRADING_MODEL``).
* Adaptive thinking (Claude Opus 5.5 cannot disable thinking); depth controlled with
  ``output_config.effort`` - set explicitly because this model defaults to "medium".
* Structured outputs: ``client.beta.messages.parse`` with ``output_config.format`` carrying the
  output model's JSON schema, built exactly as ``parse(output_format=<pydantic model>)`` builds it
  (``TypeAdapter(model).json_schema()`` through ``anthropic.transform_schema``), so the reply is
  schema-constrained. ``output_format`` itself is not passed: with it the SDK validates the JSON
  inside the request call, so a truncated or refused reply would surface as a raw pydantic
  ``ValidationError`` with no stop reason and no token usage in the audit record. The reply is
  validated here instead, AFTER ``stop_reason`` is checked.
* Failure mapping (every call is appended to ``calls`` with its stop reason, usage and error):
  ``refusal`` -> ``LLMRefusalError``; ``max_tokens`` -> ``LLMError`` (truncated); JSON that does not
  validate as the output model (e.g. a numeric bound the API schema cannot enforce) ->
  ``LLMOutputError`` carrying the pydantic error; API / connection errors -> ``LLMError``.
* Output budget: thinking counts toward ``max_tokens``, so ``xhigh`` / ``max`` effort raise the
  budget to at least ``EFFORT_MIN_MAX_TOKENS``. Budgets above ``NONSTREAMING_MAX_TOKENS`` are
  streamed (a long non-streaming request would outlast the HTTP timeout; the SDK requires streaming
  for them under its default timeout); the final message is handled the same way.
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

from pydantic import TypeAdapter, ValidationError

from aitrading.core.models import LLMCallRecord
from aitrading.llm.base import LLMError, LLMRefusalError, T, output_error

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NONSTREAMING_MAX_TOKENS = 16_000  # larger output budgets are streamed
EFFORT_MIN_MAX_TOKENS = {"xhigh": 32_000, "max": 64_000}  # thinking at these efforts outgrows 16k


def output_schema(output_model: type) -> dict[str, Any]:
    """``output_config.format`` for ``output_model`` (the transformation ``messages.parse`` applies)."""
    import anthropic

    return {"type": "json_schema", "schema": anthropic.transform_schema(TypeAdapter(output_model).json_schema())}


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

    def _send(self, kwargs: dict[str, Any]) -> Any:
        if kwargs["max_tokens"] > NONSTREAMING_MAX_TOKENS:
            with self._client.beta.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
                if getattr(resp, "_request_id", None) is None:
                    try:
                        resp._request_id = getattr(stream, "request_id", None)
                    except Exception:  # noqa: BLE001 - the request id is audit metadata only
                        pass
                return resp
        # parse() without output_format sends output_config.format as given and validates nothing
        return self._client.beta.messages.parse(**kwargs)

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

        level = effort or self.effort
        max_tokens = max(int(max_tokens), EFFORT_MIN_MAX_TOKENS.get(level, 0))
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": level, "format": output_schema(output_model)},
        )
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        record = LLMCallRecord(purpose=purpose, model=self.model)
        t0 = time.monotonic()
        try:
            resp = self._send(kwargs)
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
        except anthropic.APIError as e:  # e.g. an error event inside a stream
            record.error = f"api_error: {e.message}"
            raise LLMError(f"[{purpose}] API error: {e.message}") from e
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
        texts = [b.text for b in resp.content or [] if getattr(b, "type", None) == "text" and getattr(b, "text", "")]
        if not texts:
            record.error = "unparsed"
            raise LLMError(f"[{purpose}] no structured output in response (stop_reason={resp.stop_reason})")
        first: ValidationError | None = None
        for text in texts:  # like ParsedBetaMessage.parsed_output: the first text block that validates
            try:
                return output_model.model_validate_json(text)
            except ValidationError as e:
                first = first or e
        assert first is not None
        record.error = f"invalid_output: {first.error_count()} error(s)"
        raise output_error(purpose, output_model, first) from first
