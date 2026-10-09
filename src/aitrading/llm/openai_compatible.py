"""Any OpenAI-compatible chat-completions server as a :class:`~aitrading.llm.base.StructuredLLM`.

With this, every LLM role in the platform runs on Claude (``AnthropicLLM``) or on any server that
speaks the OpenAI ``/chat/completions`` protocol. The roles are the screen and strategy translators,
the explainer, the backtest interpreter, the idea extractor, the hypothesis generator and the LLM
judge. Compatible servers include:

* open-weights models served locally by vLLM, SGLang, Ollama, llama.cpp ``llama-server`` or LM Studio;
* hosted endpoints: OpenAI, DeepSeek, Together, Groq, OpenRouter, Azure-style gateways.

Configuration (constructor arguments, else environment):

* ``base_url`` - ``AITRADING_LLM_BASE_URL``, else ``OPENAI_BASE_URL``, else ``https://api.openai.com/v1``.
  Examples: ``http://localhost:8000/v1`` (vLLM), ``http://localhost:11434/v1`` (Ollama).
* ``api_key`` - ``AITRADING_LLM_API_KEY``, else ``OPENAI_API_KEY``. Local servers usually need none.
* ``model`` - required: the served model name (``AITRADING_MODEL`` when not given).
* ``max_output_tokens`` - cap on the completion budget (``AITRADING_LLM_MAX_TOKENS``, default 8192).
  Local models often have small context windows; callers may ask for up to 16k.

Structured output, in ``structured_mode`` order, degrading automatically when a server rejects a mode:

1. ``json_schema`` - ``response_format = {"type": "json_schema", "json_schema": {...}}``. ``$ref`` /
   ``$defs`` are inlined because several engines do not resolve references. ``strict`` is false,
   because pydantic schemas with defaults do not meet OpenAI's strict-mode rules.
2. ``json_object`` - ``response_format = {"type": "json_object"}`` plus the schema in the system prompt.
3. ``prompt`` - no ``response_format``; the schema is in the system prompt.

Whatever the mode, the reply is validated here against the pydantic model:

* ``<think>...</think>`` blocks and code fences are stripped;
* the first JSON object in the text is used;
* a reply that does not validate raises ``LLMOutputError``, so the translators run their repair rounds
  exactly as they do with Claude.

Other request handling:

* A 400 response asking for ``max_completion_tokens`` instead of ``max_tokens`` (newer OpenAI
  models) switches the parameter.
* A 400 response that names the context window (vLLM: "maximum context length is N tokens ... (M in
  the messages ...") shrinks the completion budget to fit.
* 408 / 409 / 429 / 5xx responses and connection errors are retried with exponential backoff,
  honouring ``Retry-After``.
* ``reasoning_effort`` is sent only when enabled (``send_reasoning_effort=True`` or
  ``AITRADING_LLM_REASONING_EFFORT=1``), because most servers reject unknown parameters.

Failure mapping matches ``AnthropicLLM``. Every call is appended to ``calls`` as an ``LLMCallRecord``
with the model, request id, token usage (``cached_tokens`` counts as cache reads), stop reason
(``finish_reason``), latency and error.

* refusals (``message.refusal`` or ``finish_reason == "content_filter"``) -> ``LLMRefusalError``;
* truncation (``finish_reason == "length"``) -> ``LLMError``;
* 401 / 403 -> ``LLMError`` containing "authentication/permission error";
* other failures -> ``LLMError``.

Not available through this provider: the Claude-specific features that call the Anthropic API
directly. These are web search in ``discover`` (Claude's server-side web search tool) and the
interactive research agent (tool runner).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from typing import Any, Callable

from pydantic import TypeAdapter, ValidationError

from aitrading.core.models import LLMCallRecord
from aitrading.llm.base import LLMError, LLMRefusalError, T, output_error

__all__ = [
    "OpenAICompatibleLLM",
    "DEFAULT_BASE_URL",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "ENV_MAX_TOKENS",
    "ENV_REASONING_EFFORT",
    "STRUCTURED_MODES",
    "inline_refs",
    "extract_json",
    "configured_base_url",
    "configured_api_key",
]

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
ENV_BASE_URL = "AITRADING_LLM_BASE_URL"
ENV_API_KEY = "AITRADING_LLM_API_KEY"
ENV_MAX_TOKENS = "AITRADING_LLM_MAX_TOKENS"
ENV_REASONING_EFFORT = "AITRADING_LLM_REASONING_EFFORT"
STRUCTURED_MODES = ("json_schema", "json_object", "prompt")
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MIN_COMPLETION_TOKENS = 256
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def configured_base_url() -> str | None:
    """The base URL from the environment (``AITRADING_LLM_BASE_URL``, else ``OPENAI_BASE_URL``), or None."""
    for k in (ENV_BASE_URL, "OPENAI_BASE_URL"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def configured_api_key() -> str | None:
    for k in (ENV_API_KEY, "OPENAI_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------------------------------------
# Schema and reply helpers
# ------------------------------------------------------------------------------------------------


def inline_refs(schema: dict[str, Any], *, max_depth: int = 32) -> dict[str, Any]:
    """Copy of a JSON schema with every local ``$ref`` (``#/$defs/X`` or ``#/definitions/X``) replaced
    by the definition it points to and the ``$defs`` / ``definitions`` sections dropped. Raises
    ``ValueError`` for recursive references deeper than ``max_depth``."""
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}

    def walk(node: Any, depth: int) -> Any:
        if depth > max_depth:
            raise ValueError("schema references nest too deeply (recursive model?)")
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                target = defs.get(ref.rsplit("/", 1)[-1])
                if target is None:
                    raise ValueError(f"unresolvable schema reference {ref}")
                merged = {**copy.deepcopy(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return walk(merged, depth + 1)
            return {k: walk(v, depth) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [walk(x, depth) for x in node]
        return node

    return walk(schema, 0)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S | re.I)


def _first_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text`` (string- and escape-aware), or None."""
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start: i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def extract_json(text: str) -> str:
    """The JSON object text inside a model reply: drops ``<think>`` blocks (closed, or an unclosed
    one that ends at ``</think>``), code fences and surrounding prose. Raises ``ValueError`` when
    there is no JSON object."""
    t = _THINK_RE.sub("", text or "")
    if "</think>" in t:  # an opening tag the server stripped (e.g. a chat template that adds it)
        t = t.rsplit("</think>", 1)[1]
    t = t.strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        json.loads(t)
        return t
    except ValueError:
        pass
    obj = _first_object(t)
    if obj is None:
        raise ValueError("no JSON object in the reply")
    return obj


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some servers return content parts
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content)


def _user_text(user: str | list[dict[str, Any]]) -> str:
    if isinstance(user, str):
        return user
    parts: list[str] = []
    for block in user:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif block.get("type") == "document" and isinstance(block.get("source"), dict) and block["source"].get("type") == "text":
            title = block.get("title") or "document"
            parts.append(f"<document title=\"{title}\">\n{block['source'].get('data', '')}\n</document>")
        else:
            raise LLMError(f"content block type {block.get('type')!r} is not supported by OpenAI-compatible servers")
    return "\n\n".join(parts)


def _error_message(resp: Any) -> str:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return (getattr(resp, "text", "") or "")[:500]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:500]
        if err:
            return str(err)[:500]
        if body.get("message"):
            return str(body["message"])[:500]
        if body.get("detail"):
            return str(body["detail"])[:500]
    return str(body)[:500]


_CTX_RE = re.compile(r"maximum context length is (\d+) tokens.*?(\d+) in the messages", re.S | re.I)


def _make_client(timeout_s: float) -> Any:
    import httpx

    return httpx.Client(timeout=timeout_s)


# ------------------------------------------------------------------------------------------------
# The provider
# ------------------------------------------------------------------------------------------------


class OpenAICompatibleLLM:
    """StructuredLLM over an OpenAI-compatible ``/chat/completions`` endpoint (see the module docstring)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        effort: str = "high",
        structured_mode: str = "json_schema",
        send_reasoning_effort: bool | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 3,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        model = (model or os.environ.get("AITRADING_MODEL", "")).strip()
        if not model:
            raise LLMError("no model name: pass --model (the served model, e.g. qwen2.5:14b or gpt-4.1) or set AITRADING_MODEL")
        if structured_mode not in STRUCTURED_MODES:
            raise ValueError(f"structured_mode must be one of {STRUCTURED_MODES}")
        self.model = model
        self.base_url = (base_url or configured_base_url() or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else (configured_api_key() or "")
        self.effort = effort
        self.structured_mode = structured_mode
        self.send_reasoning_effort = (_truthy(os.environ.get(ENV_REASONING_EFFORT))
                                      if send_reasoning_effort is None else bool(send_reasoning_effort))
        env_cap = os.environ.get(ENV_MAX_TOKENS, "").strip()
        self.max_output_tokens = int(max_output_tokens or (int(env_cap) if env_cap.isdigit() else DEFAULT_MAX_OUTPUT_TOKENS))
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep
        self._max_tokens_key = "max_tokens"
        self.calls: list[LLMCallRecord] = []

    def __repr__(self) -> str:
        return f"OpenAICompatibleLLM(model={self.model!r}, base_url={self.base_url!r}, mode={self.structured_mode!r})"

    @property
    def name(self) -> str:
        return self.model

    def _http(self) -> Any:
        if self._client is None:
            self._client = _make_client(self.timeout_s)
        return self._client

    # ------------------------------------------------------------------ request building
    @staticmethod
    def _schema(output_model: type) -> dict[str, Any]:
        schema = TypeAdapter(output_model).json_schema()
        try:
            return inline_refs(schema)
        except ValueError:
            return schema

    def _payload(self, system: str, user: str | list[dict[str, Any]], output_model: type, mode: str, level: str,
                 max_tokens: int) -> dict[str, Any]:
        schema = self._schema(output_model)
        if mode == "json_schema":
            sys_text = f"{system}\n\nReply with one JSON object that matches the response schema, and nothing else."
        else:
            sys_text = (f"{system}\n\n# Output format\nReply with ONE JSON object and nothing else (no prose, no code "
                        f"fences). It must validate against this JSON schema:\n{json.dumps(schema, separators=(',', ':'))}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": _user_text(user)}],
            self._max_tokens_key: int(max_tokens),
        }
        if mode == "json_schema":
            name = _NAME_RE.sub("_", output_model.__name__)[:64] or "output"
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": False}}
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.send_reasoning_effort and level in _EFFORT_MAP:
            payload["reasoning_effort"] = _EFFORT_MAP[level]
        return payload

    def _post(self, purpose: str, payload: dict[str, Any]) -> Any:
        """POST with retries on transient failures; returns the final response (any status)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        attempt = 0
        while True:
            try:
                resp = self._http().post(url, headers=headers, json=payload)
            except Exception as exc:  # noqa: BLE001 - httpx transport errors (connect, read timeout ...)
                if attempt >= self.max_retries:
                    raise LLMError(f"[{purpose}] connection error talking to {self.base_url} ({type(exc).__name__}: "
                                   f"{exc}); is the server running?") from exc
                self._sleep(min(2.0 ** attempt, 20.0))
                attempt += 1
                continue
            if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                wait = min(2.0 ** attempt, 20.0)
                ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
                try:
                    wait = min(max(float(ra), 0.0), 60.0) if ra is not None else wait
                except ValueError:
                    pass
                self._sleep(wait)
                attempt += 1
                continue
            return resp

    # ------------------------------------------------------------------ public API
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
        level = effort or self.effort
        budget = max(MIN_COMPLETION_TOKENS, min(int(max_tokens), self.max_output_tokens))
        mode = self.structured_mode
        record = LLMCallRecord(purpose=purpose, model=self.model)
        t0 = time.monotonic()
        try:
            resp, mode = self._request(purpose, system, user, output_model, mode, level, budget, record)
        finally:
            record.latency_s = round(time.monotonic() - t0, 3)
            self.calls.append(record)
        return self._parse(purpose, resp, output_model, record, budget)

    def _request(self, purpose: str, system: str, user: Any, output_model: type, mode: str, level: str, budget: int,
                 record: LLMCallRecord) -> tuple[Any, str]:
        tried_keys: set[str] = set()
        for _ in range(8):  # bounded number of adaptations (mode, token parameter, budget, effort)
            try:
                payload = self._payload(system, user, output_model, mode, level, budget)
            except LLMError as e:
                record.error = f"bad_request: {e}"
                raise LLMError(f"[{purpose}] {e}") from e
            resp = self._post(purpose, payload)
            status = resp.status_code
            if status < 400:
                if mode != self.structured_mode:
                    log.warning("%s at %s rejected structured mode %s; using %s", self.model, self.base_url,
                                self.structured_mode, mode)
                    self.structured_mode = mode  # remember for later calls
                return resp, mode
            msg = _error_message(resp)
            low = msg.lower()
            if status == 400:
                if ("max_completion_tokens" in low or ("max_tokens" in low and "not supported" in low)) \
                        and self._max_tokens_key == "max_tokens" and "max_tokens" not in tried_keys:
                    tried_keys.add("max_tokens")
                    self._max_tokens_key = "max_completion_tokens"
                    continue
                m = _CTX_RE.search(msg)
                if m:
                    fit = int(m.group(1)) - int(m.group(2)) - 16
                    if MIN_COMPLETION_TOKENS <= fit < budget:
                        budget = fit
                        continue
                if "reasoning_effort" in low and self.send_reasoning_effort:
                    self.send_reasoning_effort = False
                    continue
                if mode != "prompt" and any(k in low for k in ("response_format", "json_schema", "json_object",
                                                               "structured output", "guided", "grammar")):
                    mode = STRUCTURED_MODES[STRUCTURED_MODES.index(mode) + 1]
                    continue
                record.error = f"bad_request: {msg[:200]}"
                raise LLMError(f"[{purpose}] request rejected by {self.base_url}: {msg}")
            if status in (401, 403):
                record.error = f"auth: {msg[:200]}"
                raise LLMError(f"[{purpose}] authentication/permission error: {msg}")
            if status == 404:
                record.error = "not_found"
                raise LLMError(f"[{purpose}] model or endpoint not found (base URL {self.base_url}, model {self.model}): {msg}")
            if status == 429:
                record.error = "rate_limited"
                raise LLMError(f"[{purpose}] rate limited after {self.max_retries} retries")
            record.error = f"api_status_{status}"
            raise LLMError(f"[{purpose}] API error {status}: {msg}")
        record.error = "bad_request: too many adaptations"
        raise LLMError(f"[{purpose}] the server kept rejecting the request ({self.base_url})")

    def _parse(self, purpose: str, resp: Any, output_model: type[T], record: LLMCallRecord, budget: int) -> T:
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            record.error = "unparsed"
            raise LLMError(f"[{purpose}] the server's reply is not JSON") from e
        record.request_id = (data.get("id") if isinstance(data, dict) else None) or resp.headers.get("x-request-id")
        record.model = (data.get("model") if isinstance(data, dict) else None) or self.model
        usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        record.input_tokens = int(usage.get("prompt_tokens") or 0)
        record.output_tokens = int(usage.get("completion_tokens") or 0)
        record.cache_read_input_tokens = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            record.error = "unparsed"
            raise LLMError(f"[{purpose}] no choices in the reply")
        choice = choices[0] or {}
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        record.stop_reason = finish
        if msg.get("refusal"):
            record.error = "refusal:model"
            raise LLMRefusalError(f"[{purpose}] model declined: {str(msg['refusal'])[:300]}", category="model")
        if finish == "content_filter":
            record.error = "refusal:content_filter"
            raise LLMRefusalError(f"[{purpose}] the server's content filter blocked the reply", category="content_filter")
        text = _content_text(msg.get("content"))
        if finish == "length":
            record.error = "max_tokens"
            raise LLMError(f"[{purpose}] output truncated at max_tokens={budget} (raise {ENV_MAX_TOKENS} if the model "
                           "allows a longer completion)")
        if not text.strip():
            record.error = "unparsed"
            raise LLMError(f"[{purpose}] empty reply (finish_reason={finish})")
        try:
            obj_text = extract_json(text)
        except ValueError:
            obj_text = text  # let pydantic produce the JSON error, so callers can run a repair round
        try:
            return output_model.model_validate_json(obj_text)
        except ValidationError as e:
            record.error = f"invalid_output: {e.error_count()} error(s)"
            raise output_error(purpose, output_model, e) from e
