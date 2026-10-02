"""Tests for aitrading.llm.anthropic_client.AnthropicLLM against a mocked Messages API (offline).

The real SDK runs end to end; only the HTTP transport is replaced, so request shapes, response
parsing and error mapping are exercised as in production.
"""

from __future__ import annotations

import json

import anthropic
import pytest

httpx2 = pytest.importorskip("httpx2")  # the HTTP library of anthropic>=1.11
from pydantic import BaseModel, Field

from aitrading.llm.anthropic_client import (
    EFFORT_MIN_MAX_TOKENS,
    FALLBACK_BETA,
    NONSTREAMING_MAX_TOKENS,
    AnthropicLLM,
    output_schema,
)
from aitrading.llm.base import LLMError, LLMOutputError, LLMRefusalError


class Out(BaseModel):
    a: str
    b: int = Field(ge=0, le=10)  # the API schema cannot enforce this bound (transform_schema moves it to the description)


def _message(content, stop_reason, *, stop_details=None, input_tokens=1200, output_tokens=16_000):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": content,
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_input_tokens": 900,
                  "cache_creation_input_tokens": 0},
    }


def _sse(message: dict) -> bytes:
    """The ``message`` as a Messages API event stream (text blocks only)."""
    start = {**message, "content": [], "stop_reason": None, "usage": {**message["usage"], "output_tokens": 1}}
    events = [("message_start", {"type": "message_start", "message": start})]
    for i, block in enumerate(message["content"]):
        events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "text", "text": ""}}))
        events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": {"type": "text_delta", "text": block["text"]}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                     "usage": {"output_tokens": message["usage"]["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def make_llm(message: dict, **kw):
    seen: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        seen.append({"body": body, "headers": dict(request.headers)})
        if body.get("stream"):
            return httpx2.Response(200, content=_sse(message), headers={"content-type": "text/event-stream",
                                                                         "request-id": "req_stream"})
        return httpx2.Response(200, json=message, headers={"request-id": "req_123"})

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return AnthropicLLM(client=client, **kw), seen


def ask(llm, **kw):
    return llm.structured(purpose="explain:ACME", system="sys", user="u", output_model=Out, **kw)


def test_valid_reply_is_parsed_and_the_request_is_shaped_as_documented():
    llm, seen = make_llm(_message([{"type": "text", "text": '{"a": "x", "b": 5}'}], "end_turn"))
    assert ask(llm) == Out(a="x", b=5)
    body, headers = seen[0]["body"], seen[0]["headers"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16_000 and "stream" not in body
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high", "format": output_schema(Out)}
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default" and FALLBACK_BETA in headers.get("anthropic-beta", "")
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    rec = llm.calls[-1]
    assert (rec.stop_reason, rec.request_id, rec.error) == ("end_turn", "req_123", None)
    assert (rec.input_tokens, rec.output_tokens, rec.cache_read_input_tokens) == (1200, 16_000, 900)


def test_truncated_reply_is_an_llm_error_with_a_complete_audit_record():
    # partial JSON + stop_reason=max_tokens: messages.parse() raised a raw pydantic ValidationError here
    llm, _ = make_llm(_message([{"type": "text", "text": '{"a": "hel'}], "max_tokens"))
    with pytest.raises(LLMError, match="truncated at max_tokens=16000") as info:
        ask(llm)
    assert not isinstance(info.value, LLMOutputError)
    rec = llm.calls[-1]
    assert (rec.stop_reason, rec.error, rec.input_tokens, rec.output_tokens) == ("max_tokens", "max_tokens", 1200, 16_000)


def test_refusal_with_partial_output_is_a_refusal_error():
    llm, _ = make_llm(_message([{"type": "text", "text": '{"a": '}], "refusal", stop_details={"type": "refusal", "category": "cyber"}))
    with pytest.raises(LLMRefusalError) as info:
        ask(llm)
    assert info.value.category == "cyber"
    assert llm.calls[-1].stop_reason == "refusal" and llm.calls[-1].error == "refusal:cyber"


def test_schema_violating_reply_is_an_llm_output_error():
    llm, _ = make_llm(_message([{"type": "text", "text": '{"a": "x", "b": 50}'}], "end_turn"))
    with pytest.raises(LLMOutputError, match="b: Input should be less than or equal to 10") as info:
        ask(llm)
    assert isinstance(info.value, LLMError)
    assert [e["loc"] for e in info.value.errors()] == [("b",)]
    rec = llm.calls[-1]
    assert rec.stop_reason == "end_turn" and rec.error.startswith("invalid_output") and rec.output_tokens == 16_000


def test_no_text_block_is_an_llm_error():
    llm, _ = make_llm(_message([], "end_turn"))
    with pytest.raises(LLMError, match="no structured output"):
        ask(llm)
    assert llm.calls[-1].error == "unparsed"


def test_api_errors_are_mapped_and_recorded():
    def handler(request):
        return httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad schema"}})

    client = anthropic.Anthropic(api_key="k", max_retries=0, http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    llm = AnthropicLLM(client=client)
    with pytest.raises(LLMError, match="request rejected"):
        ask(llm)
    assert llm.calls[-1].error.startswith("bad_request")


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_high_efforts_raise_the_budget_and_stream(effort):
    llm, seen = make_llm(_message([{"type": "text", "text": '{"a": "y", "b": 1}'}], "end_turn"))
    assert ask(llm, effort=effort) == Out(a="y", b=1)
    body = seen[0]["body"]
    assert body["max_tokens"] == EFFORT_MIN_MAX_TOKENS[effort] > NONSTREAMING_MAX_TOKENS
    assert body["stream"] is True and body["output_config"]["effort"] == effort
    assert llm.calls[-1].stop_reason == "end_turn" and llm.calls[-1].request_id == "req_stream"


def test_streamed_truncation_is_an_llm_error():
    llm, _ = make_llm(_message([{"type": "text", "text": '{"a": "trunc'}], "max_tokens"), effort="max")
    with pytest.raises(LLMError, match="truncated at max_tokens=64000"):
        ask(llm)
    assert llm.calls[-1].error == "max_tokens"
