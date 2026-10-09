"""The OpenAI-compatible provider: every LLM role runs on open models (vLLM, Ollama, llama.cpp ...) or
any OpenAI-compatible API, offline here against a mock transport that imitates real servers."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from pydantic import BaseModel, Field

from aitrading import cli
from aitrading.llm import openai_compatible as oc
from aitrading.llm.base import LLMError, LLMOutputError, LLMRefusalError
from aitrading.llm.openai_compatible import OpenAICompatibleLLM, extract_json, inline_refs
from aitrading.strategy.nl import StrategyTranslator
from aitrading.strategy.spec import StrategySpec


class Inner(BaseModel):
    feature: str
    weight: float = 1.0


class Out(BaseModel):
    name: str
    parts: list[Inner] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)


def reply(content, *, finish="stop", usage=None, refusal=None, model="served-model", rid="chatcmpl-1"):
    msg = {"role": "assistant", "content": content}
    if refusal is not None:
        msg["refusal"] = refusal
    return httpx.Response(200, json={"id": rid, "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                     "usage": usage or {"prompt_tokens": 120, "completion_tokens": 30,
                                                        "prompt_tokens_details": {"cached_tokens": 100}}})


class Server:
    """Scripted responses; records every request body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        r = self.responses.pop(0)
        return r(request) if callable(r) else r


def llm_for(server: Server, **kw) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM("served-model", base_url="http://llm.local/v1/", api_key=kw.pop("api_key", "k"),
                               client=httpx.Client(transport=httpx.MockTransport(server)), sleep=lambda s: None, **kw)


GOOD = json.dumps({"name": "x", "parts": [{"feature": "return_3d_pct"}], "score": 0.4})


def test_json_schema_request_and_parsed_reply():
    srv = Server(reply(GOOD))
    llm = llm_for(srv)
    out = llm.structured(purpose="t", system="You are a test.", user="hello", output_model=Out, max_tokens=16_000)
    assert out == Out(name="x", parts=[Inner(feature="return_3d_pct")], score=0.4)
    body = srv.requests[0]
    assert body["model"] == "served-model" and body["max_tokens"] == 8192  # capped for small local context windows
    assert body["messages"][0]["role"] == "system" and body["messages"][1] == {"role": "user", "content": "hello"}
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "Out" and rf["json_schema"]["strict"] is False
    schema_text = json.dumps(rf["json_schema"]["schema"])
    assert "$ref" not in schema_text and "$defs" not in schema_text and "feature" in schema_text
    assert "reasoning_effort" not in body and srv.headers[0]["authorization"] == "Bearer k"
    rec = llm.calls[-1]
    assert (rec.purpose, rec.model, rec.request_id, rec.stop_reason) == ("t", "served-model", "chatcmpl-1", "stop")
    assert (rec.input_tokens, rec.output_tokens, rec.cache_read_input_tokens, rec.error) == (120, 30, 100, None)


def test_reasoning_models_fences_and_prose_are_handled():
    srv = Server(reply("<think>The schema wants a name...</think>\n```json\n" + GOOD + "\n```"),
                 reply("Sure! Here it is: " + GOOD + " Hope this helps."),
                 reply([{"type": "text", "text": GOOD}]))
    llm = llm_for(srv)
    for _ in range(3):
        assert llm.structured(purpose="t", system="s", user="u", output_model=Out).name == "x"
    assert extract_json("reasoning... </think> {\"a\": \"}\"}") == '{"a": "}"}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_invalid_output_raises_llm_output_error_for_repair_rounds():
    srv = Server(reply(json.dumps({"name": "x", "score": 7})), reply("I cannot produce JSON"))
    llm = llm_for(srv)
    with pytest.raises(LLMOutputError) as e:
        llm.structured(purpose="t", system="s", user="u", output_model=Out)
    assert any(err["loc"] == ("score",) for err in e.value.errors()) and llm.calls[-1].error.startswith("invalid_output")
    with pytest.raises(LLMOutputError):
        llm.structured(purpose="t", system="s", user="u", output_model=Out)


def test_unsupported_response_format_degrades_and_is_remembered():
    bad = httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported by this model"}})
    bad2 = httpx.Response(400, json={"error": {"message": "Invalid value for response_format: json_object"}})
    srv = Server(bad, bad2, reply(GOOD), reply(GOOD))
    llm = llm_for(srv)
    assert llm.structured(purpose="t", system="SYS", user="u", output_model=Out).name == "x"
    assert [r.get("response_format", {}).get("type") for r in srv.requests] == ["json_schema", "json_object", None]
    assert "JSON schema" in srv.requests[2]["messages"][0]["content"] and '"score"' in srv.requests[2]["messages"][0]["content"]
    llm.structured(purpose="t", system="SYS", user="u", output_model=Out)
    assert "response_format" not in srv.requests[3] and llm.structured_mode == "prompt"


def test_max_completion_tokens_and_context_window_adaptation():
    srv = Server(httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."}}),
                 httpx.Response(400, json={"error": {"message": "This model's maximum context length is 4096 tokens. However, you "
                                                                "requested 9192 tokens (1000 in the messages, 8192 in the completion)."}}),
                 reply(GOOD))
    llm = llm_for(srv)
    llm.structured(purpose="t", system="s", user="u", output_model=Out)
    assert "max_tokens" in srv.requests[0] and srv.requests[1]["max_completion_tokens"] == 8192
    assert srv.requests[2]["max_completion_tokens"] == 4096 - 1000 - 16


def test_transient_errors_are_retried_and_reported():
    slept: list[float] = []
    srv = Server(httpx.Response(429, headers={"retry-after": "2"}, json={"error": "slow down"}),
                 httpx.Response(503, text="overloaded"), reply(GOOD))
    llm = OpenAICompatibleLLM("m", base_url="http://x/v1", api_key="", sleep=slept.append,
                              client=httpx.Client(transport=httpx.MockTransport(srv)))
    assert llm.structured(purpose="t", system="s", user="u", output_model=Out).name == "x"
    assert slept == [2.0, 2.0] and "authorization" not in srv.headers[0]  # no key, no header (local servers)
    always = Server(*[httpx.Response(429, json={"error": "x"})] * 4)
    llm2 = OpenAICompatibleLLM("m", base_url="http://x/v1", sleep=lambda s: None, max_retries=3,
                               client=httpx.Client(transport=httpx.MockTransport(always)))
    with pytest.raises(LLMError, match="rate limited"):
        llm2.structured(purpose="t", system="s", user="u", output_model=Out)

    def boom(request):
        raise httpx.ConnectError("refused")

    down = OpenAICompatibleLLM("m", base_url="http://x/v1", sleep=lambda s: None, max_retries=1,
                               client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(LLMError, match="is the server running"):
        down.structured(purpose="t", system="s", user="u", output_model=Out)
    assert down.calls[-1].latency_s >= 0


def test_auth_refusal_truncation_and_not_found():
    cases = [
        (httpx.Response(401, json={"error": {"message": "Incorrect API key"}}), LLMError, "authentication/permission error"),
        (reply(None, refusal="I can't help with that."), LLMRefusalError, "declined"),
        (reply("", finish="content_filter"), LLMRefusalError, "content filter"),
        (reply('{"name": "x", "sco', finish="length"), LLMError, "truncated"),
        (httpx.Response(404, json={"error": {"message": "model 'nope' not found"}}), LLMError, "not found"),
    ]
    for resp, exc, text in cases:
        llm = llm_for(Server(resp))
        with pytest.raises(exc, match=text):
            llm.structured(purpose="t", system="s", user="u", output_model=Out)
        assert llm.calls[-1].error
    assert cli.is_auth_error(LLMError("[t] authentication/permission error: Incorrect API key"))


def test_inline_refs_and_model_required(monkeypatch):
    schema = {"$defs": {"A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}}, "B": {"type": "string"}},
              "type": "object", "properties": {"a": {"$ref": "#/$defs/A", "description": "kept"}}}
    assert inline_refs(schema) == {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}},
                                                                          "description": "kept"}}}
    rec = {"$defs": {"N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}}, "$ref": "#/$defs/N"}
    with pytest.raises(ValueError):
        inline_refs(rec)
    monkeypatch.delenv("AITRADING_MODEL", raising=False)
    with pytest.raises(LLMError, match="--model"):
        OpenAICompatibleLLM(base_url="http://x/v1")


def test_a_platform_role_runs_on_an_open_model():
    """The strategy translator (validate-and-repair rounds included) works unchanged on an open model."""
    spec = StrategySpec(name="momentum_3d", idea="x", kind="cross_sectional", rebalance="weekly",
                        signal=[{"feature": "return_3d_pct", "direction": "higher_is_better"}])
    bad = spec.model_copy(update={"signal": [{"feature": "return_3_day", "direction": "higher_is_better"}]})
    srv = Server(reply("<think>pick a feature</think>" + bad.model_dump_json()), reply(spec.model_dump_json()))
    tr = StrategyTranslator(llm_for(srv), today=date(2026, 10, 1))
    out = tr.translate("3-day momentum")
    assert out.attempts == 2 and out.spec.signal[0].feature == "return_3d_pct" and out.translator == "served-model"
    assert "unknown feature 'return_3_day'" in srv.requests[1]["messages"][1]["content"]


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


OBS = "US mid-caps with RSI under 40 and FCF yield above 4%"


def test_cli_selects_the_open_model_engine(capsys, monkeypatch):
    from aitrading.screen.spec import Condition, RankFactor, ScreenSpec

    screen = ScreenSpec(name="t", observation=OBS, conditions=[Condition(feature="rsi_14", op="<", value=40),
                                                               Condition(feature="fcf_yield_pct", op=">", value=4)],
                        ranking=[RankFactor(feature="fcf_yield_pct", direction="higher_is_better")])
    srv = Server(reply(screen.model_dump_json()))
    monkeypatch.setattr(oc, "_make_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(srv)))
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    code = cli.main(["spec", "--llm", "openai", "--llm-base-url", "http://localhost:11434/v1", "--model", "qwen2.5:14b", OBS])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert [c.describe() for c in ScreenSpec.model_validate_json(out.out).conditions] == ["rsi_14 < 40", "fcf_yield_pct > 4"]
    assert "translator: qwen2.5:14b" in out.err and srv.requests[0]["model"] == "qwen2.5:14b"


def test_cli_open_model_engine_from_the_environment_and_guidance(capsys, monkeypatch):
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AITRADING_LLM", "openai")
    assert cli.llm_provider() == "openai" and not cli.llm_configured()
    monkeypatch.setenv("AITRADING_LLM_BASE_URL", "http://localhost:8000/v1")
    assert cli.llm_configured()
    monkeypatch.delenv("AITRADING_MODEL", raising=False)
    code = cli.main(["spec", "--claude", OBS])  # LLM required, but no --model for the open endpoint
    err = capsys.readouterr().err
    assert code == 1 and "--model" in err and "--offline" in err and "Ollama" in err and "ANTHROPIC_API_KEY" not in err


def test_cli_default_engine_is_still_claude(capsys, monkeypatch):
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    assert cli.llm_provider() == "claude" and not cli.llm_configured()
    code = cli.main(["spec", OBS])
    err = capsys.readouterr().err
    assert code == 0 and "No Anthropic credentials found" in err and "--llm openai" in err
