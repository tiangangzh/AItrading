"""Observation -> ScreenSpec translation: LLM translator (via ScriptedLLM) and the offline heuristic."""

from __future__ import annotations

import random
import re

import pytest
from pydantic import BaseModel

from aitrading.llm.base import LLMError, LLMRefusalError, ScriptedLLM
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.nl import (
    SYSTEM_PROMPT_TEMPLATE,
    HeuristicScreenTranslator,
    NLScreenTranslator,
    ScreenTranslationError,
    TranslationResult,
    build_system_prompt,
    translate_observation,
)
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

CAT = default_catalog()

CANONICAL = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but "
    "have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI "
    "under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where "
    "short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then read "
    "the latest earnings calls and explain the dislocation."
)
CANONICAL_CONDITIONS = [
    ("market_cap_usd_bn", "between", 2.0, 20.0),
    ("sma_50_vs_sma_200_pct", ">", 0.0, None),
    ("return_12m_ex_1m_pct", ">", 0.0, None),
    ("drawdown_from_52w_high_pct", "between", -40.0, -15.0),
    ("max_volume_ratio_20d", ">=", 2.0, None),
    ("rsi_14", "<", 40.0, None),
    ("fcf_yield_pct", ">", 4.0, None),
    ("revenue_growth_yoy_pct", ">", 8.0, None),
    ("short_interest_pct_float", ">", 6.0, None),
]
CANONICAL_RANKING = [
    ("fcf_yield_pct", "higher_is_better", 1.0),
    ("revenue_growth_yoy_pct", "higher_is_better", 1.0),
    ("drawdown_from_52w_high_pct", "lower_is_better", 1.0),
]


def canonical_spec(observation: str = CANONICAL) -> ScreenSpec:
    return ScreenSpec(
        name="midcap_uptrend_pullback",
        observation=observation,
        conditions=[Condition(feature=f, op=op, value=v, value_high=hi) for f, op, v, hi in CANONICAL_CONDITIONS],
        ranking=[RankFactor(feature=f, direction=d, weight=w) for f, d, w in CANONICAL_RANKING],
    )


def conds(spec: ScreenSpec) -> list[tuple]:
    return [(c.feature, c.op, c.value, c.value_high) for c in spec.conditions]


def cond_set(spec: ScreenSpec) -> set[tuple]:
    out = set()
    for c in spec.conditions:
        if c.op in ("in", "not_in"):
            out.add((c.feature, c.op, tuple(c.values or ())))
        elif c.other_feature:
            out.add((c.feature, c.op, c.other_feature, c.multiplier))
        elif c.op == "between":
            out.add((c.feature, c.op, c.value, c.value_high))
        else:
            out.add((c.feature, c.op, c.value))
    return out


def ranking(spec: ScreenSpec) -> list[tuple]:
    return [(f.feature, f.direction, f.weight) for f in spec.ranking]


class RecordingLLM:
    """StructuredLLM test double that records every keyword argument (ScriptedLLM does not keep effort)."""

    name = "recording"

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list = []
        self.kwargs: list[dict] = []

    def structured(self, *, purpose, system, user, output_model, effort=None, max_tokens=16_000):
        self.kwargs.append(dict(purpose=purpose, system=system, user=user, output_model=output_model, effort=effort))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ================================================================================================
# NLScreenTranslator
# ================================================================================================


def test_nl_happy_path():
    llm = ScriptedLLM({"nl_screen": lambda p, s, u, m: canonical_spec()})
    res = NLScreenTranslator(llm).translate(CANONICAL)
    assert isinstance(res, TranslationResult)
    assert res.attempts == 1 and res.errors_by_round == [[]] and res.translator == "scripted"
    assert conds(res.spec) == CANONICAL_CONDITIONS and ranking(res.spec) == CANONICAL_RANKING
    assert res.spec.validate_against(CAT) == []
    p = llm.prompts[0]
    assert p["purpose"] == "nl_screen" and p["output_model"] == "ScreenSpec"
    assert f"<observation>\n{CANONICAL}\n</observation>" in p["user"]
    assert "ScreenSpec" in p["user"]
    assert len(llm.calls) == 1 and llm.calls[0].purpose == "nl_screen"


def test_nl_passes_effort_and_output_model():
    llm = RecordingLLM([canonical_spec()])
    NLScreenTranslator(llm, effort="max").translate(CANONICAL)
    kw = llm.kwargs[0]
    assert kw["effort"] == "max" and kw["output_model"] is ScreenSpec and kw["purpose"] == "nl_screen"
    llm2 = RecordingLLM([canonical_spec()])
    NLScreenTranslator(llm2).translate(CANONICAL)
    assert llm2.kwargs[0]["effort"] == "high"


def test_nl_repairs_unknown_feature():
    bad = canonical_spec()
    bad.conditions[6] = Condition(feature="fcf_yeild_pct", op=">", value=4)

    def responder(purpose, system, user, model):
        return bad if purpose == "nl_screen" else canonical_spec()

    llm = ScriptedLLM({"nl_screen": responder})
    res = NLScreenTranslator(llm).translate(CANONICAL)
    assert res.attempts == 2
    assert [p["purpose"] for p in llm.prompts] == ["nl_screen", "nl_screen:repair"]
    assert any("unknown feature 'fcf_yeild_pct'" in e for e in res.errors_by_round[0])
    assert "did you mean" in " ".join(res.errors_by_round[0])
    assert res.errors_by_round[1] == []
    repair_user = llm.prompts[1]["user"]
    assert CANONICAL in repair_user and "fcf_yeild_pct" in repair_user and "<errors>" in repair_user
    assert "<previous_spec>" in repair_user and '"conditions"' in repair_user
    assert llm.prompts[1]["system"] == llm.prompts[0]["system"]
    assert conds(res.spec) == CANONICAL_CONDITIONS


def test_nl_repair_uses_longest_prefix_responder():
    bad = canonical_spec()
    bad.ranking = [RankFactor(feature="gics_sector", direction="higher_is_better")]
    llm = ScriptedLLM({"nl_screen": lambda *a: bad, "nl_screen:repair": lambda *a: canonical_spec()})
    res = NLScreenTranslator(llm).translate(CANONICAL)
    assert res.attempts == 2 and "cannot rank on category feature 'gics_sector'" in res.errors_by_round[0]


def test_nl_still_invalid_raises():
    bad = canonical_spec()
    bad.conditions.append(Condition(feature="dividend_yield_pct", op=">", value=3))
    llm = ScriptedLLM({"nl_screen": lambda *a: bad})
    with pytest.raises(ScreenTranslationError) as ei:
        NLScreenTranslator(llm, max_repair_rounds=2).translate(CANONICAL)
    err = ei.value
    assert isinstance(err, ValueError)
    assert len(llm.prompts) == 3
    assert [p["purpose"] for p in llm.prompts] == ["nl_screen", "nl_screen:repair", "nl_screen:repair"]
    assert any("dividend_yield_pct" in e for e in err.errors)
    assert len(err.errors_by_round) == 3 and all(err.errors_by_round)
    assert err.spec is not None and err.spec.observation == CANONICAL


def test_nl_zero_repair_rounds():
    bad = canonical_spec()
    bad.ranking = []
    llm = ScriptedLLM({"nl_screen": lambda *a: bad})
    with pytest.raises(ScreenTranslationError) as ei:
        NLScreenTranslator(llm, max_repair_rounds=0).translate(CANONICAL)
    assert len(llm.prompts) == 1 and "ranking needs at least one factor" in ei.value.errors


def test_nl_rejects_negative_repair_rounds():
    with pytest.raises(ValueError):
        NLScreenTranslator(ScriptedLLM({}), max_repair_rounds=-1)


def test_nl_system_prompt_identical_across_observations():
    llm = ScriptedLLM({"nl_screen": lambda p, s, u, m: canonical_spec()})
    tr = NLScreenTranslator(llm)
    other = "Large-cap tech names above the 200-day with RSI under 30."
    tr.translate(CANONICAL)
    tr.translate(other)
    s0, s1 = llm.prompts[0]["system"], llm.prompts[1]["system"]
    assert s0 == s1 == tr.system_prompt == build_system_prompt(CAT)
    assert other not in s0
    assert "Large-cap tech names" not in s0
    assert llm.prompts[0]["user"] != llm.prompts[1]["user"]
    # A fresh translator renders the identical bytes (cache hits across runs).
    assert NLScreenTranslator(ScriptedLLM({})).system_prompt == s0


def test_nl_system_prompt_contains_catalog_and_rules():
    sp = build_system_prompt()
    assert CAT.to_prompt() in sp
    assert "{catalog}" not in sp and "{catalog}" in SYSTEM_PROMPT_TEMPLATE
    for f in CAT:
        assert f.name in sp
    for phrase in [
        "conditions` are ANDed", "any_of", "NaN", "inclusive", "other_feature", "multiplier",
        "USD billions", "never 0.05", "NEGATIVE", "between -40 and -15", "percentage points",
        "unsupported_requests", "verbatim", "rsi_14 < 30", "max_volume_ratio_20d >= 2", "eps_revision_3m_pct < 0",
        "iv_rank_1y", "rationale", "Information Technology",
    ]:
        assert phrase in sp, phrase
    assert "2026" not in sp and not re.search(r"\b\d{4}-\d{2}-\d{2}\b", sp)  # no date stamps: cacheable


def test_nl_system_prompt_custom_catalog():
    small = FeatureCatalog([CAT["rsi_14"], CAT["fcf_yield_pct"]])
    sp = build_system_prompt(small)
    assert small.to_prompt() in sp
    assert "- price_vs_sma_200_pct (number)" not in sp


def test_nl_overwrites_observation_and_clamps_top_n():
    paraphrased = canonical_spec("mid caps that pulled back")
    too_many = paraphrased.model_copy(update={"top_n": 500})  # model_copy skips validation, like a lax client
    llm = ScriptedLLM({"nl_screen": lambda *a: too_many})
    res = NLScreenTranslator(llm).translate(CANONICAL)
    assert res.spec.observation == CANONICAL
    assert res.spec.top_n == 100
    zero = canonical_spec().model_copy(update={"top_n": 0})
    res0 = NLScreenTranslator(ScriptedLLM({"nl_screen": lambda *a: zero})).translate(CANONICAL)
    assert res0.spec.top_n == 1


def test_nl_schema_error_triggers_repair():
    raw = canonical_spec().model_dump(mode="json")
    raw["top_n"] = 500  # violates le=100 -> pydantic ValidationError inside the LLM client

    def responder(purpose, system, user, model):
        return raw if purpose == "nl_screen" else canonical_spec()

    llm = ScriptedLLM({"nl_screen": responder})
    res = NLScreenTranslator(llm).translate(CANONICAL)
    assert res.attempts == 2
    assert any("top_n" in e for e in res.errors_by_round[0])
    assert "not a valid ScreenSpec" in llm.prompts[1]["user"]


def test_nl_llm_errors_propagate():
    with pytest.raises(LLMRefusalError):
        NLScreenTranslator(RecordingLLM([LLMRefusalError("declined", category="cyber")])).translate(CANONICAL)
    with pytest.raises(LLMError):
        NLScreenTranslator(ScriptedLLM({})).translate(CANONICAL)  # no responder


@pytest.mark.parametrize("obs", ["", "   ", "\n\t"])
def test_nl_empty_observation(obs):
    llm = RecordingLLM([])
    with pytest.raises(ScreenTranslationError):
        NLScreenTranslator(llm).translate(obs)
    assert llm.kwargs == []


def test_nl_spec_never_mutated_in_place():
    original = canonical_spec("different text")
    llm = ScriptedLLM({"nl_screen": lambda *a: original})
    NLScreenTranslator(llm).translate(CANONICAL)
    assert original.observation == "different text"


def test_translate_observation_falls_back_to_heuristic():
    res = translate_observation(CANONICAL, RecordingLLM([LLMError("boom")]))
    assert res.translator == "heuristic"
    assert conds(res.spec) == CANONICAL_CONDITIONS
    assert any("LLM translation failed" in a for a in res.spec.assumptions)


def test_translate_observation_without_llm_and_with_llm():
    assert translate_observation(CANONICAL).translator == "heuristic"
    res = translate_observation(CANONICAL, ScriptedLLM({"nl_screen": lambda *a: canonical_spec()}))
    assert res.translator == "scripted"
    with pytest.raises(ScreenTranslationError):
        translate_observation("  ", RecordingLLM([]))


def test_translate_observation_falls_back_on_invalid_specs():
    bad = canonical_spec()
    bad.ranking = []
    res = translate_observation(CANONICAL, ScriptedLLM({"nl_screen": lambda *a: bad}), max_repair_rounds=1)
    assert res.translator == "heuristic" and any("ScreenTranslationError" in a for a in res.spec.assumptions)


# ================================================================================================
# HeuristicScreenTranslator - canonical observation
# ================================================================================================


@pytest.fixture(scope="module")
def h() -> HeuristicScreenTranslator:
    return HeuristicScreenTranslator()


def test_heuristic_canonical_exact(h):
    res = h.translate(CANONICAL)
    s = res.spec
    assert res.translator == "heuristic" and res.attempts == 1 and res.errors_by_round == [[]]
    assert conds(s) == CANONICAL_CONDITIONS
    assert ranking(s) == CANONICAL_RANKING
    assert s.any_of == []
    assert s.universe == UniverseSpec()
    assert s.top_n == 10
    assert s.observation == CANONICAL
    assert s.unsupported_requests == []
    assert s.validate_against(CAT) == []
    assert all(c.rationale for c in s.conditions) and all(f.rationale for f in s.ranking)
    assert any("narrative" in a and "earnings calls" in a for a in s.assumptions)
    assert s.name == "mid_cap_uptrend_momentum_pullback"


def test_heuristic_canonical_matches_scripted_llm_spec(h):
    """Offline and LLM paths agree on the canonical demo."""
    a = h.translate(CANONICAL).spec
    b = canonical_spec()
    assert conds(a) == conds(b) and ranking(a) == ranking(b) and a.universe == b.universe


def test_heuristic_is_deterministic(h):
    a = h.translate(CANONICAL).spec.model_dump()
    b = HeuristicScreenTranslator().translate(CANONICAL).spec.model_dump()
    assert a == b


def test_heuristic_canonical_unicode_dashes_and_case(h):
    obs = CANONICAL.replace("15-40%", "15–40%").replace("$2-20B", "$2—20B").upper()
    s = h.translate(obs).spec
    assert conds(s) == CANONICAL_CONDITIONS and ranking(s) == CANONICAL_RANKING
    assert s.observation == obs


# ================================================================================================
# HeuristicScreenTranslator - variants
# ================================================================================================

VARIANTS = [
    # market-cap bands
    ("$2-20B companies", {("market_cap_usd_bn", "between", 2.0, 20.0)}),
    ("$2bn to $20bn market cap", {("market_cap_usd_bn", "between", 2.0, 20.0)}),
    ("mid-cap stocks", {("market_cap_usd_bn", "between", 2.0, 10.0)}),
    ("small-cap names", {("market_cap_usd_bn", "between", 0.3, 2.0)}),
    ("large-cap stocks", {("market_cap_usd_bn", ">", 10.0)}),
    ("market cap above $10B", {("market_cap_usd_bn", ">", 10.0)}),
    ("$500M-$2B", {("market_cap_usd_bn", "between", 0.5, 2.0)}),
    ("market cap between $1bn and $5bn", {("market_cap_usd_bn", "between", 1.0, 5.0)}),
    ("small and mid caps", {("market_cap_usd_bn", "between", 0.3, 10.0)}),
    ("mega caps", {("market_cap_usd_bn", ">", 200.0)}),
    ("mid-caps ($3-15B)", {("market_cap_usd_bn", "between", 3.0, 15.0)}),
    # trend
    ("above the 200-day", {("price_vs_sma_200_pct", ">", 0.0)}),
    ("trading below their 50-day moving average", {("price_vs_sma_50_pct", "<", 0.0)}),
    ("50-day above 200-day", {("sma_50_vs_sma_200_pct", ">", 0.0)}),
    ("50dma > 200dma", {("sma_50_vs_sma_200_pct", ">", 0.0)}),
    ("golden cross", {("sma_50_vs_sma_200_pct", ">", 0.0)}),
    ("recent golden cross", {("golden_cross_20d", "==", 1.0)}),
    ("death cross", {("sma_50_vs_sma_200_pct", "<", 0.0)}),
    ("in an uptrend", {("sma_50_vs_sma_200_pct", ">", 0.0)}),
    ("trading more than 10% below the 50-day", {("price_vs_sma_50_pct", "<", -10.0)}),
    ("within 3% of the 200-day", {("price_vs_sma_200_pct", "between", -3.0, 3.0)}),
    ("above a rising 200-day", {("price_vs_sma_200_pct", ">", 0.0), ("sma_200_slope_1m_pct", ">", 0.0)}),
    ("above the 20-day", {("price", ">", "sma_20", 1.0)}),
    # momentum
    ("positive 12-1 momentum", {("return_12m_ex_1m_pct", ">", 0.0)}),
    ("negative 12-1 momentum", {("return_12m_ex_1m_pct", "<", 0.0)}),
    ("12-1 momentum above 20%", {("return_12m_ex_1m_pct", ">", 20.0)}),
    ("outperforming the market over 6 months", {("rel_strength_6m_pp", ">", 0.0)}),
    ("outperformed the S&P over the last 3 months", {("rel_strength_3m_pp", ">", 0.0)}),
    ("underperforming the market over the past year", {("rel_strength_12m_pp", "<", 0.0)}),
    ("up more than 20% over the last 3 months", {("return_3m_pct", ">", 20.0)}),
    # drawdown / range
    ("down 15-40% from highs", {("drawdown_from_52w_high_pct", "between", -40.0, -15.0)}),
    ("pulled back 20-30% from their 52-week highs", {("drawdown_from_52w_high_pct", "between", -30.0, -20.0)}),
    ("within 5% of highs", {("drawdown_from_52w_high_pct", ">=", -5.0)}),
    ("within 10% of their 52-week high", {("drawdown_from_52w_high_pct", ">=", -10.0)}),
    ("more than 25% below the 52-week high", {("drawdown_from_52w_high_pct", "<", -25.0)}),
    ("down 20% from their highs", {("drawdown_from_52w_high_pct", "<=", -20.0)}),
    ("a 15-40% pullback", {("drawdown_from_52w_high_pct", "between", -40.0, -15.0)}),
    ("near 52-week highs", {("drawdown_from_52w_high_pct", ">=", -5.0)}),
    ("near 52-week lows", {("above_52w_low_pct", "<=", 10.0)}),
    # oscillators
    ("RSI under 35", {("rsi_14", "<", 35.0)}),
    ("RSI(14) below 30", {("rsi_14", "<", 30.0)}),
    ("14-day RSI above 60", {("rsi_14", ">", 60.0)}),
    ("RSI between 30 and 50", {("rsi_14", "between", 30.0, 50.0)}),
    ("oversold", {("rsi_14", "<", 30.0)}),
    ("overbought", {("rsi_14", ">", 70.0)}),
    ("oversold (RSI under 40)", {("rsi_14", "<", 40.0)}),
    ("bullish MACD crossover", {("macd_bullish_cross_10d", "==", 1.0)}),
    # volume
    ("heavy volume", {("max_volume_ratio_20d", ">=", 2.0)}),
    ("volume surge", {("rel_volume_5d", ">=", 1.5)}),
    ("relative volume above 1.5", {("rel_volume_20d", ">", 1.5)}),
    ("3x average volume", {("max_volume_ratio_20d", ">=", 3.0)}),
    # fundamentals
    ("FCF yield above 5%", {("fcf_yield_pct", ">", 5.0)}),
    ("free cash flow yield of at least 6%", {("fcf_yield_pct", ">=", 6.0)}),
    ("8%+ FCF yield", {("fcf_yield_pct", ">=", 8.0)}),
    ("revenue growth above 12%", {("revenue_growth_yoy_pct", ">", 12.0)}),
    ("sales growth > 10%", {("revenue_growth_yoy_pct", ">", 10.0)}),
    ("double-digit revenue growth", {("revenue_growth_yoy_pct", ">=", 10.0)}),
    ("accelerating revenue growth", {("revenue_growth_last_q_yoy_pct", ">", "revenue_growth_yoy_pct", 1.0)}),
    ("gross margin above 50%", {("gross_margin_pct", ">", 50.0)}),
    ("operating margins above 20%", {("operating_margin_pct", ">", 20.0)}),
    ("gross margins expanding", {("gross_margin_change_yoy_pp", ">", 0.0)}),
    ("operating margin up 200bps", {("operating_margin_change_yoy_pp", ">=", 2.0)}),
    ("EV/EBITDA below 9", {("ev_to_ebitda", "<", 9.0)}),
    ("EV/EBITDA under 12x", {("ev_to_ebitda", "<", 12.0)}),
    ("trading below 10x EBITDA", {("ev_to_ebitda", "<", 10.0)}),
    ("net cash", {("net_debt_usd_bn", "<", 0.0)}),
    ("net debt to EBITDA below 2", {("net_debt_to_ebitda", "<", 2.0)}),
    ("net debt/EBITDA under 1.5x", {("net_debt_to_ebitda", "<", 1.5)}),
    ("net debt below 2x EBITDA", {("net_debt_to_ebitda", "<", 2.0)}),
    ("ROE above 15%", {("roe_pct", ">", 15.0)}),
    # positioning / estimates / options
    ("short interest above 8% of float", {("short_interest_pct_float", ">", 8.0)}),
    ("short interest > 12%", {("short_interest_pct_float", ">", 12.0)}),
    ("high short interest", {("short_interest_pct_float", ">", 10.0)}),
    ("days to cover above 4", {("days_to_cover", ">", 4.0)}),
    ("short interest rising", {("short_interest_change_1m_pct", ">", 0.0)}),
    ("estimates rising", {("eps_revision_3m_pct", ">", 0.0)}),
    ("estimates being cut", {("eps_revision_3m_pct", "<", 0.0)}),
    ("upward revisions", {("eps_revision_3m_pct", ">", 0.0)}),
    ("analysts cutting estimates", {("eps_revision_3m_pct", "<", 0.0)}),
    ("revenue estimates rising", {("revenue_revision_3m_pct", ">", 0.0)}),
    ("beat earnings", {("last_eps_surprise_pct", ">", 0.0)}),
    ("IV rank above 60", {("iv_rank_1y", ">", 60.0)}),
    ("implied volatility above realized", {("iv_to_realized_vol_ratio", ">", 1.0)}),
    ("put/call ratio above 1.5", {("put_call_volume_ratio", ">", 1.5)}),
    # sectors / events / literal names
    ("tech stocks", {("gics_sector", "in", ("Information Technology",))}),
    ("in Health Care", {("gics_sector", "in", ("Health Care",))}),
    ("in Energy or Materials", {("gics_sector", "in", ("Energy", "Materials"))}),
    ("mid-cap industrials", {("market_cap_usd_bn", "between", 2.0, 10.0), ("gics_sector", "in", ("Industrials",))}),
    ("reported earnings in the last 2 weeks", {("days_since_last_earnings", "<=", 14.0)}),
    ("rsi_14 <= 35 and fcf_yield_pct > 5", {("rsi_14", "<=", 35.0), ("fcf_yield_pct", ">", 5.0)}),
    ("golden_cross_20d == 1", {("golden_cross_20d", "==", 1.0)}),
]


@pytest.mark.parametrize("obs,expected", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_heuristic_variant_phrases(h, obs, expected):
    s = h.translate(obs).spec
    assert cond_set(s) == expected
    assert s.validate_against(CAT) == []
    assert s.unsupported_requests == []
    assert s.observation == obs


FULL_VARIANTS = [
    (
        "Small-cap tech stocks above their 200-day with RSI below 30, ranked by FCF yield",
        [("market_cap_usd_bn", "between", 0.3, 2.0), ("gics_sector", "in", None, None), ("price_vs_sma_200_pct", ">", 0.0, None),
         ("rsi_14", "<", 30.0, None)],
        [("fcf_yield_pct", "higher_is_better", 1.0)],
    ),
    (
        "Large caps with market cap above $10B, golden cross, outperforming the market over 6 months, within 5% of "
        "52-week highs, overbought. Sort by relative strength.",
        [("market_cap_usd_bn", ">", 10.0, None), ("sma_50_vs_sma_200_pct", ">", 0.0, None), ("rel_strength_6m_pp", ">", 0.0, None),
         ("drawdown_from_52w_high_pct", ">=", -5.0, None), ("rsi_14", ">", 70.0, None)],
        [("rel_strength_6m_pp", "higher_is_better", 1.0)],
    ),
    (
        "$2bn to $20bn companies down 20-30% from highs on a volume surge, EV/EBITDA below 8, net cash. Rank by lowest "
        "EV/EBITDA and biggest drawdown.",
        [("market_cap_usd_bn", "between", 2.0, 20.0), ("drawdown_from_52w_high_pct", "between", -30.0, -20.0),
         ("rel_volume_5d", ">=", 1.5, None), ("ev_to_ebitda", "<", 8.0, None), ("net_debt_usd_bn", "<", 0.0, None)],
        [("ev_to_ebitda", "lower_is_better", 1.0), ("drawdown_from_52w_high_pct", "lower_is_better", 1.0)],
    ),
    (
        "Health care names with gross margin above 60%, operating margin above 15%, revenue growth above 10%, estimates "
        "rising, IV rank above 50; rank by revenue growth (weight 2) and gross margin",
        [("gics_sector", "in", None, None), ("gross_margin_pct", ">", 60.0, None), ("operating_margin_pct", ">", 15.0, None),
         ("revenue_growth_yoy_pct", ">", 10.0, None), ("eps_revision_3m_pct", ">", 0.0, None), ("iv_rank_1y", ">", 50.0, None)],
        [("revenue_growth_yoy_pct", "higher_is_better", 2.0), ("gross_margin_pct", "higher_is_better", 1.0)],
    ),
    (
        "Mid-cap industrials below the 50-day, net debt to EBITDA below 2, short interest above 10% of float, days to "
        "cover above 5, estimates being cut. Rank by short interest and days to cover. Top 15.",
        [("market_cap_usd_bn", "between", 2.0, 10.0), ("gics_sector", "in", None, None), ("price_vs_sma_50_pct", "<", 0.0, None),
         ("net_debt_to_ebitda", "<", 2.0, None), ("short_interest_pct_float", ">", 10.0, None), ("days_to_cover", ">", 5.0, None),
         ("eps_revision_3m_pct", "<", 0.0, None)],
        [("short_interest_pct_float", "higher_is_better", 1.0), ("days_to_cover", "higher_is_better", 1.0)],
    ),
    (
        "Find small and mid caps in Energy or Materials with FCF yield above 8%, trading below 10x EBITDA, ordered by "
        "cheapest EV/EBITDA",
        [("market_cap_usd_bn", "between", 0.3, 10.0), ("gics_sector", "in", None, None), ("fcf_yield_pct", ">", 8.0, None),
         ("ev_to_ebitda", "<", 10.0, None)],
        [("ev_to_ebitda", "lower_is_better", 1.0)],
    ),
    (
        "U.S. large-caps that are oversold with estimates rising and heavy volume after a 25%+ pullback from the "
        "52-week high, prioritise by most oversold",
        [("market_cap_usd_bn", ">", 10.0, None), ("rsi_14", "<", 30.0, None), ("eps_revision_3m_pct", ">", 0.0, None),
         ("max_volume_ratio_20d", ">=", 2.0, None), ("drawdown_from_52w_high_pct", "<=", -25.0, None)],
        [("rsi_14", "lower_is_better", 1.0)],
    ),
    (
        "Mid-caps in a golden cross with positive 12-1 momentum, gross margin above 40% and IV rank above 70, rank by "
        "momentum then IV rank",
        [("market_cap_usd_bn", "between", 2.0, 10.0), ("sma_50_vs_sma_200_pct", ">", 0.0, None),
         ("return_12m_ex_1m_pct", ">", 0.0, None), ("gross_margin_pct", ">", 40.0, None), ("iv_rank_1y", ">", 70.0, None)],
        [("return_12m_ex_1m_pct", "higher_is_better", 1.0), ("iv_rank_1y", "higher_is_better", 1.0)],
    ),
    (
        "Companies with market cap above $10B within 10% of highs, above the 200-day, short interest above 5% of float "
        "and days to cover above 3",
        [("market_cap_usd_bn", ">", 10.0, None), ("drawdown_from_52w_high_pct", ">=", -10.0, None),
         ("price_vs_sma_200_pct", ">", 0.0, None), ("short_interest_pct_float", ">", 5.0, None), ("days_to_cover", ">", 3.0, None)],
        None,
    ),
]


@pytest.mark.parametrize("obs,expected,rank", FULL_VARIANTS, ids=[f"full{i}" for i in range(len(FULL_VARIANTS))])
def test_heuristic_full_observations(h, obs, expected, rank):
    s = h.translate(obs).spec
    got = [(c.feature, c.op, c.value, c.value_high) for c in s.conditions]
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g[:2] == e[:2] and (e[2] is None or g[2:] == e[2:]), (g, e)
    if rank is not None:
        assert ranking(s) == rank
    assert s.unsupported_requests == []
    assert s.validate_against(CAT) == []


def test_heuristic_top_n_parsed(h):
    assert h.translate("Mid-cap stocks above the 200-day. Rank by FCF yield. Top 15.").spec.top_n == 15
    assert h.translate("show me 25 names above the 200-day").spec.top_n == 25
    s = h.translate("oversold mid-caps, top 500")
    assert s.spec.top_n == 100 and any("clamped" in a for a in s.spec.assumptions)


def test_heuristic_sector_exclusion_goes_to_universe(h):
    s = h.translate("mid-caps excluding Financials and Utilities, ex-real estate").spec
    assert s.universe.exclude_sectors == ["Financials", "Utilities", "Real Estate"]
    assert all(c.feature != "gics_sector" for c in s.conditions)
    s2 = h.translate("non-financial large caps").spec
    assert s2.universe.exclude_sectors == ["Financials"]


def test_heuristic_sector_inclusions_merge(h):
    s = h.translate("tech stocks and healthcare names with RSI below 30").spec
    sectors = [c for c in s.conditions if c.feature == "gics_sector"]
    assert len(sectors) == 1 and sectors[0].op == "in"
    assert sectors[0].values == ["Information Technology", "Health Care"]


def test_heuristic_universe_floors(h):
    s = h.translate("Stocks above $10 with average daily dollar volume above $20mn and RSI under 30").spec
    assert s.universe.min_price == 10 and s.universe.min_avg_dollar_volume_usd_mn == 20
    s2 = h.translate("price above $10, no stocks under $3").spec
    assert s2.universe.min_price == 10  # strictest floor wins
    s3 = h.translate("ADV above $1bn, liquid names").spec
    assert s3.universe.min_avg_dollar_volume_usd_mn == 1000


def test_heuristic_any_of_from_or(h):
    s = h.translate("mid-caps with RSI under 30 or more than 30% below the 52-week high").spec
    assert [(c.feature, c.op, c.value) for c in s.conditions] == [("market_cap_usd_bn", "between", 2.0)]
    assert len(s.any_of) == 1
    assert {(c.feature, c.op, c.value) for c in s.any_of[0]} == {("rsi_14", "<", 30.0), ("drawdown_from_52w_high_pct", "<", -30.0)}
    assert s.validate_against(CAT) == []


def test_heuristic_unsupported_and_negation(h):
    s = h.translate("Stocks with dividend yield above 3% and insider buying, RSI under 30").spec
    assert cond_set(s) == {("rsi_14", "<", 30.0)}
    assert "dividend yield above 3%" in s.unsupported_requests and "insider buying" in s.unsupported_requests
    assert any("Not understood" in a for a in s.assumptions)
    n = h.translate("Names that are not oversold with a golden cross in the last 2 weeks").spec
    assert cond_set(n) == {("golden_cross_20d", "==", 1.0)}
    assert any("not oversold" in u for u in n.unsupported_requests)
    g = h.translate("European stocks with FCF yield above 5%").spec
    assert cond_set(g) == {("fcf_yield_pct", ">", 5.0)}
    assert any("non-US" in u for u in g.unsupported_requests) and g.universe.country == "US"


def test_heuristic_all_time_high_is_flagged(h):
    s = h.translate("30% below all-time highs").spec
    assert cond_set(s) == {("drawdown_from_52w_high_pct", "<=", -30.0)}
    assert any("all-time" in u for u in s.unsupported_requests)


def test_heuristic_unknown_rank_term(h):
    s = h.translate("oversold mid-caps, rank by insider ownership and FCF yield").spec
    assert ranking(s) == [("fcf_yield_pct", "higher_is_better", 1.0)]
    assert any("insider ownership" in u for u in s.unsupported_requests)


def test_heuristic_default_ranking_uses_catalog_direction(h):
    s = h.translate("mid-caps with FCF yield above 5%, EV/EBITDA under 10 and RSI below 40").spec
    assert ranking(s) == [("fcf_yield_pct", "higher_is_better", 1.0), ("ev_to_ebitda", "lower_is_better", 1.0)]
    assert any("No ranking stated" in a for a in s.assumptions)


def test_heuristic_default_ranking_falls_back_to_condition_direction(h):
    s = h.translate("RSI under 30, short interest above 10% of float").spec
    assert ranking(s) == [("rsi_14", "lower_is_better", 1.0), ("short_interest_pct_float", "higher_is_better", 1.0)]


def test_heuristic_default_ranking_liquidity_fallback(h):
    s = h.translate("mid-cap tech stocks").spec  # only a between and a category condition
    assert ranking(s) == [("avg_dollar_volume_20d_usd_mn", "higher_is_better", 1.0)]


def test_heuristic_nothing_understood_is_still_valid(h):
    s = h.translate("Companies with great management and a wide moat").spec
    assert s.conditions == [] and s.any_of == []
    assert s.validate_against(CAT) == []
    assert s.unsupported_requests and s.name == "screen"


@pytest.mark.parametrize("obs", ["", "   ", "\n"])
def test_heuristic_empty_raises(h, obs):
    with pytest.raises(ScreenTranslationError):
        h.translate(obs)


def test_heuristic_custom_catalog_drops_unknown_features():
    small = FeatureCatalog([CAT["market_cap_usd_bn"], CAT["rsi_14"], CAT["fcf_yield_pct"]])
    s = HeuristicScreenTranslator(small).translate(CANONICAL).spec
    assert {c.feature for c in s.conditions} == {"market_cap_usd_bn", "rsi_14", "fcf_yield_pct"}
    assert ranking(s) == [("fcf_yield_pct", "higher_is_better", 1.0)]
    assert s.validate_against(small) == []
    assert any("unknown feature" in u for u in s.unsupported_requests)
    assert any("rank by" in u for u in s.unsupported_requests)


def test_heuristic_custom_catalog_without_liquidity_feature():
    small = FeatureCatalog([CAT["gics_sector"], CAT["rsi_14"]])
    s = HeuristicScreenTranslator(small).translate("tech stocks").spec
    assert s.validate_against(small) == [] and ranking(s) == [("rsi_14", "higher_is_better", 1.0)]


_FRAGMENTS = [
    "mid-caps ($2-20B)", "50-day above 200-day", "positive 12-1 momentum", "pulled back 15-40% from their 52-week highs",
    "on heavy volume", "oversold (RSI under 40)", "FCF yield above 4%", "revenue growth above 8%",
    "short interest above 6% of float", "rank by FCF yield, growth and the size of the drawdown",
    "then read the latest earnings calls", "ex-financials", "in tech or health care", "not oversold", "golden cross",
    "within 5% of highs", "near 52-week lows", "IV rank above 50", "EV/EBITDA below 8x", "net cash", "days to cover > 5",
    "estimates being cut", "top 25", "$500M to $2bn", "market cap under $300M", "5% above the 20-day",
    "less than 5% below the 50-day", "RSI between 30 and 50", "or", "and", ",", "(", ")", "dividend yield above 3%",
    "beta below 1", "volatility under 30%", "relative volume above 1.5", "3x average volume", "earnings in the next 2 weeks",
    "at least 5 analysts", "gross margin up 200bps", "operating margins contracting", "rank by lowest P/E (weight 3)",
    "sort by gibberish", "European", "price above $10", "-15", "%%", "$$", "12-1", "200-day", "between", "above", "100%",
    "∞", "—", "rsi_14 >= 70", "golden_cross_20d == 1", "upside to target above 20%", "put/call oi ratio above 2",
]


def test_heuristic_output_always_valid():
    rng = random.Random(20261002)
    h = HeuristicScreenTranslator()
    for _ in range(400):
        obs = " ".join(rng.choice(_FRAGMENTS) for _ in range(rng.randint(1, 10)))
        res = h.translate(obs)
        assert res.spec.validate_against(CAT) == [], obs
        assert res.spec.observation == obs
        assert 1 <= res.spec.top_n <= 100 and res.spec.ranking


def test_heuristic_rationales_quote_the_observation(h):
    s = h.translate(CANONICAL).spec
    by_feature = {c.feature: c.rationale for c in s.conditions}
    assert "RSI under 40" in by_feature["rsi_14"]
    assert "$2-20B" in by_feature["market_cap_usd_bn"]
    assert "heavy volume" in by_feature["max_volume_ratio_20d"]


class _Dummy(BaseModel):
    x: int = 1


def test_scripted_llm_rejects_wrong_model_for_nl():
    """A responder returning an incompatible model is surfaced as a schema error round, then repaired."""
    calls = {"n": 0}

    def responder(purpose, system, user, model):
        calls["n"] += 1
        return _Dummy().model_dump() if calls["n"] == 1 else canonical_spec()

    res = NLScreenTranslator(ScriptedLLM({"nl_screen": responder})).translate(CANONICAL)
    assert res.attempts == 2 and res.errors_by_round[0] and res.errors_by_round[1] == []
