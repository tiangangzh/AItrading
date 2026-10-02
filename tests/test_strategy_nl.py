"""Idea library + idea -> StrategySpec translation (heuristic and Claude via ScriptedLLM)."""

from __future__ import annotations

from datetime import date

import pytest

from aitrading.llm.base import LLMError, ScriptedLLM
from aitrading.screen.catalog import default_catalog
from aitrading.screen.spec import Condition
from aitrading.strategy.library import TEMPLATES, get_template, suggest_templates, templates_prompt
from aitrading.strategy.nl import (
    HeuristicStrategyTranslator,
    StrategyTranslation,
    StrategyTranslationError,
    StrategyTranslator,
    build_system_prompt,
    spec_errors,
    translate_idea,
)
from aitrading.strategy.spec import StrategySpec

CAT = default_catalog()
TODAY = date(2026, 10, 2)


@pytest.fixture
def h() -> HeuristicStrategyTranslator:
    return HeuristicStrategyTranslator(today=TODAY)


# ------------------------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------------------------

REQUIRED_KEYS = [
    "capm", "ff3", "carhart4", "ff5", "momentum_12_1", "short_term_reversal", "low_volatility", "low_beta", "value_fcf",
    "value_composite", "quality", "qarp", "size", "estimate_revisions", "short_interest", "rsi_reversal",
    "trend_200dma_spy", "golden_cross_spy", "dislocation_screen",
    "value_book_to_market", "profitability", "investment", "gross_profitability",
]


def test_library_has_required_templates():
    assert set(REQUIRED_KEYS) <= set(TEMPLATES)


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_every_template_validates(key):
    t = TEMPLATES[key]
    spec = t.spec()
    assert isinstance(spec, StrategySpec)
    assert spec.validate_against(CAT) == []
    assert spec_errors(spec, CAT) == []
    assert spec.name == key == t.key
    assert t.title and t.aliases and t.references and t.description
    assert spec.unsupported_requests == []


def test_template_builds_fresh_copies():
    a = get_template("momentum_12_1").spec()
    a.signal.clear()
    a.assumptions.append("mutated")
    b = get_template("momentum_12_1").spec()
    assert len(b.signal) == 1 and "mutated" not in b.assumptions


@pytest.mark.parametrize("key,model", [("capm", "capm"), ("ff3", "ff3"), ("carhart4", "carhart4"), ("ff5", "ff5")])
def test_factor_model_templates(key, model):
    s = TEMPLATES[key].spec()
    assert s.kind == "factor_model" and s.factor_model == model and s.attribution_model == model
    assert s.signal == [] and s.filters == []


def test_ff3_title():
    assert TEMPLATES["ff3"].title == "Fama-French 3-factor model"


def _sig(key):
    return [(c.feature, c.direction) for c in TEMPLATES[key].spec().signal]


def test_cross_sectional_template_definitions():
    m = TEMPLATES["momentum_12_1"].spec()
    assert _sig("momentum_12_1") == [("return_12m_ex_1m_pct", "higher_is_better")]
    assert (m.portfolio.n_quantiles, m.portfolio.style, m.rebalance) == (10, "long_short", "monthly")
    assert any("Jegadeesh" in r and "1993" in r for r in TEMPLATES["momentum_12_1"].references)

    assert _sig("short_term_reversal") == [("return_1m_pct", "lower_is_better")]
    assert TEMPLATES["short_term_reversal"].spec().rebalance == "monthly"

    lv = TEMPLATES["low_volatility"].spec()
    assert _sig("low_volatility") == [("volatility_60d_pct", "lower_is_better")]
    assert (lv.portfolio.style, lv.portfolio.n_quantiles, lv.portfolio.selection) == ("long_only", 5, "quantile")
    refs = " ".join(TEMPLATES["low_volatility"].references)
    assert "Ang" in refs and "2006" in refs and "Baker" in refs

    assert _sig("low_beta") == [("beta_1y", "lower_is_better")]
    assert any("Frazzini" in r and "2014" in r for r in TEMPLATES["low_beta"].references)
    assert _sig("value_fcf") == [("fcf_yield_pct", "higher_is_better")]
    assert _sig("value_composite") == [
        ("book_to_market", "higher_is_better"), ("fcf_yield_pct", "higher_is_better"),
        ("earnings_yield_ntm_pct", "higher_is_better"), ("ev_to_ebitda", "lower_is_better"),
    ]
    assert _sig("quality") == [
        ("roe_pct", "higher_is_better"), ("gross_margin_pct", "higher_is_better"), ("fcf_conversion_pct", "higher_is_better"),
        ("net_debt_to_ebitda", "lower_is_better"),
    ]
    refs = " ".join(TEMPLATES["quality"].references)
    assert "Asness" in refs and "Novy-Marx" in refs
    assert set(_sig("qarp")) == set(_sig("quality")) | set(_sig("value_composite"))
    q = TEMPLATES["qarp"].spec()
    quality_w = sum(c.weight for c in q.signal if c.feature in {"roe_pct", "gross_margin_pct", "fcf_conversion_pct", "net_debt_to_ebitda"})
    assert quality_w == pytest.approx(sum(c.weight for c in q.signal) / 2)  # 50 / 50 quality + value
    assert _sig("size") == [("market_cap_usd_bn", "lower_is_better")]
    assert _sig("estimate_revisions") == [("eps_revision_3m_pct", "higher_is_better")]
    assert _sig("short_interest") == [("short_interest_pct_float", "lower_is_better")]
    assert _sig("rsi_reversal") == [("rsi_14", "lower_is_better")]
    assert TEMPLATES["rsi_reversal"].spec().rebalance == "weekly"


def test_factor_characteristic_templates():
    assert _sig("value_book_to_market") == [("book_to_market", "higher_is_better")]
    assert _sig("profitability") == [("operating_profitability_pct", "higher_is_better")]
    assert _sig("investment") == [("asset_growth_yoy_pct", "lower_is_better")]  # conservative minus aggressive
    assert _sig("gross_profitability") == [("gross_profitability_pct", "higher_is_better")]
    for key in ("value_book_to_market", "profitability", "investment", "gross_profitability"):
        s = TEMPLATES[key].spec()
        assert s.kind == "cross_sectional" and s.rebalance == "monthly"
        assert (s.portfolio.style, s.portfolio.selection, s.portfolio.n_quantiles, s.portfolio.weighting) == (
            "long_short", "quantile", 5, "equal")
        assert s.attribution_model == "ff3"
        assert not TEMPLATES[key].needs_institutional_data  # filings-based, point-in-time in the free edition
    refs = lambda k: " ".join(TEMPLATES[k].references)  # noqa: E731
    assert "Fama & French (1992)" in refs("value_book_to_market")
    assert "Fama & French (2015)" in refs("profitability") and "Cooper, Gulen & Schill (2008)" in refs("investment")
    assert "Novy-Marx (2013)" in refs("gross_profitability")
    assert "HML" in TEMPLATES["value_book_to_market"].title and "RMW" in TEMPLATES["profitability"].title
    assert "CMA" in TEMPLATES["investment"].title and "Novy-Marx" in TEMPLATES["gross_profitability"].title


def test_time_series_templates():
    t = TEMPLATES["trend_200dma_spy"].spec()
    assert t.kind == "time_series" and t.time_series.assets == ["SPY"] and t.time_series.when_flat == "cash"
    assert [(c.feature, c.op, c.value) for c in t.time_series.entry] == [("price_vs_sma_200_pct", ">", 0.0)]
    assert any("Faber" in r and "2007" in r for r in TEMPLATES["trend_200dma_spy"].references)
    g = TEMPLATES["golden_cross_spy"].spec()
    assert [(c.feature, c.op, c.value) for c in g.time_series.entry] == [("sma_50_vs_sma_200_pct", ">", 0.0)]


def test_dislocation_screen_has_canonical_conditions():
    s = TEMPLATES["dislocation_screen"].spec()
    assert s.kind == "screen" and s.portfolio.weighting == "equal" and s.rebalance == "monthly"
    got = [(c.feature, c.op, c.value, c.value_high) for c in s.filters]
    assert got == [
        ("market_cap_usd_bn", "between", 2, 20),
        ("sma_50_vs_sma_200_pct", ">", 0, None),
        ("return_12m_ex_1m_pct", ">", 0, None),
        ("drawdown_from_52w_high_pct", "between", -40, -15),
        ("max_volume_ratio_20d", ">=", 2, None),
        ("rsi_14", "<", 40, None),
        ("fcf_yield_pct", ">", 4, None),
        ("revenue_growth_yoy_pct", ">", 8, None),
        ("short_interest_pct_float", ">", 6, None),
    ]


@pytest.mark.parametrize("key", ["short_interest", "estimate_revisions", "dislocation_screen", "value_composite", "qarp"])
def test_weak_free_data_is_flagged(key):
    t = TEMPLATES[key]
    assert t.needs_institutional_data
    assert "no point-in-time history in the free edition" in t.description
    assert "institutional data" in t.description


def test_price_only_templates_not_flagged():
    for key in ["momentum_12_1", "low_volatility", "trend_200dma_spy", "ff3", "value_book_to_market", "profitability",
                "investment", "gross_profitability"]:
        assert not TEMPLATES[key].needs_institutional_data


def test_templates_prompt_is_stable_and_complete():
    p1, p2 = templates_prompt(), templates_prompt()
    assert p1 == p2
    for t in TEMPLATES.values():
        assert f"### {t.key} - {t.title}" in p1
    assert '"feature":"return_12m_ex_1m_pct"' in p1


def test_suggest_templates_deterministic():
    s = suggest_templates("bananas and rockets")
    assert len(s) == 5 and s == suggest_templates("bananas and rockets")
    assert set(s) <= {t.title for t in TEMPLATES.values()}
    assert suggest_templates("fama french three factor model", 1) == ["Fama-French 3-factor model"]


# ------------------------------------------------------------------------------------------------
# Heuristic translator: template matching
# ------------------------------------------------------------------------------------------------

PHRASINGS = [
    ("3 factor model", "ff3"),
    ("Fama French three factor", "ff3"),
    ("Fama-French 3-factor model", "ff3"),
    ("three-factor model", "ff3"),
    ("FF3", "ff3"),
    ("Fama-French five factor model", "ff5"),
    ("5 factor model with RMW and CMA", "ff5"),
    ("Carhart four factor model", "carhart4"),
    ("4-factor model", "carhart4"),
    ("CAPM", "capm"),
    ("capital asset pricing model", "capm"),
    ("12-1 momentum top decile", "momentum_12_1"),
    ("momentum deciles since 2015 long only", "momentum_12_1"),
    ("buy last year's winners", "momentum_12_1"),
    ("short-term reversal", "short_term_reversal"),
    ("buy last month's losers, 1-month reversal", "short_term_reversal"),
    ("low vol stocks", "low_volatility"),
    ("minimum variance stocks", "low_volatility"),
    ("betting against beta", "low_beta"),
    ("low beta stocks", "low_beta"),
    ("free cash flow yield value", "value_fcf"),
    ("cheap stocks", "value_fcf"),
    ("value composite", "value_composite"),
    ("cheap on EV/EBITDA", "value_composite"),
    ("quality minus junk", "quality"),
    ("high quality companies", "quality"),
    ("quality at a reasonable price", "qarp"),
    ("quality and value", "qarp"),
    ("small cap premium", "size"),
    ("size factor", "size"),
    ("analyst earnings revisions", "estimate_revisions"),
    ("avoid heavily shorted stocks", "short_interest"),
    ("short interest", "short_interest"),
    ("RSI oversold reversal weekly", "rsi_reversal"),
    ("buy SPY when above 200 day", "trend_200dma_spy"),
    ("long SPY above its 200-day", "trend_200dma_spy"),
    ("market timing with the 200 day moving average", "trend_200dma_spy"),
    ("Faber 10-month SMA timing", "trend_200dma_spy"),
    ("trend following on the S&P 500", "trend_200dma_spy"),
    ("golden cross on SPY", "golden_cross_spy"),
    ("SPY when the 50-day is above the 200-day", "golden_cross_spy"),
    ("dislocation screen", "dislocation_screen"),
    ("value weighted momentum", "momentum_12_1"),
    ("book to market", "value_book_to_market"),
    ("HML", "value_book_to_market"),
    ("HML value factor, decile spread", "value_book_to_market"),
    ("buy low price-to-book stocks", "value_book_to_market"),
    ("profitability", "profitability"),
    ("operating profitability long short", "profitability"),
    ("robust minus weak", "profitability"),
    ("asset growth", "investment"),
    ("low asset growth stocks beat high asset growth", "investment"),
    ("conservative minus aggressive investment factor", "investment"),
    ("Novy-Marx", "gross_profitability"),
    ("Novy-Marx gross profitability premium", "gross_profitability"),
    ("gross profits to assets", "gross_profitability"),
]


@pytest.mark.parametrize("idea,key", PHRASINGS)
def test_heuristic_maps_phrasings(h, idea, key):
    tr = h.translate(idea)
    assert isinstance(tr, StrategyTranslation)
    assert tr.template == key and tr.spec.name == key
    assert tr.spec.idea == idea
    assert tr.translator == "heuristic" and tr.attempts == 1 and tr.errors_by_round == [[]]
    assert spec_errors(tr.spec, CAT) == []
    assert any(f"'{key}' template" in a for a in tr.spec.assumptions)
    assert h.match(idea)[0] == key


def test_heuristic_covers_at_least_25_phrasings():
    assert len(PHRASINGS) >= 25
    assert len({k for _, k in PHRASINGS}) == len(REQUIRED_KEYS)


def test_heuristic_is_deterministic(h):
    a = h.translate("momentum deciles since 2015 long only").spec.model_dump_json()
    b = HeuristicStrategyTranslator(today=TODAY).translate("momentum deciles since 2015 long only").spec.model_dump_json()
    assert a == b


# ------------------------------------------------------------------------------------------------
# Heuristic translator: parameter extraction
# ------------------------------------------------------------------------------------------------


def test_momentum_deciles_since_2015_long_only(h):
    s = h.translate("momentum deciles since 2015 long only").spec
    assert s.signal[0].feature == "return_12m_ex_1m_pct"
    assert s.portfolio.n_quantiles == 10 and s.portfolio.selection == "quantile"
    assert s.portfolio.style == "long_only"
    assert s.start == date(2015, 1, 1) and s.end is None
    assert s.rebalance == "monthly" and s.costs_bps == 10.0 and s.attribution_model == "ff3"
    joined = " | ".join(s.assumptions)
    assert "'deciles' -> n_quantiles=10" in joined
    assert "'long only' -> style=long_only" in joined
    assert "'since 2015' -> start=2015-01-01" in joined
    assert "Defaults kept: rebalance monthly" in joined


def test_buy_spy_above_200_day(h):
    s = h.translate("buy SPY when above 200 day").spec
    assert s.kind == "time_series"
    assert s.time_series.assets == ["SPY"] and s.benchmark == "SPY"
    assert [(c.feature, c.op, c.value) for c in s.time_series.entry] == [("price_vs_sma_200_pct", ">", 0.0)]
    assert s.time_series.when_flat == "cash"
    assert s.attribution_model == "capm"


@pytest.mark.parametrize(
    "idea,n",
    [("momentum terciles", 3), ("momentum quintiles", 5), ("momentum quartiles", 4), ("momentum deciles", 10),
     ("momentum, top 10%", 10), ("momentum, top 20%", 5), ("momentum with 8 buckets", 8), ("momentum top half", 2)],
)
def test_quantile_extraction(h, idea, n):
    s = h.translate(idea).spec
    assert s.portfolio.n_quantiles == n and s.portfolio.selection == "quantile"


def test_top_n_extraction(h):
    s = h.translate("top 50 momentum stocks equal weight").spec
    assert s.portfolio.selection == "top_n" and s.portfolio.top_n == 50
    assert s.portfolio.weighting == "equal"
    s = h.translate("hold 30 stocks with the best quality").spec
    assert s.portfolio.selection == "top_n" and s.portfolio.top_n == 30


@pytest.mark.parametrize(
    "idea,freq",
    [("momentum rebalanced weekly", "weekly"), ("monthly momentum", "monthly"), ("quarterly value", "value_q"),
     ("value rebalanced annually", "annual"), ("momentum rebalanced every 3 months", "quarterly"), ("daily low vol", "daily")],
)
def test_rebalance_extraction(h, idea, freq):
    s = h.translate(idea).spec
    assert s.rebalance == ("quarterly" if freq == "value_q" else freq)


def test_rsi_template_default_weekly_and_override(h):
    assert h.translate("RSI oversold").spec.rebalance == "weekly"
    assert h.translate("RSI oversold, rebalanced monthly").spec.rebalance == "monthly"


@pytest.mark.parametrize(
    "idea,start,end",
    [
        ("momentum from 2012 to 2020", date(2012, 1, 1), date(2020, 12, 31)),
        ("momentum 2012-2020", date(2012, 1, 1), date(2020, 12, 31)),
        ("momentum between 2010 and 2015", date(2010, 1, 1), date(2015, 12, 31)),
        ("momentum since 2019-06", date(2019, 6, 1), None),
        ("momentum until 2018", None, date(2018, 12, 31)),
        ("momentum before 2020", None, date(2019, 12, 31)),
        ("momentum after 2010", date(2011, 1, 1), None),
        ("low vol over the last 5 years", date(2021, 10, 2), None),
        ("momentum until 2030", None, None),  # future end -> latest available
    ],
)
def test_date_extraction(h, idea, start, end):
    s = h.translate(idea).spec
    assert (s.start, s.end) == (start, end)


def test_last_n_years_handles_leap_day():
    s = HeuristicStrategyTranslator(today=date(2028, 2, 29)).translate("momentum over the past 1 year").spec
    assert s.start == date(2027, 2, 28)


@pytest.mark.parametrize(
    "idea,bps",
    [("momentum with 25 bps costs", 25.0), ("momentum, 5bp", 5.0), ("momentum with no transaction costs", 0.0),
     ("momentum, frictionless", 0.0), ("momentum, 0.2% round-trip costs", 10.0), ("momentum with 0.15% transaction costs", 15.0)],
)
def test_costs_extraction(h, idea, bps):
    assert h.translate(idea).spec.costs_bps == pytest.approx(bps)


@pytest.mark.parametrize(
    "idea,bps,src",
    [
        ("momentum, costs of 0.2%", 20.0, "'costs of 0.2%' -> costs_bps=20"),
        ("momentum, 0.2% per trade", 20.0, "'0.2% per trade' -> costs_bps=20"),
        ("momentum with transaction costs of 0.15 percent", 15.0, "costs_bps=15"),
        ("momentum, commission: 0.05%", 5.0, "costs_bps=5"),
        ("momentum, 0.1% each way", 10.0, "costs_bps=10"),
        ("momentum with costs of 0.4% per round trip", 20.0, "round-trip figure -> 20 bps one-way"),
        ("momentum with transaction costs of 25 bps", 25.0, "costs_bps=25"),
        ("top 10% momentum with 0.15% transaction costs", 15.0, "costs_bps=15"),
    ],
)
def test_costs_written_after_the_cost_word(h, idea, bps, src):
    """Regression: 'costs of X%' and 'X% per trade' used to be ignored, keeping 10 bps silently."""
    s = h.translate(idea).spec
    assert s.costs_bps == pytest.approx(bps)
    joined = " | ".join(s.assumptions)
    assert src in joined
    assert "costs 10 bps one-way" not in joined  # the 'Defaults kept' note must not claim the default
    assert s.unsupported_requests == []


@pytest.mark.parametrize(
    "idea,needle",
    [
        ("momentum with commissions of 2 cents per share", "transaction cost 'commissions of 2 cents per share'"),
        ("momentum, $5 commission per trade", "transaction cost '$5 commission'"),
        ("momentum, costs of 1% a year", "'costs of 1%' a year (an annual cost figure"),
        ("momentum, 1% management fee", "'management fee' (management / performance fees are not modelled"),
    ],
)
def test_unparsed_cost_amounts_are_reported(h, idea, needle):
    s = h.translate(idea).spec
    assert s.costs_bps == 10.0
    assert any(needle in u and ("kept 10 bps" in u or "not modelled" in u) for u in s.unsupported_requests), s.unsupported_requests


@pytest.mark.parametrize("idea", ["momentum, max 5% per side", "momentum 2015-2020 net of fees", "momentum, top 10%"])
def test_non_cost_numbers_are_not_costs(h, idea):
    s = h.translate(idea).spec
    assert s.costs_bps == 10.0
    assert not any("cost" in u or "fee" in u for u in s.unsupported_requests)


@pytest.mark.parametrize(
    "idea,w",
    [("value weighted momentum", "value"), ("equal-weighted momentum", "equal"), ("cap weighted quality", "value"),
     ("inverse volatility weighted momentum", "inverse_vol")],
)
def test_weighting_extraction(h, idea, w):
    assert h.translate(idea).spec.portfolio.weighting == w


def test_long_short_cues(h):
    s = h.translate("buy the top decile of momentum and short the bottom decile").spec
    assert s.portfolio.style == "long_short" and s.portfolio.n_quantiles == 10
    s = h.translate("low volatility long-short").spec
    assert s.portfolio.style == "long_short"  # template default (long-only) overridden
    s = h.translate("12-1 momentum top decile").spec
    assert s.portfolio.style == "long_short"
    assert any("top-quantile leg is reported on its own" in a for a in s.assumptions)


@pytest.mark.parametrize(
    "idea,key",
    [
        ("buy last month's losers and sell last month's winners", "short_term_reversal"),
        ("Buy last month's losers and sell last month's winners, rebalanced monthly.", "short_term_reversal"),  # library text
        ("buy the cheapest quintile on FCF yield and sell the most expensive", "value_fcf"),
        ("buy low beta and sell high beta stocks", "low_beta"),
        ("buy the top decile of momentum, selling the bottom decile", "momentum_12_1"),
        ("buy quality, short junk", "quality"),
    ],
)
def test_sell_leg_means_long_short(h, idea, key):
    """Regression: 'buy X and sell Y' used to become long-only because only 'short ...' was a long-short cue."""
    tr = h.translate(idea)
    assert tr.template == key
    assert tr.spec.portfolio.style == "long_short"
    assert not any("style=long_only" in a for a in tr.spec.assumptions)


def test_library_reversal_description_round_trips(h):
    desc = TEMPLATES["short_term_reversal"].description
    assert "sell last month's winners" in desc.lower()
    s = h.translate(desc).spec
    assert s.name == "short_term_reversal" and s.portfolio.style == "long_short"


@pytest.mark.parametrize(
    "idea",
    ["buy the winners, sell them after 12 months", "momentum, buy the top decile and sell when it drops out",
     "buy last year's winners"],
)
def test_sell_without_a_short_leg_stays_long_only(h, idea):
    assert h.translate(idea).spec.portfolio.style == "long_only"


def test_reversal_winners_is_not_a_second_momentum_idea(h):
    s = h.translate("buy last month's losers and sell last month's winners").spec
    assert [c.feature for c in s.signal] == ["return_1m_pct"]
    assert any("'winners' is read as part of the 'short_term_reversal' idea" in a and "not tested as a separate 'momentum_12_1'" in a
               for a in s.assumptions)


@pytest.mark.parametrize(
    "idea,name,features",
    [
        ("12-1 momentum combined with 1-month reversal", "short_term_reversal_plus_momentum_12_1",
         ["return_1m_pct", "return_12m_ex_1m_pct"]),
        ("price momentum minus 1 month reversal", "short_term_reversal_plus_momentum_12_1", ["return_1m_pct", "return_12m_ex_1m_pct"]),
        ("earnings revisions plus 12-1 price momentum", "estimate_revisions_plus_momentum_12_1",
         ["eps_revision_3m_pct", "return_12m_ex_1m_pct"]),
        ("earnings momentum and price momentum", "estimate_revisions_plus_momentum_12_1", ["eps_revision_3m_pct", "return_12m_ex_1m_pct"]),
        ("low beta and low volatility composite", "low_beta_plus_low_volatility", ["beta_1y", "volatility_60d_pct"]),
        ("RSI oversold combined with 12-1 momentum", "rsi_reversal_plus_momentum_12_1", ["rsi_14", "return_12m_ex_1m_pct"]),
        ("heavily shorted stocks, 1-month reversal", "short_interest_plus_short_term_reversal",
         ["short_interest_pct_float", "return_1m_pct"]),
    ],
)
def test_genuine_second_idea_is_combined_not_dropped(h, idea, name, features):
    """Regression: _SUBSUMES used to drop an explicitly requested second idea without a trace."""
    tr = h.translate(idea)
    s = tr.spec
    assert tr.template is None and s.name == name
    assert [c.feature for c in s.signal] == features
    # equal total weight per idea
    assert [c.weight for c in s.signal] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert any("Also matched" in a for a in s.assumptions)
    assert spec_errors(s, CAT) == []


@pytest.mark.parametrize(
    "idea,key,absorbed",
    [
        ("earnings momentum", "estimate_revisions", None),               # same words: no note needed
        ("estimate momentum", "estimate_revisions", None),
        ("trend following on the S&P 500", "trend_200dma_spy", None),
        ("quality and value", "qarp", None),
        ("RSI oversold reversal weekly", "rsi_reversal", "'reversal' is read as part of the 'rsi_reversal' idea"),
        ("cheap on EV/EBITDA", "value_composite", "'cheap' is read as part of the 'value_composite' idea"),
        ("betting against beta, low risk", "low_beta", "'low risk' is read as part of the 'low_beta' idea"),
        ("short squeeze reversal", "short_interest", "'reversal' is read as part of the 'short_interest' idea"),
    ],
)
def test_descriptive_matches_are_absorbed_visibly(h, idea, key, absorbed):
    tr = h.translate(idea)
    assert tr.template == key and tr.spec.name == key
    notes = [a for a in tr.spec.assumptions if "is read as part of" in a]
    if absorbed is None:
        assert notes == []
    else:
        assert len(notes) == 1 and notes[0].startswith(absorbed)
    assert tr.spec.unsupported_requests == []


@pytest.mark.parametrize("idea", ["SMB and HML", "Fama-French 3 factor model with SMB and HML"])
def test_hml_inside_the_three_factor_model_is_not_a_second_idea(h, idea):
    s = h.translate(idea).spec
    assert s.name == "ff3" and s.kind == "factor_model"
    assert not any("is read as part of" in a for a in s.assumptions) and s.unsupported_requests == []


def test_book_to_market_is_a_catalog_signal_not_unsupported(h):
    for idea in ["book to market", "low price to book", "B/M deciles", "Fama-French 5 factor model"]:
        assert not any("book" in u for u in h.translate(idea).spec.unsupported_requests), idea
    s = h.translate("cheap stocks on book to market").spec
    assert [c.feature for c in s.signal] == ["book_to_market"]
    s = h.translate("book to market and FCF yield").spec  # two value ideas -> composite
    assert s.name == "value_book_to_market_plus_value_fcf" and [c.feature for c in s.signal] == ["book_to_market", "fcf_yield_pct"]


def test_profitability_flavours(h):
    s = h.translate("Novy-Marx gross profitability").spec
    assert [c.feature for c in s.signal] == ["gross_profitability_pct"]
    assert not any("is read as part of" in a for a in s.assumptions)  # 'profitability' belongs to the same phrase
    s = h.translate("quality minus junk").spec
    assert s.name == "quality"
    s = h.translate("gross profitability combined with operating profitability").spec
    assert s.name == "gross_profitability_plus_profitability"
    assert [c.feature for c in s.signal] == ["gross_profitability_pct", "operating_profitability_pct"]
    s = h.translate("5 factor model with RMW and CMA").spec
    assert s.name == "ff5" and s.unsupported_requests == []


@pytest.mark.parametrize(
    "idea,feature",
    [("low P/E stocks", "earnings_yield_ttm_pct"), ("value on price to earnings", "earnings_yield_ttm_pct"),
     ("low forward P/E", "earnings_yield_ntm_pct")],
)
def test_pe_value_uses_trailing_earnings_unless_forward(h, idea, feature):
    s = h.translate(idea).spec
    assert s.name == "value_fcf" and [c.feature for c in s.signal] == [feature]


def test_factor_model_family_needs_no_absorption_note(h):
    s = h.translate("Fama-French five factor model").spec
    assert s.name == "ff5" and not any("is read as part of" in a for a in s.assumptions)
    # but an extra cross-sectional idea next to a factor model is reported, not dropped
    s = h.translate("Fama-French 5 factor plus momentum").spec
    assert s.name == "ff5" and any("12-1 cross-sectional momentum" in u for u in s.unsupported_requests)


def test_attribution_phrase_sets_model_not_template(h):
    tr = h.translate("momentum deciles, alpha against the Fama-French 5-factor model")
    assert tr.template == "momentum_12_1" and tr.spec.attribution_model == "ff5"
    tr = h.translate("regress low beta on the CAPM")
    assert tr.template == "low_beta" and tr.spec.attribution_model == "capm"
    tr = h.translate("betting against beta, alpha vs 3 factor")
    assert tr.template == "low_beta" and tr.spec.attribution_model == "ff3"


def test_lookback_variants(h):
    assert h.translate("6 month momentum").spec.signal[0].feature == "return_6m_pct"
    assert h.translate("3-month return momentum").spec.signal[0].feature == "return_3m_pct"
    assert h.translate("12 month momentum").spec.signal[0].feature == "return_12m_ex_1m_pct"
    s = h.translate("9 month momentum").spec
    assert s.signal[0].feature == "return_12m_ex_1m_pct"
    assert any("9-month momentum lookback" in u for u in s.unsupported_requests)
    assert h.translate("momentum rebalanced every 3 months").spec.signal[0].feature == "return_12m_ex_1m_pct"
    assert h.translate("20-day low volatility").spec.signal[0].feature == "volatility_20d_pct"


def test_sector_neutral_and_exclusions(h):
    s = h.translate("quality ex-financials and utilities, sector neutral").spec
    assert all(c.sector_neutral for c in s.signal)
    assert s.universe.exclude_sectors == ["Financials", "Utilities"]


def test_cap_bucket_filter(h):
    s = h.translate("6 month momentum in small caps").spec
    assert [(c.feature, c.op, c.value) for c in s.filters] == [("market_cap_usd_bn", "<", 2.0)]
    s = h.translate("quality in mid caps").spec
    assert [(c.feature, c.op, c.value, c.value_high) for c in s.filters] == [("market_cap_usd_bn", "between", 2.0, 10.0)]


def test_stock_moving_average_becomes_filter(h):
    tr = h.translate("momentum stocks above their 200-day moving average")
    assert tr.spec.kind == "cross_sectional" and tr.template == "momentum_12_1"
    assert [(c.feature, c.op, c.value) for c in tr.spec.filters] == [("price_vs_sma_200_pct", ">", 0.0)]
    tr = h.translate("stocks above their 200 day")
    assert tr.spec.kind == "screen" and tr.template is None
    assert [(c.feature, c.op) for c in tr.spec.filters] == [("price_vs_sma_200_pct", ">")]


def test_timing_variants(h):
    s = h.translate("long SPY above its 200-day, otherwise short").spec
    assert s.time_series.when_flat == "short"
    s = h.translate("Long QQQ above its 50-day").spec
    assert s.time_series.assets == ["QQQ"] and s.benchmark == "QQQ"
    assert [(c.feature, c.op, c.value) for c in s.time_series.entry] == [("price_vs_sma_50_pct", ">", 0.0)]
    s = h.translate("SPY and QQQ above the 200 day").spec
    assert s.time_series.assets == ["SPY", "QQQ"] and s.benchmark is None
    s = h.translate("nasdaq above its 200-day moving average").spec
    assert s.time_series.assets == ["QQQ"]
    s = h.translate("SPY above its 100-day moving average").spec
    assert s.time_series.entry[0].feature == "price_vs_sma_200_pct"
    assert any("100-day moving average" in u for u in s.unsupported_requests)
    s = h.translate("Faber 10-month SMA timing").spec
    assert any("10-month" in u for u in s.unsupported_requests)


def test_combination_of_two_cross_sectional_ideas(h):
    tr = h.translate("momentum and low volatility")
    s = tr.spec
    assert tr.template is None and s.name == "low_volatility_plus_momentum_12_1"
    w = {c.feature: c.weight for c in s.signal}
    assert w == {"volatility_60d_pct": pytest.approx(1.0), "return_12m_ex_1m_pct": pytest.approx(1.0)}
    assert s.portfolio.style == "long_short" and s.portfolio.n_quantiles == 5  # different templates -> platform default
    assert spec_errors(s, CAT) == []


def test_unsupported_parts_are_reported(h):
    s = h.translate("momentum with a 10% stop loss and 2x leverage").spec
    assert any("stop-loss" in u for u in s.unsupported_requests)
    assert any("leverage" in u for u in s.unsupported_requests)
    assert h.translate("high quality low leverage stocks").spec.unsupported_requests == []
    s = h.translate("momentum in the Russell 2000 since 2015").spec
    assert any("russell 2000" in u for u in s.unsupported_requests)
    s = h.translate("momentum combined with the 3 factor model").spec
    assert s.kind in ("cross_sectional", "factor_model")


def test_unknown_idea_raises_with_suggestions(h):
    with pytest.raises(StrategyTranslationError) as ei:
        h.translate("a strategy that trades on the phase of the moon")
    e = ei.value
    assert isinstance(e, ValueError)
    assert e.errors and len(e.suggestions) == 5
    assert set(e.suggestions) <= {t.title for t in TEMPLATES.values()}
    assert e.suggestions == suggest_templates("a strategy that trades on the phase of the moon", 5)
    assert h.match("a strategy that trades on the phase of the moon") is None


def test_empty_idea_raises(h):
    with pytest.raises(StrategyTranslationError):
        h.translate("   ")


# ------------------------------------------------------------------------------------------------
# Claude translator (ScriptedLLM)
# ------------------------------------------------------------------------------------------------


def _respond_with(*specs):
    """Responder that returns the given specs in order (the last one repeats)."""
    seq = list(specs)

    def responder(purpose, system, user, output_model):
        assert output_model is StrategySpec
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return responder


def _bad_spec() -> StrategySpec:
    s = TEMPLATES["momentum_12_1"].spec()
    s.signal[0] = s.signal[0].model_copy(update={"feature": "momentum_12m"})
    return s


def test_llm_translator_happy_path():
    good = TEMPLATES["ff3"].spec()
    good.idea = "something the model paraphrased"
    llm = ScriptedLLM({"strategy_spec": _respond_with(good)})
    tr = StrategyTranslator(llm, today=TODAY).translate("build me the 3 factor model")
    assert tr.spec.idea == "build me the 3 factor model"
    assert tr.attempts == 1 and tr.errors_by_round == [[]]
    assert tr.template == "ff3" and tr.translator == "scripted"
    assert [c.purpose for c in llm.calls] == ["strategy_spec"]
    user = llm.prompts[0]["user"]
    assert "<idea>\nbuild me the 3 factor model\n</idea>" in user
    assert "2026-10-02" in user
    assert llm.prompts[0]["output_model"] == "StrategySpec"


def test_llm_translator_custom_name_has_no_template():
    s = TEMPLATES["momentum_12_1"].spec()
    s.name = "momentum_with_twist"
    tr = StrategyTranslator(ScriptedLLM({"strategy_spec": _respond_with(s)})).translate("momentum with a twist")
    assert tr.template is None


def test_llm_translator_repair_round():
    good = TEMPLATES["momentum_12_1"].spec()
    llm = ScriptedLLM({"strategy_spec": _respond_with(_bad_spec(), good)})
    tr = StrategyTranslator(llm).translate("12-1 momentum")
    assert tr.attempts == 2
    assert [c.purpose for c in llm.calls] == ["strategy_spec", "strategy_spec:repair"]
    assert tr.errors_by_round[1] == []
    assert any("unknown feature 'momentum_12m'" in e for e in tr.errors_by_round[0])
    assert any("did you mean" in e for e in tr.errors_by_round[0])
    repair_user = llm.prompts[1]["user"]
    assert "unknown feature 'momentum_12m'" in repair_user
    assert "<previous_spec>" in repair_user and '"momentum_12m"' in repair_user
    assert "<idea>\n12-1 momentum\n</idea>" in repair_user
    assert llm.prompts[0]["system"] == llm.prompts[1]["system"]
    assert tr.spec.signal[0].feature == "return_12m_ex_1m_pct"


def test_llm_translator_failure_after_repairs():
    llm = ScriptedLLM({"strategy_spec": _respond_with(_bad_spec())})
    with pytest.raises(StrategyTranslationError) as ei:
        StrategyTranslator(llm, max_repair_rounds=2).translate("12-1 momentum")
    assert len(llm.calls) == 3
    assert [c.purpose for c in llm.calls] == ["strategy_spec", "strategy_spec:repair", "strategy_spec:repair"]
    assert any("momentum_12m" in e for e in ei.value.errors)
    assert len(ei.value.suggestions) == 5


def test_llm_translator_no_repair_rounds():
    llm = ScriptedLLM({"strategy_spec": _respond_with(_bad_spec())})
    with pytest.raises(StrategyTranslationError):
        StrategyTranslator(llm, max_repair_rounds=0).translate("12-1 momentum")
    assert len(llm.calls) == 1


def test_llm_translator_structural_errors_are_repaired():
    bad = StrategySpec(name="x", idea="x", kind="screen", filters=[Condition(feature="gics_sector", op=">", value=1.0)])
    errs = spec_errors(bad, CAT)
    assert any("category feature 'gics_sector'" in e for e in errs)
    good = TEMPLATES["dislocation_screen"].spec()
    llm = ScriptedLLM({"strategy_spec": _respond_with(bad, good)})
    tr = StrategyTranslator(llm).translate("dislocations")
    assert tr.attempts == 2 and tr.template == "dislocation_screen"


def test_llm_errors_propagate():
    def boom(*a):
        raise LLMError("down")

    with pytest.raises(LLMError):
        StrategyTranslator(ScriptedLLM({"strategy_spec": boom})).translate("momentum")


def test_system_prompt_is_byte_stable_and_complete():
    llm = ScriptedLLM({"strategy_spec": _respond_with(TEMPLATES["ff3"].spec())})
    t1 = StrategyTranslator(llm, today=TODAY)
    t1.translate("3 factor model")
    StrategyTranslator(llm, today=date(2030, 1, 1)).translate("a completely different idea about momentum")
    s1, s2 = llm.prompts[0]["system"], llm.prompts[1]["system"]
    assert s1 == s2 == build_system_prompt(CAT) == t1.system_prompt
    assert CAT.to_prompt() in s1
    assert templates_prompt() in s1
    for t in TEMPLATES.values():
        assert t.key in s1 and t.title in s1
    for needle in ["cross_sectional", "screen", "factor_model", "time_series", "unsupported_requests", "costs_bps: 10",
                   "market_cap_usd_bn", "Copy the idea verbatim", "Never choose parameters"]:
        assert needle in s1
    assert "momentum about" not in s1 and "2026-10-02" not in s1  # no per-request content


def test_translate_idea_fallback_to_heuristic():
    def boom(*a):
        raise LLMError("rate limited")

    tr = translate_idea("3 factor model", ScriptedLLM({"strategy_spec": boom}), today=TODAY)
    assert tr.translator == "heuristic" and tr.template == "ff3"
    assert any("used the offline heuristic translator" in a for a in tr.spec.assumptions)
    assert translate_idea("3 factor model", today=TODAY).template == "ff3"
    good = TEMPLATES["low_beta"].spec()
    tr = translate_idea("betting against beta", ScriptedLLM({"strategy_spec": _respond_with(good)}), today=TODAY)
    assert tr.translator == "scripted"


def test_translate_idea_unknown_everywhere_raises():
    llm = ScriptedLLM({"strategy_spec": _respond_with(_bad_spec())})
    with pytest.raises(StrategyTranslationError) as ei:
        translate_idea("the phase of the moon", llm, today=TODAY)
    assert ei.value.suggestions
