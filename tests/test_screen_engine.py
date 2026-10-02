"""Tests for aitrading.screen.engine (local ScreenSpec evaluation and the screen funnel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aitrading.core.models import FunnelStep
from aitrading.screen.catalog import default_catalog
from aitrading.screen.engine import (
    ScreenOutcome,
    ScreenValidationError,
    apply_universe,
    evaluate_condition,
    run_screen,
)
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

NAN = float("nan")
CATALOG = default_catalog()
NO_UNIVERSE = UniverseSpec(country="", security_types=[], min_price=None, min_avg_dollar_volume_usd_mn=None)

DEMO_OBSERVATION = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but "
    "have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI under "
    "40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where short "
    "interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then read the "
    "latest earnings calls and explain the dislocation."
)


def make_frame(rows: dict[str, dict]) -> pd.DataFrame:
    """Feature frame with every catalog column (NaN unless given) plus the reference columns."""
    cols = CATALOG.names() + ["name", "security_type", "country"]
    df = pd.DataFrame(index=pd.Index(list(rows), name="ticker"), columns=cols, dtype=object)
    numeric = [f.name for f in CATALOG if f.dtype != "category"]
    df[numeric] = np.nan
    df = df.astype({c: "float64" for c in numeric})
    defaults = {"name": None, "security_type": "common_stock", "country": "US", "price": 50.0,
                "avg_dollar_volume_20d_usd_mn": 20.0, "gics_sector": "Industrials"}
    for t, vals in rows.items():
        for k, v in {**defaults, "name": f"{t} Inc", **vals}.items():
            df.loc[t, k] = v
    return df


def C(feature: str, op: str, value=None, value_high=None, values=None, other=None, mult=1.0) -> Condition:
    return Condition(feature=feature, op=op, value=value, value_high=value_high, values=values,
                     other_feature=other, multiplier=mult)


def spec(conditions, any_of=(), universe=NO_UNIVERSE, ranking=None) -> ScreenSpec:
    return ScreenSpec(
        name="t", observation="test", universe=universe, conditions=list(conditions), any_of=[list(g) for g in any_of],
        ranking=ranking or [RankFactor(feature="fcf_yield_pct", direction="higher_is_better")],
    )


@pytest.fixture()
def frame():
    return make_frame({
        "AAA": {"rsi_14": 30.0, "fcf_yield_pct": 5.0, "price_vs_sma_50_pct": -5.0, "sma_50": 40.0, "sma_200": 35.0,
                "gics_sector": "Health Care", "golden_cross_20d": 1.0},
        "BBB": {"rsi_14": 40.0, "fcf_yield_pct": 4.0, "price_vs_sma_50_pct": 2.0, "sma_50": 30.0, "sma_200": 32.0,
                "gics_sector": "Energy", "golden_cross_20d": 0.0},
        "CCC": {"rsi_14": 55.0, "fcf_yield_pct": NAN, "price_vs_sma_50_pct": NAN, "sma_50": NAN, "sma_200": 20.0,
                "gics_sector": None, "golden_cross_20d": NAN},
        "DDD": {"rsi_14": NAN, "fcf_yield_pct": 0.1 + 0.2, "price_vs_sma_50_pct": 0.0, "sma_50": 10.0, "sma_200": 5.0,
                "gics_sector": " health care ", "golden_cross_20d": 1.0},
    })


def passing(mask: pd.Series) -> list[str]:
    return sorted(mask.index[mask.to_numpy()])


# --------------------------------------------------------------------------------------------
# evaluate_condition
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, value, expected",
    [
        (">", 40.0, ["CCC"]),
        (">=", 40.0, ["BBB", "CCC"]),
        ("<", 40.0, ["AAA"]),
        ("<=", 40.0, ["AAA", "BBB"]),
        ("==", 40.0, ["BBB"]),
        ("!=", 40.0, ["AAA", "CCC"]),  # DDD is NaN: never satisfies, not even '!='
    ],
)
def test_comparison_operators(frame, op, value, expected):
    mask = evaluate_condition(C("rsi_14", op, value), frame)
    assert mask.dtype == bool and mask.index.equals(frame.index)
    assert passing(mask) == expected


def test_between_is_inclusive_and_nan_excluded(frame):
    assert passing(evaluate_condition(C("rsi_14", "between", 30.0, 40.0), frame)) == ["AAA", "BBB"]
    assert passing(evaluate_condition(C("rsi_14", "between", 30.0, 30.0), frame)) == ["AAA"]
    assert passing(evaluate_condition(C("rsi_14", "between", 41.0, 54.9), frame)) == []


def test_numeric_equality_uses_isclose(frame):
    assert passing(evaluate_condition(C("fcf_yield_pct", "==", 0.3), frame)) == ["DDD"]  # 0.1 + 0.2
    assert passing(evaluate_condition(C("fcf_yield_pct", "!=", 0.3), frame)) == ["AAA", "BBB"]


def test_bool_feature_equality(frame):
    assert passing(evaluate_condition(C("golden_cross_20d", "==", 1), frame)) == ["AAA", "DDD"]
    assert passing(evaluate_condition(C("golden_cross_20d", "==", 0), frame)) == ["BBB"]


def test_other_feature_comparison_with_multiplier(frame):
    # sma_50 > sma_200: AAA 40>35, BBB 30<32, CCC NaN, DDD 10>5
    assert passing(evaluate_condition(C("sma_50", ">", other="sma_200"), frame)) == ["AAA", "DDD"]
    # sma_50 >= 1.2 x sma_200: AAA 40 < 42, DDD 10 >= 6
    assert passing(evaluate_condition(C("sma_50", ">=", other="sma_200", mult=1.2), frame)) == ["DDD"]
    # sma_50 == 2 x sma_200 (isclose)
    assert passing(evaluate_condition(C("sma_50", "==", other="sma_200", mult=2.0), frame)) == ["DDD"]
    # NaN on the right-hand side excludes too
    f2 = frame.copy()
    f2.loc["AAA", "sma_200"] = NAN
    assert passing(evaluate_condition(C("sma_50", "!=", other="sma_200"), f2)) == ["BBB", "DDD"]


def test_category_in_not_in_case_insensitive(frame):
    assert passing(evaluate_condition(C("gics_sector", "in", values=["HEALTH CARE"]), frame)) == ["AAA", "DDD"]
    assert passing(evaluate_condition(C("gics_sector", "in", values=["energy", "Utilities"]), frame)) == ["BBB"]
    # CCC has no sector: missing data never satisfies, not even 'not_in'
    assert passing(evaluate_condition(C("gics_sector", "not_in", values=["Energy"]), frame)) == ["AAA", "DDD"]


def test_category_equality_with_values_or_value(frame):
    assert passing(evaluate_condition(C("gics_sector", "==", values=["energy"]), frame)) == ["BBB"]
    assert passing(evaluate_condition(C("gics_sector", "!=", values=["Energy"]), frame)) == ["AAA", "DDD"]
    f2 = frame.copy()
    f2["gics_industry"] = ["45", "35", None, "45"]
    assert passing(evaluate_condition(C("gics_industry", "==", 45), f2)) == ["AAA", "DDD"]


def test_non_catalog_text_column_is_category(frame):
    assert passing(evaluate_condition(C("security_type", "==", values=["COMMON_STOCK"]), frame)) == ["AAA", "BBB", "CCC", "DDD"]


def test_infinite_values_count_as_missing(frame):
    f2 = frame.copy()
    f2.loc["AAA", "rsi_14"] = np.inf
    assert passing(evaluate_condition(C("rsi_14", ">", 0.0), f2)) == ["BBB", "CCC"]


def test_object_dtype_numbers_are_coerced(frame):
    f2 = frame.copy().astype({"rsi_14": object})
    f2.loc["AAA", "rsi_14"] = "30"
    f2.loc["BBB", "rsi_14"] = None
    assert passing(evaluate_condition(C("rsi_14", "<", 35.0), f2)) == ["AAA"]


def test_evaluate_condition_errors(frame):
    with pytest.raises(ScreenValidationError):
        evaluate_condition(C("no_such_feature", ">", 1.0), frame)
    with pytest.raises(ScreenValidationError):
        evaluate_condition(C("rsi_14", "between", 1.0), frame)  # missing value_high
    with pytest.raises(ScreenValidationError):
        evaluate_condition(C("gics_sector", ">", 1.0), frame)
    with pytest.raises(ScreenValidationError):
        evaluate_condition(C("gics_sector", "in", values=[]), frame)
    with pytest.raises(ScreenValidationError):
        evaluate_condition(C("sma_50", ">", 1.0, other="sma_200"), frame)  # both value and other


def test_empty_frame_condition():
    empty = make_frame({})
    mask = evaluate_condition(C("rsi_14", "<", 40.0), empty)
    assert mask.empty and mask.dtype == bool


# --------------------------------------------------------------------------------------------
# apply_universe
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def uframe():
    return make_frame({
        "A": {},
        "B": {"country": "ca"},
        "C": {"security_type": "ADR"},
        "D": {"price": 4.99},
        "E": {"avg_dollar_volume_20d_usd_mn": 4.0},
        "F": {"gics_sector": "Energy"},
        "G": {"price": NAN, "country": "us"},
        "H": {"avg_dollar_volume_20d_usd_mn": NAN},
        "I": {"price": 5.0, "avg_dollar_volume_20d_usd_mn": 5.0},
    })


def test_apply_universe_defaults(uframe):
    mask, funnel = apply_universe(UniverseSpec(), uframe)
    assert passing(mask) == ["A", "F", "I"]
    assert [s.label for s in funnel] == [
        "country == US", "security_type in [common_stock]", "price >= 5", "avg_dollar_volume_20d_usd_mn >= 5",
    ]
    assert [(s.passed_alone, s.remaining, s.missing_data) for s in funnel] == [
        (8, 8, 0), (8, 7, 0), (7, 5, 1), (7, 3, 1),
    ]


def test_apply_universe_exclude_sectors_and_disabled_filters(uframe):
    uframe.loc["I", "gics_sector"] = None
    u = UniverseSpec(country="US", security_types=["common_stock", "adr"], min_price=None,
                     min_avg_dollar_volume_usd_mn=None, exclude_sectors=["energy"])
    mask, funnel = apply_universe(u, uframe)
    assert passing(mask) == ["A", "C", "D", "E", "G", "H"]
    assert [s.label for s in funnel] == ["country == US", "security_type in [common_stock, adr]", "gics_sector not in [energy]"]
    last = funnel[-1]
    assert (last.passed_alone, last.remaining, last.missing_data) == (7, 6, 1)  # I: unknown sector


def test_apply_universe_all_disabled(uframe):
    mask, funnel = apply_universe(NO_UNIVERSE, uframe)
    assert mask.all() and funnel == []


def test_apply_universe_missing_column():
    f = make_frame({"A": {}}).drop(columns=["country"])
    with pytest.raises(ScreenValidationError):
        apply_universe(UniverseSpec(), f)
    mask, _ = apply_universe(UniverseSpec(country=""), f)
    assert mask.all()


# --------------------------------------------------------------------------------------------
# run_screen
# --------------------------------------------------------------------------------------------


def test_run_screen_funnel_arithmetic(frame):
    s = spec([C("rsi_14", "<=", 40.0), C("fcf_yield_pct", ">", 1.0), C("gics_sector", "in", values=["Health Care"])])
    out = run_screen(s, frame)
    assert isinstance(out, ScreenOutcome)
    assert out.universe_size == 4
    assert out.survivors == ["AAA"]
    assert passing(out.mask) == ["AAA"] and out.mask.dtype == bool
    assert out.funnel == [
        FunnelStep(label="rsi_14 <= 40", passed_alone=2, remaining=2, missing_data=1),  # DDD NaN
        FunnelStep(label="fcf_yield_pct > 1", passed_alone=2, remaining=2, missing_data=0),
        FunnelStep(label="gics_sector in [Health Care]", passed_alone=2, remaining=1, missing_data=0),
    ]


def test_missing_data_counts_only_remaining_names(frame):
    # CCC is dropped by the first condition, so its NaN fcf_yield_pct is not counted again
    s = spec([C("rsi_14", ">", 50.0), C("fcf_yield_pct", ">", 0.0)])
    out = run_screen(s, frame)
    assert [(f.passed_alone, f.remaining, f.missing_data) for f in out.funnel] == [(1, 1, 1), (3, 0, 1)]
    s = spec([C("rsi_14", "<", 50.0), C("fcf_yield_pct", ">", 0.0)])
    out = run_screen(s, frame)
    assert [(f.passed_alone, f.remaining, f.missing_data) for f in out.funnel] == [(2, 2, 1), (3, 2, 0)]


def test_any_of_groups(frame):
    s = spec(
        [C("fcf_yield_pct", ">", 0.0)],
        any_of=[
            [C("rsi_14", "<", 35.0), C("price_vs_sma_50_pct", ">", 1.0)],  # AAA | BBB
            [C("gics_sector", "in", values=["Energy"]), C("golden_cross_20d", "==", 1)],  # BBB | AAA, DDD
        ],
    )
    out = run_screen(s, frame)
    assert out.survivors == ["AAA", "BBB"]
    g1, g2 = out.funnel[1], out.funnel[2]
    assert g1.label == "any of: rsi_14 < 35 | price_vs_sma_50_pct > 1"
    assert (g1.passed_alone, g1.remaining) == (2, 2)
    # DDD (rsi NaN, price_vs_sma_50 0) fails both, one input missing -> missing data
    assert g1.missing_data == 1
    assert g2.label == "any of: gics_sector in [Energy] | golden_cross_20d == 1"
    assert (g2.passed_alone, g2.remaining, g2.missing_data) == (3, 2, 0)


def test_any_of_name_passing_one_alternative_is_not_missing(frame):
    s = spec([], any_of=[[C("rsi_14", "<", 100.0), C("fcf_yield_pct", ">", 100.0)]])
    out = run_screen(s, frame)
    assert out.survivors == ["AAA", "BBB", "CCC"]  # CCC passes via rsi despite NaN fcf
    assert out.funnel[0].missing_data == 1  # only DDD (rsi NaN, fcf 0.3 fails)


def test_survivors_sorted_and_universe_steps_first():
    f = make_frame({"ZZZ": {"rsi_14": 10.0}, "MMM": {"rsi_14": 20.0}, "AAA": {"rsi_14": 30.0, "price": 1.0}})
    out = run_screen(spec([C("rsi_14", "<", 40.0)], universe=UniverseSpec()), f)
    assert out.survivors == ["MMM", "ZZZ"]
    assert [s.label for s in out.funnel][:4] == [
        "country == US", "security_type in [common_stock]", "price >= 5", "avg_dollar_volume_20d_usd_mn >= 5",
    ]
    assert out.funnel[-1] == FunnelStep(label="rsi_14 < 40", passed_alone=3, remaining=2, missing_data=0)


def test_empty_frame_run_screen():
    out = run_screen(spec([C("rsi_14", "<", 40.0)], universe=UniverseSpec()), make_frame({}))
    assert out.survivors == [] and out.universe_size == 0
    assert all(s.passed_alone == 0 and s.remaining == 0 for s in out.funnel)


def test_no_conditions_keeps_universe(frame):
    out = run_screen(spec([]), frame)
    assert out.survivors == ["AAA", "BBB", "CCC", "DDD"] and out.funnel == []


def test_validation_errors_are_collected(frame):
    bad = spec(
        [C("rsi14", "<", 40.0), C("rsi_14", "between", 50.0, 40.0), C("gics_sector", ">", 1.0), C("rsi_14", "in", values=["x"])],
        any_of=[[C("rsi_14", "<", 1.0)]],
        ranking=[RankFactor(feature="gics_sector", direction="higher_is_better")],
    )
    with pytest.raises(ScreenValidationError) as ei:
        run_screen(bad, frame)
    err = ei.value
    assert isinstance(err, ValueError)
    text = " | ".join(err.errors)
    assert "unknown feature 'rsi14'" in text and "did you mean" in text
    assert "value must be <= value_high" in text
    assert "only supports in/not_in" in text
    assert "'in'/'not_in' only apply to category features" in text
    assert "at least two alternatives" in text
    assert "cannot rank on category feature" in text
    assert len(err.errors) >= 6
    assert str(err).startswith("invalid screen: ")


def test_validation_unit_mismatch(frame):
    with pytest.raises(ScreenValidationError, match="unit mismatch"):
        run_screen(spec([C("rsi_14", ">", other="price")]), frame)


def test_validation_missing_frame_columns(frame):
    f = frame.drop(columns=["rsi_14", "fcf_yield_pct", "price"])
    with pytest.raises(ScreenValidationError) as ei:
        run_screen(spec([C("rsi_14", "<", 40.0)], universe=UniverseSpec()), f)
    msg = " ".join(ei.value.errors)
    assert "rsi_14" in msg and "fcf_yield_pct" in msg and "price" in msg  # ranking + universe columns too


def test_validation_duplicate_tickers(frame):
    f = pd.concat([frame, frame.loc[["AAA"]]])
    with pytest.raises(ScreenValidationError, match="duplicate tickers: AAA"):
        run_screen(spec([C("rsi_14", "<", 40.0)]), f)


def test_canonical_demo_spec():
    demo = ScreenSpec(
        name="pullback-in-uptrend-fcf",
        observation=DEMO_OBSERVATION,
        conditions=[
            C("market_cap_usd_bn", "between", 2, 20),
            C("sma_50_vs_sma_200_pct", ">", 0),
            C("return_12m_ex_1m_pct", ">", 0),
            C("drawdown_from_52w_high_pct", "between", -40, -15),
            C("max_volume_ratio_20d", ">=", 2),
            C("rsi_14", "<", 40),
            C("fcf_yield_pct", ">", 4),
            C("revenue_growth_yoy_pct", ">", 8),
            C("short_interest_pct_float", ">", 6),
        ],
        ranking=[
            RankFactor(feature="fcf_yield_pct", direction="higher_is_better"),
            RankFactor(feature="revenue_growth_yoy_pct", direction="higher_is_better"),
            RankFactor(feature="drawdown_from_52w_high_pct", direction="lower_is_better"),
        ],
    )
    good = {"market_cap_usd_bn": 6.0, "sma_50_vs_sma_200_pct": 3.0, "return_12m_ex_1m_pct": 25.0,
            "drawdown_from_52w_high_pct": -22.0, "max_volume_ratio_20d": 2.6, "rsi_14": 34.0, "fcf_yield_pct": 6.5,
            "revenue_growth_yoy_pct": 12.0, "short_interest_pct_float": 8.0}
    f = make_frame({
        "GOOD": good,
        "EDGE": {**good, "market_cap_usd_bn": 20.0, "drawdown_from_52w_high_pct": -15.0, "max_volume_ratio_20d": 2.0},
        "BIG": {**good, "market_cap_usd_bn": 25.0},
        "SHALLOW": {**good, "drawdown_from_52w_high_pct": -10.0},
        "NOSI": {**good, "short_interest_pct_float": NAN},
        "PENNY": {**good, "price": 3.0},
        "ADR": {**good, "security_type": "adr"},
    })
    out = run_screen(demo, f)
    assert out.survivors == ["EDGE", "GOOD"]
    assert out.universe_size == 7
    labels = [s.label for s in out.funnel]
    assert labels[4:] == [c.describe() for c in demo.conditions]
    assert out.funnel[4].label == "market_cap_usd_bn between 2 and 20"
    assert out.funnel[-1].label == "short_interest_pct_float > 6"
    assert out.funnel[-1].missing_data == 1  # NOSI
    remaining = [s.remaining for s in out.funnel]
    assert remaining == sorted(remaining, reverse=True) and remaining[-1] == 2
