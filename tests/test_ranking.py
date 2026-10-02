"""Tests for aitrading.rank.scoring (percentile composite ranking of screen survivors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aitrading.core.models import RankedCandidate
from aitrading.rank.scoring import factor_percentile, rank_candidates
from aitrading.screen.spec import RankFactor

NAN = float("nan")
HI, LO = "higher_is_better", "lower_is_better"


def F(feature: str, direction: str = HI, weight: float = 1.0) -> RankFactor:
    return RankFactor(feature=feature, direction=direction, weight=weight)


def make_frame(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    if "name" not in df.columns:
        df["name"] = [f"{t} Corp" for t in df.index]
    return df


@pytest.fixture()
def frame():
    return make_frame({
        "AAA": {"fcf_yield_pct": 8.0, "revenue_growth_yoy_pct": 10.0, "drawdown_from_52w_high_pct": -20.0, "rsi_14": 31.0},
        "BBB": {"fcf_yield_pct": 6.0, "revenue_growth_yoy_pct": 30.0, "drawdown_from_52w_high_pct": -35.0, "rsi_14": 25.0},
        "CCC": {"fcf_yield_pct": 4.5, "revenue_growth_yoy_pct": 9.0, "drawdown_from_52w_high_pct": -16.0, "rsi_14": 38.0},
        "DDD": {"fcf_yield_pct": NAN, "revenue_growth_yoy_pct": 20.0, "drawdown_from_52w_high_pct": -25.0, "rsi_14": 33.0},
        "EEE": {"fcf_yield_pct": 99.0, "revenue_growth_yoy_pct": 1.0, "drawdown_from_52w_high_pct": -50.0, "rsi_14": 20.0},
    })


DEMO_RANKING = [F("fcf_yield_pct"), F("revenue_growth_yoy_pct"), F("drawdown_from_52w_high_pct", LO)]


# --------------------------------------------------------------------------------------------
# factor_percentile
# --------------------------------------------------------------------------------------------


def test_percentile_direction_ties_and_nan():
    s = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    assert factor_percentile(s).tolist() == [0.0, 0.5, 1.0]
    assert factor_percentile(s, higher_is_better=False).tolist() == [1.0, 0.5, 0.0]
    ties = pd.Series([1.0, 1.0, 2.0], index=list("abc"))
    assert factor_percentile(ties).tolist() == [0.25, 0.25, 1.0]
    assert factor_percentile(ties, higher_is_better=False).tolist() == [0.75, 0.75, 0.0]
    with_nan = pd.Series([1.0, NAN, 3.0, np.inf], index=list("abcd"))
    assert factor_percentile(with_nan).tolist() == [0.0, 0.5, 1.0, 0.5]


def test_percentile_degenerate_cross_sections():
    assert factor_percentile(pd.Series([5.0], index=["a"]), higher_is_better=False).tolist() == [1.0]
    assert factor_percentile(pd.Series([5.0, NAN], index=["a", "b"]), higher_is_better=False).tolist() == [1.0, 0.5]
    assert factor_percentile(pd.Series([NAN, NAN], index=["a", "b"])).tolist() == [0.5, 0.5]
    assert factor_percentile(pd.Series([7.0, 7.0, 7.0], index=list("abc"))).tolist() == [0.5, 0.5, 0.5]
    assert factor_percentile(pd.Series([], dtype=float)).empty


def test_percentile_is_robust_to_outliers():
    base = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    outlier = pd.Series([1.0, 2.0, 1e12], index=list("abc"))
    assert factor_percentile(base).tolist() == factor_percentile(outlier).tolist()


# --------------------------------------------------------------------------------------------
# rank_candidates
# --------------------------------------------------------------------------------------------


def test_demo_ranking_hand_computed(frame):
    survivors = ["AAA", "BBB", "CCC", "DDD"]
    out = rank_candidates(frame, DEMO_RANKING, survivors, top_n=10)
    assert all(isinstance(c, RankedCandidate) for c in out)
    # fcf (valid AAA 8, BBB 6, CCC 4.5): AAA 1, BBB .5, CCC 0, DDD NaN -> .5
    # growth (10, 30, 9, 20): AAA 1/3, BBB 1, CCC 0, DDD 2/3
    # drawdown lower is better (-20, -35, -16, -25): AAA 1/3, BBB 1, CCC 0, DDD 2/3
    expected = {
        "AAA": (1.0 + 1 / 3 + 1 / 3) / 3,
        "BBB": (0.5 + 1.0 + 1.0) / 3,
        "CCC": 0.0,
        "DDD": (0.5 + 2 / 3 + 2 / 3) / 3,
    }
    assert [c.ticker for c in out] == ["BBB", "DDD", "AAA", "CCC"]
    assert [c.rank for c in out] == [1, 2, 3, 4]
    for c in out:
        assert c.score == pytest.approx(expected[c.ticker], abs=1e-12)
        assert 0.0 <= c.score <= 1.0
    bbb = out[0]
    assert bbb.name == "BBB Corp"
    assert bbb.factor_scores == pytest.approx({"fcf_yield_pct": 0.5, "revenue_growth_yoy_pct": 1.0, "drawdown_from_52w_high_pct": 1.0})
    assert list(bbb.factor_scores) == ["fcf_yield_pct", "revenue_growth_yoy_pct", "drawdown_from_52w_high_pct"]
    assert bbb.features == {"fcf_yield_pct": 6.0, "revenue_growth_yoy_pct": 30.0, "drawdown_from_52w_high_pct": -35.0}
    ddd = out[1]
    assert ddd.features["fcf_yield_pct"] is None and ddd.factor_scores["fcf_yield_pct"] == 0.5


def test_percentiles_use_survivors_only(frame):
    # EEE (fcf 99) is not a survivor and must not affect anyone's score
    a = rank_candidates(frame, [F("fcf_yield_pct")], ["AAA", "BBB"], top_n=5)
    assert [(c.ticker, c.score) for c in a] == [("AAA", 1.0), ("BBB", 0.0)]


def test_weights_are_normalised(frame):
    survivors = ["AAA", "BBB", "CCC"]
    w = rank_candidates(frame, [F("fcf_yield_pct", weight=3.0), F("revenue_growth_yoy_pct", weight=1.0)], survivors, 3)
    scores = {c.ticker: c.score for c in w}
    # fcf: AAA 1, BBB .5, CCC 0 ; growth: AAA .5, BBB 1, CCC 0
    assert scores["AAA"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.5)
    assert scores["BBB"] == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    assert scores["CCC"] == 0.0
    scaled = rank_candidates(frame, [F("fcf_yield_pct", weight=30.0), F("revenue_growth_yoy_pct", weight=10.0)], survivors, 3)
    assert [(c.ticker, c.score) for c in scaled] == [(c.ticker, c.score) for c in w]


def test_ties_broken_by_ticker():
    f = make_frame({"ZZZ": {"x": 1.0, "y": 2.0}, "AAA": {"x": 2.0, "y": 1.0}, "MMM": {"x": 1.5, "y": 1.5}})
    out = rank_candidates(f, [F("x"), F("y")], ["ZZZ", "AAA", "MMM"], 3)
    assert [c.ticker for c in out] == ["AAA", "MMM", "ZZZ"]
    assert {c.score for c in out} == {0.5}


def test_top_n_truncates(frame):
    out = rank_candidates(frame, DEMO_RANKING, ["AAA", "BBB", "CCC", "DDD"], top_n=2)
    assert [c.ticker for c in out] == ["BBB", "DDD"] and [c.rank for c in out] == [1, 2]


def test_single_survivor_scores_one(frame):
    out = rank_candidates(frame, DEMO_RANKING, ["CCC"], top_n=10)
    assert len(out) == 1
    c = out[0]
    assert c.score == 1.0 and c.rank == 1
    assert set(c.factor_scores.values()) == {1.0}  # lower_is_better is not mirrored to 0 for a lone value


def test_single_survivor_with_missing_factor(frame):
    out = rank_candidates(frame, DEMO_RANKING, ["DDD"], top_n=1)
    assert out[0].score == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_all_nan_factor_is_neutral(frame):
    f = frame.assign(empty=NAN)
    out = rank_candidates(f, [F("fcf_yield_pct"), F("empty")], ["AAA", "BBB", "CCC"], 3)
    assert [c.factor_scores["empty"] for c in out] == [0.5, 0.5, 0.5]
    assert [c.score for c in out] == pytest.approx([0.75, 0.5, 0.25])


def test_empty_survivors(frame):
    assert rank_candidates(frame, DEMO_RANKING, [], top_n=5) == []


def test_duplicate_survivors_and_order_independence(frame):
    a = rank_candidates(frame, DEMO_RANKING, ["DDD", "AAA", "BBB", "CCC", "AAA"], 10)
    b = rank_candidates(frame, DEMO_RANKING, ["AAA", "BBB", "CCC", "DDD"], 10)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_features_dict_types_and_extras(frame):
    f = frame.assign(
        gics_sector=["Industrials", "Energy", None, "Health Care", "Utilities"],
        golden_cross_20d=np.array([True, False, True, False, True]),
        num_analysts=np.array([12, 5, 3, 0, 1], dtype=np.int64),
        last_earnings=pd.to_datetime(["2026-08-04"] * 5),
    )
    out = rank_candidates(
        f, [F("fcf_yield_pct")], ["AAA", "CCC"], 2,
        extra_features=["rsi_14", "gics_sector", "golden_cross_20d", "num_analysts", "last_earnings", "fcf_yield_pct", "not_a_column"],
    )
    aaa, ccc = out
    assert list(aaa.features) == ["fcf_yield_pct", "rsi_14", "gics_sector", "golden_cross_20d", "num_analysts", "last_earnings", "not_a_column"]
    assert aaa.features["rsi_14"] == 31.0 and type(aaa.features["rsi_14"]) is float
    assert aaa.features["gics_sector"] == "Industrials"
    assert aaa.features["golden_cross_20d"] == 1.0 and type(aaa.features["golden_cross_20d"]) is float
    assert aaa.features["num_analysts"] == 12.0 and type(aaa.features["num_analysts"]) is float
    assert aaa.features["last_earnings"].startswith("2026-08-04")
    assert aaa.features["not_a_column"] is None
    assert ccc.features["gics_sector"] is None
    assert type(aaa.score) is float and all(type(v) is float for v in aaa.factor_scores.values())
    aaa.model_dump_json()  # JSON-serialisable


def test_name_fallback_to_ticker(frame):
    f = frame.copy()
    f.loc["AAA", "name"] = None
    out = rank_candidates(f, [F("fcf_yield_pct")], ["AAA"], 1)
    assert out[0].name == "AAA"
    out = rank_candidates(f.drop(columns=["name"]), [F("fcf_yield_pct")], ["BBB"], 1)
    assert out[0].name == "BBB"


def test_duplicate_factor_same_direction_merges(frame):
    merged = rank_candidates(frame, [F("fcf_yield_pct"), F("fcf_yield_pct"), F("revenue_growth_yoy_pct")], ["AAA", "BBB", "CCC"], 3)
    weighted = rank_candidates(frame, [F("fcf_yield_pct", weight=2.0), F("revenue_growth_yoy_pct")], ["AAA", "BBB", "CCC"], 3)
    assert [(c.ticker, c.score) for c in merged] == [(c.ticker, c.score) for c in weighted]
    assert list(merged[0].factor_scores) == ["fcf_yield_pct", "revenue_growth_yoy_pct"]


def test_errors(frame):
    with pytest.raises(ValueError, match="at least one factor"):
        rank_candidates(frame, [], ["AAA"], 1)
    with pytest.raises(ValueError, match="not in the feature frame: nope"):
        rank_candidates(frame, [F("nope")], ["AAA"], 1)
    with pytest.raises(ValueError, match="survivors not in the feature frame: XYZ"):
        rank_candidates(frame, [F("fcf_yield_pct")], ["AAA", "XYZ"], 1)
    with pytest.raises(ValueError, match="top_n"):
        rank_candidates(frame, [F("fcf_yield_pct")], ["AAA"], 0)
    with pytest.raises(ValueError, match="winsor"):
        rank_candidates(frame, [F("fcf_yield_pct")], ["AAA"], 1, winsor=(0.9, 0.1))
    with pytest.raises(ValueError, match="conflicting directions"):
        rank_candidates(frame, [F("fcf_yield_pct"), F("fcf_yield_pct", LO)], ["AAA"], 1)
    dup = pd.concat([frame, frame.loc[["AAA"]]])
    with pytest.raises(ValueError, match="duplicate"):
        rank_candidates(dup, [F("fcf_yield_pct")], ["AAA"], 1)


def test_winsor_does_not_change_percentile_scores(frame):
    a = rank_candidates(frame, DEMO_RANKING, ["AAA", "BBB", "CCC", "DDD", "EEE"], 5)
    b = rank_candidates(frame, DEMO_RANKING, ["AAA", "BBB", "CCC", "DDD", "EEE"], 5, winsor=(0.0, 1.0))
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_large_universe_is_fast_and_deterministic():
    rng = np.random.default_rng(7)
    n = 3000
    tickers = [f"T{i:04d}" for i in range(n)]
    f = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n), "name": tickers}, index=pd.Index(tickers, name="ticker")
    )
    f.loc[f.index[::10], "b"] = NAN
    ranking = [F("a"), F("b", LO, 2.0), F("c", weight=0.5)]
    one = rank_candidates(f, ranking, tickers, 100)
    two = rank_candidates(f, ranking, list(reversed(tickers)), 100)
    assert [c.model_dump() for c in one] == [c.model_dump() for c in two]
    assert len(one) == 100 and [c.rank for c in one] == list(range(1, 101))
    scores = [c.score for c in one]
    assert scores == sorted(scores, reverse=True)
