"""Tests for aitrading.positioning.features.compute_positioning_features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields
from aitrading.positioning import compute_positioning_features
from aitrading.screen.catalog import POSITIONING_FEATURES


def short_interest(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: ticker -> (si shares, si shares 1m ago, float shares)."""
    df = pd.DataFrame.from_dict(
        rows, orient="index",
        columns=[fields.SHORT_INTEREST_SHARES, fields.SHORT_INTEREST_SHARES_1M_AGO, fields.FLOAT_SHARES],
    )
    df[fields.SI_SETTLEMENT_DATE] = pd.Timestamp("2026-09-15")
    df.index.name = "ticker"
    return df


def options(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: ticker -> (iv, iv 1y high, iv 1y low, put vol, call vol, put OI, call OI)."""
    df = pd.DataFrame.from_dict(rows, orient="index", columns=fields.OPTIONS_COLUMNS)
    df.index.name = "ticker"
    return df


@pytest.fixture()
def base():
    si = short_interest({"AAA": (8e6, 6.4e6, 1e8), "BBB": (2e6, 2e6, 4e7)})
    opt = options({"AAA": (0.45, 0.60, 0.30, 3000, 2000, 5e4, 2.5e4), "BBB": (0.20, 0.50, 0.25, 100, 400, 10, 40)})
    avg_vol = pd.Series({"AAA": 2e6, "BBB": 5e5})
    rv = pd.Series({"AAA": 30.0, "BBB": 25.0})
    return si, opt, avg_vol, rv


def test_columns_index_and_dtype(base):
    out = compute_positioning_features(*base, ["AAA", "BBB"])
    assert list(out.columns) == POSITIONING_FEATURES
    assert out.index.name == "ticker" and list(out.index) == ["AAA", "BBB"]
    assert (out.dtypes == "float64").all()


def test_each_feature_hand_checked(base):
    out = compute_positioning_features(*base, ["AAA", "BBB"])
    a, b = out.loc["AAA"], out.loc["BBB"]
    assert a["iv_30d_pct"] == pytest.approx(45.0)
    assert a["iv_rank_1y"] == pytest.approx(50.0)  # (0.45 - 0.30) / (0.60 - 0.30)
    assert a["iv_to_realized_vol_ratio"] == pytest.approx(1.5)  # 45 / 30
    assert a["put_call_volume_ratio"] == pytest.approx(1.5)
    assert a["put_call_oi_ratio"] == pytest.approx(2.0)
    assert a["short_interest_pct_float"] == pytest.approx(8.0)
    assert a["days_to_cover"] == pytest.approx(4.0)
    assert a["short_interest_change_1m_pct"] == pytest.approx(25.0)
    assert b["iv_30d_pct"] == pytest.approx(20.0)
    assert b["iv_rank_1y"] == 0.0  # below the vendor's 1y low: clipped
    assert b["iv_to_realized_vol_ratio"] == pytest.approx(0.8)
    assert b["put_call_volume_ratio"] == pytest.approx(0.25)
    assert b["put_call_oi_ratio"] == pytest.approx(0.25)
    assert b["short_interest_pct_float"] == pytest.approx(5.0)
    assert b["days_to_cover"] == pytest.approx(4.0)
    assert b["short_interest_change_1m_pct"] == pytest.approx(0.0)


def test_iv_rank_flat_range_and_clipping():
    opt = options({
        "FLAT": (0.30, 0.30, 0.30, 1, 1, 1, 1),
        "INV": (0.30, 0.20, 0.40, 1, 1, 1, 1),  # high < low: no valid range
        "NEWHIGH": (0.70, 0.60, 0.30, 1, 1, 1, 1),
        "LOW": (0.30, 0.60, 0.30, 1, 1, 1, 1),
    })
    out = compute_positioning_features(None, opt, None, None, ["FLAT", "INV", "NEWHIGH", "LOW"])
    assert math.isnan(out.loc["FLAT", "iv_rank_1y"]) and math.isnan(out.loc["INV", "iv_rank_1y"])
    assert out.loc["NEWHIGH", "iv_rank_1y"] == 100.0
    assert out.loc["LOW", "iv_rank_1y"] == 0.0


def test_zero_and_negative_denominators_give_nan():
    si = short_interest({"Z": (1e6, 0.0, 0.0), "N": (1e6, -5.0, -1e7)})
    opt = options({"Z": (0.3, 0.5, 0.2, 10, 0, 10, 0), "N": (0.3, 0.5, 0.2, 10, -3, 10, -1)})
    out = compute_positioning_features(si, opt, pd.Series({"Z": 0.0, "N": -1.0}), pd.Series({"Z": 0.0, "N": -2.0}), ["Z", "N"])
    for t in ("Z", "N"):
        for name in ("iv_to_realized_vol_ratio", "put_call_volume_ratio", "put_call_oi_ratio",
                     "short_interest_pct_float", "days_to_cover", "short_interest_change_1m_pct"):
            assert math.isnan(out.loc[t, name]), (t, name)
        assert out.loc[t, "iv_30d_pct"] == pytest.approx(30.0)


def test_zero_numerators_are_valid():
    si = short_interest({"A": (0.0, 1e6, 1e8)})
    opt = options({"A": (0.3, 0.5, 0.3, 0, 10, 0, 10)})
    out = compute_positioning_features(si, opt, pd.Series({"A": 1e6}), pd.Series({"A": 20.0}), ["A"]).loc["A"]
    assert out["put_call_volume_ratio"] == 0.0 and out["put_call_oi_ratio"] == 0.0
    assert out["short_interest_pct_float"] == 0.0 and out["days_to_cover"] == 0.0
    assert out["short_interest_change_1m_pct"] == -100.0
    assert out["iv_rank_1y"] == 0.0


def test_missing_tickers_and_partial_inputs(base):
    si, opt, avg_vol, rv = base
    out = compute_positioning_features(si, opt.drop(index="BBB"), avg_vol.drop("AAA"), rv, ["CCC", "BBB", "AAA"])
    assert list(out.index) == ["CCC", "BBB", "AAA"]
    assert out.loc["CCC"].isna().all()
    options_feats = ["iv_30d_pct", "iv_rank_1y", "iv_to_realized_vol_ratio", "put_call_volume_ratio", "put_call_oi_ratio"]
    assert out.loc["BBB", options_feats].isna().all()
    assert out.loc["BBB", "short_interest_pct_float"] == pytest.approx(5.0)
    assert math.isnan(out.loc["AAA", "days_to_cover"])  # no average volume for AAA
    assert out.loc["AAA", "short_interest_pct_float"] == pytest.approx(8.0)


def test_nan_inputs_propagate(base):
    si, opt, avg_vol, rv = base
    si.loc["AAA", fields.FLOAT_SHARES] = np.nan
    opt.loc["AAA", fields.IV_30D_ATM] = np.nan
    rv["BBB"] = np.nan
    out = compute_positioning_features(si, opt, avg_vol, rv, ["AAA", "BBB"])
    assert math.isnan(out.loc["AAA", "short_interest_pct_float"])
    assert out.loc["AAA", ["iv_30d_pct", "iv_rank_1y", "iv_to_realized_vol_ratio"]].isna().all()
    assert out.loc["AAA", "days_to_cover"] == pytest.approx(4.0)
    assert math.isnan(out.loc["BBB", "iv_to_realized_vol_ratio"])
    assert out.loc["BBB", "iv_30d_pct"] == pytest.approx(20.0)


def test_empty_and_none_inputs():
    out = compute_positioning_features(
        pd.DataFrame(columns=fields.SHORT_INTEREST_COLUMNS), pd.DataFrame(columns=fields.OPTIONS_COLUMNS),
        pd.Series(dtype=float), pd.Series(dtype=float), ["A", "B"],
    )
    assert list(out.index) == ["A", "B"] and out.isna().all().all()
    none = compute_positioning_features(None, None, None, None, ["A"])
    assert none.isna().all().all() and list(none.columns) == POSITIONING_FEATURES
    empty = compute_positioning_features(None, None, None, None, [])
    assert empty.empty and list(empty.columns) == POSITIONING_FEATURES


def test_missing_columns_object_values_ticker_column_and_duplicates():
    si = pd.DataFrame({
        "ticker": ["A", "B", "B"],
        fields.SHORT_INTEREST_SHARES: ["5000000", "1e6", "2e6"],  # strings from a vendor feed
        fields.FLOAT_SHARES: [1e8, 1e8, 4e7],
    })  # no 1m-ago column at all
    opt = options({"A": (0.3, 0.5, 0.2, 10, 20, 30, 60)}).drop(columns=[fields.PUT_VOLUME])
    out = compute_positioning_features(si, opt, None, None, ["A", "B", "A"])
    assert list(out.index) == ["A", "B"]  # duplicate requested ticker dropped
    assert out.loc["A", "short_interest_pct_float"] == pytest.approx(5.0)
    assert out.loc["B", "short_interest_pct_float"] == pytest.approx(5.0)  # last duplicate row wins: 2e6 / 4e7
    assert out["short_interest_change_1m_pct"].isna().all()
    assert math.isnan(out.loc["A", "put_call_volume_ratio"])
    assert out.loc["A", "put_call_oi_ratio"] == pytest.approx(0.5)


def test_canonical_demo_threshold_semantics(base):
    """short_interest_pct_float is in % of float (the demo screen uses '> 6')."""
    out = compute_positioning_features(*base, ["AAA", "BBB"])
    assert (out["short_interest_pct_float"] > 6).tolist() == [True, False]
