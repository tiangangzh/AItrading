"""Tests for aitrading.technical.features.compute_technical_features and helpers."""

from __future__ import annotations

import math
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aitrading.data.base import PricePanel
from aitrading.screen.catalog import TECHNICAL_FEATURES
from aitrading.technical import indicators as ind
from aitrading.technical.features import (
    average_volume_shares,
    compute_technical_features,
    cross_sectional_percentile,
    evaluation_date,
)

AS_OF = date(2026, 9, 30)  # a Wednesday
N_ROWS = 300
INDEX = pd.bdate_range(end=AS_OF, periods=N_ROWS)
I = np.arange(N_ROWS, dtype=float)
BOOL_FEATURES = ["golden_cross_20d", "macd_bullish_cross_10d"]


def panel_from(closes: dict, index=INDEX, highs=None, lows=None, volumes=None) -> PricePanel:
    close = pd.DataFrame(closes, index=index)
    high = pd.DataFrame(highs, index=index) if highs is not None else close * 1.01
    low = pd.DataFrame(lows, index=index) if lows is not None else close * 0.99
    vol = pd.DataFrame(volumes, index=index) if volumes is not None else close * 0 + 1e6
    return PricePanel(open=close.copy(), high=high, low=low, close=close, volume=vol)


def bench_series(index=INDEX) -> pd.Series:
    """Deterministic benchmark with varying daily returns (non-zero variance)."""
    i = np.arange(len(index), dtype=float)
    rets = np.r_[0.0, 0.0005 + 0.01 * np.sin(0.7 * i[1:])]
    return pd.Series(1000.0 * np.cumprod(1.0 + rets), index=index, name="SPX")


def bench_return_pct(bench: pd.Series, w: int) -> float:
    return (bench.iloc[-1] / bench.iloc[-1 - w] - 1) * 100


@pytest.fixture(scope="module")
def demo():
    """Deterministic panel with hand-checkable tickers."""
    rng = np.random.default_rng(42)
    bench = bench_series()
    lin = 100.0 + I
    lin_vol = np.full(N_ROWS, 1e6)
    lin_vol[-5] = 3e6
    bench_ret = bench.pct_change().fillna(0.0).to_numpy()
    walk = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, N_ROWS)))
    gap = walk.copy()
    gap[200] = np.nan
    ipo = np.full(N_ROWS, np.nan)
    ipo[-30:] = 20.0 + 0.1 * np.arange(30)
    halt = lin.copy()
    halt[-1] = np.nan
    closes = {
        "LIN": lin,
        "GEO": 50.0 * 1.002 ** I,
        "BETA": 80.0 * np.cumprod(1.0 + 1.5 * bench_ret),
        "WALK": walk,
        "GAP": gap,
        "IPO": ipo,
        "HALT": halt,
        "DEAD": np.full(N_ROWS, np.nan),
    }
    highs = {k: v * 1.01 for k, v in closes.items()}
    lows = {k: v * 0.99 for k, v in closes.items()}
    highs["LIN"], lows["LIN"] = lin + 1.0, lin - 1.0
    volumes = {k: np.full(N_ROWS, 1e6) for k in closes}
    volumes["LIN"] = lin_vol
    prices = panel_from(closes, highs=highs, lows=lows, volumes=volumes)
    return prices, bench, compute_technical_features(prices, bench, AS_OF)


# --- shape / contract --------------------------------------------------------------------------


def test_exact_columns_index_and_dtype(demo):
    prices, _, out = demo
    assert list(out.columns) == TECHNICAL_FEATURES
    assert out.index.name == "ticker"
    assert list(out.index) == prices.tickers
    assert (out.dtypes == "float64").all()


def test_lookahead_safety_appending_future_rows(demo):
    prices, bench, out = demo
    future = pd.bdate_range(start=AS_OF + pd.Timedelta(days=1), periods=15)
    rng = np.random.default_rng(0)

    def extend(frame: pd.DataFrame) -> pd.DataFrame:
        extra = pd.DataFrame(rng.uniform(1, 1e7, (len(future), frame.shape[1])), index=future, columns=frame.columns)
        return pd.concat([frame, extra])

    longer = PricePanel(*(extend(getattr(prices, f)) for f in ("open", "high", "low", "close", "volume")))
    longer_bench = pd.concat([bench, pd.Series(rng.uniform(1, 1e5, len(future)), index=future)])
    pd.testing.assert_frame_equal(compute_technical_features(longer, longer_bench, AS_OF), out)


def test_as_of_between_sessions_uses_last_session_before(demo):
    prices, bench, _ = demo
    sunday, friday = date(2026, 9, 27), date(2026, 9, 25)
    assert evaluation_date(prices, sunday) == pd.Timestamp(friday)
    pd.testing.assert_frame_equal(
        compute_technical_features(prices, bench, sunday), compute_technical_features(prices, bench, friday)
    )
    assert evaluation_date(prices, AS_OF) == pd.Timestamp(AS_OF)
    assert evaluation_date(prices, date(2000, 1, 1)) is None


# --- hand-checked values -----------------------------------------------------------------------


def test_linear_ticker_hand_checked(demo):
    _, bench, out = demo
    r = out.loc["LIN"]
    p = 399.0  # 100 + 299
    assert r["price"] == p
    assert r["sma_20"] == pytest.approx(389.5)
    assert r["sma_50"] == pytest.approx(374.5)
    assert r["sma_200"] == pytest.approx(299.5)
    assert r["price_vs_sma_50_pct"] == pytest.approx((p / 374.5 - 1) * 100)
    assert r["price_vs_sma_200_pct"] == pytest.approx((p / 299.5 - 1) * 100)
    assert r["sma_50_vs_sma_200_pct"] == pytest.approx((374.5 / 299.5 - 1) * 100)
    assert r["sma_200_slope_1m_pct"] == pytest.approx((299.5 / 278.5 - 1) * 100)
    assert r["golden_cross_20d"] == 0.0  # sma_50 above sma_200 all along: no crossing
    assert r["return_1m_pct"] == pytest.approx((p / 378 - 1) * 100)
    assert r["return_3m_pct"] == pytest.approx((p / 336 - 1) * 100)
    assert r["return_6m_pct"] == pytest.approx((p / 273 - 1) * 100)
    assert r["return_12m_pct"] == pytest.approx((p / 147 - 1) * 100)
    assert r["return_12m_ex_1m_pct"] == pytest.approx((378 / 147 - 1) * 100)
    for label, w in (("3m", 63), ("6m", 126), ("12m", 252)):
        bench_ret = bench_return_pct(bench, w)
        assert r[f"rel_strength_{label}_pp"] == pytest.approx(r[f"return_{label}_pct"] - bench_ret)
    assert r["rsi_14"] == 100.0
    assert r["high_52w"] == 400.0
    assert r["low_52w"] == 147.0  # lowest low of rows 48..299 = (100 + 48) - 1
    assert r["drawdown_from_52w_high_pct"] == pytest.approx(-0.25)
    assert r["above_52w_low_pct"] == pytest.approx((p / 147 - 1) * 100)
    assert r["days_since_52w_high"] == 0.0
    assert r["atr_14_pct"] == pytest.approx(2.0 / p * 100)  # every true range is exactly 2
    logret = np.diff(np.log(100.0 + I))
    assert r["volatility_20d_pct"] == pytest.approx(np.std(logret[-20:], ddof=1) * math.sqrt(252) * 100)
    assert r["volatility_60d_pct"] == pytest.approx(np.std(logret[-60:], ddof=1) * math.sqrt(252) * 100)
    # volume: 1mn shares a day with a 3mn spike five sessions ago
    assert r["avg_dollar_volume_20d_usd_mn"] == pytest.approx(429.0)
    assert r["rel_volume_5d"] == pytest.approx((7e6 / 5) / (62e6 / 60))
    assert r["rel_volume_20d"] == pytest.approx((22e6 / 20) / (122e6 / 120))
    assert r["max_volume_ratio_20d"] == pytest.approx(3e6 / (122e6 / 120))
    assert math.isnan(r["up_down_volume_ratio_50d"])  # no down-close sessions: zero denominator


def test_macd_values_match_independent_pandas(demo):
    prices, _, out = demo
    for t in ("LIN", "GEO", "WALK"):
        c = prices.close[t]
        line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        sig = line.ewm(span=9, adjust=False).mean()
        assert out.loc[t, "macd_line"] == pytest.approx(line.iloc[-1], rel=1e-9)
        assert out.loc[t, "macd_signal"] == pytest.approx(sig.iloc[-1], rel=1e-9)
        assert out.loc[t, "macd_histogram"] == pytest.approx(line.iloc[-1] - sig.iloc[-1], rel=1e-7, abs=1e-12)
        assert out.loc[t, "macd_histogram_pct_price"] == pytest.approx(
            (line.iloc[-1] - sig.iloc[-1]) / c.iloc[-1] * 100, rel=1e-7, abs=1e-12
        )


def test_geometric_ticker(demo):
    _, _, out = demo
    r = out.loc["GEO"]
    assert r["return_1m_pct"] == pytest.approx((1.002**21 - 1) * 100)
    assert r["return_12m_ex_1m_pct"] == pytest.approx((1.002**231 - 1) * 100)
    assert r["volatility_20d_pct"] == pytest.approx(0.0, abs=1e-6)
    assert r["rsi_14"] == 100.0
    assert r["drawdown_from_52w_high_pct"] == pytest.approx((1 / 1.01 - 1) * 100)


def test_beta_and_relative_strength_vs_benchmark(demo):
    _, bench, out = demo
    r = out.loc["BETA"]
    assert r["beta_1y"] == pytest.approx(1.5, rel=1e-9)
    rb = bench.pct_change().to_numpy()
    for label, w in (("3m", 63), ("6m", 126), ("12m", 252)):
        own = (np.prod(1 + 1.5 * rb[-w:]) - 1) * 100
        assert r[f"return_{label}_pct"] == pytest.approx(own)
        assert r[f"rel_strength_{label}_pp"] == pytest.approx(own - bench_return_pct(bench, w))
    # the linear ticker has no co-movement with the benchmark beyond chance: beta is finite
    assert math.isfinite(out.loc["LIN", "beta_1y"])


def test_return_6m_percentile(demo):
    _, _, out = demo
    expected = cross_sectional_percentile(out["return_6m_pct"])
    pd.testing.assert_series_equal(out["return_6m_percentile"], expected, check_names=False)
    valid = out["return_6m_pct"].dropna()
    assert out.loc[valid.idxmax(), "return_6m_percentile"] == 100.0
    assert out.loc[valid.idxmin(), "return_6m_percentile"] == 0.0
    assert out.loc[["IPO", "HALT", "DEAD"], "return_6m_percentile"].isna().all()


def test_cross_sectional_percentile_ties_nan_and_single():
    s = pd.Series([10.0, np.nan, 30.0, 20.0, 20.0], index=list("abcde"))
    out = cross_sectional_percentile(s)
    # ranks among 4 valid values: 1, 4, 2.5, 2.5 -> (rank - 1) / 3 * 100
    assert out["a"] == 0.0 and out["c"] == 100.0 and math.isnan(out["b"])
    assert out["d"] == pytest.approx(50.0) and out["e"] == pytest.approx(50.0)
    assert cross_sectional_percentile(pd.Series([5.0, np.nan])).tolist()[0] == 50.0
    assert cross_sectional_percentile(pd.Series([np.nan, np.nan])).isna().all()
    assert cross_sectional_percentile(pd.Series([], dtype=float)).empty


# --- NaN handling ------------------------------------------------------------------------------


def test_no_trade_on_evaluation_row_gives_all_nan(demo):
    _, _, out = demo
    assert out.loc["HALT"].isna().all()
    assert out.loc["DEAD"].isna().all()


def test_short_history_ipo(demo):
    _, _, out = demo
    r = out.loc["IPO"]  # 30 bars
    for name in ("price", "sma_20", "return_1m_pct", "rsi_14", "macd_line", "volatility_20d_pct", "atr_14_pct",
                 "avg_dollar_volume_20d_usd_mn"):
        assert not math.isnan(r[name]), name
    for name in ("sma_50", "sma_200", "return_3m_pct", "return_12m_ex_1m_pct", "rel_strength_3m_pp", "macd_signal",
                 "macd_bullish_cross_10d", "golden_cross_20d", "high_52w", "drawdown_from_52w_high_pct",
                 "days_since_52w_high", "volatility_60d_pct", "beta_1y", "rel_volume_5d", "rel_volume_20d",
                 "max_volume_ratio_20d", "up_down_volume_ratio_50d", "return_6m_percentile"):
        assert math.isnan(r[name]), name
    assert r["price"] == pytest.approx(22.9)
    assert r["return_1m_pct"] == pytest.approx((22.9 / 20.8 - 1) * 100)


def test_interior_gap_skipped_as_bars(demo):
    """A missing print is skipped: the ticker's features equal those of its own bar series."""
    prices, bench, out = demo
    keep = prices.close["GAP"].notna()
    idx = INDEX[keep.to_numpy()]
    own = panel_from(
        {"GAP": prices.close["GAP"][keep].to_numpy()},
        index=idx,
        highs={"GAP": prices.high["GAP"][keep].to_numpy()},
        lows={"GAP": prices.low["GAP"][keep].to_numpy()},
        volumes={"GAP": prices.volume["GAP"][keep].to_numpy()},
    )
    alone = compute_technical_features(own, bench, AS_OF).loc["GAP"].drop("return_6m_percentile")
    pd.testing.assert_series_equal(out.loc["GAP"].drop("return_6m_percentile"), alone, rtol=1e-9, check_names=False)
    assert out.loc["GAP"].drop("return_6m_percentile").notna().all()


def test_missing_volume_only_blanks_volume_features():
    closes = {"A": 100 + I, "B": 100 + I}
    vols = {"A": np.full(N_ROWS, 1e6), "B": np.full(N_ROWS, 1e6)}
    vols["B"][-3] = np.nan
    out = compute_technical_features(panel_from(closes, volumes=vols), bench_series(), AS_OF)
    vol_feats = ["avg_dollar_volume_20d_usd_mn", "rel_volume_5d", "rel_volume_20d", "max_volume_ratio_20d",
                 "up_down_volume_ratio_50d"]
    assert out.loc["B", vol_feats].isna().all()
    others = [c for c in TECHNICAL_FEATURES if c not in vol_feats]
    pd.testing.assert_series_equal(out.loc["A", others], out.loc["B", others], check_names=False)


def test_missing_or_bad_high_low_fall_back_to_close():
    rng = np.random.default_rng(9)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, N_ROWS)))
    highs = {"X": np.full(N_ROWS, np.nan), "Y": c * 0.95}  # missing / below the close (bad data)
    lows = {"X": np.full(N_ROWS, np.nan), "Y": c * 1.05}
    out = compute_technical_features(panel_from({"X": c, "Y": c}, highs=highs, lows=lows), bench_series(), AS_OF)
    cs = pd.Series(c)
    close_only_atr = ind.atr_wilder(cs, cs, cs, 14).iloc[-1]  # true range = |close change|
    for t in ("X", "Y"):
        r = out.loc[t]
        assert r["high_52w"] == pytest.approx(c[-252:].max())
        assert r["low_52w"] == pytest.approx(c[-252:].min())
        assert r["drawdown_from_52w_high_pct"] <= 0 and r["above_52w_low_pct"] >= 0
        assert r["days_since_52w_high"] == float(251 - np.argmax(c[-252:]))
        assert r["atr_14_pct"] == pytest.approx(close_only_atr / c[-1] * 100)
    pd.testing.assert_series_equal(out.loc["X"], out.loc["Y"], check_names=False)


def test_non_positive_close_is_treated_as_missing_bar():
    c = 100 + I
    bad = c.copy()
    bad[150] = 0.0
    bad[151] = -3.0
    out = compute_technical_features(panel_from({"OK": c, "BAD": bad}), bench_series(), AS_OF)
    bars = np.delete(c, [150, 151])
    assert out.loc["BAD", "sma_200"] == pytest.approx(bars[-200:].mean())
    assert out.loc["BAD", "low_52w"] == pytest.approx(bars[-252:].min() * 0.99)
    # skipping two bars shifts every window by two bars: return_12m spans 254 sessions of the panel
    assert out.loc["BAD", "return_12m_pct"] == pytest.approx((399 / 145 - 1) * 100)
    assert out.loc["BAD"].drop("return_6m_percentile").notna().sum() == out.loc["OK"].drop("return_6m_percentile").notna().sum()


def test_holiday_padding_row_is_not_a_session(demo):
    prices, bench, out = demo
    idx = INDEX.append(pd.DatetimeIndex([pd.Timestamp("2026-10-01")]))
    pad = lambda f: pd.concat([f, pd.DataFrame(np.nan, index=idx[-1:], columns=f.columns)])  # noqa: E731
    padded = PricePanel(*(pad(getattr(prices, f)) for f in ("open", "high", "low", "close", "volume")))
    assert evaluation_date(padded, date(2026, 10, 1)) == pd.Timestamp(AS_OF)
    pd.testing.assert_frame_equal(compute_technical_features(padded, bench, date(2026, 10, 1)), out)


def test_benchmark_none_and_alignment(demo):
    prices, bench, out = demo
    no_bench = compute_technical_features(prices, None, AS_OF)
    bench_cols = ["rel_strength_3m_pp", "rel_strength_6m_pp", "rel_strength_12m_pp", "beta_1y"]
    assert no_bench[bench_cols].isna().all().all()
    pd.testing.assert_frame_equal(no_bench.drop(columns=bench_cols), out.drop(columns=bench_cols))
    # benchmark stamped at the close (16:00) and missing an unrelated day aligns as-of by date
    shifted = bench.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=16)
    shifted = shifted.drop(shifted.index[100])
    realigned = compute_technical_features(prices, shifted, AS_OF)
    pd.testing.assert_series_equal(realigned["rel_strength_3m_pp"], out["rel_strength_3m_pp"])


def test_empty_inputs():
    empty = panel_from({}, index=INDEX)
    out = compute_technical_features(empty, bench_series(), AS_OF)
    assert out.empty and list(out.columns) == TECHNICAL_FEATURES
    prices = panel_from({"A": 100 + I})
    early = compute_technical_features(prices, bench_series(), date(2000, 1, 1))
    assert list(early.index) == ["A"] and early.isna().all().all()
    assert average_volume_shares(prices, date(2000, 1, 1)).isna().all()


# --- event features ----------------------------------------------------------------------------


def _expected_cross(a: pd.Series, b: pd.Series, lookback: int, t: int) -> float:
    wa, wb = a.iloc[t - lookback : t + 1], b.iloc[t - lookback : t + 1]
    if wa.isna().any() or wb.isna().any():
        return math.nan
    above = (wa > wb).to_numpy()
    return float(any(above[1:] & ~above[:-1]))


def test_golden_cross_detection():
    n = 420
    idx = pd.bdate_range(end=AS_OF, periods=n)
    i = np.arange(n, dtype=float)
    close = np.where(i < 260, 200 - 0.3 * i, 200 - 0.3 * 260 + 1.2 * (i - 260))
    prices = panel_from({"GC": close}, index=idx)
    c = prices.close["GC"]
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    above = (s50 > s200).to_numpy()
    events = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    assert len(events) == 1
    e = int(events[0])
    for t in (e - 1, e, e + 10, e + 19, e + 20, e + 40):
        got = compute_technical_features(prices, None, idx[t].date()).loc["GC", "golden_cross_20d"]
        assert got == (1.0 if e <= t <= e + 19 else 0.0), t
        assert got == _expected_cross(s50, s200, 20, t)
    # fewer than 220 bars: the look-back is not fully observable -> NaN
    assert math.isnan(compute_technical_features(prices, None, idx[218].date()).loc["GC", "golden_cross_20d"])
    assert compute_technical_features(prices, None, idx[219].date()).loc["GC", "golden_cross_20d"] == 0.0


def test_macd_bullish_cross_detection():
    n = 260
    idx = pd.bdate_range(end=AS_OF, periods=n)
    close = 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 60) + 0.05 * np.arange(n)
    prices = panel_from({"SIN": close}, index=idx)
    c = prices.close["SIN"]
    line = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    line = line.where(np.arange(n) >= 25)
    sig = line.ewm(span=9, adjust=False, min_periods=9).mean()
    seen = set()
    for t in range(30, n, 3):
        got = compute_technical_features(prices, None, idx[t].date()).loc["SIN", "macd_bullish_cross_10d"]
        exp = _expected_cross(line, sig, 10, t)
        assert (math.isnan(got) and math.isnan(exp)) or got == exp, t
        seen.add(None if math.isnan(got) else got)
    assert {None, 0.0, 1.0} <= seen  # warm-up, no recent cross and recent cross all exercised


# --- properties / helpers / performance -------------------------------------------------------


def test_invariants_on_random_panel():
    rng = np.random.default_rng(123)
    n, k = 320, 40
    idx = pd.bdate_range(end=AS_OF, periods=n)
    c = 30 * np.exp(np.cumsum(rng.normal(0, 0.03, (n, k)), axis=0))
    c[rng.random((n, k)) < 0.01] = np.nan
    cols = [f"S{j}" for j in range(k)]
    close = pd.DataFrame(c, idx, cols)
    high = close * rng.uniform(0.97, 1.05, (n, k))  # deliberately sometimes below the close
    low = close * rng.uniform(0.95, 1.03, (n, k))
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, (n, k)), idx, cols)
    out = compute_technical_features(PricePanel(close, high, low, close, vol), bench_series(idx), AS_OF)
    assert (out["drawdown_from_52w_high_pct"].dropna() <= 0).all()
    assert (out["above_52w_low_pct"].dropna() >= 0).all()
    assert out["rsi_14"].dropna().between(0, 100).all()
    assert out["return_6m_percentile"].dropna().between(0, 100).all()
    for b in BOOL_FEATURES:
        assert out[b].dropna().isin([0.0, 1.0]).all()
    assert (out["days_since_52w_high"].dropna().between(0, 251)).all()
    traded = close.iloc[-1].notna()
    assert out.loc[~traded.to_numpy()].isna().all().all()


def test_average_volume_shares(demo):
    prices, _, _ = demo
    avg = average_volume_shares(prices, AS_OF)
    assert avg.index.name == "ticker" and list(avg.index) == prices.tickers
    assert avg.name == "avg_volume_20d_shares"
    assert avg["LIN"] == pytest.approx(22e6 / 20)
    assert avg["IPO"] == pytest.approx(1e6)
    assert math.isnan(avg["HALT"]) and math.isnan(avg["DEAD"])
    assert average_volume_shares(prices, AS_OF, window=5)["LIN"] == pytest.approx(7e6 / 5)
    assert math.isnan(average_volume_shares(prices, AS_OF, window=40)["IPO"])
    with pytest.raises(ValueError):
        average_volume_shares(prices, AS_OF, window=0)


def test_performance_3000_tickers_520_sessions():
    rng = np.random.default_rng(0)
    n, k = 520, 3000
    idx = pd.bdate_range(end=AS_OF, periods=n)
    cols = [f"T{j:04d}" for j in range(k)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n, k)), axis=0)), idx, cols)
    close.iloc[:200, :300] = np.nan  # recent listings
    close.iloc[300, 300:330] = np.nan  # a few interior gaps
    prices = PricePanel(close, close * 1.01, close * 0.99, close, close * 0 + 5e5)
    bench = pd.Series(4000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), idx)
    compute_technical_features(prices.subset(cols[:50]), bench, AS_OF)  # warm-up imports
    timings = []
    for _ in range(2):  # best of two filters scheduler noise on shared CI machines
        start = time.perf_counter()
        out = compute_technical_features(prices, bench, AS_OF)
        timings.append(time.perf_counter() - start)
    assert out.shape == (k, len(TECHNICAL_FEATURES))
    assert out["rsi_14"].notna().sum() > 2500
    assert min(timings) < 3.0, f"took {timings}"


def test_canonical_demo_pullback_pattern_passes_technical_legs():
    """An established uptrend followed by a heavy-volume 15-40% pullback satisfies the technical
    conditions of the canonical demo spec; a steady compounder does not."""
    n = 320
    idx = pd.bdate_range(end=AS_OF, periods=n)
    rng = np.random.default_rng(2026)
    up = 40 * np.exp(np.cumsum(np.full(250, 0.003) + rng.normal(0, 0.004, 250)))
    down = up[-1] * np.exp(np.cumsum(np.full(70, -0.0042) + rng.normal(0, 0.006, 70)))
    pull = np.r_[up, down]
    vol = np.full(n, 1e6)
    vol[-12] = 3.2e6  # capitulation day
    prices = panel_from({"PULL": pull, "STEADY": 40 * 1.003 ** np.arange(n)}, index=idx,
                        volumes={"PULL": vol, "STEADY": np.full(n, 1e6)})
    out = compute_technical_features(prices, bench_series(idx), AS_OF)
    r = out.loc["PULL"]
    assert r["sma_50_vs_sma_200_pct"] > 0
    assert r["return_12m_ex_1m_pct"] > 0
    assert -40 <= r["drawdown_from_52w_high_pct"] <= -15
    assert r["max_volume_ratio_20d"] >= 2
    assert r["rsi_14"] < 40
    s = out.loc["STEADY"]
    assert not (-40 <= s["drawdown_from_52w_high_pct"] <= -15) and s["rsi_14"] == 100.0
