"""Tests for aitrading.technical.indicators: reference values, hand-computed cases, properties."""

from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from aitrading.technical import indicators as ind

# StockCharts "RSI" ChartSchool worked example (14-period, closes and published RSI values).
STOCKCHARTS_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433, 46.0826,
    45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028, 46.0328, 46.4116, 46.2222, 45.6439,
    46.2122, 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672,
    43.4205, 42.6628, 43.1314,
]
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42, 39.99,
    41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]


def _walk(n: int, cols: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.02, size=(n, cols))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), columns=[f"c{i}" for i in range(cols)])


def _wilder_reference(x: list[float], n: int) -> list[float]:
    """Plain-Python Wilder average of a gap-free sequence."""
    out = [math.nan] * len(x)
    if len(x) < n:
        return out
    avg = sum(x[:n]) / n
    out[n - 1] = avg
    for i in range(n, len(x)):
        avg = (avg * (n - 1) + x[i]) / n
        out[i] = avg
    return out


def _rsi_reference(close: list[float], n: int = 14) -> list[float]:
    changes = [close[i] - close[i - 1] for i in range(1, len(close))]
    g = _wilder_reference([max(c, 0.0) for c in changes], n)
    lo = _wilder_reference([max(-c, 0.0) for c in changes], n)
    out = [math.nan]
    for a, b in zip(g, lo):
        out.append(math.nan if math.isnan(a) or a + b == 0 else 100 * a / (a + b))
    return out


# --- SMA / EMA ---------------------------------------------------------------------------------


def test_sma_hand_computed():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.sma(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_sma_dataframe_columnwise_and_matches_pandas():
    df = _walk(300, cols=4)
    df.iloc[100, 1] = np.nan
    out = ind.sma(df, 20)
    assert out.shape == df.shape and out.index.equals(df.index) and out.columns.equals(df.columns)
    expected = df.rolling(20, min_periods=20).mean()
    pd.testing.assert_frame_equal(out, expected, rtol=1e-12, atol=1e-12)
    # a missing value blanks every window containing it
    assert out["c1"].iloc[100:120].isna().all()
    assert out["c1"].iloc[120:].notna().all()


def test_sma_window_validation_and_short_input():
    with pytest.raises(ValueError):
        ind.sma(pd.Series([1.0, 2.0]), 0)
    with pytest.raises(ValueError):
        ind.sma(pd.Series([1.0, 2.0]), 2.5)  # type: ignore[arg-type]
    assert ind.sma(pd.Series([1.0, 2.0]), 5).isna().all()
    assert ind.sma(pd.Series([], dtype=float), 3).empty


def test_ema_hand_computed():
    # span 3 -> alpha 0.5, seeded with the first value
    out = ind.ema(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert out.tolist() == pytest.approx([1.0, 1.5, 2.25, 3.125])


def test_ema_matches_pandas_and_min_periods():
    df = _walk(200)
    pd.testing.assert_frame_equal(ind.ema(df, 12), df.ewm(span=12, adjust=False).mean(), rtol=1e-12)
    out = ind.ema(df, 12, min_periods=12)
    assert out.iloc[:11].isna().all().all() and out.iloc[11:].notna().all().all()


def test_ema_skips_missing_rows():
    out = ind.ema(pd.Series([np.nan, 1.0, 2.0, np.nan, 4.0]), 3)
    assert math.isnan(out.iloc[0]) and math.isnan(out.iloc[3])
    assert out.iloc[[1, 2, 4]].tolist() == pytest.approx([1.0, 1.5, 2.75])


def test_wilder_smooth_hand_computed():
    out = ind.wilder_smooth(pd.Series([2.0, 3.0, 4.0, 5.0, 6.0]), 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == pytest.approx([3.0, 11 / 3, 40 / 9])


# --- RSI ---------------------------------------------------------------------------------------


def test_rsi_stockcharts_worked_example():
    rsi = ind.rsi_wilder(pd.Series(STOCKCHARTS_CLOSES), 14)
    assert rsi.iloc[:14].isna().all()  # needs 14 changes = 15 closes
    assert rsi.iloc[14:].round(2).tolist() == pytest.approx(STOCKCHARTS_RSI, abs=1e-9)


def test_rsi_matches_reference_loop_and_is_bounded():
    df = _walk(400, cols=5, seed=11)
    rsi = ind.rsi_wilder(df, 14)
    for col in df:
        ref = _rsi_reference(df[col].tolist(), 14)
        np.testing.assert_allclose(rsi[col].to_numpy(), ref, rtol=1e-9, atol=1e-9, equal_nan=True)
    vals = rsi.to_numpy()[~np.isnan(rsi.to_numpy())]
    assert (vals >= 0).all() and (vals <= 100).all()


def test_rsi_edge_series():
    up = ind.rsi_wilder(pd.Series(np.arange(1.0, 41.0)), 14)
    assert (up.iloc[14:] == 100.0).all()
    down = ind.rsi_wilder(pd.Series(np.arange(40.0, 0.0, -1.0)), 14)
    assert (down.iloc[14:] == 0.0).all()
    # constant series: average gain and loss are both zero -> undefined (NaN), by design
    assert ind.rsi_wilder(pd.Series([10.0] * 30), 14).isna().all()
    # insufficient history
    assert ind.rsi_wilder(pd.Series(np.arange(1.0, 15.0)), 14).isna().all()


def test_rsi_dataframe_with_different_start_dates():
    df = _walk(120, cols=3, seed=3)
    df.iloc[:40, 1] = np.nan  # late listing
    out = ind.rsi_wilder(df, 14)
    late = ind.rsi_wilder(df["c1"].iloc[40:], 14)
    pd.testing.assert_series_equal(out["c1"].iloc[40:], late, rtol=1e-12)
    assert out["c1"].iloc[: 40 + 14].isna().all()


def test_rsi_missing_close_is_nan_and_series_continues():
    s = pd.Series(STOCKCHARTS_CLOSES)
    s.iloc[25] = np.nan
    out = ind.rsi_wilder(s, 14)
    assert math.isnan(out.iloc[25]) and math.isnan(out.iloc[26])  # change into/out of the gap missing
    assert out.iloc[27:].notna().all()


# --- MACD --------------------------------------------------------------------------------------


def test_macd_hand_computed():
    line, sig, hist = ind.macd(pd.Series([1.0, 3.0, 2.0, 4.0, 6.0]), fast=2, slow=3, signal=2)
    assert line.iloc[:2].isna().all() and sig.iloc[:3].isna().all() and hist.iloc[:3].isna().all()
    assert line.iloc[2:].tolist() == pytest.approx([1 / 9, 10 / 27, 101 / 162])
    assert sig.iloc[3:].tolist() == pytest.approx([23 / 81, 124 / 243])
    assert hist.iloc[3:].tolist() == pytest.approx([7 / 81, 55 / 486])


def test_macd_default_matches_pandas_ewm_and_warmup():
    df = _walk(150)
    line, sig, hist = ind.macd(df)
    e12 = df.ewm(span=12, adjust=False).mean()
    e26 = df.ewm(span=26, adjust=False).mean()
    exp_line = (e12 - e26).where(pd.Series(np.arange(150) >= 25, index=df.index), axis=0)
    pd.testing.assert_frame_equal(line, exp_line, rtol=1e-10, atol=1e-12)
    assert sig.iloc[:33].isna().all().all() and sig.iloc[33:].notna().all().all()
    pd.testing.assert_frame_equal(hist, line - sig)


def test_macd_validation():
    with pytest.raises(ValueError):
        ind.macd(pd.Series([1.0, 2.0]), fast=26, slow=12)


# --- true range / ATR --------------------------------------------------------------------------


def test_true_range_hand_computed():
    high = pd.Series([10.0, 12.0, 11.0, 11.0])
    low = pd.Series([8.0, 9.0, 7.0, 10.5])
    close = pd.Series([9.0, 11.0, 10.0, 9.5])
    # row 3: gap - previous close 10.0, high 11, low 10.5 -> max(0.5, 1.0, 0.5)
    assert ind.true_range(high, low, close).tolist() == pytest.approx([2.0, 3.0, 4.0, 1.0])


def test_true_range_missing_inputs():
    high = pd.Series([10.0, np.nan, 11.0])
    low = pd.Series([8.0, 9.0, 9.5])
    close = pd.Series([np.nan, 9.5, 10.0])
    tr = ind.true_range(high, low, close)
    assert tr.iloc[0] == 2.0 and math.isnan(tr.iloc[1]) and tr.iloc[2] == pytest.approx(1.5)


def test_atr_hand_computed_and_reference():
    atr = ind.atr_wilder(pd.Series([2.0, 3, 4, 5, 6]), pd.Series([0.0] * 5), pd.Series([1.0] * 5), n=3)
    assert atr.iloc[:2].isna().all()
    assert atr.iloc[2:].tolist() == pytest.approx([3.0, 11 / 3, 40 / 9])
    df = _walk(200, cols=2, seed=5)
    hi, lo = df * 1.01, df * 0.985
    out = ind.atr_wilder(hi, lo, df, 14)
    for col in df:
        tr = ind.true_range(hi[col], lo[col], df[col]).tolist()
        np.testing.assert_allclose(out[col].to_numpy(), _wilder_reference(tr, 14), rtol=1e-10, equal_nan=True)


# --- returns / volatility ----------------------------------------------------------------------


def test_log_returns_and_non_positive_prices():
    out = ind.log_returns(pd.Series([100.0, 110.0, 0.0, 50.0, -5.0, 60.0]))
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(math.log(1.1))
    assert out.iloc[2:].isna().all()


def test_realized_vol_hand_computed():
    rets = [0.01, -0.01, 0.02]
    close = pd.Series(100 * np.exp(np.cumsum([0.0, *rets])))
    out = ind.realized_vol(close, 3)
    assert out.iloc[:3].isna().all()
    assert out.iloc[3] == pytest.approx(statistics.stdev(rets) * math.sqrt(252))
    assert ind.realized_vol(close, 3, periods_per_year=1).iloc[3] == pytest.approx(statistics.stdev(rets))
    with pytest.raises(ValueError):
        ind.realized_vol(close, 1)


def test_realized_vol_matches_pandas_and_constant_is_zero():
    df = _walk(300, cols=3)
    exp = np.log(df / df.shift(1)).rolling(20, min_periods=20).std() * math.sqrt(252)
    pd.testing.assert_frame_equal(ind.realized_vol(df, 20), exp, rtol=1e-8, atol=1e-12)
    flat = ind.realized_vol(pd.Series([50.0] * 30), 20)
    assert flat.iloc[20:].tolist() == [0.0] * 10


def test_total_return():
    out = ind.total_return(pd.Series([100.0, 110.0, 121.0, 0.0, 10.0]), 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(0.21)
    assert math.isnan(out.iloc[3])  # price 0
    assert out.iloc[4] == pytest.approx(10.0 / 121.0 - 1)
    assert ind.total_return(pd.Series([0.0, 5.0]), 1).isna().all()  # zero denominator


def test_rolling_high_low():
    s = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    assert ind.rolling_high(s, 3).iloc[2:].tolist() == [4.0, 4.0, 5.0, 9.0, 9.0]
    assert ind.rolling_low(s, 3).iloc[2:].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0]
    df = _walk(100, cols=3)
    df.iloc[50, 0] = np.nan
    pd.testing.assert_frame_equal(ind.rolling_high(df, 10), df.rolling(10, min_periods=10).max())
    pd.testing.assert_frame_equal(ind.rolling_low(df, 10), df.rolling(10, min_periods=10).min())


# --- beta --------------------------------------------------------------------------------------


def test_rolling_beta_exact_linear_relation():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 300))
    assets = pd.DataFrame({"x2": 2 * bench + 0.001, "neg": -0.5 * bench, "noise": rng.normal(0, 0.01, 300)})
    beta = ind.rolling_beta(assets, bench, 60)
    assert beta.iloc[:59].isna().all().all()
    np.testing.assert_allclose(beta["x2"].iloc[59:], 2.0, rtol=1e-9)
    np.testing.assert_allclose(beta["neg"].iloc[59:], -0.5, rtol=1e-9)
    window = slice(240, 300)
    expected = np.cov(assets["noise"].iloc[window], bench.iloc[window])[0, 1] / np.var(bench.iloc[window], ddof=1)
    assert beta["noise"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_rolling_beta_series_dataframe_benchmark_and_nan():
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0, 0.01, 100))
    asset = 1.5 * bench
    assert ind.rolling_beta(asset, bench, 20).iloc[-1] == pytest.approx(1.5)
    frame = pd.DataFrame({"a": asset, "b": asset})
    per_col = pd.DataFrame({"a": bench, "b": 2 * bench})
    out = ind.rolling_beta(frame, per_col, 20)
    assert out["a"].iloc[-1] == pytest.approx(1.5) and out["b"].iloc[-1] == pytest.approx(0.75)
    gappy = asset.copy()
    gappy.iloc[90] = np.nan
    b = ind.rolling_beta(gappy, bench, 20)
    assert b.iloc[90:].isna().all() and b.iloc[89] == pytest.approx(1.5)
    # zero benchmark variance -> NaN
    assert ind.rolling_beta(asset, pd.Series([0.001] * 100), 20).isna().all()
    with pytest.raises(TypeError):
        ind.rolling_beta(asset, [0.0] * 100, 20)  # type: ignore[arg-type]


# --- crosses -----------------------------------------------------------------------------------


def test_crossed_above_within():
    a = pd.Series([1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 3.0])
    b = pd.Series([2.0] * 7)
    assert ind.crossed_above_within(a, b, 2).tolist() == [False, False, True, True, False, False, True]
    assert ind.crossed_above_within(a, 2.0, 1).tolist() == [False, False, True, False, False, False, True]
    # touching then rising counts (from <= to >); staying above never counts
    assert ind.crossed_above_within(pd.Series([2.0, 3.0]), pd.Series([2.0, 2.0]), 5).tolist() == [False, True]
    assert not ind.crossed_above_within(pd.Series([3.0, 4.0, 5.0]), 2.0, 3).any()
    # missing values never count as a crossing
    assert not ind.crossed_above_within(pd.Series([1.0, np.nan, 3.0]), 2.0, 3).any()
    with pytest.raises(ValueError):
        ind.crossed_above_within(a, b, 0)


def test_crossed_above_within_dataframe():
    a = pd.DataFrame({"x": [1.0, 3.0, 3.0], "y": [3.0, 1.0, 3.0]})
    b = pd.DataFrame({"x": [2.0, 2.0, 2.0], "y": [2.0, 2.0, 2.0]})
    out = ind.crossed_above_within(a, b, 1)
    assert out.dtypes.eq(bool).all()
    assert out["x"].tolist() == [False, True, False] and out["y"].tolist() == [False, False, True]


def test_shapes_preserved_for_series_and_frames():
    df = _walk(60)
    s = df["c0"].rename("px")
    for fn in (lambda x: ind.sma(x, 5), lambda x: ind.ema(x, 5), lambda x: ind.rsi_wilder(x, 5),
               lambda x: ind.realized_vol(x, 5), lambda x: ind.total_return(x, 5), ind.log_returns,
               lambda x: ind.atr_wilder(x * 1.01, x * 0.99, x, 5), lambda x: ind.macd(x, 3, 6, 3)[0]):
        out_s, out_df = fn(s), fn(df)
        assert isinstance(out_s, pd.Series) and out_s.index.equals(s.index) and out_s.name == "px"
        assert isinstance(out_df, pd.DataFrame) and out_df.shape == df.shape and out_df.columns.equals(df.columns)
