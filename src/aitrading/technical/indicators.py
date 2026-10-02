"""Vectorised technical indicators.

Every function accepts a ``pd.Series`` or a wide ``pd.DataFrame`` (dates x tickers, applied
column-wise) and returns the same shape, index and columns. Windows count rows; what a row is is
the caller's choice (the feature engine passes each ticker's own trading sessions, see
``aitrading.technical.features``). Ratios are returned as fractions, not percentages.

NaN conventions
---------------
* Rolling statistics need a complete window (``min_periods == n``): the first ``n - 1`` rows are
  NaN, and a missing value anywhere in the window makes the output NaN.
* Recursive averages (EMA, Wilder) skip missing inputs (the next valid value continues the
  recursion as if adjacent) and are NaN on rows whose own input is missing.
* EMA is pandas ``ewm(span, adjust=False)``: seeded with the first valid value,
  ``alpha = 2 / (span + 1)``. Wilder averages are seeded with the simple mean of the first ``n``
  consecutive valid inputs, then ``avg_t = (avg_{t-1} * (n - 1) + x_t) / n``.
* Prices <= 0 are treated as missing in return calculations (no division by zero, no log <= 0).
* RSI is NaN when both average gain and average loss are zero (a constant series has no defined
  RSI); it is exactly 100 when the average loss is zero and 0 when the average gain is zero.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

P = TypeVar("P", pd.Series, pd.DataFrame)


def _check_window(n: int, name: str = "n", minimum: int = 1) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {n!r}")


def _wrap(like: P, values: np.ndarray) -> P:
    if isinstance(like, pd.Series):
        return pd.Series(values, index=like.index, name=like.name)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def _positive(x: P) -> P:
    return x.where(x > 0)


# --- moving averages -------------------------------------------------------------------------


def sma(x: P, n: int) -> P:
    """Simple moving average over ``n`` rows (complete window required)."""
    _check_window(n)
    return x.rolling(n, min_periods=n).mean()


def ema(x: P, span: int, *, min_periods: int = 0) -> P:
    """Exponential moving average, pandas ``ewm(span, adjust=False)`` (seeded with the first value).

    ``min_periods`` masks the warm-up: output is NaN until that many valid inputs have been seen.
    """
    _check_window(span, "span")
    out = x.ewm(span=span, adjust=False, ignore_na=True, min_periods=min_periods).mean()
    return out.where(x.notna())


def wilder_smooth(x: P, n: int) -> P:
    """Wilder's running average: seed = mean of the first ``n`` consecutive valid values (placed on
    the n-th of them), then ``(prev * (n - 1) + x_t) / n``. NaN before the seed and where x is NaN."""
    _check_window(n)
    seed = x.rolling(n, min_periods=n).mean()
    started = seed.notna().cummax()
    first = started & ~started.shift(1, fill_value=False)
    y = x.where(started & ~first).mask(first, seed)
    out = y.ewm(alpha=1.0 / n, adjust=False, ignore_na=True).mean()
    return out.where(started & x.notna())


# --- oscillators -----------------------------------------------------------------------------


def rsi_wilder(close: P, n: int = 14) -> P:
    """Classic Wilder RSI (0-100). The first value appears once ``n`` price changes exist (row ``n``
    for a gap-free series): first averages are simple means of the first n gains / losses, then
    Wilder-smoothed. NaN when average gain and loss are both zero (e.g. a constant series)."""
    _check_window(n)
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), n)
    avg_loss = wilder_smooth((-delta).clip(lower=0.0), n)
    total = avg_gain + avg_loss
    return (100.0 * avg_gain / total).where(total > 0)


def macd(close: P, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[P, P, P]:
    """MACD ``(line, signal, histogram)``: line = EMA(fast) - EMA(slow) of close, signal = EMA(signal)
    of the line, histogram = line - signal. The line needs ``slow`` closes, the signal ``signal``
    line values (``slow + signal - 1`` closes); earlier rows are NaN."""
    _check_window(fast, "fast")
    _check_window(slow, "slow")
    _check_window(signal, "signal")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    line = ema(close, fast, min_periods=fast) - ema(close, slow, min_periods=slow)
    sig = ema(line, signal, min_periods=signal)
    return line, sig, line - sig


# --- range / volatility ----------------------------------------------------------------------


def true_range(high: P, low: P, close: P) -> P:
    """max(high - low, |high - prev close|, |low - prev close|); high - low when there is no
    previous close (first row, or the previous close is missing). NaN if high or low is missing."""
    prev = close.shift(1)
    hl = high - low
    hc = (high - prev).abs()
    lc = (low - prev).abs()
    tr = np.fmax(hl.to_numpy(dtype=float), np.fmax(hc.to_numpy(dtype=float), lc.to_numpy(dtype=float)))
    return _wrap(hl, tr).where(hl.notna())


def atr_wilder(high: P, low: P, close: P, n: int = 14) -> P:
    """Wilder average true range: first value = mean of the first ``n`` true ranges (the first of
    which is high - low), then Wilder-smoothed."""
    return wilder_smooth(true_range(high, low, close), n)


def log_returns(close: P) -> P:
    """ln(close_t / close_{t-1}); NaN on the first row and around non-positive / missing prices."""
    c = _positive(close)
    return np.log(c / c.shift(1))


def realized_vol(close: P, n: int, periods_per_year: int = 252) -> P:
    """Annualised sample stdev (ddof=1) of the last ``n`` log returns, as a fraction (needs n + 1
    closes)."""
    _check_window(n, minimum=2)
    return log_returns(close).rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year)


def total_return(close: P, n: int) -> P:
    """close_t / close_{t-n} - 1 as a fraction; NaN if either price is missing or <= 0."""
    _check_window(n)
    c = _positive(close)
    return c / c.shift(n) - 1.0


def rolling_high(x: P, n: int) -> P:
    """Rolling maximum over ``n`` rows (complete window required)."""
    _check_window(n)
    return x.rolling(n, min_periods=n).max()


def rolling_low(x: P, n: int) -> P:
    """Rolling minimum over ``n`` rows (complete window required)."""
    _check_window(n)
    return x.rolling(n, min_periods=n).min()


# --- regression ------------------------------------------------------------------------------


def rolling_beta(asset_returns: P, bench_returns: pd.Series | pd.DataFrame, n: int) -> P:
    """Rolling OLS beta cov(asset, bench) / var(bench) over ``n`` rows.

    ``bench_returns`` is a Series (aligned on the index and broadcast to every column) or a
    DataFrame with the same columns (column-matched, for per-ticker benchmark paths). All ``n``
    pairs must be present; NaN when the benchmark variance is (numerically) zero.
    """
    _check_window(n, minimum=2)
    a = asset_returns.to_frame() if isinstance(asset_returns, pd.Series) else asset_returns
    if isinstance(bench_returns, pd.DataFrame):
        bvals = bench_returns.reindex(index=a.index, columns=a.columns).to_numpy(dtype=float)
    elif isinstance(bench_returns, pd.Series):
        bvals = np.broadcast_to(bench_returns.reindex(a.index).to_numpy(dtype=float)[:, None], a.shape)
    else:
        raise TypeError("bench_returns must be a pandas Series or DataFrame")
    avals = a.to_numpy(dtype=float)
    valid = np.isfinite(avals) & np.isfinite(bvals)
    av = np.where(valid, avals, np.nan)
    bv = np.where(valid, bvals, np.nan)

    def rsum(v: np.ndarray) -> np.ndarray:
        return pd.DataFrame(v).rolling(n, min_periods=n).sum().to_numpy()

    sa, sb, sab, sbb = rsum(av), rsum(bv), rsum(av * bv), rsum(bv * bv)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        var = sbb - sb * sb / n
        beta = np.where(var > 1e-12 * np.maximum(sbb, np.finfo(float).tiny), cov / var, np.nan)
    if isinstance(asset_returns, pd.Series):
        return pd.Series(beta[:, 0], index=asset_returns.index, name=asset_returns.name)
    return pd.DataFrame(beta, index=a.index, columns=a.columns)


# --- events ----------------------------------------------------------------------------------


def crossed_above_within(a: P, b: P | float, lookback: int) -> P:
    """Boolean: ``a`` crossed from <= ``b`` (previous row) to > ``b`` (this row) on any of the last
    ``lookback`` rows, including the current one. ``b`` may be a same-shaped object or a scalar.
    A crossing needs both rows' values present; missing data never counts as a crossing (the caller
    decides whether the look-back was fully observable)."""
    _check_window(lookback, "lookback")
    if np.isscalar(b):
        above = a > b
        prev_at_or_below = a.shift(1) <= b
    else:
        above = a > b
        prev_at_or_below = a.shift(1) <= b.shift(1)
    event = (above & prev_at_or_below).astype(float)
    return event.rolling(lookback, min_periods=1).max() > 0
