"""Vectorised technical indicators.

Every function accepts a ``pd.Series`` or a wide ``pd.DataFrame`` (dates x tickers, applied
column-wise) and returns the same shape, index and columns as float. Windows count rows; what a row
is is the caller's choice (the feature engine passes each ticker's own trading sessions, see
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

Implementation: rolling statistics run as a single pandas (Cython) pass over the columns laid end
to end with ``n`` NaN rows between them, so windows never mix tickers; recursions loop over rows
with numpy operations across all columns. Both are O(rows x columns) and wide-frame friendly.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import pandas as pd

P = TypeVar("P", pd.Series, pd.DataFrame)
_Stat = Literal["sum", "mean", "std", "max", "min"]


# --- array kernels (2-D float arrays, rows x columns) ----------------------------------------


def _check_window(n: int, name: str = "n", minimum: int = 1) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {n!r}")


def _arr(x: pd.Series | pd.DataFrame) -> np.ndarray:
    a = x.to_numpy(dtype="float64", na_value=np.nan)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _like(x: P, a: np.ndarray) -> P:
    if isinstance(x, pd.Series):
        return pd.Series(a[:, 0], index=x.index, name=x.name)
    return pd.DataFrame(a, index=x.index, columns=x.columns)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    if k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


def _window_diff(cum: np.ndarray, n: int) -> np.ndarray:
    """Sums over windows of n rows ending at rows n-1 .. t-1, from a cumulative sum with a zero row."""
    return cum[n:] - cum[:-n]


def _rolling(a: np.ndarray, n: int, stat: _Stat) -> np.ndarray:
    """Complete-window rolling statistic of every column (std is the sample stdev, ddof=1).

    sum / mean / std use cumulative sums of values centred on each column's first valid value
    (keeps the running sums small); a window holding any non-finite value is NaN. max / min run as
    one pandas pass over the columns laid end to end, separated by n NaN rows.
    """
    t, k = a.shape
    out = np.full((t, k), np.nan)
    if t < n or k == 0:
        return out
    finite = np.isfinite(a)
    if stat in ("max", "min"):
        padded = np.full((t + n, k), np.nan)
        padded[n:] = np.where(finite, a, np.nan)
        r = getattr(pd.Series(padded.T.ravel()).rolling(n, min_periods=n), stat)()
        return r.to_numpy().reshape(k, t + n).T[n:].copy()
    all_finite = bool(finite.all())
    first = np.argmax(finite, axis=0)
    centre = np.where(finite.any(axis=0), a[first, np.arange(k)], 0.0)
    x = a - centre
    if not all_finite:
        x[~finite] = 0.0
    cum = np.zeros((t + 1, k))
    np.cumsum(x, axis=0, out=cum[1:])
    s1 = _window_diff(cum, n)
    if stat == "sum":
        res = s1 + n * centre
    elif stat == "mean":
        res = s1 / n + centre
    elif stat == "std":
        if n < 2:
            raise ValueError("std needs a window of at least 2")
        np.cumsum(x * x, axis=0, out=cum[1:])
        var = (_window_diff(cum, n) - s1 * s1 / n) / (n - 1)
        res = np.sqrt(np.maximum(var, 0.0))
    else:  # pragma: no cover - guarded by _Stat
        raise ValueError(f"unknown rolling statistic {stat!r}")
    if not all_finite:
        bad = np.zeros((t + 1, k), dtype=np.int32)
        np.cumsum(~finite, axis=0, out=bad[1:])
        res[_window_diff(bad, n) > 0] = np.nan
    out[n - 1 :] = res
    return out


def _ewm(a: np.ndarray, alpha: float, min_periods: int = 1) -> np.ndarray:
    """``s_t = (1 - alpha) s_prev + alpha x_t`` seeded with the first valid value, skipping missing
    rows; NaN on missing rows and until ``min_periods`` valid values have been seen."""
    t, k = a.shape
    out = np.empty((t, k))
    ok = np.isfinite(a)
    row_ok = ok.all(axis=1)
    state = np.full(k, np.nan)
    seen = np.zeros(k, dtype=np.int64)
    keep = 1.0 - alpha
    ready = False  # every column seeded and warmed up: complete rows take the cheap update
    for i in range(t):
        x = a[i]
        if ready and row_ok[i]:
            state *= keep
            state += alpha * x
            out[i] = state
            continue
        new = keep * state + alpha * x
        fresh = ok[i] & np.isnan(state)
        new[fresh] = x[fresh]
        state = np.where(ok[i], new, state)
        seen += ok[i]
        warm = seen >= min_periods
        out[i] = np.where(ok[i] & warm, state, np.nan)
        ready = bool(warm.all()) and not np.isnan(state).any()
    return out


def _wilder(a: np.ndarray, n: int) -> np.ndarray:
    seed = _rolling(a, n, "mean")
    started = np.maximum.accumulate(np.isfinite(seed), axis=0)
    first = started.copy()
    first[1:] &= ~started[:-1]
    y = np.where(started & ~first, a, np.nan)
    y[first] = seed[first]
    return np.where(started & np.isfinite(a), _ewm(y, 1.0 / n), np.nan)


# --- moving averages -------------------------------------------------------------------------


def sma(x: P, n: int) -> P:
    """Simple moving average over ``n`` rows (complete window required)."""
    _check_window(n)
    return _like(x, _rolling(_arr(x), n, "mean"))


def ema(x: P, span: int, *, min_periods: int = 0) -> P:
    """Exponential moving average, pandas ``ewm(span, adjust=False)`` (seeded with the first value).

    ``min_periods`` masks the warm-up: output is NaN until that many valid inputs have been seen.
    """
    _check_window(span, "span")
    return _like(x, _ewm(_arr(x), 2.0 / (span + 1.0), max(int(min_periods), 1)))


def wilder_smooth(x: P, n: int) -> P:
    """Wilder's running average: seed = mean of the first ``n`` consecutive valid values (placed on
    the n-th of them), then ``(prev * (n - 1) + x_t) / n``. NaN before the seed and where x is NaN."""
    _check_window(n)
    return _like(x, _wilder(_arr(x), n))


# --- oscillators -----------------------------------------------------------------------------


def rsi_wilder(close: P, n: int = 14) -> P:
    """Classic Wilder RSI (0-100). The first value appears once ``n`` price changes exist (row ``n``
    for a gap-free series): first averages are simple means of the first n gains / losses, then
    Wilder-smoothed. NaN when average gain and loss are both zero (e.g. a constant series)."""
    _check_window(n)
    c = _arr(close)
    delta = c - _shift(c, 1)
    avg_gain = _wilder(np.clip(delta, 0.0, None), n)
    avg_loss = _wilder(np.clip(-delta, 0.0, None), n)
    total = avg_gain + avg_loss
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(total > 0, 100.0 * avg_gain / total, np.nan)
    return _like(close, rsi)


def macd(close: P, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[P, P, P]:
    """MACD ``(line, signal, histogram)``: line = EMA(fast) - EMA(slow) of close, signal = EMA(signal)
    of the line, histogram = line - signal. The line needs ``slow`` closes, the signal ``signal``
    line values (``slow + signal - 1`` closes); earlier rows are NaN."""
    _check_window(fast, "fast")
    _check_window(slow, "slow")
    _check_window(signal, "signal")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    c = _arr(close)
    line = _ewm(c, 2.0 / (fast + 1.0), fast) - _ewm(c, 2.0 / (slow + 1.0), slow)
    sig = _ewm(line, 2.0 / (signal + 1.0), signal)
    return _like(close, line), _like(close, sig), _like(close, line - sig)


# --- range / volatility ----------------------------------------------------------------------


def _true_range(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev = _shift(c, 1)
    hl = h - lo
    tr = np.fmax(hl, np.fmax(np.abs(h - prev), np.abs(lo - prev)))
    return np.where(np.isnan(hl), np.nan, tr)


def true_range(high: P, low: P, close: P) -> P:
    """max(high - low, |high - prev close|, |low - prev close|); high - low when there is no
    previous close (first row, or the previous close is missing). NaN if high or low is missing."""
    return _like(high, _true_range(_arr(high), _arr(low), _arr(close)))


def atr_wilder(high: P, low: P, close: P, n: int = 14) -> P:
    """Wilder average true range: first value = mean of the first ``n`` true ranges (the first of
    which is high - low), then Wilder-smoothed."""
    _check_window(n)
    return _like(high, _wilder(_true_range(_arr(high), _arr(low), _arr(close)), n))


def _positive(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(a) & (a > 0), a, np.nan)


def log_returns(close: P) -> P:
    """ln(close_t / close_{t-1}); NaN on the first row and around non-positive / missing prices."""
    c = _positive(_arr(close))
    return _like(close, np.log(c / _shift(c, 1)))


def realized_vol(close: P, n: int, periods_per_year: int = 252) -> P:
    """Annualised sample stdev (ddof=1) of the last ``n`` log returns, as a fraction (needs n + 1
    closes)."""
    _check_window(n, minimum=2)
    c = _positive(_arr(close))
    return _like(close, _rolling(np.log(c / _shift(c, 1)), n, "std") * np.sqrt(periods_per_year))


def total_return(close: P, n: int) -> P:
    """close_t / close_{t-n} - 1 as a fraction; NaN if either price is missing or <= 0."""
    _check_window(n)
    c = _positive(_arr(close))
    return _like(close, c / _shift(c, n) - 1.0)


def rolling_high(x: P, n: int) -> P:
    """Rolling maximum over ``n`` rows (complete window required)."""
    _check_window(n)
    return _like(x, _rolling(_arr(x), n, "max"))


def rolling_low(x: P, n: int) -> P:
    """Rolling minimum over ``n`` rows (complete window required)."""
    _check_window(n)
    return _like(x, _rolling(_arr(x), n, "min"))


# --- regression ------------------------------------------------------------------------------


def rolling_beta(asset_returns: P, bench_returns: pd.Series | pd.DataFrame, n: int) -> P:
    """Rolling OLS beta cov(asset, bench) / var(bench) over ``n`` rows.

    ``bench_returns`` is a Series (aligned on the index and broadcast to every column) or a
    DataFrame with the same columns (column-matched, for per-ticker benchmark paths). All ``n``
    pairs must be present; NaN when the benchmark variance is (numerically) zero.
    """
    _check_window(n, minimum=2)
    a = _arr(asset_returns)
    if isinstance(bench_returns, pd.DataFrame):
        if isinstance(asset_returns, pd.Series):
            raise TypeError("a DataFrame benchmark needs DataFrame asset returns")
        b = _arr(bench_returns.reindex(index=asset_returns.index, columns=asset_returns.columns))
    elif isinstance(bench_returns, pd.Series):
        b = np.broadcast_to(_arr(bench_returns.reindex(asset_returns.index)), a.shape)
    else:
        raise TypeError("bench_returns must be a pandas Series or DataFrame")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(valid, a, np.nan), np.where(valid, b, np.nan)
    sa, sb = _rolling(a, n, "sum"), _rolling(b, n, "sum")
    sab, sbb = _rolling(a * b, n, "sum"), _rolling(b * b, n, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        var = sbb - sb * sb / n
        beta = np.where(var > 1e-12 * np.maximum(sbb, np.finfo(float).tiny), cov / var, np.nan)
    return _like(asset_returns, beta)


# --- events ----------------------------------------------------------------------------------


def crossed_above_within(a: P, b: P | float, lookback: int) -> P:
    """Boolean: ``a`` crossed from <= ``b`` (previous row) to > ``b`` (this row) on any of the last
    ``lookback`` rows, including the current one. ``b`` may be a same-shaped object or a scalar.
    A crossing needs both rows' values present; missing data never counts as a crossing (the caller
    decides whether the look-back was fully observable)."""
    _check_window(lookback, "lookback")
    av = _arr(a)
    bv = np.full(av.shape, float(b)) if np.isscalar(b) else _arr(b)
    with np.errstate(invalid="ignore"):
        event = (av > bv) & (_shift(av, 1) <= _shift(bv, 1))
    hits = np.cumsum(event, axis=0)
    lagged = np.zeros_like(hits)
    if lookback < hits.shape[0]:
        lagged[lookback:] = hits[:-lookback]
    within = hits - lagged > 0
    if isinstance(a, pd.Series):
        return pd.Series(within[:, 0], index=a.index, name=a.name)
    return pd.DataFrame(within, index=a.index, columns=a.columns)
