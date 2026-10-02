"""Technical feature engine: PricePanel -> one row per ticker of the catalog's technical features.

Conventions
-----------
* Look-ahead safety: the panel and the benchmark are cut to dates <= ``as_of`` before anything is
  computed. Features are evaluated on the *evaluation row*: the last session on or before as_of.
  A panel row on which no ticker has a valid close (holiday / padding row) is not a session.
* Sessions are each ticker's own bars: rows where its close is a positive finite number. Interior
  gaps (halts, missing vendor prints) are skipped, as charting packages draw bars, so one missing
  print does not blank a 200-session window. With gap-free data a bar is simply a panel row.
* A ticker without a valid close on the evaluation row did not trade that day: every feature is NaN
  (nothing is evaluated on a stale price) and it is left out of the return_6m percentile.
* Windows need complete inputs: fewer bars than a definition needs, or a missing volume inside a
  window that uses volume, gives NaN. Ratios with a non-positive denominator are NaN.
* Bar high / low are sanitised to fmax(high, close) / fmin(low, close) (a missing high or low falls
  back to the close), so high_52w >= price >= low_52w and the drawdown is always <= 0.
* The benchmark is aligned as-of to the panel dates (last level on or before each date). Relative
  strength and beta compare a ticker with the benchmark over the same dates (the ticker's bars).
* Returns are simple close-to-close returns (beta uses daily simple returns); volatility uses log
  returns (sample stdev, annualised with 252).
* Bool features are 1.0 / 0.0, NaN when the inputs over the whole look-back are not available.
* return_6m_percentile = (rank - 1) / (N - 1) x 100 over the tickers with a valid return_6m_pct
  (ties share the average rank; 50 when N == 1): 0 = weakest, 100 = strongest.

Bars needed (gap-free data): price 1; returns n + 1 (12-1 momentum 253); sma_n n; sma_200 slope
221; golden cross 220; rsi_14 15; macd line 26, signal 34, bullish cross 44; 52w range 252;
volatility n + 1; atr_14 14; beta 253; volume ratios 60 / 120; up/down volume 51.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from aitrading.data.base import PricePanel
from aitrading.screen.catalog import TECHNICAL_FEATURES
from aitrading.technical import indicators as ind

W_1M, W_3M, W_6M, W_12M = 21, 63, 126, 252
AsOf = date | pd.Timestamp | str


@dataclass(frozen=True)
class _Bars:
    """Each ticker's bars right-aligned so that row -1 is its latest bar (NaN-padded on top)."""

    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    bench: np.ndarray  # benchmark level on each of the ticker's bar dates
    traded: np.ndarray  # (N,) bool: valid close on the evaluation row

    @property
    def length(self) -> int:
        return self.close.shape[0]


def _dates(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _rows_upto(index: pd.Index, as_of: AsOf) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Positions of rows dated <= as_of in ascending date order (last duplicate date wins)."""
    dates = _dates(index)
    pos = np.flatnonzero(dates <= pd.Timestamp(as_of).normalize())
    if not dates[pos].is_monotonic_increasing:
        pos = pos[np.argsort(dates[pos], kind="stable")]
    d = dates[pos]
    if len(d) > 1:
        pos = pos[np.r_[d[1:] != d[:-1], True]]
    return pos, dates[pos]


def _values(frame: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    a = frame.to_numpy(dtype="float64", na_value=np.nan)
    if len(pos) and pos[-1] - pos[0] + 1 == len(pos):  # contiguous ascending rows: a view
        return a[pos[0] : pos[-1] + 1]
    return a[pos]


def _sessions(prices: PricePanel, as_of: AsOf) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Panel rows that are sessions on or before as_of: ``(positions, dates, close, valid)``.

    A row on which no ticker has a valid close (holiday / padding row) is not a session.
    """
    pos, dates = _rows_upto(prices.close.index, as_of)
    close = _values(prices.close, pos)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(close) & (close > 0)
    if close.shape[1]:
        session = valid.any(axis=1)
        if not session.all():
            pos, dates, close, valid = pos[session], dates[session], close[session], valid[session]
    return pos, dates, close, valid


def evaluation_date(prices: PricePanel, as_of: AsOf) -> pd.Timestamp | None:
    """The session features are evaluated on (last session <= as_of), or None if there is none."""
    _, dates, _, _ = _sessions(prices, as_of)
    return dates[-1] if len(dates) else None


def _aligned_benchmark(benchmark: pd.Series | None, dates: pd.DatetimeIndex, as_of: AsOf) -> np.ndarray:
    if benchmark is None or len(benchmark) == 0 or len(dates) == 0:
        return np.full(len(dates), np.nan)
    b = pd.to_numeric(benchmark, errors="coerce")
    pos, bdates = _rows_upto(b.index, as_of)
    vals = b.to_numpy(dtype="float64", na_value=np.nan)[pos]
    ok = np.isfinite(vals) & (vals > 0)
    b = pd.Series(vals[ok], index=bdates[ok])
    if b.empty:
        return np.full(len(dates), np.nan)
    return b.reindex(b.index.union(dates)).ffill().reindex(dates).to_numpy(dtype=float)


def _bar_aligner(valid: np.ndarray):
    """Function moving each column's valid-bar values to the bottom rows, keeping their order."""
    if valid.all():
        return lambda a: a
    order = np.argsort(valid, axis=0, kind="stable")  # missing rows first, then bars in date order
    padding = np.arange(valid.shape[0])[:, None] < (~valid).sum(axis=0)[None, :]

    def align(a: np.ndarray) -> np.ndarray:
        out = np.take_along_axis(np.asarray(a, dtype=float), order, axis=0)
        out[padding] = np.nan
        return out

    return align


def _bars(prices: PricePanel, benchmark: pd.Series | None, as_of: AsOf) -> _Bars:
    pos, dates, close, valid = _sessions(prices, as_of)
    n = close.shape[1]
    if close.shape[0] == 0:
        empty = np.empty((0, n))
        return _Bars(empty, empty, empty, empty, empty, np.zeros(n, dtype=bool))
    with np.errstate(invalid="ignore"):
        close = np.where(valid, close, np.nan)
        high = np.fmax(_values(prices.high, pos), close)
        low = np.fmin(_values(prices.low, pos), close)
        vol = _values(prices.volume, pos)
        vol = np.where(np.isfinite(vol) & (vol >= 0), vol, np.nan)
    bench = np.broadcast_to(_aligned_benchmark(benchmark, dates, as_of)[:, None], close.shape)
    align = _bar_aligner(valid)
    return _Bars(
        close=align(close),
        high=align(high),
        low=align(low),
        volume=align(vol),
        bench=align(bench),
        traded=valid[-1].copy(),
    )


# --- small numpy helpers on right-aligned bars (row -1 = latest bar) -------------------------


def _row(a: np.ndarray, lag: int) -> np.ndarray:
    """Row ``lag`` bars before the latest (NaN if the panel is too short)."""
    if lag >= a.shape[0]:
        return np.full(a.shape[1], np.nan)
    return a[-1 - lag]


def _tail(a: np.ndarray, n: int, lag: int = 0) -> np.ndarray | None:
    """The n bars ending ``lag`` bars before the latest, or None if the panel is too short."""
    if n + lag > a.shape[0]:
        return None
    return a[a.shape[0] - lag - n : a.shape[0] - lag]


def _tail_mean(a: np.ndarray, n: int) -> np.ndarray:
    w = _tail(a, n)
    return np.full(a.shape[1], np.nan) if w is None else w.mean(axis=0)


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den where den > 0 and both finite, else NaN."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    np.divide(num, den, out=out, where=ok)
    return out


def _pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """(num / den - 1) x 100 with the _div guards."""
    return (_div(num, den) - 1.0) * 100.0


def _last(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float)[-1] if len(frame) else np.full(frame.shape[1], np.nan)


def _cross_flag(a: pd.DataFrame, b: pd.DataFrame, lookback: int) -> np.ndarray:
    """1.0 / 0.0 if ``a`` crossed above ``b`` within ``lookback`` bars; NaN unless both series are
    available on all lookback + 1 bars (every possible crossing was observable)."""
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    wa, wb = _tail(av, lookback + 1), _tail(bv, lookback + 1)
    if wa is None:
        return np.full(av.shape[1], np.nan)
    observable = np.isfinite(wa).all(axis=0) & np.isfinite(wb).all(axis=0)
    crossed = _last(ind.crossed_above_within(a, b, lookback).astype(float))
    return np.where(observable, crossed, np.nan)


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    """(rank - 1) / (N - 1) x 100 among the non-NaN values (average rank for ties, 50 when N == 1);
    NaN stays NaN."""
    ranks = values.rank(method="average")
    n = int(values.notna().sum())
    if n <= 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=values.index, name=values.name)
    return (ranks - 1.0) / (n - 1.0) * 100.0


# --- public API ------------------------------------------------------------------------------


def compute_technical_features(prices: PricePanel, benchmark: pd.Series | None, as_of: AsOf) -> pd.DataFrame:
    """Catalog technical features for every ticker in ``prices``, evaluated on the last row <= as_of.

    Returns a float frame indexed by ``ticker`` (panel column order) with exactly the columns
    ``aitrading.screen.catalog.TECHNICAL_FEATURES``, in that order. See the module docstring for
    the session, NaN and alignment conventions.
    """
    tickers = pd.Index(list(prices.close.columns), name="ticker")
    bars = _bars(prices, benchmark, as_of)
    n = len(tickers)
    nan = np.full(n, np.nan)
    if bars.length == 0 or n == 0:
        return pd.DataFrame(np.nan, index=tickers, columns=TECHNICAL_FEATURES, dtype="float64")

    cols = pd.RangeIndex(n)
    frame = lambda a: pd.DataFrame(a, columns=cols)  # noqa: E731
    C, H, L, V, B = bars.close, bars.high, bars.low, bars.volume, bars.bench
    Cdf = frame(C)
    price = _row(C, 0)
    f: dict[str, np.ndarray] = {"price": price}

    with np.errstate(invalid="ignore", divide="ignore"):
        wc, wv = _tail(C, 20), _tail(V, 20)
        f["avg_dollar_volume_20d_usd_mn"] = nan if wc is None else (wc * wv).mean(axis=0) / 1e6

        # trend
        recent = frame(_tail_or_all(C, 200 + W_1M))  # sma_200 today and 21 bars ago
        s50, s200 = ind.sma(recent, 50), ind.sma(recent, 200)
        f["sma_20"] = _tail_mean(C, 20)
        f["sma_50"] = _last(s50)
        f["sma_200"] = _last(s200)
        f["price_vs_sma_50_pct"] = _pct(price, f["sma_50"])
        f["price_vs_sma_200_pct"] = _pct(price, f["sma_200"])
        f["sma_50_vs_sma_200_pct"] = _pct(f["sma_50"], f["sma_200"])
        f["sma_200_slope_1m_pct"] = _pct(f["sma_200"], _row(s200.to_numpy(dtype=float), W_1M))
        f["golden_cross_20d"] = _cross_flag(s50, s200, 20)

        # momentum and relative strength (benchmark over the ticker's own bar dates)
        bench_now = _row(B, 0)
        for label, w in (("1m", W_1M), ("3m", W_3M), ("6m", W_6M), ("12m", W_12M)):
            f[f"return_{label}_pct"] = _pct(price, _row(C, w))
            if label != "1m":
                f[f"rel_strength_{label}_pp"] = f[f"return_{label}_pct"] - _pct(bench_now, _row(B, w))
        f["return_12m_ex_1m_pct"] = _pct(_row(C, W_1M), _row(C, W_12M))

        # oscillators
        f["rsi_14"] = _last(ind.rsi_wilder(Cdf, 14))
        line, sig, hist = ind.macd(Cdf, 12, 26, 9)
        f["macd_line"], f["macd_signal"], f["macd_histogram"] = _last(line), _last(sig), _last(hist)
        f["macd_histogram_pct_price"] = _div(f["macd_histogram"], price) * 100.0
        f["macd_bullish_cross_10d"] = _cross_flag(line, sig, 10)

        # range / drawdown
        wh, wl = _tail(H, W_12M), _tail(L, W_12M)
        if wh is None:
            f["high_52w"] = f["low_52w"] = f["days_since_52w_high"] = nan
        else:
            hi = wh.max(axis=0)  # NaN propagates: complete window required
            f["high_52w"], f["low_52w"] = hi, wl.min(axis=0)
            latest_at_high = np.argmax(wh[::-1] == hi[None, :], axis=0).astype(float)
            f["days_since_52w_high"] = np.where(np.isfinite(hi), latest_at_high, np.nan)
        f["drawdown_from_52w_high_pct"] = _pct(price, f["high_52w"])
        f["above_52w_low_pct"] = _pct(price, f["low_52w"])

        # volatility
        f["volatility_20d_pct"] = _last(ind.realized_vol(frame(_tail_or_all(C, 21)), 20)) * 100.0
        f["volatility_60d_pct"] = _last(ind.realized_vol(frame(_tail_or_all(C, 61)), 60)) * 100.0
        f["atr_14_pct"] = _div(_last(ind.atr_wilder(frame(H), frame(L), Cdf, 14)), price) * 100.0
        tail = _tail_or_all(C, W_12M + 1), _tail_or_all(B, W_12M + 1)
        rets = ind.total_return(frame(tail[0]), 1), ind.total_return(frame(tail[1]), 1)
        f["beta_1y"] = _last(ind.rolling_beta(rets[0], rets[1], W_12M))

        # volume
        vol_120 = _tail_mean(V, 120)
        f["rel_volume_5d"] = _div(_tail_mean(V, 5), _tail_mean(V, 60))
        f["rel_volume_20d"] = _div(_tail_mean(V, 20), vol_120)
        w20 = _tail(V, 20)
        f["max_volume_ratio_20d"] = _div(nan if w20 is None else w20.max(axis=0), vol_120)
        f["up_down_volume_ratio_50d"] = _up_down_volume_ratio(C, V, 50)

    out = pd.DataFrame({k: f[k] for k in TECHNICAL_FEATURES if k != "return_6m_percentile"}, index=tickers)
    out.loc[~bars.traded, :] = np.nan
    out.insert(
        TECHNICAL_FEATURES.index("return_6m_percentile"),
        "return_6m_percentile",
        cross_sectional_percentile(out["return_6m_pct"]),
    )
    return out[TECHNICAL_FEATURES].astype("float64")


def _tail_or_all(a: np.ndarray, n: int) -> np.ndarray:
    """Last n bars (all bars if fewer) - limits rolling work to what the latest value needs."""
    return a[-n:] if a.shape[0] > n else a


def _up_down_volume_ratio(C: np.ndarray, V: np.ndarray, n: int) -> np.ndarray:
    """Sum of volume on up-close bars / sum on down-close bars over the last n bars (n + 1 closes);
    unchanged closes count for neither side; NaN if any input is missing or there is no down volume."""
    wc, wv = _tail(C, n + 1), _tail(V, n)
    if wc is None:
        return np.full(C.shape[1], np.nan)
    chg = np.diff(wc, axis=0)
    complete = np.isfinite(chg).all(axis=0) & np.isfinite(wv).all(axis=0)
    up = np.where(chg > 0, wv, 0.0).sum(axis=0)
    down = np.where(chg < 0, wv, 0.0).sum(axis=0)
    return np.where(complete, _div(up, down), np.nan)


def average_volume_shares(prices: PricePanel, as_of: AsOf, window: int = 20) -> pd.Series:
    """Mean daily volume (shares) over each ticker's last ``window`` bars on or before as_of.

    Same conventions as the feature frame: NaN when the ticker has no valid close on the evaluation
    row, has fewer than ``window`` bars, or a volume is missing inside the window.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    tickers = pd.Index(list(prices.close.columns), name="ticker")
    bars = _bars(prices, None, as_of)
    name = f"avg_volume_{window}d_shares"
    if bars.length == 0:
        return pd.Series(np.nan, index=tickers, name=name, dtype="float64")
    avg = np.where(bars.traded, _tail_mean(bars.volume, window), np.nan)
    return pd.Series(avg, index=tickers, name=name, dtype="float64")
