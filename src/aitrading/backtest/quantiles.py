"""Quantile (bucket) analysis and rank information coefficients of a cross-sectional signal."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aitrading.backtest.models import QuantileAnalysis

__all__ = [
    "quantile_buckets",
    "spearman",
    "quantile_analysis",
    "forward_returns_from_close",
]


def quantile_buckets(signal: pd.Series, n_quantiles: int) -> pd.Series:
    """Assign each name with a finite signal to a bucket 1 (worst) .. ``n_quantiles`` (best).

    Names are ordered by signal (higher = better); ties are broken deterministically by ticker
    (alphabetically earlier ticker ranks better). With names ordered worst-first at positions
    p = 0..n-1, bucket = floor(p * n_quantiles / n) + 1, so bucket sizes differ by at most one
    (the worst buckets get the extra names) and every bucket is non-empty when n >= n_quantiles.
    NaN / inf signals are excluded. Returns an int Series indexed by ticker (worst-first order).
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    s = pd.Series(signal, dtype=float)
    s = s[np.isfinite(s.to_numpy())]
    if s.index.has_duplicates:
        raise ValueError("signal index (tickers) must be unique")
    # stable sorts: ticker descending, then signal ascending -> worst first, ties: later ticker worse
    s = s.sort_index(ascending=False, kind="mergesort").sort_values(ascending=True, kind="mergesort")
    n = len(s)
    pos = np.arange(n)
    buckets = (pos * n_quantiles) // max(n, 1) + 1
    return pd.Series(buckets.astype(int), index=s.index, name="bucket")


def spearman(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if either side is constant."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank(method="average").to_numpy()
    if ra.size < 2:
        return float("nan")
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float(da @ da) * float(db @ db))
    if denom <= 0:
        return float("nan")
    return float(da @ db) / denom


def _norm_keys(d: dict) -> dict[pd.Timestamp, pd.Series]:
    return {pd.Timestamp(k): v for k, v in d.items()}


def quantile_analysis(
    signals: dict[pd.Timestamp, pd.Series],
    forward_returns: dict[pd.Timestamp, pd.Series],
    *,
    n_quantiles: int,
    periods_per_year: float,
) -> QuantileAnalysis:
    """Bucket returns and rank ICs of a signal against next-period returns.

    For every date present in both dicts, names with a finite signal and a finite forward return
    are bucketed with :func:`quantile_buckets` (1 = worst, n = best). Dates with fewer than
    ``3 * n_quantiles`` such names are skipped for both the bucket returns and the IC.

    * bucket return per date = equal-weighted mean forward return of the names in the bucket;
    * annual bucket return = geometric CAGR of the bucket's return series,
      prod(1 + r_t) ** (periods_per_year / T) - 1 (assumes consecutive, non-overlapping holding
      periods, as produced by :func:`forward_returns_from_close`; includes volatility drag);
    * spread = annual(best) - annual(worst);
    * monotonicity = Spearman correlation of bucket number vs annual bucket return (0.0 if the
      bucket returns are all equal);
    * IC = Spearman correlation of signal vs forward return per date; ``ic_t_stat`` =
      mean / std(ddof=1) * sqrt(#dates) (0.0 when undefined: < 2 dates or zero dispersion);
      ``ic_hit_rate_pct`` = % of dates with IC > 0.
    Raises ``ValueError`` if no date has enough names.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    sig = _norm_keys(signals)
    fwd = _norm_keys(forward_returns)
    min_names = 3 * n_quantiles

    rows: list[np.ndarray] = []
    ics: list[float] = []
    for d in sorted(set(sig) & set(fwd)):
        df = pd.concat(
            [pd.Series(sig[d], dtype=float).rename("s"), pd.Series(fwd[d], dtype=float).rename("f")],
            axis=1,
            join="inner",
        )
        df = df[np.isfinite(df["s"].to_numpy()) & np.isfinite(df["f"].to_numpy())]
        if len(df) < min_names:
            continue
        buckets = quantile_buckets(df["s"], n_quantiles)
        means = df["f"].groupby(buckets).mean().reindex(range(1, n_quantiles + 1))
        rows.append(means.to_numpy(dtype=float))
        ic = spearman(df["s"], df["f"])
        if math.isfinite(ic):
            ics.append(ic)

    if not rows:
        raise ValueError(
            f"quantile analysis: no date has at least {min_names} names with both a signal and a "
            "forward return"
        )
    B = np.vstack(rows)
    T = B.shape[0]
    wealth = np.prod(1.0 + B, axis=0)
    annual = np.where(wealth > 0, np.abs(wealth) ** (periods_per_year / T) - 1.0, -1.0)
    annual_pct = [float(x) * 100.0 for x in annual]

    mono = spearman(np.arange(1, n_quantiles + 1), annual)
    ic_arr = np.asarray(ics, dtype=float)
    ic_mean = float(ic_arr.mean()) if ic_arr.size else 0.0
    ic_t = 0.0
    if ic_arr.size >= 2:
        sd = float(ic_arr.std(ddof=1))
        if sd > 1e-12:
            ic_t = ic_mean / sd * math.sqrt(ic_arr.size)
    hit = float((ic_arr > 0).mean()) * 100.0 if ic_arr.size else 0.0

    return QuantileAnalysis(
        n_quantiles=n_quantiles,
        annual_return_by_quantile_pct=annual_pct,
        spread_annual_pct=annual_pct[-1] - annual_pct[0],
        monotonicity=float(mono) if math.isfinite(mono) else 0.0,
        ic_mean=ic_mean,
        ic_t_stat=ic_t,
        ic_hit_rate_pct=hit,
    )


def forward_returns_from_close(
    close: pd.DataFrame, dates: list[pd.Timestamp], execution_lag: int = 1
) -> dict[pd.Timestamp, pd.Series]:
    """Forward holding-period returns per signal date, consistent with ``engine.simulate``.

    For consecutive signal dates t_i < t_{i+1} the forward return of a name is measured from the
    close ``execution_lag`` sessions after t_i (its execution day, see
    ``engine.execution_index``) to the close of the execution day of t_{i+1}. Only names with a
    valid price on the entry execution day are included. A name whose price goes missing during
    the period (delisting / data gap) is valued at its last valid price from then on - exactly as
    ``simulate`` does - so delisted losers are not silently dropped (no survivorship bias).
    The last signal date (no following date) and dates whose executions fall outside the price
    data are omitted. Keys are the signal dates t_i.
    """
    from aitrading.backtest.engine import _frozen_growth, _prepare_close, execution_index

    close = _prepare_close(close)
    idx = close.index
    arr = close.to_numpy(dtype=float)
    cols = close.columns
    ds = sorted({pd.Timestamp(d) for d in dates})
    pos = [execution_index(idx, d, execution_lag) for d in ds]
    out: dict[pd.Timestamp, pd.Series] = {}
    for i in range(len(ds) - 1):
        p0, p1 = pos[i], pos[i + 1]
        if p0 is None or p1 is None or p1 <= p0:
            continue
        entry = arr[p0]
        valid = np.isfinite(entry) & (entry > 0)
        if not valid.any():
            continue
        growth, _ = _frozen_growth(arr[p0 : p1 + 1][:, valid])
        out[ds[i]] = pd.Series(growth[-1] - 1.0, index=cols[valid], name=ds[i])
    return out
