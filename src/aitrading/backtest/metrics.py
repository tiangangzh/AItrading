"""Return-series statistics for the backtest engine (numpy / pandas only).

Conventions
-----------
* Inputs are *periodic simple returns as fractions* (0.01 = 1%) indexed by date.
* Reported statistics follow the units in the ``PerformanceStats`` field names: ``*_pct`` fields
  are percent, ratios (Sharpe, Sortino, Calmar, information ratio, beta, t-stats) are unitless.
* ``periods_per_year`` is inferred from the median spacing of the index unless given:
  daily 252, weekly 52, monthly 12, quarterly 4, annual 1.
* Optional statistics that are undefined (zero variance, too few observations, no drawdown, ...)
  are reported as ``None`` - never ``inf`` / ``NaN`` - so results serialise to JSON and back.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd

from aitrading.backtest.models import PerformanceStats

__all__ = [
    "infer_periods_per_year",
    "compound",
    "drawdown_series",
    "max_drawdown",
    "default_nw_lags",
    "newey_west_tstat",
    "performance_stats",
]

# Standard deviations below this are treated as exactly zero (constant series; float noise).
_ZERO_STD = 1e-12

_COMPOUND_RULES = {
    "W": "W-FRI",
    "WEEKLY": "W-FRI",
    "W-FRI": "W-FRI",
    "M": "ME",
    "ME": "ME",
    "MONTHLY": "ME",
    "Q": "QE",
    "QE": "QE",
    "QUARTERLY": "QE",
    "A": "YE",
    "Y": "YE",
    "YE": "YE",
    "ANNUAL": "YE",
    "YEARLY": "YE",
}


# ------------------------------------------------------------------------------------------------
# Frequency helpers
# ------------------------------------------------------------------------------------------------


def infer_periods_per_year(index: Union[pd.Index, pd.Series, pd.DataFrame]) -> float:
    """Infer the number of periods per year from the median spacing (in days) of a date index.

    Median spacing <= 4 days -> 252 (trading days; weekends/holidays make the median 1 day),
    <= 10 -> 52 (weekly), <= 45 -> 12 (monthly), <= 140 -> 4 (quarterly), otherwise 1 (annual).
    Accepts an index or a Series/DataFrame (its index is used). Raises ``ValueError`` with fewer
    than two distinct dates.
    """
    if isinstance(index, (pd.Series, pd.DataFrame)):
        index = index.index
    idx = pd.DatetimeIndex(index).dropna().unique().sort_values()
    if len(idx) < 2:
        raise ValueError("need at least two dates to infer the periodicity of a return series")
    # TimedeltaIndex.total_seconds() is resolution-agnostic (pandas >= 3 may use s/ms/us units)
    spacing_days = float(np.median((idx[1:] - idx[:-1]).total_seconds())) / 86_400.0
    if spacing_days <= 4:
        return 252.0
    if spacing_days <= 10:
        return 52.0
    if spacing_days <= 45:
        return 12.0
    if spacing_days <= 140:
        return 4.0
    return 1.0


def compound(returns: Union[pd.Series, pd.DataFrame], freq: str) -> Union[pd.Series, pd.DataFrame]:
    """Compound periodic returns into lower-frequency period returns: prod(1 + r) - 1.

    ``freq`` is 'W' (weeks ending Friday), 'M', 'Q' or 'A'/'Y' (also accepted: 'weekly',
    'monthly', 'quarterly', 'annual' and the pandas aliases 'ME', 'QE', 'YE'). Results are
    labelled with the *calendar period end* (e.g. 2020-01-31, 2020-03-31, the Friday of the week)
    so series compounded the same way align exactly. NaN returns are skipped inside a period;
    periods without any observation are dropped (not reported as a 0% return).
    """
    key = str(freq).upper()
    if key not in _COMPOUND_RULES:
        raise ValueError(f"unsupported compounding frequency {freq!r}; use 'W', 'M', 'Q' or 'A'")
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.DatetimeIndex(returns.index)
    out = (1.0 + returns.sort_index()).resample(_COMPOUND_RULES[key]).prod(min_count=1) - 1.0
    if isinstance(out, pd.DataFrame):
        return out.dropna(how="all")
    return out.dropna()


# ------------------------------------------------------------------------------------------------
# Drawdowns
# ------------------------------------------------------------------------------------------------


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown (fraction, <= 0) of the compounded wealth path versus its running peak.

    Wealth starts at 1 before the first return, so a loss in the very first period is a
    drawdown. NaN returns are treated as 0 (flat).
    """
    r = pd.Series(returns, dtype=float).fillna(0.0)
    wealth = np.cumprod(1.0 + r.to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, wealth / peak - 1.0, -1.0)
    return pd.Series(dd, index=r.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """Return ``(max_drawdown, longest_underwater_periods)``.

    ``max_drawdown`` is the most negative peak-to-trough decline as a fraction (<= 0).
    The duration is the longest run of consecutive periods spent below a previous peak (the
    run ends in the period the old peak is regained; an unrecovered drawdown counts to the end).
    """
    dd = drawdown_series(returns).to_numpy()
    if dd.size == 0:
        return 0.0, 0
    underwater = dd < -1e-12
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(min(dd.min(), 0.0)), int(longest)


# ------------------------------------------------------------------------------------------------
# Newey-West
# ------------------------------------------------------------------------------------------------


def default_nw_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 * (n / 100) ** (2 / 9))."""
    if n <= 1:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean of ``x`` with a Newey-West (Bartlett kernel) long-run variance.

    long-run variance S = g0 + 2 * sum_{l=1..L} (1 - l / (L + 1)) * g_l, with
    g_l = (1/n) * sum_t (x_t - mean)(x_{t-l} - mean); var(mean) = S / n * n / (n - 1).
    The n / (n - 1) small-sample factor makes ``lags=0`` reproduce the classic
    t = mean / (s / sqrt(n)) with the ddof=1 standard deviation exactly.
    ``lags=None`` uses :func:`default_nw_lags`. NaNs are dropped. Returns NaN when undefined
    (fewer than two observations or zero variance).
    """
    arr = pd.Series(x, dtype=float).to_numpy()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    L = default_nw_lags(n) if lags is None else int(lags)
    if L < 0:
        raise ValueError("lags must be >= 0")
    L = min(L, n - 1)
    mean = float(arr.mean())
    dev = arr - mean
    s = float(dev @ dev) / n
    for lag in range(1, L + 1):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        s += 2.0 * (1.0 - lag / (L + 1.0)) * gamma
    var_mean = max(s, 0.0) / n * n / (n - 1.0)
    se = math.sqrt(var_mean)
    if se <= _ZERO_STD * max(1.0, abs(mean)):
        return float("nan")
    return mean / se


# ------------------------------------------------------------------------------------------------
# Alignment helpers
# ------------------------------------------------------------------------------------------------


def _clean(series: pd.Series, name: str) -> pd.Series:
    s = pd.Series(series, dtype=float)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.DatetimeIndex(s.index)
    s = s.sort_index()
    if s.index.has_duplicates:
        raise ValueError(f"{name} has duplicate dates")
    if np.isinf(s.to_numpy()).any():
        raise ValueError(f"{name} contains infinite values")
    return s.dropna()


_CALENDAR_PERIOD = {52.0: "W-FRI", 12.0: "M", 4.0: "Q", 1.0: "Y"}


def _compound_onto(series: pd.Series, target: pd.DatetimeIndex, ppy: float) -> pd.Series:
    """Compound a finer series into the intervals (previous target date, target date].

    The first interval starts at the beginning of the calendar week / month / quarter / year
    containing the first target date (for those periodicities), otherwise one target spacing
    before it. Intervals without observations are NaN.
    """
    if len(target) < 2:
        return pd.Series(np.nan, index=target)
    code = _CALENDAR_PERIOD.get(float(ppy))
    if code is not None:
        lower = target[0].to_period(code).start_time
        keep = series.index >= lower
    else:
        keep = series.index > target[0] - (target[1] - target[0])
    s = series[keep & (series.index <= target[-1])]
    pos = target.searchsorted(s.index, side="left")
    grouped = (1.0 + s).groupby(pos).prod() - 1.0
    out = pd.Series(np.nan, index=target)
    out.iloc[grouped.index.to_numpy()] = grouped.to_numpy()
    return out


# Monthly-or-coarser risk-free series are per-calendar-period rates; match them by period.
_RF_PERIOD = {12.0: "M", 4.0: "Q", 1.0: "Y"}


def _align_rf(rf: pd.Series, index: pd.DatetimeIndex, ppy: float) -> pd.Series:
    """Risk-free per-period rate on ``index`` (fractions per return period).

    * Monthly / quarterly / annual rf is a rate *for that calendar period*, whatever day it is
      labelled with (the Kenneth French RF sits on the calendar month-end, FRED monthly series on
      the 1st), so it is matched to the returns by calendar period - never as-of, which would
      charge month m (returns dated on its last trading day, or daily returns inside it) the
      rate of month m-1:
      - same or coarser than the returns: (1 + rf) ** (rf_ppy / ppy) - 1 on every return date of
        that period (e.g. a monthly rate spread geometrically over the days of the month);
      - finer than the returns (monthly rf, quarterly returns): the periods are compounded into
        the return interval that contains them.
    * Daily / weekly rf: compounded over each return interval when finer than the returns,
      otherwise converted geometrically and forward-filled ("as of") onto the return dates.
    Return dates without rf (before its first or after its last period) take the nearest
    available value (rf published with a lag keeps its latest rate); if rf has no data, 0.
    """
    rf = _clean(rf, "rf")
    if rf.empty:
        return pd.Series(0.0, index=index)
    rf_ppy = infer_periods_per_year(rf.index) if len(rf) >= 2 else ppy
    code = _RF_PERIOD.get(float(rf_ppy)) if len(rf) >= 2 else None
    finer = rf_ppy > 1.5 * ppy and len(index) >= 2
    if code is not None:
        by_period = rf.groupby(rf.index.to_period(code)).last()
        if finer:
            # label each period at its first day so a return dated on the last trading day
            # (e.g. 2022-12-30) still owns its own calendar month's rate
            starts = pd.Series(by_period.to_numpy(), index=by_period.index.to_timestamp(how="start"))
            out = _compound_onto(starts, index, ppy)
        else:
            conv = (1.0 + by_period) ** (rf_ppy / ppy) - 1.0
            out = pd.Series(conv.reindex(index.to_period(code)).to_numpy(), index=index)
    elif finer:
        out = _compound_onto(rf, index, ppy)
    else:
        conv = (1.0 + rf) ** (rf_ppy / ppy) - 1.0
        out = conv.reindex(conv.index.union(index)).ffill().reindex(index)
    return out.ffill().bfill().fillna(0.0)


def _align_benchmark(bench: pd.Series, index: pd.DatetimeIndex, ppy: float) -> pd.Series:
    """Benchmark returns on the return dates: exact date match, or compounded when finer."""
    bench = _clean(bench, "benchmark")
    if len(bench) >= 2 and infer_periods_per_year(bench.index) > 1.5 * ppy and len(index) >= 2:
        return _compound_onto(bench, index, ppy)
    return bench.reindex(index)


def _opt(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = float(np.std(x, ddof=1))
    return 0.0 if s <= _ZERO_STD else s


# ------------------------------------------------------------------------------------------------
# Performance statistics
# ------------------------------------------------------------------------------------------------


def performance_stats(
    returns: pd.Series,
    *,
    label: str,
    rf: pd.Series | float | None = None,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    periods_per_year: float | None = None,
) -> PerformanceStats:
    """Annualised statistics of a periodic return series (fractions in, units per field name out).

    * total return = prod(1 + r) - 1; CAGR = (1 + total) ** (1 / years) - 1 with
      years = n_periods / periods_per_year (-100% if wealth is wiped out).
    * volatility = std(r, ddof=1) * sqrt(ppy).
    * Sharpe = mean(r - rf) / std(r - rf) * sqrt(ppy). ``rf`` (fractions per its own period) is
      aligned to the return dates and converted to the return periodicity (see ``_align_rf``);
      a float is taken as the per-period rate; ``None`` means rf = 0 (Sharpe still computed).
    * Sortino = mean(r - rf) / DD * sqrt(ppy), DD = sqrt(mean(min(r - rf, 0) ** 2)) (target 0,
      averaged over all periods).
    * max drawdown / longest underwater duration per :func:`max_drawdown`; Calmar = CAGR / |maxDD|.
    * hit rate = share of periods with r > 0; best / worst single period.
    * skew / excess kurtosis = bias-corrected sample moments (pandas), ``None`` if n < 8.
    * mean_return_t_stat = Newey-West t-stat of the mean raw periodic return.
    * avg_turnover_pct = mean of the given one-way turnover series (per rebalance, fraction of
      book, e.g. ``SimulationResult.turnover`` = 0.5 * traded notional), in %; it is not halved
      here, so do not pass the two-way traded notional.
    * benchmark (same periodicity, or finer - then compounded onto the return dates): beta =
      cov(r, b) / var(b), tracking error = std(r - b) * sqrt(ppy), IR = mean(r - b) / std(r - b)
      * sqrt(ppy), on the overlapping dates (>= 2 needed).
    NaN returns are dropped. Raises ``ValueError`` for empty / all-NaN input or < 2 observations.
    """
    if returns is None:
        raise ValueError(f"{label}: return series is None")
    r = _clean(returns, f"{label} returns")
    if r.empty:
        raise ValueError(f"{label}: return series is empty or all-NaN; nothing to evaluate")
    if len(r) < 2:
        raise ValueError(f"{label}: need at least 2 non-NaN returns to compute statistics, got {len(r)}")

    ppy = float(periods_per_year) if periods_per_year else infer_periods_per_year(r.index)
    if ppy <= 0:
        raise ValueError("periods_per_year must be positive")
    x = r.to_numpy()
    n = x.size

    wealth = float(np.prod(1.0 + x))
    total = wealth - 1.0
    years = n / ppy
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else -1.0
    std = _std(x)
    vol = std * math.sqrt(ppy)

    if rf is None:
        rf_arr = np.zeros(n)
    elif isinstance(rf, (int, float)):
        rf_arr = np.full(n, float(rf))
    else:
        rf_arr = _align_rf(rf, r.index, ppy).to_numpy()
    excess = x - rf_arr
    ex_mean = float(excess.mean())
    ex_std = _std(excess)
    sharpe = ex_mean / ex_std * math.sqrt(ppy) if ex_std > 0 else None
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    sortino = ex_mean / downside * math.sqrt(ppy) if downside > _ZERO_STD else None

    mdd, mdd_dur = max_drawdown(r)
    calmar = cagr / abs(mdd) if mdd < -1e-12 else None

    skew = kurt = None
    if n >= 8:
        skew = _opt(r.skew()) if std > 0 else None
        kurt = _opt(r.kurt()) if std > 0 else None

    avg_turnover = None
    if turnover is not None:
        t = pd.Series(turnover, dtype=float).dropna()
        if not t.empty:
            avg_turnover = _opt(t.mean() * 100.0)

    beta = te = ir = None
    if benchmark is not None:
        b = _align_benchmark(benchmark, r.index, ppy)
        both = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
        if len(both) >= 2:
            rv, bv = both["r"].to_numpy(), both["b"].to_numpy()
            var_b = float(np.var(bv, ddof=1))
            if var_b > _ZERO_STD**2:
                beta = _opt(float(np.cov(rv, bv, ddof=1)[0, 1]) / var_b)
            active = rv - bv
            a_std = _std(active)
            te = a_std * math.sqrt(ppy) * 100.0
            ir = float(active.mean()) / a_std * math.sqrt(ppy) if a_std > 0 else None

    return PerformanceStats(
        label=label,
        start=r.index[0].date(),
        end=r.index[-1].date(),
        n_periods=int(n),
        periods_per_year=ppy,
        total_return_pct=total * 100.0,
        cagr_pct=cagr * 100.0,
        volatility_pct=vol * 100.0,
        sharpe=_opt(sharpe),
        sortino=_opt(sortino),
        max_drawdown_pct=mdd * 100.0,
        max_drawdown_duration_periods=mdd_dur,
        calmar=_opt(calmar),
        hit_rate_pct=float(np.mean(x > 0)) * 100.0,
        best_period_pct=float(x.max()) * 100.0,
        worst_period_pct=float(x.min()) * 100.0,
        skew=skew,
        excess_kurtosis=kurt,
        mean_return_t_stat=_opt(newey_west_tstat(r)),
        avg_turnover_pct=avg_turnover,
        beta_to_benchmark=_opt(beta),
        tracking_error_pct=_opt(te),
        information_ratio=_opt(ir),
    )
