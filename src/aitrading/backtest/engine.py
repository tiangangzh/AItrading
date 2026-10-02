"""Vectorised portfolio simulator: rebalance calendar, execution lag, drift, turnover, costs.

Execution convention (shared with ``quantiles.forward_returns_from_close``)
-------------------------------------------------------------------------
Target weights decided with information up to the close of *signal date* t are traded at the
close of the trading session ``execution_lag`` sessions later (default 1 = the next session's
close), so a strategy never trades on the same close its signal was computed from. If t is not
a trading day, the first session after t counts as one session later (lag 0 then means "the
first session on or after t").

Portfolio accounting
--------------------
* Weights are fractions of the book (NAV); they may be negative (shorts) and need not sum to 1.
  The remainder ``1 - sum(w)`` is cash earning 0% here (the stats layer handles the risk-free
  rate), so a long-short book with legs +1 / -1 holds 100% cash collateral.
* Between executions weights DRIFT with prices (buy-and-hold). Daily portfolio return
  r_p(d) = sum_i w_i(d-1, drifted) * r_i(d), with r_i the close-to-close simple return.
* At an execution: traded notional = sum_i |w_target_i - w_drifted_i| (buys + sells, fraction
  of book); cost = traded * costs_bps / 1e4 (``costs_bps`` is a one-way cost, paid on every
  unit bought or sold), subtracted from that day's return (net = gross - cost). The reported
  turnover is the conventional ONE-WAY turnover = traded / 2: fully replacing a long-only book
  is 100%, building a long-only book from cash 50%, building a +1 / -1 long-short book 100%.
* Missing prices (NaN / inf) are told apart by whether the name trades again later in ``close``:
  - interior gap (trading halt, missing print - a later valid price exists): the position keeps
    its exposure, is carried at its last price while the price is missing and is marked to the
    new price when trading resumes, so a move during the gap lands in P&L on that day. If the
    gap covers an execution day the position cannot be traded: it is kept at its drifted weight
    (no turnover, no cost) next to the new target and traded at a later rebalance.
  - terminal delisting (no later valid price): on the first missing day the position is marked
    at its last price times ``1 + delisting_return`` (default 0.0 - no delisting return, which
    is optimistic for performance-related delistings, cf. Shumway 1997), held as cash from then
    on and dropped at the next rebalance (no turnover). A warning names each delisted name once.
* Names in a target without a valid price on the execution day are dropped and the remaining
  weights are NOT renormalised (a warning is added).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from aitrading.backtest.quantiles import quantile_buckets

__all__ = [
    "SimulationResult",
    "rebalance_dates",
    "execution_index",
    "simulate",
    "build_quantile_weights",
]

_PERIOD_CODES = {
    "weekly": "W-SUN",
    "monthly": "M",
    "quarterly": "Q",
    "annual": "Y",
}


# ------------------------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------------------------


def rebalance_dates(
    trading_days: pd.DatetimeIndex,
    freq: str,
    start=None,
    end=None,
) -> list[pd.Timestamp]:
    """Last trading day of each week (Mon-Sun) / month / quarter / calendar year in the data.

    ``daily`` returns every trading day. Period ends are determined on the full
    ``trading_days`` index and then restricted to [start, end] (inclusive), so a mid-month
    ``end`` does not create a fake month-end. The last period of the data may be partial: its
    "end" is the last available trading day (useful as the latest signal date).
    """
    idx = pd.DatetimeIndex(trading_days).dropna().unique().sort_values()
    key = str(freq).lower()
    if key == "daily":
        ends = idx
    elif key in _PERIOD_CODES:
        if len(idx) == 0:
            return []
        s = pd.Series(idx, index=idx)
        ends = pd.DatetimeIndex(s.groupby(idx.to_period(_PERIOD_CODES[key])).max().to_numpy())
    else:
        raise ValueError(f"unknown rebalance frequency {freq!r}; expected daily/weekly/monthly/quarterly/annual")
    if start is not None:
        ends = ends[ends >= pd.Timestamp(start)]
    if end is not None:
        ends = ends[ends <= pd.Timestamp(end)]
    return [pd.Timestamp(d) for d in ends]


def execution_index(trading_days: pd.DatetimeIndex, signal_date, execution_lag: int = 1) -> int | None:
    """Position in ``trading_days`` (sorted) of the execution session for ``signal_date``.

    lag >= 1: the session ``lag`` sessions after the signal session (if the signal date is not
    a session, the first session after it counts as one). lag 0: the first session on or after
    the signal date. Returns ``None`` when the execution day lies beyond the data.
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    t = pd.Timestamp(signal_date)
    if execution_lag == 0:
        p = int(trading_days.searchsorted(t, side="left"))
    else:
        p = int(trading_days.searchsorted(t, side="right")) + execution_lag - 1
    return p if p < len(trading_days) else None


# ------------------------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Output of :func:`simulate`.

    * ``daily_returns`` - net of costs, one value per trading day from the first execution day
      to the end of the window (the first day is -cost: the book is built at that close).
    * ``gross_returns`` - before costs, same index.
    * ``turnover`` - ONE-WAY turnover per execution = 0.5 * sum |w_target - w_drifted|
      (fraction of book; 1.0 = the whole book replaced), indexed by execution date. This is the
      series to pass to ``metrics.performance_stats(turnover=...)``.
    * ``traded`` - traded notional per execution = sum |w_target - w_drifted| (buys + sells,
      = 2 * turnover); ``costs`` = traded * costs_bps / 1e4. Net = gross - cost on those days.
    * ``weights_history`` - post-trade holdings by execution date: the executed target weights
      (names without a price that day dropped) plus any held position that could not be traded
      that day because of a price gap (at its drifted weight).
    """

    daily_returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series]
    warnings: list[str] = field(default_factory=list)
    traded: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, name="traded"))

    def latest_weights(self) -> pd.Series:
        """Most recently executed target weights (empty Series if none)."""
        if not self.weights_history:
            return pd.Series(dtype=float)
        return self.weights_history[max(self.weights_history)]


def _prepare_close(close: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(close, pd.DataFrame):
        raise TypeError("close must be a DataFrame (dates x tickers)")
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.DatetimeIndex(close.index)
    if close.index.has_duplicates:
        raise ValueError("close has duplicate dates")
    if close.columns.has_duplicates:
        raise ValueError("close has duplicate tickers")
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    return close


def _price_state(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(raw, filled, last_valid)`` for a (dates x names) price matrix.

    ``raw`` has non-finite prices replaced by NaN; ``filled`` is ``raw`` forward-filled per
    column (NaN only before a column's first price); ``last_valid[c]`` is the row of column c's
    last finite price (-1 if it never has one). A missing price at row r <= last_valid[c] is an
    interior gap (trading resumes later); rows > last_valid[c] are after a terminal delisting.
    """
    raw = np.where(np.isfinite(arr), arr, np.nan)
    filled = pd.DataFrame(raw).ffill().to_numpy(dtype=float)
    ok = np.isfinite(raw)
    n = raw.shape[0]
    last_valid = np.where(ok.any(axis=0), n - 1 - ok[::-1].argmax(axis=0), -1) if n else np.zeros(0, int)
    return raw, filled, last_valid.astype(int)


def _holding_growth(
    filled: np.ndarray,
    last_valid: np.ndarray,
    p0: int,
    p1: int,
    cols: np.ndarray,
    delisting_return: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Price relatives P_d / P_{p0} for rows d = p0..p1 of columns ``cols``.

    ``filled[p0, cols]`` must be positive (the entry / carried price). Interior gaps are carried
    at the last price (forward fill) and marked to market when trading resumes. Rows after a
    column's last valid price (terminal delisting) stay at the last price times
    ``1 + delisting_return``. Returns ``(growth, first_dead)``: growth is (p1-p0+1, m) and
    ``first_dead[k]`` the relative row of column k's first post-delisting row (p1-p0+1 if none).
    """
    seg = filled[p0 : p1 + 1][:, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = seg / seg[0]
    rows = g.shape[0]
    first_dead = np.clip(last_valid[cols] - p0 + 1, 0, rows)
    if delisting_return != 0.0:
        dead = np.arange(rows)[:, None] >= first_dead[None, :]
        g = np.where(dead, g * (1.0 + delisting_return), g)
    return g, first_dead


def _clean_weights(w) -> pd.Series:
    s = pd.Series(w, dtype=float) if not isinstance(w, pd.Series) else w.astype(float)
    if s.index.has_duplicates:
        s = s.groupby(level=0, sort=False).sum()
    s = s[np.isfinite(s.to_numpy()) & (s.to_numpy() != 0.0)]
    return s


def _fmt(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _names(names, limit: int = 10) -> str:
    names = [str(n) for n in names]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", +{len(names) - limit} more"


def simulate(
    target_weights: dict[pd.Timestamp, pd.Series],
    close: pd.DataFrame,
    *,
    costs_bps: float = 10.0,
    execution_lag: int = 1,
    start=None,
    end=None,
    delisting_return: float = 0.0,
) -> SimulationResult:
    """Simulate a rebalanced portfolio from target weights keyed by *signal* date.

    See the module docstring for the full convention. In short: weights for signal date t are
    executed at the close ``execution_lag`` sessions later; between executions weights drift
    with prices; daily return = sum(w_{d-1, drifted} * r_d); traded notional at each execution =
    sum |w_target - w_drifted|, cost = traded * costs_bps / 1e4 deducted on the execution day,
    reported one-way turnover = traded / 2. A held name with a temporary price gap is carried at
    its last price and marked to market when trading resumes (it is not traded while its price
    is missing); after a terminal delisting it is marked at its last price times
    ``1 + delisting_return`` and becomes cash. Target names without a price on the execution day
    are dropped without renormalising.

    ``start`` / ``end`` bound the simulation window: only executions on days within
    [start, end] are used, the return series starts on the first such execution day and ends
    on the last trading day <= ``end`` (default: last row of ``close``). If two signal dates
    map to the same execution day, the later signal wins. A target may be empty (all cash).
    ``delisting_return`` (fraction, > -1) is booked on the first missing day of a name with no
    later price in ``close``; the default 0.0 applies none, which is optimistic when names were
    delisted for performance reasons (a warning says so). Raises ``ValueError`` when no signal
    date can be executed inside the window.

    Vectorised per holding period: one matrix product over the (days x held names) slice.
    """
    if costs_bps < 0:
        raise ValueError("costs_bps must be >= 0")
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    if not np.isfinite(delisting_return) or delisting_return <= -1.0:
        raise ValueError("delisting_return must be a finite fraction > -1")
    close = _prepare_close(close)
    idx = close.index
    if len(idx) == 0:
        raise ValueError("close has no rows")
    arr, filled, last_valid = _price_state(close.to_numpy(dtype=float))
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    warnings: list[str] = []

    # ---- map signal dates to execution sessions -------------------------------------------
    schedule: dict[int, tuple[pd.Timestamp, object]] = {}
    unexecutable: list[pd.Timestamp] = []
    collisions: list[str] = []
    for t in sorted(target_weights, key=pd.Timestamp):
        ts = pd.Timestamp(t)
        p = execution_index(idx, ts, execution_lag)
        if p is None:
            if end_ts is None or ts < end_ts:
                unexecutable.append(ts)
            continue
        d = idx[p]
        if (start_ts is not None and d < start_ts) or (end_ts is not None and d > end_ts):
            continue
        if p in schedule:
            collisions.append(f"{_fmt(schedule[p][0])} -> {_fmt(ts)} (both execute {_fmt(d)})")
        schedule[p] = (ts, target_weights[t])
    if unexecutable:
        warnings.append(
            f"{len(unexecutable)} signal date(s) not executed: fewer than {max(execution_lag, 1)} "
            f"session(s) of price data after them ({_names([_fmt(d) for d in unexecutable])})"
        )
    if collisions:
        warnings.append(
            "several signal dates map to the same execution day; the later signal was used: "
            + "; ".join(collisions[:5])
            + (f"; +{len(collisions) - 5} more" if len(collisions) > 5 else "")
        )
    if not schedule:
        raise ValueError("no target weights can be executed inside the simulation window")

    exec_pos = sorted(schedule)
    first_pos = exec_pos[0]
    last_pos = (int(idx.searchsorted(end_ts, side="right")) - 1) if end_ts is not None else len(idx) - 1
    n_days = last_pos - first_pos + 1
    gross = np.zeros(n_days)
    cost_by_day = np.zeros(n_days)

    turnover_vals: list[float] = []
    traded_vals: list[float] = []
    cost_vals: list[float] = []
    exec_dates: list[pd.Timestamp] = []
    history: dict[pd.Timestamp, pd.Series] = {}
    drifted = pd.Series(dtype=float)  # pre-trade weights at the current execution
    delist_warned: set[str] = set()
    gap_first: dict[str, pd.Timestamp] = {}
    ruined = False

    for j, p in enumerate(exec_pos):
        d = idx[p]
        target = _clean_weights(schedule[p][1])

        # drop target names without a valid price on the execution day
        if len(target):
            cpos = close.columns.get_indexer(target.index)
            px = np.where(cpos >= 0, arr[p, np.maximum(cpos, 0)], np.nan)
            ok = np.isfinite(px) & (px > 0)
            if not ok.all():
                dropped = list(target.index[~ok])
                target = target[ok]
                warnings.append(
                    f"{_fmt(d)}: dropped {len(dropped)} target name(s) with no price on the execution "
                    f"day ({_names(dropped)}); remaining weights not renormalised "
                    f"(sum {float(target.sum()):.4f})"
                )

        # held names in a price gap today cannot be traded: they stay at their drifted weight
        stuck = pd.Series(dtype=float)
        if len(drifted):
            live = np.isfinite(arr[p, close.columns.get_indexer(drifted.index)])
            if not live.all():
                stuck, drifted = drifted[~live], drifted[live]
                warnings.append(
                    f"{_fmt(d)}: {len(stuck)} held name(s) had no price on the execution day "
                    f"(trading halt / data gap) and could not be traded ({_names(stuck.index)}); "
                    "kept at their drifted weight until a later rebalance"
                )

        t_al, d_al = target.align(drifted, fill_value=0.0)
        traded = float(np.abs(t_al.to_numpy() - d_al.to_numpy()).sum())
        cost = traded * costs_bps / 1e4
        exec_dates.append(d)
        traded_vals.append(traded)
        turnover_vals.append(0.5 * traded)
        cost_vals.append(cost)
        cost_by_day[p - first_pos] = cost
        if len(stuck):
            # a stuck position's currency value is untouched by today's trading costs (paid from
            # cash), so re-express its pre-trade weight against the post-cost NAV
            g_p = gross[p - first_pos]
            post = 1.0 + g_p - cost
            if post > 0:
                stuck = stuck * ((1.0 + g_p) / post)
            book = pd.concat([target, stuck]).astype(float)
        else:
            book = target.copy()
        history[d] = book.copy()

        # ---- holding period (p, next_p] ------------------------------------------------------
        next_p = exec_pos[j + 1] if j + 1 < len(exec_pos) else last_pos
        if next_p <= p:
            drifted = book.copy()
            continue
        if len(book) == 0:
            drifted = pd.Series(dtype=float)
            continue
        w = book.to_numpy()
        cpos = close.columns.get_indexer(book.index)
        growth, first_dead = _holding_growth(filled, last_valid, p, next_p, cpos, delisting_return)
        nav = 1.0 + (growth - 1.0) @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = nav[1:] / nav[:-1] - 1.0
        if (nav[1:] <= 0).any():
            q = int(np.argmax(nav[1:] <= 0))  # first day the book is wiped out
            rets[q] = -1.0
            rets[q + 1 :] = 0.0
            if not ruined:
                warnings.append(
                    f"{_fmt(idx[p + 1 + q])}: portfolio value fell to <= 0 (ruin); "
                    "compounded returns are -100% from here on"
                )
                ruined = True
        gross[p + 1 - first_pos : next_p + 1 - first_pos] = rets

        rows = growth.shape[0]
        # interior gaps (price missing, trading resumes later): reported once per name
        rel = np.arange(rows)[:, None]
        gap = ~np.isfinite(arr[p : next_p + 1][:, cpos]) & (rel >= 1) & (rel < first_dead[None, :])
        for k in np.flatnonzero(gap.any(axis=0)):
            gap_first.setdefault(str(book.index[k]), idx[p + int(gap[:, k].argmax())])
        for k in np.flatnonzero(first_dead < rows):
            name = str(book.index[k])
            if name not in delist_warned:
                delist_warned.add(name)
                how = (
                    f"marked at its last price with a {delisting_return * 100:+.1f}% delisting return"
                    if delisting_return != 0.0
                    else "valued at its last price with no delisting return (0%; optimistic if it "
                    "was delisted for performance reasons)"
                )
                warnings.append(
                    f"{name}: no price from {_fmt(idx[p + int(first_dead[k])])} on while held "
                    f"(delisted, or its data ends); {how}, held as cash and dropped at the next "
                    "rebalance"
                )
        alive = first_dead >= rows
        if nav[-1] > 0:
            dw = pd.Series(w[alive] * growth[-1, alive] / nav[-1], index=book.index[alive])
            drifted = dw[dw.to_numpy() != 0.0]
        else:
            drifted = pd.Series(dtype=float)

    if gap_first:
        warnings.append(
            f"{len(gap_first)} held name(s) had missing prices while held (trading halt / data "
            "gap); carried at their last price and marked to market when trading resumed: "
            + _names([f"{k} from {_fmt(v)}" for k, v in gap_first.items()])
        )

    day_index = idx[first_pos : last_pos + 1]
    gross_s = pd.Series(gross, index=day_index, name="gross_return")
    net_s = pd.Series(gross - cost_by_day, index=day_index, name="net_return")
    ex_index = pd.DatetimeIndex(exec_dates)
    return SimulationResult(
        daily_returns=net_s,
        gross_returns=gross_s,
        turnover=pd.Series(turnover_vals, index=ex_index, name="turnover", dtype=float),
        costs=pd.Series(cost_vals, index=ex_index, name="costs", dtype=float),
        weights_history=history,
        warnings=warnings,
        traded=pd.Series(traded_vals, index=ex_index, name="traded", dtype=float),
    )


# ------------------------------------------------------------------------------------------------
# Portfolio construction
# ------------------------------------------------------------------------------------------------


def _cap_weights(w: np.ndarray, cap: float | None) -> np.ndarray:
    """Normalise positive weights to sum 1, then cap each at ``cap`` redistributing the excess
    proportionally among uncapped names (iterated until no name exceeds the cap). If
    n * cap <= 1 the cap is binding for everyone: all names get ``cap`` and the leg sums to
    n * cap (< 1, the rest stays in cash)."""
    w0 = w / w.sum()
    if cap is None or w0.size == 0:
        return w0
    if cap * w0.size <= 1.0 + 1e-12:
        return np.full(w0.size, float(cap))
    out = w0.copy()
    fixed = np.zeros(w0.size, dtype=bool)
    for _ in range(w0.size):
        over = (out > cap * (1.0 + 1e-12)) & ~fixed
        if not over.any():
            break
        fixed |= over
        free = ~fixed
        out[fixed] = cap
        out[free] = w0[free] / w0[free].sum() * (1.0 - cap * fixed.sum())
    return out


def _leg_weights(
    names: list,
    weighting: str,
    market_cap: pd.Series | None,
    volatility: pd.Series | None,
    max_weight: float | None,
) -> pd.Series:
    """Positive leg weights summing to 1 (before the cap). ``names`` ordered most extreme first."""
    if not names:
        return pd.Series(dtype=float)
    if weighting == "equal":
        raw = pd.Series(1.0, index=names)
    elif weighting == "signal":
        k = len(names)
        raw = pd.Series(np.arange(k, 0, -1, dtype=float), index=names)
    elif weighting == "value":
        raw = pd.Series(market_cap, dtype=float).reindex(names)
    elif weighting == "inverse_vol":
        vol = pd.Series(volatility, dtype=float).reindex(names)
        with np.errstate(divide="ignore"):
            raw = 1.0 / vol
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    raw = raw[np.isfinite(raw.to_numpy()) & (raw.to_numpy() > 0)]
    if raw.empty:
        return pd.Series(dtype=float)
    return pd.Series(_cap_weights(raw.to_numpy(), max_weight), index=raw.index)


def build_quantile_weights(
    signal: pd.Series,
    *,
    n_quantiles: int = 5,
    style: Literal["long_only", "long_short"] = "long_short",
    selection: Literal["quantile", "top_n"] = "quantile",
    top_n: int | None = None,
    weighting: Literal["equal", "value", "signal", "inverse_vol"] = "equal",
    market_cap: pd.Series | None = None,
    volatility: pd.Series | None = None,
    max_weight: float | None = None,
) -> pd.Series:
    """Signed portfolio weights from a cross-sectional signal (higher = better).

    Convention: the long leg (best bucket, or the ``top_n`` best names) sums to +1. For
    ``long_short`` the short leg (worst bucket, or the ``top_n`` worst names) sums to -1, i.e.
    gross exposure 2 and net 0 per unit of capital - the simulated return is the long-leg return
    minus the short-leg return (cash collateral earns 0 in ``simulate``), so regress it as a
    self-financing portfolio (``excess=False``).

    * NaN / inf signals are excluded; buckets per :func:`quantile_buckets` (ties broken by
      ticker, alphabetically earlier = better). Fewer valid names than ``n_quantiles`` ->
      empty Series (no portfolio). ``top_n`` long-short uses min(top_n, n // 2) names per side
      so the legs never overlap.
    * weighting within each leg: ``equal``; ``value`` (proportional to ``market_cap``);
      ``inverse_vol`` (proportional to 1 / ``volatility``); ``signal`` (proportional to the rank
      inside the leg: the most extreme name gets k, the least extreme 1). Names lacking a
      positive, finite market cap / volatility are left out of that leg.
    * ``max_weight`` caps |weight| per name within a leg, redistributing the excess
      proportionally to the uncapped names; if the cap cannot be met (k * cap < 1) every name
      gets the cap and the leg holds the remainder in cash.
    Returns a float Series indexed by ticker (sorted), without zero weights.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    if style not in ("long_only", "long_short"):
        raise ValueError(f"unknown style {style!r}")
    if selection not in ("quantile", "top_n"):
        raise ValueError(f"unknown selection {selection!r}")
    if weighting == "value" and market_cap is None:
        raise ValueError("value weighting needs market_cap")
    if weighting == "inverse_vol" and volatility is None:
        raise ValueError("inverse_vol weighting needs volatility")
    if max_weight is not None and not (0 < max_weight <= 1):
        raise ValueError("max_weight must be in (0, 1]")

    buckets = quantile_buckets(signal, n_quantiles)  # worst first, deterministic ties
    worst_first = list(buckets.index)
    best_first = worst_first[::-1]
    n = len(best_first)

    if selection == "quantile":
        if n < n_quantiles:
            return pd.Series(dtype=float, name="weight")
        b = buckets.to_numpy()
        long_names = list(buckets.index[b == n_quantiles])[::-1]  # best first
        short_names = list(buckets.index[b == 1])  # worst first
    else:
        if not top_n or top_n < 1:
            raise ValueError("selection='top_n' needs top_n >= 1")
        k = min(top_n, n) if style == "long_only" else min(top_n, n // 2)
        long_names = best_first[:k]
        short_names = worst_first[:k]

    long_w = _leg_weights(long_names, weighting, market_cap, volatility, max_weight)
    parts = [long_w]
    if style == "long_short":
        short_w = _leg_weights(short_names, weighting, market_cap, volatility, max_weight)
        parts.append(-short_w)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.Series(dtype=float, name="weight")
    out = pd.concat(parts).astype(float)
    out = out[out != 0.0].sort_index()
    out.name = "weight"
    return out
