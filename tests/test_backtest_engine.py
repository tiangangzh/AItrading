"""Tests for the backtest simulator, portfolio construction and quantile analysis.

Every expected number below is computed by hand in the comments (prices chosen so the
arithmetic is exact), plus a brute-force reference simulator for random data.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.engine import (
    SimulationResult,
    build_quantile_weights,
    execution_index,
    rebalance_dates,
    simulate,
)
from aitrading.backtest.quantiles import (
    forward_returns_from_close,
    quantile_analysis,
    quantile_buckets,
    spearman,
)

D = pd.bdate_range("2024-01-01", periods=5)  # Mon 1 .. Fri 5 Jan 2024
d0, d1, d2, d3, d4 = D


def _close(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols, index=D[: len(next(iter(cols.values())))], dtype=float)


# A: 0%, +10%, +10%, 0% ; B: 0%, -10%, 0%, +10%
BASE = dict(A=[10, 10, 11, 12.1, 12.1], B=[10, 10, 9, 9, 9.9])


# ------------------------------------------------------------------------------------------------
# rebalance calendar / execution mapping
# ------------------------------------------------------------------------------------------------


def test_rebalance_dates_month_week_quarter_year_ends():
    days = pd.bdate_range("2020-01-01", "2021-03-31")
    m = rebalance_dates(days, "monthly")
    assert m[:3] == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]
    assert m[-1] == pd.Timestamp("2021-03-31")
    q = rebalance_dates(days, "quarterly")
    assert q[:2] == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")]
    a = rebalance_dates(days, "annual")
    assert a == [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-03-31")]  # last period partial
    w = rebalance_dates(days, "weekly")
    assert w[0] == pd.Timestamp("2020-01-03") and all(d.dayofweek == 4 for d in w[:-1])
    assert rebalance_dates(days[:7], "daily") == list(days[:7])


def test_rebalance_dates_holidays_and_window():
    # 2020-07-31 removed (pretend holiday) -> July's last trading day is the 30th
    days = pd.bdate_range("2020-06-01", "2020-09-30").drop(pd.Timestamp("2020-07-31"))
    m = rebalance_dates(days, "monthly")
    assert pd.Timestamp("2020-07-30") in m
    # a mid-month end does not create a fake month-end; start/end are inclusive
    w = rebalance_dates(days, "monthly", start="2020-06-30", end="2020-09-15")
    assert w == [pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-30"), pd.Timestamp("2020-08-31")]
    with pytest.raises(ValueError):
        rebalance_dates(days, "fortnightly")


def test_execution_index_lag_and_non_trading_signal_dates():
    idx = pd.bdate_range("2024-01-01", periods=6)  # Mon 1 .. Mon 8
    assert execution_index(idx, idx[0], 1) == 1
    assert execution_index(idx, idx[0], 2) == 2
    assert execution_index(idx, idx[0], 0) == 0
    sat = pd.Timestamp("2024-01-06")
    assert idx[execution_index(idx, sat, 1)] == pd.Timestamp("2024-01-08")  # first session after
    assert idx[execution_index(idx, sat, 0)] == pd.Timestamp("2024-01-08")  # never the Friday close
    assert execution_index(idx, idx[-1], 1) is None


# ------------------------------------------------------------------------------------------------
# simulate: hand-computed examples
# ------------------------------------------------------------------------------------------------


def test_drift_and_first_day_cost():
    close = _close(**BASE)
    res = simulate({d0: pd.Series({"A": 0.5, "B": 0.5})}, close, costs_bps=10)
    assert isinstance(res, SimulationResult)
    # executed at d1's close; series starts on the execution day
    assert list(res.daily_returns.index) == [d1, d2, d3, d4]
    # d2: .5*10% + .5*(-10%) = 0 ; drifted w = (.55, .45)
    # d3: .55 * 10% = 5.5% ; NAV 1.055, w = (.605, .45)/1.055
    # d4: .45/1.055 * 10% = .045/1.055
    expected_gross = [0.0, 0.0, 0.055, 0.045 / 1.055]
    np.testing.assert_allclose(res.gross_returns.to_numpy(), expected_gross, atol=1e-12)
    # building the book from cash: turnover 1.0, cost 10bp charged on d1
    assert res.turnover.to_dict() == {d1: pytest.approx(1.0)}
    assert res.costs.to_dict() == {d1: pytest.approx(0.001)}
    np.testing.assert_allclose(res.daily_returns.to_numpy(), [-0.001, 0.0, 0.055, 0.045 / 1.055], atol=1e-12)
    # buy-and-hold wealth: .5*12.1/10 + .5*9.9/10 = 1.1
    assert np.prod(1 + res.gross_returns) == pytest.approx(1.1)
    assert res.weights_history[d1].to_dict() == {"A": 0.5, "B": 0.5}
    assert res.warnings == []


def test_rebalance_turnover_and_costs():
    close = _close(**BASE)
    w = pd.Series({"A": 0.5, "B": 0.5})
    res = simulate({d0: w, d2: w}, close, costs_bps=10)
    # second execution at d3's close; pre-trade drifted weights (.605, .45)/1.055
    drift_a = 0.605 / 1.055
    turnover = abs(0.5 - drift_a) + abs(0.5 - (1 - drift_a))  # = 2 * .0775 / 1.055
    assert turnover == pytest.approx(0.14691943127962084)
    assert res.turnover[d3] == pytest.approx(turnover)
    assert res.costs[d3] == pytest.approx(turnover * 10 / 1e4)
    # d3: return earned on the old drifted book, minus the cost of trading at the close
    assert res.gross_returns[d3] == pytest.approx(0.055)
    assert res.daily_returns[d3] == pytest.approx(0.055 - turnover * 1e-3)
    # d4: back to 50/50 -> .5 * 0% + .5 * 10% = 5%
    assert res.gross_returns[d4] == pytest.approx(0.05)
    assert list(res.turnover.index) == [d1, d3]


def test_execution_lag_two_and_zero():
    close = _close(**BASE)
    w = {d0: pd.Series({"A": 0.5, "B": 0.5})}
    lag2 = simulate(w, close, costs_bps=0, execution_lag=2)
    # executed at d2's close: d3 = .5*10% = 5% ; d4 = (.5/1.05) * 10% = .05/1.05
    assert list(lag2.daily_returns.index) == [d2, d3, d4]
    np.testing.assert_allclose(lag2.daily_returns.to_numpy(), [0.0, 0.05, 0.05 / 1.05], atol=1e-12)
    lag0 = simulate(w, close, costs_bps=0, execution_lag=0)
    # executed at d0's close (same-close trading, only on explicit request)
    assert lag0.daily_returns.index[0] == d0
    np.testing.assert_allclose(lag0.daily_returns.to_numpy(), [0.0, 0.0, 0.0, 0.055, 0.045 / 1.055], atol=1e-12)


def test_signal_on_execution_day_does_not_see_that_close():
    # A jumps +100% on d2. A signal computed at d1's close (say, knowing nothing of d2) executed
    # with lag 1 is filled at d2's close -> it must NOT earn the d2 jump.
    close = _close(A=[10, 10, 20, 20, 20])
    res = simulate({d1: pd.Series({"A": 1.0})}, close, costs_bps=0)
    assert res.daily_returns.index[0] == d2
    assert res.daily_returns.abs().max() == 0.0


def test_long_short_weights_drift():
    close = _close(**BASE)
    res = simulate({d0: pd.Series({"A": 1.0, "B": -1.0})}, close, costs_bps=10)
    assert res.turnover[d1] == pytest.approx(2.0)  # gross exposure 2
    # d2: 1*10% - 1*(-10%) = 20%; NAV = 1 (cash) + 1.1 - .9 = 1.2
    # d3: A +10% -> 0.11 / 1.2 ; NAV 1.31
    # d4: B +10% on a -0.9 position -> -0.09 / 1.31
    np.testing.assert_allclose(
        res.gross_returns.to_numpy(), [0.0, 0.2, 0.11 / 1.2, -0.09 / 1.31], atol=1e-12
    )
    assert res.daily_returns[d1] == pytest.approx(-0.002)


def test_delisted_holding_frozen_and_dropped_with_one_warning():
    close = _close(A=[10, 10, 11, 12.1, 12.1], B=[10, 10, 9, np.nan, np.nan])
    w = pd.Series({"A": 0.5, "B": 0.5})
    res = simulate({d0: w, d3: w}, close, costs_bps=0)
    # d3: B has no price -> 0 return; A: .55 * 10% = 5.5% ; d4: A flat -> 0
    np.testing.assert_allclose(res.gross_returns.to_numpy(), [0.0, 0.0, 0.055, 0.0], atol=1e-12)
    # second execution on d4: B (no price) dropped from the target, A not renormalised;
    # the frozen B position became cash, so only A trades: |.5 - .605/1.055|
    assert res.weights_history[d4].to_dict() == {"A": 0.5}
    assert res.turnover[d4] == pytest.approx(abs(0.5 - 0.605 / 1.055))
    delist = [m for m in res.warnings if m.startswith("B:")]
    assert len(delist) == 1 and "2024-01-04" in delist[0]
    assert any("dropped 1 target name(s)" in m and "(B)" in m for m in res.warnings)


def test_data_gap_keeps_zero_return_until_next_rebalance():
    # B's price comes back on d4, but the position stays frozen at its last price (0% return)
    close = _close(A=[10, 10, 10, 10, 10], B=[10, 10, 9, np.nan, 12])
    res = simulate({d0: pd.Series({"A": 0.5, "B": 0.5})}, close, costs_bps=0)
    np.testing.assert_allclose(res.gross_returns.to_numpy(), [0.0, -0.05, 0.0, 0.0], atol=1e-12)
    assert sum(m.startswith("B:") for m in res.warnings) == 1


def test_missing_price_on_execution_day_dropped_not_renormalised():
    close = _close(A=[10, 10, 11, 11, 11], C=[10, np.nan, 10, 10, 10])
    res = simulate({d0: pd.Series({"A": 0.5, "C": 0.3, "ZZZ": 0.2})}, close, costs_bps=10)
    assert res.weights_history[d1].to_dict() == {"A": 0.5}
    assert res.turnover[d1] == pytest.approx(0.5)
    assert res.gross_returns[d2] == pytest.approx(0.05)  # .5 * 10%, the rest is cash
    msg = [m for m in res.warnings if "dropped 2 target name(s)" in m]
    assert msg and "C" in msg[0] and "ZZZ" in msg[0] and "not renormalised" in msg[0]


def test_cash_target_window_and_unexecutable_signals():
    close = _close(**BASE)
    flat = simulate({d0: pd.Series(dtype=float)}, close)
    assert (flat.daily_returns == 0).all() and flat.turnover[d1] == 0.0

    w = pd.Series({"A": 1.0})
    res = simulate({d0: w, d2: w, d4: w}, close, costs_bps=0, start=d2, end=d3)
    # d0's execution (d1) is before start -> the series starts at d3 (execution of d2) and ends at end
    assert list(res.daily_returns.index) == [d3]
    assert list(res.turnover.index) == [d3]
    assert not any("not executed" in m for m in res.warnings)  # d4's signal is past `end`: silent

    res = simulate({d0: w, d4: w}, close)
    assert any("1 signal date(s) not executed" in m and "2024-01-05" in m for m in res.warnings)

    with pytest.raises(ValueError, match="no target weights"):
        simulate({d4: w}, close)
    with pytest.raises(ValueError):
        simulate({d0: w}, close, costs_bps=-1)


def test_signals_mapping_to_same_execution_day_use_the_later_one():
    idx = pd.bdate_range("2024-01-01", periods=7)  # Mon 1 .. Tue 9
    close = pd.DataFrame({"A": [10.0] * 6 + [11.0], "B": [10.0] * 7}, index=idx)
    sat, sun = pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")
    res = simulate({sat: pd.Series({"B": 1.0}), sun: pd.Series({"A": 1.0})}, close, costs_bps=0)
    assert list(res.weights_history) == [pd.Timestamp("2024-01-08")]
    assert res.weights_history[pd.Timestamp("2024-01-08")].to_dict() == {"A": 1.0}
    assert res.gross_returns.iloc[-1] == pytest.approx(0.1)
    assert any("later signal" in m for m in res.warnings)


def test_ruin_is_clipped_at_minus_100pct():
    close = _close(A=[10, 10, 10, 10, 10], B=[10, 10, 25, 30, 30])
    res = simulate({d0: pd.Series({"A": 1.0, "B": -1.0})}, close, costs_bps=0)
    # d2: B +150% on a -1 short -> NAV = 1 - 1.5 < 0 -> -100%
    assert res.gross_returns[d2] == -1.0
    assert np.prod(1 + res.daily_returns) == 0.0
    assert any("ruin" in m for m in res.warnings)


def _reference_simulate(target_weights, close, costs_bps, lag):
    """Brute-force day-by-day simulator (positions in currency units) used as an oracle."""
    idx = close.index
    execs = {}
    for t, w in sorted(target_weights.items()):
        p = execution_index(idx, t, lag)
        if p is not None:
            execs[p] = w
    first = min(execs)
    pos: dict[str, float] = {}  # ticker -> currency value
    frozen: set[str] = set()
    cash, nav = 1.0, 1.0
    out = []
    for p in range(first, len(idx)):
        value = cash
        for k in list(pos):
            if k not in frozen and p > first:
                p0, p1 = close[k].iloc[p - 1], close[k].iloc[p]
                if np.isfinite(p1):
                    pos[k] *= p1 / p0
                else:
                    frozen.add(k)  # valued at its last price from here on
            value += pos[k]
        r = value / nav - 1.0
        if p in execs:
            for k in frozen:  # frozen positions become cash at the rebalance
                cash += pos.pop(k)
            frozen.clear()
            drifted = {k: v / value for k, v in pos.items()}
            tgt = {k: v for k, v in execs[p].items() if v != 0 and np.isfinite(close[k].iloc[p])}
            turnover = sum(abs(tgt.get(k, 0.0) - drifted.get(k, 0.0)) for k in set(tgt) | set(drifted))
            r -= turnover * costs_bps / 1e4
            value = nav * (1.0 + r)
            pos = {k: w * value for k, w in tgt.items()}
            cash = value - sum(pos.values())
        nav = value
        out.append(r)
    return pd.Series(out, index=idx[first:])


def test_matches_bruteforce_reference_on_random_data():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-01", periods=120)
    tick = [f"S{i}" for i in range(12)]
    px = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.02, (120, 12)), axis=0)), index=idx, columns=tick)
    px.iloc[70:, 3] = np.nan  # delisting
    px.iloc[30:33, 5] = np.nan  # data gap
    targets = {}
    for d in rebalance_dates(idx, "weekly")[:-1]:
        w = pd.Series(rng.normal(0, 1, 12), index=tick)
        targets[d] = (w / w.abs().sum() * 1.5).round(4)
    for bps, lag in [(0, 1), (25, 1), (10, 2)]:
        res = simulate(targets, px, costs_bps=bps, execution_lag=lag)
        ref = _reference_simulate(targets, px, bps, lag)
        pd.testing.assert_series_equal(res.daily_returns, ref, check_names=False, check_freq=False, atol=1e-12, rtol=0)


def test_speed_500_names_10_years_monthly():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=2520)
    names = [f"T{i:03d}" for i in range(500)]
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (len(idx), 500)), axis=0)), index=idx, columns=names
    )
    dates = rebalance_dates(idx, "monthly")
    targets = {d: pd.Series(rng.normal(size=500), index=names) / 500 for d in dates}
    t0 = time.perf_counter()
    res = simulate(targets, px)
    elapsed = time.perf_counter() - t0
    assert len(res.daily_returns) > 2400 and len(res.turnover) == len(dates) - 1
    assert elapsed < 2.0, f"simulate took {elapsed:.2f}s"


# ------------------------------------------------------------------------------------------------
# build_quantile_weights
# ------------------------------------------------------------------------------------------------

SIG = pd.Series({f"N{i:02d}": float(i) for i in range(1, 11)})  # N10 best, N01 worst


def test_quintile_long_short_equal_weight_legs():
    w = build_quantile_weights(SIG, n_quantiles=5, style="long_short", selection="quantile", weighting="equal")
    assert w.to_dict() == {"N01": -0.5, "N02": -0.5, "N09": 0.5, "N10": 0.5}
    assert w.sum() == pytest.approx(0.0) and w.abs().sum() == pytest.approx(2.0)
    lo = build_quantile_weights(SIG, n_quantiles=5, style="long_only")
    assert lo.to_dict() == {"N09": 0.5, "N10": 0.5}


def test_nan_excluded_and_too_few_names():
    s = SIG.copy()
    s["N10"] = np.nan
    s["N09"] = np.inf
    w = build_quantile_weights(s, n_quantiles=4, style="long_short")
    # 8 valid names (N01..N08), quartiles of 2
    assert w.to_dict() == {"N01": -0.5, "N02": -0.5, "N07": 0.5, "N08": 0.5}
    assert build_quantile_weights(SIG.iloc[:3], n_quantiles=5).empty


def test_ties_broken_by_ticker():
    s = pd.Series({"D": 1.0, "B": 1.0, "C": 1.0, "A": 1.0})
    # all tied: alphabetical order = best first -> A best, D worst
    w = build_quantile_weights(s, n_quantiles=4, style="long_short")
    assert w.to_dict() == {"A": 1.0, "D": -1.0}
    w2 = build_quantile_weights(s.iloc[::-1], n_quantiles=4, style="long_short")
    assert w2.to_dict() == w.to_dict()  # independent of input order
    assert list(quantile_buckets(s, 2).sort_index().items()) == [("A", 2), ("B", 2), ("C", 1), ("D", 1)]


def test_bucket_sizes_differ_by_at_most_one():
    b = quantile_buckets(SIG, 3)  # 10 names -> sizes 4, 3, 3 (worst bucket takes the extra)
    assert b.value_counts().sort_index().tolist() == [4, 3, 3]
    assert set(b[b == 3].index) == {"N08", "N09", "N10"}


def test_top_n_selection():
    w = build_quantile_weights(SIG, selection="top_n", top_n=3, style="long_short")
    assert w.to_dict() == pytest.approx({"N01": -1 / 3, "N02": -1 / 3, "N03": -1 / 3, "N08": 1 / 3, "N09": 1 / 3, "N10": 1 / 3})
    # long-short never overlaps: at most n // 2 per side
    w = build_quantile_weights(SIG.iloc[:5], selection="top_n", top_n=4, style="long_short")
    assert (w > 0).sum() == 2 and (w < 0).sum() == 2
    with pytest.raises(ValueError):
        build_quantile_weights(SIG, selection="top_n")


def test_value_weighting_with_iterative_cap():
    sig = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    cap = pd.Series({"A": 50.0, "B": 30.0, "C": 10.0, "D": 10.0})
    w = build_quantile_weights(sig, n_quantiles=2, style="long_only", selection="top_n", top_n=4,
                               weighting="value", market_cap=cap)
    assert w.to_dict() == pytest.approx({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1})
    # cap 35%: A -> .35, excess .15 spread pro-rata -> B .39 > cap -> B -> .35;
    # the remaining .30 is split pro-rata between C and D -> .15 each
    w = build_quantile_weights(sig, n_quantiles=2, style="long_only", selection="top_n", top_n=4,
                               weighting="value", market_cap=cap, max_weight=0.35)
    assert w.to_dict() == pytest.approx({"A": 0.35, "B": 0.35, "C": 0.15, "D": 0.15})
    # infeasible cap (4 * .2 < 1): everyone at the cap, 20% left in cash
    w = build_quantile_weights(sig, n_quantiles=2, style="long_only", selection="top_n", top_n=4, max_weight=0.2)
    assert w.to_dict() == pytest.approx({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2})
    with pytest.raises(ValueError):
        build_quantile_weights(sig, weighting="value")


def test_cap_applies_per_leg_for_long_short():
    sig = pd.Series({f"N{i}": float(i) for i in range(1, 9)})
    cap = pd.Series({f"N{i}": float(i) for i in range(1, 9)})
    w = build_quantile_weights(sig, n_quantiles=2, style="long_short", weighting="value",
                               market_cap=cap, max_weight=0.3)
    assert w[w > 0].sum() == pytest.approx(1.0) and w[w < 0].sum() == pytest.approx(-1.0)
    assert w.abs().max() <= 0.3 + 1e-12


def test_signal_and_inverse_vol_weighting():
    w = build_quantile_weights(SIG, selection="top_n", top_n=3, style="long_short", weighting="signal")
    # within each leg the most extreme name gets rank 3: 3/6, 2/6, 1/6
    assert w.to_dict() == pytest.approx(
        {"N10": 3 / 6, "N09": 2 / 6, "N08": 1 / 6, "N01": -3 / 6, "N02": -2 / 6, "N03": -1 / 6}
    )
    vol = pd.Series({"N10": 0.1, "N09": 0.2, "N08": np.nan})
    w = build_quantile_weights(SIG, selection="top_n", top_n=3, style="long_only",
                               weighting="inverse_vol", volatility=vol)
    # N08 has no volatility -> excluded; 1/.1 : 1/.2 = 2 : 1
    assert w.to_dict() == pytest.approx({"N10": 2 / 3, "N09": 1 / 3})


# ------------------------------------------------------------------------------------------------
# quantile analysis / forward returns
# ------------------------------------------------------------------------------------------------


def test_forward_returns_consistent_with_simulate():
    close = _close(A=[10, 10, 11, 12.1, 12.1], B=[10, 10, 9, np.nan, np.nan], C=[np.nan, np.nan, 5, 5, 6])
    fwd = forward_returns_from_close(close, [d0, d2], execution_lag=1)
    # entry at d1 close, exit at d3 close; B frozen at its last price 9; C had no entry price
    assert list(fwd) == [d0]
    assert fwd[d0].to_dict() == pytest.approx({"A": 0.21, "B": -0.1})
    # the same period through the simulator: a 50/50 buy-and-hold book earns the average
    w = pd.Series({"A": 0.5, "B": 0.5})
    res = simulate({d0: w, d2: w}, close, costs_bps=0)
    period = np.prod(1 + res.gross_returns.loc[d2:d3]) - 1
    assert period == pytest.approx(0.5 * 0.21 + 0.5 * (-0.1))
    # lag 0: d0 close -> d2 close
    fwd0 = forward_returns_from_close(close, [d0, d2], execution_lag=0)
    assert fwd0[d0].to_dict() == pytest.approx({"A": 0.1, "B": -0.1})


def test_spearman_basics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_quantile_analysis_planted_monotone_signal():
    rng = np.random.default_rng(3)
    names = [f"S{i:03d}" for i in range(100)]
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    sigs, fwds = {}, {}
    for d in dates:
        s = pd.Series(rng.normal(size=100), index=names)
        sigs[d] = s
        fwds[d] = 0.01 * s + pd.Series(rng.normal(0, 0.02, 100), index=names)
    qa = quantile_analysis(sigs, fwds, n_quantiles=5, periods_per_year=12)
    assert qa.n_quantiles == 5 and len(qa.annual_return_by_quantile_pct) == 5
    assert qa.monotonicity == pytest.approx(1.0)
    assert qa.ic_mean > 0.3 and qa.ic_t_stat > 10 and qa.ic_hit_rate_pct == 100.0
    assert qa.spread_annual_pct == pytest.approx(
        qa.annual_return_by_quantile_pct[-1] - qa.annual_return_by_quantile_pct[0]
    )
    assert qa.spread_annual_pct > 0


def test_quantile_analysis_hand_computed():
    # 6 names, 2 buckets, two dates; forward return = signal / 100 exactly
    names = list("ABCDEF")
    s = pd.Series([1.0, 2, 3, 4, 5, 6], index=names)
    f1 = s / 100  # bottom bucket (A,B,C) mean 2%, top (D,E,F) 5%
    f2 = -s / 100  # reversed on the second date: bottom -2%, top -5%
    t1, t2 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
    qa = quantile_analysis({t1: s, t2: s}, {t1: f1, t2: f2}, n_quantiles=2, periods_per_year=12)
    # annual = (prod(1 + r)) ** (12 / 2) - 1
    bottom = ((1.02 * 0.98) ** 6 - 1) * 100
    top = ((1.05 * 0.95) ** 6 - 1) * 100
    assert qa.annual_return_by_quantile_pct == pytest.approx([bottom, top])
    assert qa.ic_mean == pytest.approx(0.0)  # +1 then -1
    assert qa.ic_hit_rate_pct == 50.0
    assert qa.monotonicity == pytest.approx(-1.0)  # top bucket compounds worse (vol drag)


def test_quantile_analysis_skips_thin_dates_and_raises_when_empty():
    names = list("ABCDE")
    s = pd.Series([1.0, 2, 3, 4, 5], index=names)
    t = pd.Timestamp("2020-01-31")
    with pytest.raises(ValueError, match="at least 6 names"):
        quantile_analysis({t: s}, {t: s / 100}, n_quantiles=2, periods_per_year=12)
