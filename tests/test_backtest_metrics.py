"""Tests for return statistics, Newey-West t-stats and factor regressions (closed-form checks)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.metrics import (
    _align_rf,
    compound,
    default_nw_lags,
    drawdown_series,
    infer_periods_per_year,
    max_drawdown,
    newey_west_tstat,
    performance_stats,
)
from aitrading.backtest.models import FactorRegression, PerformanceStats
from aitrading.backtest.regression import FACTOR_COLUMNS, factor_regression, ols_newey_west

ME = pd.date_range("2018-01-31", periods=24, freq="ME")


# ------------------------------------------------------------------------------------------------
# frequency / compounding
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq, ppy", [("B", 252.0), ("W-FRI", 52.0), ("ME", 12.0), ("QE", 4.0), ("YE", 1.0)]
)
def test_infer_periods_per_year(freq, ppy):
    idx = pd.date_range("2010-01-01", periods=40, freq=freq)
    assert infer_periods_per_year(idx) == ppy
    assert infer_periods_per_year(pd.Series(0.0, index=idx.as_unit("s"))) == ppy  # unit-agnostic
    with pytest.raises(ValueError):
        infer_periods_per_year(idx[:1])


def test_compound_monthly_quarterly_and_gaps():
    days = pd.bdate_range("2024-01-01", "2024-03-31")
    r = pd.Series(0.01, index=days)
    m = compound(r, "M")
    n_jan = int((days.month == 1).sum())  # 23 weekdays in Jan 2024
    assert n_jan == 23
    assert list(m.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    assert m.iloc[0] == pytest.approx(1.01**23 - 1)
    q = compound(r, "Q")
    assert q.iloc[0] == pytest.approx(1.01 ** len(days) - 1)
    # compounding monthly results again gives the same quarterly number
    assert compound(m, "Q").iloc[0] == pytest.approx(q.iloc[0])
    # a period without data is dropped, NaNs inside a period are skipped
    gap = pd.Series([0.1, np.nan, 0.1], index=pd.to_datetime(["2024-01-10", "2024-01-11", "2024-03-05"]))
    g = compound(gap, "monthly")
    assert list(g.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-03-31")]
    assert g.iloc[0] == pytest.approx(0.1)
    w = compound(r.iloc[:10], "W")
    assert all(d.dayofweek == 4 for d in w.index)  # weeks labelled by Friday
    df = compound(pd.DataFrame({"a": r, "b": r * 2}), "A")
    assert df.shape == (1, 2)
    with pytest.raises(ValueError):
        compound(r, "fortnight")


# ------------------------------------------------------------------------------------------------
# drawdowns
# ------------------------------------------------------------------------------------------------


def test_known_drawdown_and_duration():
    r = pd.Series([0.5, -0.5, 0.0, 1.0], index=ME[:4])
    # wealth 1.5, .75, .75, 1.5 -> -50%, under water for 2 periods, recovered on the 4th
    np.testing.assert_allclose(drawdown_series(r).to_numpy(), [0.0, -0.5, -0.5, 0.0])
    assert max_drawdown(r) == (pytest.approx(-0.5), 2)
    # a first-period loss is a drawdown from the initial wealth of 1; unrecovered counts to the end
    r2 = pd.Series([-0.1, 0.05], index=ME[:2])
    np.testing.assert_allclose(drawdown_series(r2).to_numpy(), [-0.1, -0.055])
    assert max_drawdown(r2) == (pytest.approx(-0.1), 2)
    assert max_drawdown(pd.Series([0.01, 0.02], index=ME[:2])) == (0.0, 0)


# ------------------------------------------------------------------------------------------------
# Newey-West
# ------------------------------------------------------------------------------------------------


def test_newey_west_lag0_equals_plain_tstat():
    x = pd.Series(np.random.default_rng(1).normal(0.01, 0.05, 200))
    plain = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert newey_west_tstat(x, lags=0) == pytest.approx(plain, rel=1e-12)


def test_newey_west_hand_computed_lag1():
    # x = 1..4: mean 2.5, g0 = 1.25, g1 = .3125, S = 1.25 + 2 * .5 * .3125 = 1.5625
    # var(mean) = S / 4 * 4/3 = 25/48 -> t = 2.5 / sqrt(25/48) = 2 * sqrt(3)
    assert newey_west_tstat(pd.Series([1.0, 2, 3, 4]), lags=1) == pytest.approx(2 * math.sqrt(3))


def test_newey_west_default_lags_and_degenerate():
    assert default_nw_lags(100) == 4  # floor(4 * 1)
    assert default_nw_lags(120) == 4  # floor(4 * 1.2 ** (2/9)) = floor(4.16)
    assert default_nw_lags(1000) == 6  # floor(4 * 10 ** (2/9)) = floor(6.67)
    x = pd.Series(np.random.default_rng(2).normal(size=120))
    assert newey_west_tstat(x) == pytest.approx(newey_west_tstat(x, lags=4))
    assert math.isnan(newey_west_tstat(pd.Series([0.01] * 10)))
    assert math.isnan(newey_west_tstat(pd.Series([0.01])))
    # NaNs are dropped
    assert newey_west_tstat(pd.Series([1.0, np.nan, 2, 3, 4]), lags=1) == pytest.approx(2 * math.sqrt(3))


def test_newey_west_widens_se_for_autocorrelated_series():
    rng = np.random.default_rng(5)
    e = rng.normal(size=600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.7 * ar[t - 1] + e[t]
    x = pd.Series(ar + 0.2)
    assert abs(newey_west_tstat(x)) < abs(newey_west_tstat(x, lags=0))


# ------------------------------------------------------------------------------------------------
# performance_stats
# ------------------------------------------------------------------------------------------------


def test_constant_returns_closed_form():
    r = pd.Series(0.01, index=ME)
    s = performance_stats(r, label="const")
    assert isinstance(s, PerformanceStats)
    assert s.periods_per_year == 12 and s.n_periods == 24
    assert s.start == ME[0].date() and s.end == ME[-1].date()
    assert s.total_return_pct == pytest.approx((1.01**24 - 1) * 100)
    assert s.cagr_pct == pytest.approx((1.01**12 - 1) * 100)  # 2 years
    assert s.volatility_pct == 0.0
    assert s.sharpe is None and s.sortino is None and s.calmar is None  # no variance / no drawdown
    assert s.max_drawdown_pct == 0.0 and s.max_drawdown_duration_periods == 0
    assert s.hit_rate_pct == 100.0
    assert s.best_period_pct == pytest.approx(1.0) and s.worst_period_pct == pytest.approx(1.0)
    assert s.skew is None and s.excess_kurtosis is None and s.mean_return_t_stat is None
    assert s.avg_turnover_pct is None and s.beta_to_benchmark is None
    s.model_dump_json()  # serialisable (no NaN / inf)


def test_alternating_returns_closed_form():
    r = pd.Series([0.1, -0.1] * 12, index=ME)
    s = performance_stats(r, label="alt")
    assert s.total_return_pct == pytest.approx((0.99**12 - 1) * 100)
    assert s.cagr_pct == pytest.approx((0.99**6 - 1) * 100)
    # mean 0, sum of squares 24 * .01 -> var = .24 / 23
    assert s.volatility_pct == pytest.approx(math.sqrt(0.24 / 23) * math.sqrt(12) * 100)
    assert s.sharpe == pytest.approx(0.0, abs=1e-12)
    assert s.sortino == pytest.approx(0.0, abs=1e-12)
    # peak 1.1 after the first month; trough .99 ** 12 at the end, under water 23 months
    assert s.max_drawdown_pct == pytest.approx((0.99**12 / 1.1 - 1) * 100)
    assert s.max_drawdown_duration_periods == 23
    assert s.calmar == pytest.approx((0.99**6 - 1) / (1 - 0.99**12 / 1.1))
    assert s.hit_rate_pct == 50.0
    assert s.best_period_pct == pytest.approx(10.0) and s.worst_period_pct == pytest.approx(-10.0)
    assert s.skew == pytest.approx(0.0, abs=1e-12)
    # two-point symmetric distribution: g2 = -2; G2 = ((n+1) g2 + 6)(n-1) / ((n-2)(n-3))
    assert s.excess_kurtosis == pytest.approx((25 * -2 + 6) * 23 / (22 * 21))


def test_sharpe_sortino_with_risk_free():
    r = pd.Series([0.02, 0.0, 0.03, -0.01] * 6, index=ME)
    rf = pd.Series(0.001, index=ME)
    s = performance_stats(r, label="x", rf=rf)
    ex = r - 0.001
    assert s.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(12))
    dd = math.sqrt((np.minimum(ex, 0) ** 2).mean())
    assert s.sortino == pytest.approx(ex.mean() / dd * math.sqrt(12))
    # rf None -> rf = 0, float rf -> per-period rate
    assert performance_stats(r, label="x").sharpe == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(12))
    assert performance_stats(r, label="x", rf=0.001).sharpe == pytest.approx(s.sharpe)
    # rf forward-filled onto the return dates (a quarterly rf series -> converted to monthly)
    rf_q = pd.Series(0.003, index=pd.date_range("2017-12-31", periods=9, freq="QE"))
    s_q = performance_stats(r, label="x", rf=rf_q)
    exq = r - (1.003 ** (4 / 12) - 1)
    assert s_q.sharpe == pytest.approx(exq.mean() / exq.std(ddof=1) * math.sqrt(12))


def test_daily_risk_free_compounded_onto_monthly_returns():
    days = pd.bdate_range("2018-01-01", "2019-12-31")
    rf_d = pd.Series(0.0001, index=days)
    r = pd.Series([0.02, -0.01, 0.015] * 8, index=ME)
    s = performance_stats(r, label="x", rf=rf_d)
    n_days = pd.Series(1, index=days).resample("ME").sum().to_numpy()
    rf_m = 1.0001**n_days - 1
    ex = r.to_numpy() - rf_m
    assert s.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(12))


def test_month_end_labelled_rf_matches_daily_returns_by_calendar_month():
    # Kenneth French RF sits on calendar month-ends; daily returns inside month m must be charged
    # month m's rate (an as-of join charged them month m-1's: Feb got .01, Mar got .02)
    days = pd.bdate_range("2024-01-01", "2024-03-31")  # 23 / 21 / 21 weekdays
    rf = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]))
    a = _align_rf(rf, days, 252.0)
    per_day = {1: 1.01 ** (12 / 252) - 1, 2: 1.02 ** (12 / 252) - 1, 3: 1.03 ** (12 / 252) - 1}
    np.testing.assert_allclose(a.to_numpy(), [per_day[m] for m in days.month], rtol=1e-12)
    monthly = (1 + a).groupby(days.month).prod() - 1
    # 21 trading days x 12/252 = exactly one month for Feb / Mar
    np.testing.assert_allclose(monthly.to_numpy(), [1.01 ** (23 / 21) - 1, 0.02, 0.03], rtol=1e-12)
    # FRED-style labels (1st of the month) mean the same months
    rf_fred = pd.Series(rf.to_numpy(), index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    pd.testing.assert_series_equal(_align_rf(rf_fred, days, 252.0), a)
    # end to end: Sharpe of zero daily returns uses the same-month rates
    r = pd.Series(np.where(np.arange(len(days)) % 2 == 0, 0.01, -0.01), index=days)
    ex = r.to_numpy() - a.to_numpy()
    s = performance_stats(r, label="d", rf=rf)
    assert s.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(252))


def test_month_end_rf_matches_monthly_returns_on_last_trading_day():
    bme = pd.date_range("2021-01-01", periods=24, freq="BME")  # 2021-01-29, 2021-02-26, ...
    assert bme[0] == pd.Timestamp("2021-01-29")
    me = pd.date_range("2021-01-31", periods=24, freq="ME")
    rf = pd.Series(np.linspace(0.0001, 0.0040, 24), index=me)
    r = pd.Series([0.02, -0.01, 0.015, 0.0] * 6, index=bme)
    np.testing.assert_allclose(_align_rf(rf, bme, 12.0).to_numpy(), rf.to_numpy(), rtol=1e-12)
    ex = r.to_numpy() - rf.to_numpy()  # same calendar month, not the previous one
    s = performance_stats(r, label="m", rf=rf)
    assert s.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * math.sqrt(12))
    dd = math.sqrt(float(np.mean(np.minimum(ex, 0) ** 2)))
    assert s.sortino == pytest.approx(ex.mean() / dd * math.sqrt(12))
    # rf published with a lag (ends 2 months early): the latest available rate is carried
    late = _align_rf(rf.iloc[:-2], bme, 12.0)
    assert late.iloc[-1] == pytest.approx(rf.iloc[-3]) and late.iloc[-2] == pytest.approx(rf.iloc[-3])


def test_month_end_rf_compounded_onto_quarterly_returns_on_last_trading_day():
    # 2022-12-31 is a Saturday: the Q4 return is dated 2022-12-30 and must own December's rf
    bqe = pd.date_range("2022-01-01", periods=8, freq="BQE")
    assert pd.Timestamp("2022-12-30") in bqe
    me = pd.date_range("2022-01-31", periods=24, freq="ME")
    rf = pd.Series(0.001 * np.arange(1, 25), index=me)
    a = _align_rf(rf, bqe, 4.0)
    expected = (1 + rf).groupby(np.arange(24) // 3).prod().to_numpy() - 1
    np.testing.assert_allclose(a.to_numpy(), expected, rtol=1e-12)


def test_benchmark_beta_tracking_error_ir_and_turnover():
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0.01, 0.04, 24), index=ME)
    active = pd.Series(rng.normal(0.002, 0.01, 24), index=ME)
    r = 1.5 * b + active
    s = performance_stats(r, label="x", benchmark=b, turnover=pd.Series([0.2, 0.4, np.nan]))
    assert s.beta_to_benchmark == pytest.approx(np.cov(r, b)[0, 1] / b.var())
    d = r - b
    assert s.tracking_error_pct == pytest.approx(d.std(ddof=1) * math.sqrt(12) * 100)
    assert s.information_ratio == pytest.approx(d.mean() / d.std(ddof=1) * math.sqrt(12))
    assert s.avg_turnover_pct == pytest.approx(30.0)
    # exact linear relation: beta 2, no active variance -> IR undefined, TE 0
    s2 = performance_stats(2 * b, label="x", benchmark=b)
    assert s2.beta_to_benchmark == pytest.approx(2.0)
    # benchmark given at daily frequency is compounded onto the monthly return dates
    days = pd.bdate_range("2018-01-01", "2019-12-31")
    bd = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    bm = compound(bd, "M")
    rm = 0.8 * bm + pd.Series(rng.normal(0, 0.005, 24), index=bm.index)
    s3 = performance_stats(rm, label="x", benchmark=bd)
    assert s3.beta_to_benchmark == pytest.approx(performance_stats(rm, label="x", benchmark=bm).beta_to_benchmark)


def test_t_stat_ppy_override_and_nan_handling():
    r = pd.Series([0.02, np.nan, 0.0, 0.03, -0.01] * 5, index=pd.date_range("2018-01-31", periods=25, freq="ME"))
    s = performance_stats(r, label="x", periods_per_year=12)
    clean = r.dropna()
    assert s.n_periods == 20
    assert s.mean_return_t_stat == pytest.approx(newey_west_tstat(clean))
    days = pd.bdate_range("2020-01-01", periods=10)
    s_d = performance_stats(pd.Series([0.001, -0.002] * 5, index=days), label="d")
    assert s_d.periods_per_year == 252
    assert s_d.skew is not None  # n = 10 >= 8
    s_short = performance_stats(pd.Series([0.01, 0.02, -0.01], index=days[:3]), label="s")
    assert s_short.skew is None and s_short.excess_kurtosis is None  # n < 8


def test_wipeout_and_errors():
    r = pd.Series([0.1, -1.0, 0.5], index=ME[:3])
    s = performance_stats(r, label="ruin")
    assert s.total_return_pct == pytest.approx(-100.0) and s.cagr_pct == -100.0
    assert s.max_drawdown_pct == pytest.approx(-100.0)
    with pytest.raises(ValueError, match="empty or all-NaN"):
        performance_stats(pd.Series(dtype=float), label="e")
    with pytest.raises(ValueError, match="empty or all-NaN"):
        performance_stats(pd.Series([np.nan, np.nan], index=ME[:2]), label="e")
    with pytest.raises(ValueError, match="at least 2"):
        performance_stats(pd.Series([0.01], index=ME[:1]), label="e")
    with pytest.raises(ValueError, match="infinite"):
        performance_stats(pd.Series([0.01, np.inf], index=ME[:2]), label="e")


# ------------------------------------------------------------------------------------------------
# OLS / factor regression
# ------------------------------------------------------------------------------------------------


def test_ols_recovers_known_betas():
    rng = np.random.default_rng(11)
    n = 2000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    y = 0.5 + 0.9 * X["x1"] - 0.3 * X["x2"] + rng.normal(0, 0.1, n)
    coef, t, r2, nobs = ols_newey_west(y, X)
    assert list(coef.index) == ["const", "x1", "x2"]
    np.testing.assert_allclose(coef.to_numpy(), [0.5, 0.9, -0.3], atol=0.01)
    assert nobs == n and r2 > 0.98 and (t.abs() > 50).all()


def test_ols_exact_fit_and_alignment():
    x = pd.DataFrame({"x": [0.0, 1, 2, 3, 4, 5]}, index=list("abcdef"))
    y = pd.Series([1.0, 3, 5, 7, 9, np.nan], index=list("abcdef"))
    coef, t, r2, n = ols_newey_west(y.drop("a"), x)  # 'a' missing in y, 'f' is NaN -> 4 rows
    assert n == 4
    assert coef["const"] == pytest.approx(1.0) and coef["x"] == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)
    assert t.isna().all()  # zero residuals -> t-stats undefined
    with pytest.raises(ValueError):
        ols_newey_west(pd.Series([1.0, 2.0]), pd.DataFrame({"x": [1.0, 2.0]}))


def test_ols_intercept_only_lag0_equals_plain_tstat():
    y = pd.Series(np.random.default_rng(9).normal(0.01, 0.05, 150))
    coef, t, _, _ = ols_newey_west(y, pd.DataFrame(index=y.index), lags=0)
    plain = y.mean() / (y.std(ddof=1) / math.sqrt(len(y)))
    assert coef["const"] == pytest.approx(y.mean())
    assert t["const"] == pytest.approx(plain)
    # and matches the metrics implementation for any lag
    _, t4, _, _ = ols_newey_west(y, pd.DataFrame(index=y.index), lags=4)
    assert t4["const"] == pytest.approx(newey_west_tstat(y, lags=4))


def _factors(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.007, 0.045, n),
            "SMB": rng.normal(0.002, 0.03, n),
            "HML": rng.normal(0.003, 0.03, n),
            "RMW": rng.normal(0.003, 0.02, n),
            "CMA": rng.normal(0.002, 0.02, n),
            "Mom": rng.normal(0.006, 0.04, n),
            "RF": np.full(n, 0.001),
        },
        index=idx,
    )


def test_factor_regression_ff3_recovers_alpha_and_betas():
    f = _factors()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.002, len(f))
    r = f["RF"] + 0.002 + 1.1 * f["Mkt-RF"] + 0.4 * f["SMB"] - 0.2 * f["HML"] + noise
    reg = factor_regression(r, f, model="ff3", factor_source="Kenneth French Data Library")
    assert isinstance(reg, FactorRegression)
    assert reg.model == "ff3" and reg.n == 120 and reg.factor_source == "Kenneth French Data Library"
    assert set(reg.betas) == {"Mkt-RF", "SMB", "HML"}
    assert reg.betas["Mkt-RF"] == pytest.approx(1.1, abs=0.02)
    assert reg.betas["SMB"] == pytest.approx(0.4, abs=0.02)
    assert reg.betas["HML"] == pytest.approx(-0.2, abs=0.02)
    # alpha annualised arithmetically: 0.2% per month * 12 = 2.4% a year
    coef, t, r2, _ = ols_newey_west(r - f["RF"], f[["Mkt-RF", "SMB", "HML"]])
    assert reg.alpha_annual_pct == pytest.approx(coef["const"] * 12 * 100)
    assert reg.alpha_annual_pct == pytest.approx(2.4, abs=0.6)
    assert reg.alpha_t_stat == pytest.approx(t["const"]) and reg.alpha_t_stat > 5
    assert reg.r_squared == pytest.approx(r2) and reg.r_squared > 0.95
    reg.model_dump_json()


def test_factor_regression_models_alignment_and_errors():
    f = _factors()
    r = 0.9 * f["Mkt-RF"] + 0.3 * f["Mom"] + np.random.default_rng(2).normal(0, 0.01, len(f))
    # self-financing long-short: no RF subtraction
    reg = factor_regression(r, f.drop(columns="RF"), model="carhart4", factor_source="x", excess=False)
    assert list(reg.betas) == FACTOR_COLUMNS["carhart4"]
    assert reg.betas["Mom"] == pytest.approx(0.3, abs=0.05)
    reg5 = factor_regression(r.iloc[:60], f, model="ff5", factor_source="x")  # inner join -> 60 obs
    assert reg5.n == 60 and set(reg5.betas) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}
    capm = factor_regression(r, f, model="capm", factor_source="x")
    assert list(capm.betas) == ["Mkt-RF"]
    with pytest.raises(ValueError, match="at least 24"):
        factor_regression(r.iloc[:23], f, model="ff3", factor_source="x")
    with pytest.raises(ValueError, match="RF"):
        factor_regression(r, f.drop(columns="RF"), model="ff3", factor_source="x")
    with pytest.raises(ValueError, match="RMW"):
        factor_regression(r, f.drop(columns=["RMW"]), model="ff5", factor_source="x")
    with pytest.raises(ValueError, match="unknown factor model"):
        factor_regression(r, f, model="ff7", factor_source="x")
