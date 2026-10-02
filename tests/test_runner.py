"""Tests for aitrading.backtest.runner (StrategyRunner) and aitrading.backtest.panel (PreloadedProvider).

Fully offline: a small long-history SyntheticProvider and an injected, deterministic factor loader
that mimics the Kenneth French files (Mkt-RF from the synthetic index, seeded long-short factors, a
constant RF) stand in for the network.
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.metrics import compound
from aitrading.backtest.models import BacktestResult
from aitrading.backtest.panel import PreloadedProvider, slice_panel
from aitrading.backtest.protocols import BacktestRunner
from aitrading.backtest.regression import FACTOR_COLUMNS
from aitrading.backtest.runner import RunDetails, StrategyRunner, composite_signal
from aitrading.core import fields as F
from aitrading.data.base import PricePanel
from aitrading.data.synthetic import SyntheticProvider
from aitrading.screen.engine import evaluate_condition
from aitrading.screen.features import FeatureEngine
from aitrading.screen.spec import Condition, UniverseSpec
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.spec import PortfolioConstruction, SignalComponent, StrategySpec, TimeSeriesRule

START = date(2019, 1, 1)
END = date(2024, 12, 31)
OPEN_UNIVERSE = UniverseSpec(min_price=None, min_avg_dollar_volume_usd_mn=None)
DAILY_RF = 0.0001
MONTHLY_RF = 0.002


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider(n_tickers=60, seed=11, start=date(2017, 1, 2), end=END)


class FakeFrench:
    """Deterministic official-like factor frames; records every (model, frequency) request."""

    def __init__(self, provider: SyntheticProvider, fail: bool = False) -> None:
        self.provider = provider
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, frequency: str) -> pd.DataFrame:
        self.calls.append((model, frequency))
        if self.fail:
            raise RuntimeError("French library offline")
        bench = self.provider.get_benchmark_history(date(2017, 1, 2), END)
        r = (bench / bench.shift(1) - 1.0).dropna()
        rf = DAILY_RF
        if frequency == "monthly":
            r = compound(r, "M")
            rf = MONTHLY_RF
        rng = np.random.default_rng(42 if frequency == "monthly" else 43)
        n = len(r)
        df = pd.DataFrame(
            {
                "Mkt-RF": r.to_numpy() - rf,
                "SMB": rng.normal(0.001, 0.02, n),
                "HML": rng.normal(0.0, 0.02, n),
                "RMW": rng.normal(0.002, 0.015, n),
                "CMA": rng.normal(0.001, 0.012, n),
                "Mom": rng.normal(0.004, 0.035, n),
                "RF": rf,
            },
            index=r.index,
        )
        out = df[FACTOR_COLUMNS[model] + ["RF"]].copy()
        out.attrs["source"] = "Fake French Data Library"
        return out


@pytest.fixture(scope="module")
def french(provider) -> FakeFrench:
    return FakeFrench(provider)


@pytest.fixture(scope="module")
def runner(provider, french) -> StrategyRunner:
    return StrategyRunner(provider, factor_loader=french)


def momentum_spec(**update) -> StrategySpec:
    spec = TEMPLATES["momentum_12_1"].spec()
    spec = spec.model_copy(update={"start": START, "end": END, "universe": OPEN_UNIVERSE,
                                   "portfolio": spec.portfolio.model_copy(update={"n_quantiles": 5}), **update})
    return spec


def low_vol_spec(**update) -> StrategySpec:
    spec = TEMPLATES["low_volatility"].spec()
    return spec.model_copy(update={
        "start": START, "end": END, "universe": OPEN_UNIVERSE,
        "portfolio": PortfolioConstruction(style="long_only", selection="quantile", n_quantiles=5, weighting="value"),
        **update,
    })


@pytest.fixture(scope="module")
def momentum_run(runner) -> tuple[BacktestResult, RunDetails]:
    res = runner.backtest(momentum_spec())
    assert runner.last_run is not None
    return res, runner.last_run


@pytest.fixture(scope="module")
def low_vol_run(runner) -> tuple[BacktestResult, RunDetails]:
    res = runner.backtest(low_vol_spec())
    assert runner.last_run is not None
    return res, runner.last_run


def _features_at(provider, tickers, t, features) -> pd.DataFrame:
    """Recompute features at t straight from the provider (universe with point-in-time market caps)."""
    uni = provider.get_universe(OPEN_UNIVERSE, t)
    uni = uni.reindex([x for x in tickers if x in uni.index])
    return FeatureEngine(provider).build(uni, t, features=set(features)).frame


# ------------------------------------------------------------------------------------------------
# composite_signal
# ------------------------------------------------------------------------------------------------


def test_composite_signal_rank_zscore_direction_and_missing_components():
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, np.nan], "b": [10.0, 40.0, 20.0, 30.0, 5.0],
         "gics_sector": ["X", "X", "Y", "Y", None]},
        index=["P", "Q", "R", "S", "T"],
    )
    rank_a = composite_signal(frame, [SignalComponent(feature="a", direction="higher_is_better")])
    assert rank_a.loc[["P", "Q", "R", "S"]].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert math.isnan(rank_a["T"])
    low = composite_signal(frame, [SignalComponent(feature="a", direction="lower_is_better")])
    assert low.loc[["P", "S"]].tolist() == pytest.approx([1.0, 0.0])
    # weighted average over the components a name has; T only has b
    both = composite_signal(frame, [SignalComponent(feature="a", direction="higher_is_better", weight=3.0),
                                    SignalComponent(feature="b", direction="higher_is_better", weight=1.0)])
    rb = (frame["b"].rank() - 1) / 4
    assert both["P"] == pytest.approx((3 * 0.0 + rb["P"]) / 4)
    assert both["T"] == pytest.approx(rb["T"])
    # z-scores are winsorised at +/-3
    big = pd.DataFrame({"x": [0.0] * 30 + [1000.0]}, index=[f"N{i}" for i in range(31)])
    z = composite_signal(big, [SignalComponent(feature="x", direction="higher_is_better", transform="zscore")])
    assert z.max() == pytest.approx(3.0) and z.min() > -1
    # sector-neutral ranks: within X and within Y; no sector -> no score
    sn = composite_signal(frame, [SignalComponent(feature="b", direction="higher_is_better", sector_neutral=True)])
    assert sn.loc[["P", "Q", "R", "S"]].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert math.isnan(sn["T"])


# ------------------------------------------------------------------------------------------------
# PreloadedProvider
# ------------------------------------------------------------------------------------------------


class CountingProvider:
    """Delegates to a provider and counts calls per method."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.counts: dict[str, int] = {}

    def __getattr__(self, item):
        attr = getattr(self.inner, item)
        if not callable(attr) or not item.startswith("get_"):
            return attr

        def wrapped(*a, **k):
            self.counts[item] = self.counts.get(item, 0) + 1
            return attr(*a, **k)

        return wrapped


def test_preloaded_provider_slices_delegates_and_memoises(provider):
    counting = CountingProvider(provider)
    tickers = provider.tickers[:5]
    s0, e0 = date(2019, 1, 1), date(2020, 12, 31)
    panel = provider.get_price_history(tickers, s0, e0)
    pre = PreloadedProvider(counting, panel, window=(s0, e0))
    assert pre.name == "synthetic" and pre.capabilities == provider.capabilities

    got = pre.get_price_history(tickers[:3], date(2019, 6, 3), date(2019, 9, 30))
    direct = provider.get_price_history(tickers[:3], date(2019, 6, 3), date(2019, 9, 30))
    pd.testing.assert_frame_equal(got.close, direct.close, check_names=False)
    pd.testing.assert_frame_equal(got.volume, direct.volume, check_names=False)
    assert counting.counts.get("get_price_history", 0) == 0  # served from the preload

    # a ticker missing from the preload is fetched once for the whole window, then served locally
    extra = provider.tickers[10]
    pre.get_price_history([extra], date(2019, 6, 3), date(2019, 6, 28))
    pre.get_price_history([extra, tickers[0]], date(2020, 1, 2), date(2020, 3, 31))
    assert counting.counts["get_price_history"] == 1

    # outside the window: delegated unchanged
    pre.get_price_history(tickers[:1], date(2018, 1, 2), date(2019, 3, 1))
    assert counting.counts["get_price_history"] == 2 and pre.delegated_price_requests == 1

    # benchmark: fetched once over the window and sliced
    b1 = pre.get_benchmark_history(date(2019, 2, 1), date(2019, 2, 28))
    b2 = pre.get_benchmark_history(date(2020, 2, 3), date(2020, 2, 28))
    assert counting.counts["get_benchmark_history"] == 1
    pd.testing.assert_series_equal(b2, provider.get_benchmark_history(date(2020, 2, 3), date(2020, 2, 28)), check_names=False)
    assert b1.index.min() >= pd.Timestamp(2019, 2, 1) and b1.index.max() <= pd.Timestamp(2019, 2, 28)

    # snapshots are memoised per (method, tickers, as_of) and returned as copies
    f1 = pre.get_fundamentals(tickers, date(2020, 6, 30))
    f1.iloc[:, :] = np.nan
    f2 = pre.get_fundamentals(tickers, date(2020, 6, 30))
    assert counting.counts["get_fundamentals"] == 1 and f2.notna().any().any()
    pre.get_fundamentals(tickers, date(2020, 7, 31))
    assert counting.counts["get_fundamentals"] == 2
    pre.get_universe(OPEN_UNIVERSE, date(2020, 6, 30))
    pre.get_universe(OPEN_UNIVERSE, date(2020, 6, 30))
    assert counting.counts["get_universe"] == 1


def test_slice_panel_reindexes_columns(provider):
    panel = provider.get_price_history(provider.tickers[:3], date(2019, 1, 1), date(2019, 3, 1))
    out = slice_panel(panel, [provider.tickers[2], "NOPE"], date(2019, 2, 1), date(2019, 2, 28))
    assert list(out.close.columns) == [provider.tickers[2], "NOPE"]
    assert out.close["NOPE"].isna().all() and out.close.index.min() >= pd.Timestamp(2019, 2, 1)


# ------------------------------------------------------------------------------------------------
# Cross-sectional long-short (momentum)
# ------------------------------------------------------------------------------------------------


def test_runner_implements_protocol(runner):
    assert isinstance(runner, BacktestRunner)
    assert runner.provider_name == "synthetic"


def test_momentum_long_short_end_to_end(momentum_run, runner):
    res, run = momentum_run
    assert isinstance(res, BacktestResult)
    assert res.provider == "synthetic" and res.rebalance == "monthly" and res.end == END
    assert {"strategy", "benchmark", "long", "short"} <= set(res.stats)
    assert {"strategy", "benchmark", "long", "short"} <= set(res.returns)
    # stats on DAILY returns, stored series MONTHLY (calendar month-ends)
    assert res.stats["strategy"].periods_per_year == 252.0
    assert all(pd.Timestamp(d) == pd.Timestamp(d) + pd.offsets.MonthEnd(0) for d in res.dates)
    assert all(len(v) == len(res.dates) for v in res.returns.values())
    assert 60 <= len(res.dates) <= 73
    # the stored monthly strategy series compounds the daily one
    daily = run.daily_returns
    monthly = compound(daily, "M")
    stored = pd.Series(res.returns["strategy"], index=pd.DatetimeIndex(res.dates), dtype=float).dropna()
    assert np.allclose(stored.to_numpy(), monthly.reindex(stored.index).to_numpy())
    # long-short book: +1 long leg, -1 short leg at every rebalance
    for t, w in run.target_weights.items():
        assert w[w > 0].sum() == pytest.approx(1.0) and w[w < 0].sum() == pytest.approx(-1.0)
    assert sum(res.latest_holdings.values()) == pytest.approx(0.0, abs=0.05)
    assert any(v < 0 for v in res.latest_holdings.values()) and any(v > 0 for v in res.latest_holdings.values())
    # turnover, regression (self-financing: excess=False), quantiles
    assert res.stats["strategy"].avg_turnover_pct is not None and res.stats["strategy"].avg_turnover_pct > 0
    assert res.regression is not None and res.regression.model == "ff3"
    assert res.regression.factor_source == "Fake French Data Library"
    assert res.quantiles is not None and res.quantiles.n_quantiles == 5
    assert len(res.quantiles.annual_return_by_quantile_pct) == 5
    # provenance
    usage = {d.dataset: d for d in res.data_usage}
    assert {"prices", "universe", "benchmark", "ff_factors_official", "risk_free"} <= set(usage)
    assert all(d.point_in_time for d in res.data_usage)
    assert "survivorship" in usage["universe"].notes
    assert "fundamentals" not in usage  # momentum uses prices only
    # serialisable
    again = BacktestResult.model_validate_json(res.model_dump_json())
    assert again.stats["strategy"].sharpe == res.stats["strategy"].sharpe


def test_survivorship_warning_is_first_and_prominent(momentum_run):
    res, _ = momentum_run
    assert res.warnings[0].startswith("SURVIVORSHIP BIAS")
    assert "as of 2024-12-31" in res.warnings[0]
    from aitrading.strategy.interpret import HeuristicInterpreter, verdict_caps

    # survivorship caps the verdict at "promising"; nothing in a clean run reads as look-ahead
    assert [c for c, _ in verdict_caps(res)] == ["promising"]
    interp, _ = HeuristicInterpreter().interpret(res)
    assert interp.verdict in ("promising", "weak", "likely_spurious", "inconclusive")


def test_run_id_is_deterministic_with_label_suffix(runner, momentum_run):
    res, run = momentum_run
    spec = momentum_spec()
    rid = runner._run_id(spec, START, END, None, tickers=run.universe)
    assert res.run_id == rid and rid.startswith("momentum-12-1-")
    assert runner._run_id(spec, START, END, "First half!", tickers=run.universe) == rid + "-first-half"
    assert runner._run_id(spec.model_copy(update={"costs_bps": 25.0}), START, END, None, tickers=run.universe) != rid
    # the universe is part of the id
    assert runner._run_id(spec, START, END, None, tickers=run.universe[:-1]) != rid


def test_run_id_tells_differently_configured_providers_apart():
    """Regression: the id hashed only the provider NAME, so the same spec on SyntheticProvider seed 7 and seed 8
    (or the free provider on two ticker lists) got the same id and IdeaLab overwrote the first run's files."""
    from types import SimpleNamespace

    spec = momentum_spec()

    def rid(provider, tickers=("AAA", "BBB")) -> str:
        return StrategyRunner(provider, factor_loader=None)._run_id(spec, START, END, None, tickers=list(tickers))

    seed7 = rid(SimpleNamespace(name="synthetic", seed=7, n_tickers=40))
    assert seed7 == rid(SimpleNamespace(name="synthetic", seed=7, n_tickers=40))  # deterministic
    assert seed7 != rid(SimpleNamespace(name="synthetic", seed=8, n_tickers=40))
    assert seed7 != rid(SimpleNamespace(name="synthetic", seed=7, n_tickers=41))
    free_a = rid(SimpleNamespace(name="free", tickers=["AAPL", "MSFT"]))
    assert free_a != rid(SimpleNamespace(name="free", tickers=["AAPL", "NVDA"]))
    assert free_a == rid(SimpleNamespace(name="free", tickers=["MSFT", "AAPL"]))  # order does not matter
    assert seed7 != rid(SimpleNamespace(name="synthetic", seed=7, n_tickers=40), tickers=("AAA",))
    # a provider-supplied fingerprint wins over the attributes
    fp1 = rid(SimpleNamespace(name="vendor", fingerprint=lambda: "account-1", seed=1))
    assert fp1 == rid(SimpleNamespace(name="vendor", fingerprint=lambda: "account-1", seed=2))
    assert fp1 != rid(SimpleNamespace(name="vendor", fingerprint="account-2"))


def test_short_leg_is_the_shorted_basket_return(momentum_run):
    res, run = momentum_run
    lng = pd.Series(res.returns["long"], dtype=float)
    sht = pd.Series(res.returns["short"], dtype=float)
    strat = pd.Series(res.returns["strategy"], dtype=float)
    ok = lng.notna() & sht.notna() & strat.notna()
    spread = (lng - sht)[ok]
    # long - short tracks the strategy (differences: compounding / drift and the short leg's costs)
    assert np.corrcoef(spread, strat[ok])[0, 1] > 0.95


def test_target_portfolio_equals_the_backtest_weights(runner, momentum_run):
    _, run = momentum_run
    spec = momentum_spec()
    for t in (run.rebalance_dates[3], run.rebalance_dates[len(run.rebalance_dates) // 2], run.rebalance_dates[-1]):
        live = runner.target_portfolio(spec, t.date())
        pd.testing.assert_series_equal(live.sort_index(), run.target_weights[t].sort_index(), check_names=False)


# ------------------------------------------------------------------------------------------------
# Long-only, value weighted (low volatility)
# ------------------------------------------------------------------------------------------------


def test_low_vol_long_only_top_quintile_value_weighted(provider, low_vol_run, runner):
    res, run = low_vol_run
    assert "long" not in res.stats and "short" not in res.stats
    assert res.regression is not None  # excess=True (long-only)
    assert res.quantiles is not None
    t = run.rebalance_dates[len(run.rebalance_dates) // 2]
    w = run.target_weights[t]
    assert (w > 0).all() and w.sum() == pytest.approx(1.0)
    frame = _features_at(provider, run.target_weights[t].index.tolist(), t.date(), {"volatility_60d_pct", "market_cap_usd_bn", "price"})
    cap = frame["market_cap_usd_bn"].reindex(w.index)
    assert np.allclose(w.to_numpy(), (cap / cap.sum()).to_numpy())
    # the held names are the lowest-volatility quintile of the eligible names
    allf = _features_at(provider, provider.get_universe(OPEN_UNIVERSE, t.date()).index.tolist(), t.date(), {"volatility_60d_pct", "price"})
    vol = allf["volatility_60d_pct"].dropna()
    assert vol.reindex(w.index).max() <= vol.quantile(0.25)
    # idle cash earns RF only when there is idle cash: a fully invested book earns ~ the simulated return
    usage = {d.dataset for d in res.data_usage}
    assert "risk_free" in usage
    live = runner.target_portfolio(low_vol_spec(), t.date())
    pd.testing.assert_series_equal(live.sort_index(), w.sort_index(), check_names=False)


# ------------------------------------------------------------------------------------------------
# Screen kind (the canonical dislocation conditions)
# ------------------------------------------------------------------------------------------------


def test_screen_kind_holds_exactly_the_names_passing_every_condition(provider, runner):
    base = TEMPLATES["dislocation_screen"].spec()
    spec = base.model_copy(update={"start": date(2023, 1, 1), "end": END, "universe": OPEN_UNIVERSE})
    res = runner.backtest(spec)
    run = runner.last_run
    assert res.quantiles is None and "long" not in res.stats
    held_dates = [t for t, w in run.target_weights.items() if len(w)]
    checked = 0
    for t in held_dates[:3] + [list(run.target_weights)[-1]]:
        w = run.target_weights[t]
        uni = provider.get_universe(OPEN_UNIVERSE, t.date())
        frame = FeatureEngine(provider).build(uni, t.date(), features=spec.features() | {"price"}).frame
        mask = pd.Series(True, index=frame.index)
        for c in spec.filters:
            mask &= evaluate_condition(c, frame)
        mask &= frame["price"].notna()
        assert sorted(w.index) == sorted(frame.index[mask])
        if len(w):
            assert np.allclose(w.to_numpy(), 1.0 / len(w))  # equal weighted, long-only
            checked += 1
    if not held_dates:
        assert any("no name passed the screen" in x for x in res.warnings)
    # the screen needs fundamentals and short interest: both recorded as point-in-time
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["fundamentals"].point_in_time and usage["short_interest"].point_in_time


def test_screen_kind_relaxed_holds_names_with_value_weights(runner):
    spec = StrategySpec(
        name="cheap_uptrend", idea="cheap stocks in an uptrend", kind="screen", start=date(2021, 1, 1), end=END,
        universe=OPEN_UNIVERSE,
        filters=[Condition(feature="fcf_yield_pct", op=">", value=2.0), Condition(feature="price_vs_sma_200_pct", op=">", value=0.0)],
        portfolio=PortfolioConstruction(style="long_only", weighting="value", max_weight=0.25),
    )
    res = runner.backtest(spec)
    run = runner.last_run
    sizes = [len(w) for w in run.target_weights.values()]
    assert max(sizes) >= 2
    for w in run.target_weights.values():
        if len(w):
            assert (w > 0).all() and w.max() <= 0.25 + 1e-12 and w.sum() <= 1.0 + 1e-9
    assert res.stats["strategy"].n_periods > 200


# ------------------------------------------------------------------------------------------------
# Time-series rule
# ------------------------------------------------------------------------------------------------


def _trend_spec(ticker: str, start: date = START, **rule) -> StrategySpec:
    return StrategySpec(
        name="trend_200", idea=f"long {ticker} above its 200-day", kind="time_series", start=start, end=END,
        time_series=TimeSeriesRule(assets=[ticker], entry=[Condition(feature="price_vs_sma_200_pct", op=">", value=0.0)], **rule),
        benchmark=ticker, attribution_model="capm",
    )


def test_time_series_trend_rule_on_a_synthetic_ticker(provider, runner):
    ticker = provider.get_universe(OPEN_UNIVERSE, END).index[3]
    spec = _trend_spec(ticker)
    res = runner.backtest(spec)
    run = runner.last_run
    assert not any("SURVIVORSHIP" in w for w in res.warnings)  # explicit assets, not a universe
    values = {float(w.get(ticker, 0.0)) for w in run.target_weights.values()}
    assert values <= {0.0, 1.0} and values == {0.0, 1.0}
    for t in run.rebalance_dates[::9]:
        f = FeatureEngine(provider).build(provider.get_universe(None, t.date()).loc[[ticker]], t.date(),
                                          features={"price_vs_sma_200_pct", "price"}).frame
        expected = 1.0 if f["price_vs_sma_200_pct"].iloc[0] > 0 else 0.0
        assert float(run.target_weights[t].get(ticker, 0.0)) == expected
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["benchmark"].source.endswith(ticker)
    assert res.regression is not None and res.regression.model == "capm"
    # idle cash earns the daily RF: on a fully flat day the strategy return is exactly RF
    sim = run.simulation
    flat_days = [d for d in run.daily_returns.index[5:] if abs(sim.gross_returns.get(d, 1.0)) == 0.0
                 and sim.costs.get(d, 0.0) == 0.0]
    assert flat_days and all(run.daily_returns[d] == pytest.approx(DAILY_RF) for d in flat_days[:20])


def test_time_series_short_when_flat_and_exit_state(provider, runner):
    ticker = provider.get_universe(OPEN_UNIVERSE, END).index[3]
    short = _trend_spec(ticker, start=date(2021, 1, 1), when_flat="short")
    runner.backtest(short)
    values = {float(w.get(ticker, 0.0)) for w in runner.last_run.target_weights.values()}
    assert values == {-1.0, 1.0}
    # an exit rule makes the position sticky: once in, stay until the exit holds
    sticky = _trend_spec(ticker, start=date(2021, 1, 1), exit=[Condition(feature="price_vs_sma_50_pct", op="<", value=-8.0)])
    runner.backtest(sticky)
    run = runner.last_run
    longs = [float(w.get(ticker, 0.0)) for w in run.target_weights.values()]
    assert 1.0 in longs
    t = run.rebalance_dates[-2]
    assert runner.target_portfolio(sticky, t.date()).get(ticker, 0.0) == run.target_weights[t].get(ticker, 0.0)


def test_short_book_earns_rf_on_capital_plus_short_proceeds(provider, runner):
    """Regression: idle cash was clipped at 1, so a -1 short earned rf on 1 instead of on 1 + the short
    proceeds: the short state returned rf - r instead of 2 rf - r (its excess return -r instead of -(r - rf))."""
    ticker = provider.get_universe(OPEN_UNIVERSE, END).index[3]
    runner.backtest(_trend_spec(ticker, start=date(2021, 1, 1), when_flat="short"))
    run = runner.last_run
    sim = run.simulation
    execs = pd.Series(sorted(sim.weights_history))
    n_short = 0
    for d in run.daily_returns.index[1:]:
        before = execs[execs < d]
        if not len(before):
            continue
        book = float(sim.weights_history[before.iloc[-1]].sum())
        cash = max(0.0, 1.0 - book)
        assert run.daily_returns[d] == pytest.approx(sim.daily_returns[d] + cash * DAILY_RF, abs=1e-15)
        n_short += book < 0
        if book < 0:
            assert cash == pytest.approx(2.0)
    assert n_short > 20


class LateListing:
    """The base provider with ``ticker``'s prices missing before ``first`` (a later IPO)."""

    def __init__(self, base: SyntheticProvider, ticker: str, first: date) -> None:
        self.base, self.ticker, self.first = base, ticker, pd.Timestamp(first)

    def __getattr__(self, item):
        return getattr(self.base, item)

    def get_price_history(self, tickers, start, end) -> PricePanel:
        p = self.base.get_price_history(tickers, start, end)
        frames = []
        for fr in (p.open, p.high, p.low, p.close, p.volume):
            fr = fr.copy()
            if self.ticker in fr.columns:
                fr.loc[fr.index < self.first, self.ticker] = np.nan
            frames.append(fr)
        return PricePanel(*frames)


def test_time_series_short_never_opens_a_position_on_missing_data(provider, french):
    """Regression: with when_flat='short', an asset whose entry inputs were missing (not listed yet, then the
    200-session warm-up of its 200-day average) was shorted at -1/n on every such date."""
    uni = provider.get_universe(OPEN_UNIVERSE, END).index
    late, other = uni[5], uni[3]
    wrapped = LateListing(provider, late, date(2021, 6, 22))
    r = StrategyRunner(wrapped, factor_loader=french)
    spec = StrategySpec(
        name="trend_pair", idea="long each asset above its 200-day, short below", kind="time_series", start=START, end=END,
        time_series=TimeSeriesRule(assets=[late, other], entry=[Condition(feature="price_vs_sma_200_pct", op=">", value=0.0)],
                                   when_flat="short"),
        attribution_model=None,
    )
    res = r.backtest(spec)
    run = r.last_run
    engine = FeatureEngine(wrapped)
    missing = present = 0
    for t, w in run.target_weights.items():
        frame = engine.build(provider.get_universe(None, t.date()).reindex([late]), t.date(),
                             features={"price_vs_sma_200_pct", "price"}).frame
        x = frame.at[late, "price_vs_sma_200_pct"]
        if np.isfinite(x):
            present += 1
            assert float(w.get(late, 0.0)) == (0.5 if x > 0 else -0.5)
        else:
            missing += 1
            assert float(w.get(late, 0.0)) == 0.0, t  # neither long nor short without a signal
        assert float(w.get(other, 0.0)) in (0.5, -0.5)  # the other asset always has data
    assert missing >= 20 and present >= 20
    assert any(w.startswith(f"time-series rule: {late} had no price or no data for its entry conditions") and
               "neither long nor short" in w for w in res.warnings)
    assert not any("no price on the execution day" in w for w in res.warnings)  # no target in a name without a price
    # the live target follows the same rule
    t_missing = pd.Timestamp("2021-12-31")
    assert float(r.target_portfolio(spec, t_missing.date()).get(late, 0.0)) == 0.0


def test_time_series_on_an_asset_without_prices_fails_clearly(provider, runner):
    spec = _trend_spec("SPY")  # the synthetic market has no SPY
    with pytest.raises(ValueError, match=r"no price data for 1 ticker\(s\) \(SPY\)"):
        runner.backtest(spec)


def test_repeated_dated_warnings_are_compressed():
    from aitrading.backtest.runner import _compress_dated

    ws = [f"2020-0{m}-28: dropped {m} target name(s) with no price on the execution day (X{m})" for m in range(1, 7)]
    out = _compress_dated(["first", *ws, "2021-01-29: 1 held name(s) had no price on the execution day (Y)", "last"])
    assert out[0] == "first" and out[-1] == "last"
    assert out[1:3] == ws[:2]
    assert out[3].startswith("... and 4 more warning(s) like 'dropped # target name(s)'") and "2020-03-28 to 2020-06-28" in out[3]
    assert out[4].startswith("2021-01-29: 1 held name(s)")


def test_runner_step_logic():
    step = StrategyRunner._ts_step
    assert step(False, True, False, False) is True
    assert step(True, False, False, False) is False
    assert step(True, False, False, True) is True  # exit rule present: stay long until it fires
    assert step(True, True, True, True) is False
    assert step(False, True, True, True) is False  # do not enter while the exit condition holds


# ------------------------------------------------------------------------------------------------
# Factor model
# ------------------------------------------------------------------------------------------------


def test_ff3_factor_model_with_injected_official_factors(runner, french):
    spec = TEMPLATES["ff3"].spec().model_copy(update={"start": START, "end": END})
    res = runner.backtest(spec)
    run = runner.last_run
    assert {"Mkt-RF", "SMB", "HML", "strategy", "benchmark"} <= set(res.returns)
    assert {"Mkt-RF", "SMB", "HML", "strategy"} <= set(res.stats)
    assert res.stats["SMB"].periods_per_year == 12.0
    checks = {c.factor: c for c in res.factor_checks}
    assert set(checks) == {"Mkt-RF", "SMB", "HML"}
    # the constructed market tracks the official-like Mkt-RF (both from the same synthetic market)
    assert checks["Mkt-RF"].correlation_with_official > 0.9
    assert checks["Mkt-RF"].n_overlap_periods >= 60
    # strategy = equal-weighted average of the non-market factors, net of costs on the factor-mimicking turnover
    f = run.factor_returns
    ok = f[["SMB", "HML"]].notna().all(axis=1)
    assert np.allclose(f.loc[ok, "strategy_gross"], f.loc[ok, ["SMB", "HML"]].mean(axis=1))
    assert np.allclose(f.loc[ok, "strategy"], f.loc[ok, "strategy_gross"] - f.loc[ok, "costs"])
    assert (f["costs"] >= 0).all() and f["costs"].sum() > 0
    assert res.stats["strategy"].avg_turnover_pct > 0
    assert any("net of 10 bps one-way costs" in w and "execution lag" in w for w in res.warnings)
    assert ("ff3", "monthly") in french.calls
    assert res.regression is not None and res.regression.model == "ff3"
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["fundamentals"].point_in_time and "construction checks" in usage["ff_factors_official"].notes
    # factor-mimicking holdings: a self-financing long-short book
    assert res.latest_holdings and sum(res.latest_holdings.values()) == pytest.approx(0.0, abs=1e-9)
    live = runner.target_portfolio(spec, END)
    assert set(live.index) == set(res.latest_holdings)
    from aitrading.strategy.interpret import HeuristicInterpreter

    interp, _ = HeuristicInterpreter().interpret(res)
    assert interp.verdict in ("robust", "promising", "weak", "likely_spurious", "inconclusive")


def test_capm_and_carhart4_factor_models(runner):
    capm = runner.backtest(TEMPLATES["capm"].spec().model_copy(update={"start": START, "end": END, "costs_bps": 0.0}))
    assert set(c.factor for c in capm.factor_checks) == {"Mkt-RF"}
    s = pd.Series(capm.returns["strategy"], dtype=float)
    m = pd.Series(capm.returns["Mkt-RF"], dtype=float)
    assert np.allclose(s.dropna(), m.dropna())  # without costs the headline is the constructed market
    assert sum(capm.latest_holdings.values()) == pytest.approx(1.0)  # the value-weighted market
    # with costs: buying the market from cash costs 10 bps in the first month, then only re-weighting
    net = runner.backtest(TEMPLATES["capm"].spec().model_copy(update={"start": START, "end": END}))
    sn = pd.Series(net.returns["strategy"], dtype=float).dropna()
    assert sn.iloc[0] == pytest.approx(s.dropna().iloc[0] - 0.001, abs=1e-12)
    assert (sn <= s.dropna() + 1e-15).all()
    c4 = runner.backtest(TEMPLATES["carhart4"].spec().model_copy(update={"start": START, "end": END}))
    assert "Mom" in c4.returns and "Mom" in c4.stats


def test_factor_model_without_official_data_still_constructs(provider):
    r = StrategyRunner(provider, factor_loader=FakeFrench(provider, fail=True))
    res = r.backtest(TEMPLATES["ff3"].spec().model_copy(update={"start": START, "end": END}))
    assert all(c.correlation_with_official is None for c in res.factor_checks)
    assert res.regression is None
    assert any("official ff3 factors unavailable" in w for w in res.warnings)
    assert any("risk-free rate unavailable" in w for w in res.warnings)


def _ff3_spec(**update) -> StrategySpec:
    return TEMPLATES["ff3"].spec().model_copy(update={"start": START, "end": END, **update})


def test_factor_model_charges_costs_on_the_factor_mimicking_turnover(runner):
    """Regression: kind=factor_model ignored spec.costs_bps (identical Sharpe at 0 and 200 bps), so the
    replication suite's cost checks passed vacuously and 'net of costs' / 'after costs' were untrue."""
    gross = runner.backtest(_ff3_spec(costs_bps=0.0))
    g = runner.last_run.factor_returns
    dear = runner.backtest(_ff3_spec(costs_bps=200.0))
    d = runner.last_run.factor_returns
    # the factor series are the research objects (gross, like the official ones): unchanged
    for c in ("Mkt-RF", "SMB", "HML"):
        assert gross.returns[c] == dear.returns[c]
    assert (g["costs"] == 0.0).all() and np.allclose(g["strategy"].dropna(), g["strategy_gross"].dropna())
    # the headline strategy bears 200 bps x the traded notional of each month-end trade
    ok = d["strategy"].notna()
    assert (d.loc[ok, "costs"] > 0).all()
    assert np.allclose(d.loc[ok, "strategy"], g.loc[ok, "strategy"] - d.loc[ok, "costs"])
    assert dear.stats["strategy"].sharpe < gross.stats["strategy"].sharpe
    assert dear.stats["strategy"].avg_turnover_pct == pytest.approx(gross.stats["strategy"].avg_turnover_pct)
    assert dear.run_id != gross.run_id


def test_factor_mimicking_weights_use_the_june_formation_session(provider, runner):
    """Regression: target_portfolio kept the PRIOR year's June formation until calendar June 30, while
    construct_factors forms on the last June session (Fri 2024-06-28; also 2019-06-28), so a paper account
    rebalancing on that date traded last year's factor portfolio for a month."""
    spec = _ff3_spec(costs_bps=0.0)
    runner.backtest(spec)
    run = runner.last_run
    f = run.factor_returns
    close = provider.get_price_history(provider.tickers, date(2017, 1, 2), END).close
    trades = sorted(run.target_weights)
    assert pd.Timestamp("2024-06-28") in trades and pd.Timestamp("2019-06-28") in trades
    checked = 0
    for k, t in enumerate(trades[:-1]):
        nxt = trades[k + 1]
        lab = (nxt + pd.offsets.MonthEnd(0)).normalize()
        if not np.isfinite(f["strategy_gross"].get(lab, np.nan)):
            continue
        w = run.target_weights[t]
        # the traded weights earn exactly the constructed factors' gross return of the next month
        realised = float((w * (close.loc[nxt, w.index] / close.loc[t, w.index] - 1.0)).sum())
        assert realised == pytest.approx(f.at[lab, "strategy_gross"], abs=1e-12), t
        checked += 1
    assert checked >= 50
    # live: on the formation session the NEW formation is traded, the day before still the old one
    on = runner.target_portfolio(spec, date(2024, 6, 28))
    pd.testing.assert_series_equal(on.sort_index(), run.target_weights[pd.Timestamp("2024-06-28")].sort_index(),
                                   check_names=False)
    before = runner.target_portfolio(spec, date(2024, 6, 27))
    assert set(np.sign(before).items()) != set(np.sign(on).items())


def test_factor_model_leaves_out_an_incomplete_last_month(runner):
    """Regression: with the data ending 2024-12-03, a two-session December was reported as a monthly factor
    return (labelled 2024-12-31, annualised at 12 a year, in the attribution)."""
    res = runner.backtest(_ff3_spec(end=date(2024, 12, 3)))
    assert res.dates[-1] == date(2024, 11, 30)
    assert res.stats["SMB"].end <= date(2024, 11, 30)
    assert any("the last month (2024-12) is incomplete" in w for w in res.warnings)
    full = runner.backtest(_ff3_spec())  # data to 2024-12-31: December is complete
    assert full.dates[-1] == date(2024, 12, 31)
    assert not any("is incomplete" in w for w in full.warnings)


class Delisted:
    """The base provider with ``ticker``'s prices missing after ``last`` (it stops trading)."""

    def __init__(self, base: SyntheticProvider, ticker: str, last: date) -> None:
        self.base, self.ticker, self.last = base, ticker, pd.Timestamp(last)

    def __getattr__(self, item):
        return getattr(self.base, item)

    def get_price_history(self, tickers, start, end) -> PricePanel:
        p = self.base.get_price_history(tickers, start, end)
        frames = []
        for fr in (p.open, p.high, p.low, p.close, p.volume):
            fr = fr.copy()
            if self.ticker in fr.columns:
                fr.loc[fr.index > self.last, self.ticker] = np.nan
            frames.append(fr)
        return PricePanel(*frames)


def test_factor_model_panel_books_the_delisting_return(provider, french):
    ticker = provider.get_universe(OPEN_UNIVERSE, END).index[7]
    r = StrategyRunner(Delisted(provider, ticker, date(2022, 3, 15)), factor_loader=french)
    spec = _ff3_spec(delisting_return=-0.3)
    ctx = r._context(spec, START, END, [], with_benchmark=False)
    days = r._month_days(ctx)
    labels = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in days]).normalize()
    close_m, rets, dead = r._monthly_panel(ctx, days, labels, -0.3)
    assert (ticker, pd.Timestamp("2022-03-15")) in dead
    raw = ctx.close[ticker]
    to_last = raw.loc["2022-03-15"] / raw.loc[:"2022-02-28"].dropna().iloc[-1] - 1.0
    assert rets.at[pd.Timestamp("2022-03-31"), ticker] == pytest.approx((1 + to_last) * 0.7 - 1)
    assert np.isnan(rets.at[pd.Timestamp("2022-04-30"), ticker])
    _, rets0, _ = r._monthly_panel(ctx, days, labels, 0.0)
    assert rets0.at[pd.Timestamp("2022-03-31"), ticker] == pytest.approx(to_last)
    res = r.backtest(spec)
    assert any("stopped trading inside the window" in w and ticker in w and "-30.0% delisting return" in w
               for w in res.warnings)


# ------------------------------------------------------------------------------------------------
# No look-ahead
# ------------------------------------------------------------------------------------------------


class FutureShock:
    """The base provider with every price after ``cut`` scrambled (benchmark too)."""

    def __init__(self, base: SyntheticProvider, cut: date) -> None:
        self.base = base
        self.cut = pd.Timestamp(cut)

    def __getattr__(self, item):
        return getattr(self.base, item)

    def _shock(self, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
        after = frame.index > self.cut
        rng = np.random.default_rng(seed)
        factors = np.exp(rng.normal(0.0, 0.3, size=(int(after.sum()), frame.shape[1])))
        out = frame.copy()
        out.loc[after] = frame.loc[after].to_numpy() * np.cumprod(factors, axis=0)
        return out

    def get_price_history(self, tickers, start, end) -> PricePanel:
        p = self.base.get_price_history(tickers, start, end)
        c = self._shock(p.close, 1)
        scale = c / p.close
        return PricePanel(p.open * scale, p.high * scale, p.low * scale, c, p.volume)

    def get_benchmark_history(self, start, end, symbol=None):
        s = self.base.get_benchmark_history(start, end, symbol)
        return self._shock(s.to_frame(), 2).iloc[:, 0]


def test_perturbing_prices_after_a_rebalance_does_not_change_its_weights(provider, momentum_run, french):
    _, run = momentum_run
    cut = run.rebalance_dates[30]
    shocked = StrategyRunner(FutureShock(provider, cut.date()), factor_loader=french)
    shocked.backtest(momentum_spec())
    srun = shocked.last_run
    for t in run.rebalance_dates:
        if t <= cut:
            pd.testing.assert_series_equal(srun.target_weights[t], run.target_weights[t])
    later = [t for t in run.rebalance_dates if t > cut + timedelta(days=60)]
    assert any(not srun.target_weights[t].equals(run.target_weights[t]) for t in later)


# ------------------------------------------------------------------------------------------------
# Costs, presence rules, validation, prices
# ------------------------------------------------------------------------------------------------


def test_costs_reduce_returns(runner):
    res = runner.backtest(momentum_spec(start=date(2022, 1, 1)))
    gross = runner.backtest(momentum_spec(start=date(2022, 1, 1), costs_bps=0.0))
    dear = runner.backtest(momentum_spec(start=date(2022, 1, 1), costs_bps=50.0))
    assert gross.stats["strategy"].total_return_pct > res.stats["strategy"].total_return_pct > dear.stats["strategy"].total_return_pct
    assert gross.stats["strategy"].avg_turnover_pct == pytest.approx(dear.stats["strategy"].avg_turnover_pct)


def test_quantiles_and_regression_presence_rules(provider, french):
    r = StrategyRunner(provider, factor_loader=french)
    # too few names for deciles: no quantile analysis, with a warning
    few = momentum_spec(start=date(2023, 1, 1), filters=[Condition(feature="market_cap_usd_bn", op=">", value=20.0)],
                        portfolio=PortfolioConstruction(n_quantiles=10, style="long_short"))
    res = r.backtest(few)
    assert res.quantiles is None and any("quantile analysis skipped" in w for w in res.warnings)
    # under 24 months: no regression, with a warning
    short = momentum_spec(start=date(2023, 6, 1))
    res = r.backtest(short)
    assert res.regression is None and any("complete month" in w for w in res.warnings)
    # attribution disabled
    res = r.backtest(momentum_spec(start=date(2022, 1, 1), attribution_model=None))
    assert res.regression is None
    # factor data unavailable: regression skipped, rf = 0
    offline = StrategyRunner(provider, factor_loader=FakeFrench(provider, fail=True))
    res = offline.backtest(momentum_spec(start=date(2022, 1, 1)))
    assert res.regression is None
    assert any("factor attribution (ff3) skipped" in w for w in res.warnings)
    assert any("risk-free rate unavailable" in w for w in res.warnings)
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["risk_free"].source == "none" and usage["ff_factors_official"].coverage == "unavailable"


def test_leading_dates_without_history_are_dropped(provider, french):
    r = StrategyRunner(provider, factor_loader=french)
    res = r.backtest(momentum_spec(start=date(2017, 3, 1), end=date(2019, 12, 31)))
    assert any("no portfolio could be formed at the first" in w for w in res.warnings)
    assert not any("price data starts on" in w for w in res.warnings)  # the data does start before 2017-03-01
    assert res.start >= date(2017, 12, 1)  # 12-1 momentum needs a year of prices
    assert all(len(w) for w in r.last_run.target_weights.values())
    early = r.backtest(momentum_spec(start=date(2016, 6, 1), end=date(2019, 12, 31)))
    assert any("price data starts on 2017-01-02, after the requested start 2016-06-01" in w for w in early.warnings)
    assert early.start == res.start


def test_default_window_and_validation(provider, french):
    r = StrategyRunner(provider, factor_loader=french, default_years=3)
    spec = TEMPLATES["short_term_reversal"].spec().model_copy(update={"universe": OPEN_UNIVERSE})
    res = r.backtest(spec)
    assert res.end == END and res.start >= date(2021, 12, 1)
    bad = spec.model_copy(update={"signal": [SignalComponent(feature="no_such_feature", direction="higher_is_better")]})
    with pytest.raises(ValueError, match="no_such_feature"):
        r.backtest(bad)
    with pytest.raises(ValueError, match="invalid strategy spec"):
        r.target_portfolio(bad, END)


def test_progress_and_latest_prices(provider, french):
    lines: list[str] = []
    r = StrategyRunner(provider, factor_loader=french, progress=lines.append)
    r.backtest(momentum_spec(start=date(2023, 1, 1)))
    assert any("rebalance" in x for x in lines) and any("Finished" in x for x in lines)
    tk = provider.tickers[:3]
    px = r.latest_prices([*tk, "NOPE"], date(2024, 6, 15))  # a Saturday: last close on or before
    direct = provider.get_price_history(tk, date(2024, 6, 1), date(2024, 6, 14)).close.iloc[-1]
    assert np.allclose(px[tk].to_numpy(), direct.to_numpy()) and math.isnan(px["NOPE"])


class StaleSnapshots:
    """A synthetic provider that behaves like the free edition: snapshot data is current-only."""

    def __init__(self, base: SyntheticProvider) -> None:
        self.base = base
        self.warnings: list[str] = []

    def __getattr__(self, item):
        return getattr(self.base, item)

    def is_stale(self, as_of: date) -> bool:
        return as_of < date(2024, 12, 20)

    def get_short_interest(self, tickers, as_of):
        if self.is_stale(as_of):
            self.warnings.append("snapshot only")
            return self.base.get_short_interest([], as_of).reindex(list(tickers))
        return self.base.get_short_interest(tickers, as_of)


def test_snapshot_features_without_history_are_flagged(provider, french):
    r = StrategyRunner(StaleSnapshots(provider), factor_loader=french)
    spec = TEMPLATES["short_interest"].spec().model_copy(update={"start": date(2023, 1, 1), "end": END, "universe": OPEN_UNIVERSE})
    res = r.backtest(spec)
    assert any("no point-in-time history in the free edition" in w for w in res.warnings)
    assert any(w.startswith("provider synthetic: snapshot only") for w in res.warnings)
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["short_interest"].point_in_time is False
    from aitrading.strategy.interpret import verdict_caps

    assert verdict_caps(res)[0][0] == "inconclusive"


class StaleEstimates(StaleSnapshots):
    """Like the free edition: at stale dates the estimates are blank except the last earnings date, which
    comes from dated filings (SEC 8-K) and stays point-in-time."""

    def get_estimates(self, tickers, as_of):
        est = self.base.get_estimates(tickers, as_of)
        if self.is_stale(as_of):
            keep = est[[F.LAST_EARNINGS_DATE]] if F.LAST_EARNINGS_DATE in est.columns else None
            est = est.copy()
            for c in est.columns:
                if c != F.LAST_EARNINGS_DATE:
                    est[c] = np.nan
            if keep is not None:
                est[F.LAST_EARNINGS_DATE] = keep[F.LAST_EARNINGS_DATE]
        return est


def test_days_since_last_earnings_is_not_flagged_as_a_snapshot_feature(provider, french):
    """Regression: every feature reading the estimates dataset was flagged 'no point-in-time history' on a
    snapshot-only provider - including days_since_last_earnings, which the free provider fills from SEC 8-K
    dates - and the estimates data-usage entry forced the verdict to 'inconclusive'."""
    from aitrading.strategy.interpret import verdict_caps

    r = StrategyRunner(StaleEstimates(provider), factor_loader=french)
    spec = StrategySpec(name="post_earnings", idea="stocks that reported in the last 30 days", kind="screen",
                        start=date(2023, 1, 1), end=END, universe=OPEN_UNIVERSE,
                        filters=[Condition(feature="days_since_last_earnings", op="<=", value=30.0)])
    res = r.backtest(spec)
    assert not any("no point-in-time history" in w for w in res.warnings)
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["estimates"].point_in_time is True and "point-in-time" in usage["estimates"].notes
    assert "inconclusive" not in [c for c, _ in verdict_caps(res)]
    held = [len(w) for t, w in r.last_run.target_weights.items() if t < pd.Timestamp("2024-12-01")]
    assert held and sum(1 for n in held if n > 0) >= len(held) // 2  # the screen works at historical dates
    # a real snapshot-only estimate feature next to it is still flagged
    mixed = spec.model_copy(update={"filters": [*spec.filters, Condition(feature="num_analysts", op=">=", value=1.0)]})
    res2 = r.backtest(mixed)
    assert any("feature num_analysts has no point-in-time history" in w for w in res2.warnings)
    assert {d.dataset: d for d in res2.data_usage}["estimates"].point_in_time is False


def test_market_cap_features_are_listed_completely():
    """MARKET_CAP_FEATURES must hold every catalog feature computed from the universe market cap."""
    from aitrading.backtest.runner import MARKET_CAP_FEATURES

    sp = SyntheticProvider(n_tickers=30, seed=3, start=date(2023, 1, 2), end=date(2024, 6, 28))
    t = date(2024, 6, 28)
    uni = sp.get_universe(None, t)
    a = FeatureEngine(sp).build(uni, t).frame
    doubled = uni.copy()
    doubled[F.MARKET_CAP] = doubled[F.MARKET_CAP] * 2.0
    b = FeatureEngine(sp).build(doubled, t).frame
    num = [c for c in a.columns if pd.api.types.is_float_dtype(a[c])]
    changed = {c for c in num if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True)}
    assert changed == set(MARKET_CAP_FEATURES)


class Vendor:
    """Synthetic data served as an anonymous vendor (not 'synthetic'): market caps only via universe snapshots.

    ``shock``: from that date on the snapshots report 30% fewer shares for every other name (buybacks);
    ``pit``: the provider's ``point_in_time_market_cap`` declaration (None = undeclared);
    ``split``: (ticker, factor) - that name's prices are divided by ``factor`` (adjusted for a later split).
    """

    name = "vendor"

    def __init__(self, base: SyntheticProvider, *, shock: date | None = None, pit: bool | None = None,
                 split: tuple[str, float] | None = None, as_traded: bool | None = None) -> None:
        self.base, self.shock, self.split = base, shock, split
        self.calls: list[date] = []
        if pit is not None:
            self.point_in_time_market_cap = pit
        if as_traded is not None:
            self.prices_as_traded = as_traded

    def __getattr__(self, item):
        return getattr(self.base, item)

    def get_universe(self, spec, as_of):
        self.calls.append(as_of)
        u = self.base.get_universe(spec, as_of)
        if self.shock is not None and as_of >= self.shock:
            u = u.copy()
            u.loc[u.index[::2], F.MARKET_CAP] *= 0.7
        return u

    def get_price_history(self, tickers, start, end) -> PricePanel:
        p = self.base.get_price_history(tickers, start, end)
        if self.split is None or self.split[0] not in p.close.columns:
            return p
        tk, k = self.split
        frames = []
        for name, fr in zip(("open", "high", "low", "close", "volume"), (p.open, p.high, p.low, p.close, p.volume)):
            fr = fr.copy()
            fr[tk] = fr[tk] * (k if name == "volume" else 1.0 / k)
            frames.append(fr)
        return PricePanel(*frames)


def test_market_caps_are_point_in_time_snapshots_rolled_forward(provider, french):
    """Regression: on every non-synthetic provider the cap at t was END cap x P(t) / P(end) - the END share
    count and later dividends at every date - while data_usage said point-in-time."""
    vendor = Vendor(provider)
    r = StrategyRunner(vendor, factor_loader=french)
    spec = low_vol_spec(start=date(2021, 1, 1))  # value-weighted: needs market caps
    res = r.backtest(spec)
    snapshot_dates = sorted({pd.Timestamp(d) for d in vendor.calls} - {pd.Timestamp(END)})
    assert 6 <= len(snapshot_dates) <= 12  # about two a year, not one per rebalance
    full_ctx = r._context(spec, START, END, [], with_benchmark=False)
    assert all(d.month in (6, 12) and r._is_month_end_session(full_ctx, d) for d in snapshot_dates)
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["market_cap"].point_in_time is True and "snapshot" in usage["market_cap"].source
    from aitrading.strategy.interpret import verdict_caps

    assert "inconclusive" not in [c for c, _ in verdict_caps(res)]
    # the cap at t = snapshot at the latest June / December session x the adjusted-price ratio
    ctx = r._context(spec, date(2021, 1, 1), END, [], with_benchmark=False)
    t, a = pd.Timestamp("2022-10-31"), pd.Timestamp("2022-06-30")
    cap = r._mcap_at(ctx, t)
    snap = provider.get_universe(OPEN_UNIVERSE, a.date())[F.MARKET_CAP].reindex(cap.index)
    px = provider.get_price_history(list(cap.index), a.date(), t.date()).close
    expected = snap * px.loc[t] / px.loc[a]
    ok = expected.notna()
    assert ok.sum() > 20 and np.allclose(cap[ok], expected[ok])
    # no look-ahead: share-count changes reported after a date leave every earlier value weight unchanged
    shocked = StrategyRunner(Vendor(provider, shock=date(2023, 1, 1)), factor_loader=french)
    shocked.backtest(spec)
    base_w, shock_w = r.last_run.target_weights, shocked.last_run.target_weights
    for d in base_w:
        if d < pd.Timestamp("2022-12-30"):
            pd.testing.assert_series_equal(shock_w[d], base_w[d])
    assert any(not shock_w[d].equals(base_w[d]) for d in base_w if d > pd.Timestamp("2023-02-01"))
    # strategies that use no market cap make no snapshot calls at all
    plain = Vendor(provider)
    StrategyRunner(plain, factor_loader=french).backtest(momentum_spec(start=date(2023, 1, 1)))
    assert {pd.Timestamp(d) for d in plain.calls} == {pd.Timestamp(END)}


def test_end_scaled_market_caps_are_reported_as_not_point_in_time(provider, french):
    from aitrading.strategy.interpret import verdict_caps

    r = StrategyRunner(Vendor(provider, pit=False), factor_loader=french)
    res = r.backtest(low_vol_spec(start=date(2022, 1, 1)))
    usage = {d.dataset: d for d in res.data_usage}
    assert usage["market_cap"].point_in_time is False
    assert any(w.startswith("MARKET CAPS NOT POINT-IN-TIME") for w in res.warnings)
    assert verdict_caps(res)[0][0] == "inconclusive"
    # a provider whose snapshot fails falls back to the estimate, flagged the same way
    class Broken(Vendor):
        def get_universe(self, spec, as_of):
            if pd.Timestamp(as_of) != pd.Timestamp(END):
                raise RuntimeError("no history")
            return super().get_universe(spec, as_of)

    res2 = StrategyRunner(Broken(provider), factor_loader=french).backtest(low_vol_spec(start=date(2022, 1, 1)))
    u2 = {d.dataset: d for d in res2.data_usage}["market_cap"]
    assert u2.point_in_time is False and u2.notes.startswith("NOT point-in-time")
    assert any("gave no universe snapshot with market caps" in w for w in res2.warnings)


def test_price_floor_is_not_applied_to_adjusted_prices_at_past_dates(provider, french):
    """Regression: the universe min_price floor ran on split- and dividend-adjusted closes at every past date,
    so a stock that split later (adjusted close under $5 back then) was dropped using splits not yet made."""
    uni = provider.get_universe(OPEN_UNIVERSE, END)
    px = provider.get_price_history(list(uni.index), START, END).close
    tk = str(px.max().sort_values().index[0])  # its 40:1-split-adjusted price stays far below $5
    assert float(px[tk].max()) / 40.0 < 5.0
    floor = UniverseSpec(min_price=5.0, min_avg_dollar_volume_usd_mn=None)
    spec = momentum_spec(start=date(2022, 1, 1), universe=floor)
    adjusted = StrategyRunner(Vendor(provider, pit=True, split=(tk, 40.0)), factor_loader=french)
    res = adjusted.backtest(spec)
    sig = adjusted.last_run.signals
    assert all(tk in s.index for t, s in sig.items() if t < pd.Timestamp(END))  # eligible: the floor is suspended
    assert tk not in sig[pd.Timestamp(END)].index  # the latest data date: adjusted = traded price, floor applies
    assert any(w.startswith("price floor (universe min_price $5) not applied") for w in res.warnings)
    assert "price floor is not applied" in {d.dataset: d for d in res.data_usage}["universe"].notes
    ctx = adjusted._context(spec, date(2022, 1, 1), END, [], with_benchmark=False)
    assert adjusted._universe_filters(ctx, pd.Timestamp("2023-06-30")).min_price is None
    assert adjusted._universe_filters(ctx, pd.Timestamp(END)).min_price == 5.0  # current prices: floor applies
    # prices the stocks really traded at (declared as traded): the floor applies at every date
    traded = StrategyRunner(Vendor(provider, pit=True, split=(tk, 40.0), as_traded=True), factor_loader=french)
    res_t = traded.backtest(spec)
    assert all(tk not in s.index for s in traded.last_run.signals.values())
    assert not any("price floor" in w for w in res_t.warnings)


def test_performance_300_names_ten_years_monthly_under_60s():
    p = SyntheticProvider(n_tickers=300, start=date(2015, 1, 2))
    r = StrategyRunner(p, factor_loader=None)
    t0 = time.perf_counter()
    res = r.backtest(TEMPLATES["momentum_12_1"].spec())
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"10-year monthly backtest of 300 names took {elapsed:.1f}s"
    assert res.stats["strategy"].n_periods > 2400
