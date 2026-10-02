"""Offline tests of the strategy store and the paper-trading account (no broker, no network)."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aitrading.trading.paper as paper_mod
import aitrading.trading.store as store_mod
from aitrading.backtest.metrics import performance_stats
from aitrading.backtest.models import BacktestResult
from aitrading.backtest.protocols import BacktestRunner
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.spec import SignalComponent, StrategySpec
from aitrading.trading import (
    ALREADY_REBALANCED,
    LedgerError,
    PaperAccount,
    StoreError,
    StrategyExistsError,
    StrategyNotFoundError,
    StrategyStore,
    backtest_return_over_window,
    default_store_root,
    is_rebalance_day,
    next_rebalance_date,
    scheduled_rebalance_on_or_before,
    slugify_strategy_name,
)

D = date
START = D(2026, 10, 2)  # Friday, mid-month (month end: Friday 2026-10-30)
HIST = D(2026, 9, 25)  # earlier history, so marks get an anchor price (two business days back)
UTC = timezone.utc


# ------------------------------------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------------------------------------


class FakeRunner:
    """Deterministic BacktestRunner: piecewise-constant target weights and per-date price tables.

    ``weights[d]`` applies from date d onwards; ``prices[ticker][d]`` is the close on d and
    ``latest_prices`` returns the close on or before ``as_of`` (NaN if there is none, or if the
    latest entry is NaN - "unavailable").
    """

    provider_name = "fake"

    def __init__(self, prices=None, weights=None):
        self.prices: dict[str, dict[date, float]] = {t: dict(v) for t, v in (prices or {}).items()}
        self.weights: dict[date, dict[str, float]] = dict(weights or {})
        self.target_calls: list[date] = []
        self.price_calls: list[tuple[tuple[str, ...], date]] = []
        self.fail_target = False

    def set_prices(self, d: date, **px: float) -> None:
        for t, p in px.items():
            self.prices.setdefault(t, {})[d] = p

    def set_weights(self, d: date, **w: float) -> None:
        self.weights[d] = dict(w)

    def backtest(self, spec, *, label=None):  # pragma: no cover - not used by paper trading
        raise AssertionError("paper trading must not run a full backtest")

    def target_portfolio(self, spec, as_of):
        self.target_calls.append(as_of)
        if self.fail_target:
            raise RuntimeError("data provider unavailable")
        keys = [d for d in self.weights if d <= as_of]
        if not keys:
            return pd.Series(dtype=float)
        return pd.Series(self.weights[max(keys)], dtype=float)

    def latest_prices(self, tickers, as_of):
        self.price_calls.append((tuple(tickers), as_of))
        out = {}
        for t in tickers:
            hist = self.prices.get(t, {})
            ds = [d for d in hist if d <= as_of]
            out[t] = hist[max(ds)] if ds else np.nan
        return pd.Series(out, dtype=float)


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float = 1.0) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def make_spec(name: str = "test_strategy", rebalance: str = "monthly", costs_bps: float = 10.0, **kw) -> StrategySpec:
    return StrategySpec(
        name=name,
        idea="Long the strongest 12-1 momentum names",
        kind="cross_sectional",
        rebalance=rebalance,
        costs_bps=costs_bps,
        signal=[SignalComponent(feature="momentum_12_1", direction="higher_is_better")],
        **kw,
    )


def make_backtest(dates: list[date], rets: list[float | None], spec: StrategySpec | None = None, run_id: str = "bt-1") -> BacktestResult:
    spec = spec or make_spec()
    clean = pd.Series([0.0 if r is None else r for r in rets], index=pd.DatetimeIndex(dates))
    stats = {"strategy": performance_stats(clean, label="strategy", periods_per_year=12)}
    return BacktestResult(
        run_id=run_id,
        idea=spec.idea,
        spec=spec.model_dump(mode="json"),
        provider="fake",
        llm="offline",
        start=dates[0],
        end=dates[-1],
        rebalance=spec.rebalance,
        returns={"strategy": list(rets)},
        dates=list(dates),
        stats=stats,
        latest_holdings={"AAA": 0.5, "BBB": 0.5},
        started_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
    )


REAL_UTCNOW = paper_mod._utcnow
SIM_NOW = datetime(2028, 1, 3, 12, 0, tzinfo=UTC)  # simulated "now" after every as_of used below


@pytest.fixture(autouse=True)
def _simulated_now(monkeypatch):
    """Accounts without an explicit clock refuse an as_of in the future; the scenarios below play
    out over the coming months, so the default clock is moved past them."""
    monkeypatch.setattr(paper_mod, "_utcnow", lambda: SIM_NOW)


@pytest.fixture()
def store(tmp_path) -> StrategyStore:
    return StrategyStore(tmp_path / "strategies", clock=Clock(datetime(2026, 10, 2, 9, 0, tzinfo=UTC)))


@pytest.fixture()
def runner() -> FakeRunner:
    r = FakeRunner()
    r.set_prices(HIST, AAA=100.0, BBB=50.0, CCC=20.0, DDD=10.0)
    r.set_prices(START, AAA=100.0, BBB=50.0, CCC=20.0, DDD=10.0)
    r.set_weights(START, AAA=0.5, BBB=0.3, CCC=0.2)
    return r


def ledger_nav(acct: PaperAccount) -> float:
    led = acct.ledger
    return led.cash + sum(sh * led.last_prices[t].price for t, sh in led.positions.items())


def assert_nav_identity(acct: PaperAccount, expected: float | None = None) -> None:
    led = acct.ledger
    nav = ledger_nav(acct)
    if led.nav_history:
        assert nav == pytest.approx(led.nav_history[-1][1], rel=1e-12, abs=1e-6)
    if expected is not None:
        assert nav == pytest.approx(expected, rel=1e-12, abs=1e-6)
    assert acct.nav == pytest.approx(nav, rel=1e-12, abs=1e-6)


# ------------------------------------------------------------------------------------------------
# Rebalance calendar
# ------------------------------------------------------------------------------------------------


def test_fake_runner_implements_protocol():
    assert isinstance(FakeRunner(), BacktestRunner)


@pytest.mark.parametrize(
    "freq, as_of, scheduled, nxt, is_day",
    [
        ("monthly", D(2026, 10, 2), D(2026, 9, 30), D(2026, 10, 30), False),
        ("monthly", D(2026, 10, 30), D(2026, 10, 30), D(2026, 11, 30), True),
        ("monthly", D(2026, 10, 31), D(2026, 10, 30), D(2026, 11, 30), False),  # Saturday after month end
        ("monthly", D(2026, 2, 27), D(2026, 2, 27), D(2026, 3, 31), True),  # Feb 28 2026 is a Saturday
        ("weekly", D(2026, 10, 2), D(2026, 10, 2), D(2026, 10, 9), True),
        ("weekly", D(2026, 10, 4), D(2026, 10, 2), D(2026, 10, 9), False),  # Sunday
        ("weekly", D(2026, 10, 7), D(2026, 10, 2), D(2026, 10, 9), False),
        ("quarterly", D(2026, 10, 2), D(2026, 9, 30), D(2026, 12, 31), False),
        ("quarterly", D(2026, 12, 31), D(2026, 12, 31), D(2027, 3, 31), True),
        ("annual", D(2026, 10, 2), D(2025, 12, 31), D(2026, 12, 31), False),
        ("annual", D(2026, 12, 31), D(2026, 12, 31), D(2027, 12, 31), True),
        ("daily", D(2026, 10, 2), D(2026, 10, 2), D(2026, 10, 5), True),
        ("daily", D(2026, 10, 3), D(2026, 10, 2), D(2026, 10, 5), False),  # Saturday
    ],
)
def test_rebalance_calendar(freq, as_of, scheduled, nxt, is_day):
    assert scheduled_rebalance_on_or_before(as_of, freq) == scheduled
    assert next_rebalance_date(as_of, freq) == nxt
    assert is_rebalance_day(as_of, freq) is is_day


def test_rebalance_calendar_accepts_timestamps_and_strings():
    assert is_rebalance_day(pd.Timestamp("2026-10-30"), "monthly")
    assert is_rebalance_day("2026-10-30", "monthly")
    with pytest.raises(ValueError, match="unknown rebalance frequency"):
        is_rebalance_day(START, "hourly")


# ------------------------------------------------------------------------------------------------
# Paper account: first run, costs, shorts
# ------------------------------------------------------------------------------------------------


def test_first_run_trades_into_target_mid_period(store, runner):
    spec = make_spec()
    acct = PaperAccount(store, "Momentum", initial_capital=100_000.0)
    assert acct.started is None and acct.nav == 100_000.0
    assert not is_rebalance_day(START, "monthly")  # mid-month, but never started -> trade now

    rep = acct.rebalance(runner, spec, START)

    assert rep.rebalanced and rep.trigger == "initial" and rep.skipped_reason is None
    assert rep.nav_before == pytest.approx(100_000.0)
    # sized on the post-cost NAV: N' = N - 10bp x N' (fully invested long-only book)
    n_post = 100_000.0 / 1.001
    assert rep.total_cost == pytest.approx(100_000.0 - n_post, rel=1e-6)
    assert rep.nav_after == pytest.approx(rep.nav_before - rep.total_cost, rel=1e-12)
    assert rep.nav_after == pytest.approx(n_post, rel=1e-6)
    assert {o.ticker for o in rep.orders} == {"AAA", "BBB", "CCC"}
    assert all(o.side == "buy" and o.reason == "rebalance" for o in rep.orders)
    for o in rep.orders:
        assert o.notional == pytest.approx(o.shares * o.price)
        assert o.cost == pytest.approx(o.notional * 10 / 1e4)
    pos = acct.positions
    assert pos["AAA"] == pytest.approx(0.5 * n_post / 100.0, rel=1e-6)
    assert pos["BBB"] == pytest.approx(0.3 * n_post / 50.0, rel=1e-6)
    assert pos["CCC"] == pytest.approx(0.2 * n_post / 20.0, rel=1e-6)
    assert abs(acct.cash) < 0.01  # fully invested, no overdraft
    assert rep.cash_after == pytest.approx(acct.cash)
    assert acct.started == START
    assert rep.next_rebalance == D(2026, 10, 30)
    assert_nav_identity(acct, rep.nav_after)
    # holdings weights equal the target weights after a rebalance
    weights = {h["ticker"]: h["weight"] for h in acct.holdings()}
    assert weights == pytest.approx({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, abs=1e-6)
    assert rep.target_weights == {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    assert runner.target_calls == [START]


def test_costs_default_to_spec_and_can_be_overridden(store, runner):
    spec = make_spec(costs_bps=25.0)
    rep = PaperAccount(store, "spec-costs").rebalance(runner, spec, START)
    assert rep.total_cost == pytest.approx(100_000.0 - 100_000.0 / 1.0025, rel=1e-6)

    free = PaperAccount(store, "free", costs_bps=0.0)
    rep0 = free.rebalance(runner, spec, START)
    assert rep0.total_cost == 0.0 and all(o.cost == 0.0 for o in rep0.orders)
    assert rep0.nav_after == pytest.approx(rep0.nav_before)
    assert free.summary()["costs_bps"] == 0.0

    with pytest.raises(ValueError):
        PaperAccount(store, "bad", costs_bps=-1)
    with pytest.raises(ValueError):
        PaperAccount(store, "bad", initial_capital=0)


def test_long_short_book_cash_accounting(store, runner):
    runner.set_weights(START, AAA=0.5, BBB=0.5, CCC=-0.5, DDD=-0.5)
    spec = make_spec()
    acct = PaperAccount(store, "LS")
    rep = acct.rebalance(runner, spec, START)

    n_post = 100_000.0 / 1.002  # traded notional = 2 x N'
    assert rep.nav_after == pytest.approx(n_post, rel=1e-6)
    pos = acct.positions
    assert pos["CCC"] == pytest.approx(-0.5 * n_post / 20.0, rel=1e-6)
    assert pos["DDD"] == pytest.approx(-0.5 * n_post / 10.0, rel=1e-6)
    sides = {o.ticker: o.side for o in rep.orders}
    assert sides == {"AAA": "buy", "BBB": "buy", "CCC": "sell", "DDD": "sell"}
    # sells are listed (and executed) first
    assert [o.side for o in rep.orders] == ["sell", "sell", "buy", "buy"]
    # short proceeds credited: cash = initial - longs + shorts - costs = N'
    assert acct.cash == pytest.approx(n_post, rel=1e-6)
    assert_nav_identity(acct, rep.nav_after)
    s = acct.summary()
    assert s["gross_exposure_pct"] == pytest.approx(200.0, abs=1e-3)
    assert s["net_exposure_pct"] == pytest.approx(0.0, abs=1e-3)
    assert {h["ticker"]: h["side"] for h in s["holdings"]}["CCC"] == "short"

    # both shorts fall 10%: the book gains 2 x 0.5 x N' x 10%
    d = D(2026, 10, 9)
    runner.set_prices(d, AAA=100.0, BBB=50.0, CCC=18.0, DDD=9.0)
    nav = acct.mark_to_market(runner, d)
    assert nav == pytest.approx(n_post * 1.10, rel=1e-6)
    assert_nav_identity(acct, nav)

    # buying to cover debits cash
    runner.set_weights(d, AAA=1.0)
    rep2 = acct.rebalance(runner, spec, d, force=True)
    covers = [o for o in rep2.orders if o.ticker in ("CCC", "DDD")]
    assert covers and all(o.side == "buy" for o in covers)
    assert "CCC" not in acct.positions and "DDD" not in acct.positions
    assert_nav_identity(acct, rep2.nav_after)
    assert rep2.nav_after == pytest.approx(rep2.nav_before - rep2.total_cost)


def test_whole_shares_truncate_toward_zero(store, runner):
    runner.set_prices(START, AAA=300.0, BBB=70.0, CCC=33.0, DDD=10.0)
    runner.set_weights(START, AAA=0.5, BBB=0.3, CCC=-0.2)
    acct = PaperAccount(store, "whole", allow_fractional=False, costs_bps=0.0)
    acct.rebalance(runner, make_spec(), START)
    pos = acct.positions
    assert all(float(s).is_integer() for s in pos.values())
    assert pos["AAA"] == 166.0  # 50,000 / 300 = 166.67 -> 166
    assert pos["BBB"] == 428.0  # 30,000 / 70 = 428.57 -> 428
    assert pos["CCC"] == -606.0  # -20,000 / 33 = -606.06 -> -606 (toward zero)
    assert_nav_identity(acct, 100_000.0)


# ------------------------------------------------------------------------------------------------
# Paper account: schedule, idempotency, force
# ------------------------------------------------------------------------------------------------


def test_second_rebalance_same_day_is_idempotent(store, runner):
    acct = PaperAccount(store, "idem")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    before = acct.ledger_path.read_bytes()
    n_target, n_prices = len(runner.target_calls), len(runner.price_calls)

    rep = acct.rebalance(runner, spec, START)

    assert rep.skipped_reason == ALREADY_REBALANCED == "already rebalanced today"
    assert not rep.rebalanced and rep.orders == []
    assert rep.nav_before == rep.nav_after == pytest.approx(ledger_nav(acct))
    assert acct.ledger_path.read_bytes() == before
    assert (len(runner.target_calls), len(runner.price_calls)) == (n_target, n_prices)


def test_non_rebalance_day_only_marks_to_market(store, runner):
    acct = PaperAccount(store, "mtm")
    spec = make_spec()
    first = acct.rebalance(runner, spec, START)
    shares = acct.positions
    d = D(2026, 10, 9)
    runner.set_prices(d, AAA=110.0, BBB=45.0, CCC=20.0)
    runner.set_weights(d, AAA=1.0)  # would trade if it were a rebalance day

    rep = acct.rebalance(runner, spec, d)

    assert not rep.rebalanced and rep.orders == [] and rep.trigger is None
    assert rep.skipped_reason.startswith("not a rebalance day")
    assert "2026-10-30" in rep.skipped_reason
    assert runner.target_calls == [START]  # the target is not even computed
    assert acct.positions == shares
    expected = acct.cash + shares["AAA"] * 110 + shares["BBB"] * 45 + shares["CCC"] * 20
    assert rep.nav_before == rep.nav_after == pytest.approx(expected)
    assert rep.nav_after != pytest.approx(first.nav_after)
    led = acct.ledger
    assert led.runs == [(START, True), (d, False)]
    assert [x for x, _ in led.nav_history] == [START, d]
    assert led.last_rebalance == START
    assert_nav_identity(acct, expected)

    # running again on the same non-rebalance day re-marks (one history entry per date)
    runner.set_prices(d, AAA=111.0)
    acct.rebalance(runner, spec, d)
    assert len(acct.ledger.nav_history) == 2
    assert_nav_identity(acct)


def test_force_rebalances_on_any_day(store, runner):
    acct = PaperAccount(store, "force")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    d = D(2026, 10, 14)
    runner.set_prices(d, AAA=100.0, BBB=50.0, CCC=20.0)
    runner.set_weights(d, AAA=0.2, BBB=0.2, CCC=0.6)

    rep = acct.rebalance(runner, spec, d, force=True)

    assert rep.rebalanced and rep.trigger == "forced"
    weights = {h["ticker"]: h["weight"] for h in acct.holdings()}
    assert weights == pytest.approx({"AAA": 0.2, "BBB": 0.2, "CCC": 0.6}, abs=1e-6)
    assert acct.ledger.last_rebalance == d
    # force also re-runs a day that already rebalanced
    rep2 = acct.rebalance(runner, spec, d, force=True)
    assert rep2.rebalanced and rep2.orders == []  # already at target
    assert_nav_identity(acct, rep2.nav_after)


def test_scheduled_month_end_rebalance(store, runner):
    acct = PaperAccount(store, "monthly")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    month_end = D(2026, 10, 30)
    runner.set_prices(month_end, AAA=120.0, BBB=40.0, CCC=25.0, DDD=10.0)
    runner.set_weights(month_end, AAA=0.4, DDD=0.6)

    rep = acct.rebalance(runner, spec, month_end)

    assert rep.rebalanced and rep.trigger == "scheduled" and not rep.warnings
    assert set(acct.positions) == {"AAA", "DDD"}
    by = {o.ticker: o for o in rep.orders}
    assert by["BBB"].side == "sell" and by["CCC"].side == "sell" and by["DDD"].side == "buy"
    assert rep.nav_after == pytest.approx(rep.nav_before - rep.total_cost)
    assert rep.next_rebalance == D(2026, 11, 30)
    assert_nav_identity(acct, rep.nav_after)


def test_missed_rebalance_is_done_late_with_warning(store, runner):
    acct = PaperAccount(store, "late")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    d = D(2026, 11, 3)  # PC was off on 2026-10-30
    runner.set_prices(d, AAA=100.0, BBB=50.0, CCC=20.0)
    runner.set_weights(d, AAA=1.0)
    rep = acct.rebalance(runner, spec, d)
    assert rep.rebalanced and rep.trigger == "late"
    assert any("2026-10-30" in w and "missed" in w for w in rep.warnings)
    assert set(acct.positions) == {"AAA"}


def test_weekend_run_after_month_end_is_not_late(store, runner):
    acct = PaperAccount(store, "weekend")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    rep = acct.rebalance(runner, spec, D(2026, 10, 31))  # Saturday after Friday month end
    assert rep.rebalanced and rep.trigger == "scheduled" and not rep.warnings


def test_initial_rebalance_on_a_weekend_then_daily_schedule(store, runner):
    acct = PaperAccount(store, "daily")
    spec = make_spec(rebalance="daily")
    sat = D(2026, 10, 3)
    assert acct.rebalance(runner, spec, sat).trigger == "initial"
    assert acct.rebalance(runner, spec, D(2026, 10, 4)).skipped_reason.startswith("not a rebalance day")
    assert acct.rebalance(runner, spec, D(2026, 10, 5)).trigger == "scheduled"


def test_runs_must_move_forward(store, runner):
    acct = PaperAccount(store, "chrono")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    before = acct.ledger_path.read_bytes()
    with pytest.raises(ValueError, match="only moves"):
        acct.rebalance(runner, spec, D(2026, 9, 30))
    with pytest.raises(ValueError):
        acct.mark_to_market(runner, D(2026, 10, 1))
    assert acct.ledger_path.read_bytes() == before


def test_future_as_of_is_refused(store, runner, monkeypatch):
    """Regression: a mistyped future date (2027 for 2026) was accepted and then blocked every real
    run until that date, since runs must move forward."""
    clock = Clock(datetime(2026, 10, 2, 12, 0, tzinfo=UTC))
    acct = PaperAccount(store, "typo", clock=clock)
    spec = make_spec()
    with pytest.raises(ValueError, match="in the future"):
        acct.rebalance(runner, spec, D(2027, 10, 2))
    with pytest.raises(ValueError, match="in the future"):
        acct.mark_to_market(runner, D(2027, 10, 2))
    assert not acct.ledger_path.exists() and runner.target_calls == []  # nothing recorded or computed

    assert acct.rebalance(runner, spec, START).trigger == "initial"  # the real run still works
    before = acct.ledger_path.read_bytes()
    with pytest.raises(ValueError, match="in the future"):
        acct.rebalance(runner, spec, D(2026, 10, 30), force=True)
    with pytest.raises(ValueError, match="in the future"):
        acct.mark_to_market(runner, D(2026, 10, 5))
    assert acct.ledger_path.read_bytes() == before

    # one day of slack for time zones ahead of UTC (Asia-Pacific mornings are still "yesterday" in UTC)
    runner.set_prices(D(2026, 10, 3), AAA=101.0)
    assert acct.mark_to_market(runner, D(2026, 10, 3)) > 0
    clock.now = datetime(2026, 10, 4, 23, 30, tzinfo=timezone(timedelta(hours=-5)))  # 2026-10-05 04:30 UTC
    acct.mark_to_market(runner, D(2026, 10, 6))  # UTC date + 1: allowed
    assert acct.ledger.runs[-1][0] == D(2026, 10, 6)
    # the default clock is the real UTC time
    monkeypatch.setattr(paper_mod, "_utcnow", REAL_UTCNOW)
    real = PaperAccount(store, "real clock")
    with pytest.raises(ValueError, match="in the future"):
        real.rebalance(runner, spec, datetime.now(UTC).date() + timedelta(days=2))


def test_rebalance_skipped_when_the_target_cannot_be_priced(store, runner):
    """Regression: a rebalance day with no target prices (provider outage) was recorded as a
    completed rebalance, so the book sat in cash for the whole period."""
    spec = make_spec()
    runner.set_weights(START, AAA=0.5, BBB=0.5)
    month_end = D(2026, 10, 30)
    runner.set_prices(month_end, AAA=np.nan, BBB=np.nan, CCC=np.nan, DDD=np.nan)

    # a new account on an outage day: nothing traded, not started, retried on the next run
    acct = PaperAccount(store, "outage")
    rep = acct.rebalance(runner, spec, month_end)
    assert not rep.rebalanced and rep.orders == [] and rep.trigger is None
    assert "data outage" in rep.skipped_reason and "retried on the next run" in rep.skipped_reason
    assert "2 of 2 target names (100%" in rep.skipped_reason
    assert any("data outage" in w for w in rep.warnings)
    led = acct.ledger
    assert led.started is None and led.last_rebalance is None and led.positions == {}
    assert led.runs == [(month_end, False)] and acct.cash == 100_000.0
    assert acct.summary()["status"] == "not_started"
    assert acct.rebalance(runner, spec, month_end).skipped_reason != ALREADY_REBALANCED  # same-day retry allowed

    nov2 = D(2026, 11, 2)
    runner.set_prices(nov2, AAA=100.0, BBB=50.0)
    rep2 = acct.rebalance(runner, spec, nov2)  # data back: the initial rebalance happens now
    assert rep2.rebalanced and rep2.trigger == "initial" and set(acct.positions) == {"AAA", "BBB"}
    assert acct.started == nov2

    # a running account: the scheduled rebalance is kept pending and done late once prices return
    acct2 = PaperAccount(store, "outage 2")
    acct2.rebalance(runner, spec, START)
    held = acct2.positions
    runner.set_weights(month_end, CCC=0.5, DDD=0.5)
    rep3 = acct2.rebalance(runner, spec, month_end)
    assert not rep3.rebalanced and acct2.positions == held
    assert rep3.nav_before == rep3.nav_after == pytest.approx(ledger_nav(acct2))
    led2 = acct2.ledger
    assert led2.last_rebalance == START and led2.runs[-1] == (month_end, False)
    assert {t: m.missing_since for t, m in led2.last_prices.items()} == {"AAA": month_end, "BBB": month_end}
    s = acct2.summary()
    assert s["next_rebalance"] == "2026-10-30" and s["rebalance_due"] is True
    runner.set_prices(nov2, CCC=20.0, DDD=10.0)
    rep4 = acct2.rebalance(runner, spec, nov2)
    assert rep4.rebalanced and rep4.trigger == "late" and set(acct2.positions) == {"CCC", "DDD"}
    assert_nav_identity(acct2, rep4.nav_after)

    # a forced rebalance during an outage says how to retry
    acct3 = PaperAccount(store, "outage 3")
    acct3.rebalance(runner, spec, START)
    d = D(2026, 10, 14)  # mid-period: only a forced rebalance would trade
    runner.set_prices(d, AAA=np.nan, BBB=np.nan)
    runner.set_weights(d, AAA=0.5, BBB=0.5)
    rep5 = acct3.rebalance(runner, spec, d, force=True)
    assert not rep5.rebalanced and "force=True" in rep5.skipped_reason
    assert acct3.ledger.last_rebalance == START


def test_partial_missing_target_prices_still_rebalance(store, runner):
    """Only a majority of unpriced target weight counts as an outage: half or less still trades."""
    spec = make_spec(costs_bps=0.0)
    runner.set_prices(START, EEE=np.nan)
    runner.set_weights(START, AAA=0.25, BBB=0.25, EEE=-0.5)  # exactly half of the gross weight unpriced
    rep = PaperAccount(store, "half").rebalance(runner, spec, START)
    assert rep.rebalanced and {o.ticker for o in rep.orders} == {"AAA", "BBB"}
    runner.set_weights(START, AAA=0.2, EEE=0.3)  # 60% unpriced
    rep2 = PaperAccount(store, "most").rebalance(runner, spec, START)
    assert not rep2.rebalanced and "60%" in rep2.skipped_reason


def test_summary_shows_a_pending_scheduled_rebalance(store, runner):
    """Regression: next_rebalance was computed from the last run, so a mark-to-market run on or
    after the scheduled date showed the following period although the rebalance was due."""
    spec = make_spec()
    acct = PaperAccount(store, "pending")
    assert acct.summary()["next_rebalance"] is None and acct.summary()["rebalance_due"] is None
    acct.rebalance(runner, spec, START)
    s = acct.summary()
    assert (s["next_rebalance"], s["rebalance_due"]) == ("2026-10-30", False)

    acct.mark_to_market(runner, D(2026, 10, 9))
    assert (acct.summary()["next_rebalance"], acct.summary()["rebalance_due"]) == ("2026-10-30", False)
    acct.mark_to_market(runner, D(2026, 10, 30))  # the scheduled month end, but no rebalance
    s = acct.summary()
    assert (s["last_rebalance"], s["last_run"]) == ("2026-10-02", "2026-10-30")
    assert (s["next_rebalance"], s["rebalance_due"]) == ("2026-10-30", True)

    rep = acct.rebalance(runner, spec, D(2026, 11, 2))
    assert rep.trigger == "late"  # what the summary announced
    s = acct.summary()
    assert (s["next_rebalance"], s["rebalance_due"]) == ("2026-11-30", False)
    assert json.loads(json.dumps(s))["rebalance_due"] is False


def test_min_trade_size_skips_small_orders_but_not_full_exits(store, runner):
    acct = PaperAccount(store, "small")
    spec = make_spec(costs_bps=0.0)
    acct.rebalance(runner, spec, START)
    d = D(2026, 10, 7)
    runner.set_prices(d, AAA=100.0, BBB=50.0, CCC=20.0)
    runner.set_weights(d, AAA=0.5004, BBB=0.2996, CCC=0.2)  # 0.04% moves < 0.1% of NAV
    rep = acct.rebalance(runner, spec, d, force=True)
    assert rep.rebalanced and rep.orders == [] and rep.small_orders_skipped == 2
    assert any("smaller than 0.1% of NAV" in n for n in rep.notes)

    # CCC collapses to a sliver of the book; a full exit is executed however small
    d2 = D(2026, 10, 8)
    runner.set_prices(d2, AAA=100.0, BBB=50.0, CCC=0.002)
    runner.set_weights(d2, AAA=0.5, BBB=0.5)
    rep2 = acct.rebalance(runner, spec, d2, force=True)
    exits = [o for o in rep2.orders if o.ticker == "CCC"]
    assert len(exits) == 1 and exits[0].notional < 0.001 * rep2.nav_before
    assert "CCC" not in acct.positions
    assert_nav_identity(acct, rep2.nav_after)


def test_empty_target_moves_to_cash(store, runner):
    acct = PaperAccount(store, "flat")
    spec = make_spec(costs_bps=0.0)
    acct.rebalance(runner, spec, START)
    runner.set_weights(D(2026, 10, 30))  # flat: no positions
    rep = acct.rebalance(runner, spec, D(2026, 10, 30))
    assert rep.rebalanced and acct.positions == {}
    assert acct.cash == pytest.approx(rep.nav_after)
    assert_nav_identity(acct, rep.nav_after)


# ------------------------------------------------------------------------------------------------
# Paper account: missing prices, delistings, corporate actions
# ------------------------------------------------------------------------------------------------


def test_missing_prices_are_not_traded_and_held_positions_are_carried(store, runner):
    runner.set_prices(START, EEE=np.nan)
    runner.set_weights(START, AAA=0.5, BBB=0.3, EEE=0.2)
    acct = PaperAccount(store, "gaps")
    spec = make_spec()
    rep = acct.rebalance(runner, spec, START)
    assert "EEE" not in acct.positions
    assert any("EEE" in w and "not traded" in w for w in rep.warnings)
    assert acct.cash == pytest.approx(0.2 * rep.nav_after, rel=1e-3)  # weights not renormalised
    assert_nav_identity(acct, rep.nav_after)

    # BBB has no price for a few days: carried at its last price, cannot be sold
    d = D(2026, 10, 9)
    runner.set_prices(d, AAA=105.0, BBB=np.nan)
    bbb = acct.positions["BBB"]
    nav = acct.mark_to_market(runner, d)
    assert nav == pytest.approx(acct.cash + acct.positions["AAA"] * 105.0 + bbb * 50.0)
    assert any("BBB" in w and "carried" in w for w in acct.summary()["warnings"])
    assert acct.ledger.last_prices["BBB"].price == 50.0
    assert acct.ledger.last_prices["BBB"].as_of == START

    runner.set_weights(d, AAA=1.0)
    rep2 = acct.rebalance(runner, spec, d, force=True)
    assert acct.positions["BBB"] == bbb  # not sold
    assert "BBB" not in {o.ticker for o in rep2.orders}
    assert any("BBB" in w and "carried" in w for w in rep2.warnings)
    assert_nav_identity(acct, rep2.nav_after)

    # the price comes back: BBB is sold at the next rebalance
    d2 = D(2026, 10, 12)
    runner.set_prices(d2, AAA=105.0, BBB=48.0)
    rep3 = acct.rebalance(runner, spec, d2, force=True)
    assert "BBB" not in acct.positions
    assert {o.ticker: o.side for o in rep3.orders}["BBB"] == "sell"
    assert_nav_identity(acct, rep3.nav_after)


def test_long_missing_price_is_treated_as_delisting(store, runner):
    spec = make_spec(delisting_return=-0.3)
    acct = PaperAccount(store, "delist", costs_bps=0.0, stale_price_days=10)
    acct.rebalance(runner, spec, START)
    bbb = acct.positions["BBB"]
    runner.set_prices(D(2026, 10, 5), BBB=np.nan)

    # first run without a price: the gap starts here
    rep0 = acct.rebalance(runner, spec, D(2026, 10, 5))
    assert "BBB" in acct.positions and rep0.orders == []
    assert acct.ledger.last_prices["BBB"].missing_since == D(2026, 10, 5)
    assert {h["ticker"]: h["price_missing_since"] for h in acct.holdings()}["BBB"] == "2026-10-05"

    # 10 business days since it went missing: still carried
    d = D(2026, 10, 19)
    rep = acct.rebalance(runner, spec, d)
    assert "BBB" in acct.positions and rep.orders == []
    assert any("BBB" in w and "carried" in w and "missing since the run of 2026-10-05" in w for w in rep.warnings)

    # more than 10 business days: closed at last price x (1 - 30%)
    d2 = D(2026, 10, 20)
    nav_before_close = ledger_nav(acct)
    rep2 = acct.rebalance(runner, spec, d2)
    assert "BBB" not in acct.positions
    (o,) = rep2.orders
    assert o.reason == "delisting" and o.side == "sell" and o.shares == pytest.approx(bbb)
    assert o.price == pytest.approx(50.0 * 0.7)
    assert rep2.nav_after == pytest.approx(nav_before_close - bbb * 50.0 * 0.3)
    assert any("11 business days" in w and "delisted" in w for w in rep2.warnings)
    assert acct.summary()["trades"][0]["reason"] == "delisting"
    assert_nav_identity(acct, rep2.nav_after)


def test_one_day_price_gap_on_a_monthly_run_is_not_a_delisting(store, runner):
    """Regression: the gap is measured from when the price went missing, not from the last run
    that saw a price - a monthly account must not force-close a name over a one-day data gap."""
    spec = make_spec(delisting_return=-0.3)
    runner.set_weights(START, AAA=0.5, BBB=0.5)
    acct = PaperAccount(store, "monthly gap", costs_bps=0.0, stale_price_days=10)
    acct.rebalance(runner, spec, START)
    aaa = acct.positions["AAA"]
    for ts in pd.bdate_range("2026-10-05", "2026-10-29"):
        runner.set_prices(ts.date(), AAA=100.0, BBB=50.0)
    month_end = D(2026, 10, 30)
    runner.set_prices(month_end, AAA=np.nan, BBB=50.0)  # one-day provider gap

    rep = acct.rebalance(runner, spec, month_end)  # 20 business days after the last run

    assert rep.rebalanced and rep.trigger == "scheduled"
    assert not [o for o in rep.orders if o.reason == "delisting"]
    assert acct.positions["AAA"] == aaa  # carried (cannot be traded without a price), not closed
    assert not any("delisted" in w for w in rep.warnings)
    assert any("AAA" in w and "carried" in w for w in rep.warnings)
    assert rep.nav_after == pytest.approx(100_000.0)  # no -30% delisting loss booked
    assert acct.ledger.last_prices["AAA"].missing_since == month_end

    # the price comes back: the gap is cleared
    runner.set_prices(D(2026, 11, 2), AAA=101.0, BBB=50.0)
    acct.mark_to_market(runner, D(2026, 11, 2), spec=spec)
    mark = acct.ledger.last_prices["AAA"]
    assert mark.missing_since is None and mark.price == 101.0 and "AAA" in acct.positions

    # a later gap starts counting afresh from its own first run, not from 2026-10-30
    runner.set_prices(D(2026, 11, 13), AAA=np.nan)
    acct.mark_to_market(runner, D(2026, 11, 13), spec=spec)
    acct.mark_to_market(runner, D(2026, 11, 20), spec=spec)  # 5 business days later
    assert "AAA" in acct.positions
    assert acct.ledger.last_prices["AAA"].missing_since == D(2026, 11, 13)
    raw = json.loads(acct.ledger_path.read_text(encoding="utf-8"))
    assert raw["last_prices"]["AAA"]["missing_since"] == "2026-11-13"  # persisted across runs

    # still missing at the next monthly run, 11 business days after it went missing: delisted
    rep2 = acct.rebalance(runner, spec, D(2026, 11, 30))
    assert "AAA" not in acct.positions
    assert [(o.ticker, o.reason) for o in rep2.orders if o.reason == "delisting"] == [("AAA", "delisting")]
    assert any("AAA has had no price for 11 business days" in w for w in rep2.warnings)
    assert_nav_identity(acct, rep2.nav_after)


def test_delisting_needs_at_least_two_runs_without_a_price(store, runner):
    spec = make_spec(delisting_return=-0.5)
    acct = PaperAccount(store, "two runs", costs_bps=0.0, stale_price_days=0)
    acct.rebalance(runner, spec, START)
    d = D(2026, 12, 15)  # long after the last priced run, but the first run without a price
    runner.set_prices(d, AAA=np.nan, BBB=50.0, CCC=20.0)
    acct.mark_to_market(runner, d, spec=spec)
    assert "AAA" in acct.positions
    acct.mark_to_market(runner, D(2026, 12, 16), spec=spec)  # second run, a business day later
    assert "AAA" not in acct.positions


def test_split_in_adjusted_history_scales_the_position(store, runner):
    acct = PaperAccount(store, "split", costs_bps=0.0)
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    aaa = acct.positions["AAA"]
    nav0 = ledger_nav(acct)

    mark = acct.ledger.last_prices["AAA"]
    assert (mark.price, mark.as_of, mark.anchor_date, mark.anchor_price) == (100.0, START, D(2026, 9, 30), 100.0)

    # 2:1 split: the provider halves all earlier history and AAA now trades at 51
    d = D(2026, 10, 9)
    runner.set_prices(HIST, AAA=50.0)
    runner.set_prices(START, AAA=50.0)
    runner.set_prices(d, AAA=51.0, BBB=50.0, CCC=20.0)
    nav = acct.mark_to_market(runner, d, spec=spec)

    assert acct.positions["AAA"] == pytest.approx(2 * aaa)
    assert nav == pytest.approx(nav0 + 2 * aaa * 1.0)  # a 2% gain, not a 49% loss
    assert any("AAA" in w and "split" in w for w in acct.summary()["warnings"])
    assert_nav_identity(acct, nav)


def test_dividend_adjustment_is_reinvested_quietly(store, runner):
    acct = PaperAccount(store, "dividend", costs_bps=0.0)
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    ccc = acct.positions["CCC"]
    d = D(2026, 10, 9)
    runner.set_prices(HIST, CCC=19.8)  # 1% dividend: the provider scales all earlier history
    runner.set_prices(START, CCC=19.8)
    runner.set_prices(d, AAA=100.0, BBB=50.0, CCC=19.8)
    rep = acct.rebalance(runner, spec, d)
    assert acct.positions["CCC"] == pytest.approx(ccc * 20.0 / 19.8, rel=1e-6)
    assert not rep.warnings and any("CCC" in n for n in rep.notes)
    assert rep.nav_after == pytest.approx(rep.nav_before)
    # the dividend kept the NAV whole although the price fell to 19.8
    assert rep.nav_after == pytest.approx(ledger_nav(acct))
    assert ledger_nav(acct) == pytest.approx(acct.ledger.nav_history[0][1], rel=1e-9)


def test_intraday_run_is_not_mistaken_for_a_corporate_action(store, runner):
    """A run during the session records a partial price; when the final close later replaces it,
    the anchor (two business days back) is unchanged, so nothing is rescaled."""
    acct = PaperAccount(store, "intraday", costs_bps=0.0)
    spec = make_spec()
    acct.rebalance(runner, spec, START)  # AAA filled at 100 (intraday print)
    aaa = acct.positions["AAA"]
    nav0 = ledger_nav(acct)
    runner.set_prices(START, AAA=103.0)  # final close of START
    d = D(2026, 10, 5)
    runner.set_prices(d, AAA=104.0, BBB=50.0, CCC=20.0)
    rep = acct.rebalance(runner, spec, d)
    assert acct.positions["AAA"] == aaa
    assert not rep.warnings and not rep.notes
    assert rep.nav_after == pytest.approx(nav0 + aaa * 4.0)  # fill at 100 -> mark at 104


class _LiveRunner:
    """Fixed target weights; prices from a real ``StrategyRunner`` on the real free provider."""

    provider_name = "free"

    def __init__(self, inner, weights: dict[str, float]):
        self.inner = inner
        self.weights = weights

    def backtest(self, spec, *, label=None):  # pragma: no cover - not used by paper trading
        raise AssertionError("paper trading must not run a full backtest")

    def target_portfolio(self, spec, as_of):
        return pd.Series(self.weights, dtype=float)

    def latest_prices(self, tickers, as_of):
        return self.inner.latest_prices(tickers, as_of)

    def prices_at(self, tickers, dates):
        return self.inner.prices_at(tickers, dates)


def _free_runner(tmp_path: Path, cache, today: date, close: np.ndarray, adj: np.ndarray | None = None) -> _LiveRunner:
    """A real FreeDataProvider (fake Yahoo serving ``close`` / ``adj`` as XYZ's history from
    2026-07-01) sharing the on-disk ``cache`` across runs, like CLI runs on one PC."""
    from test_free_provider import FakeYF, make_sec

    from aitrading.backtest.runner import StrategyRunner
    from aitrading.data.free import FreeDataProvider

    idx = pd.bdate_range("2026-07-01", periods=len(close), name="Date")
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                          "Adj Close": close if adj is None else adj, "Volume": 1e6}, index=idx)
    sec, _ = make_sec(tmp_path, cache=cache)
    prov = FreeDataProvider(["XYZ"], cache=cache, sec=sec, yf_module=FakeYF(frames={"XYZ": frame}),
                            today=lambda: today, max_workers=1, sleep=lambda _s: None)
    return _LiveRunner(StrategyRunner(prov, factor_loader=None, today=today), {"XYZ": 1.0})


@pytest.mark.parametrize("second_run", [D(2026, 9, 15), D(2026, 9, 18)])
def test_split_is_detected_through_the_free_providers_disk_cache(tmp_path, store, second_run):
    """Regression: the anchor re-read went through the same (ticker, start, end) window as when it
    was recorded, which the free provider caches on disk for 7 days once it is in the past - so a
    run within a week of a 2:1 split re-read the pre-split anchor (factor 1) while today's price
    was post-split: a fake -50% loss and no warning."""
    from aitrading.data.cache import DiskCache

    cache = DiskCache(tmp_path / "cache")  # enabled, shared by both runs
    n = len(pd.bdate_range("2026-07-01", "2026-10-01"))
    acct = PaperAccount(store, "split cache", costs_bps=0.0)
    spec = make_spec()
    rep = acct.rebalance(_free_runner(tmp_path, cache, D(2026, 9, 14), np.full(n, 100.0)), spec, D(2026, 9, 14))
    assert rep.rebalanced and acct.positions == {"XYZ": pytest.approx(1000.0)}
    assert acct.ledger.last_prices["XYZ"].anchor_price == 100.0

    # XYZ splits 2:1 before the second run: Yahoo restates the whole history at 50
    nav = acct.mark_to_market(_free_runner(tmp_path, cache, second_run, np.full(n, 50.0)), second_run, spec=spec)
    assert acct.positions["XYZ"] == pytest.approx(2000.0)
    assert nav == pytest.approx(100_000.0)
    assert any("XYZ" in w and "factor 2.0000" in w for w in acct.summary()["warnings"])
    assert_nav_identity(acct, nav)


def test_dividend_is_booked_by_daily_runs_through_the_disk_cache(tmp_path, store):
    """Regression: with daily runs every ex-dividend date fell between two runs less than 7 days
    apart, so the cached anchor hid every dividend and the paper NAV drifted below the backtest."""
    from aitrading.data.cache import DiskCache

    cache = DiskCache(tmp_path / "cache")
    idx = pd.bdate_range("2026-07-01", "2026-10-01")
    acct = PaperAccount(store, "dividend cache", costs_bps=0.0)
    spec = make_spec()
    acct.rebalance(_free_runner(tmp_path, cache, D(2026, 9, 14), np.full(len(idx), 100.0)), spec, D(2026, 9, 14))
    # 1% dividend, ex-date 2026-09-15: the close drops to 99 and Yahoo scales the earlier Adj Close by 0.99
    close = np.where(idx >= pd.Timestamp("2026-09-15"), 99.0, 100.0)
    adj = np.where(idx >= pd.Timestamp("2026-09-15"), 99.0, 99.0)
    rep = acct.rebalance(_free_runner(tmp_path, cache, D(2026, 9, 15), close, adj), spec, D(2026, 9, 15))
    assert acct.positions["XYZ"] == pytest.approx(1000.0 / 0.99, rel=1e-6)
    assert any("XYZ" in x for x in rep.notes) and not rep.warnings
    assert ledger_nav(acct) == pytest.approx(100_000.0, rel=1e-9)


def test_price_table_uses_one_download_and_falls_back_per_date(runner):
    calls: list[tuple[tuple[str, ...], tuple[date, ...]]] = []

    class WithPricesAt(FakeRunner):
        def prices_at(self, tickers, dates):
            calls.append((tuple(tickers), tuple(dates)))
            return pd.DataFrame({t: [self.latest_prices([t], d)[t] for d in dates] for t in tickers},
                                index=pd.DatetimeIndex(dates))

    r = WithPricesAt(prices=runner.prices)
    table = PaperAccount._price_table(r, ["AAA", "BBB", "AAA"], [START, HIST, START])
    assert calls == [(("AAA", "BBB"), (HIST, START))]
    assert table[START].to_dict() == {"AAA": 100.0, "BBB": 50.0} and list(table) == [HIST, START]
    fallback = PaperAccount._price_table(runner, ["AAA", "ZZZ"], [START, HIST])
    assert fallback[HIST]["AAA"] == 100.0 and math.isnan(fallback[START]["ZZZ"])
    assert [c[1] for c in runner.price_calls] == [HIST, START]


def test_whole_share_account_gets_cash_in_lieu(store, runner):
    acct = PaperAccount(store, "lieu", costs_bps=0.0, allow_fractional=False)
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    ccc = acct.positions["CCC"]
    nav0 = ledger_nav(acct)
    runner.set_prices(HIST, CCC=19.8)
    runner.set_prices(START, CCC=19.8)
    d = D(2026, 10, 9)
    runner.set_prices(d, CCC=19.8)
    nav = acct.mark_to_market(runner, d)
    new = acct.positions["CCC"]
    assert float(new).is_integer() and new == float(int(ccc * 20.0 / 19.8))
    assert nav == pytest.approx(nav0)  # the dividend (shares + cash in lieu) offsets the price drop
    assert_nav_identity(acct, nav)


def test_runner_failure_leaves_ledger_untouched(store, runner):
    acct = PaperAccount(store, "fail")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    before = acct.ledger_path.read_bytes()
    runner.fail_target = True
    with pytest.raises(RuntimeError, match="unavailable"):
        acct.rebalance(runner, spec, D(2026, 10, 30))
    assert acct.ledger_path.read_bytes() == before
    runner.fail_target = False
    assert acct.rebalance(runner, spec, D(2026, 10, 30)).rebalanced


def test_target_cleanup_duplicates_and_nan(store, runner):
    class Messy(FakeRunner):
        def target_portfolio(self, spec, as_of):
            return pd.Series([0.3, 0.2, np.nan, 0.0], index=["AAA", "AAA", "BBB", "CCC"])

    messy = Messy(prices=runner.prices)
    acct = PaperAccount(store, "messy", costs_bps=0.0)
    rep = acct.rebalance(messy, make_spec(), START)
    assert rep.target_weights == {"AAA": pytest.approx(0.5)}
    assert set(acct.positions) == {"AAA"}
    assert any("more than once" in w for w in rep.warnings)
    assert any("not numbers" in w for w in rep.warnings)


def test_nav_identity_holds_through_a_random_walk(store):
    rng = np.random.default_rng(7)
    tickers = ["T%02d" % i for i in range(12)]
    px = dict(zip(tickers, rng.uniform(5, 200, len(tickers))))
    r = FakeRunner()
    spec = make_spec(rebalance="weekly", costs_bps=15.0)
    acct = PaperAccount(store, "random walk", allow_fractional=False, initial_capital=250_000)
    days = pd.bdate_range("2026-10-02", "2027-01-29")
    for i, ts in enumerate(days):
        d = ts.date()
        for t in tickers:
            px[t] *= float(np.exp(rng.normal(0, 0.02)))
        quote = {t: (np.nan if rng.random() < 0.03 else px[t]) for t in tickers}
        r.set_prices(d, **quote)
        if i % 5 == 0:
            names = rng.choice(tickers, size=6, replace=False)
            w = rng.normal(0, 0.3, size=6)
            r.set_weights(d, **{n: float(x) for n, x in zip(names, w)})
        rep = acct.rebalance(r, spec, d, force=bool(i % 17 == 0))
        assert rep.nav_after == pytest.approx(rep.nav_before - rep.total_cost, rel=1e-12, abs=1e-6)
        assert_nav_identity(acct, rep.nav_after)
        assert all(float(s).is_integer() for s in acct.positions.values())
    led = acct.ledger
    assert len(led.nav_history) == len(days)
    assert sum(1 for _, reb in led.runs if reb) >= 17  # every Friday plus forced runs
    assert led.total_costs == pytest.approx(sum(t.cost for t in led.trades))


# ------------------------------------------------------------------------------------------------
# Persistence, reset, summary
# ------------------------------------------------------------------------------------------------


def test_ledger_persistence_round_trip(store, runner):
    spec = make_spec()
    acct = PaperAccount(store, "persist", initial_capital=50_000)
    acct.rebalance(runner, spec, START)
    d = D(2026, 10, 9)
    runner.set_prices(d, AAA=101.0, BBB=49.0, CCC=21.0)
    acct.mark_to_market(runner, d)

    raw = json.loads(acct.ledger_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["started"] == "2026-10-02"
    assert raw["runs"] == [["2026-10-02", True], ["2026-10-09", False]]
    assert set(raw["positions"]) == {"AAA", "BBB", "CCC"}
    assert len(raw["trades"]) == 3

    again = PaperAccount(store, "persist", initial_capital=1.0)  # existing ledger wins
    assert again.ledger == acct.ledger
    assert again.ledger.initial_capital == 50_000
    assert again.summary() == acct.summary()
    assert json.loads(json.dumps(again.summary())) is not None  # JSON-serialisable
    # two account objects on one ledger stay consistent (each operation re-reads the file)
    d2 = D(2026, 10, 12)
    runner.set_prices(d2, AAA=102.0)
    again.mark_to_market(runner, d2)
    assert acct.ledger.runs[-1] == (d2, False)
    assert_nav_identity(acct)


def test_ledger_writes_are_atomic(store, runner, monkeypatch):
    acct = PaperAccount(store, "atomic")
    spec = make_spec()
    acct.rebalance(runner, spec, START)
    before = acct.ledger_path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        acct.rebalance(runner, spec, D(2026, 10, 30))
    monkeypatch.undo()

    assert acct.ledger_path.read_bytes() == before
    assert not list(acct.path.glob("*.tmp")) and not list(acct.path.glob(".*"))
    assert acct.ledger.last_rebalance == START  # in-memory state was not advanced either
    assert acct.rebalance(runner, spec, D(2026, 10, 30)).rebalanced


def test_corrupt_or_newer_ledger_is_refused_and_reset_recovers(store, runner):
    spec = make_spec()
    acct = PaperAccount(store, "corrupt")
    acct.rebalance(runner, spec, START)
    acct.ledger_path.write_text("{not json", encoding="utf-8")
    fresh = PaperAccount(store, "corrupt", initial_capital=40_000)  # a new process: constructing works
    for op in (lambda a: a.ledger, lambda a: a.summary(), lambda a: a.holdings(), lambda a: a.nav,
               lambda a: a.rebalance(runner, spec, D(2026, 10, 30)), lambda a: a.mark_to_market(runner, D(2026, 10, 30))):
        with pytest.raises(LedgerError, match="call reset"):
            op(fresh)
    assert acct.ledger_path.read_text(encoding="utf-8") == "{not json"  # refused operations wrote nothing

    # regression: the recovery the error message suggests works from a freshly built account
    archived = PaperAccount(store, "corrupt", initial_capital=40_000).reset()
    assert archived is not None and archived.read_text(encoding="utf-8") == "{not json"
    assert fresh.started is None and fresh.ledger.initial_capital == 40_000
    assert acct.rebalance(runner, spec, START).trigger == "initial"

    data = json.loads(acct.ledger_path.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    acct.ledger_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LedgerError, match="schema_version"):
        PaperAccount(store, "corrupt").summary()
    assert PaperAccount(store, "corrupt").reset() is not None
    assert acct.started is None


@pytest.mark.parametrize(
    "raw",
    [b"\xff\xfe{\x00", b'{"name": "caf\xe9"}', b"\x00\x01\x02\x89PNG garbage", b""],
    ids=["utf16-garbage", "cp1252", "binary", "empty"],
)
def test_non_utf8_ledger_is_a_ledger_error_and_reset_recovers(store, runner, raw):
    """Regression: bytes that do not decode as JSON text raised a raw UnicodeDecodeError, which
    neither the callers (LedgerError) nor reset() handled."""
    spec = make_spec()
    acct = PaperAccount(store, "binary")
    acct.rebalance(runner, spec, START)
    acct.ledger_path.write_bytes(raw)
    with pytest.raises(LedgerError):
        PaperAccount(store, "binary").ledger
    with pytest.raises(StoreError):
        store_mod.read_json(acct.ledger_path)
    archived = acct.reset()  # an existing object ...
    assert archived is not None and archived.read_bytes() == raw
    acct.rebalance(runner, spec, START)
    acct.ledger_path.write_bytes(raw)
    assert PaperAccount(store, "binary").reset() is not None  # ... and a new one both recover
    assert acct.started is None and len(acct.archived_ledgers()) == 2


def test_store_reads_json_saved_by_windows_editors_and_skips_undecodable_meta(store):
    spec = make_spec()
    d = store.save("bom", spec, notes="café")
    meta = (d / "meta.json").read_text(encoding="utf-8")
    (d / "meta.json").write_bytes(b"\xef\xbb\xbf" + meta.encode("utf-8"))  # "UTF-8 with BOM"
    assert store.load_meta("bom")["notes"] == "café"
    (d / "meta.json").write_bytes(meta.encode("utf-16"))  # Notepad's "Unicode"
    assert store.load_meta("bom")["notes"] == "café"

    store.save("ok", spec)
    (d / "meta.json").write_bytes(meta.encode("cp1252"))  # not JSON text: skipped, not a crash
    with pytest.raises(StoreError):
        store.load_meta("bom")
    assert [r["slug"] for r in store.list()] == ["ok"]
    (store.path("ok") / "backtest.json").write_bytes(b"\xff\xfe\xfd")
    assert store.load_backtest("ok") is None  # unreadable backtest is ignored (logged)


def test_reset_archives_the_ledger(store, runner):
    clock = Clock(datetime(2026, 10, 5, 14, 30, 0, tzinfo=UTC))
    acct = PaperAccount(store, "reset me", clock=clock)
    spec = make_spec()
    assert acct.reset() is None  # no ledger yet: nothing to archive (a fresh one is written)
    assert acct.ledger_path.exists() and acct.started is None
    acct.rebalance(runner, spec, START)
    old = json.loads(acct.ledger_path.read_text(encoding="utf-8"))

    archived = acct.reset(initial_capital=25_000)

    assert archived is not None and archived.name == "ledger-20261005T143000Z.json"
    assert json.loads(archived.read_text(encoding="utf-8")) == old
    assert re.fullmatch(r"ledger-\d{8}T\d{6}Z(-\d+)?\.json", archived.name)
    led = acct.ledger
    assert led.started is None and led.positions == {} and led.trades == [] and led.nav_history == []
    assert led.cash == led.initial_capital == 25_000
    assert acct.summary()["status"] == "not_started"

    # same-second reset gets a distinct archive name; capital defaults to the current one
    acct.rebalance(runner, spec, START)
    archived2 = acct.reset()
    assert archived2.name == "ledger-20261005T143000Z-1.json"
    assert acct.ledger.initial_capital == 25_000
    assert acct.archived_ledgers() == [archived, archived2]  # oldest first
    # a reset account starts over from the next run
    assert acct.rebalance(runner, spec, START).trigger == "initial"


def test_archived_ledgers_are_oldest_first_across_same_second_suffixes(store):
    """Regression: a name sort put ``-1`` before the unsuffixed archive ('-' < '.') and ``-10``
    before ``-2``."""
    clock = Clock(datetime(2026, 10, 5, 14, 30, 0, tzinfo=UTC))
    acct = PaperAccount(store, "many resets", clock=clock)
    assert acct.reset(initial_capital=1.0) is None
    created = [acct.reset(initial_capital=float(i)) for i in range(2, 14)]  # 12 resets in one second
    clock.tick(1)
    created.append(acct.reset(initial_capital=99.0))
    assert [p.name for p in created[:3]] == [
        "ledger-20261005T143000Z.json", "ledger-20261005T143000Z-1.json", "ledger-20261005T143000Z-2.json",
    ]  # fmt: skip
    assert created[10].name == "ledger-20261005T143000Z-10.json"
    assert created[-1].name == "ledger-20261005T143001Z.json"

    ordered = acct.archived_ledgers()
    assert ordered == created
    # each reset archived the ledger the previous one wrote, so the capitals give the true order
    capitals = [json.loads(p.read_text(encoding="utf-8"))["initial_capital"] for p in ordered]
    assert capitals == [float(i) for i in range(1, 14)]
    # a hand-made copy does not disturb the order of the real archives
    (acct.path / "ledger-backup.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in acct.archived_ledgers()] == ["ledger-backup.json"] + [p.name for p in created]


def test_mark_to_market_before_start_is_cash(store, runner):
    acct = PaperAccount(store, "not started", initial_capital=10_000)
    assert acct.mark_to_market(runner, START) == 10_000
    assert not acct.ledger_path.exists()
    s = acct.summary()
    assert s["status"] == "not_started" and s["nav"] == 10_000 and s["started"] is None
    assert s["since_start_return_pct"] is None and s["backtest_expected_return_pct"] is None
    assert s["nav_history"] == [] and s["holdings"] == [] and s["trades"] == []


def test_summary_contents(store, runner):
    spec = make_spec()
    acct = PaperAccount(store, "Summary Strat")
    acct.rebalance(runner, spec, START)
    d = D(2026, 10, 30)
    runner.set_prices(d, AAA=110.0, BBB=50.0, CCC=20.0, DDD=10.0)
    runner.set_weights(d, AAA=0.6, DDD=0.4)
    acct.rebalance(runner, spec, d)

    s = acct.summary()
    expected_keys = {
        "name", "slug", "status", "started", "last_run", "last_rebalance", "next_rebalance", "rebalance",
        "initial_capital", "cash", "nav", "long_value", "short_value", "gross_exposure_pct",
        "net_exposure_pct", "since_start_return_pct", "backtest_expected_return_pct", "backtest_window",
        "max_drawdown_pct", "total_costs", "n_trades", "nav_history", "holdings", "trades", "runs",
        "warnings", "backtest_comparison_notes", "costs_bps", "allow_fractional", "ledger_path",
    }  # fmt: skip
    assert expected_keys <= set(s)
    assert s["name"] == "Summary Strat" and s["slug"] == "summary-strat" and s["status"] == "active"
    assert (s["started"], s["last_run"], s["last_rebalance"]) == ("2026-10-02", "2026-10-30", "2026-10-30")
    assert s["next_rebalance"] == "2026-11-30" and s["rebalance"] == "monthly"
    assert s["nav_history"] == [(x.isoformat(), v) for x, v in acct.ledger.nav_history]
    assert all(isinstance(x, str) and isinstance(v, float) for x, v in s["nav_history"])
    assert s["nav"] == pytest.approx(ledger_nav(acct))
    assert s["cash"] == pytest.approx(acct.cash)
    assert s["since_start_return_pct"] == pytest.approx((s["nav"] / 100_000 - 1) * 100)
    assert s["max_drawdown_pct"] <= 0
    # trades latest first
    dates = [t["date"] for t in s["trades"]]
    assert dates == sorted(dates, reverse=True) and dates[0] == "2026-10-30"
    assert s["n_trades"] == len(s["trades"]) == len(acct.ledger.trades)
    assert set(s["trades"][0]) == {"date", "ticker", "side", "shares", "price", "notional", "cost", "reason"}
    # holdings with weights that sum with cash to 100%
    h = {x["ticker"]: x for x in s["holdings"]}
    assert set(h) == {"AAA", "DDD"}
    assert h["AAA"]["weight"] == pytest.approx(0.6, abs=1e-6) and h["AAA"]["target_weight"] == 0.6
    assert sum(x["weight"] for x in s["holdings"]) + s["cash"] / s["nav"] == pytest.approx(1.0)
    assert [x["ticker"] for x in s["holdings"]] == ["AAA", "DDD"]  # largest first
    assert s["total_costs"] == pytest.approx(sum(t["cost"] for t in s["trades"]))
    assert s["costs_bps"] is None  # nothing configured and no saved spec
    assert s["backtest_expected_return_pct"] is None  # no saved backtest


def test_summary_compares_with_a_covering_backtest(store, runner):
    spec = make_spec()
    month_ends = [D(2026, 8, 31), D(2026, 9, 30), D(2026, 10, 30), D(2026, 11, 30)]
    bt = make_backtest(month_ends, [0.01, -0.02, 0.03, None], spec)
    store.save("covered", spec, bt)
    acct = PaperAccount(store, "covered")
    acct.rebalance(runner, spec, START)
    assert acct.summary()["backtest_expected_return_pct"] is None  # window (10-02, 10-02] is empty

    runner.set_prices(D(2026, 10, 30), AAA=101.0)
    acct.rebalance(runner, spec, D(2026, 10, 30))
    s = acct.summary()
    assert s["backtest_expected_return_pct"] == pytest.approx(3.0)
    assert s["backtest_window"] == {"start": "2026-10-30", "end": "2026-10-30", "n_periods": 1}
    assert s["costs_bps"] == 10.0  # from the saved spec

    acct.mark_to_market(runner, D(2026, 11, 30))
    assert acct.summary()["backtest_expected_return_pct"] == pytest.approx(3.0)  # None return counts as 0

    # a backtest that ended before paper trading started does not cover the window
    old_bt = make_backtest(month_ends[:2], [0.01, -0.02], spec)
    assert acct.summary(backtest=old_bt)["backtest_expected_return_pct"] is None
    # an explicitly passed backtest overrides the saved one
    other = make_backtest(month_ends, [0.0, 0.0, 0.10, 0.10], spec)
    assert acct.summary(backtest=other)["backtest_expected_return_pct"] == pytest.approx(21.0)


def test_summary_notes_that_the_backtest_credits_interest_on_cash(store, runner):
    """The paper ledger's cash earns 0% while the backtest of a long-only / timing strategy credits
    RF on idle cash (2x on short proceeds): the summary must say so next to the comparison."""
    from aitrading.backtest.models import DataUsage
    from aitrading.trading import backtest_credits_cash_interest

    spec = make_spec(portfolio={"style": "long_only"})
    month_ends = [D(2026, 9, 30), D(2026, 10, 30)]
    rf = DataUsage(dataset="risk_free", source="Kenneth French (daily 1-month T-bill)", coverage="x", point_in_time=True)
    bt = make_backtest(month_ends, [0.0, 0.01], spec).model_copy(update={"data_usage": [rf]})
    acct = PaperAccount(store, "cash note")
    acct.rebalance(runner, spec, START)
    assert acct.summary()["backtest_comparison_notes"] == []  # no backtest
    notes = acct.summary(backtest=bt)["backtest_comparison_notes"]
    assert len(notes) == 1 and "T-bill" in notes[0] and "0%" in notes[0]

    assert backtest_credits_cash_interest(bt)
    no_rf = bt.model_copy(update={"data_usage": [rf.model_copy(update={"source": "none"})]})
    assert not backtest_credits_cash_interest(no_rf)
    ls = make_spec(portfolio={"style": "long_short"})
    assert not backtest_credits_cash_interest(bt.model_copy(update={"spec": ls.model_dump(mode="json")}))
    assert not backtest_credits_cash_interest(bt.model_copy(update={"spec": {**bt.spec, "kind": "factor_model"}}))
    assert acct.summary(backtest=no_rf)["backtest_comparison_notes"] == []


def test_backtest_return_over_window_edge_cases():
    dates = [D(2026, 9, 30), D(2026, 10, 30), D(2026, 11, 30)]
    bt = make_backtest(dates, [0.01, 0.02, 0.03])
    res = backtest_return_over_window(bt, D(2026, 10, 2), D(2026, 11, 30))
    assert res["return_pct"] == pytest.approx((1.02 * 1.03 - 1) * 100) and res["n_periods"] == 2
    assert backtest_return_over_window(bt, D(2026, 9, 1), D(2026, 11, 30)) is None  # starts before coverage
    assert backtest_return_over_window(bt, D(2026, 10, 2), D(2026, 12, 15)) is None  # ends after coverage
    assert backtest_return_over_window(bt, D(2026, 10, 2), D(2026, 10, 20)) is None  # no period end inside
    assert backtest_return_over_window(bt, D(2026, 10, 2), D(2026, 10, 2)) is None
    assert backtest_return_over_window(bt, D(2026, 10, 2), D(2026, 11, 30), series="benchmark") is None


def test_spec_drift_warning(store, runner):
    spec = make_spec()
    store.save("drift", spec)
    acct = PaperAccount(store, "drift")
    assert not acct.rebalance(runner, spec, START).warnings
    changed = spec.model_copy(update={"costs_bps": 50.0})
    rep = acct.rebalance(runner, changed, D(2026, 10, 30))
    assert any("differs from the spec saved" in w for w in rep.warnings)


# ------------------------------------------------------------------------------------------------
# Strategy store
# ------------------------------------------------------------------------------------------------


def test_store_save_load_round_trip(store):
    spec = make_spec()
    bt = make_backtest([D(2026, 8, 31), D(2026, 9, 30)], [0.01, 0.02], spec)
    path = store.save("Momentum 12-1", spec, bt, notes="first try")

    assert path == store.path("Momentum 12-1") == store.root / "momentum-12-1"
    assert {p.name for p in path.iterdir()} == {"spec.json", "backtest.json", "meta.json"}
    spec2, bt2, meta = store.load("Momentum 12-1")
    assert spec2 == spec
    assert bt2 == bt
    assert meta["schema_version"] == 1
    assert meta["name"] == "Momentum 12-1" and meta["slug"] == "momentum-12-1"
    assert meta["idea"] == spec.idea and meta["notes"] == "first try" and meta["template"] is None
    assert meta["created"] == meta["updated"] == "2026-10-02T09:00:00+00:00"
    assert meta["kind"] == "cross_sectional" and meta["rebalance"] == "monthly" and meta["has_backtest"]
    assert meta["backtest"]["run_id"] == "bt-1"
    assert meta["backtest"]["total_return_pct"] == pytest.approx((1.01 * 1.02 - 1) * 100)
    # files are the plain model JSON (readable without the store)
    assert StrategySpec.model_validate_json((path / "spec.json").read_text(encoding="utf-8")) == spec
    assert BacktestResult.model_validate_json((path / "backtest.json").read_text(encoding="utf-8")) == bt
    # lookups by any spelling that maps to the same slug
    assert store.load(" MOMENTUM 12-1")[0] == spec
    assert "MOMENTUM 12-1" in store and "other" not in store


def test_store_without_backtest_and_template_detection(store):
    key = "momentum_12_1" if "momentum_12_1" in TEMPLATES else next(iter(TEMPLATES))
    spec = TEMPLATES[key].spec()
    store.save("from template", spec)
    spec2, bt, meta = store.load("from template")
    assert spec2 == spec and bt is None and meta["template"] == key and not meta["has_backtest"]
    assert meta["backtest"] is None


def test_store_overwrite_rules(store):
    clock = store._clock
    spec = make_spec()
    bt = make_backtest([D(2026, 8, 31), D(2026, 9, 30)], [0.01, 0.02], spec)
    store.save("strat", spec, bt, notes="keep me")
    with pytest.raises(StrategyExistsError) as exc:
        store.save("STRAT", spec)  # same slug, different case
    assert isinstance(exc.value, FileExistsError)

    clock.tick(3600)
    spec2 = make_spec(costs_bps=20.0)
    store.save("strat", spec2, overwrite=True)
    s, b, meta = store.load("strat")
    assert s == spec2
    assert b is None and not (store.path("strat") / "backtest.json").exists()  # stale backtest removed
    assert meta["created"] == "2026-10-02T09:00:00+00:00"
    assert meta["updated"] == "2026-10-02T10:00:00+00:00"
    assert meta["notes"] == "keep me"

    meta2 = store.set_notes("strat", "changed my mind")
    assert meta2["notes"] == "changed my mind" == store.load_meta("strat")["notes"]


def test_store_overwrite_records_the_new_specs_template(store):
    """Regression: an overwrite kept the old meta template even when the new spec came from a
    different library template (or none), contradicting spec.json."""
    a, b = list(TEMPLATES)[:2]
    store.save("prov", TEMPLATES[a].spec())
    assert store.load_meta("prov")["template"] == a
    store.save("prov", TEMPLATES[b].spec(), overwrite=True)
    assert store.load_meta("prov")["template"] == b
    store.save("prov", make_spec(name="hand made"), overwrite=True)  # no longer from any template
    assert store.load_meta("prov")["template"] is None
    assert [r["template"] for r in store.list()] == [None]

    # an explicitly given template is kept while the spec stays the same spec ...
    store.save("custom", make_spec(name="my_momo"), template=a)
    store.save("custom", make_spec(name="my_momo", costs_bps=30.0), overwrite=True)
    assert store.load_meta("custom")["template"] == a
    # ... and an explicit template always wins
    store.save("custom", TEMPLATES[b].spec(), overwrite=True, template="my-own")
    assert store.load_meta("custom")["template"] == "my-own"
    store.save("custom", make_spec(name="something else"), overwrite=True)
    assert store.load_meta("custom")["template"] is None


def test_store_overwrite_keeps_the_paper_ledger(store, runner):
    spec = make_spec()
    store.save("live", spec)
    PaperAccount(store, "live").rebalance(runner, spec, START)
    store.save("live", spec, make_backtest([D(2026, 9, 30), D(2026, 10, 30)], [0.0, 0.01], spec), overwrite=True)
    assert PaperAccount(store, "live").started == START


def test_store_list_and_delete(store, runner):
    spec = make_spec()
    store.save("Beta idea", spec)
    store.save("alpha idea", spec, make_backtest([D(2026, 8, 31), D(2026, 9, 30)], [0.01, 0.02], spec))
    PaperAccount(store, "Beta idea").rebalance(runner, spec, START)
    (store.root / "not-a-strategy").mkdir()
    (store.root / ".hidden").mkdir()

    rows = store.list()
    assert [r["slug"] for r in rows] == ["alpha-idea", "beta-idea"]
    a, b = rows
    assert a["name"] == "alpha idea" and a["has_backtest"] and a["backtest"]["run_id"] == "bt-1"
    assert not a["paper_trading"]
    assert b["name"] == "Beta idea" and not b["has_backtest"] and b["backtest"] is None and b["paper_trading"]
    assert set(a) >= {"name", "slug", "path", "idea", "template", "notes", "kind", "rebalance", "created", "updated"}

    store.delete("Beta idea")
    assert [r["slug"] for r in store.list()] == ["alpha-idea"]
    assert not store.path("Beta idea").exists()
    with pytest.raises(StrategyNotFoundError) as exc:
        store.load("Beta idea")
    assert isinstance(exc.value, KeyError) and "Beta idea" in str(exc.value)
    with pytest.raises(StrategyNotFoundError):
        store.delete("Beta idea")
    assert StrategyStore(store.root / "nowhere").list() == []


def test_store_refuses_newer_schema(store):
    spec = make_spec()
    d = store.save("future", spec)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["schema_version"] = 2
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StoreError, match="schema_version"):
        store.load("future")
    assert store.list() == []  # skipped (logged), not misread


def test_store_handles_missing_meta(store):
    spec = make_spec()
    d = store.save("handmade", spec)
    (d / "meta.json").unlink()
    s, bt, meta = store.load("handmade")
    assert s == spec and bt is None and meta["name"] == "handmade" and meta["schema_version"] == 1
    assert [r["slug"] for r in store.list()] == ["handmade"]


def test_store_writes_are_atomic(store, monkeypatch):
    spec = make_spec()
    d = store.save("atomic", spec)
    before = (d / "spec.json").read_bytes()

    def boom(src, dst):
        raise OSError("crash during write")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.save("atomic", make_spec(costs_bps=99.0), overwrite=True)
    monkeypatch.undo()
    assert (d / "spec.json").read_bytes() == before
    assert not [p for p in d.iterdir() if p.name.endswith(".tmp")]
    assert store.load("atomic")[0] == spec

    with pytest.raises(ValueError):  # NaN is not valid JSON: refused, nothing left behind
        store_mod.write_json_atomic(d / "x.json", {"x": float("nan")})
    assert not (d / "x.json").exists() and not [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_store_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AITRADING_HOME", str(tmp_path / "home"))
    assert default_store_root() == tmp_path / "home" / "strategies"
    s = StrategyStore()
    assert s.root == tmp_path / "home" / "strategies"
    s.save("env strat", make_spec())
    assert (tmp_path / "home" / "strategies" / "env-strat" / "spec.json").is_file()

    monkeypatch.delenv("AITRADING_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
    assert default_store_root() == tmp_path / "user" / ".aitrading" / "strategies"


WINDOWS_UNSAFE = [
    "CON",
    "nul",
    "Com1",
    "LPT9",
    'a<b>c:d"e/f\\g|h?i*j',
    "  trailing dots and spaces. . ",
    "...",
    "../../escape",
    "C:\\Windows\\System32",
    "tab\tand\nnewline",
    "Qualité à prix raisonnable",
    "动量策略",
    "价值策略",
    "Ünïcödé + 动量",
    "x" * 300,
    "emoji 🚀 rocket",
]


@pytest.mark.parametrize("name", WINDOWS_UNSAFE)
def test_windows_unsafe_names_are_slugified(store, name):
    slug = slugify_strategy_name(name)
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug), slug
    assert len(slug) <= 64
    assert slug.split(".")[0] not in {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(10)} | {f"lpt{i}" for i in range(10)}
    assert not slug.endswith((".", " "))
    assert slugify_strategy_name(slug) == slug  # idempotent
    d = store.save(name, make_spec())
    assert d.parent == store.root and d.name == slug  # never escapes the store root
    spec, _, meta = store.load(name)
    assert meta["name"] == name.strip()
    acct = PaperAccount(store, name)
    assert acct.ledger_path.parent == d


def test_slug_examples_and_collisions():
    assert slugify_strategy_name("Momentum 12-1") == "momentum-12-1"
    assert slugify_strategy_name("CON") == "con_"
    assert slugify_strategy_name("Qualité") == "qualite"
    assert slugify_strategy_name("MyStrat") == slugify_strategy_name("mystrat")
    assert slugify_strategy_name("动量策略") != slugify_strategy_name("价值策略")
    assert slugify_strategy_name("x" * 300) != slugify_strategy_name("x" * 299)
    assert slugify_strategy_name("Ünïcödé + 动量") != slugify_strategy_name("Ünïcödé + 价值")
    for bad in ["", "   "]:
        with pytest.raises(ValueError):
            slugify_strategy_name(bad)
    with pytest.raises(TypeError):
        slugify_strategy_name(None)  # type: ignore[arg-type]


def test_no_broker_integration():
    """Paper trading is simulated only: the package must not reference any broker API."""
    import aitrading.trading.paper as paper_mod

    src = Path(paper_mod.__file__).read_text(encoding="utf-8").lower() + Path(store_mod.__file__).read_text(encoding="utf-8").lower()
    for broker in ("ib_insync", "ibapi", "alpaca", "tradier", "oanda", "requests.post", "httpx"):
        assert broker not in src
