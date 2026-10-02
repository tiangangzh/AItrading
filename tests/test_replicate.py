"""Tests for aitrading.discovery.replicate (replication / robustness suite), fully offline.

A FakeRunner implements the BacktestRunner protocol and returns hand-built BacktestResults whose
Sharpe ratio depends on the spec it is given (window, costs, quantiles, rebalance), so every check,
pass rule and verdict branch can be exercised deterministically.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.models import BacktestResult, DataUsage, FactorRegression, PerformanceStats
from aitrading.backtest.protocols import BacktestRunner
from aitrading.discovery.models import IdeaCandidate, IdeaExtraction, ReplicationReport, RobustnessCheck, SourceDocument
from aitrading.discovery.replicate import (
    MIN_BASE_YEARS,
    PARTIAL_MIN_PASS_SHARE,
    REPLICATES_MIN_PASS_SHARE,
    cost_passed,
    perturbation_passed,
    post_publication_passed,
    replicate,
    replication_verdict,
    subperiod_passed,
)
from aitrading.screen.spec import Condition
from aitrading.strategy.spec import PortfolioConstruction, SignalComponent, StrategySpec, TimeSeriesRule

NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2025, 12, 31)


# ------------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------------


def make_stats(start: date, end: date, sharpe: float | None, *, n: int | None = None, ppy: float = 12.0) -> PerformanceStats:
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates) if n is None else n
    first = dates[0].date() if len(dates) else start
    last = dates[-1].date() if len(dates) else end
    return PerformanceStats(
        label="strategy",
        start=first,
        end=last,
        n_periods=n,
        periods_per_year=ppy,
        total_return_pct=50.0,
        cagr_pct=None if sharpe is None else round(8.0 * sharpe, 6),
        volatility_pct=12.0,
        sharpe=sharpe,
        sortino=None,
        max_drawdown_pct=-15.0,
        max_drawdown_duration_periods=6,
        calmar=None,
        hit_rate_pct=55.0,
        best_period_pct=6.0,
        worst_period_pct=-7.0,
        skew=None,
        excess_kurtosis=None,
        mean_return_t_stat=None,
    )


def make_result(
    spec: StrategySpec,
    start: date,
    end: date,
    sharpe: float | None,
    *,
    alpha_t: float | None = None,
    label: str = "base",
    warnings: list[str] | None = None,
    data_usage: list[DataUsage] | None = None,
    with_stats: bool = True,
) -> BacktestResult:
    dates = [d.date() for d in pd.date_range(start, end, freq="ME")]
    stats = {"strategy": make_stats(start, end, sharpe)} if with_stats else {}
    if with_stats and sharpe is not None and not math.isfinite(sharpe):
        # PerformanceStats accepts NaN, which the suite must treat as "no Sharpe"
        stats["strategy"] = stats["strategy"].model_copy(update={"sharpe": sharpe})
    reg = None
    if alpha_t is not None:
        reg = FactorRegression(model="ff3", factor_source="test factors", n=len(dates), alpha_annual_pct=3.0,
                               alpha_t_stat=alpha_t, betas={"mkt_rf": 0.1}, beta_t_stats={"mkt_rf": 1.0}, r_squared=0.1)
    return BacktestResult(
        run_id=f"run-{label}",
        idea=spec.idea,
        spec=spec.model_dump(mode="json"),
        provider="fake",
        llm="offline",
        start=start,
        end=end,
        rebalance=spec.rebalance,
        returns={"strategy": [0.01] * len(dates)},
        dates=dates,
        stats=stats,
        regression=reg,
        warnings=list(warnings or []),
        data_usage=list(data_usage or []),
        started_at=NOW,
    )


def default_sharpe(spec: StrategySpec, start: date, end: date, label: str | None) -> float:
    """Base (10 bps, quintiles, monthly, full window) = 1.0; costs, quantiles, rebalance and window shift it."""
    s = 1.2 - 0.02 * spec.costs_bps  # 0 bps -> 1.2, 10 bps -> 1.0, 25 bps -> 0.7
    if spec.portfolio.n_quantiles == 10:
        s += 0.1
    if spec.rebalance == "quarterly":
        s -= 0.1
    if end <= date(2021, 1, 1):
        s += 0.1  # first half
    elif start >= date(2020, 12, 1):
        s -= 0.1  # second half
    return s


class FakeRunner:
    """BacktestRunner returning crafted results; records every (label, spec) it was asked to run."""

    provider_name = "fake"

    def __init__(
        self,
        sharpe_fn=default_sharpe,
        *,
        alpha_fn=lambda sharpe, spec, label: None if sharpe is None else 2.5 * sharpe,
        fail: dict[str, Exception] | None = None,
        warnings_fn=lambda label: [],
        data_usage_fn=lambda label: [],
        no_stats: set[str] | None = None,
        window: tuple[date, date] = (DEFAULT_START, DEFAULT_END),
    ):
        self.sharpe_fn = sharpe_fn
        self.alpha_fn = alpha_fn
        self.fail = dict(fail or {})
        self.warnings_fn = warnings_fn
        self.data_usage_fn = data_usage_fn
        self.no_stats = set(no_stats or ())
        self.window = window
        self.calls: list[tuple[str | None, StrategySpec]] = []

    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        self.calls.append((label, spec))
        if label in self.fail:
            raise self.fail[label]
        if "*" in self.fail and label != "base":
            raise self.fail["*"]
        start = spec.start or self.window[0]
        end = spec.end or self.window[1]
        sharpe = self.sharpe_fn(spec, start, end, label)
        return make_result(
            spec, start, end, sharpe,
            alpha_t=self.alpha_fn(sharpe, spec, label),
            label=label or "run",
            warnings=self.warnings_fn(label),
            data_usage=self.data_usage_fn(label),
            with_stats=label not in self.no_stats,
        )

    def target_portfolio(self, spec: StrategySpec, as_of: date) -> pd.Series:
        return pd.Series(dtype=float)

    def latest_prices(self, tickers: list[str], as_of: date) -> pd.Series:
        return pd.Series(np.nan, index=list(tickers), dtype=float)

    def labels(self) -> list[str | None]:
        return [lbl for lbl, _ in self.calls]

    def spec_for(self, label: str) -> StrategySpec:
        return next(s for lbl, s in self.calls if lbl == label)


class StatsWindowRunner(FakeRunner):
    """FakeRunner whose strategy stats can cover a different window (or sample length) than the
    result's [start, end]: ``stats_window(label, start, end)`` returns ``(start, end, n_periods or
    None)`` to override, or ``None`` to keep the FakeRunner's stats."""

    def __init__(self, stats_window, sharpe_fn=default_sharpe, **kw):
        super().__init__(sharpe_fn, **kw)
        self.stats_window = stats_window

    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        res = super().backtest(spec, label=label)
        override = self.stats_window(label, res.start, res.end)
        if override is None:
            return res
        s, e, n = override
        st = make_stats(s, e, res.stats["strategy"].sharpe, n=n)
        return res.model_copy(update={"stats": {"strategy": st}})


def cs_spec(**kw) -> StrategySpec:
    portfolio = kw.pop("portfolio", PortfolioConstruction(style="long_short", selection="quantile", n_quantiles=5))
    return StrategySpec(
        name="mom_12_1",
        idea="Long-short quintiles on 12-1 momentum",
        kind="cross_sectional",
        signal=[SignalComponent(feature="momentum_12_1", direction="higher_is_better")],
        portfolio=portfolio,
        **kw,
    )


def screen_spec(**kw) -> StrategySpec:
    return StrategySpec(
        name="cheap_quality",
        idea="Cheap, profitable stocks",
        kind="screen",
        filters=[Condition(feature="pe_ratio", op="<", value=15.0)],
        **kw,
    )


def ts_spec(entry: list[Condition] | None = None, **kw) -> StrategySpec:
    entry = entry or [Condition(feature="price_vs_sma_200_pct", op=">", value=0.0)]
    return StrategySpec(
        name="trend_200dma_spy",
        idea="Long SPY above its 200-day moving average",
        kind="time_series",
        time_series=TimeSeriesRule(assets=["SPY"], entry=entry),
        attribution_model="capm",
        benchmark="SPY",
        **kw,
    )


def fm_spec() -> StrategySpec:
    return StrategySpec(name="ff3", idea="Fama-French 3-factor model", kind="factor_model", factor_model="ff3")


def make_candidate(
    *,
    published: date | None = date(2019, 6, 15),
    reported_sharpe: float | None = None,
    reported_t: float | None = None,
    reported_ret: float | None = None,
    sample_period: str | None = None,
) -> IdeaCandidate:
    src = SourceDocument(
        source_type="arxiv",
        url="https://arxiv.org/abs/1901.00001",
        title="Momentum, again",
        authors=["A. Author"],
        published=published,
        text="We document that 12-1 momentum earns a Sharpe ratio of 1.2.",
        fetched_at=NOW,
        source_name="arXiv q-fin.PM",
    )
    ext = IdeaExtraction(
        is_trading_idea=True,
        title="12-1 momentum",
        summary="Stocks that rose over the past year keep rising.",
        claimed_effect="Winners beat losers by about 1% a month.",
        signal_description="Return from t-12 to t-1.",
        asset_class="us_equities",
        holding_period="1 month",
        reported_sharpe=reported_sharpe,
        reported_annual_return_pct=reported_ret,
        reported_t_stat=reported_t,
        sample_period=sample_period,
        evidence_quotes=["We document that 12-1 momentum earns a Sharpe ratio of 1.2."],
        data_requirements=["prices"],
        testability="testable_now",
        missing_data=[],
        proposed_strategy_idea="Long-short quintiles on 12-1 momentum, monthly rebalance, US stocks",
        credibility_notes=[],
    )
    return IdeaCandidate(idea_id="idea-mom", source=src, extraction=ext, discovered_at=NOW)


def check(report: ReplicationReport, name: str) -> RobustnessCheck:
    return next(c for c in report.checks if c.name == name)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# ------------------------------------------------------------------------------------------------
# Protocol and basic flow
# ------------------------------------------------------------------------------------------------


def test_fake_runner_implements_protocol():
    assert isinstance(FakeRunner(), BacktestRunner)


def test_base_run_and_check_order_for_cross_sectional_spec():
    runner = FakeRunner()
    spec = cs_spec()
    base, report = replicate(None, spec, runner)

    assert runner.calls[0][0] == "base"
    assert runner.calls[0][1] is spec  # the base runs the spec itself
    assert runner.labels() == ["base", "first_half", "second_half", "costs_0bps", "costs_25bps",
                               "deciles_instead_of_quintiles", "quarterly_rebalance"]
    assert [c.name for c in report.checks] == runner.labels()[1:]
    assert base.run_id == "run-base"
    assert report.backtest_run_id == "run-base"
    assert report.idea_id == "mom_12_1"  # no candidate -> the spec name
    assert report.base_sharpe == pytest.approx(1.0)
    assert report.base_alpha_t_stat == pytest.approx(2.5)
    assert report.claimed_sharpe is None and report.claimed_t_stat is None and report.replication_ratio is None
    assert all(c.passed for c in report.checks)
    assert report.verdict == "replicates"


def test_checks_carry_stats_from_their_own_runs():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner)
    fh = check(report, "first_half")
    assert fh.sharpe == pytest.approx(1.1)
    assert fh.cagr_pct == pytest.approx(8.8)
    assert fh.alpha_t_stat == pytest.approx(2.75)
    assert fh.n_periods == 60
    assert "2016-01-31 to 2020-12-31" in fh.note
    assert "60 periods" in fh.note
    assert check(report, "costs_0bps").sharpe == pytest.approx(1.2)
    assert check(report, "costs_25bps").sharpe == pytest.approx(0.7)


def test_spec_is_never_mutated():
    spec = cs_spec()
    before = spec.model_dump()
    replicate(make_candidate(), spec, FakeRunner())
    assert spec.model_dump() == before


def test_report_round_trips_and_attaches_to_candidate():
    cand = make_candidate(reported_sharpe=1.2, reported_t=3.1)
    _, report = replicate(cand, cs_spec(), FakeRunner())
    again = ReplicationReport.model_validate_json(report.model_dump_json())
    assert again == report
    updated = IdeaCandidate.model_validate(cand.model_copy(update={"replication": report}).model_dump(mode="json"))
    assert updated.replication.verdict == report.verdict
    assert report.idea_id == "idea-mom"


# ------------------------------------------------------------------------------------------------
# Sub-period splits
# ------------------------------------------------------------------------------------------------


def test_split_windows_use_the_base_window_midpoint():
    runner = FakeRunner()
    replicate(None, cs_spec(), runner)  # spec has no dates: the runner chooses 2016-01-01..2025-12-31
    fh, sh = runner.spec_for("first_half"), runner.spec_for("second_half")
    assert (fh.start, fh.end) == (date(2016, 1, 1), date(2020, 12, 31))
    assert (sh.start, sh.end) == (date(2021, 1, 1), date(2025, 12, 31))
    # everything else about the spec is unchanged
    assert fh.model_dump(exclude={"start", "end"}) == cs_spec().model_dump(exclude={"start", "end"})


def test_split_follows_explicit_spec_window():
    runner = FakeRunner()
    replicate(None, cs_spec(start=date(2018, 3, 1), end=date(2025, 2, 28)), runner)
    fh, sh = runner.spec_for("first_half"), runner.spec_for("second_half")
    mid = date(2018, 3, 1) + (date(2025, 2, 28) - date(2018, 3, 1)) // 2
    assert fh.start == date(2018, 3, 1) and fh.end == mid
    assert sh.start == date.fromordinal(mid.toordinal() + 1) and sh.end == date(2025, 2, 28)


def test_six_year_window_splits_into_two_three_year_halves():
    runner = FakeRunner(window=(date(2020, 1, 1), date(2025, 12, 31)))
    _, report = replicate(None, cs_spec(), runner, min_years_per_split=3.0)
    assert "first_half" in runner.labels() and "second_half" in runner.labels()
    assert runner.spec_for("first_half").end == date(2022, 12, 31)
    assert check(report, "first_half").passed is not None


def test_split_skipped_when_halves_are_too_short():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner, min_years_per_split=6.0)
    assert "first_half" not in runner.labels() and "second_half" not in runner.labels()
    for name in ("first_half", "second_half"):
        c = check(report, name)
        assert c.passed is None
        assert c.sharpe is None and c.n_periods == 0
        assert "5.0 years" in c.note and "6" in c.note
    # skipped checks do not count against the verdict
    assert report.verdict == "replicates"


def test_subperiod_fails_when_one_half_is_negative():
    def sharpe(spec, start, end, label):
        return -0.3 if label == "second_half" else default_sharpe(spec, start, end, label)

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe))
    assert check(report, "first_half").passed is True
    sh = check(report, "second_half")
    assert sh.passed is False
    assert "not positive" in sh.note
    assert "second_half" in report.summary


def test_split_uses_the_realised_return_window_not_the_requested_one():
    # regression: the result says 2010-2025 but strategy returns only exist from 2022 (4 years). The
    # old split (2010-2017 / 2018-2025) ran an empty first half and a "second half" equal to the base.
    runner = StatsWindowRunner(lambda label, s, e: (max(s, date(2022, 1, 1)), e, None),
                               window=(date(2010, 1, 1), DEFAULT_END))
    _, report = replicate(None, cs_spec(), runner)
    assert "first_half" not in runner.labels() and "second_half" not in runner.labels()
    for name in ("first_half", "second_half"):
        c = check(report, name)
        assert c.passed is None
        assert "4.0-year return window" in c.note and "2.0 years" in c.note


def test_split_midpoint_follows_late_starting_returns():
    runner = StatsWindowRunner(lambda label, s, e: (max(s, date(2016, 1, 1)), e, None),
                               window=(date(2010, 1, 1), DEFAULT_END))
    _, report = replicate(None, cs_spec(), runner)
    fh, sh = runner.spec_for("first_half"), runner.spec_for("second_half")
    # the returns cover 2016-2025: split that (one month before the first return date), not 2010-2025
    assert (fh.start, fh.end) == (date(2015, 12, 31), date(2020, 12, 30))
    assert (sh.start, sh.end) == (date(2020, 12, 31), DEFAULT_END)
    assert check(report, "first_half").passed is True and check(report, "second_half").passed is True


def test_sparse_sample_does_not_pass_the_split_length_test():
    # 10 calendar years but only 48 monthly returns (4 years): the shorter of the two decides
    runner = StatsWindowRunner(lambda label, s, e: (s, e, 48) if label == "base" else None)
    _, report = replicate(None, cs_spec(), runner)
    assert "first_half" not in runner.labels()
    assert "2.0 years" in check(report, "first_half").note


def test_split_run_outside_its_window_is_not_evaluated():
    # a runner that ignores the requested half and returns the full base window does not test that half
    runner = StatsWindowRunner(lambda label, s, e: (DEFAULT_START, DEFAULT_END, None) if label == "first_half" else None)
    _, report = replicate(None, cs_spec(), runner)
    c = check(report, "first_half")
    assert c.passed is None and c.sharpe == pytest.approx(1.1)
    assert "not evaluated" in c.note and "not inside the requested window (2016-01-01 to 2020-12-31)" in c.note
    assert check(report, "second_half").passed is True
    assert "1 could not be run" in report.summary


def test_split_run_with_too_few_years_is_not_evaluated():
    runner = StatsWindowRunner(lambda label, s, e: (date(2019, 7, 1), e, None) if label == "first_half" else None)
    _, report = replicate(None, cs_spec(), runner)
    c = check(report, "first_half")
    assert c.passed is None
    assert c.n_periods == 18
    assert "only 1.5 years of returns in the requested window (3 required)" in c.note


def test_split_run_tolerates_one_formation_period():
    # a 3-year half whose first month is spent forming the book (35 monthly returns) still counts
    runner = StatsWindowRunner(lambda label, s, e: (date(2020, 2, 1), e, None) if label == "first_half" else None,
                               window=(date(2020, 1, 1), DEFAULT_END))
    _, report = replicate(None, cs_spec(), runner, min_years_per_split=3.0)
    c = check(report, "first_half")
    assert c.n_periods == 35
    assert c.passed is True


# ------------------------------------------------------------------------------------------------
# Post-publication
# ------------------------------------------------------------------------------------------------


def test_post_publication_runs_from_the_publication_date():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=date(2019, 6, 15)), cs_spec(), runner)
    # the in-sample reference run comes right before the post-publication run
    assert runner.labels()[3:5] == ["pre_publication", "post_publication"]
    pre = runner.spec_for("pre_publication")
    assert (pre.start, pre.end) == (date(2016, 1, 1), date(2019, 6, 14))
    pp = runner.spec_for("post_publication")
    assert (pp.start, pp.end) == (date(2019, 6, 15), date(2025, 12, 31))
    c = check(report, "post_publication")
    assert c.passed is True
    assert "McLean" in c.description and "pre-publication" in c.description
    # post 1.00 vs pre-publication 1.10 (the default first-half bonus)
    assert "91% of the pre-publication Sharpe 1.10" in c.note
    assert "pre-publication reference 2016-01-31 to 2019-05-31, 41 periods" in c.note
    assert "After publication the Sharpe ratio was 1.00 (91% of the pre-publication figure of 1.10)" in report.summary
    # the reference run is not a check of its own
    assert "pre_publication" not in [c.name for c in report.checks]


def test_post_publication_decay_fails_below_half_of_pre_publication():
    def sharpe(spec, start, end, label):
        return {"post_publication": 0.4, "pre_publication": 1.0}.get(label) or default_sharpe(spec, start, end, label)

    _, report = replicate(make_candidate(), cs_spec(), FakeRunner(sharpe))
    c = check(report, "post_publication")
    assert c.passed is False
    assert "40% of the pre-publication Sharpe 1.00" in c.note and "needs >= 50% of the pre-publication Sharpe" in c.note


def test_post_publication_exactly_half_passes():
    def sharpe(spec, start, end, label):
        return {"post_publication": 0.5, "pre_publication": 1.0}.get(label) or default_sharpe(spec, start, end, label)

    _, report = replicate(make_candidate(), cs_spec(), FakeRunner(sharpe))
    assert check(report, "post_publication").passed is True


def test_post_publication_decay_is_measured_against_the_pre_publication_sharpe():
    # regression (McLean & Pontiff compare with the in-sample period): 2000-2025, published 2016,
    # pre-publication Sharpe 1.0, post 0.4 (a 60% decay), blended full sample about 0.76. Against the
    # full sample the 50% rule would pass (0.4 >= 0.38); against the pre-publication Sharpe it fails.
    def sharpe(spec, start, end, label):
        return {"base": 0.76, "pre_publication": 1.0, "post_publication": 0.4}.get(label, 0.9)

    runner = FakeRunner(sharpe, window=(date(2000, 1, 1), DEFAULT_END))
    _, report = replicate(make_candidate(published=date(2016, 1, 1)), cs_spec(), runner)
    assert post_publication_passed(0.4, 0.76) is True  # what the old full-sample comparison concluded
    pre = runner.spec_for("pre_publication")
    assert (pre.start, pre.end) == (date(2000, 1, 1), date(2015, 12, 31))
    c = check(report, "post_publication")
    assert c.passed is False
    assert "40% of the pre-publication Sharpe 1.00" in c.note
    assert "After publication the Sharpe ratio was 0.40 (40% of the pre-publication figure of 1.00)" in report.summary


def test_post_publication_falls_back_to_full_sample_when_pre_window_is_short():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=date(2018, 1, 1)), cs_spec(), runner)
    assert "pre_publication" not in runner.labels()
    c = check(report, "post_publication")
    assert c.passed is True
    assert "of the full-sample Sharpe 1.00" in c.note
    assert "compared with the full-sample Sharpe instead of the pre-publication one" in c.note
    assert "covers only 2.0 years (< 3 required)" in c.note
    assert "of the full-sample figure of 1.00" in report.summary


def test_post_publication_falls_back_when_the_pre_publication_run_fails():
    runner = FakeRunner(fail={"pre_publication": RuntimeError("no data before 2019")})
    msgs: list[str] = []
    _, report = replicate(make_candidate(), cs_spec(), runner, progress=msgs.append)
    c = check(report, "post_publication")
    assert c.passed is True  # 1.00 vs the full-sample 1.00
    assert "full-sample" in c.note and "pre-publication run failed (RuntimeError: no data before 2019)" in c.note
    assert any("pre_publication failed" in m for m in msgs)
    assert "pre_publication" not in [x.name for x in report.checks]


def test_post_publication_falls_back_when_the_pre_publication_run_misses_its_window():
    runner = StatsWindowRunner(lambda label, s, e: (DEFAULT_START, DEFAULT_END, None) if label == "pre_publication" else None)
    _, report = replicate(make_candidate(), cs_spec(), runner)
    c = check(report, "post_publication")
    assert "full-sample" in c.note and "the pre-publication run's returns" in c.note and "not inside" in c.note


def test_post_publication_after_a_negative_pre_publication_sharpe_is_no_decay():
    def sharpe(spec, start, end, label):
        return {"pre_publication": -0.2, "post_publication": 0.5}.get(label) or default_sharpe(spec, start, end, label)

    _, report = replicate(make_candidate(), cs_spec(), FakeRunner(sharpe))
    c = check(report, "post_publication")
    assert c.passed is True
    assert "pre-publication Sharpe -0.20 is not positive, so there is no decay to measure" in c.note
    assert "(the pre-publication Sharpe was -0.20, so there is no decay to measure)" in report.summary


def test_pre_publication_run_warnings_are_collected():
    def warnings(label):
        return ["XYZ: no price from 2017-03-31 on while held (delisted)"] if label == "pre_publication" else []

    _, report = replicate(make_candidate(), cs_spec(), FakeRunner(warnings_fn=warnings))
    assert any("XYZ: no price" in c and "[check run: pre_publication]" in c for c in report.caveats)


def test_post_publication_skipped_when_too_recent():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=date(2023, 6, 1)), cs_spec(), runner)
    assert "post_publication" not in runner.labels()
    c = check(report, "post_publication")
    assert c.passed is None
    assert "2.6 years" in c.note
    assert any("Post-publication decay check" in cv for cv in report.caveats)


def test_post_publication_after_backtest_end_is_skipped():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=date(2026, 3, 1)), cs_spec(), runner)
    assert "post_publication" not in runner.labels()
    assert check(report, "post_publication").passed is None


def test_post_publication_before_backtest_start_is_not_rerun():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=date(2010, 1, 1)), cs_spec(), runner)
    assert "post_publication" not in runner.labels()
    c = check(report, "post_publication")
    assert c.passed is None
    assert "already after publication" in c.note


def test_post_publication_without_date_is_reported_as_not_run():
    runner = FakeRunner()
    _, report = replicate(make_candidate(published=None), cs_spec(), runner)
    c = check(report, "post_publication")
    assert c.passed is None and "no publication date" in c.note
    assert "post_publication" not in runner.labels()


def test_no_post_publication_check_without_candidate():
    _, report = replicate(None, cs_spec(), FakeRunner())
    assert all(c.name != "post_publication" for c in report.checks)


# ------------------------------------------------------------------------------------------------
# Pass rules
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sharpe", "base", "expected"),
    [(0.5, 1.0, True), (2.0, 1.0, True), (0.0, 1.0, False), (-0.5, 1.0, False), (0.5, -1.0, False),
     (-0.5, -1.0, False), (None, 1.0, None), (0.5, None, None)],
)
def test_subperiod_rule(sharpe, base, expected):
    assert subperiod_passed(sharpe, base) is expected


@pytest.mark.parametrize(
    ("sharpe", "base", "expected"),
    [(0.5, 1.0, True), (0.49, 1.0, False), (1.5, 1.0, True), (0.0, 1.0, False), (-0.1, 1.0, False),
     (0.3, -0.2, True), (None, 1.0, None), (0.5, None, None)],
)
def test_post_publication_rule(sharpe, base, expected):
    assert post_publication_passed(sharpe, base) is expected


@pytest.mark.parametrize(("sharpe", "expected"), [(0.01, True), (0.0, False), (-1.0, False), (None, None)])
def test_cost_rule(sharpe, expected):
    assert cost_passed(sharpe) is expected
    assert cost_passed(sharpe, -5.0) is expected  # the base Sharpe is irrelevant


@pytest.mark.parametrize(
    ("sharpe", "base", "expected"),
    [(0.5, 1.0, True), (0.49, 1.0, False), (2.0, 1.0, True), (0.0, 1.0, False), (-0.6, 1.0, False),
     # the effect under test is a positive premium: keeping a negative (or zero) base's sign is not robustness
     (-0.6, -1.0, False), (-0.4, -1.0, False), (-2.0, -1.0, False), (0.6, -1.0, False), (0.5, 0.0, False),
     (-0.5, 0.0, False), (None, 1.0, None), (0.5, None, None)],
)
def test_perturbation_rule(sharpe, base, expected):
    assert perturbation_passed(sharpe, base) is expected


def test_nan_sharpe_in_a_check_cannot_be_evaluated():
    def sharpe(spec, start, end, label):
        return float("nan") if label == "costs_0bps" else default_sharpe(spec, start, end, label)

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe))
    c = check(report, "costs_0bps")
    assert c.sharpe is None and c.passed is None
    assert "cannot be evaluated" in c.note
    ReplicationReport.model_validate_json(report.model_dump_json())  # still serialisable (no NaN)


def test_check_without_strategy_stats_cannot_be_evaluated():
    runner = FakeRunner(no_stats={"quarterly_rebalance"})
    _, report = replicate(None, cs_spec(), runner)
    c = check(report, "quarterly_rebalance")
    assert c.passed is None and c.sharpe is None
    assert c.n_periods == 120  # falls back to the number of strategy returns


# ------------------------------------------------------------------------------------------------
# Cost checks
# ------------------------------------------------------------------------------------------------


def test_cost_checks_pass_their_cost_level_to_the_runner():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner, cost_levels_bps=(0.0, 25.0, 12.5))
    assert runner.spec_for("costs_0bps").costs_bps == 0.0
    assert runner.spec_for("costs_25bps").costs_bps == 25.0
    assert runner.spec_for("costs_12.5bps").costs_bps == 12.5
    assert runner.spec_for("costs_25bps").start is None  # same window as the base: only costs change
    assert "25 bps" in check(report, "costs_25bps").description and "base: 10 bps" in check(report, "costs_25bps").description


def test_cost_check_fails_when_sharpe_turns_negative():
    def sharpe(spec, start, end, label):
        return 1.4 - 0.04 * spec.costs_bps  # 10 bps -> 1.0, 25 bps -> 0.4, 40 bps -> -0.2

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe), cost_levels_bps=(0.0, 25.0, 40.0))
    assert check(report, "costs_25bps").passed is True
    c40 = check(report, "costs_40bps")
    assert c40.passed is False and "not positive" in c40.note
    assert any("does not survive realistic trading costs (costs_40bps)" in cv for cv in report.caveats)


def test_cost_level_equal_to_base_reuses_the_base_run():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(costs_bps=25.0), runner)
    assert "costs_25bps" not in runner.labels()
    c = check(report, "costs_25bps")
    assert c.passed is True
    assert c.sharpe == pytest.approx(report.base_sharpe)
    assert "reused" in c.note


def test_costs_at_or_below_the_base_are_informational_and_not_counted():
    # regression: factor model at 25 bps over 5 years (halves skipped). The only cost checks are the
    # gross run (easier than the base) and the reused base itself, so nothing harder than the base ran
    # and the verdict must not be "replicates".
    def sharpe(spec, start, end, label):
        return 0.8 if spec.costs_bps == 0 else 0.6

    runner = FakeRunner(sharpe, alpha_fn=lambda s, spec, label: 2.1, window=(date(2021, 1, 1), DEFAULT_END))
    spec = fm_spec().model_copy(update={"costs_bps": 25.0})
    msgs: list[str] = []
    _, report = replicate(None, spec, runner, progress=msgs.append)
    assert runner.labels() == ["base", "costs_0bps"]
    for name in ("costs_0bps", "costs_25bps"):
        c = check(report, name)
        assert c.passed is True  # still shown as passing ...
        assert c.note.startswith("informational, not counted in the verdict")  # ... but not counted
        assert "informational only" in c.description
    assert report.verdict == "inconclusive"
    assert "no robustness check harder than the base run could be run" in report.summary
    assert "costs_0bps, costs_25bps are informational" in report.summary
    assert msgs[-1] == "Replication verdict: inconclusive (0/0 checks passed)"
    # they still feed the gross vs net caveat
    assert any(c.startswith("Gross vs net") and "0bps: 0.80, 25bps: 0.60" in c for c in report.caveats)


def test_gross_pass_cannot_lift_a_failed_cost_stress_test():
    # regression: the only real stress test (25 bps) fails; the free 0 bps pass used to make it 1/2 = 50%
    def sharpe(spec, start, end, label):
        return {0.0: 0.3, 10.0: 0.1, 25.0: -0.05}[spec.costs_bps]

    runner = FakeRunner(sharpe, alpha_fn=lambda s, spec, label: 2.1, window=(date(2021, 1, 1), DEFAULT_END))
    _, report = replicate(None, fm_spec(), runner)
    assert check(report, "costs_0bps").passed is True
    assert check(report, "costs_25bps").passed is False
    assert report.verdict == "fails_to_replicate"
    assert "0 of 1 robustness checks passed (failed: costs_25bps)" in report.summary
    assert "costs_0bps is informational" in report.summary


def test_cost_level_between_zero_and_base_is_informational():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner, cost_levels_bps=(0.0, 5.0, 25.0))
    assert runner.spec_for("costs_5bps").costs_bps == 5.0
    assert check(report, "costs_5bps").note.startswith("informational")
    assert not check(report, "costs_25bps").note.startswith("informational")
    assert "passes if its Sharpe is still > 0" in check(report, "costs_25bps").description
    assert "costs_0bps, costs_5bps are informational" in report.summary
    assert "5 of 5 robustness checks passed" in report.summary  # halves, costs_25bps, 2 perturbations


def test_zero_cost_base_counts_every_positive_cost_level():
    _, report = replicate(None, cs_spec(costs_bps=0.0), FakeRunner())
    assert check(report, "costs_0bps").note.startswith("informational")
    assert not check(report, "costs_25bps").note.startswith("informational")


def test_replication_verdict_leaves_informational_checks_out_of_the_tally():
    checks = _checks(True, False, True, True)  # c2, c3 = gross / reused-base cost checks
    kw = dict(base_sharpe=1.0, base_alpha_t=2.5, base_years=10.0, replication_ratio=None)
    assert replication_verdict(checks=checks, **kw)[0] == "replicates"  # 3/4 if they counted
    verdict, reason = replication_verdict(checks=checks, informational={"c2", "c3"}, **kw)
    assert verdict == "partially_replicates" and "1 of 2 checks pass" in reason
    verdict, reason = replication_verdict(checks=_checks(None, True), informational={"c1"}, **kw)
    assert verdict == "inconclusive" and "harder than the base run" in reason


def test_failed_gross_check_is_not_blamed_on_costs():
    # regression: a non-positive Sharpe at 0 bps means no gross effect, not "killed by trading costs"
    def sharpe(spec, start, end, label):
        return -0.1 - 0.02 * spec.costs_bps  # 0 bps -> -0.1, 10 bps -> -0.3, 25 bps -> -0.6

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe))
    text = "\n".join(report.caveats)
    assert "The Sharpe is not positive even before trading costs (costs_0bps): there is no gross effect" in text
    assert "does not survive realistic trading costs (costs_25bps)." in text
    assert "costs_0bps)" not in text.split("does not survive realistic trading costs")[1]


def test_failed_low_cost_check_says_the_effect_is_missing_at_base_costs():
    def sharpe(spec, start, end, label):
        return 0.05 - 0.02 * spec.costs_bps  # 0 -> 0.05, 5 -> -0.05, 10 -> -0.15, 25 -> -0.45

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe), cost_levels_bps=(0.0, 5.0, 25.0))
    text = "\n".join(report.caveats)
    assert "not positive even at or below the base run's 10 bps costs (costs_5bps)" in text
    assert "does not survive realistic trading costs (costs_25bps)." in text
    assert "before trading costs (costs_0bps)" not in text  # the gross run is positive


def test_duplicate_cost_levels_are_run_once():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner, cost_levels_bps=(0, 0.0, 25))
    assert [c.name for c in report.checks if c.name.startswith("costs_")] == ["costs_0bps", "costs_25bps"]
    assert runner.labels().count("costs_0bps") == 1


def test_gross_vs_net_caveat():
    _, report = replicate(None, cs_spec(), FakeRunner())
    gross = [c for c in report.caveats if c.startswith("Gross vs net")]
    assert len(gross) == 1
    assert "1.00 at 10 bps" in gross[0] and "0bps: 1.20" in gross[0] and "25bps: 0.70" in gross[0]


def test_zero_cost_base_is_flagged_as_gross():
    _, report = replicate(None, cs_spec(costs_bps=0.0), FakeRunner())
    assert any("assumes zero transaction costs" in c for c in report.caveats)
    assert "costs_0bps" in [c.name for c in report.checks]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        replicate(None, cs_spec(), FakeRunner(), cost_levels_bps=(-5.0,))
    with pytest.raises(ValueError):
        replicate(None, cs_spec(), FakeRunner(), cost_levels_bps=(float("nan"),))
    with pytest.raises(ValueError):
        replicate(None, cs_spec(), FakeRunner(), min_years_per_split=0)


# ------------------------------------------------------------------------------------------------
# Parameter perturbations
# ------------------------------------------------------------------------------------------------


def test_cross_sectional_perturbations_quintiles_to_deciles_and_quarterly():
    runner = FakeRunner()
    _, report = replicate(None, cs_spec(), runner)
    dec = runner.spec_for("deciles_instead_of_quintiles")
    assert dec.portfolio.n_quantiles == 10
    assert dec.portfolio.style == "long_short" and dec.rebalance == "monthly"
    q = runner.spec_for("quarterly_rebalance")
    assert q.rebalance == "quarterly" and q.portfolio.n_quantiles == 5
    assert check(report, "deciles_instead_of_quintiles").sharpe == pytest.approx(1.1)
    assert check(report, "quarterly_rebalance").sharpe == pytest.approx(0.9)


def test_cross_sectional_deciles_go_to_quintiles_and_quarterly_to_monthly():
    runner = FakeRunner()
    spec = cs_spec(rebalance="quarterly", portfolio=PortfolioConstruction(n_quantiles=10))
    _, report = replicate(None, spec, runner)
    assert runner.spec_for("quintiles_instead_of_deciles").portfolio.n_quantiles == 5
    assert runner.spec_for("monthly_rebalance").rebalance == "monthly"


def test_cross_sectional_terciles_go_to_quintiles():
    runner = FakeRunner()
    replicate(None, cs_spec(portfolio=PortfolioConstruction(n_quantiles=3)), runner)
    assert runner.spec_for("quintiles_instead_of_terciles").portfolio.n_quantiles == 5


def test_cross_sectional_top_n_is_doubled():
    runner = FakeRunner()
    spec = cs_spec(portfolio=PortfolioConstruction(style="long_only", selection="top_n", top_n=20))
    replicate(None, spec, runner)
    assert runner.spec_for("top_40_instead_of_top_20").portfolio.top_n == 40
    assert not any(lbl and "instead_of_quintiles" in lbl for lbl in runner.labels())


def test_screen_only_perturbs_rebalance():
    runner = FakeRunner()
    replicate(None, screen_spec(), runner)
    assert runner.labels() == ["base", "first_half", "second_half", "costs_0bps", "costs_25bps", "quarterly_rebalance"]


def test_weekly_rebalance_alternative_is_monthly():
    runner = FakeRunner()
    replicate(None, screen_spec(rebalance="weekly"), runner)
    assert runner.spec_for("monthly_rebalance").rebalance == "monthly"


def test_time_series_gets_a_stricter_threshold():
    runner = FakeRunner()
    spec = ts_spec()
    _, report = replicate(None, spec, runner)
    alt = runner.spec_for("stricter_entry_threshold")
    assert alt.time_series.entry[0].value == pytest.approx(1.0)
    assert alt.time_series.entry[0].feature == "price_vs_sma_200_pct"
    assert spec.time_series.entry[0].value == 0.0  # original untouched
    assert "quarterly_rebalance" not in runner.labels()
    assert "> 1 instead of > 0" in check(report, "stricter_entry_threshold").description


def test_time_series_nonzero_threshold_moves_ten_percent_stricter():
    runner = FakeRunner()
    replicate(None, ts_spec([Condition(feature="rsi_14", op="<", value=30.0)]), runner)
    assert runner.spec_for("stricter_entry_threshold").time_series.entry[0].value == pytest.approx(27.0)


def test_time_series_with_several_conditions_has_no_perturbation():
    runner = FakeRunner()
    entry = [Condition(feature="price_vs_sma_200_pct", op=">", value=0.0),
             Condition(feature="rsi_14", op="<", value=70.0)]
    replicate(None, ts_spec(entry), runner)
    assert runner.labels() == ["base", "first_half", "second_half", "costs_0bps", "costs_25bps"]


def test_factor_model_has_no_perturbations():
    runner = FakeRunner()
    replicate(None, fm_spec(), runner)
    assert runner.labels() == ["base", "first_half", "second_half", "costs_0bps", "costs_25bps"]


def test_perturbation_fails_when_sharpe_collapses():
    def sharpe(spec, start, end, label):
        return 0.3 if label == "deciles_instead_of_quintiles" else default_sharpe(spec, start, end, label)

    _, report = replicate(None, cs_spec(), FakeRunner(sharpe))
    c = check(report, "deciles_instead_of_quintiles")
    assert c.passed is False
    assert "30% of the base" in c.note


# ------------------------------------------------------------------------------------------------
# Verdicts
# ------------------------------------------------------------------------------------------------


def test_verdict_replicates_with_claim_close_to_replication():
    _, report = replicate(make_candidate(reported_sharpe=1.25, reported_t=3.0), cs_spec(), FakeRunner())
    assert report.claimed_sharpe == 1.25
    assert report.claimed_t_stat == 3.0
    assert report.replication_ratio == pytest.approx(0.8)
    assert report.verdict == "replicates"
    assert "80% of the claimed Sharpe" in report.summary


def test_verdict_partial_when_claim_far_above_replication():
    _, report = replicate(make_candidate(reported_sharpe=2.5), cs_spec(), FakeRunner())
    assert report.replication_ratio == pytest.approx(0.4)
    assert report.verdict == "partially_replicates"
    assert "only 40% of the claimed Sharpe" in report.summary


def test_verdict_partial_when_alpha_not_significant():
    runner = FakeRunner(alpha_fn=lambda s, spec, label: 1.5)
    _, report = replicate(None, cs_spec(), runner)
    assert report.base_alpha_t_stat == 1.5
    assert report.verdict == "partially_replicates"
    assert "not significant" in report.summary


def test_verdict_uses_sharpe_without_regression():
    runner = FakeRunner(alpha_fn=lambda s, spec, label: None)
    _, report = replicate(None, cs_spec(), runner)
    assert report.base_alpha_t_stat is None
    assert report.verdict == "replicates"  # Sharpe 1.0 >= 0.5

    def weak(spec, start, end, label):
        return 0.4 if label in ("base", "costs_0bps") else 0.35

    _, report = replicate(None, cs_spec(), FakeRunner(weak, alpha_fn=lambda s, spec, label: None))
    assert report.verdict == "partially_replicates"
    assert "below 0.5" in report.summary


def test_verdict_fails_when_base_sharpe_negative():
    def neg(spec, start, end, label):
        return -0.5

    _, report = replicate(None, cs_spec(), FakeRunner(neg, alpha_fn=lambda s, spec, label: -1.0))
    assert report.verdict == "fails_to_replicate"
    assert "not positive" in report.summary


def test_replicates_requires_a_positive_base_sharpe():
    # regression: a significant alpha t-stat used to be enough even when the strategy loses money
    kw = dict(base_alpha_t=2.5, base_years=10.0, replication_ratio=None, checks=_checks(True, True, True, True))
    for base in (-0.1, 0.0):
        verdict, reason = replication_verdict(base_sharpe=base, **kw)
        assert verdict == "fails_to_replicate" and "not positive" in reason
    assert replication_verdict(base_sharpe=0.01, **kw)[0] == "replicates"


def test_money_losing_strategy_with_significant_alpha_t_does_not_replicate():
    # regression (reviewer's probe): base -0.1, 0 bps 0.1, 25 bps -0.3, both perturbations -0.2, alpha t
    # 2.5. Negative perturbations used to "keep the sign" and the verdict was "replicates".
    def sharpe(spec, start, end, label):
        return {"base": -0.1, "costs_0bps": 0.1, "costs_25bps": -0.3}.get(label, -0.2)

    runner = FakeRunner(sharpe, alpha_fn=lambda s, spec, label: 2.5, window=(date(2021, 1, 1), DEFAULT_END))
    _, report = replicate(None, cs_spec(), runner)
    for name in ("deciles_instead_of_quintiles", "quarterly_rebalance"):
        c = check(report, name)
        assert c.passed is False
        assert "no positive premium to keep" in c.note
    assert report.verdict == "fails_to_replicate"
    assert "the full-sample Sharpe is -0.10 (not positive)" in report.summary
    assert "0 of 3 robustness checks passed" in report.summary


def test_verdict_fails_when_most_checks_fail():
    def fragile(spec, start, end, label):
        return 1.0 if label in ("base", "costs_0bps") else -0.2

    _, report = replicate(None, cs_spec(), FakeRunner(fragile))
    runnable = [c for c in report.checks if c.passed is not None]
    assert sum(c.passed for c in runnable) == 1 and len(runnable) == 6
    assert report.verdict == "fails_to_replicate"


def test_verdict_inconclusive_for_short_base():
    runner = FakeRunner(window=(date(2024, 1, 1), date(2025, 12, 31)))
    _, report = replicate(make_candidate(published=date(2024, 6, 1)), cs_spec(), runner)
    assert report.verdict == "inconclusive"
    assert "2.0 years" in report.summary
    assert any("Only 2.0 years" in c for c in report.caveats)
    assert check(report, "first_half").passed is None
    assert check(report, "post_publication").passed is None


def test_verdict_inconclusive_when_every_check_fails():
    runner = FakeRunner(fail={"*": RuntimeError("provider offline")})
    _, report = replicate(None, cs_spec(), runner)
    assert all(c.passed is None for c in report.checks)
    assert all("provider offline" in c.note for c in report.checks)
    assert report.verdict == "inconclusive"
    assert "None of the robustness checks could be run." in report.summary


def test_verdict_inconclusive_when_base_has_no_statistics():
    runner = FakeRunner(no_stats={"base"})
    _, report = replicate(None, cs_spec(), runner)
    assert report.base_sharpe is None
    assert report.verdict == "inconclusive"
    assert "no Sharpe ratio" in report.summary


def _checks(*passed):
    return [RobustnessCheck(name=f"c{i}", description="", sharpe=1.0, cagr_pct=1.0, alpha_t_stat=None, n_periods=12,
                            passed=p) for i, p in enumerate(passed)]


def test_replication_verdict_thresholds():
    kw = dict(base_sharpe=1.0, base_alpha_t=2.0, base_years=MIN_BASE_YEARS, replication_ratio=None)
    assert REPLICATES_MIN_PASS_SHARE == 0.75 and PARTIAL_MIN_PASS_SHARE == 0.5
    assert replication_verdict(checks=_checks(True, True, True, False), **kw)[0] == "replicates"  # exactly 75%
    assert replication_verdict(checks=_checks(True, True, False, False), **kw)[0] == "partially_replicates"  # 50%
    assert replication_verdict(checks=_checks(True, False, False, False), **kw)[0] == "fails_to_replicate"
    # skipped checks (None) are not in the denominator
    assert replication_verdict(checks=_checks(True, True, True, False, None, None), **kw)[0] == "replicates"
    # the claim gates "replicates" but not "partially"
    assert replication_verdict(checks=_checks(True, True, True, True), **{**kw, "replication_ratio": 0.5})[0] == "replicates"
    assert replication_verdict(checks=_checks(True, True, True, True), **{**kw, "replication_ratio": 0.49})[0] == "partially_replicates"
    # alpha t just below 2
    assert replication_verdict(checks=_checks(True, True), **{**kw, "base_alpha_t": 1.99})[0] == "partially_replicates"
    # no regression -> Sharpe criterion
    assert replication_verdict(checks=_checks(True, True), **{**kw, "base_alpha_t": None, "base_sharpe": 0.5})[0] == "replicates"
    assert replication_verdict(checks=_checks(True, True), **{**kw, "base_alpha_t": None, "base_sharpe": 0.49})[0] == "partially_replicates"
    # inconclusive branches
    assert replication_verdict(checks=_checks(True), **{**kw, "base_years": 2.5})[0] == "inconclusive"
    assert replication_verdict(checks=_checks(None, None), **kw)[0] == "inconclusive"
    assert replication_verdict(checks=_checks(True), **{**kw, "base_sharpe": None})[0] == "inconclusive"
    # negative Sharpe with all checks passing is still not "partially"
    assert replication_verdict(checks=_checks(True, True), **{**kw, "base_sharpe": -0.1, "base_alpha_t": 0.0})[0] == "fails_to_replicate"


# ------------------------------------------------------------------------------------------------
# Claimed vs replicated
# ------------------------------------------------------------------------------------------------


def test_claimed_vs_replicated_caveats():
    cand = make_candidate(reported_sharpe=1.25, reported_t=4.0, sample_period="1963-2019")
    _, report = replicate(cand, cs_spec(), FakeRunner())
    text = "\n".join(report.caveats)
    assert "gross (pre-cost), long-short" in text
    assert "Replicated Sharpe 1.00 vs claimed 1.25: replication ratio 0.80" in text
    assert "Before costs (0 bps) the replicated Sharpe is 1.20, a gross replication ratio of 0.96" in text
    assert "t-stat of 4.00" in text and "alpha t-stat is 2.50" in text
    assert "overlaps the source's sample (1963-2019) in 2016-2019" in text


def test_nonpositive_claim_gives_no_ratio():
    _, report = replicate(make_candidate(reported_sharpe=-0.2), cs_spec(), FakeRunner())
    assert report.claimed_sharpe == -0.2
    assert report.replication_ratio is None
    assert any("not positive, so no replication ratio" in c for c in report.caveats)
    assert report.verdict == "replicates"  # an unusable claim does not block the verdict


def test_annual_return_claim_compares_cagr():
    _, report = replicate(make_candidate(reported_ret=12.0), cs_spec(), FakeRunner())
    assert report.claimed_sharpe is None and report.replication_ratio is None
    cav = [c for c in report.caveats if "claims about 12.0% a year" in c]
    assert len(cav) == 1
    assert "compounded 8.0% a year" in cav[0] and "9.6% before costs" in cav[0]


def test_monthly_sharpe_claim_is_flagged():
    _, report = replicate(make_candidate(reported_sharpe=0.28), cs_spec(), FakeRunner())
    assert report.replication_ratio == pytest.approx(1.0 / 0.28)
    assert any("monthly (non-annualised)" in c and "0.97" in c for c in report.caveats)


def test_out_of_sample_window_is_noted():
    _, report = replicate(make_candidate(sample_period="1927 to 2009"), cs_spec(), FakeRunner())
    assert any("does not overlap the source's sample (1927-2009)" in c for c in report.caveats)


def test_long_only_replication_of_a_claim_is_flagged():
    spec = cs_spec(portfolio=PortfolioConstruction(style="long_only"))
    _, report = replicate(make_candidate(reported_sharpe=1.0), spec, FakeRunner())
    assert any("long-only portfolio" in c for c in report.caveats)


# ------------------------------------------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------------------------------------------


def test_failed_check_is_recorded_and_suite_continues():
    runner = FakeRunner(fail={"second_half": RuntimeError("no prices before 2021")})
    msgs: list[str] = []
    _, report = replicate(None, cs_spec(), runner, progress=msgs.append)
    c = check(report, "second_half")
    assert c.passed is None and c.sharpe is None and c.n_periods == 0
    assert "RuntimeError" in c.note and "no prices before 2021" in c.note
    # later checks still ran
    assert runner.labels()[-1] == "quarterly_rebalance"
    assert len(report.checks) == 6
    assert any("second_half failed" in m for m in msgs)
    assert report.verdict == "replicates"  # 5/5 runnable checks pass


def test_runner_returning_garbage_is_a_failed_check():
    class Garbage(FakeRunner):
        def backtest(self, spec, *, label=None):
            if label == "costs_25bps":
                self.calls.append((label, spec))
                return {"not": "a result"}
            return super().backtest(spec, label=label)

    _, report = replicate(None, cs_spec(), Garbage())
    c = check(report, "costs_25bps")
    assert c.passed is None and "TypeError" in c.note


def test_base_failure_raises():
    runner = FakeRunner(fail={"base": ConnectionError("data provider down")})
    with pytest.raises(ConnectionError, match="data provider down"):
        replicate(None, cs_spec(), runner)
    assert runner.labels() == ["base"]


@pytest.mark.parametrize("garbage", [None, {"not": "a result"}])
def test_base_run_returning_garbage_raises_a_clear_type_error(garbage):
    # regression: used to crash later with "'NoneType' object has no attribute 'stats'"
    class Garbage(FakeRunner):
        def backtest(self, spec, *, label=None):
            self.calls.append((label, spec))
            return garbage

    runner = Garbage()
    with pytest.raises(TypeError, match=r"runner returned \w+ for the base run, not a BacktestResult"):
        replicate(None, cs_spec(), runner)
    assert runner.labels() == ["base"]


# ------------------------------------------------------------------------------------------------
# Progress, summary and caveats
# ------------------------------------------------------------------------------------------------


def test_progress_callback_reports_each_step():
    msgs: list[str] = []
    _, report = replicate(make_candidate(published=date(2025, 1, 1)), cs_spec(costs_bps=25.0), FakeRunner(),
                          progress=msgs.append)
    assert "base backtest" in msgs[0] and "mom_12_1" in msgs[0]
    n = len(report.checks)
    for i, c in enumerate(report.checks, start=1):
        assert any(m.startswith(f"Replication check {i}/{n}: {c.name}") for m in msgs)
    assert any("post_publication skipped" in m for m in msgs)
    assert any("costs_25bps (same as the base run, reused)" in m for m in msgs)
    # base Sharpe 0.7 at 25 bps -> alpha t 1.75 < 2 -> partially replicates; costs_0bps and the reused
    # costs_25bps are at or below the base's 25 bps, so only the 4 other checks count
    assert report.verdict == "partially_replicates"
    assert msgs[-1] == "Replication verdict: partially replicates (4/4 checks passed)"
    assert len(msgs) == n + 2


def test_progress_is_optional():
    base, report = replicate(None, cs_spec(), FakeRunner(), progress=None)
    assert report.verdict == "replicates"


@pytest.mark.parametrize("verdict_case", ["replicates", "partial", "fails", "inconclusive"])
def test_summary_has_two_to_four_plain_sentences(verdict_case):
    cand = make_candidate(reported_sharpe=1.25)
    if verdict_case == "replicates":
        runner = FakeRunner()
    elif verdict_case == "partial":
        runner = FakeRunner(alpha_fn=lambda s, spec, label: 1.0)
    elif verdict_case == "fails":
        runner = FakeRunner(lambda spec, start, end, label: -0.4)
    else:
        runner = FakeRunner(window=(date(2024, 1, 1), date(2025, 6, 30)))
    _, report = replicate(cand, cs_spec(), runner)
    parts = sentences(report.summary)
    assert 2 <= len(parts) <= 4, parts
    assert "Verdict:" in report.summary
    assert report.summary.startswith("The replication earned a Sharpe ratio of")


def test_warnings_are_collected_from_all_runs_and_deduplicated():
    shared = "Universe = today's S&P 500 constituents: survivorship bias (delisted names missing)."

    def warnings(label):
        out = [shared, "3 signal date(s) not executed: fewer than 1 session(s) of price data after them (2025-12-31)"]
        if label == "first_half":
            out.append("ABC: no price from 2018-04-30 on while held (delisted, or its data ends)")
        return out

    def usage(label):
        return [
            DataUsage(dataset="prices", source="fake", coverage="100/100 tickers", point_in_time=True),
            DataUsage(dataset="fundamentals", source="snapshot vendor", coverage="today only", point_in_time=False,
                      notes="Snapshot, not as-filed."),
            DataUsage(dataset="risk_free", source="FRED", coverage="full", point_in_time=True),
        ]

    _, report = replicate(None, cs_spec(), FakeRunner(warnings_fn=warnings, data_usage_fn=usage))
    cav = report.caveats
    assert sum(1 for c in cav if shared in c) == 1
    assert sum(1 for c in cav if "signal date(s) not executed" in c) == 1
    assert any(c.startswith("Survivorship bias") and "every run" in c for c in cav)
    assert sum(1 for c in cav if "Dataset 'fundamentals'" in c and "not point-in-time" in c) == 1
    delist = [c for c in cav if "ABC: no price" in c]
    assert len(delist) == 1 and "[check run: first_half]" in delist[0]
    assert len(cav) == len(set(cav))
    # data-related warnings come before generic ones
    i_shared = next(i for i, c in enumerate(cav) if shared in c)
    i_generic = next(i for i, c in enumerate(cav) if "signal date(s) not executed" in c)
    assert i_shared < i_generic
    # no post-publication check (no candidate) -> the data caveat is mentioned in the summary
    assert "Data caveats" in report.summary


def test_negated_survivorship_is_not_flagged():
    def warnings(label):
        return ["Universe is survivorship-bias-free (includes delisted names)."]

    _, report = replicate(None, cs_spec(), FakeRunner(warnings_fn=warnings))
    assert not any(c.startswith("Survivorship bias") for c in report.caveats)


def test_many_generic_warnings_are_capped():
    def warnings(label):
        return [f"generic note number {i}" for i in range(20)]

    _, report = replicate(None, cs_spec(), FakeRunner(warnings_fn=warnings))
    generic = [c for c in report.caveats if "generic note number" in c]
    assert len(generic) == 8
    assert any("12 further backtest warnings not listed" in c for c in report.caveats)


def test_unsupported_requests_become_caveats():
    spec = cs_spec(unsupported_requests=["analyst revisions"])
    _, report = replicate(None, spec, FakeRunner())
    assert any("analyst revisions" in c for c in report.caveats)
