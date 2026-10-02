"""Offline tests for the HTML reports, SVG charts and the local dashboard (aitrading.report)."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aitrading.backtest.metrics import performance_stats
from aitrading.backtest.models import (
    BacktestInterpretation,
    BacktestResult,
    CitedMetric,
    DataUsage,
    FactorConstructionCheck,
    FactorRegression,
    QuantileAnalysis,
)
from aitrading.backtest.protocols import BacktestRunner
from aitrading.core.models import (
    DislocationThesis,
    EvidenceCheck,
    FunnelStep,
    GroundingReport,
    InvestmentIdea,
    LLMCallRecord,
    PipelineResult,
    QuantEvidence,
    QuoteEvidence,
    RankedCandidate,
)
from aitrading.discovery.models import (
    IdeaCandidate,
    IdeaExtraction,
    ReplicationReport,
    RobustnessCheck,
    SourceDocument,
)
from aitrading.report import svg
from aitrading.report.dashboard import open_in_browser, write_dashboard
from aitrading.report.html import (
    DISCLAIMER,
    describe_strategy,
    render_backtest_html,
    render_ideas_inbox,
    render_pipeline_html,
    render_strategy_page,
    returns_frame,
    template_info,
)
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.spec import PortfolioConstruction, SignalComponent, StrategySpec, TimeSeriesRule

EVIL = "<script>alert(1)</script>"
SVG_NS = "{http://www.w3.org/2000/svg}"


# ------------------------------------------------------------------------------------------------
# Fixtures: hand-built results and a fake runner
# ------------------------------------------------------------------------------------------------


def _month_ends(n: int, start: str = "2016-01-31") -> list[date]:
    return [d.date() for d in pd.date_range(start, periods=n, freq="ME")]


def _returns(n: int, mu: float, sigma: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def make_result(
    *,
    idea: str = "12-1 momentum, long the top decile, short the bottom decile",
    spec: StrategySpec | dict | None = None,
    n: int = 96,
    mu: float = 0.008,
    warnings: list[str] | None = None,
    run_id: str = "run-001",
    interpretation: BacktestInterpretation | None | str = "default",
    holdings: dict[str, float] | None = None,
    extra_series: dict[str, list[float | None]] | None = None,
    finished: datetime | None = None,
) -> BacktestResult:
    dates = _month_ends(n)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    strat = _returns(n, mu, 0.04, 1)
    bench = _returns(n, 0.007, 0.045, 2)
    long_ = _returns(n, 0.011, 0.05, 3)
    short = _returns(n, 0.004, 0.055, 4)
    series = {"strategy": strat, "benchmark": bench, "long": long_, "short": short}
    stats = {k: performance_stats(pd.Series(v, index=idx), label=k, benchmark=pd.Series(bench, index=idx) if k != "benchmark" else None)
             for k, v in series.items()}
    if spec is None:
        spec = TEMPLATES["momentum_12_1"].spec()
    spec_json = spec.model_dump(mode="json") if isinstance(spec, StrategySpec) else spec
    reg = FactorRegression(model="ff3", factor_source="Kenneth French Data Library", n=n, alpha_annual_pct=3.1, alpha_t_stat=2.4,
                           betas={"Mkt-RF": 0.12, "SMB": -0.31, "HML": -0.45}, beta_t_stats={"Mkt-RF": 1.1, "SMB": -2.0, "HML": -3.3},
                           r_squared=0.21)
    quant = QuantileAnalysis(n_quantiles=5, annual_return_by_quantile_pct=[-2.5, 3.0, 6.1, 8.0, 12.4], spread_annual_pct=14.9,
                             monotonicity=1.0, ic_mean=0.031, ic_t_stat=2.8, ic_hit_rate_pct=58.0)
    checks = [
        FactorConstructionCheck(factor="SMB", correlation_with_official=0.62, annual_premium_constructed_pct=1.1,
                                annual_premium_official_pct=1.9, n_overlap_periods=n),
        FactorConstructionCheck(factor="HML", correlation_with_official=0.81, annual_premium_constructed_pct=2.0,
                                annual_premium_official_pct=2.4, n_overlap_periods=n),
        FactorConstructionCheck(factor="Mom", correlation_with_official=0.31, annual_premium_constructed_pct=None,
                                annual_premium_official_pct=4.0, n_overlap_periods=n),
    ]
    if holdings is None:
        holdings = {f"L{i:02d}": 0.02 + i * 0.0005 for i in range(40)}
        holdings.update({f"S{i:02d}": -(0.02 + i * 0.0004) for i in range(30)})
    if interpretation == "default":
        sharpe = stats["strategy"].sharpe
        interpretation = BacktestInterpretation(
            summary="Momentum earned a positive spread with a monotonic quantile pattern.",
            verdict="promising",
            key_findings=["Quantile returns rise monotonically.", "Alpha t-stat of 2.4 is weak evidence."],
            cited_metrics=[
                CitedMetric(path="stats.strategy.sharpe", value=round(sharpe, 2), meaning="strategy Sharpe ratio"),
                CitedMetric(path="regression.alpha_t_stat", value=9.99, meaning="deliberately wrong citation"),
            ],
            biases_and_caveats=["Survivorship bias: the universe is today's constituents."],
            next_experiments=["Test deciles instead of quintiles."],
        )
    returns: dict[str, list[float | None]] = {k: [float(x) for x in v] for k, v in series.items()}
    if extra_series:
        returns.update(extra_series)
    return BacktestResult(
        run_id=run_id,
        idea=idea,
        spec=spec_json,
        provider="synthetic",
        llm="heuristic",
        start=dates[0] - timedelta(days=31),
        end=dates[-1],
        rebalance="monthly",
        returns=returns,
        dates=dates,
        stats=stats,
        regression=reg,
        quantiles=quant,
        factor_checks=checks,
        latest_holdings=holdings,
        data_usage=[
            DataUsage(dataset="prices", source="synthetic", coverage="150/150 tickers, 2016-01 to 2023-12", point_in_time=True),
            DataUsage(dataset="fundamentals", source="SEC EDGAR", coverage="140/150", point_in_time=False, notes="restated values"),
        ],
        interpretation=interpretation if isinstance(interpretation, BacktestInterpretation) or interpretation is None else None,
        llm_calls=[LLMCallRecord(purpose="interpret", model="claude-test", input_tokens=1200, output_tokens=300, latency_s=2.5)],
        warnings=warnings if warnings is not None else
        ["Survivorship bias: universe = current constituents; delisted names are missing.", "3 names had price gaps."],
        started_at=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        finished_at=finished or datetime(2026, 9, 30, 12, 5, tzinfo=timezone.utc),
    )


class FakeRunner:
    """Deterministic in-memory implementation of the BacktestRunner protocol."""

    provider_name = "fake"

    def __init__(self, tickers: list[str] | None = None):
        self.tickers = tickers or ["AAA", "BBB", "CCC", "DDD"]

    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        seed = sum(map(ord, spec.name)) % 97
        mu = 0.002 + seed / 10_000
        return make_result(idea=spec.idea, spec=spec, mu=mu, run_id=label or f"fake-{spec.name}", warnings=[])

    def target_portfolio(self, spec: StrategySpec, as_of: date) -> pd.Series:
        w = 1.0 / len(self.tickers)
        if spec.portfolio.style == "long_short" and spec.kind == "cross_sectional":
            half = len(self.tickers) // 2
            return pd.Series({t: (w * 2 if i < half else -w * 2) for i, t in enumerate(self.tickers)})
        return pd.Series({t: w for t in self.tickers})

    def latest_prices(self, tickers: list[str], as_of: date) -> pd.Series:
        bump = 1.0 + (as_of.toordinal() % 10) / 100.0
        return pd.Series({t: (10.0 + i) * bump for i, t in enumerate(tickers)}, dtype=float)


def _svgs(page: str) -> list[str]:
    return re.findall(r"<svg\b.*?</svg>", page, flags=re.S)


class _TagChecker(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "area", "base", "col", "embed", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.in_svg = 0

    def handle_starttag(self, tag, attrs):
        if tag == "svg":
            self.in_svg += 1
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()
        if tag == "svg":
            self.in_svg -= 1


def assert_well_formed(page: str) -> None:
    assert page.startswith("<!DOCTYPE html>")
    checker = _TagChecker()
    checker.feed(page)
    assert not checker.errors, checker.errors[:5]
    assert not checker.stack, checker.stack
    for s in _svgs(page):
        ET.fromstring(s)  # every embedded chart is valid XML
    # self-contained: no external scripts, stylesheets or images
    assert "<script" not in page.lower()
    assert not re.search(r"<link\b", page, flags=re.I)
    assert not re.search(r"\bsrc\s*=", page, flags=re.I)
    assert "@import" not in page


# ------------------------------------------------------------------------------------------------
# SVG helpers
# ------------------------------------------------------------------------------------------------


def _daily(n: int = 3000, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2014-01-01", periods=n)
    r = np.random.default_rng(seed).normal(0.0003, 0.01, n)
    return pd.Series(np.cumprod(1 + r), index=idx)


def test_line_chart_is_valid_accessible_responsive_svg():
    s = svg.line_chart({"Strategy": _daily(), "Benchmark": _daily(seed=1)}, title="Growth of $1", y_label="Value")
    root = ET.fromstring(s)
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("width") == "100%"
    assert re.fullmatch(r"0 0 [\d.]+ [\d.]+", root.get("viewBox"))
    assert root.get("role") == "img"
    title = root.find(f"{SVG_NS}title")
    assert title is not None and title.text == "Growth of $1"
    assert root.get("aria-labelledby").split()[0] == title.get("id")
    assert root.find(f"{SVG_NS}desc") is not None
    # colours come from CSS variables (themeable) with fallbacks
    assert "var(--series-1" in s and "var(--series-2" in s and "var(--chart-grid" in s
    # legend for >= 2 series, with names
    assert "Strategy" in s and "Benchmark" in s
    # hover layer with native tooltips
    assert 'class="hz"' in s


def test_line_chart_breaks_lines_at_nan_gaps():
    s = pd.Series([1.0, 1.1, 1.2, np.nan, np.nan, 1.3, 1.4, 1.5], index=pd.date_range("2024-01-31", periods=8, freq="ME"))
    out = svg.line_chart({"x": s}, title="gap")
    root = ET.fromstring(out)
    paths = [p.get("d") for p in root.iter(f"{SVG_NS}path") if p.get("fill") == "none"]
    assert len(paths) == 1
    assert paths[0].count("M") == 2  # two separate segments
    assert "nan" not in out.lower().replace("n/a", "")


def test_line_chart_isolated_point_between_gaps_is_drawn_as_dot():
    s = pd.Series([1.0, 1.1, np.nan, 1.3, np.nan, 1.4, 1.5], index=pd.date_range("2024-01-31", periods=7, freq="ME"))
    root = ET.fromstring(svg.line_chart({"x": s}, title="dots"))
    circles = list(root.iter(f"{SVG_NS}circle"))
    assert len(circles) >= 2  # isolated point + end marker


def test_downsample_limits_points_and_keeps_extremes():
    s = _daily(5000)
    s.iloc[1234] = 50.0  # a spike
    s.iloc[3210] = 0.01  # a crash
    ds = svg.downsample(s, 800)
    assert len(ds) <= 800
    assert ds.index[0] == s.index[0] and ds.index[-1] == s.index[-1]
    assert ds.max() == 50.0 and ds.min() == 0.01
    assert ds.index.is_monotonic_increasing
    short = _daily(100)
    assert svg.downsample(short, 800) is short


def test_downsample_keeps_nan_gaps():
    s = _daily(4000)
    s.iloc[2000:2100] = np.nan
    ds = svg.downsample(s, 800)
    assert len(ds) <= 800
    assert ds.isna().any()
    gap = ds[(ds.index >= s.index[2000]) & (ds.index < s.index[2100])]
    assert gap.isna().all()


def test_line_chart_long_series_is_downsampled():
    out = svg.line_chart({"a": _daily(6000)}, title="long")
    root = ET.fromstring(out)
    d = [p.get("d") for p in root.iter(f"{SVG_NS}path") if p.get("fill") == "none"][0]
    n_points = d.count("L") + d.count("M")
    assert n_points <= svg.MAX_POINTS
    assert len(out) < 120_000


def test_line_chart_log_scale_and_fallback():
    s = pd.Series(np.geomspace(1, 40, 120), index=pd.date_range("2010-01-31", periods=120, freq="ME"))
    out = svg.line_chart({"w": s}, title="log", log_scale=True, y_format="${:,.0f}")
    assert "Log-scale" in out
    ET.fromstring(out)
    neg = s - 5  # contains values <= 0 -> linear fallback
    out2 = svg.line_chart({"w": neg}, title="lin", log_scale=True)
    assert "Linear line chart" in out2


def test_charts_handle_empty_and_all_nan_input():
    for out in (
        svg.line_chart({}, title="empty"),
        svg.line_chart({"a": pd.Series([np.nan, np.nan], index=pd.date_range("2020-01-01", periods=2))}, title="nan"),
        svg.area_drawdown_chart(pd.Series(dtype=float)),
        svg.bar_chart([], [], title="none"),
        svg.bar_chart(["a"], [None], title="none"),
        svg.horizontal_bar_chart(["a"], [float("nan")], title="nan"),
        svg.sparkline(pd.Series(dtype=float)),
    ):
        root = ET.fromstring(out)
        assert root.tag == f"{SVG_NS}svg"


def test_svg_text_is_escaped():
    s = _daily(50)
    out = svg.line_chart({EVIL: s, "b&b": s * 1.1}, title=EVIL, y_label=EVIL)
    assert EVIL not in out
    root = ET.fromstring(out)
    assert root.find(f"{SVG_NS}title").text == EVIL  # round-trips as text, not markup
    out2 = svg.bar_chart([EVIL, "ok"], [1.0, -2.0], title=EVIL)
    assert EVIL not in out2
    ET.fromstring(out2)
    out3 = svg.horizontal_bar_chart([EVIL], [0.5], [0.1], annotations=[EVIL], title=EVIL)
    assert EVIL not in out3
    ET.fromstring(out3)


def test_area_drawdown_chart_marks_max_drawdown():
    r = pd.Series([0.1, -0.2, -0.1, 0.05, 0.3, -0.05], index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    from aitrading.backtest.metrics import drawdown_series

    out = svg.area_drawdown_chart(drawdown_series(r), title="DD")
    ET.fromstring(out)
    dd = drawdown_series(r).min()
    assert f"Max drawdown {dd * 100:.1f}%" in out
    assert "var(--neg" in out
    assert 'class="area"' in out


def test_bar_chart_highlights_negative_and_labels_values():
    out = svg.bar_chart(["Q1", "Q2", "Q3"], [-2.0, None, 5.5], title="Quantiles", value_format="{:.1f}%")
    root = ET.fromstring(out)
    fills = [p.get("style") for p in root.iter(f"{SVG_NS}path")]
    assert any("var(--neg" in f for f in fills)
    assert any("var(--series-1" in f for f in fills)
    assert "-2.0%" in out and "5.5%" in out and "n/a" in out
    plain = svg.bar_chart(["a", "b"], [-1.0, 1.0], title="t", highlight_negative=False)
    assert "var(--neg" not in plain


def test_horizontal_bar_chart_with_errors_and_annotations():
    out = svg.horizontal_bar_chart(["Mkt-RF", "SMB"], [1.02, -0.3], [0.1, 0.2], annotations=["t=10.2", "t=-1.5"], title="Betas")
    root = ET.fromstring(out)
    assert "t=10.2" in out and "t=-1.5" in out
    assert len([ln for ln in root.iter(f"{SVG_NS}line") if ln.get("class") == "whisker"]) == 2
    with pytest.raises(ValueError):
        svg.horizontal_bar_chart(["a", "b"], [1.0])


def test_sparkline_and_unique_ids():
    a = svg.sparkline(_daily(300), title="NAV")
    b = svg.sparkline(_daily(300), title="NAV")
    ra, rb = ET.fromstring(a), ET.fromstring(b)
    assert ra.find(f"{SVG_NS}title").get("id") != rb.find(f"{SVG_NS}title").get("id")
    assert "NAV" in a


def test_nice_and_log_ticks():
    t = svg.nice_ticks(0.95, 3.4)
    assert t[0] <= 0.95 and t[-1] >= 3.4 and 3 <= len(t) <= 9
    steps = np.diff(t)
    assert np.allclose(steps, steps[0])
    assert svg.nice_ticks(5, 5)[0] < 5 < svg.nice_ticks(5, 5)[-1]
    lt = svg.log_ticks(0.8, 120)
    assert all(0.8 <= x <= 120 for x in lt) and 1.0 in lt and 100.0 in lt and len(lt) <= 8
    assert svg.log_ticks(-1, 10) == []


def test_date_axis_labels_years_or_months():
    long = svg.line_chart({"a": _daily(3000)}, title="long")
    assert re.search(r">20(1[5-9]|2\d)<", long)
    short_idx = pd.date_range("2025-01-31", periods=10, freq="ME")
    short = svg.line_chart({"a": pd.Series(np.arange(10.0) + 1, index=short_idx)}, title="short")
    assert re.search(r">(Mar|Apr|May|Jun|Jul|Aug|Sep|Oct)( 2025)?<", short)


def test_line_chart_percent_and_custom_formats():
    s = pd.Series([0.0, -0.05, -0.12, -0.03], index=pd.date_range("2024-01-31", periods=4, freq="ME"))
    out = svg.line_chart({"dd": s, "other": s / 2}, title="pct", percent=True, reference=0.0)
    assert re.search(r">-1?\d+%<", out) and 'class="reference"' in out
    out2 = svg.line_chart({"v": s + 1}, title="fmt", y_format=lambda v: f"{v:.3f}x")
    assert "x<" in out2
    ET.fromstring(out2)


def test_line_chart_accepts_date_objects_and_numeric_index():
    s = pd.Series([1.0, 2.0, 3.0], index=[date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)])
    ET.fromstring(svg.line_chart({"d": s}, title="dates"))
    n = pd.Series([1.0, 4.0, 9.0], index=[1, 2, 3])
    ET.fromstring(svg.line_chart({"n": n}, title="numbers"))


# ------------------------------------------------------------------------------------------------
# Plain-English idea text
# ------------------------------------------------------------------------------------------------


def test_describe_cross_sectional_momentum():
    text = " ".join(describe_strategy(TEMPLATES["momentum_12_1"].spec()))
    assert text.startswith("Each month, take US stocks and rank them by 12-1 momentum")
    assert "higher is better" in text
    assert "buy the top 10% (decile) and short the bottom 10%, equal weighted" in text
    assert "10 bps one-way" in text
    assert "Fama-French 3-factor model" in text


def test_describe_long_only_top_n_composite_and_filters():
    spec = StrategySpec(
        name="custom", idea="custom", kind="cross_sectional", rebalance="weekly",
        signal=[SignalComponent(feature="roe_pct", direction="higher_is_better", weight=3.0, sector_neutral=True),
                SignalComponent(feature="ev_to_ebitda", direction="lower_is_better", weight=1.0, transform="zscore")],
        filters=[Condition(feature="market_cap_usd_bn", op=">", value=2, rationale="no small caps")],
        portfolio=PortfolioConstruction(style="long_only", selection="top_n", top_n=25, weighting="inverse_vol", max_weight=0.05),
        costs_bps=25, delisting_return=-0.3, benchmark="SPY",
    )
    sentences = describe_strategy(spec)
    first = sentences[0]
    assert first.startswith("Each week")
    assert "composite of 2 signals" in first and "return on equity (higher is better, 75%)" in first
    assert "EV/EBITDA (lower is better, 25%)" in first
    assert "within each sector" in first and "z-scores" in first
    assert "buy the top 25 names" in first and "short" not in first
    assert "inverse volatility" in first and "capped at 5% per name" in first
    text = " ".join(sentences)
    assert "market cap above $2bn (no small caps)" in text
    assert "25 bps" in text and "-30%" in text and "Benchmark: SPY" in text


def test_describe_quintile_long_only():
    first = describe_strategy(TEMPLATES["low_volatility"].spec())[0]
    assert "buy the top 20% (quintile)" in first and "short" not in first
    assert "60-day volatility (lower is better)" in first


def test_describe_screen():
    sentences = describe_strategy(TEMPLATES["dislocation_screen"].spec())
    assert sentences[0].startswith("Each month, hold every one of the US stocks that passes all 9 conditions")
    assert "sits in cash" in sentences[0]
    conds = [s for s in sentences if s.startswith("Condition ")]
    assert len(conds) == 9
    assert any("market cap between $2bn and $20bn (mid caps)" in c for c in conds)
    assert any("14-day RSI below 40 (oversold)" in c for c in conds)


def test_describe_factor_model():
    text = " ".join(describe_strategy(TEMPLATES["ff3"].spec()))
    assert "Build the Fama-French 3-factor model" in text
    for f in ("Mkt-RF", "SMB", "HML"):
        assert f in text
    assert "Kenneth French" in text
    assert "bps" not in text  # costs are irrelevant for a factor build
    assert "RMW" in " ".join(describe_strategy(TEMPLATES["ff5"].spec()))


def test_describe_time_series_cash_and_short():
    text = " ".join(describe_strategy(TEMPLATES["trend_200dma_spy"].spec()))
    assert text.startswith("At each month-end, hold SPY (long) while price vs its 200-day moving average above 0%")
    assert "move to cash" in text and "Benchmark: SPY" in text
    spec = StrategySpec(name="ts", idea="ts", kind="time_series", rebalance="daily",
                        time_series=TimeSeriesRule(assets=["QQQ", "IWM"], entry=[Condition(feature="rsi_14", op="<", value=30)],
                                                   exit=[Condition(feature="rsi_14", op=">", value=70)], when_flat="short"))
    text = " ".join(describe_strategy(spec))
    assert "At every close, hold QQQ, IWM (long) while 14-day RSI below 30; otherwise go short." in text
    assert "Exit early when 14-day RSI above 70." in text
    assert "independently" in text


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_has_a_readable_description(key):
    sentences = describe_strategy(TEMPLATES[key].spec())
    assert sentences and len(sentences[0]) > 30
    assert "_" not in sentences[0].split(":")[0]  # no raw snake_case in the core rule
    title, desc, refs = template_info(key)
    assert title and desc and isinstance(refs, list)


def test_describe_strategy_with_unreadable_spec():
    assert "could not be fully read" in describe_strategy({"kind": "cross_sectional", "rebalance": "monthly", "costs_bps": 5})[0]
    assert describe_strategy(None) == ["No strategy specification was saved with this result."]
    assert template_info("no-such-template") is None


# ------------------------------------------------------------------------------------------------
# Backtest report
# ------------------------------------------------------------------------------------------------

BACKTEST_SECTIONS = ["warnings", "idea", "key-numbers", "verdict", "growth", "drawdown", "quantiles", "factors",
                     "factor-checks", "stats", "holdings", "data", "llm-audit"]


def test_backtest_report_has_every_section():
    res = make_result()
    t = TEMPLATES["momentum_12_1"]
    page = render_backtest_html(res, template_title=t.title, template_description=t.description, references=t.references)
    assert_well_formed(page)
    for anchor in BACKTEST_SECTIONS:
        assert f'id="{anchor}"' in page, anchor
        assert f'href="#{anchor}"' in page, anchor
    assert DISCLAIMER in page
    assert "The idea in plain English" in page and t.description in page
    assert "How the backtest trades it" in page and "buy the top 10% (decile)" in page
    assert "Jegadeesh &amp; Titman (1993)" in page
    # header: idea, strategy name, period, provider, LLM
    assert "12-1 momentum, long the top decile" in page
    assert "momentum_12_1" in page and "synthetic" in page and "heuristic" in page
    assert f"{res.start.isoformat()} to {res.end.isoformat()}" in page
    # verdict badge + interpretation parts
    assert "Promising" in page and "Key findings" in page and "Biases and caveats" in page and "Next experiments" in page
    # charts: growth (with long / short legs), drawdown, quantiles, factor betas, sparkline
    assert "Growth of $1" in page and "Long leg" in page and "Short leg" in page
    assert "Max drawdown" in page and "Annual return by signal bucket" in page and "Factor betas" in page
    assert len(_svgs(page)) >= 5
    # tables: regression, factor checks, stats for every series, data usage, LLM audit
    for text in ("Alpha (annual)", "Correlation with official", "Information ratio", "Point-in-time", "claude-test", "1,200"):
        assert text in page, text
    assert page.count("<th scope=\"col\" class=\"num\">Long leg</th>") >= 1
    assert "Calendar-year returns" in page


def test_cited_metrics_are_marked_verified_or_not():
    res = make_result()
    page = render_backtest_html(res)
    section = page[page.index('id="verdict"'):page.index('id="growth"')]
    assert "stats.strategy.sharpe" in section and "regression.alpha_t_stat" in section
    assert 'class="mark ok"' in section  # correct Sharpe citation
    assert 'class="mark bad"' in section  # wrong alpha t-stat citation
    assert "1 of 2 cited numbers verified" in section


def test_cited_metric_checks_passed_explicitly_and_from_warnings():
    res = make_result()
    checks = [EvidenceCheck(kind="quant", ref="stats.strategy.sharpe", claim="x", status="mismatch", detail="nope"),
              EvidenceCheck(kind="quant", ref="regression.alpha_t_stat", claim="y", status="verified", detail="ok")]
    section = render_backtest_html(res, metric_checks=checks)
    assert "1 of 2 cited numbers verified" in section
    res2 = make_result(warnings=["Citation stats.strategy.sharpe unverified: value mismatch"])
    page2 = render_backtest_html(res2, metric_checks=[EvidenceCheck(kind="quant", ref="stats.strategy.sharpe", claim="x",
                                                                    status="verified")])
    assert "0 of 2 cited numbers verified" in page2


def test_malicious_text_is_escaped_everywhere():
    spec = TEMPLATES["momentum_12_1"].spec()
    spec.name = EVIL
    spec.idea = EVIL
    spec.assumptions = [EVIL]
    spec.unsupported_requests = [EVIL]
    interp = BacktestInterpretation(summary=EVIL, verdict="weak", key_findings=[EVIL], biases_and_caveats=[EVIL],
                                    next_experiments=[EVIL], cited_metrics=[CitedMetric(path=EVIL, value=1.0, meaning=EVIL)])
    res = make_result(idea=EVIL, spec=spec, warnings=[EVIL, f"survivorship {EVIL}"], interpretation=interp,
                      holdings={EVIL: 0.5, "OK": -0.5}, extra_series={EVIL: [0.01] * 96}, run_id=EVIL)
    res.data_usage.append(DataUsage(dataset=EVIL, source=EVIL, coverage=EVIL, point_in_time=False, notes=EVIL))
    res.llm_calls.append(LLMCallRecord(purpose=EVIL, model=EVIL, error=EVIL, request_id=EVIL, stop_reason=EVIL))
    res.factor_checks.append(FactorConstructionCheck(factor=EVIL, correlation_with_official=None, annual_premium_constructed_pct=None,
                                                     annual_premium_official_pct=None, n_overlap_periods=0))
    res.regression.betas[EVIL] = 0.1
    page = render_backtest_html(res, template_title=EVIL, template_description=EVIL, references=[EVIL], home_href='"><script>x</script>')
    assert EVIL not in page
    assert "<script" not in page.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert_well_formed(page)


def test_warnings_are_prominent_and_severe_first():
    res = make_result(warnings=["minor note", "Survivorship bias: today's universe", "Look-ahead risk in fundamentals"])
    page = render_backtest_html(res)
    callout = page[page.index('id="warnings"') - 60:page.index('id="idea"')]
    assert 'class="callout bad"' in callout
    assert callout.index("Survivorship") < callout.index("minor note")
    assert callout.count('<li class="bad">') == 2
    assert page.index('id="warnings"') < page.index('id="key-numbers"')
    no_warn = render_backtest_html(make_result(warnings=[]))
    assert 'id="warnings"' not in no_warn


def test_holdings_show_top_25_per_side_by_abs_weight():
    holdings = {f"L{i:02d}": 0.01 * (i + 1) for i in range(40)}
    holdings.update({f"S{i:02d}": -0.005 * (i + 1) for i in range(30)})
    page = render_backtest_html(make_result(holdings=holdings))
    sec = page[page.index('id="holdings"'):page.index('id="data"')]
    assert "Top long positions (40)" in sec and "Top short positions (30)" in sec
    assert "+ 15 more" in sec and "+ 5 more" in sec
    assert sec.index(">L39<") < sec.index(">L38<")  # largest first
    assert ">L00<" not in sec  # 15 smallest longs are cut
    assert ">S29<" in sec and ">S00<" not in sec


def test_growth_chart_switches_to_log_scale_for_large_ranges():
    res = make_result(mu=0.06, n=96)  # ~ x100 growth
    assert "log scale" in render_backtest_html(res)
    assert "log scale" not in render_backtest_html(make_result(mu=0.002))


def test_minimal_and_malformed_results_still_render():
    res = BacktestResult(run_id="r", idea="tiny", spec={}, provider="p", llm="none", start=date(2020, 1, 1), end=date(2020, 3, 31),
                         rebalance="monthly", returns={}, dates=[], stats={}, started_at=datetime(2026, 1, 1))
    page = render_backtest_html(res)
    assert_well_formed(page)
    assert "No strategy specification" in page and "No interpretation was saved" in page and "No return series" in page
    assert "No LLM calls were made" in page
    # series longer than dates, None values, garbage spec
    dates = _month_ends(6)
    res2 = BacktestResult(run_id="r2", idea="odd", spec={"kind": "weird"}, provider="p", llm="none", start=dates[0], end=dates[-1],
                          rebalance="monthly", returns={"strategy": [0.01, None, 0.02, -0.01, None, 0.03, 0.5, 0.7]}, dates=dates,
                          stats={}, started_at=datetime(2026, 1, 1))
    page2 = render_backtest_html(res2)
    assert_well_formed(page2)
    frame = returns_frame(res2)
    assert len(frame) == 6 and frame["strategy"].isna().sum() == 2


def test_factor_model_result_without_strategy_series():
    dates = _month_ends(24)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    rets = {f: list(_returns(24, 0.003, 0.02, i)) for i, f in enumerate(["Mkt-RF", "SMB", "HML"])}
    stats = {f: performance_stats(pd.Series(v, index=idx), label=f) for f, v in rets.items()}
    res = BacktestResult(run_id="ff", idea="Fama-French 3-factor model", spec=TEMPLATES["ff3"].spec().model_dump(mode="json"),
                         provider="synthetic", llm="none", start=dates[0], end=dates[-1], rebalance="monthly", returns=rets, dates=dates,
                         stats=stats, started_at=datetime(2026, 1, 1))
    page = render_backtest_html(res)
    assert_well_formed(page)
    assert "Mkt-RF" in page and "SMB" in page and "Build the Fama-French 3-factor model" in page


def test_fake_runner_results_render_for_every_kind():
    runner = FakeRunner()
    assert isinstance(runner, BacktestRunner)
    for key in ("momentum_12_1", "low_volatility", "dislocation_screen", "trend_200dma_spy", "ff3"):
        spec = TEMPLATES[key].spec()
        res = runner.backtest(spec, label=f"run-{key}")
        page = render_backtest_html(res, **dict(zip(("template_title", "template_description", "references"), template_info(key))))
        assert_well_formed(page)
        assert describe_strategy(spec)[0] in page.replace("&#x27;", "'")


# ------------------------------------------------------------------------------------------------
# Screening (pipeline) report
# ------------------------------------------------------------------------------------------------


def make_pipeline(observation: str = "Quality mid caps that sold off on a guidance cut") -> PipelineResult:
    spec = ScreenSpec(
        name="dislocation", observation=observation,
        conditions=[Condition(feature="drawdown_from_52w_high_pct", op="<", value=-20, rationale="sharp sell-off"),
                    Condition(feature="fcf_yield_pct", op=">", value=4)],
        any_of=[[Condition(feature="rsi_14", op="<", value=35), Condition(feature="rel_volume_5d", op=">", value=2)]],
        ranking=[RankFactor(feature="fcf_yield_pct", direction="higher_is_better", weight=2, rationale="cheap"),
                 RankFactor(feature="drawdown_from_52w_high_pct", direction="lower_is_better")],
        assumptions=["mid cap = 2-10bn"], unsupported_requests=["management tone"],
    )
    thesis = DislocationThesis(
        ticker="ACME", headline="Guidance reset over-discounted", dislocation_type="guidance_reset_overreaction",
        market_narrative="Growth is over", variant_view="Backlog intact", why_dislocation_exists="Forced selling",
        quant_evidence=[QuantEvidence(feature="fcf_yield_pct", value=7.5, interpretation="cheap"),
                        QuantEvidence(feature="rsi_14", value=22.0, interpretation="oversold")],
        narrative_evidence=[QuoteEvidence(doc_id="t1", speaker="CEO", quote="Backlog grew 20% year over year.", interpretation="demand"),
                            QuoteEvidence(doc_id="t1", speaker=None, quote="We invented this quote.", interpretation="fake")],
        catalysts=["Q3 print"], risks=["Macro"], invalidation_triggers=["Backlog falls"], conviction="medium", is_actionable=True,
        data_gaps=["Segment margins"],
    )
    grounding = GroundingReport(ticker="ACME", checks=[
        EvidenceCheck(kind="quant", ref="fcf_yield_pct", claim="fcf_yield_pct=7.5", status="verified"),
        EvidenceCheck(kind="quant", ref="rsi_14", claim="rsi_14=22.0", status="mismatch", detail="actual 31.2"),
        EvidenceCheck(kind="quote", ref="t1", claim="Backlog grew 20% year over year.", status="verified"),
        EvidenceCheck(kind="quote", ref="t1", claim="We invented this quote.", status="not_found"),
    ])
    cand = RankedCandidate(ticker="ACME", name="Acme Corp", rank=1, score=0.91, factor_scores={"fcf_yield_pct": 0.95},
                           features={"fcf_yield_pct": 7.5, "gics_sector": "Industrials", "rsi_14": None})
    cand2 = RankedCandidate(ticker="BETA", name="Beta Inc", rank=2, score=0.5)
    return PipelineResult(
        run_id="screen-1", observation=observation, as_of=date(2026, 9, 30), provider="synthetic", llm="heuristic",
        spec=spec.model_dump(mode="json"), universe_size=500, feature_coverage={"fcf_yield_pct": 0.92, "rsi_14": 0.3},
        funnel=[FunnelStep(label="sell-off", passed_alone=80, remaining=80, missing_data=3),
                FunnelStep(label="fcf yield", passed_alone=120, remaining=12, missing_data=10)],
        survivors=12,
        ideas=[InvestmentIdea(candidate=cand, thesis=thesis, grounding=grounding, documents_used=["t1"]),
               InvestmentIdea(candidate=cand2, error="LLM timeout")],
        llm_calls=[LLMCallRecord(purpose="explain:ACME", model="claude-test", input_tokens=10, output_tokens=5)],
        pushdown_query="SCREEN(...)", warnings=["synthetic data"], started_at=datetime(2026, 9, 30, 9, 0),
    )


def test_pipeline_report_sections_and_grounding_marks():
    page = render_pipeline_html(make_pipeline())
    assert_well_formed(page)
    for anchor in ("warnings", "spec", "coverage", "funnel", "candidates", "theses", "llm-audit"):
        assert f'id="{anchor}"' in page, anchor
    assert DISCLAIMER in page
    card = page[page.index('id="idea-ACME"'):page.index('id="idea-BETA"')]
    assert card.count('class="mark ok"') == 2 and card.count('class="mark bad"') == 2
    assert "Backlog grew 20% year over year." in card and "We invented this quote." in card
    assert "1/4" not in card and "2/4 evidence verified" in card
    assert "LLM timeout" in page
    assert "SCREEN(...)" in page and "management tone" in page
    assert "Names remaining after each condition" in page


def test_pipeline_report_escapes_and_handles_bad_spec():
    res = make_pipeline(observation=EVIL)
    res.spec = {"conditions": EVIL}
    res.ideas[0].thesis.headline = EVIL
    res.ideas[0].candidate.name = EVIL
    page = render_pipeline_html(res)
    assert EVIL not in page and "<script" not in page.lower()
    assert "could not be parsed" in page
    assert_well_formed(page)


# ------------------------------------------------------------------------------------------------
# Strategy page (paper trading)
# ------------------------------------------------------------------------------------------------


def make_paper(backtest_expected: float | None = 2.0) -> dict:
    nav = [("2023-09-29", 100_000.0), ("2023-10-31", 101_200.0), ("2023-11-30", 103_000.0), ("2023-12-29", 102_500.0)]
    return {
        "name": "mom", "status": "active", "started": "2023-09-29", "last_run": "2023-12-29", "next_rebalance": "2024-01-31",
        "initial_capital": 100_000.0, "cash": 1_500.0, "nav": 102_500.0, "since_start_return_pct": 2.5,
        "backtest_expected_return_pct": backtest_expected, "max_drawdown_pct": -0.49, "total_costs": 55.0, "n_trades": 3,
        "gross_exposure_pct": 98.5, "net_exposure_pct": 98.5,
        "nav_history": nav,
        "holdings": [
            {"ticker": "AAA", "shares": 100.0, "side": "long", "price": 300.0, "market_value": 30_000.0, "weight": 0.2927, "target_weight": 0.3},
            {"ticker": "BBB", "shares": 500.0, "side": "long", "price": 142.0, "market_value": 71_000.0, "weight": 0.6927, "target_weight": 0.7},
        ],
        "trades": [  # latest first, as PaperAccount.summary returns them
            {"date": "2023-11-30", "ticker": "AAA", "side": "sell", "shares": 10.0, "price": 290.0, "notional": 2900.0, "cost": 2.9, "reason": "rebalance"},
            {"date": "2023-09-29", "ticker": "BBB", "side": "buy", "shares": 500.0, "price": 130.0, "notional": 65000.0, "cost": 32.5, "reason": "initial"},
            {"date": "2023-09-29", "ticker": "AAA", "side": "buy", "shares": 110.0, "price": 300.0, "notional": 33000.0, "cost": 19.6, "reason": "initial"},
        ],
        "warnings": ["price for CCC is stale"],
    }


def test_strategy_page_shows_idea_holdings_trades_and_paper_vs_backtest():
    spec = TEMPLATES["momentum_12_1"].spec()
    bt = make_result(spec=spec)  # monthly 2016-01 .. 2023-12: covers the paper window
    page = render_strategy_page("Momentum 12-1", spec, bt, make_paper())
    assert_well_formed(page)
    for anchor in ("idea", "paper", "trades", "backtest", "growth", "verdict", "holdings", "warnings"):
        assert f'id="{anchor}"' in page, anchor
    assert page.count('id="warnings"') == 1
    assert "Momentum 12-1" in page and TEMPLATES["momentum_12_1"].description in page
    paper = page[page.index('id="paper"'):page.index('id="backtest"')]
    assert "$102,500.00" in paper and "+2.50%" in paper and "+2.00%" in paper and "+0.50 pp" in paper
    assert "Paper NAV vs backtest expectation" in paper and "Backtest (same dates)" in paper
    assert ">AAA<" in paper and ">BBB<" in paper and "$71,000.00" in paper and "Target weight" in paper
    assert paper.index(">BBB<") < paper.index(">AAA<")  # largest weight first
    blotter = paper[paper.index('id="trades"'):]
    dates = re.findall(r"<td>(\d{4}-\d{2}-\d{2})</td>", blotter)
    assert dates == sorted(dates, reverse=True) and dates[0] == "2023-11-30"
    assert "price for CCC is stale" in paper
    assert DISCLAIMER in page
    assert "T-bill" not in paper  # no comparison note in the summary: none shown


def test_strategy_page_shows_backtest_comparison_notes():
    spec = TEMPLATES["low_volatility"].spec()
    paper = make_paper()
    paper["backtest_comparison_notes"] = ["The backtest credits the 1-month T-bill rate on idle cash & short proceeds."]
    page = render_strategy_page("lowvol", spec, make_result(spec=spec), paper)
    assert_well_formed(page)
    section = page[page.index('id="paper"'):page.index('id="backtest"')]
    assert "The backtest credits the 1-month T-bill rate on idle cash &amp; short proceeds." in section


def test_strategy_page_expected_path_fallbacks_and_oldest_first_trades():
    spec = TEMPLATES["low_volatility"].spec()
    paper = make_paper()
    paper["nav_history"] = [("2026-08-31", 100_000.0), ("2026-09-30", 100_900.0)]
    paper["trades"] = list(reversed(paper["trades"]))  # oldest first
    bt = make_result(spec=spec)  # ends 2023-12: does not cover 2026
    page = render_strategy_page("lowvol", spec, bt, paper)
    assert "Backtest-expected (CAGR" in page
    blotter = page[page.index('id="trades"'):]
    dates = re.findall(r"<td>(\d{4}-\d{2}-\d{2})</td>", blotter)
    assert dates[0] == "2023-11-30"
    page2 = render_strategy_page("lowvol", spec, None, make_paper(backtest_expected=1.0))
    assert "Backtest-expected" in page2 and "No backtest result is saved" in page2


def test_strategy_page_without_paper_or_backtest_and_with_dict_spec():
    page = render_strategy_page("Nothing yet", TEMPLATES["trend_200dma_spy"].spec().model_dump(mode="json"), None, None)
    assert_well_formed(page)
    assert "not being paper-traded yet" in page and "No backtest result is saved" in page
    assert "hold SPY (long)" in page
    page2 = render_strategy_page(EVIL, {"kind": EVIL}, None, {"nav_history": [[EVIL, EVIL]], "holdings": [{"ticker": EVIL}],
                                                             "trades": [{"ticker": EVIL, "side": EVIL, "date": EVIL}], "started": EVIL})
    assert EVIL not in page2 and "<script" not in page2.lower()
    assert_well_formed(page2)


def test_strategy_page_with_real_paper_account(tmp_path):
    paper_mod = pytest.importorskip("aitrading.trading.paper")
    store_mod = pytest.importorskip("aitrading.trading.store")
    spec = TEMPLATES["momentum_12_1"].spec()
    runner = FakeRunner()
    bt = runner.backtest(spec, label="bt-real")
    store = store_mod.StrategyStore(tmp_path / "strategies")
    store.save("Momentum real", spec, bt)
    acct = paper_mod.PaperAccount(store, "Momentum real")
    try:
        acct.rebalance(runner, spec, date(2023, 10, 31))
        acct.rebalance(runner, spec, date(2023, 11, 30))
    except Exception as e:  # pragma: no cover - the paper module is developed independently
        pytest.skip(f"paper account API changed: {e}")
    summary = acct.summary(backtest=bt)
    page = render_strategy_page("Momentum real", spec, bt, summary)
    assert_well_formed(page)
    assert "Trade blotter (latest first)" in page and ">AAA<" in page
    # and the dashboard picks the strategy and its ledger up from the store directory
    index = write_dashboard(tmp_path / "site", runs_dir=tmp_path / "runs", strategies_dir=tmp_path / "strategies",
                            inbox_path=tmp_path / "ideas.json")
    html_index = index.read_text(encoding="utf-8")
    assert "Momentum real" in html_index and "not paper-trading" not in html_index


# ------------------------------------------------------------------------------------------------
# Ideas inbox
# ------------------------------------------------------------------------------------------------


def make_candidate(idea_id: str = "abc123", *, status: str = "new", score: float = 0.7, url: str = "https://arxiv.org/abs/1234.5678",
                   title: str = "Industry momentum", testability: str = "testable_now", replication: bool = False) -> IdeaCandidate:
    quotes = ["Industry momentum earns 0.5% per month.", "Returns reverse after 12 months."]
    rep = None
    if replication:
        rep = ReplicationReport(
            idea_id=idea_id, backtest_run_id="bt-9", base_sharpe=0.45, base_alpha_t_stat=2.1, claimed_sharpe=0.9, claimed_t_stat=4.0,
            replication_ratio=0.5,
            checks=[RobustnessCheck(name="first_half", description="2010-2017", sharpe=0.6, cagr_pct=4.0, alpha_t_stat=2.0, n_periods=96, passed=True),
                    RobustnessCheck(name="costs_25bps", description="higher costs", sharpe=0.1, cagr_pct=0.5, alpha_t_stat=0.4, n_periods=192, passed=False),
                    RobustnessCheck(name="post_publication", description="after 2015", sharpe=None, cagr_pct=None, alpha_t_stat=None, n_periods=0, passed=None)],
            verdict="partially_replicates", summary="Half the claimed Sharpe.", caveats=["Short sample"],
        )
    return IdeaCandidate(
        idea_id=idea_id,
        source=SourceDocument(source_type="arxiv", url=url, title=f"{title} paper", authors=["A. Author", "B. Author", "C. Author", "D. Author"],
                              published=date(2026, 9, 1), text="...", fetched_at=datetime(2026, 9, 2), source_name="arXiv q-fin.PM"),
        extraction=IdeaExtraction(
            is_trading_idea=True, title=title, summary="Industries that went up keep going up.", claimed_effect="0.5% per month",
            signal_description="6-month industry return", asset_class="us_equities", holding_period="1 month", reported_sharpe=0.9,
            reported_annual_return_pct=6.0, reported_t_stat=4.0, sample_period="1963-2019", evidence_quotes=quotes,
            data_requirements=["prices", "industry codes"], testability=testability, missing_data=["GICS history"],
            proposed_strategy_idea="Long-short quintiles on 6-month industry momentum", closest_library_template="momentum_12_1",
            credibility_notes=["Peer reviewed (JF)"],
        ),
        quote_checks=[EvidenceCheck(kind="quote", ref=url, claim=quotes[0], status="verified", detail="verbatim match"),
                      EvidenceCheck(kind="quote", ref=url, claim=quotes[1], status="not_found", detail="no such passage")],
        score=score, status=status, discovered_at=datetime(2026, 9, 2, 8, 0), replication=rep,
    )


def test_ideas_inbox_cards():
    cands = [make_candidate("low", score=0.2, title="Low score"), make_candidate("hi", score=0.9, title="High score", replication=True),
             make_candidate("rej", status="rejected", score=0.99, title="Rejected one", testability="needs_institutional_data")]
    page = render_ideas_inbox(cands)
    assert_well_formed(page)
    assert page.index("High score") < page.index("Low score") < page.index("Rejected one")  # status, then score
    for c in cands:
        assert f'id="idea-{c.idea_id}"' in page
        assert f"aitrading ideas try {c.idea_id}" in page
    assert '<a href="https://arxiv.org/abs/1234.5678" rel="noopener noreferrer" target="_blank">' in page
    assert "Testable now" in page and "Needs institutional data" in page and "GICS history" in page
    assert "1/2 verified verbatim" in page and 'class="mark ok"' in page and 'class="mark bad"' in page
    assert "Industry momentum earns 0.5% per month." in page
    assert "Partially replicates" in page and "Robustness checks" in page and "costs_25bps" in page
    assert "et al." in page and "published 2026-09-01" in page
    assert "new: 2" in page and "rejected: 1" in page


def test_ideas_inbox_escapes_and_refuses_unsafe_links():
    c = make_candidate(EVIL, url="javascript:alert(1)", title=EVIL)
    c.extraction.evidence_quotes = [EVIL]
    c.quote_checks = []
    page = render_ideas_inbox([c])
    assert EVIL not in page and "<script" not in page.lower()
    assert 'href="javascript' not in page
    assert_well_formed(page)
    empty = render_ideas_inbox([])
    assert "inbox is empty" in empty
    assert_well_formed(empty)


# ------------------------------------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _hrefs(page: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', page)


def test_dashboard_with_all_artefacts_and_corrupt_files(tmp_path):
    runs = tmp_path / "runs"
    good = make_result(run_id="run/with:odd*chars", idea="Momentum idea", finished=datetime(2026, 9, 1, tzinfo=timezone.utc))
    newer = make_result(run_id="run-2", idea="Low vol idea", spec=TEMPLATES["low_volatility"].spec(),
                        finished=datetime(2026, 9, 20, tzinfo=timezone.utc))
    _write(runs / "a" / "result.json", good.model_dump_json())
    _write(runs / "b" / "result.json", newer.model_dump_json())
    _write(runs / "c" / "result.json", "{not json")
    _write(runs / "d" / "result.json", json.dumps({"hello": "world"}))
    _write(runs / "e" / "result.json", "[1, 2, 3]")
    _write(runs / "f" / "result.json", make_pipeline().model_dump_json())
    _write(runs / "g" / "result.json", make_result(run_id="CON", idea="Reserved name").model_dump_json())

    strategies = tmp_path / "strategies"
    spec = TEMPLATES["momentum_12_1"].spec()
    _write(strategies / "momentum" / "spec.json", spec.model_dump_json())
    _write(strategies / "momentum" / "backtest.json", make_result(run_id="strat-bt", spec=spec).model_dump_json())
    _write(strategies / "momentum" / "meta.json", json.dumps({"name": "Momentum 12-1", "updated": "2026-09-30T10:00:00"}))
    _write(strategies / "broken" / "spec.json", "{{{")
    _write(strategies / "badledger" / "spec.json", TEMPLATES["low_volatility"].spec().model_dump_json())
    _write(strategies / "badledger" / "ledger.json", "corrupt")
    _write(strategies / "badbt" / "spec.json", json.dumps({"name": "x", "kind": "nonsense"}))
    _write(strategies / "badbt" / "backtest.json", json.dumps({"run_id": 1}))

    inbox = tmp_path / "ideas.json"
    valid = [make_candidate(f"id{i}", score=i / 10, title=f"Idea {i}").model_dump(mode="json") for i in range(7)]
    _write(inbox, json.dumps({"schema_version": 1, "ideas": valid + [{"idea_id": "broken"}], "quarantined": [{"x": 1}]}))

    out = tmp_path / "site"
    index = write_dashboard(out, runs_dir=runs, strategies_dir=strategies, inbox_path=inbox)
    assert index == out / "index.html" and index.is_file()
    page = index.read_text(encoding="utf-8")
    assert_well_formed(page)

    # every link is relative and resolves inside the output folder
    links = [h for h in _hrefs(page) if not h.startswith("#")]
    assert links
    for h in links:
        assert not h.startswith(("/", "file:", "http")) and ":" not in h.split("#")[0] and "\\" not in h
        assert (out / h.split("#")[0]).is_file(), h
    # summary tables
    assert "Momentum 12-1" in page and "Momentum idea" in page and "Low vol idea" in page
    assert page.index("Low vol idea") < page.index("Momentum idea")  # newest backtest first
    assert "Quality mid caps that sold off" in page  # screening run
    assert "Promising" in page
    # idea inbox counts + top 5 new ideas by score
    assert "new: 7" in page
    sec = page[page.index('id="ideas"'):]
    assert sec.index("Idea 6") < sec.index("Idea 5") and "Idea 1" not in sec.split("Open the full idea inbox")[0]
    assert "aitrading ideas try id6" in sec
    # skipped files are listed, nothing crashed
    skipped = page[page.index('id="skipped"'):]
    for fragment in ("c" + ("\\" if "\\" in str(runs) else "/") + "result.json", "not a backtest or screening result",
                     "Skipped strategy broken", "paper ledger could not be read", "ignored unreadable",
                     "skipped 1 invalid idea entry", "1 quarantined entry"):
        assert fragment in skipped, fragment
    # sub pages exist and link back home
    bt_pages = sorted((out / "backtests").glob("*.html"))
    assert len(bt_pages) == 4  # three runs + the strategy's backtest
    assert "CON.html" not in [p.name for p in bt_pages] and "_CON.html" in [p.name for p in bt_pages]
    for p in bt_pages + list((out / "strategies").glob("*.html")) + list((out / "screens").glob("*.html")):
        sub = p.read_text(encoding="utf-8")
        assert 'href="../index.html"' in sub
        assert_well_formed(sub)
    assert 'href="index.html"' in (out / "ideas.html").read_text(encoding="utf-8")
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+\.html", p.name) for p in bt_pages)


def test_dashboard_with_nothing_saved(tmp_path):
    index = write_dashboard(tmp_path / "out", runs_dir=tmp_path / "nope", strategies_dir=tmp_path / "nope2",
                            inbox_path=tmp_path / "nope.json")
    page = index.read_text(encoding="utf-8")
    assert_well_formed(page)
    assert "No saved strategies" in page and "No backtest results found" in page and "No idea inbox found" in page
    assert 'id="skipped"' not in page
    assert DISCLAIMER in page


def test_dashboard_defaults_use_aitrading_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AITRADING_HOME", str(tmp_path / "home"))
    _write(tmp_path / "home" / "runs" / "x" / "result.json", make_result(run_id="home-run", idea="From home").model_dump_json())
    index = write_dashboard(tmp_path / "out")
    assert "From home" in index.read_text(encoding="utf-8")


def test_dashboard_escapes_malicious_content_and_dedupes_names(tmp_path):
    runs = tmp_path / "runs"
    _write(runs / "1" / "result.json", make_result(run_id=EVIL, idea=EVIL).model_dump_json())
    _write(runs / "2" / "result.json", make_result(run_id=EVIL + " ", idea="second").model_dump_json())
    _write(runs / "3" / "backtest.json", make_result(run_id=EVIL, idea="duplicate run id").model_dump_json())
    inbox = tmp_path / "ideas.json"
    _write(inbox, json.dumps([make_candidate(EVIL, title=EVIL).model_dump(mode="json")]))
    out = tmp_path / "out"
    index = write_dashboard(out, runs_dir=runs, strategies_dir=tmp_path / "none", inbox_path=inbox)
    page = index.read_text(encoding="utf-8")
    assert EVIL not in page and "<script" not in page.lower()
    names = sorted(p.name for p in (out / "backtests").glob("*.html"))
    assert len(names) == 2 and len(set(n.lower() for n in names)) == 2  # duplicate run id rendered once
    for p in [*(out / "backtests").glob("*.html"), out / "ideas.html"]:
        text = p.read_text(encoding="utf-8")
        assert EVIL not in text and "<script" not in text.lower()


def test_open_in_browser_never_raises(tmp_path, monkeypatch):
    import webbrowser

    target = tmp_path / "index.html"
    target.write_text("<!DOCTYPE html>", encoding="utf-8")
    seen = []
    monkeypatch.setattr(webbrowser, "open", lambda uri: seen.append(uri) or True)
    assert open_in_browser(target) is True
    assert seen and seen[0].startswith("file://") and seen[0].endswith("index.html")

    def boom(uri):
        raise RuntimeError("no browser")

    monkeypatch.setattr(webbrowser, "open", boom)
    assert open_in_browser(target) is False


# ------------------------------------------------------------------------------------------------
# Regression tests (review findings on the html dashboard)
# ------------------------------------------------------------------------------------------------


def _paper_holdings_table(page: str) -> str:
    sec = page[page.index("<h3>Current holdings</h3>"):]
    return sec[:sec.index("</table>")]


def test_paper_weights_above_150pct_of_nav_are_still_fractions():
    # a short that ran against the account: |weight| > 1.5 must not flip the table into "already percent" mode
    paper = {"name": "x", "nav": 40_000.0, "cash": 180_000.0,
             "holdings": [{"ticker": "SPY", "shares": -200.0, "price": 700.0, "market_value": -140_000.0,
                           "weight": -3.5, "target_weight": -1.0},
                          {"ticker": "QQQ", "shares": 10.0, "price": 500.0, "market_value": 5_000.0,
                           "weight": 0.125, "target_weight": 0.1}]}
    page = render_strategy_page("x", None, None, paper)
    assert_well_formed(page)
    table = _paper_holdings_table(page)
    assert "-350.00%" in table and "-100.00%" in table
    assert "12.50%" in table and "10.00%" in table
    assert "-3.50%" not in table and "0.13%" not in table
    assert table.index(">SPY<") < table.index(">QQQ<")  # largest |weight| first
    # ordinary fractions (the PaperAccount.summary contract) are unchanged
    normal = _paper_holdings_table(render_strategy_page("m", None, None, make_paper()))
    assert "69.27%" in normal and "70.00%" in normal and "29.27%" in normal


def test_paper_weights_in_percent_only_via_explicit_pct_keys():
    paper = {"name": "x", "nav": 100_000.0, "cash": 0.0,
             "holdings": [{"ticker": "AAA", "shares": 1.0, "price": 1.0, "market_value": 60_000.0,
                           "weight_pct": 60.0, "target_weight_pct": 55.0}]}
    table = _paper_holdings_table(render_strategy_page("x", None, None, paper))
    assert "60.00%" in table and "55.00%" in table and "Target weight" in table


def _page_css(page: str) -> str:
    return page[page.index("<style>") + len("<style>"):page.index("</style>")]


def test_grid_columns_never_force_horizontal_scroll_on_phones():
    css = _page_css(render_backtest_html(make_result()))
    assert "minmax(min(300px, 100%), 1fr)" in css
    assert ".two-col > * { min-width: 0; }" in css
    # no grid track minimum wider than a phone column (~260px) unless it is capped by min(..., 100%)
    for m in re.finditer(r"minmax\((\d+)px", css):
        assert int(m.group(1)) <= 200, m.group(0)
    mobile = css[css.index("@media (max-width: 640px)"):]
    mobile = mobile[:mobile.index("}\n}") + 3]
    assert ".idea-card { padding: 12px; }" in mobile


class _GridChildren(HTMLParser):
    """Collects the tag (+ class) of every direct child of each ``div.two-col``."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.grids: list[tuple[int, list[str]]] = []  # (depth of the grid div, child tags)
        self.open: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _TagChecker.VOID:
            return
        self.depth += 1
        cls = dict(attrs).get("class") or ""
        if self.open and self.depth == self.grids[self.open[-1]][0] + 1:
            self.grids[self.open[-1]][1].append(f"{tag}.{cls}" if cls else tag)
        if tag == "div" and "two-col" in cls.split():
            self.grids.append((self.depth, []))
            self.open.append(len(self.grids) - 1)

    def handle_endtag(self, tag: str) -> None:
        if tag in _TagChecker.VOID:
            return
        if self.open and self.depth == self.grids[self.open[-1]][0]:
            self.open.pop()
        self.depth -= 1


def test_holdings_more_notes_stay_with_their_own_table():
    page = render_backtest_html(make_result())  # 40 longs, 30 shorts -> "+ 15 more" and "+ 5 more"
    sec = page[page.index('id="holdings"'):page.index('id="data"')]
    parser = _GridChildren()
    parser.feed(sec)
    assert len(parser.grids) == 1
    children = parser.grids[0][1]
    assert children == ["div", "div"], children  # one grid cell per side, not table / note / table / note
    longs, shorts = sec.split("Top short positions")
    assert "+ 15 more" in longs and "+ 5 more" in shorts
    # a long-only book still yields a single cell
    only_long = render_backtest_html(make_result(holdings={f"L{i}": 0.01 for i in range(30)}))
    parser = _GridChildren()
    parser.feed(only_long[only_long.index('id="holdings"'):only_long.index('id="data"')])
    assert parser.grids[0][1] == ["div"]


def _growth_svg(page: str) -> ET.Element:
    sec = page[page.index('id="growth"'):]
    return ET.fromstring(_svgs(sec)[0])


def test_growth_chart_title_matches_the_scale_actually_drawn():
    # a -100% period takes wealth to 0: line_chart must draw a linear axis, so the title cannot say "log scale"
    res = make_result(n=24)
    res.returns["strategy"] = [0.2] * 10 + [-1.0] + [0.0] * 13
    root = _growth_svg(render_backtest_html(res))
    title, desc = root.find(f"{SVG_NS}title").text, root.find(f"{SVG_NS}desc").text
    assert "log scale" not in title and desc.startswith("Linear")
    # and whenever the title says log scale, the chart really is log-scaled
    for r in (make_result(mu=0.06, n=96), make_result(mu=0.002), res):
        root = _growth_svg(render_backtest_html(r))
        title, desc = root.find(f"{SVG_NS}title").text, root.find(f"{SVG_NS}desc").text
        assert ("log scale" in title) == desc.startswith("Log-scale"), (title, desc[:20])


def _bridged_gaps(original: pd.Series, ds: pd.Series) -> int:
    """Consecutive kept finite points that have an original NaN between them (= a line across missing data)."""
    pos = original.index.get_indexer(ds.index)
    nan = original.isna().to_numpy()
    dv = ds.to_numpy(dtype=float)
    return sum(1 for a, b, va, vb in zip(pos[:-1], pos[1:], dv[:-1], dv[1:])
               if np.isfinite(va) and np.isfinite(vb) and nan[a + 1:b].any())


def test_downsample_never_draws_across_any_nan_gap():
    s = pd.Series(np.linspace(1.0, 2.0, 5000), index=pd.bdate_range("2000-01-03", periods=5000))
    s.iloc[100::5] = np.nan  # many separate gaps per bucket
    ds = svg.downsample(s, 800)
    assert len(ds) <= 800
    assert _bridged_gaps(s, ds) == 0
    assert ds.index[0] == s.index[0] and ds.index[-1] == s.index[-1]
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = int(rng.integers(900, 6000))
        mp = int(rng.choice([5, 8, 60, 400, 800]))
        x = _daily(n, seed=trial).to_numpy().copy()
        if trial % 2:
            x[rng.random(n) < 0.15] = np.nan
        else:
            for _ in range(int(rng.integers(1, 40))):
                a = int(rng.integers(0, n))
                x[a:a + int(rng.integers(1, 30))] = np.nan
        t = pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n))
        d = svg.downsample(t, mp)
        assert len(d) <= mp, (trial, len(d), mp)
        assert _bridged_gaps(t, d) == 0, trial
        assert d.index.is_monotonic_increasing
        if mp >= 8:  # extremes survive alongside the gap markers
            assert d.max() == t.max() and d.min() == t.min()


def test_line_chart_with_many_gaps_has_no_segment_across_missing_data():
    s = pd.Series(np.linspace(1.0, 2.0, 5000), index=pd.bdate_range("2000-01-03", periods=5000))
    s.iloc[100::5] = np.nan  # after the first 100 points, every finite run is 4 trading days long
    root = ET.fromstring(svg.line_chart({"x": s}, title="gappy"))
    d = [p.get("d") for p in root.iter(f"{SVG_NS}path") if p.get("fill") == "none"][0]
    segs = [[float(v) for v in re.findall(r"[-\d.]+", seg)[0::2]] for seg in d.split("M") if seg.strip()]
    x_first, x_last = segs[0][0], segs[-1][-1]
    px_per_day = (x_last - x_first) / (s.index[-1] - s.index[0]).days
    # beyond the gap-free head, no drawn segment may span more than one 4-day run (+ a weekend)
    widest = max((seg[-1] - seg[0]) / px_per_day for seg in segs[1:])
    assert widest <= 7.5, widest


def _tooltips(out: str) -> list[str]:
    return re.findall(r'class="hz"[^>]*><title>([^<]*)</title>', out)


def test_hover_tooltip_says_na_outside_a_series_date_span():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    late = pd.Series([10.0], index=[pd.Timestamp("2020-04-30")])
    tips = _tooltips(svg.line_chart({"A": a, "B": late}, title="t"))
    assert tips == ["2020-01-31 | A: 1 | B: n/a", "2020-02-29 | A: 2 | B: n/a",
                    "2020-03-31 | A: 3 | B: n/a", "2020-04-30 | A: 4 | B: 10"]
    # a series that ends early is n/a afterwards; inside its span the nearest value is still shown
    early = pd.Series([5.0, 6.0], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    tips = _tooltips(svg.line_chart({"A": a, "E": early}, title="t"))
    assert tips[0].endswith("E: 5") and tips[1].endswith("E: 6")
    assert tips[2].endswith("E: n/a") and tips[3].endswith("E: n/a")
    # mixed frequency: a monthly series is still shown at the daily dates inside its span
    daily = pd.Series(np.arange(40, dtype=float), index=pd.bdate_range("2020-02-03", periods=40))
    monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    tips = _tooltips(svg.line_chart({"D": daily, "M": monthly}, title="t"))
    assert all("M: n/a" not in tip for tip in tips if tip[:10] <= "2020-03-31")


def _hex_lum(h: str) -> float:
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_hex_lum(a), _hex_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_chart_tick_text_meets_wcag_aa_contrast_in_both_themes():
    css = _page_css(render_backtest_html(make_result()))
    light = css[css.index(":root {"):css.index("@media (prefers-color-scheme: dark)")]
    dark = css[css.index(':root[data-theme="dark"]'):]
    for block in (light, dark):
        muted = re.search(r"--chart-muted: (#[0-9a-f]{6})", block).group(1)
        surface = re.search(r"--chart-surface: (#[0-9a-f]{6})", block).group(1)
        assert _contrast(muted, surface) >= 4.5, (muted, surface)
    # the standalone SVG fallback (no page tokens) is readable on the light surface too
    out = svg.line_chart({"a": _daily(50)}, title="t", y_label="Value")
    fallback = re.search(r"var\(--chart-muted, (#[0-9a-f]{6})\)", out).group(1)
    assert _contrast(fallback, "#fcfcfb") >= 4.5
