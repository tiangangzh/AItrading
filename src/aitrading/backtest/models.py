"""Backtest result models (serialisable, so runs can be saved, diffed and reported)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from aitrading.core.models import LLMCallRecord


class PerformanceStats(BaseModel):
    """Annualised statistics of a periodic return series (periods_per_year inferred from frequency)."""

    label: str
    start: date
    end: date
    n_periods: int
    periods_per_year: float
    total_return_pct: float
    cagr_pct: float
    volatility_pct: float
    sharpe: float | None = Field(description="Annualised, excess of the risk-free rate when available.")
    sortino: float | None
    max_drawdown_pct: float = Field(description="Most negative peak-to-trough decline, <= 0.")
    max_drawdown_duration_periods: int
    calmar: float | None
    hit_rate_pct: float = Field(description="Share of periods with a positive return.")
    best_period_pct: float
    worst_period_pct: float
    skew: float | None
    excess_kurtosis: float | None
    mean_return_t_stat: float | None = Field(description="t-stat of the mean periodic return (Newey-West).")
    avg_turnover_pct: float | None = Field(None, description="Average one-way turnover per rebalance, % of book.")
    beta_to_benchmark: float | None = None
    tracking_error_pct: float | None = None
    information_ratio: float | None = None


class FactorRegression(BaseModel):
    model: str = Field(description='"capm", "ff3", "carhart4" or "ff5".')
    factor_source: str = Field(description="Where the factor returns came from (e.g. 'Kenneth French Data Library', 'constructed from universe').")
    n: int
    alpha_annual_pct: float
    alpha_t_stat: float = Field(description="Newey-West t-stat.")
    betas: dict[str, float]
    beta_t_stats: dict[str, float]
    r_squared: float


class QuantileAnalysis(BaseModel):
    n_quantiles: int
    annual_return_by_quantile_pct: list[float] = Field(description="Bucket 1 = worst signal ... bucket n = best signal.")
    spread_annual_pct: float = Field(description="Best minus worst bucket, annualised.")
    monotonicity: float = Field(description="Spearman correlation between bucket number and bucket return, [-1, 1].")
    ic_mean: float = Field(description="Mean cross-sectional rank IC of signal vs next-period return.")
    ic_t_stat: float
    ic_hit_rate_pct: float


class FactorConstructionCheck(BaseModel):
    factor: str
    correlation_with_official: float | None
    annual_premium_constructed_pct: float | None
    annual_premium_official_pct: float | None
    n_overlap_periods: int


class DataUsage(BaseModel):
    dataset: str = Field(description='e.g. "prices", "fundamentals", "ff_factors_official", "risk_free".')
    source: str
    coverage: str = Field(description="Human-readable coverage, e.g. '148/150 tickers, 2015-01 to 2026-09'.")
    point_in_time: bool
    notes: str = ""


class BacktestInterpretation(BaseModel):
    """LLM (or heuristic) reading of the results; every number cited is verified against the stats."""

    summary: str
    verdict: Literal["robust", "promising", "weak", "likely_spurious", "inconclusive"]
    key_findings: list[str]
    cited_metrics: list["CitedMetric"]
    biases_and_caveats: list[str]
    next_experiments: list[str]


class CitedMetric(BaseModel):
    path: str = Field(description="Dotted path into the result, e.g. 'stats.strategy.sharpe' or 'regression.alpha_t_stat'.")
    value: float
    meaning: str


BacktestInterpretation.model_rebuild()


class BacktestResult(BaseModel):
    run_id: str
    idea: str
    spec: dict = Field(description="StrategySpec as JSON.")
    provider: str
    llm: str
    start: date
    end: date
    rebalance: str
    returns: dict[str, list[float | None]] = Field(description="Periodic returns (fractions) keyed by series: strategy, benchmark, long, short, factors...")
    dates: list[date] = Field(description="Period end dates aligned with every series in 'returns'.")
    stats: dict[str, PerformanceStats] = Field(description="Keyed by series name: strategy, benchmark, long, short, ...")
    regression: FactorRegression | None = None
    quantiles: QuantileAnalysis | None = None
    factor_checks: list[FactorConstructionCheck] = Field(default_factory=list)
    latest_holdings: dict[str, float] = Field(default_factory=dict, description="Most recent portfolio weights (ticker -> weight).")
    data_usage: list[DataUsage] = Field(default_factory=list)
    interpretation: BacktestInterpretation | None = None
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
