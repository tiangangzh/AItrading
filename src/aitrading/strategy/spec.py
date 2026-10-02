"""Typed strategy specification for the idea lab (idea -> data -> strategy -> backtest -> result).

An LLM (or the offline heuristic translator) turns a research idea such as "Fama-French 3-factor
model", "12-1 momentum, long the top decile", "quality at a reasonable price" or "long SPY when it
is above its 200-day" into a ``StrategySpec``. The deterministic runner then gathers point-in-time
data, builds the signal at each rebalance date, forms portfolios and backtests them.

Kinds
-----
* ``cross_sectional`` - rank stocks on a composite signal (``signal``) at each rebalance and hold
  quantile / top-N portfolios (long-only or long-short).
* ``screen`` - hold the names passing ``filters`` (a screen) at each rebalance, equal/value weighted.
* ``factor_model`` - build an academic factor model (``factor_model``) from the universe with the
  standard Fama-French sort methodology, compare it with the official Kenneth French factors, and
  report factor premia. Any other strategy can also be regressed on a factor model via
  ``attribution_model``.
* ``time_series`` - per-asset timing rule (``time_series``): long when the entry conditions hold
  (evaluated on each asset's own technical features), otherwise flat (or short).

All feature names come from the feature catalog; anything the catalog cannot express goes into
``unsupported_requests`` instead of being approximated silently.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from aitrading.screen.spec import Condition, UniverseSpec

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.catalog import FeatureCatalog

StrategyKind = Literal["cross_sectional", "screen", "factor_model", "time_series"]
Rebalance = Literal["daily", "weekly", "monthly", "quarterly", "annual"]
FactorModelName = Literal["capm", "ff3", "carhart4", "ff5"]


class SignalComponent(BaseModel):
    feature: str = Field(description="Catalog feature used as (part of) the ranking signal.")
    direction: Literal["higher_is_better", "lower_is_better"]
    weight: float = Field(1.0, gt=0, description="Relative weight in the composite; normalised to sum to 1.")
    transform: Literal["rank", "zscore"] = Field("rank", description="Cross-sectional normalisation before combining.")
    sector_neutral: bool = Field(False, description="Normalise within GICS sector instead of across the whole universe.")


class PortfolioConstruction(BaseModel):
    style: Literal["long_only", "long_short"] = "long_short"
    selection: Literal["quantile", "top_n"] = "quantile"
    n_quantiles: int = Field(5, ge=2, le=20, description="Number of signal buckets (5 = quintiles, 10 = deciles).")
    top_n: int | None = Field(None, ge=1, le=500, description="Number of names per side when selection='top_n'.")
    weighting: Literal["equal", "value", "signal", "inverse_vol"] = "equal"
    max_weight: float | None = Field(None, gt=0, le=1, description="Cap per name (long-only / per side), e.g. 0.05.")


class TimeSeriesRule(BaseModel):
    assets: list[str] = Field(description="Tickers the rule trades independently (e.g. ['SPY']).")
    entry: list[Condition] = Field(description="ANDed conditions on the asset's own features; long when all hold.")
    exit: list[Condition] = Field(default_factory=list, description="Optional ANDed exit conditions; if empty, exit when entry stops holding.")
    when_flat: Literal["cash", "short"] = "cash"


class StrategySpec(BaseModel):
    name: str = Field(description="Short slug-like name.")
    idea: str = Field(description="The research idea, verbatim.")
    kind: StrategyKind
    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    start: date | None = Field(None, description="Backtest start; None = as far back as the data allows (default ~10y).")
    end: date | None = Field(None, description="Backtest end; None = latest available.")
    rebalance: Rebalance = "monthly"
    signal: list[SignalComponent] = Field(default_factory=list, description="cross_sectional: composite ranking signal.")
    filters: list[Condition] = Field(default_factory=list, description="Eligibility filters applied at each rebalance (the screen itself for kind='screen').")
    portfolio: PortfolioConstruction = Field(default_factory=PortfolioConstruction)
    factor_model: FactorModelName | None = Field(None, description="factor_model: which model to build.")
    time_series: TimeSeriesRule | None = None
    attribution_model: FactorModelName | None = Field("ff3", description="Factor model used to attribute the strategy's returns (alpha/betas).")
    costs_bps: float = Field(10.0, ge=0, description="One-way transaction cost in basis points, charged on turnover.")
    delisting_return: float = Field(
        0.0,
        gt=-1,
        description="Return booked when a held name stops trading for good (terminal delisting). 0 is optimistic; "
        "e.g. -0.3 approximates performance delistings (Shumway 1997).",
    )
    benchmark: str | None = Field(None, description="Benchmark ticker; None = the provider's default broad US index.")
    assumptions: list[str] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)

    def features(self) -> set[str]:
        out = {c.feature for c in self.signal}
        for c in self.filters:
            out |= c.features()
        if self.time_series:
            for c in [*self.time_series.entry, *self.time_series.exit]:
                out |= c.features()
        return out

    def validate_against(self, catalog: "FeatureCatalog") -> list[str]:
        errs: list[str] = []
        for name in sorted(self.features()):
            if name not in catalog:
                errs.append(f"unknown feature '{name}'")
        for c in self.filters + ([*self.time_series.entry, *self.time_series.exit] if self.time_series else []):
            errs.extend(c.structural_errors())
        for s in self.signal:
            if s.feature in catalog and catalog[s.feature].dtype == "category":
                errs.append(f"cannot rank on category feature '{s.feature}'")
        if self.kind == "cross_sectional" and not self.signal:
            errs.append("cross_sectional strategies need at least one signal component")
        if self.kind == "screen" and not self.filters:
            errs.append("screen strategies need at least one filter condition")
        if self.kind == "factor_model" and self.factor_model is None:
            errs.append("factor_model strategies need factor_model set (capm, ff3, carhart4 or ff5)")
        if self.kind == "time_series":
            if self.time_series is None or not self.time_series.assets or not self.time_series.entry:
                errs.append("time_series strategies need time_series.assets and time_series.entry")
        if self.portfolio.selection == "top_n" and not self.portfolio.top_n:
            errs.append("selection='top_n' needs portfolio.top_n")
        if self.start and self.end and self.start >= self.end:
            errs.append("start must be before end")
        return errs
