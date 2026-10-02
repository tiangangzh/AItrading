"""StrategySpec -> point-in-time signals -> portfolios -> BacktestResult (the idea lab's engine).

:class:`StrategyRunner` implements :class:`~aitrading.backtest.protocols.BacktestRunner`. It reuses
the platform's building blocks: ``FeatureEngine`` for every feature at every rebalance date, the
screen engine for universe filters and conditions, ``build_quantile_weights`` / ``simulate`` for
portfolios and trading, ``performance_stats`` / ``factor_regression`` / ``quantile_analysis`` for the
evidence, and ``construct_factors`` / ``compare_with_official`` for factor models.

Window
------
``end`` = ``spec.end`` or the provider's latest trading day (the synthetic provider's ``end``;
otherwise min(today, the benchmark's last price date)), capped at the last date with prices.
``start`` = ``spec.start`` or ``end - default_years``. Prices are loaded once from ``start - 420``
calendar days (the feature look-back) to ``end`` and served to the feature engine through
:class:`~aitrading.backtest.panel.PreloadedProvider`. A warning is added when the data starts after
the requested start. Leading rebalance dates at which no portfolio can be formed (not enough
history for the features yet) are dropped with a warning, so the track record does not open with
months of artificial cash; later empty portfolios (a screen nobody passes) are held as cash.

Universe (survivorship)
-----------------------
The universe is ``provider.get_universe(spec.universe, end)`` - the constituents as of the END
date - used for every date back to ``start``. Unless the provider declares point-in-time
membership (attribute ``point_in_time_universe = True``) the result carries a prominent
SURVIVORSHIP BIAS warning and the ``universe`` data-usage note says so (companies delisted or
acquired before ``end`` are missing; this caps the interpreter's verdict at "promising").
Names without a valid price on a rebalance date (not listed yet, halted) are not eligible then.

Market cap at each date is point-in-time: from ``provider.get_universe(spec.universe, t)`` for
providers whose universe snapshot is cheap and point-in-time (the synthetic provider, or any
provider with ``point_in_time_market_cap = True``); otherwise it is estimated as the END market cap
times the adjusted-price ratio P(t) / P(end) (share issuance / buybacks after t and dividends are
not reflected; the data-usage note says so). It feeds ``market_cap_usd_bn``, value weights and
every valuation ratio (book-to-market, FCF yield, ...).

Per-date portfolio (one code path for the backtest and ``target_portfolio``)
--------------------------------------------------------------------------
At each rebalance date t: ``FeatureEngine.build(universe_t, t, needed)`` with ``needed`` =
the spec's features + universe-filter inputs + ``price`` + ``market_cap_usd_bn`` (value weights) +
``volatility_60d_pct`` (inverse-vol weights) + ``gics_sector`` (sector-neutral signals); then
``apply_universe(spec.universe)``, a valid price, and every ``spec.filters`` condition
(missing data excludes).

* ``cross_sectional`` - composite signal (:func:`composite_signal`): each component is transformed
  across the eligible names (``rank`` -> percentile in [0, 1] with average ranks for ties;
  ``zscore`` -> (x - mean) / std winsorised at +/-3), within GICS sector when ``sector_neutral``
  (names without a sector get no score for that component), with the direction applied
  (lower_is_better negates), then averaged with the component weights renormalised over the
  components a name has data for. Weights: ``build_quantile_weights(signal, **portfolio)``.
* ``screen`` - every eligible name, long-only, equal / value / inverse-vol weighted (``signal``
  weighting falls back to equal: a screen has no signal), capped at ``max_weight``.
* ``time_series`` - per asset, on its own features (a single-asset universe; no universe
  filters): long when every entry condition (and every ``spec.filters`` condition) holds; with
  exit conditions the position is kept until every exit condition holds, without them it is
  closed when the entry stops holding. Weight +1/n long, -1/n when flat and ``when_flat="short"``,
  else 0. The state is replayed date by date (``target_portfolio`` replays from the window start).
* ``factor_model`` - see below.

Execution and returns
---------------------
``simulate(target_weights, close, costs_bps, execution_lag, start, end, delisting_return)``: signal
at the close of t, traded ``execution_lag`` sessions later, weights drift, costs on traded notional.
Long-short books (cross_sectional ``long_short``) are self-financing: collateral earns 0, Sharpe is
on the raw spread and the factor regression uses ``excess=False``. Every other book (long-only,
screen, time series) earns the daily risk-free rate on its idle cash (1 - sum of the executed
weights, clipped to [0, 1], drift ignored), Sharpe is in excess of RF and the regression uses
``excess=True``. Long-short runs also report the ``long`` leg (positive weights, simulated with the
same costs) and the ``short`` leg = the return of the shorted basket (absolute negative weights,
before costs), so long - short is the spread before the short leg's trading costs.

Statistics are computed on DAILY returns (RF = official daily 1-month T-bill rate when the factor
loader can provide it, else 0 with a warning; benchmark = ``spec.benchmark`` or the provider
default). The result stores MONTHLY compounded returns (calendar month-end labels) to keep files
small. Factor regression: monthly strategy returns on the official ``spec.attribution_model``
factors (complete months only; skipped with a warning below 24 months or without factor data).
Quantiles (cross_sectional with at least 2 x n_quantiles names): ``quantile_analysis`` of the stored
signals against ``forward_returns_from_close`` with the same execution lag.

Factor models (kind ``factor_model``)
-------------------------------------
A :class:`~aitrading.factors.construct.CharacteristicsPanel` on calendar month-ends: returns from
month-end closes, point-in-time market caps (above), book equity = ``total_equity`` of the latest
public filing as of each June formation date (the point-in-time lag), ff5 operating profitability /
investment = ``operating_profitability_pct`` / ``asset_growth_yoy_pct`` / 100 from the feature engine
at the June formation date, carhart4 momentum = P(m-1) / P(m-12) - 1 from month-end closes (the
monthly-granularity ``return_12m_ex_1m_pct``), NYSE flags from the universe ``exchange``, RF = the
official monthly RF. ``construct_factors(panel, model, formation="annual_june")`` builds the
factors, ``compare_with_official`` gives ``factor_checks``. ``returns`` holds every factor plus
``strategy`` = the equal-weighted average of the model's non-market long-short factors (SMB / HML /
... ; the constructed Mkt-RF for capm) so reports and the interpreter have a headline series;
statistics are per factor (raw, they are already excess / self-financing returns) and for
``strategy`` / ``benchmark``. ``target_portfolio`` / ``latest_holdings`` give the factor-mimicking
weights of that headline portfolio: the 2x3 value-weighted portfolios formed at the latest June
(momentum: at as_of) with today's market-cap weights inside each portfolio.

Run ids are deterministic: ``<spec name>-<sha256 of spec JSON + provider + window + execution lag>``
plus ``-<label>`` when a label is given. ``runner.last_run`` keeps the internals of the latest
backtest (:class:`RunDetails`: target weights and signals per date, the simulation).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Hashable

import numpy as np
import pandas as pd

from aitrading.backtest.engine import SimulationResult, build_quantile_weights, rebalance_dates, simulate
from aitrading.backtest.metrics import compound, performance_stats
from aitrading.backtest.models import (
    BacktestResult,
    DataUsage,
    FactorConstructionCheck,
    FactorRegression,
    PerformanceStats,
    QuantileAnalysis,
)
from aitrading.backtest.panel import PreloadedProvider, slice_panel
from aitrading.backtest.quantiles import forward_returns_from_close, quantile_analysis
from aitrading.backtest.regression import FACTOR_COLUMNS, MIN_REGRESSION_OBS, factor_regression
from aitrading.core import fields as F
from aitrading.data.base import PricePanel
from aitrading.factors.construct import CharacteristicsPanel, compare_with_official, construct_factors, two_by_three_sort
from aitrading.factors.french import FRENCH_SOURCE, align_factor_dates, load_french_factors
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.engine import apply_universe, evaluate_condition
from aitrading.screen.features import FEATURE_DATASETS, LOOKBACK_CALENDAR_DAYS, FeatureEngine
from aitrading.strategy.spec import SignalComponent, StrategySpec

__all__ = [
    "StrategyRunner",
    "RunDetails",
    "composite_signal",
    "PRELOAD_LOOKBACK_DAYS",
    "REBALANCES_PER_YEAR",
]

#: Calendar days of history loaded before ``start`` (the feature engine's look-back).
PRELOAD_LOOKBACK_DAYS = LOOKBACK_CALENDAR_DAYS
REBALANCES_PER_YEAR: dict[str, float] = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0, "annual": 1.0}
_SNAPSHOT_DATASETS = ("estimates", "short_interest", "options")
_MARKET = {"capm": "Mkt-RF"}
_MAX_WARNINGS = 80

FactorLoader = Callable[[str, str], pd.DataFrame]


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


def _as_date(x: Any) -> date:
    if isinstance(x, str):
        x = pd.Timestamp(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date()
    return x


def _slug(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return (s or "run")[:max_len].strip("-") or "run"


def _fmt(d: Any) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for w in items:
        if w and w not in seen:
            seen[w] = None
    return list(seen)


def _names(names: list[str], limit: int = 8) -> str:
    names = [str(n) for n in names]
    return ", ".join(names[:limit]) + (f", +{len(names) - limit} more" if len(names) > limit else "")


def _rank_pct(x: pd.Series) -> pd.Series:
    n = len(x)
    if n == 0:
        return x.astype(float)
    if n == 1:
        return pd.Series(0.5, index=x.index)
    return (x.rank(method="average") - 1.0) / (n - 1.0)


def _zscore(x: pd.Series) -> pd.Series:
    n = len(x)
    if n == 0:
        return x.astype(float)
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if not math.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=x.index)
    return ((x - float(x.mean())) / sd).clip(-3.0, 3.0)


def _transform(x: pd.Series, how: str) -> pd.Series:
    """Cross-sectional transform of the finite values of ``x`` (NaN elsewhere)."""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    v = x[np.isfinite(x.to_numpy(dtype=float))]
    if len(v):
        out.loc[v.index] = (_rank_pct(v) if how == "rank" else _zscore(v)).to_numpy(dtype=float)
    return out


def composite_signal(
    frame: pd.DataFrame, components: list[SignalComponent], *, sector_column: str = "gics_sector"
) -> pd.Series:
    """Composite cross-sectional signal (higher = better) of ``frame``'s rows.

    Per component: the feature (non-finite -> missing), negated when ``lower_is_better``, then
    transformed across the rows (``rank`` -> percentile in [0, 1], average ranks for ties, 0.5 for
    a single name; ``zscore`` -> (x - mean) / std(ddof=1) clipped at +/-3, 0 when the dispersion is
    zero), within each ``sector_column`` group when ``sector_neutral`` (rows without a sector get no
    score for that component). The composite is the weighted mean of the components a row has a
    score for (weights renormalised over those); NaN when it has none.
    """
    num = np.zeros(len(frame))
    den = np.zeros(len(frame))
    for c in components:
        if c.feature not in frame.columns:
            continue
        x = pd.to_numeric(frame[c.feature], errors="coerce").astype(float)
        x = x.where(np.isfinite(x.to_numpy()))
        if c.direction == "lower_is_better":
            x = -x
        if c.sector_neutral:
            sectors = frame[sector_column] if sector_column in frame.columns else pd.Series(None, index=frame.index)
            s = pd.Series(np.nan, index=frame.index, dtype=float)
            labels = sectors.astype(object).where(sectors.notna(), None)
            for sec in pd.unique(labels.dropna()):
                rows = labels == sec
                s.loc[rows] = _transform(x[rows], c.transform).to_numpy()
        else:
            s = _transform(x, c.transform)
        ok = np.isfinite(s.to_numpy())
        num[ok] += float(c.weight) * s.to_numpy()[ok]
        den[ok] += float(c.weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.Series(out, index=frame.index, name="signal")


# ------------------------------------------------------------------------------------------------
# Run state
# ------------------------------------------------------------------------------------------------


@dataclass
class RunDetails:
    """Internals of the latest backtest (``StrategyRunner.last_run``), for diagnostics and tests."""

    spec: StrategySpec
    start: date
    end: date
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    target_weights: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    signals: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    simulation: SimulationResult | None = None
    daily_returns: pd.Series | None = None
    factor_returns: pd.DataFrame | None = None


@dataclass
class _DateOutcome:
    weights: pd.Series
    signal: pd.Series | None
    formable: bool
    coverage: dict[str, float]
    warnings: list[str]
    n_universe: int = 0
    n_eligible: int = 0


@dataclass
class _Context:
    spec: StrategySpec
    start: date
    end: date
    preload_start: date
    universe: pd.DataFrame
    tickers: list[str]
    prices: PricePanel
    wrapped: PreloadedProvider
    engine: FeatureEngine
    needed: set[str]
    mcap_point_in_time: bool
    universe_spec: Any
    warnings: list[str] = field(default_factory=list)
    benchmark: pd.Series | None = None
    benchmark_label: str = ""
    data_start: date | None = None
    _mcap_cache: dict = field(default_factory=dict)
    _close_ffill: pd.DataFrame | None = None
    _mcap_scale: pd.Series | None = None

    @property
    def close(self) -> pd.DataFrame:
        return self.prices.close


# ------------------------------------------------------------------------------------------------
# The runner
# ------------------------------------------------------------------------------------------------


class StrategyRunner:
    """Runs a :class:`StrategySpec` point-in-time on a ``MarketDataProvider`` (see module docstring).

    Args:
        provider: the data source.
        catalog: feature catalog (default: the platform catalog).
        factor_loader: ``callable(model, frequency) -> DataFrame`` of official factor returns
            (fractions, ``RF`` included), default :func:`~aitrading.factors.french.load_french_factors`.
            ``None`` disables official factors. A failure becomes a warning; attribution / factor
            checks / RF are then skipped (RF = 0).
        default_years: window length when ``spec.start`` is None.
        execution_lag: sessions between the signal close and the trade (1 = next close).
        progress: receives one human-readable line per step.
        today: the clock for "latest available" (default ``date.today()``).
    """

    def __init__(
        self,
        provider: Any,
        *,
        catalog: FeatureCatalog | None = None,
        factor_loader: FactorLoader | None = load_french_factors,
        default_years: int = 10,
        execution_lag: int = 1,
        progress: Callable[[str], None] | None = None,
        today: date | None = None,
    ) -> None:
        if default_years < 1:
            raise ValueError("default_years must be >= 1")
        if execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        self.provider = provider
        self.catalog = catalog or default_catalog()
        self.factor_loader = factor_loader
        self.default_years = int(default_years)
        self.execution_lag = int(execution_lag)
        self.progress = progress
        self.today = today
        self.last_run: RunDetails | None = None
        self._memo: OrderedDict[Hashable, Any] = OrderedDict()
        self._factor_cache: dict[tuple[str, str], tuple[pd.DataFrame | None, str | None]] = {}
        self._price_store: list[tuple[date, date, PricePanel]] = []
        self._data_end: date | None = None

    # ------------------------------------------------------------------ protocol surface
    @property
    def provider_name(self) -> str:
        return str(getattr(self.provider, "name", type(self.provider).__name__))

    def _say(self, msg: str) -> None:
        if self.progress is not None:
            self.progress(msg)

    def _today(self) -> date:
        return self.today or date.today()

    # ------------------------------------------------------------------ official factors
    def _official(self, model: str, frequency: str) -> tuple[pd.DataFrame | None, str | None]:
        """(factors, error) from the factor loader, cached per (model, frequency)."""
        key = (model, frequency)
        if key not in self._factor_cache:
            if self.factor_loader is None:
                self._factor_cache[key] = (None, "official factor data disabled (no factor loader)")
            else:
                try:
                    df = self.factor_loader(model, frequency)
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        raise ValueError("the factor loader returned no data")
                    df = df.copy()
                    df.index = pd.DatetimeIndex(df.index)
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    df.attrs = dict(getattr(df, "attrs", {}) or {})
                    self._factor_cache[key] = (df, None)
                except Exception as exc:  # noqa: BLE001 - official data is optional
                    self._factor_cache[key] = (None, f"{type(exc).__name__}: {exc}")
        return self._factor_cache[key]

    def _daily_rf(self, warnings: list[str]) -> tuple[pd.Series | None, str]:
        df, err = self._official("ff3", "daily")
        if df is None or "RF" not in df.columns:
            why = err or "no RF column in the daily factor data"
            warnings.append(f"risk-free rate unavailable ({why}): Sharpe ratios use rf = 0 and idle cash earns 0")
            return None, why
        for w in df.attrs.get("warnings", []) or []:
            warnings.append(f"official factors: {w}")
        return df["RF"].astype(float).dropna(), str(df.attrs.get("source", FRENCH_SOURCE))

    # ------------------------------------------------------------------ window
    def _latest_data_date(self) -> date:
        if self._data_end is not None:
            return self._data_end
        prov_end = getattr(self.provider, "end", None)
        if isinstance(prov_end, date):
            self._data_end = _as_date(prov_end)
            return self._data_end
        today = self._today()
        end = today
        try:
            probe = self.provider.get_benchmark_history(today - timedelta(days=14), today)
            if probe is not None and len(probe.dropna()):
                end = min(today, pd.Timestamp(probe.dropna().index[-1]).date())
        except Exception:  # noqa: BLE001 - fall back to today
            end = today
        self._data_end = end
        return end

    def _window(self, spec: StrategySpec, warnings: list[str]) -> tuple[date, date]:
        latest = self._latest_data_date()
        if spec.end is None:
            end = latest
        else:
            end = _as_date(spec.end)
            if end > latest:
                warnings.append(f"requested end {end} is after the latest available data ({latest}); the backtest ends on {latest}")
                end = latest
        if spec.start is not None:
            start = _as_date(spec.start)
        else:
            start = (pd.Timestamp(end) - pd.DateOffset(years=self.default_years)).date()
        if start >= end:
            raise ValueError(f"backtest window is empty: start {start} is not before end {end}")
        return start, end

    # ------------------------------------------------------------------ data loading
    def _load_prices(self, tickers: list[str], start: date, end: date) -> PricePanel:
        for s0, e0, panel in self._price_store:
            if s0 <= start and end <= e0 and all(t in panel.close.columns for t in tickers):
                return slice_panel(panel, tickers, start, end)
        self._say(f"Loading prices for {len(tickers)} ticker(s), {start} to {end}")
        panel = self.provider.get_price_history(list(tickers), start, end)
        self._price_store.insert(0, (start, end, panel))
        del self._price_store[3:]
        return slice_panel(panel, tickers, start, end)

    def _needed_features(self, spec: StrategySpec) -> set[str]:
        needed = set(spec.features()) | {"price"}
        if spec.kind in ("cross_sectional", "screen"):
            u = spec.universe
            if u.min_avg_dollar_volume_usd_mn is not None:
                needed.add("avg_dollar_volume_20d_usd_mn")
            if u.exclude_sectors:
                needed.add("gics_sector")
            if spec.portfolio.weighting == "value":
                needed.add("market_cap_usd_bn")
            if spec.portfolio.weighting == "inverse_vol":
                needed.add("volatility_60d_pct")
            if any(c.sector_neutral for c in spec.signal):
                needed.add("gics_sector")
        return {f for f in needed if f in self.catalog}

    def _mcap_mode(self) -> bool:
        flag = getattr(self.provider, "point_in_time_market_cap", None)
        if flag is not None:
            return bool(flag)
        return self.provider_name == "synthetic"

    def _time_series_universe(self, spec: StrategySpec, end: date) -> pd.DataFrame:
        assets = list(dict.fromkeys(str(a) for a in spec.time_series.assets))  # type: ignore[union-attr]
        frame = pd.DataFrame(index=pd.Index(assets, name=F.TICKER), columns=F.UNIVERSE_COLUMNS, dtype=object)
        frame[F.NAME] = assets
        frame[F.MARKET_CAP] = np.nan
        technical_only = all(self.catalog[f].source == "technical" for f in spec.features() if f in self.catalog)
        if not technical_only:  # reference / fundamental conditions need the asset's universe row (market cap, sector)
            try:
                full = self.provider.get_universe(None, end)
                rows = full.reindex(assets)
                for c in F.UNIVERSE_COLUMNS:
                    if c in rows.columns:
                        frame[c] = rows[c].where(rows[c].notna(), frame[c])
            except Exception:  # noqa: BLE001 - reference data is optional for a timing rule
                pass
        frame[F.MARKET_CAP] = pd.to_numeric(frame[F.MARKET_CAP], errors="coerce").astype(float)
        return frame

    def _context(self, spec: StrategySpec, start: date, end: date, warnings: list[str], *, with_benchmark: bool = True) -> _Context:
        preload_start = start - timedelta(days=PRELOAD_LOOKBACK_DAYS)
        if spec.kind == "time_series":
            universe = self._time_series_universe(spec, end)
            universe_spec = None
        else:
            self._say(f"Loading the universe as of {end}")
            universe = self.provider.get_universe(spec.universe, end)
            if F.TICKER in universe.columns and universe.index.name != F.TICKER:
                universe = universe.set_index(F.TICKER)
            universe = universe[~universe.index.duplicated(keep="first")]
            universe.index = pd.Index([str(t) for t in universe.index], name=F.TICKER)
            universe_spec = spec.universe
            if universe.empty:
                raise ValueError(f"the provider returned an empty universe as of {end} for {spec.universe.model_dump_json()}")
        tickers = list(universe.index)
        prices = self._load_prices(tickers, preload_start, end)
        close = prices.close
        valid = close.notna().any(axis=1)
        if not valid.any():
            raise ValueError(f"no price data for {len(tickers)} ticker(s) between {preload_start} and {end}")
        last_px = pd.Timestamp(close.index[valid.to_numpy()][-1]).date()
        first_px = pd.Timestamp(close.index[valid.to_numpy()][0]).date()
        if last_px < end:
            end = last_px
        if first_px > start:
            warnings.append(
                f"price data starts on {first_px}, after the requested start {start}: the backtest can only start once the "
                f"features have enough history after {first_px}"
            )
        wrapped = PreloadedProvider(self.provider, prices, window=(preload_start, end), memo=self._memo)
        ctx = _Context(
            spec=spec, start=start, end=end, preload_start=preload_start, universe=universe, tickers=tickers, prices=prices,
            wrapped=wrapped, engine=FeatureEngine(wrapped, self.catalog), needed=self._needed_features(spec),
            mcap_point_in_time=self._mcap_mode(), universe_spec=universe_spec, warnings=warnings, data_start=first_px,
        )
        if with_benchmark:
            ctx.benchmark, ctx.benchmark_label = self._benchmark(spec, ctx)
        return ctx

    def _benchmark(self, spec: StrategySpec, ctx: _Context) -> tuple[pd.Series | None, str]:
        symbols: list[str | None] = [spec.benchmark] if spec.benchmark else []
        symbols.append(None)
        for sym in symbols:
            try:
                ser = ctx.wrapped.get_benchmark_history(ctx.preload_start, ctx.end, sym)
                ser = pd.Series(ser, dtype=float).dropna()
                ser = ser[ser > 0]
                if len(ser) >= 2:
                    label = sym or str(getattr(self.provider, "benchmark", None) or "provider default index")
                    if sym is None and spec.benchmark:
                        ctx.warnings.append(f"benchmark {spec.benchmark} unavailable; using the provider's default index ({label})")
                    return ser.sort_index(), label
            except Exception as exc:  # noqa: BLE001 - try the default next
                if sym is not None:
                    ctx.warnings.append(f"benchmark {sym} could not be loaded ({type(exc).__name__}: {exc})")
        ctx.warnings.append("no benchmark data: benchmark-relative statistics are not reported")
        return None, ""

    # ------------------------------------------------------------------ point-in-time market cap
    def _mcap_at(self, ctx: _Context, t: pd.Timestamp) -> pd.Series:
        key = pd.Timestamp(t)
        if key in ctx._mcap_cache:
            return ctx._mcap_cache[key]
        if F.MARKET_CAP not in ctx.universe.columns:
            out = pd.Series(np.nan, index=ctx.universe.index, dtype=float)
        elif ctx.mcap_point_in_time:
            snap = ctx.wrapped.get_universe(ctx.universe_spec, key.date())
            if F.TICKER in snap.columns and snap.index.name != F.TICKER:
                snap = snap.set_index(F.TICKER)
            snap = snap[~snap.index.duplicated(keep="last")]
            out = pd.to_numeric(snap[F.MARKET_CAP], errors="coerce").astype(float)
            out.index = pd.Index([str(i) for i in out.index])
            out = out.reindex(ctx.universe.index)
        else:
            if ctx._close_ffill is None:
                ff = ctx.close.reindex(columns=ctx.universe.index).ffill()
                ctx._close_ffill = ff
                last = ff.iloc[-1] if len(ff) else pd.Series(np.nan, index=ctx.universe.index)
                cap_end = pd.to_numeric(ctx.universe[F.MARKET_CAP], errors="coerce").astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ctx._mcap_scale = (cap_end / last.where(last > 0)).astype(float)
            ff = ctx._close_ffill
            pos = int(ff.index.searchsorted(key, side="right")) - 1
            row = ff.iloc[pos] if pos >= 0 else pd.Series(np.nan, index=ff.columns)
            out = (row * ctx._mcap_scale).astype(float)
        out = out.where(np.isfinite(out.to_numpy()) & (out > 0))
        ctx._mcap_cache[key] = out
        return out

    def _universe_at(self, ctx: _Context, t: pd.Timestamp) -> pd.DataFrame:
        uni = ctx.universe.copy()
        if ctx.spec.kind == "time_series" and not uni[F.MARKET_CAP].notna().any():
            return uni
        uni[F.MARKET_CAP] = self._mcap_at(ctx, t).reindex(uni.index).to_numpy()
        return uni

    # ------------------------------------------------------------------ per-date portfolio
    def _cross_section(self, ctx: _Context, t: pd.Timestamp) -> _DateOutcome:
        spec = ctx.spec
        uni = self._universe_at(ctx, t)
        ff = ctx.engine.build(uni, t.date(), features=ctx.needed)
        frame = ff.frame
        mask, _ = apply_universe(spec.universe, frame)
        px = pd.to_numeric(frame["price"], errors="coerce")
        mask &= px.notna() & (px > 0)
        for cond in spec.filters:
            mask &= evaluate_condition(cond, frame, catalog=self.catalog)
        elig = frame.loc[mask.to_numpy()]
        mcap = pd.to_numeric(elig["market_cap_usd_bn"], errors="coerce") if "market_cap_usd_bn" in elig else None
        vol = pd.to_numeric(elig["volatility_60d_pct"], errors="coerce") if "volatility_60d_pct" in elig else None
        p = spec.portfolio
        signal: pd.Series | None = None
        if spec.kind == "screen":
            weighting = p.weighting if p.weighting in ("equal", "value", "inverse_vol") else "equal"
            names = list(elig.index)
            if names:
                weights = build_quantile_weights(
                    pd.Series(0.0, index=names), n_quantiles=2, style="long_only", selection="top_n", top_n=len(names),
                    weighting=weighting, market_cap=mcap, volatility=vol, max_weight=p.max_weight,
                )
            else:
                weights = pd.Series(dtype=float, name="weight")
            filter_feats = [f for f in spec.features() if f in frame.columns]
            formable = all(ff.coverage.get(f, float(frame[f].notna().mean()) if len(frame) else 0.0) > 0 for f in filter_feats)
        else:
            sig = composite_signal(elig, spec.signal)
            signal = sig[np.isfinite(sig.to_numpy())]
            weights = build_quantile_weights(
                signal, n_quantiles=p.n_quantiles, style=p.style, selection=p.selection, top_n=p.top_n,
                weighting=p.weighting, market_cap=mcap if mcap is not None else pd.Series(dtype=float),
                volatility=vol if vol is not None else pd.Series(dtype=float), max_weight=p.max_weight,
            )
            formable = len(weights) > 0
        return _DateOutcome(weights=weights.astype(float), signal=signal, formable=formable, coverage=dict(ff.coverage),
                            warnings=list(ff.warnings), n_universe=len(frame), n_eligible=int(mask.sum()))

    def _time_series_flags(self, ctx: _Context, t: pd.Timestamp) -> tuple[dict[str, tuple[bool, bool, bool]], dict[str, float], list[str]]:
        """Per asset: (entry holds, exit holds, entry inputs available) at ``t``."""
        rule = ctx.spec.time_series
        assert rule is not None
        uni_all = self._universe_at(ctx, t)
        out: dict[str, tuple[bool, bool, bool]] = {}
        cov: dict[str, list[float]] = {}
        warns: list[str] = []
        entry_feats = sorted({f for c in rule.entry for f in c.features()})
        for asset in uni_all.index:
            ff = ctx.engine.build(uni_all.loc[[asset]], t.date(), features=ctx.needed)
            frame = ff.frame
            warns.extend(ff.warnings)
            for k, v in ff.coverage.items():
                cov.setdefault(k, []).append(v)
            entry = pd.Series(True, index=frame.index)
            for cond in [*rule.entry, *ctx.spec.filters]:
                entry &= evaluate_condition(cond, frame, catalog=self.catalog)
            if rule.exit:
                ex = pd.Series(True, index=frame.index)
                for cond in rule.exit:
                    ex &= evaluate_condition(cond, frame, catalog=self.catalog)
                exit_ok = bool(ex.iloc[0])
            else:
                exit_ok = False
            has = bool(frame[entry_feats].notna().all(axis=1).iloc[0]) if entry_feats else True
            px = pd.to_numeric(frame["price"], errors="coerce").iloc[0] if "price" in frame else np.nan
            has = has and bool(np.isfinite(px) and px > 0)
            out[str(asset)] = (bool(entry.iloc[0]), exit_ok, has)
        coverage = {k: float(np.mean(v)) for k, v in cov.items()}
        return out, coverage, warns

    @staticmethod
    def _ts_step(long: bool, entry_ok: bool, exit_ok: bool, has_exit: bool) -> bool:
        if not long:
            return entry_ok and not (has_exit and exit_ok)
        if has_exit:
            return not exit_ok
        return entry_ok

    def _ts_weights(self, ctx: _Context, state: dict[str, bool]) -> pd.Series:
        rule = ctx.spec.time_series
        assert rule is not None
        n = len(ctx.universe.index)
        w = {}
        for asset in ctx.universe.index:
            if state.get(str(asset), False):
                w[str(asset)] = 1.0 / n
            elif rule.when_flat == "short":
                w[str(asset)] = -1.0 / n
        return pd.Series(w, dtype=float, name="weight").sort_index()

    def _time_series_path(self, ctx: _Context, dates: list[pd.Timestamp]) -> tuple[dict[pd.Timestamp, pd.Series], list[bool], dict[str, list[float]], list[str]]:
        rule = ctx.spec.time_series
        assert rule is not None
        has_exit = bool(rule.exit)
        state = {str(a): False for a in ctx.universe.index}
        targets: dict[pd.Timestamp, pd.Series] = {}
        formable: list[bool] = []
        cov: dict[str, list[float]] = {}
        warns: list[str] = []
        for i, t in enumerate(dates):
            if i % 24 == 0:
                self._say(f"{ctx.spec.name}: signal {i + 1}/{len(dates)} ({_fmt(t)})")
            flags, coverage, w = self._time_series_flags(ctx, t)
            warns.extend(w)
            for k, v in coverage.items():
                cov.setdefault(k, []).append(v)
            for asset, (entry_ok, exit_ok, _has) in flags.items():
                state[asset] = self._ts_step(state[asset], entry_ok, exit_ok, has_exit)
            formable.append(any(h for _, _, h in flags.values()))
            targets[t] = self._ts_weights(ctx, state)
        return targets, formable, cov, warns

    # ------------------------------------------------------------------ backtest
    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        started = datetime.now(timezone.utc)
        errors = self._spec_errors(spec)
        if errors:
            raise ValueError("invalid strategy spec: " + "; ".join(errors))
        warnings: list[str] = []
        start, end = self._window(spec, warnings)
        n_provider_warnings = len(getattr(self.provider, "warnings", []) or [])
        self._say(f"Backtesting '{spec.name}' ({spec.kind}) from {start} to {end}")
        ctx = self._context(spec, start, end, warnings)
        survivorship = self._survivorship_warning(spec, ctx)
        rf_daily, rf_source = self._daily_rf(warnings)
        if spec.kind == "factor_model":
            result = self._factor_model_backtest(spec, ctx, rf_daily, rf_source, label, started)
        else:
            result = self._portfolio_backtest(spec, ctx, rf_daily, rf_source, label, started)
        prov_w = list(getattr(self.provider, "warnings", []) or [])[n_provider_warnings:]
        all_w = _dedupe(([survivorship] if survivorship else []) + ctx.warnings + [f"data: {w}" for w in prov_w])
        if len(all_w) > _MAX_WARNINGS:
            all_w = all_w[:_MAX_WARNINGS] + [f"... {len(all_w) - _MAX_WARNINGS} more warnings omitted"]
        result.warnings = all_w
        result.finished_at = datetime.now(timezone.utc)
        self._say(f"Finished '{spec.name}': run {result.run_id}")
        return result

    def _spec_errors(self, spec: StrategySpec) -> list[str]:
        from aitrading.strategy.nl import spec_errors  # local import: nl imports the template library

        return spec_errors(spec, self.catalog)

    def _survivorship_warning(self, spec: StrategySpec, ctx: _Context) -> str | None:
        if spec.kind == "time_series" or bool(getattr(self.provider, "point_in_time_universe", False)):
            return None
        return (
            f"SURVIVORSHIP BIAS: the universe is the provider's constituents as of {ctx.end} ({len(ctx.tickers)} names) "
            f"applied to every date back to {ctx.start}, not point-in-time membership. Companies that were delisted, "
            f"acquired or dropped before {ctx.end} are missing, which usually flatters backtested returns "
            "(point-in-time index membership needs institutional data)."
        )

    def _run_id(self, spec: StrategySpec, start: date, end: date, label: str | None) -> str:
        payload = json.dumps(
            {"spec": spec.model_dump(mode="json"), "provider": self.provider_name, "start": start.isoformat(),
             "end": end.isoformat(), "execution_lag": self.execution_lag},
            sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
        rid = f"{_slug(spec.name)}-{digest}"
        return f"{rid}-{_slug(label, 32)}" if label else rid

    def _signal_dates(self, ctx: _Context) -> list[pd.Timestamp]:
        dates = rebalance_dates(ctx.close.index, ctx.spec.rebalance, ctx.start, ctx.end)
        if not dates:
            raise ValueError(f"no {ctx.spec.rebalance} rebalance date between {ctx.start} and {ctx.end}")
        return dates

    def _portfolio_backtest(self, spec: StrategySpec, ctx: _Context, rf_daily: pd.Series | None, rf_source: str,
                            label: str | None, started: datetime) -> BacktestResult:
        dates = self._signal_dates(ctx)
        targets: dict[pd.Timestamp, pd.Series] = {}
        signals: dict[pd.Timestamp, pd.Series] = {}
        formable: list[bool] = []
        cov: dict[str, list[float]] = {}
        engine_w: list[str] = []
        n_eligible: list[int] = []
        if spec.kind == "time_series":
            targets, formable, cov, engine_w = self._time_series_path(ctx, dates)
        else:
            for i, t in enumerate(dates):
                if i % 12 == 0:
                    self._say(f"{spec.name}: rebalance {i + 1}/{len(dates)} ({_fmt(t)})")
                out = self._cross_section(ctx, t)
                targets[t] = out.weights
                if out.signal is not None:
                    signals[t] = out.signal
                formable.append(out.formable)
                engine_w.extend(out.warnings)
                n_eligible.append(out.n_eligible)
                for k, v in out.coverage.items():
                    cov.setdefault(k, []).append(v)
        ctx.warnings.extend(engine_w)
        self._coverage_warnings(spec, ctx, cov)

        # leading dates without enough history: the track record starts at the first portfolio
        if any(formable):
            k = formable.index(True)
            if k > 0:
                ctx.warnings.append(
                    f"no portfolio could be formed at the first {k} rebalance date(s) ({_fmt(dates[0])} to {_fmt(dates[k - 1])}): "
                    f"the features did not have enough history yet; the backtest starts on {_fmt(dates[k])}"
                )
                for t in dates[:k]:
                    targets.pop(t, None)
                    signals.pop(t, None)
                dates = dates[k:]
        else:
            ctx.warnings.append(
                "no portfolio could be formed at any rebalance date (no name had data for the signal / every filter); "
                "the strategy held cash throughout"
            )
        if spec.kind == "screen" and targets and all(len(w) == 0 for w in targets.values()):
            ctx.warnings.append("no name passed the screen at any rebalance date: the strategy held cash throughout")

        self._say(f"{spec.name}: simulating {len(targets)} rebalance(s)")
        try:
            sim = simulate(targets, ctx.close, costs_bps=spec.costs_bps, execution_lag=self.execution_lag,
                           start=ctx.start, end=ctx.end, delisting_return=spec.delisting_return)
        except ValueError as exc:
            raise ValueError(f"'{spec.name}' cannot be simulated from {ctx.start} to {ctx.end} "
                             f"({len(targets)} rebalance date(s)): {exc}") from exc
        ctx.warnings.extend(sim.warnings)
        long_short = spec.kind == "cross_sectional" and spec.portfolio.style == "long_short"
        strat = sim.daily_returns.rename("strategy")
        if not long_short:
            strat = strat + self._cash_credit(sim, rf_daily, strat.index)
        bench_daily = self._bench_returns(ctx, strat.index)

        series: dict[str, pd.Series] = {"strategy": strat}
        stats: dict[str, PerformanceStats] = {}
        self._add_stats(stats, "strategy", strat, None if long_short else rf_daily, bench_daily, sim.turnover, ctx.warnings)
        if bench_daily is not None:
            series["benchmark"] = bench_daily
            self._add_stats(stats, "benchmark", bench_daily, rf_daily, None, None, ctx.warnings)
        if long_short:
            long_t = {t: w[w > 0] for t, w in targets.items()}
            short_t = {t: -w[w < 0] for t, w in targets.items()}
            if any(len(w) for w in short_t.values()):
                try:
                    lsim = simulate(long_t, ctx.close, costs_bps=spec.costs_bps, execution_lag=self.execution_lag,
                                    start=ctx.start, end=ctx.end, delisting_return=spec.delisting_return)
                    ssim = simulate(short_t, ctx.close, costs_bps=0.0, execution_lag=self.execution_lag,
                                    start=ctx.start, end=ctx.end, delisting_return=spec.delisting_return)
                    series["long"] = lsim.daily_returns.rename("long")
                    series["short"] = ssim.daily_returns.rename("short")
                    self._add_stats(stats, "long", series["long"], rf_daily, bench_daily, lsim.turnover, ctx.warnings)
                    self._add_stats(stats, "short", series["short"], rf_daily, bench_daily, None, ctx.warnings)
                except ValueError as exc:
                    ctx.warnings.append(f"long / short legs not reported: {exc}")

        monthly = {k: compound(v, "M") for k, v in series.items()}
        regression = self._regression(spec, ctx, monthly["strategy"], strat, excess=not long_short)
        quantiles = self._quantiles(spec, ctx, signals, dates) if spec.kind == "cross_sectional" else None

        holdings = sim.latest_weights()
        run_start = dates[0].date() if dates else ctx.start
        data_usage = self._data_usage(spec, ctx, cov, rf_daily, rf_source, regression is not None)
        self.last_run = RunDetails(spec=spec, start=run_start, end=ctx.end, rebalance_dates=list(dates), target_weights=targets,
                                   signals=signals, simulation=sim, daily_returns=strat)
        out_dates, returns = self._monthly_table(monthly)
        return BacktestResult(
            run_id=self._run_id(spec, ctx.start, ctx.end, label),
            idea=spec.idea,
            spec=spec.model_dump(mode="json"),
            provider=self.provider_name,
            llm="none",
            start=max(run_start, ctx.start),
            end=ctx.end,
            rebalance=spec.rebalance,
            returns=returns,
            dates=out_dates,
            stats=stats,
            regression=regression,
            quantiles=quantiles,
            factor_checks=[],
            latest_holdings={str(k): float(v) for k, v in holdings.items() if float(v) != 0.0},
            data_usage=data_usage,
            warnings=[],
            started_at=started,
        )

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _cash_credit(sim: SimulationResult, rf_daily: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
        """Daily risk-free interest on idle cash: (1 - sum of the executed weights, clipped to [0, 1]) x RF,
        earned from the day after each execution (drift between executions ignored)."""
        if rf_daily is None or not len(index) or not sim.weights_history:
            return pd.Series(0.0, index=index)
        cash = pd.Series({pd.Timestamp(d): float(np.clip(1.0 - float(w.sum()), 0.0, 1.0)) for d, w in sim.weights_history.items()})
        cash = cash.sort_index().reindex(index.union(cash.index)).ffill().reindex(index).shift(1).fillna(0.0)
        rf = rf_daily.reindex(rf_daily.index.union(index)).ffill().reindex(index)
        if rf.isna().all():
            return pd.Series(0.0, index=index)
        rf = rf.bfill().fillna(0.0)
        return (cash * rf).astype(float)

    @staticmethod
    def _bench_returns(ctx: _Context, index: pd.DatetimeIndex) -> pd.Series | None:
        if ctx.benchmark is None or not len(index):
            return None
        b = ctx.benchmark
        r = (b / b.shift(1) - 1.0).dropna()
        r = r[(r.index >= index[0]) & (r.index <= index[-1])]
        if len(r) < 2:
            return None
        if pd.Timestamp(index[0]) in r.index:
            r.loc[pd.Timestamp(index[0])] = 0.0  # the book is built at that close: no benchmark exposure before it
        return r.rename("benchmark")

    @staticmethod
    def _add_stats(stats: dict[str, PerformanceStats], key: str, returns: pd.Series, rf: pd.Series | None,
                   bench: pd.Series | None, turnover: pd.Series | None, warnings: list[str],
                   periods_per_year: float | None = None) -> None:
        r = pd.Series(returns, dtype=float).dropna()
        if len(r) < 2:
            warnings.append(f"no statistics for '{key}': fewer than 2 returns")
            return
        try:
            stats[key] = performance_stats(r, label=key, rf=rf, benchmark=bench, turnover=turnover,
                                           periods_per_year=periods_per_year)
        except ValueError as exc:
            warnings.append(f"no statistics for '{key}': {exc}")

    @staticmethod
    def _monthly_table(monthly: dict[str, pd.Series]) -> tuple[list[date], dict[str, list[float | None]]]:
        idx = pd.DatetimeIndex([])
        for s in monthly.values():
            idx = idx.union(pd.DatetimeIndex(s.index))
        idx = idx.sort_values()
        out: dict[str, list[float | None]] = {}
        for k, s in monthly.items():
            vals = s.reindex(idx).to_numpy(dtype=float)
            out[k] = [float(v) if np.isfinite(v) else None for v in vals]
        return [d.date() for d in idx], out

    def _complete_months(self, ctx: _Context, daily: pd.Series) -> set[pd.Period]:
        """Calendar months fully covered by the daily return series (first to last session of the month)."""
        if not len(daily):
            return set()
        sessions = ctx.close.index[(ctx.close.index >= ctx.close.index[0])]
        per = sessions.to_period("M")
        first = pd.Series(sessions, index=sessions).groupby(per).min()
        last = pd.Series(sessions, index=sessions).groupby(per).max()
        d0, d1 = pd.Timestamp(daily.index[0]), pd.Timestamp(daily.index[-1])
        out = set()
        for p in first.index:
            # a month is complete when the series holds the return of every session in it; the first
            # session's return needs the previous close, i.e. the series must start before it
            if first[p] > d0 and last[p] <= d1:
                out.add(p)
        # the last month is only complete when the data reaches the end of that calendar month
        if len(sessions) and d1 == sessions[-1] and d1 < (d1 + pd.offsets.MonthEnd(0)) - pd.Timedelta(days=3):
            out.discard(d1.to_period("M"))
        return out

    def _regression(self, spec: StrategySpec, ctx: _Context, monthly: pd.Series, daily: pd.Series, *,
                    excess: bool) -> FactorRegression | None:
        model = spec.attribution_model
        if model is None:
            return None
        factors, err = self._official(model, "monthly")
        if factors is None:
            ctx.warnings.append(f"factor attribution ({model}) skipped: official factors unavailable ({err})")
            return None
        complete = self._complete_months(ctx, daily)
        y = monthly[np.array([pd.Timestamp(d).to_period("M") in complete for d in monthly.index], dtype=bool)]
        if len(y) < MIN_REGRESSION_OBS:
            ctx.warnings.append(
                f"factor attribution ({model}) skipped: only {len(y)} complete month(s) of returns (< {MIN_REGRESSION_OBS})"
            )
            return None
        try:
            aligned = align_factor_dates(factors, y.index)
            cols = FACTOR_COLUMNS[model] + (["RF"] if excess else [])
            data = aligned[[c for c in cols if c in aligned.columns]].dropna()
            source = str(factors.attrs.get("source", FRENCH_SOURCE))
            return factor_regression(y, data, model=model, factor_source=source, excess=excess)
        except (ValueError, KeyError) as exc:
            ctx.warnings.append(f"factor attribution ({model}) skipped: {exc}")
            return None

    def _quantiles(self, spec: StrategySpec, ctx: _Context, signals: dict[pd.Timestamp, pd.Series],
                   dates: list[pd.Timestamp]) -> QuantileAnalysis | None:
        nq = spec.portfolio.n_quantiles
        if not signals or max(len(s) for s in signals.values()) < 2 * nq:
            ctx.warnings.append(f"quantile analysis skipped: fewer than {2 * nq} names with a signal at every date")
            return None
        try:
            fwd = forward_returns_from_close(ctx.close, dates, self.execution_lag, delisting_return=spec.delisting_return)
            return quantile_analysis(signals, fwd, n_quantiles=nq, periods_per_year=REBALANCES_PER_YEAR[spec.rebalance])
        except ValueError as exc:
            ctx.warnings.append(f"quantile analysis skipped: {exc}")
            return None

    def _coverage_warnings(self, spec: StrategySpec, ctx: _Context, cov: dict[str, list[float]]) -> None:
        used = sorted(f for f in spec.features() if f in self.catalog and self.catalog[f].dtype != "category")
        snapshot_pit = self._snapshots_point_in_time(ctx)
        for f in used:
            vals = cov.get(f, [])
            mean = float(np.mean(vals)) if vals else 0.0
            snap = sorted(d for d in _SNAPSHOT_DATASETS if d in FEATURE_DATASETS.get(f, ()))
            if snap and not snapshot_pit:
                ctx.warnings.append(
                    f"feature {f} has no point-in-time history in the free edition (the {'/'.join(snap)} data is a "
                    f"current snapshot only): it is NaN at historical rebalance dates, so results exclude it - a signal or "
                    f"filter that needs it will be empty"
                )
            elif vals and max(vals) == 0.0:
                ctx.warnings.append(f"feature {f} had no data at any rebalance date; results exclude it")
            elif vals and mean < 0.5:
                ctx.warnings.append(f"feature {f} covered only {mean * 100:.0f}% of the universe on average across rebalance dates")

    def _snapshots_point_in_time(self, ctx: _Context) -> bool:
        is_stale = getattr(self.provider, "is_stale", None)
        if callable(is_stale):
            try:
                return not bool(is_stale(ctx.start))
            except Exception:  # noqa: BLE001
                return False
        return True

    def _data_usage(self, spec: StrategySpec, ctx: _Context, cov: dict[str, list[float]], rf_daily: pd.Series | None,
                    rf_source: str, used_factors: bool, *, factor_model_fund: bool = False) -> list[DataUsage]:
        name = self.provider_name
        close = ctx.close[(ctx.close.index >= pd.Timestamp(ctx.start)) & (ctx.close.index <= pd.Timestamp(ctx.end))]
        n_with = int(close.notna().any(axis=0).sum())
        first = close.index[0] if len(close) else pd.Timestamp(ctx.start)
        last = close.index[-1] if len(close) else pd.Timestamp(ctx.end)
        price_src = {"synthetic": "Synthetic market simulator (offline)", "free": "Yahoo Finance via yfinance"}.get(name, name)
        out = [DataUsage(dataset="prices", source=price_src,
                         coverage=f"{n_with}/{len(ctx.tickers)} tickers, {_fmt(first)} to {_fmt(last)}", point_in_time=True,
                         notes="split- and dividend-adjusted daily closes")]
        if spec.kind == "time_series":
            out.append(DataUsage(dataset="universe", source="strategy spec", coverage=", ".join(ctx.tickers), point_in_time=True,
                                 notes="the assets named by the timing rule"))
        else:
            pit = bool(getattr(self.provider, "point_in_time_universe", False))
            notes = ("point-in-time membership" if pit else
                     f"constituents as of {ctx.end} applied to every date (not point-in-time membership): survivorship bias")
            notes += ("; market caps point-in-time" if ctx.mcap_point_in_time else
                      "; market caps before the end date = end market cap x adjusted price ratio (share issuance, "
                      "buybacks and dividends not reflected)")
            out.append(DataUsage(dataset="universe", source=name, coverage=f"{len(ctx.tickers)} names as of {ctx.end}",
                                 point_in_time=True, notes=notes))
        feats = set(spec.features())
        fund_feats = sorted(f for f in feats if "fundamentals" in FEATURE_DATASETS.get(f, ()))
        if fund_feats or factor_model_fund:
            vals = [np.mean(cov[f]) for f in fund_feats if cov.get(f)]
            cover = f"{np.mean(vals) * 100:.0f}% average coverage of {', '.join(fund_feats)}" if vals else "book equity at each June formation"
            src = {"synthetic": "Synthetic filings (point-in-time on report date)",
                   "free": "SEC EDGAR XBRL companyfacts (point-in-time on filing date)"}.get(name, name)
            out.append(DataUsage(dataset="fundamentals", source=src, coverage=cover, point_in_time=True,
                                 notes="latest filing public on each rebalance date"))
        snapshot_pit = self._snapshots_point_in_time(ctx)
        for ds in _SNAPSHOT_DATASETS:
            used = sorted(f for f in feats if ds in FEATURE_DATASETS.get(f, ()))
            if not used:
                continue
            vals = [np.mean(cov[f]) for f in used if cov.get(f)]
            cover = f"{np.mean(vals) * 100:.0f}% average coverage of {', '.join(used)}" if vals else "none"
            out.append(DataUsage(
                dataset=ds, source=name, coverage=cover, point_in_time=snapshot_pit,
                notes="" if snapshot_pit else "current snapshot only: no history in the free edition (NaN at historical dates)",
            ))
        if ctx.benchmark is not None:
            b = ctx.benchmark
            out.append(DataUsage(dataset="benchmark", source=f"{name}: {ctx.benchmark_label}",
                                 coverage=f"{_fmt(b.index[0])} to {_fmt(b.index[-1])}", point_in_time=True))
        model = spec.factor_model if spec.kind == "factor_model" else spec.attribution_model
        if model is not None:
            f, err = self._official(model, "monthly")
            if f is not None:
                if spec.kind == "factor_model":
                    note = "construction checks" + (" and attribution" if used_factors else "")
                else:
                    note = "factor attribution" if used_factors else "loaded, but the attribution could not be run"
                out.append(DataUsage(dataset="ff_factors_official", source=str(f.attrs.get("source", FRENCH_SOURCE)),
                                     coverage=f"{model} monthly, {f.index[0]:%Y-%m} to {f.index[-1]:%Y-%m}", point_in_time=True,
                                     notes=note))
            else:
                out.append(DataUsage(dataset="ff_factors_official", source=FRENCH_SOURCE, coverage="unavailable",
                                     point_in_time=True, notes=str(err or "")[:300]))
        if rf_daily is not None and len(rf_daily):
            out.append(DataUsage(dataset="risk_free", source=f"{rf_source} (daily 1-month T-bill)",
                                 coverage=f"{_fmt(rf_daily.index[0])} to {_fmt(rf_daily.index[-1])}", point_in_time=True))
        else:
            out.append(DataUsage(dataset="risk_free", source="none", coverage="rf = 0", point_in_time=True,
                                 notes="official RF unavailable"))
        return out

    # ------------------------------------------------------------------ factor models
    def _month_days(self, ctx: _Context, upto: pd.Timestamp | None = None) -> pd.DatetimeIndex:
        """Last trading day of each calendar month in the preloaded data (up to ``upto``)."""
        idx = ctx.close.index[ctx.close.notna().any(axis=1).to_numpy()]
        if upto is not None:
            idx = idx[idx <= upto]
        if not len(idx):
            return pd.DatetimeIndex([])
        s = pd.Series(idx, index=idx)
        return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().to_numpy())

    def _book_equity(self, ctx: _Context, t: pd.Timestamp) -> pd.Series:
        fund = ctx.wrapped.get_fundamentals(list(ctx.tickers), t.date())
        if F.TICKER in fund.columns and fund.index.name != F.TICKER:
            fund = fund.set_index(F.TICKER)
        fund = fund[~fund.index.duplicated(keep="last")]
        be = pd.to_numeric(fund.get(F.TOTAL_EQUITY), errors="coerce") if F.TOTAL_EQUITY in fund.columns else None
        if be is None:
            return pd.Series(np.nan, index=ctx.universe.index)
        be.index = pd.Index([str(i) for i in be.index])
        return be.reindex(ctx.universe.index).astype(float)

    def _op_inv(self, ctx: _Context, t: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
        ff = ctx.engine.build(self._universe_at(ctx, t), t.date(), features={"operating_profitability_pct", "asset_growth_yoy_pct"})
        ctx.warnings.extend(ff.warnings)
        fr = ff.frame
        return (pd.to_numeric(fr["operating_profitability_pct"], errors="coerce") / 100.0,
                pd.to_numeric(fr["asset_growth_yoy_pct"], errors="coerce") / 100.0)

    def _factor_model_backtest(self, spec: StrategySpec, ctx: _Context, rf_daily: pd.Series | None, rf_source: str,
                               label: str | None, started: datetime) -> BacktestResult:
        model = spec.factor_model
        assert model is not None
        days = self._month_days(ctx)
        if len(days) < 3:
            raise ValueError("factor models need at least three months of prices")
        labels = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in days]).normalize()
        close_m = pd.DataFrame(ctx.close.reindex(columns=ctx.tickers).ffill().loc[days].to_numpy(),
                               index=labels, columns=ctx.tickers)
        # a name that stopped trading has no later monthly price: do not carry its last close forward
        last_valid = ctx.close.reindex(columns=ctx.tickers).apply(lambda s: s.last_valid_index())
        for tk, lv in last_valid.items():
            if lv is None or pd.isna(lv):
                close_m[tk] = np.nan
            else:
                close_m.loc[labels > (pd.Timestamp(lv) + pd.offsets.MonthEnd(0)), tk] = np.nan
        returns = close_m / close_m.shift(1) - 1.0
        self._say(f"{spec.name}: point-in-time market caps at {len(days)} month-ends")
        me = pd.DataFrame({lab: self._mcap_at(ctx, d) for lab, d in zip(labels, days)}).T.reindex(columns=ctx.tickers)
        me.index = labels
        be = op = inv = mom = None
        june = [(lab, d) for lab, d in zip(labels, days) if d.month == 6 and lab <= pd.Timestamp(ctx.end) + pd.offsets.MonthEnd(0)]
        if model != "capm":
            self._say(f"{spec.name}: book equity at {len(june)} June formation date(s)")
            be = pd.DataFrame({lab: self._book_equity(ctx, d) for lab, d in june}).T.reindex(columns=ctx.tickers)
            if be.empty:
                be = pd.DataFrame(index=pd.DatetimeIndex([]), columns=ctx.tickers, dtype=float)
            be.index = pd.DatetimeIndex(be.index)
        if model == "ff5":
            parts = {lab: self._op_inv(ctx, d) for lab, d in june}
            op = pd.DataFrame({lab: v[0] for lab, v in parts.items()}).T.reindex(columns=ctx.tickers)
            inv = pd.DataFrame({lab: v[1] for lab, v in parts.items()}).T.reindex(columns=ctx.tickers)
            op.index = pd.DatetimeIndex(op.index) if len(op) else pd.DatetimeIndex([])
            inv.index = pd.DatetimeIndex(inv.index) if len(inv) else pd.DatetimeIndex([])
        if model == "carhart4":
            mom = close_m.shift(1) / close_m.shift(12) - 1.0
        exchange = ctx.universe[F.EXCHANGE] if F.EXCHANGE in ctx.universe.columns else None
        official_m, err = self._official(model, "monthly")
        rf_m = official_m["RF"].astype(float) if official_m is not None and "RF" in official_m.columns else None
        panel = CharacteristicsPanel(returns=returns, market_cap=me, book_equity=be, operating_profitability=op,
                                     investment=inv, momentum_12_1=mom, exchange=exchange, rf=rf_m)
        self._say(f"{spec.name}: constructing {model} factors")
        frame, fwarn = construct_factors(panel, model, formation="annual_june")
        ctx.warnings.extend(f"factor construction: {w}" for w in fwarn)
        cols = FACTOR_COLUMNS[model]
        start_lim = pd.Timestamp(ctx.start) - pd.Timedelta(days=5)
        in_window = [(lab - pd.offsets.MonthBegin(1) >= start_lim) and lab <= pd.Timestamp(ctx.end) + pd.offsets.MonthEnd(0)
                     for lab in frame.index]
        fac = frame.loc[in_window, cols]
        fac = fac.loc[fac.notna().any(axis=1).cumsum() > 0]  # drop leading months before the first formation
        if fac.empty or fac.notna().sum().max() < 2:
            raise ValueError(f"could not construct {model} factors in {ctx.start} to {ctx.end} (too little data)")
        nonmarket = [c for c in cols if c != "Mkt-RF"] or ["Mkt-RF"]
        strategy = fac[nonmarket].mean(axis=1, skipna=False).rename("strategy")
        bench_m = None
        if ctx.benchmark is not None:
            b = ctx.benchmark
            bd = (b / b.shift(1) - 1.0).dropna()
            first_lab = fac.index[0]
            bd = bd[(bd.index > first_lab - pd.offsets.MonthEnd(1)) & (bd.index <= pd.Timestamp(ctx.end))]
            if len(bd) >= 2:
                bench_m = compound(bd, "M")

        stats: dict[str, PerformanceStats] = {}
        for c in cols:
            self._add_stats(stats, c, fac[c], None, None, None, ctx.warnings, periods_per_year=12.0)
        self._add_stats(stats, "strategy", strategy, None, bench_m, None, ctx.warnings, periods_per_year=12.0)
        if bench_m is not None:
            self._add_stats(stats, "benchmark", bench_m, rf_m if rf_m is not None else rf_daily, None, None, ctx.warnings,
                            periods_per_year=12.0)
        if official_m is not None:
            checks = compare_with_official(fac, official_m)
        else:
            checks = [FactorConstructionCheck(factor=c, correlation_with_official=None,
                                              annual_premium_constructed_pct=float(fac[c].dropna().mean() * 1200.0)
                                              if fac[c].notna().any() else None,
                                              annual_premium_official_pct=None, n_overlap_periods=0) for c in cols]
            ctx.warnings.append(f"official {model} factors unavailable ({err}): no comparison with the Kenneth French series")
        regression = None
        if spec.attribution_model is not None:
            y = strategy.dropna()
            fac_off, ferr = self._official(spec.attribution_model, "monthly")
            if fac_off is None:
                ctx.warnings.append(f"factor attribution ({spec.attribution_model}) skipped: official factors unavailable ({ferr})")
            elif len(y) < MIN_REGRESSION_OBS:
                ctx.warnings.append(f"factor attribution ({spec.attribution_model}) skipped: only {len(y)} month(s) (< {MIN_REGRESSION_OBS})")
            else:
                try:
                    data = align_factor_dates(fac_off, y.index)[FACTOR_COLUMNS[spec.attribution_model]].dropna()
                    regression = factor_regression(y, data, model=spec.attribution_model,
                                                   factor_source=str(fac_off.attrs.get("source", FRENCH_SOURCE)), excess=False)
                except (ValueError, KeyError) as exc:
                    ctx.warnings.append(f"factor attribution ({spec.attribution_model}) skipped: {exc}")

        try:
            holdings = self._factor_weights_at(ctx, pd.Timestamp(ctx.end))
        except Exception as exc:  # noqa: BLE001 - holdings are informational
            ctx.warnings.append(f"latest factor-portfolio holdings not computed: {type(exc).__name__}: {exc}")
            holdings = pd.Series(dtype=float)
        monthly: dict[str, pd.Series] = {c: fac[c] for c in cols}
        monthly["strategy"] = strategy
        if bench_m is not None:
            monthly["benchmark"] = bench_m[(bench_m.index >= fac.index[0]) & (bench_m.index <= fac.index[-1])]
        out_dates, returns_out = self._monthly_table(monthly)
        data_usage = self._data_usage(spec, ctx, {}, rf_daily, rf_source, regression is not None,
                                      factor_model_fund=model != "capm")
        ctx.warnings.append(
            "factor model: 'strategy' is the equal-weighted average of the constructed "
            + ("non-market factors (" + ", ".join(nonmarket) + ")" if nonmarket != ["Mkt-RF"] else "market factor (Mkt-RF)")
            + "; the factors are value-weighted 2x3 sorts formed each June (momentum monthly)"
        )
        self.last_run = RunDetails(spec=spec, start=fac.index[0].date(), end=ctx.end, rebalance_dates=list(days),
                                   factor_returns=fac.assign(strategy=strategy))
        return BacktestResult(
            run_id=self._run_id(spec, ctx.start, ctx.end, label),
            idea=spec.idea,
            spec=spec.model_dump(mode="json"),
            provider=self.provider_name,
            llm="none",
            start=max(ctx.start, (fac.index[0] - pd.offsets.MonthBegin(1)).date()),
            end=ctx.end,
            rebalance=spec.rebalance,
            returns=returns_out,
            dates=out_dates,
            stats=stats,
            regression=regression,
            quantiles=None,
            factor_checks=checks,
            latest_holdings={str(k): float(v) for k, v in holdings.items() if float(v) != 0.0},
            data_usage=data_usage,
            warnings=[],
            started_at=started,
        )

    def _factor_weights_at(self, ctx: _Context, as_of: pd.Timestamp) -> pd.Series:
        """Factor-mimicking weights of the factor model's headline portfolio at ``as_of`` (module docstring)."""
        model = ctx.spec.factor_model
        assert model is not None
        days = self._month_days(ctx, as_of)
        if not len(days):
            return pd.Series(dtype=float)
        now = days[-1]  # last session on or before as_of
        cap_now = self._mcap_at(ctx, now)
        alive = ctx.close.reindex(columns=ctx.tickers).loc[:now].iloc[-1]
        cap_now = cap_now.where(alive.notna() & (alive > 0))
        exchange = ctx.universe[F.EXCHANGE] if F.EXCHANGE in ctx.universe.columns else None

        def vw(names: list[str]) -> pd.Series:
            c = cap_now.reindex(names).dropna()
            c = c[c > 0]
            return c / c.sum() if len(c) else pd.Series(dtype=float)

        def ports(lbl: pd.Series, names: tuple[str, str, str]) -> dict[str, pd.Series]:
            return {f"{sz}/{n}": vw(list(lbl.index[lbl == f"{sz}/{n}"])) for sz in ("S", "B") for n in names}

        def combo(parts: list[tuple[float, pd.Series]]) -> pd.Series | None:
            if any(len(p) == 0 for _, p in parts):
                return None
            out = pd.Series(dtype=float)
            for k, p in parts:
                out = out.add(k * p, fill_value=0.0)
            return out

        if model == "capm":
            w = vw(list(cap_now.dropna().index))
            return w[w != 0].sort_index()
        factors: list[pd.Series] = []
        june_days = [d for d in days if d.month == 6 and pd.Timestamp(d.year, 6, 30) <= as_of]
        if june_days:
            j = june_days[-1]
            size = self._mcap_at(ctx, j)
            dec_days = [d for d in days if d.year == j.year - 1 and d.month == 12]
            me_dec = self._mcap_at(ctx, dec_days[-1]) if dec_days else size
            be = self._book_equity(ctx, j)
            with np.errstate(invalid="ignore", divide="ignore"):
                bm = (be / me_dec).where((be > 0) & (me_dec > 0))
            lbl, _ = two_by_three_sort(size, bm, exchange=exchange)
            p = ports(lbl.dropna(), ("L", "M", "H"))
            smb_bm = combo([(1 / 3, p["S/L"]), (1 / 3, p["S/M"]), (1 / 3, p["S/H"]),
                            (-1 / 3, p["B/L"]), (-1 / 3, p["B/M"]), (-1 / 3, p["B/H"])])
            hml = combo([(0.5, p["S/H"]), (0.5, p["B/H"]), (-0.5, p["S/L"]), (-0.5, p["B/L"])])
            if model == "ff5":
                op, inv = self._op_inv(ctx, j)
                with np.errstate(invalid="ignore"):
                    op = op.where(be.reindex(op.index) > 0)
                lo, _ = two_by_three_sort(size, op, exchange=exchange, labels=("W", "N", "R"))
                li, _ = two_by_three_sort(size, inv, exchange=exchange, labels=("C", "N", "A"))
                po, pi = ports(lo.dropna(), ("W", "N", "R")), ports(li.dropna(), ("C", "N", "A"))
                smb_op = combo([(1 / 3, po[k]) for k in ("S/W", "S/N", "S/R")] + [(-1 / 3, po[k]) for k in ("B/W", "B/N", "B/R")])
                smb_inv = combo([(1 / 3, pi[k]) for k in ("S/C", "S/N", "S/A")] + [(-1 / 3, pi[k]) for k in ("B/C", "B/N", "B/A")])
                if smb_bm is not None and smb_op is not None and smb_inv is not None:
                    factors.append((smb_bm / 3.0).add(smb_op / 3.0, fill_value=0.0).add(smb_inv / 3.0, fill_value=0.0))
                for f in (hml,
                          combo([(0.5, po["S/R"]), (0.5, po["B/R"]), (-0.5, po["S/W"]), (-0.5, po["B/W"])]),
                          combo([(0.5, pi["S/C"]), (0.5, pi["B/C"]), (-0.5, pi["S/A"]), (-0.5, pi["B/A"])])):
                    if f is not None:
                        factors.append(f)
            else:
                factors.extend(f for f in (smb_bm, hml) if f is not None)
        if model == "carhart4" and len(days) >= 13:
            px = ctx.close.reindex(columns=ctx.tickers).ffill()
            p1, p12 = px.loc[days[-2]], px.loc[days[-13]]
            mom = (p1 / p12 - 1.0).where((p1 > 0) & (p12 > 0))
            lm, _ = two_by_three_sort(cap_now, mom, exchange=exchange, labels=("Down", "Mid", "Up"))
            pm = ports(lm.dropna(), ("Down", "Mid", "Up"))
            f = combo([(0.5, pm["S/Up"]), (0.5, pm["B/Up"]), (-0.5, pm["S/Down"]), (-0.5, pm["B/Down"])])
            if f is not None:
                factors.append(f)
        if not factors:
            return pd.Series(dtype=float)
        total = pd.Series(dtype=float)
        for f in factors:
            total = total.add(f / len(factors), fill_value=0.0)
        total = total[np.abs(total.to_numpy()) > 1e-15]
        return total.sort_index().rename("weight")

    # ------------------------------------------------------------------ live surface
    def target_portfolio(self, spec: StrategySpec, as_of: date) -> pd.Series:
        """Signed target weights after rebalancing on ``as_of`` (only data up to the close of ``as_of``).

        Same per-date construction as :meth:`backtest`; the universe is the provider's as of
        ``as_of``. Time-series rules with exit conditions replay their state from the window start
        (``spec.start`` or ``as_of - default_years``).
        """
        errors = self._spec_errors(spec)
        if errors:
            raise ValueError("invalid strategy spec: " + "; ".join(errors))
        as_of = _as_date(as_of)
        warnings: list[str] = []
        latest = self._latest_data_date()
        if as_of > latest:
            as_of = latest
        t = pd.Timestamp(as_of)
        if spec.kind == "time_series" and spec.time_series is not None and spec.time_series.exit:
            start = _as_date(spec.start) if spec.start is not None and _as_date(spec.start) < as_of else \
                (t - pd.DateOffset(years=self.default_years)).date()
        elif spec.kind == "factor_model":
            start = (t - pd.DateOffset(months=13)).date()
        else:
            start = as_of - timedelta(days=1)
        ctx = self._context(spec, start, as_of, warnings, with_benchmark=False)
        t = pd.Timestamp(ctx.end)
        if spec.kind == "factor_model":
            return self._factor_weights_at(ctx, t)
        if spec.kind == "time_series":
            if spec.time_series is not None and spec.time_series.exit:
                dates = [d for d in rebalance_dates(ctx.close.index, spec.rebalance, ctx.start, t)]
                if not dates or dates[-1] != t:
                    dates.append(t)
            else:
                dates = [t]
            targets, _, _, _ = self._time_series_path(ctx, dates)
            return targets[dates[-1]]
        return self._cross_section(ctx, t).weights

    def latest_prices(self, tickers: list[str], as_of: date) -> pd.Series:
        """Adjusted close on or before ``as_of`` (within 31 days) per ticker; NaN when unavailable."""
        req = list(dict.fromkeys(str(t) for t in tickers))
        if not req:
            return pd.Series(dtype=float)
        as_of = _as_date(as_of)
        try:
            panel = self.provider.get_price_history(req, as_of - timedelta(days=31), as_of)
            close = panel.close
            close = close.loc[close.index <= pd.Timestamp(as_of)]
            last = close.ffill().iloc[-1] if len(close) else pd.Series(np.nan, index=req)
        except Exception:  # noqa: BLE001 - unavailable prices are NaN
            last = pd.Series(np.nan, index=req)
        last.index = [str(i) for i in last.index]
        out = pd.to_numeric(last, errors="coerce").reindex(req).astype(float)
        return out.where(np.isfinite(out.to_numpy()) & (out > 0))
