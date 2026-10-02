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

Price floor: ``UniverseSpec.min_price`` is a floor on the price a stock TRADED at. Providers serve
split- and dividend-adjusted closes (the ``PricePanel`` contract), which later splits and dividends
restate downwards, so on adjusted prices the floor is suspended at dates before the provider's latest
data date (a warning and the ``universe`` data-usage note say so; the liquidity floor, which is
split-invariant, still applies - but at past dates it is computed from dividend-adjusted closes, so it
understates the dollar volume of names that paid dividends later, more so the further back). It
applies at every date when the prices are as traded (the synthetic market, or a provider declaring
``prices_as_traded = True``). For the same reason a run that uses a ratio of a per-share vendor value
to the close (``PER_SHARE_PRICE_RATIOS``: ``pe_ntm``, ``earnings_yield_ntm_pct``,
``target_price_upside_pct``) at those dates gets a "not point-in-time" warning (later splits and
dividends restate the denominator), which caps the verdict at inconclusive; it only fires for
providers that serve historical estimates (the free provider's are blank at past dates).

Market cap (only computed when something uses it: ``MARKET_CAP_FEATURES`` - size and the valuation
ratios book-to-market, FCF / earnings yield, EV ... - value weights, or a factor model; otherwise the
universe's cap column is blank so the END snapshot can never stand in for a past date). Modes
(:meth:`StrategyRunner._mcap_mode`):

* ``snapshot`` - ``provider.get_universe(spec.universe, t)`` at every date (the cheap offline
  synthetic provider, or a provider declaring ``point_in_time_market_cap = True``);
* ``anchored`` (default) - the provider's universe snapshot (point-in-time by the provider contract;
  e.g. the free provider prices SEC cover-page shares as of the date) at the last session of each
  June and December ``a`` - about two calls a year - rolled forward to ``t`` with the adjusted-price
  ratio P(t) / P(a), which only uses prices in (a, t]. Share changes between snapshots are not
  reflected, dividends since ``a`` count as reinvested, a name listed after ``a`` has no cap until
  the next snapshot. The June / December snapshots are exactly the Fama-French size / B/M dates;
* ``end_scaled`` - a provider declaring ``point_in_time_market_cap = False``: END cap x P(t) / P(end),
  which feeds the END share count and later dividends to every date. A ``MARKET CAPS NOT
  POINT-IN-TIME`` warning and a ``market_cap`` data-usage entry with ``point_in_time=False`` make the
  interpreter treat it as look-ahead. An anchored snapshot that fails falls back to this estimate for
  the dates that depend on it, flagged the same way.

A provider that had to price some names of a past snapshot with a later (e.g. today's) share count
lists them in the snapshot's ``attrs[MCAP_CURRENT_SHARES_ATTR]``; the ``market_cap`` entry is then
``point_in_time=False`` and a ``MARKET CAPS NOT POINT-IN-TIME`` warning names them.

Universe labels (sector, industry, exchange, security type) are the END snapshot's values at every
date. Unless the provider never reclassifies (the synthetic market), the ``universe`` data-usage note
says so, and an ``END-DATE LABELS`` warning appears when they drive the run (sector exclusions,
sector-neutral signals, label conditions, NYSE breakpoints of a factor model).

A name is given a cap only if it traded within 10 days of the date (no stale caps for dead names).
The ``market_cap`` data-usage entry records the mode.

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
  else 0. Missing data never opens a position: an asset without a price at t (not listed yet,
  halted) is flat with its state reset, and one without its entry inputs (e.g. during the warm-up
  of a 200-day average) is neither long nor short (an open long under an exit rule keeps following
  it); a warning lists the dates per asset. The state is replayed date by date
  (``target_portfolio`` replays from the window start).
* ``factor_model`` - see below.

Execution and returns
---------------------
``simulate(target_weights, close, costs_bps, execution_lag, start, end, delisting_return)``: signal
at the close of t, traded ``execution_lag`` sessions later, weights drift, costs on traded notional.
Long-short books (cross_sectional ``long_short``) are self-financing: collateral earns 0, Sharpe is
on the raw spread and the factor regression uses ``excess=False``. Every other book (long-only,
screen, time series) earns the daily risk-free rate on its cash (max(0, 1 - sum of the executed
weights), drift ignored: a net-short time-series book keeps its capital plus the short-sale
proceeds, e.g. 2 for a -1 short, at RF - a full rebate - so the short state's excess return is
-(r - rf)), Sharpe is in excess of RF and the regression uses ``excess=True``. Long-short runs also report the ``long`` leg (positive weights, simulated with the
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
The universe is the provider's (country / security types) without the price and liquidity floors
or ``spec.filters``: every name with data enters the sorts, as in Fama-French.
A :class:`~aitrading.factors.construct.CharacteristicsPanel` on calendar month-ends: returns from
month-end closes, point-in-time market caps (above), book equity = ``total_equity`` of the latest
public filing as of each June formation date (the point-in-time lag), ff5 operating profitability /
investment = ``operating_profitability_pct`` / ``asset_growth_yoy_pct`` / 100 from the feature engine
at the June formation date, carhart4 momentum = P(m-1) / P(m-12) - 1 from month-end closes (the
monthly-granularity ``return_12m_ex_1m_pct``), NYSE flags from the universe ``exchange``, RF = the
official monthly RF. ``construct_factors(panel, model, formation="annual_june")`` builds the
factors, ``compare_with_official`` gives ``factor_checks``. A name that stops trading has no later
monthly price; a non-zero ``spec.delisting_return`` is booked on its first missing session (as in
``simulate``). A last month the data does not cover to its end (the data ending more than 3 days
before the calendar month-end) is left out of every series, the statistics and the attribution.
``returns`` holds every factor (gross, like the official ones) plus ``strategy`` = the
equal-weighted average of the model's non-market long-short factors (SMB / HML / ... ; the
constructed Mkt-RF for capm) NET of ``spec.costs_bps`` one-way costs on the traded notional of the
factor-mimicking portfolio: its weights at every month-end session, drifted through the month with
the monthly returns, against the next month-end's weights (the cost of the trade at the end of month
m is deducted from month m+1's return; the first trade builds the book from cash). Factors and
``strategy`` are formed and traded at the month-end close (the Fama-French convention; the runner's
``execution_lag`` is not applied - a warning says so). Statistics are per factor (raw, they are
already excess / self-financing returns) and for ``strategy`` (with its turnover) / ``benchmark``.
``target_portfolio`` / ``latest_holdings`` give the factor-mimicking weights of that headline
portfolio: the 2x3 value-weighted portfolios of the latest June formation whose formation SESSION
(the last trading session of June, as in ``construct_factors``) is on or before the date (momentum:
at the date), with the market-cap weights of the date inside each portfolio; over the backtest these
weights reproduce the constructed gross factor returns exactly.

Run ids are deterministic: ``<spec name>-<hash>`` plus ``-<label>`` when a label is given; the hash
covers the spec JSON, the provider name and configuration (:meth:`StrategyRunner.provider_fingerprint`:
a provider ``fingerprint``, else its class, seed / n_tickers / start / end / benchmark and its ticker
list), the universe (sorted tickers), the window and the execution lag - so the same idea on two
differently configured providers never shares an id (or an idea-lab run folder). ``runner.last_run``
keeps the internals of the latest backtest (:class:`RunDetails`: target weights and signals per date,
the simulation; for factor models the monthly factor-mimicking weights and gross / cost / net series).
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
from aitrading.data.base import MCAP_CURRENT_SHARES_ATTR, PricePanel
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
    "snapshot_only_features",
    "MARKET_CAP_FEATURES",
    "PRELOAD_LOOKBACK_DAYS",
    "REBALANCES_PER_YEAR",
]

#: Calendar days of history loaded before ``start`` (the feature engine's look-back).
PRELOAD_LOOKBACK_DAYS = LOOKBACK_CALENDAR_DAYS
MARKET_PROXIES = frozenset({"SPY", "VOO", "IVV", "VTI", "^GSPC", "SPX", "^SPX", "SPX INDEX", "MARKET", "US MARKET"})
REBALANCES_PER_YEAR: dict[str, float] = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0, "annual": 1.0}
_SNAPSHOT_DATASETS = ("estimates", "short_interest", "options")
#: Features that read a snapshot dataset but stay point-in-time when the provider's snapshots are
#: current-only: the last earnings date comes from dated filings (SEC 8-K release dates on the free
#: provider) and falls back to the fundamentals' report date.
_POINT_IN_TIME_DESPITE_SNAPSHOT = frozenset({"days_since_last_earnings"})
#: Catalog features computed from the universe market cap: they need a point-in-time cap at each date.
MARKET_CAP_FEATURES = frozenset({
    "market_cap_usd_bn", "enterprise_value_usd_bn", "fcf_yield_pct", "ev_to_ebitda", "ev_to_sales",
    "cash_pct_market_cap", "book_to_market", "earnings_yield_ttm_pct",
})
#: Ratios of a per-share vendor value (consensus EPS, target price) to the close. On adjusted prices at a
#: date before the latest data date the close is restated for later splits and dividends, so the ratio
#: is not point-in-time (same list as ``pipeline.PER_SHARE_PRICE_RATIOS``).
PER_SHARE_PRICE_RATIOS = ("pe_ntm", "earnings_yield_ntm_pct", "target_price_upside_pct")
#: Calendar months whose last session is a market-cap snapshot date in ``anchored`` mode (the
#: Fama-French formation months: June for size, December for book-to-market).
MCAP_ANCHOR_MONTHS = (6, 12)
#: ``MCAP_CURRENT_SHARES_ATTR`` (defined in ``aitrading.data.base``, re-exported here): the ``attrs`` key
#: of a universe snapshot listing the tickers whose market cap uses a share count from AFTER its
#: ``as_of``, i.e. caps that are NOT point-in-time. The runner then reports market caps as look-ahead.
#: Universe label columns (catalog features) that can drive a run: sector exclusions, sector-neutral
#: signals, exchange conditions and NYSE breakpoints.
_LABEL_FEATURES = frozenset({F.GICS_SECTOR, F.GICS_INDUSTRY, F.EXCHANGE})
#: A market cap is only estimated for a name that traded within this many calendar days of the date.
_MCAP_STALE_DAYS = 10
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


_DATED = re.compile(r"^(\d{4}-\d{2}-\d{2}): (.*)$", re.S)


def _compress_dated(warnings: list[str], keep: int = 2) -> list[str]:
    """Collapse repeated per-date warnings ("2020-01-31: dropped 2 target name(s) ...") of the same kind
    (the first four words after the date, numbers ignored): the first ``keep`` are kept, the rest
    become one "... and N more" line, so they cannot crowd out the other warnings."""
    groups: dict[str, list[str]] = {}
    order: list[tuple[str, str]] = []
    for w in warnings:
        m = _DATED.match(w)
        if not m:
            order.append(("", w))
            continue
        kind = " ".join(re.sub(r"\d+", "#", m.group(2)).split()[:4])
        if kind not in groups:
            order.append((kind, ""))
        groups.setdefault(kind, []).append(w)
    out: list[str] = []
    for kind, w in order:
        if not kind:
            out.append(w)
            continue
        items = groups[kind]
        out.extend(items[:keep])
        if len(items) > keep:
            first_dates = [_DATED.match(x).group(1) for x in items[keep:]]  # type: ignore[union-attr]
            out.append(f"... and {len(items) - keep} more warning(s) like '{kind}' ({first_dates[0]} to {first_dates[-1]})")
    return out


def _names(names: list[str], limit: int = 8) -> str:
    names = [str(n) for n in names]
    return ", ".join(names[:limit]) + (f", +{len(names) - limit} more" if len(names) > limit else "")


def snapshot_only_features(features: set[str] | list[str]) -> dict[str, list[str]]:
    """Feature -> the snapshot datasets (estimates / short interest / options) it needs, for the
    features that have NO point-in-time history on a provider whose snapshots are current-only.

    ``days_since_last_earnings`` is left out: it reads the estimates dataset, but its input (the last
    earnings date) comes from dated filings and falls back to the fundamentals' report date.
    """
    out: dict[str, list[str]] = {}
    for f in sorted(set(features)):
        if f in _POINT_IN_TIME_DESPITE_SNAPSHOT:
            continue
        ds = [d for d in _SNAPSHOT_DATASETS if d in FEATURE_DATASETS.get(f, ())]
        if ds:
            out[f] = ds
    return out


def _sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    universe: list[str] = field(default_factory=list)  # the tickers the run used (part of the run id)


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
    mcap_mode: str  # "snapshot" | "anchored" | "end_scaled" (see StrategyRunner._mcap_mode)
    universe_spec: Any
    mcap_needed: bool = True  # False: no feature / weighting uses market cap, so none is computed
    prices_as_traded: bool = True  # False: adjusted prices (price floors suspended at past dates)
    price_floor_from: date | None = None  # dates on or after it apply the price floor on adjusted prices
    warnings: list[str] = field(default_factory=list)
    benchmark: pd.Series | None = None
    benchmark_label: str = ""
    data_start: date | None = None
    eligible_counts: list[int] = field(default_factory=list)
    floor_suspended: list[pd.Timestamp] = field(default_factory=list)
    per_share_biased: dict[str, list[pd.Timestamp]] = field(default_factory=dict)  # ratio -> dates with values on adjusted past prices
    ts_unavailable: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    mcap_fallback: list[str] = field(default_factory=list)  # snapshot dates estimated from the END cap
    mcap_current_shares: dict[str, list[str]] = field(default_factory=dict)  # snapshot date -> names capped with a later share count
    _mcap_cache: dict = field(default_factory=dict)
    _mcap_snapshots: dict = field(default_factory=dict)
    _mcap_anchors: pd.DatetimeIndex | None = None
    _close_ffill: pd.DataFrame | None = None
    _last_valid_pos: np.ndarray | None = None
    _mcap_scale: pd.Series | None = None
    _sessions: pd.DatetimeIndex | None = None
    _formations: dict = field(default_factory=dict)

    @property
    def close(self) -> pd.DataFrame:
        return self.prices.close

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """Dates with at least one valid close (the trading sessions in the loaded data)."""
        if self._sessions is None:
            self._sessions = self.close.index[self.close.notna().any(axis=1).to_numpy()]
        return self._sessions

    def close_ffill(self) -> pd.DataFrame:
        """Closes of the universe forward-filled (row lookups only ever read rows up to the date)."""
        if self._close_ffill is None:
            self._close_ffill = self.close.reindex(columns=self.universe.index).ffill()
        return self._close_ffill

    def fresh(self, t: pd.Timestamp) -> pd.Series:
        """True for names with a valid close within ``_MCAP_STALE_DAYS`` calendar days on or before ``t``."""
        cols = self.universe.index
        idx = self.close.index
        if self._last_valid_pos is None:
            valid = self.close.reindex(columns=cols).notna().to_numpy()
            pos = np.where(valid, np.arange(len(idx))[:, None], -1)
            self._last_valid_pos = np.maximum.accumulate(pos, axis=0) if len(idx) else pos
        ts = pd.Timestamp(t)
        p = int(idx.searchsorted(ts, side="right")) - 1
        if p < 0:
            return pd.Series(False, index=cols)
        lp = self._last_valid_pos[p]
        ok = lp >= 0
        age = np.full(len(lp), np.inf)
        age[ok] = (ts.to_datetime64() - idx.to_numpy()[lp[ok]]) / np.timedelta64(1, "D")
        return pd.Series(ok & (age <= _MCAP_STALE_DAYS), index=cols)


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

    def _market_proxy_fill(self, spec: StrategySpec, prices: PricePanel, start: date, end: date, warnings: list[str]) -> PricePanel:
        """A timing rule on 'the market' (SPY, ^GSPC, ...) uses the provider's benchmark index when the
        provider has no such ticker (e.g. the simulated market), with a warning; OHLC = close, no volume."""
        missing = [t for t in prices.close.columns if t.upper() in MARKET_PROXIES and prices.close[t].isna().all()]
        if not missing:
            return prices
        try:
            bench = pd.Series(self.provider.get_benchmark_history(start, end), dtype=float)
        except Exception:  # noqa: BLE001 - leave the gap; the caller reports missing prices
            return prices
        bench = bench[bench > 0]
        if len(bench) < 2:
            return prices
        idx = prices.close.index.union(pd.DatetimeIndex(bench.index))
        frames = {k: getattr(prices, k).reindex(idx) for k in ("open", "high", "low", "close", "volume")}
        aligned = bench.reindex(idx)
        for t in missing:
            for k in ("open", "high", "low", "close"):
                frames[k][t] = aligned
            frames["volume"][t] = np.nan
        label = getattr(self.provider, "benchmark", None) or getattr(bench, "name", None) or "benchmark"
        warnings.append(
            f"{', '.join(missing)}: not available from provider '{self.provider_name}'; the provider's market index ({label}) "
            "is used as a stand-in (no volume, so volume-based conditions cannot fire)"
        )
        return PricePanel(**frames)

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

    def _mcap_mode(self) -> str:
        """How market caps at past dates are obtained (module docstring, "Market cap").

        * ``snapshot`` - ``provider.get_universe(spec, t)`` at every date: providers declaring
          ``point_in_time_market_cap = True`` and the (cheap, offline) synthetic provider.
        * ``anchored`` - the default: ``get_universe(spec, a)`` (point-in-time by the provider
          contract) only at the last session of each June and December ``a``, rolled forward to
          ``t`` with the adjusted-price ratio P(t) / P(a) (about two snapshot calls a year).
        * ``end_scaled`` - providers declaring ``point_in_time_market_cap = False`` (their snapshots
          at past dates are not point-in-time): END cap x P(t) / P(end), reported as NOT
          point-in-time in ``data_usage`` and the warnings.
        """
        flag = getattr(self.provider, "point_in_time_market_cap", None)
        if flag is None:
            return "snapshot" if self.provider_name == "synthetic" else "anchored"
        return "snapshot" if bool(flag) else "end_scaled"

    def _prices_as_traded(self) -> bool:
        """True when the provider's closes are the prices the stocks traded at (no later split /
        dividend restatement): providers declaring ``prices_as_traded = True`` and the synthetic
        market (no corporate actions). Every other provider serves adjusted prices (the
        ``PricePanel`` contract)."""
        flag = getattr(self.provider, "prices_as_traded", None)
        if flag is not None:
            return bool(flag)
        return self.provider_name == "synthetic"

    @staticmethod
    def _needs_mcap(spec: StrategySpec, needed: set[str]) -> bool:
        return spec.kind == "factor_model" or bool(set(needed) & MARKET_CAP_FEATURES)

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
        if spec.kind == "time_series":
            prices = self._market_proxy_fill(spec, prices, preload_start, end, warnings)
        close = prices.close
        valid = close.notna().any(axis=1)
        if not valid.any():
            raise ValueError(f"provider '{self.provider_name}' has no price data for {len(tickers)} ticker(s) "
                             f"({_names(tickers)}) between {preload_start} and {end}")
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
        needed = self._needed_features(spec)
        ctx = _Context(
            spec=spec, start=start, end=end, preload_start=preload_start, universe=universe, tickers=tickers, prices=prices,
            wrapped=wrapped, engine=FeatureEngine(wrapped, self.catalog), needed=needed,
            mcap_mode=self._mcap_mode(), universe_spec=universe_spec, mcap_needed=self._needs_mcap(spec, needed),
            prices_as_traded=self._prices_as_traded(), price_floor_from=self._latest_data_date(),
            warnings=warnings, data_start=first_px,
        )
        if with_benchmark:
            ctx.benchmark, ctx.benchmark_label = self._benchmark(spec, ctx)
        return ctx

    def _benchmark(self, spec: StrategySpec, ctx: _Context) -> tuple[pd.Series | None, str]:
        symbols: list[str | None] = [spec.benchmark] if spec.benchmark else []
        symbols.append(None)
        for sym in symbols:
            try:
                raw = ctx.wrapped.get_benchmark_history(ctx.preload_start, ctx.end, sym)
                ser = pd.Series(raw, dtype=float).dropna()
                ser = ser[ser > 0]
                if len(ser) >= 2:
                    series_name = getattr(raw, "name", None)
                    label = sym or str(getattr(self.provider, "benchmark", None)
                                       or (series_name if isinstance(series_name, str) and series_name else None)
                                       or "provider default index")
                    if sym is None and spec.benchmark:
                        ctx.warnings.append(f"benchmark {spec.benchmark} unavailable; using the provider's default index ({label})")
                    return ser.sort_index(), label
            except Exception as exc:  # noqa: BLE001 - try the default next
                if sym is not None:
                    ctx.warnings.append(f"benchmark {sym} could not be loaded ({type(exc).__name__}: {exc})")
        ctx.warnings.append("no benchmark data: benchmark-relative statistics are not reported")
        return None, ""

    # ------------------------------------------------------------------ point-in-time market cap
    @staticmethod
    def _close_row(ctx: _Context, t: pd.Timestamp) -> pd.Series:
        """Forward-filled closes on the last session on or before ``t`` (NaN before the data)."""
        ff = ctx.close_ffill()
        pos = int(ff.index.searchsorted(pd.Timestamp(t), side="right")) - 1
        return ff.iloc[pos] if pos >= 0 else pd.Series(np.nan, index=ff.columns)

    def _month_end_sessions(self, ctx: _Context, months: tuple[int, ...] | None = None) -> pd.DatetimeIndex:
        """Sessions in the loaded data that are the LAST trading session of their calendar month
        (restricted to ``months`` when given). The last session of the data counts only when the
        next business day falls in another month (the data may stop mid-month)."""
        s = ctx.sessions
        if not len(s):
            return pd.DatetimeIndex([])
        per = s.to_period("M")
        last = pd.Series(s, index=s).groupby(per).max()
        out = [d for d in pd.DatetimeIndex(last.to_numpy()) if months is None or d.month in months]
        if out and out[-1] == s[-1] and (s[-1] + pd.offsets.BDay(1)).month == s[-1].month:
            out = out[:-1]
        return pd.DatetimeIndex(out)

    def _is_month_end_session(self, ctx: _Context, d: pd.Timestamp) -> bool:
        """True when ``d`` is the last trading session of its month: the next session in the data is
        in a later month, or ``d`` ends the data and the next business day is in another month."""
        s = ctx.sessions
        d = pd.Timestamp(d)
        later = s[s > d]
        if len(later):
            return later[0].to_period("M") != d.to_period("M")
        return (d + pd.offsets.BDay(1)).month != d.month

    def _snapshot_caps(self, ctx: _Context, a: pd.Timestamp) -> pd.Series:
        """Market caps from the provider's universe snapshot as of ``a`` (point-in-time by the
        provider contract), indexed like the universe. In ``anchored`` mode a failed or empty
        snapshot falls back to the END-cap estimate for that date, with a NOT point-in-time warning."""
        key = pd.Timestamp(a)
        if key in ctx._mcap_snapshots:
            return ctx._mcap_snapshots[key]
        flagged: set[str] = set()
        try:
            snap = ctx.wrapped.get_universe(ctx.universe_spec, key.date())
            raw_flag = snap.attrs.get(MCAP_CURRENT_SHARES_ATTR) if isinstance(getattr(snap, "attrs", None), dict) else None
            if raw_flag is not None:
                flagged = {str(raw_flag)} if isinstance(raw_flag, str) else {str(x) for x in raw_flag}
            if F.TICKER in snap.columns and snap.index.name != F.TICKER:
                snap = snap.set_index(F.TICKER)
            snap = snap[~snap.index.duplicated(keep="last")]
            if F.MARKET_CAP in snap.columns:
                out = pd.to_numeric(snap[F.MARKET_CAP], errors="coerce").astype(float)
            else:
                out = pd.Series(np.nan, index=snap.index, dtype=float)
            out.index = pd.Index([str(i) for i in out.index])
            out = out.reindex(ctx.universe.index)
            if ctx.mcap_mode == "anchored" and not np.isfinite(out.to_numpy()).any() \
                    and np.isfinite(self._mcap_end_scaled(ctx, key).to_numpy()).any():
                raise ValueError("the snapshot has no market cap for any name")
        except Exception as exc:  # noqa: BLE001 - anchored mode degrades to the END-cap estimate, flagged
            if ctx.mcap_mode != "anchored":
                raise
            if not ctx.mcap_fallback:
                ctx.warnings.append(
                    f"market caps: provider '{self.provider_name}' gave no universe snapshot with market caps as of "
                    f"{_fmt(key)} ({type(exc).__name__}: {str(exc)[:160]}); the end-date market cap x the adjusted-price "
                    "ratio is used for the dates that depend on it, so share changes and dividends after those dates leak "
                    "into them (not point-in-time)"
                )
            ctx.mcap_fallback.append(_fmt(key))
            out = self._mcap_end_scaled(ctx, key)
            flagged = set()
        names = sorted(n for n in flagged if n in out.index and np.isfinite(out[n]))
        if names:
            ctx.mcap_current_shares[_fmt(key)] = names
        ctx._mcap_snapshots[key] = out
        return out

    def _mcap_end_scaled(self, ctx: _Context, t: pd.Timestamp) -> pd.Series:
        """END market cap x P_adj(t) / P_adj(end): the END share count and later dividends are used at
        every earlier date (NOT point-in-time)."""
        if ctx._mcap_scale is None:
            ff = ctx.close_ffill()
            last = ff.iloc[-1] if len(ff) else pd.Series(np.nan, index=ctx.universe.index)
            cap_end = pd.to_numeric(ctx.universe[F.MARKET_CAP], errors="coerce").astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                ctx._mcap_scale = (cap_end / last.where(last > 0)).astype(float)
        return (self._close_row(ctx, t) * ctx._mcap_scale).astype(float)

    def _mcap_anchor(self, ctx: _Context, t: pd.Timestamp) -> pd.Timestamp:
        """The snapshot date for ``t`` in anchored mode: the latest last-session-of-June/December on
        or before ``t``, or ``t`` itself before the first one."""
        if ctx._mcap_anchors is None:
            ctx._mcap_anchors = self._month_end_sessions(ctx, MCAP_ANCHOR_MONTHS)
        anchors = ctx._mcap_anchors
        k = int(anchors.searchsorted(pd.Timestamp(t), side="right")) - 1
        return anchors[k] if k >= 0 else pd.Timestamp(t)

    def _mcap_at(self, ctx: _Context, t: pd.Timestamp) -> pd.Series:
        """Market cap of every universe name at ``t`` (NaN when unknown), per ``ctx.mcap_mode``."""
        key = pd.Timestamp(t)
        if key in ctx._mcap_cache:
            return ctx._mcap_cache[key]
        if F.MARKET_CAP not in ctx.universe.columns:
            out = pd.Series(np.nan, index=ctx.universe.index, dtype=float)
        elif ctx.mcap_mode == "snapshot":
            out = self._snapshot_caps(ctx, key)
        else:
            if ctx.mcap_mode == "anchored":
                a = self._mcap_anchor(ctx, key)
                base = self._snapshot_caps(ctx, a)
                p_a, p_t = self._close_row(ctx, a), self._close_row(ctx, key)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = (base * (p_t / p_a.where(p_a > 0))).astype(float)
            else:
                out = self._mcap_end_scaled(ctx, key)
            out = out.where(ctx.fresh(key).reindex(out.index, fill_value=False))
        out = out.where(np.isfinite(out.to_numpy()) & (out > 0))
        ctx._mcap_cache[key] = out
        return out

    def _universe_at(self, ctx: _Context, t: pd.Timestamp) -> pd.DataFrame:
        uni = ctx.universe.copy()
        if not ctx.mcap_needed:
            # nothing uses market cap: never let the END snapshot's caps stand in for a past date
            uni[F.MARKET_CAP] = np.nan
            return uni
        if ctx.spec.kind == "time_series" and not uni[F.MARKET_CAP].notna().any():
            return uni
        uni[F.MARKET_CAP] = self._mcap_at(ctx, t).reindex(uni.index).to_numpy()
        return uni

    # ------------------------------------------------------------------ per-date portfolio
    def _universe_filters(self, ctx: _Context, t: pd.Timestamp) -> Any:
        """``spec.universe`` as applied at ``t``. On adjusted prices the price floor is suspended at
        dates before the latest data date: an adjusted close is restated for every later split and
        dividend, so "adjusted close < min_price" says nothing about the price the stock traded at on
        ``t`` and would exclude later winners because of splits that had not happened yet."""
        u = ctx.spec.universe
        if u.min_price is None or ctx.prices_as_traded:
            return u
        if ctx.price_floor_from is not None and pd.Timestamp(t).date() >= ctx.price_floor_from:
            return u
        ctx.floor_suspended.append(pd.Timestamp(t))
        return u.model_copy(update={"min_price": None})

    def _cross_section(self, ctx: _Context, t: pd.Timestamp) -> _DateOutcome:
        spec = ctx.spec
        uni = self._universe_at(ctx, t)
        ff = ctx.engine.build(uni, t.date(), features=ctx.needed)
        frame = ff.frame
        self._note_per_share_ratios(ctx, t, frame)
        mask, _ = apply_universe(self._universe_filters(ctx, t), frame)
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

    def _time_series_flags(self, ctx: _Context, t: pd.Timestamp) -> tuple[dict[str, tuple[bool, bool, bool, bool]], dict[str, float], list[str]]:
        """Per asset: (entry holds, exit holds, entry inputs and price available, price available) at ``t``."""
        rule = ctx.spec.time_series
        assert rule is not None
        uni_all = self._universe_at(ctx, t)
        out: dict[str, tuple[bool, bool, bool, bool]] = {}
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
            has_px = bool(np.isfinite(px) and px > 0)
            out[str(asset)] = (bool(entry.iloc[0]), exit_ok, has and has_px, has_px)
        coverage = {k: float(np.mean(v)) for k, v in cov.items()}
        return out, coverage, warns

    @staticmethod
    def _ts_step(long: bool, entry_ok: bool, exit_ok: bool, has_exit: bool) -> bool:
        if not long:
            return entry_ok and not (has_exit and exit_ok)
        if has_exit:
            return not exit_ok
        return entry_ok

    def _ts_weights(self, ctx: _Context, state: dict[str, bool], available: dict[str, bool]) -> pd.Series:
        """+1/n for long assets; -1/n for flat assets when ``when_flat='short'`` - only where the rule
        could be evaluated at this date (``available``): missing data never opens a position."""
        rule = ctx.spec.time_series
        assert rule is not None
        n = len(ctx.universe.index)
        w = {}
        for asset in ctx.universe.index:
            a = str(asset)
            if state.get(a, False):
                w[a] = 1.0 / n
            elif rule.when_flat == "short" and available.get(a, False):
                w[a] = -1.0 / n
        return pd.Series(w, dtype=float, name="weight").sort_index()

    def _time_series_path(self, ctx: _Context, dates: list[pd.Timestamp]) -> tuple[dict[pd.Timestamp, pd.Series], list[bool], dict[str, list[float]], list[str]]:
        """Replay the timing rule over ``dates``. An asset without a price at ``t`` (not listed yet,
        halted, delisted) holds no position and its state resets to flat. An asset with a price but
        without its entry inputs (e.g. the 200-day average during the first 200 sessions) cannot open
        a position either way - neither long nor, with ``when_flat='short'``, short; a long position
        already open under an exit rule keeps following that exit rule. Dates per asset without a
        signal are kept in ``ctx.ts_unavailable`` (reported as a warning)."""
        rule = ctx.spec.time_series
        assert rule is not None
        has_exit = bool(rule.exit)
        state = {str(a): False for a in ctx.universe.index}
        targets: dict[pd.Timestamp, pd.Series] = {}
        formable: list[bool] = []
        cov: dict[str, list[float]] = {}
        warns: list[str] = []
        ctx.ts_unavailable = {}
        for i, t in enumerate(dates):
            if i % 24 == 0:
                self._say(f"{ctx.spec.name}: signal {i + 1}/{len(dates)} ({_fmt(t)})")
            flags, coverage, w = self._time_series_flags(ctx, t)
            warns.extend(w)
            for k, v in coverage.items():
                cov.setdefault(k, []).append(v)
            available: dict[str, bool] = {}
            for asset, (entry_ok, exit_ok, has, has_px) in flags.items():
                if not has_px:
                    state[asset] = False
                elif has or (state[asset] and has_exit):
                    state[asset] = self._ts_step(state[asset], entry_ok, exit_ok, has_exit)
                else:
                    state[asset] = False
                available[asset] = has
                if not has:
                    ctx.ts_unavailable.setdefault(asset, []).append(t)
            formable.append(any(available.values()))
            targets[t] = self._ts_weights(ctx, state, available)
        return targets, formable, cov, warns

    def _ts_unavailable_warnings(self, ctx: _Context, dates: list[pd.Timestamp]) -> None:
        rule = ctx.spec.time_series
        if rule is None or not dates:
            return
        first = dates[0]
        for asset, ds in sorted(ctx.ts_unavailable.items()):
            ds = [d for d in ds if d >= first]
            if not ds:
                continue
            side = "neither long nor short" if rule.when_flat == "short" else "flat"
            ctx.warnings.append(
                f"time-series rule: {asset} had no price or no data for its entry conditions at {len(ds)} of {len(dates)} "
                f"rebalance date(s) ({_fmt(ds[0])} to {_fmt(ds[-1])}); the rule cannot be evaluated there, so no position "
                f"is held in it ({side})"
            )

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
        rf_daily, rf_source = self._daily_rf(warnings)
        if spec.kind == "factor_model":
            result = self._factor_model_backtest(spec, ctx, rf_daily, rf_source, label, started)
        else:
            result = self._portfolio_backtest(spec, ctx, rf_daily, rf_source, label, started)
        survivorship = self._survivorship_warning(spec, ctx, result.start)
        mcap = self._mcap_warning(ctx)
        labels = self._label_warning(spec, ctx)
        prov_w = list(getattr(self.provider, "warnings", []) or [])[n_provider_warnings:]
        all_w = _dedupe(([survivorship] if survivorship else []) + ([mcap] if mcap else [])
                        + ([labels] if labels else []) + ctx.warnings
                        + [f"provider {self.provider_name}: {w}" for w in prov_w])
        if len(all_w) > _MAX_WARNINGS:
            all_w = all_w[:_MAX_WARNINGS] + [f"... {len(all_w) - _MAX_WARNINGS} more warnings omitted"]
        result.warnings = all_w
        result.finished_at = datetime.now(timezone.utc)
        self._say(f"Finished '{spec.name}': run {result.run_id}")
        return result

    def _mcap_warning(self, ctx: _Context) -> str | None:
        if not ctx.mcap_needed or not ctx._mcap_cache:
            return None
        if ctx.mcap_mode != "end_scaled":
            if not ctx.mcap_current_shares:
                return None
            names = sorted({n for v in ctx.mcap_current_shares.values() for n in v})
            dates = sorted(ctx.mcap_current_shares)
            return (
                f"MARKET CAPS NOT POINT-IN-TIME for {len(names)} name(s) ({_names(names)}): provider '{self.provider_name}' "
                f"had no point-in-time share count for them, so their market cap on {len(dates)} snapshot date(s) "
                f"({dates[0]} to {dates[-1]}) uses a later (current) share count. Buybacks and issuance after those dates "
                "feed size, value weights and valuation ratios (book-to-market, FCF / earnings yield, EV) at earlier "
                "dates: look-ahead bias."
            )
        return (
            f"MARKET CAPS NOT POINT-IN-TIME: provider '{self.provider_name}' declares no point-in-time market caps, so the "
            f"market cap at each date before {ctx.end} is the end-date market cap x the adjusted-price ratio. Share counts "
            f"known only on {ctx.end} (buybacks, issuance) and later dividends feed every earlier date: size, value weights "
            "and valuation ratios (book-to-market, FCF / earnings yield, EV) carry look-ahead bias."
        )

    def _spec_errors(self, spec: StrategySpec) -> list[str]:
        from aitrading.strategy.nl import spec_errors  # local import: nl imports the template library

        return spec_errors(spec, self.catalog)

    def _survivorship_warning(self, spec: StrategySpec, ctx: _Context, first: date) -> str | None:
        if spec.kind == "time_series" or bool(getattr(self.provider, "point_in_time_universe", False)):
            return None
        return (
            f"SURVIVORSHIP BIAS: the universe is the provider's constituents as of {ctx.end} ({len(ctx.tickers)} names) "
            f"applied to every date back to {first} instead of the index membership on each date. Companies that were delisted, "
            f"acquired or dropped before {ctx.end} are missing, which usually flatters backtested returns "
            "(point-in-time index membership needs institutional data)."
        )

    def _labels_static(self) -> bool:
        """True when the provider's sector / industry / exchange labels never change over time (the
        synthetic market), so the END snapshot's labels are valid at every date."""
        return self.provider_name == "synthetic"

    def _label_warning(self, spec: StrategySpec, ctx: _Context) -> str | None:
        """Universe labels are END-date values applied to every date: say so when they drive the run."""
        if spec.kind == "time_series" or self._labels_static():
            return None
        used = sorted(ctx.needed & _LABEL_FEATURES)
        if spec.kind == "factor_model" and F.EXCHANGE in ctx.universe.columns and ctx.universe[F.EXCHANGE].notna().any():
            used = sorted({*used, F.EXCHANGE})
        if not used:
            return None
        return (
            f"END-DATE LABELS: {' / '.join(used)} come from the provider's universe as of {ctx.end} and are applied to every "
            "rebalance date; a name reclassified or relisted later is filtered, sector-neutralised or given NYSE "
            "breakpoints by its later label at earlier dates (e.g. GOOGL and META were Information Technology and DIS "
            "Consumer Discretionary until the GICS Communication Services sector was created in September 2018)."
        )

    def provider_fingerprint(self) -> dict[str, str]:
        """What identifies the provider's data beyond its name: ``provider.fingerprint`` (a string or a
        callable returning one) when the provider defines it, else its class and its configuration
        attributes (``seed``, ``n_tickers``, ``start``, ``end``, ``benchmark``, and a hash of its
        ``tickers`` list), so two differently configured providers never share a run id."""
        p = self.provider
        try:
            fp = getattr(p, "fingerprint", None)
            fp = fp() if callable(fp) else fp
        except Exception:  # noqa: BLE001 - a broken fingerprint falls back to the attributes
            fp = None
        if fp is not None:
            return {"fingerprint": str(fp)}
        out = {"class": f"{type(p).__module__}.{type(p).__qualname__}"}
        for attr in ("seed", "n_tickers", "start", "end", "benchmark"):
            try:
                v = getattr(p, attr, None)
            except Exception:  # noqa: BLE001
                v = None
            if isinstance(v, (bool, int, float, str, date)):
                out[attr] = str(v)
        try:
            tickers = getattr(p, "tickers", None)
        except Exception:  # noqa: BLE001
            tickers = None
        if isinstance(tickers, (list, tuple)):
            out["tickers"] = _sha(sorted(str(t) for t in tickers))[:16]
        return out

    def _run_id(self, spec: StrategySpec, start: date, end: date, label: str | None, *,
                tickers: list[str] | tuple[str, ...] = ()) -> str:
        """``<spec name>-<hash>[-<label>]``; the hash covers the spec, the provider (name and
        :meth:`provider_fingerprint`), the universe (sorted ``tickers``), the window and the execution lag."""
        digest = _sha({"spec": spec.model_dump(mode="json"), "provider": self.provider_name,
                       "provider_config": self.provider_fingerprint(),
                       "universe": _sha(sorted(str(t) for t in tickers))[:16],
                       "start": start.isoformat(), "end": end.isoformat(), "execution_lag": self.execution_lag})[:10]
        rid = f"{_slug(spec.name)}-{digest}"
        return f"{rid}-{_slug(label, 32)}" if label else rid

    @staticmethod
    def _note_per_share_ratios(ctx: _Context, t: pd.Timestamp, frame: pd.DataFrame) -> None:
        """Record the per-share price ratios the run uses that have values at ``t`` on adjusted past prices
        (the dates at which the price floor is suspended): their denominator is restated for later splits
        and dividends. The free provider's estimates are blank at past dates and the synthetic market
        is as traded, so this only fires for providers serving historical estimates (e.g. BQL)."""
        if ctx.prices_as_traded or (ctx.price_floor_from is not None and pd.Timestamp(t).date() >= ctx.price_floor_from):
            return
        for f in PER_SHARE_PRICE_RATIOS:
            if f in ctx.needed and f in frame.columns and pd.to_numeric(frame[f], errors="coerce").notna().any():
                ctx.per_share_biased.setdefault(f, []).append(pd.Timestamp(t))

    def _per_share_ratio_warning(self, ctx: _Context, dates: list[pd.Timestamp]) -> None:
        kept = set(dates)
        biased = {f: sorted(set(ts) & kept) for f, ts in ctx.per_share_biased.items()}
        biased = {f: ts for f, ts in biased.items() if ts}
        if not biased:
            return
        first = min(ts[0] for ts in biased.values())
        last = max(ts[-1] for ts in biased.values())
        one = len(biased) == 1
        ctx.warnings.append(
            f"{', '.join(f for f in PER_SHARE_PRICE_RATIOS if f in biased)} at rebalance dates {_fmt(first)} to {_fmt(last)} "
            f"{'divides' if one else 'divide'} per-share vendor values (consensus EPS / target price) by a close that provider "
            f"'{self.provider_name}' has adjusted for splits and dividends after those dates, so "
            f"{'it is' if one else 'they are'} not point-in-time (upside and earnings yield read high for later dividend "
            "payers and splitters)"
        )

    def _price_floor_warning(self, ctx: _Context, dates: list[pd.Timestamp]) -> None:
        kept = set(dates)
        sus = sorted(t for t in set(ctx.floor_suspended) if t in kept)
        if not sus:
            return
        floor = ctx.spec.universe.min_price
        ctx.warnings.append(
            f"price floor (universe min_price ${floor:g}) not applied at {len(sus)} rebalance date(s) ({_fmt(sus[0])} to "
            f"{_fmt(sus[-1])}): provider '{self.provider_name}' serves split- and dividend-adjusted prices, so an adjusted "
            f"close under ${floor:g} does not mean the stock traded under ${floor:g} then (later splits and dividends lower "
            "earlier adjusted prices) and the floor would drop later winners using splits that had not happened yet. It "
            f"applies from {ctx.price_floor_from} (current prices); the liquidity floor and every filter still apply (the "
            "liquidity floor is split-invariant, but dollar volume from dividend-adjusted closes understates later "
            "dividend payers at past dates)."
        )

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

        # leading dates without enough history: the track record starts at the first portfolio. Not when the
        # gap comes from snapshot data without history (it would leave only the last dates: report cash instead)
        snapshot_gap = not self._snapshots_point_in_time(ctx) and bool(snapshot_only_features(spec.features()))
        if any(formable):
            k = formable.index(True)
            if k > 0 and not snapshot_gap and len(dates) - k >= 2:
                ctx.warnings.append(
                    f"no portfolio could be formed at the first {k} rebalance date(s) ({_fmt(dates[0])} to {_fmt(dates[k - 1])}): "
                    f"the features did not have enough history yet; the backtest starts on {_fmt(dates[k])}"
                )
                for t in dates[:k]:
                    targets.pop(t, None)
                    signals.pop(t, None)
                dates = dates[k:]
                n_eligible = n_eligible[k:]
        else:
            ctx.warnings.append(
                "no portfolio could be formed at any rebalance date (no name had data for the signal / every filter); "
                "the strategy held cash throughout"
            )
        if spec.kind == "screen" and targets and all(len(w) == 0 for w in targets.values()):
            ctx.warnings.append("no name passed the screen at any rebalance date: the strategy held cash throughout")
        if spec.kind == "time_series":
            self._ts_unavailable_warnings(ctx, dates)
        self._price_floor_warning(ctx, dates)
        self._per_share_ratio_warning(ctx, dates)
        ctx.eligible_counts = n_eligible

        self._say(f"{spec.name}: simulating {len(targets)} rebalance(s)")
        try:
            sim = simulate(targets, ctx.close, costs_bps=spec.costs_bps, execution_lag=self.execution_lag,
                           start=ctx.start, end=ctx.end, delisting_return=spec.delisting_return)
        except ValueError as exc:
            raise ValueError(f"'{spec.name}' cannot be simulated from {ctx.start} to {ctx.end} "
                             f"({len(targets)} rebalance date(s)): {exc}") from exc
        ctx.warnings.extend(_compress_dated(sim.warnings))
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
                                   signals=signals, simulation=sim, daily_returns=strat, universe=list(ctx.tickers))
        out_dates, returns = self._monthly_table(monthly)
        return BacktestResult(
            run_id=self._run_id(spec, ctx.start, ctx.end, label, tickers=ctx.tickers),
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
        """Daily risk-free interest on cash: max(0, 1 - sum of the executed weights) x RF, earned from the
        day after each execution (drift between executions ignored). A net-short book holds its capital
        plus the short-sale proceeds in cash (e.g. 2 for a -1 short), which earn RF (full rebate), so a
        short position's excess return is -(r - rf). Leverage (sum > 1) is not financed (it never arises:
        long books sum to at most 1)."""
        if rf_daily is None or not len(index) or not sim.weights_history:
            return pd.Series(0.0, index=index)
        cash = pd.Series({pd.Timestamp(d): max(0.0, 1.0 - float(w.sum())) for d, w in sim.weights_history.items()})
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
        snap_only = snapshot_only_features(used)
        for f in used:
            vals = cov.get(f, [])
            mean = float(np.mean(vals)) if vals else 0.0
            snap = snap_only.get(f, [])
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

    def _mcap_usage(self, ctx: _Context) -> DataUsage | None:
        """The ``market_cap`` provenance entry, when market caps fed the run (``point_in_time=False``
        when any date used the END-cap estimate, which caps the interpreter's verdict)."""
        if not ctx.mcap_needed or not ctx._mcap_cache:
            return None
        name = self.provider_name
        n = len(ctx._mcap_snapshots)
        current = sorted({x for v in ctx.mcap_current_shares.values() for x in v})
        current_note = ""
        if current:
            dates = sorted(ctx.mcap_current_shares)
            current_note = (f"NOT point-in-time for {len(current)} name(s) ({_names(current)}) on {len(dates)} snapshot "
                            f"date(s) ({dates[0]} to {dates[-1]}): the provider priced them with a later (current) share "
                            "count (look-ahead in size, value weights and valuation ratios); ")
        if ctx.mcap_mode == "snapshot":
            return DataUsage(dataset="market_cap", source=f"{name}: universe snapshot at each date",
                             coverage=f"{len(ctx._mcap_cache)} date(s)", point_in_time=not current,
                             notes=current_note + "market cap as of each date from the provider")
        if ctx.mcap_mode == "end_scaled":
            return DataUsage(dataset="market_cap", source=f"{name}: end-date market cap x adjusted-price ratio",
                             coverage=f"{len(ctx._mcap_cache)} date(s)", point_in_time=False,
                             notes=(f"the provider declares no point-in-time market caps: the share count of {ctx.end} and "
                                    "later dividends are used at every earlier date (look-ahead in size, value weights and "
                                    "valuation ratios)"))
        fb = sorted(set(ctx.mcap_fallback))
        notes = ("provider snapshots on the last session of each June and December (and on dates before the first one), "
                 "rolled forward to each date with the adjusted-price ratio: share changes between snapshots are not "
                 "reflected, dividends since the snapshot count as reinvested, and a name listed after a snapshot has no "
                 "market cap until the next one")
        if fb:
            notes = (f"NOT point-in-time on {len(fb)} snapshot date(s) ({fb[0]} to {fb[-1]}): the provider gave no snapshot, "
                     "so the end-date market cap x adjusted-price ratio was used; otherwise " + notes)
        pit = not fb and not current
        return DataUsage(dataset="market_cap",
                         source=f"{name}: universe snapshots" + (" (point-in-time)" if pit else ""),
                         coverage=f"{n} snapshot date(s) for {len(ctx._mcap_cache)} date(s)", point_in_time=pit,
                         notes=current_note + notes)

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
                     f"constituents as of {ctx.end} applied to every date (today's survivors): survivorship bias")
            if not self._labels_static():
                notes += (f"; sector, industry, exchange and security-type labels are the values as of {ctx.end}, applied "
                          "to every date (later reclassifications are used at earlier dates)")
            sus = sorted(set(ctx.floor_suspended))
            if sus and spec.kind in ("cross_sectional", "screen"):
                notes += (f"; the ${spec.universe.min_price:g} price floor is not applied before {ctx.price_floor_from} "
                          "(adjusted prices are restated for later splits and dividends)")
            coverage = f"{len(ctx.tickers)} names as of {ctx.end}"
            if ctx.eligible_counts:
                counts = np.asarray(ctx.eligible_counts)
                coverage += (f"; eligible per rebalance after filters: median {int(np.median(counts))} "
                             f"(min {int(counts.min())}, max {int(counts.max())})")
            elif spec.kind == "factor_model":
                coverage += " (no price / liquidity floors: every name with data enters the sorts)"
            out.append(DataUsage(dataset="universe", source=name, coverage=coverage, point_in_time=True, notes=notes))
        mcap_usage = self._mcap_usage(ctx)
        if mcap_usage is not None:
            out.append(mcap_usage)
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
        snap_only = snapshot_only_features(feats)
        for ds in _SNAPSHOT_DATASETS:
            used = sorted(f for f in feats if ds in FEATURE_DATASETS.get(f, ()))
            if not used:
                continue
            vals = [np.mean(cov[f]) for f in used if cov.get(f)]
            cover = f"{np.mean(vals) * 100:.0f}% average coverage of {', '.join(used)}" if vals else "none"
            current_only = [f for f in used if ds in snap_only.get(f, ())]
            if snapshot_pit:
                pit_ds, notes = True, ""
            elif current_only:
                pit_ds, notes = False, ("current snapshot only: no history in the free edition (NaN at historical dates) for "
                                        + ", ".join(current_only))
            else:
                pit_ds, notes = True, (f"only {', '.join(used)}, which is point-in-time: dated earnings releases / filings "
                                       "(the snapshot-only estimate fields are not used)")
            out.append(DataUsage(dataset=ds, source=name, coverage=cover, point_in_time=pit_ds, notes=notes))
        if ctx.benchmark is not None:
            b = ctx.benchmark[ctx.benchmark.index >= pd.Timestamp(ctx.start)]
            if len(b):
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
        idx = ctx.sessions
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

    def _monthly_panel(self, ctx: _Context, days: pd.DatetimeIndex, labels: pd.DatetimeIndex,
                       delisting_return: float) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, pd.Timestamp]]]:
        """(month-end closes, monthly returns, names that stopped trading) on the calendar month-end labels.

        A name that stopped trading has no later monthly price (its last close is not carried forward).
        When ``delisting_return`` is not 0 it is booked on the name's first missing session, as
        ``simulate`` does: compounded into that month's return, or as the next month's return when the
        last close was the month's last session."""
        raw = ctx.close.reindex(columns=ctx.tickers)
        close_m = pd.DataFrame(raw.ffill().loc[days].to_numpy(), index=labels, columns=ctx.tickers)
        last_valid = raw.apply(lambda s: s.last_valid_index())
        sessions = ctx.sessions
        data_last = sessions[-1] if len(sessions) else None
        dead: list[tuple[str, pd.Timestamp]] = []
        for tk, lv in last_valid.items():
            if lv is None or pd.isna(lv):
                close_m[tk] = np.nan
                continue
            lv = pd.Timestamp(lv)
            close_m.loc[labels > (lv + pd.offsets.MonthEnd(0)), tk] = np.nan
            if data_last is not None and lv < data_last:
                dead.append((str(tk), lv))
        returns = close_m / close_m.shift(1) - 1.0
        if delisting_return != 0.0:
            for tk, lv in dead:
                nxt = sessions[int(sessions.searchsorted(lv, side="right"))]
                lab_lv = (lv + pd.offsets.MonthEnd(0)).normalize()
                lab_nx = (nxt + pd.offsets.MonthEnd(0)).normalize()
                if lab_nx == lab_lv:
                    if lab_lv in returns.index and np.isfinite(returns.at[lab_lv, tk]):
                        returns.at[lab_lv, tk] = (1.0 + returns.at[lab_lv, tk]) * (1.0 + delisting_return) - 1.0
                elif lab_nx in returns.index:
                    returns.at[lab_nx, tk] = delisting_return
        return close_m, returns, dead

    def _factor_trades(self, ctx: _Context, months: pd.DatetimeIndex, labels: pd.DatetimeIndex,
                       days: pd.DatetimeIndex, returns: pd.DataFrame) -> tuple[dict[pd.Timestamp, pd.Series], pd.Series]:
        """Factor-mimicking weights at each month-end session from the one before ``months[0]`` to the one
        before ``months[-1]``, and the traded notional sum |w_target - w_drifted| of the trade at the end
        of each month, keyed by the NEXT month's label (the month whose return bears its cost). The book
        is built from cash at the first trade; between trades weights drift with the monthly returns
        (names without a return - stopped trading - leave the book without a trade)."""
        pos = {lab: k for k, lab in enumerate(labels)}
        k0, k1 = pos[months[0]] - 1, pos[months[-1]] - 1
        weights: dict[pd.Timestamp, pd.Series] = {}
        traded: dict[pd.Timestamp, float] = {}
        prev = pd.Series(dtype=float)
        for k in range(max(k0, 0), k1 + 1):
            w = self._factor_weights_at(ctx, days[k])
            weights[days[k]] = w
            drift = pd.Series(dtype=float)
            if len(prev):
                r = returns.loc[labels[k]].reindex(prev.index)
                ok = r.notna().to_numpy()
                nav = 1.0 + float((prev[ok] * r[ok]).sum())
                if nav > 0:
                    drift = prev[ok] * (1.0 + r[ok]) / nav
            t_al, d_al = w.align(drift, fill_value=0.0)
            traded[labels[k + 1]] = float(np.abs(t_al.to_numpy() - d_al.to_numpy()).sum())
            prev = w
        return weights, pd.Series(traded, dtype=float)

    def _factor_model_backtest(self, spec: StrategySpec, ctx: _Context, rf_daily: pd.Series | None, rf_source: str,
                               label: str | None, started: datetime) -> BacktestResult:
        model = spec.factor_model
        assert model is not None
        days = self._month_days(ctx)
        if len(days) < 3:
            raise ValueError("factor models need at least three months of prices")
        if spec.rebalance != "monthly":
            ctx.warnings.append(
                f"rebalance '{spec.rebalance}' does not apply to a factor model: the factor-mimicking portfolios are "
                "re-weighted at every month-end (sorts re-formed each June, the Fama-French convention), so the run "
                "is reported, and its costs counted, as monthly"
            )
        labels = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in days]).normalize()
        close_m, returns, dead = self._monthly_panel(ctx, days, labels, float(spec.delisting_return))
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
        # a last month the data does not cover to its end is not a monthly return: leave it out
        last_session = ctx.sessions[-1] if len(ctx.sessions) else pd.Timestamp(ctx.end)
        last_label = (last_session + pd.offsets.MonthEnd(0)).normalize()
        if len(fac) and fac.index[-1] == last_label and last_session < last_label - pd.Timedelta(days=3):
            fac = fac.iloc[:-1]
            ctx.warnings.append(
                f"the last month ({last_label:%Y-%m}) is incomplete (the data ends on {_fmt(last_session)}): it is left out "
                "of the factor returns, the statistics and the attribution"
            )
        if fac.empty or fac.notna().sum().max() < 2:
            raise ValueError(f"could not construct {model} factors in {ctx.start} to {ctx.end} (too little data)")
        nonmarket = [c for c in cols if c != "Mkt-RF"] or ["Mkt-RF"]
        gross = fac[nonmarket].mean(axis=1, skipna=False).rename("strategy")

        # trading the headline portfolio: costs on the traded notional of the factor-mimicking weights
        costs = pd.Series(0.0, index=gross.index)
        turnover: pd.Series | None = None
        trade_weights: dict[pd.Timestamp, pd.Series] = {}
        first = gross.first_valid_index()
        if first is not None:
            self._say(f"{spec.name}: factor-mimicking portfolio turnover")
            try:
                trade_weights, traded = self._factor_trades(ctx, gross.index[gross.index >= first], labels, days, returns)
                costs = (traded * float(spec.costs_bps) / 1e4).reindex(gross.index).fillna(0.0)
                turnover = (traded / 2.0).reindex(gross.index).dropna()
                cost_note = (f"net of {spec.costs_bps:g} bps one-way costs on the traded notional of the factor-mimicking "
                             "portfolio (re-weighted at every month-end, re-formed each June"
                             + ("; momentum re-formed monthly" if model == "carhart4" else "") + ")")
            except Exception as exc:  # noqa: BLE001 - report gross returns rather than fail the run
                cost_note = (f"GROSS of trading costs: the factor-mimicking turnover could not be computed "
                             f"({type(exc).__name__}: {str(exc)[:160]})")
        else:
            cost_note = "gross of trading costs (no month with every factor)"
        strategy = (gross - costs).rename("strategy")

        bench_m = None
        if ctx.benchmark is not None:
            b = ctx.benchmark
            bd = (b / b.shift(1) - 1.0).dropna()
            first_lab = fac.index[0]
            bd = bd[(bd.index > first_lab - pd.offsets.MonthEnd(1)) & (bd.index <= pd.Timestamp(ctx.end))]
            if len(bd) >= 2:
                bench_m = compound(bd, "M")
                bench_m = bench_m[bench_m.index <= fac.index[-1]]

        stats: dict[str, PerformanceStats] = {}
        for c in cols:
            self._add_stats(stats, c, fac[c], None, None, None, ctx.warnings, periods_per_year=12.0)
        self._add_stats(stats, "strategy", strategy, None, bench_m, turnover, ctx.warnings, periods_per_year=12.0)
        if bench_m is not None and len(bench_m) >= 2:
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
        what = ("non-market factors (" + ", ".join(nonmarket) + ")" if nonmarket != ["Mkt-RF"] else "market factor (Mkt-RF)")
        ctx.warnings.append(
            f"factor model: 'strategy' is the equal-weighted average of the constructed {what}, {cost_note}; the factor "
            "series themselves are gross, like the official ones. The factors are value-weighted 2x3 sorts formed each June "
            "(momentum monthly), formed and traded at the month-end close (the Fama-French convention: the "
            f"{self.execution_lag}-session execution lag of the other strategy kinds is not applied)"
        )
        in_window_dead = [(tk, lv) for tk, lv in dead if pd.Timestamp(ctx.start) <= lv]
        if in_window_dead:
            names = _names([f"{tk} ({_fmt(lv)})" for tk, lv in in_window_dead])
            if spec.delisting_return != 0.0:
                ctx.warnings.append(f"{len(in_window_dead)} name(s) stopped trading inside the window ({names}): a "
                                    f"{spec.delisting_return * 100:+.1f}% delisting return is booked after the last price")
            else:
                ctx.warnings.append(f"{len(in_window_dead)} name(s) stopped trading inside the window ({names}): no delisting "
                                    "return is booked after the last price (0%; optimistic if they were delisted for "
                                    "performance reasons)")
        details = fac.assign(strategy_gross=gross, costs=costs, strategy=strategy)
        self.last_run = RunDetails(spec=spec, start=fac.index[0].date(), end=ctx.end,
                                   rebalance_dates=sorted(trade_weights) if trade_weights else list(days),
                                   target_weights=trade_weights, factor_returns=details, universe=list(ctx.tickers))
        return BacktestResult(
            run_id=self._run_id(spec, ctx.start, ctx.end, label, tickers=ctx.tickers),
            idea=spec.idea,
            spec=spec.model_dump(mode="json"),
            provider=self.provider_name,
            llm="none",
            start=max(ctx.start, (fac.index[0] - pd.offsets.MonthBegin(1)).date()),
            end=ctx.end,
            rebalance="monthly",  # what was simulated, whatever spec.rebalance says (warning above)
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

    def _june_sorts(self, ctx: _Context, j: pd.Timestamp) -> dict[str, pd.Series]:
        """2x3 portfolio labels of the June formation at session ``j`` (B/M; ff5 also OP and INV),
        exactly as ``construct_factors`` forms them: size = cap at ``j``, B/M = book equity public at
        ``j`` / cap at the last December session before it. Cached per context."""
        key = ("june", pd.Timestamp(j))
        if key in ctx._formations:
            return ctx._formations[key]
        model = ctx.spec.factor_model
        exchange = ctx.universe[F.EXCHANGE] if F.EXCHANGE in ctx.universe.columns else None
        days = self._month_days(ctx, j)
        size = self._mcap_at(ctx, j)
        dec_days = [d for d in days if d.year == j.year - 1 and d.month == 12]
        me_dec = self._mcap_at(ctx, dec_days[-1]) if dec_days else size
        be = self._book_equity(ctx, j)
        with np.errstate(invalid="ignore", divide="ignore"):
            bm = (be / me_dec).where((be > 0) & (me_dec > 0))
        out = {"bm": two_by_three_sort(size, bm, exchange=exchange)[0].dropna()}
        if model == "ff5":
            op, inv = self._op_inv(ctx, j)
            with np.errstate(invalid="ignore"):
                op = op.where(be.reindex(op.index) > 0)
            out["op"] = two_by_three_sort(size, op, exchange=exchange, labels=("W", "N", "R"))[0].dropna()
            out["inv"] = two_by_three_sort(size, inv, exchange=exchange, labels=("C", "N", "A"))[0].dropna()
        ctx._formations[key] = out
        return out

    def _factor_weights_at(self, ctx: _Context, as_of: pd.Timestamp) -> pd.Series:
        """Factor-mimicking weights of the factor model's headline portfolio at ``as_of`` (module docstring).

        The annual sorts are the latest June formation whose session (the last trading session of June,
        the date ``construct_factors`` forms on) is on or before ``as_of`` - on that session itself the
        NEW formation is used. Within each 2x3 portfolio names are weighted by their market cap at the
        last session on or before ``as_of``."""
        model = ctx.spec.factor_model
        assert model is not None
        days = self._month_days(ctx, as_of)
        if not len(days):
            return pd.Series(dtype=float)
        now = days[-1]  # last session on or before as_of
        cap_now = self._mcap_at(ctx, now)
        pos = int(ctx.close.index.searchsorted(now, side="right")) - 1
        alive = pd.to_numeric(ctx.close.iloc[pos].reindex(ctx.tickers), errors="coerce") if pos >= 0 \
            else pd.Series(np.nan, index=ctx.tickers)
        cap_now = cap_now.where(alive.reindex(cap_now.index).notna() & (alive.reindex(cap_now.index) > 0))
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
        june_days = [d for d in days if d.month == 6 and self._is_month_end_session(ctx, d)]
        if june_days:
            sorts = self._june_sorts(ctx, june_days[-1])
            p = ports(sorts["bm"], ("L", "M", "H"))
            smb_bm = combo([(1 / 3, p["S/L"]), (1 / 3, p["S/M"]), (1 / 3, p["S/H"]),
                            (-1 / 3, p["B/L"]), (-1 / 3, p["B/M"]), (-1 / 3, p["B/H"])])
            hml = combo([(0.5, p["S/H"]), (0.5, p["B/H"]), (-0.5, p["S/L"]), (-0.5, p["B/L"])])
            if model == "ff5":
                po, pi = ports(sorts["op"], ("W", "N", "R")), ports(sorts["inv"], ("C", "N", "A"))
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
            px = ctx.close_ffill()
            p1, p12 = px.loc[days[-2]], px.loc[days[-13]]
            with np.errstate(invalid="ignore", divide="ignore"):
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
        return self.prices_at(req, [as_of]).loc[pd.Timestamp(as_of)].rename(None)

    def prices_at(self, tickers: list[str], dates: list[date]) -> pd.DataFrame:
        """Adjusted close on or before each of ``dates`` (within 31 days) per ticker, all taken from ONE
        price download over ``(min(dates) - 31 days, max(dates)]``; NaN when unavailable.

        Rows: the distinct ``dates`` (as Timestamps, ascending); columns: ``tickers`` (deduplicated, in order).
        Because every row comes from the same download, all of them share one adjustment vintage:
        paper trading compares a re-read past price with today's price to detect splits and
        dividends, which a past window served from a provider's cache (adjusted before the event)
        would hide.
        """
        req = list(dict.fromkeys(str(t) for t in tickers))
        days = sorted({_as_date(d) for d in dates})
        index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
        out = pd.DataFrame(np.nan, index=index, columns=req, dtype=float)
        if not req or not days:
            return out
        try:
            panel = self.provider.get_price_history(req, days[0] - timedelta(days=31), days[-1])
            close = panel.close.copy()
            close.columns = [str(c) for c in close.columns]
            close = close.loc[:, ~close.columns.duplicated(keep="last")]
            close = close.apply(pd.to_numeric, errors="coerce").sort_index()
        except Exception:  # noqa: BLE001 - unavailable prices are NaN
            return out
        for d, ts in zip(days, index):
            rows = close.loc[(close.index >= pd.Timestamp(d - timedelta(days=31))) & (close.index <= ts)]
            if len(rows):
                out.loc[ts] = rows.ffill().iloc[-1].reindex(req).astype(float)
        return out.where(np.isfinite(out.to_numpy()) & (out > 0))
