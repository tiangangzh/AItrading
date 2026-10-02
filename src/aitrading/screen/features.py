"""Feature engine: provider datasets -> one row per ticker with every catalog feature.

``FeatureEngine.build`` fetches only the datasets the requested features need, runs the technical,
fundamental and positioning engines and adds the reference features, returning a frame the screen
engine (``aitrading.screen.engine``) and the ranker consume directly.

Conventions
-----------
* Index ``ticker`` in universe order (duplicates dropped). Columns: the raw reference columns
  ``name``, ``country`` and ``security_type``, then every catalog feature in catalog order. Every
  catalog column is always present; a feature that was not computed, or whose dataset is
  unavailable, is NaN.
* Datasets: prices (+ benchmark for relative strength and beta) are fetched over the
  ``LOOKBACK_CALENDAR_DAYS`` calendar days ending on as_of (enough for 252-session windows plus the
  12-1 momentum lag and the 200-day slope). Fundamentals, estimates, short interest and options are
  point-in-time snapshots as of as_of. A dataset is fetched only when a requested feature needs it.
  Once a dataset is fetched, every feature computable from it is filled in, requested or not.
* A dataset the provider does not declare in ``capabilities`` is skipped with a warning naming the
  missing capability and the requested features left NaN. A provider error on an optional dataset
  (benchmark, estimates, short interest, options) degrades the same way; an error on prices or
  fundamentals propagates (as ``ProviderError``), since a screen without them would silently drop
  every name.
* Cross-sectional features (``CROSS_SECTIONAL_FEATURES``: ``return_6m_percentile``) are relative to
  the tickers passed in, so build them on the screening universe, not on a pre-filtered subset.
* ``coverage`` is the non-NaN fraction over the frame's rows for each requested catalog feature
  (0.0 for an empty universe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from aitrading.core import fields
from aitrading.data.base import Capability, MarketDataProvider, PricePanel, ProviderError
from aitrading.fundamental.features import compute_fundamental_features
from aitrading.positioning.features import compute_positioning_features
from aitrading.screen.catalog import (
    FUNDAMENTAL_FEATURES,
    POSITIONING_FEATURES,
    TECHNICAL_FEATURES,
    FeatureCatalog,
    default_catalog,
)
from aitrading.technical.features import average_volume_shares, compute_technical_features

__all__ = [
    "LOOKBACK_CALENDAR_DAYS",
    "RAW_REFERENCE_COLUMNS",
    "CROSS_SECTIONAL_FEATURES",
    "FEATURE_DATASETS",
    "FeatureFrame",
    "FeatureEngine",
]

LOOKBACK_CALENDAR_DAYS = 420
RAW_REFERENCE_COLUMNS = [fields.NAME, fields.COUNTRY, fields.SECURITY_TYPE]
CROSS_SECTIONAL_FEATURES = frozenset({"return_6m_percentile"})

PRICES, BENCHMARK, FUNDAMENTALS, ESTIMATES, SHORT_INTEREST, OPTIONS = (
    "prices", "benchmark", "fundamentals", "estimates", "short_interest", "options",
)
_CAPABILITY = {
    PRICES: Capability.PRICES,
    BENCHMARK: Capability.PRICES,
    FUNDAMENTALS: Capability.FUNDAMENTALS,
    ESTIMATES: Capability.ESTIMATES,
    SHORT_INTEREST: Capability.SHORT_INTEREST,
    OPTIONS: Capability.OPTIONS,
}
_REQUIRED = frozenset({PRICES, FUNDAMENTALS})  # provider errors on these propagate

_FUNDAMENTAL_DEPS: dict[str, frozenset[str]] = {
    # market cap comes from the universe; price from the price panel
    "pe_ntm": frozenset({ESTIMATES, PRICES}),
    "earnings_yield_ntm_pct": frozenset({ESTIMATES, PRICES}),
    "target_price_upside_pct": frozenset({ESTIMATES, PRICES}),
    "revenue_growth_ntm_est_pct": frozenset({ESTIMATES, FUNDAMENTALS}),
    "eps_growth_ntm_est_pct": frozenset({ESTIMATES}),
    "eps_revision_3m_pct": frozenset({ESTIMATES}),
    "revenue_revision_3m_pct": frozenset({ESTIMATES}),
    "num_analysts": frozenset({ESTIMATES}),
    "last_eps_surprise_pct": frozenset({ESTIMATES}),
    "days_since_last_earnings": frozenset({ESTIMATES, FUNDAMENTALS}),  # falls back to the report date
    "days_to_next_earnings": frozenset({ESTIMATES}),
}
_POSITIONING_DEPS: dict[str, frozenset[str]] = {
    "iv_30d_pct": frozenset({OPTIONS}),
    "iv_rank_1y": frozenset({OPTIONS}),
    "iv_to_realized_vol_ratio": frozenset({OPTIONS, PRICES}),
    "put_call_volume_ratio": frozenset({OPTIONS}),
    "put_call_oi_ratio": frozenset({OPTIONS}),
    "short_interest_pct_float": frozenset({SHORT_INTEREST}),
    "days_to_cover": frozenset({SHORT_INTEREST, PRICES}),
    "short_interest_change_1m_pct": frozenset({SHORT_INTEREST}),
}
_BENCHMARK_FEATURES = frozenset({"rel_strength_3m_pp", "rel_strength_6m_pp", "rel_strength_12m_pp", "beta_1y"})


def _deps(name: str) -> frozenset[str]:
    if name in TECHNICAL_FEATURES:
        return frozenset({PRICES, BENCHMARK}) if name in _BENCHMARK_FEATURES else frozenset({PRICES})
    if name in FUNDAMENTAL_FEATURES:
        return _FUNDAMENTAL_DEPS.get(name, frozenset({FUNDAMENTALS}))
    if name in POSITIONING_FEATURES:
        return _POSITIONING_DEPS[name]
    return frozenset()  # reference features come from the universe frame


FEATURE_DATASETS: dict[str, frozenset[str]] = {f.name: _deps(f.name) for f in default_catalog()}
"""Catalog feature -> datasets it is computed from (empty for reference features)."""


@dataclass
class FeatureFrame:
    frame: pd.DataFrame  # index 'ticker'; RAW_REFERENCE_COLUMNS + every catalog feature
    coverage: dict[str, float]  # requested catalog feature -> non-NaN fraction of rows
    warnings: list[str] = field(default_factory=list)
    universe: pd.DataFrame = field(default_factory=pd.DataFrame)  # the universe rows the frame covers


def _as_date(d: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(d, str):
        d = pd.Timestamp(d)
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    return d


def _by_ticker(data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Provider frame keyed by ticker (index or ``ticker`` column), last duplicate wins."""
    if data is None:
        return None
    if fields.TICKER in data.columns and data.index.name != fields.TICKER:
        data = data.set_index(fields.TICKER)
    return data[~data.index.duplicated(keep="last")]


class FeatureEngine:
    """Computes catalog features for a universe from a ``MarketDataProvider`` (see module docstring)."""

    def __init__(self, provider: MarketDataProvider, catalog: FeatureCatalog | None = None):
        self.provider = provider
        self.catalog = catalog or default_catalog()

    # -- planning ----------------------------------------------------------------------------

    def _has(self, dataset: str) -> bool:
        caps = getattr(self.provider, "capabilities", None)
        return caps is None or _CAPABILITY[dataset] in caps

    def required_datasets(self, features: set[str] | None = None) -> set[str]:
        """Datasets needed for ``features`` (None = every catalog feature)."""
        names = self.catalog.names() if features is None else [f for f in features if f in FEATURE_DATASETS]
        out: set[str] = set()
        for f in names:
            out |= FEATURE_DATASETS.get(f, frozenset())
        return out

    # -- build -------------------------------------------------------------------------------

    def build(self, universe: pd.DataFrame, as_of: date, features: set[str] | None = None) -> FeatureFrame:
        """Feature frame for every ticker in ``universe`` as of ``as_of``.

        ``features`` selects what must be computed (None = all catalog features); unknown names are
        ignored with a warning. Raises ``ProviderError`` when prices or fundamentals are needed and
        the provider fails to return them.
        """
        as_of = _as_date(as_of)
        warnings: list[str] = []
        universe = universe[~universe.index.duplicated(keep="first")]
        tickers = [str(t) for t in universe.index]
        index = pd.Index(tickers, name=fields.TICKER)
        universe = universe.set_axis(index)

        if features is None:
            requested = self.catalog.names()
        else:
            unknown = sorted(f for f in features if f not in self.catalog)
            if unknown:
                warnings.append(f"unknown feature(s) ignored: {', '.join(unknown)}")
            requested = [f for f in self.catalog.names() if f in features]

        needed = self.required_datasets(set(requested))
        if not tickers:
            needed = set()
        unavailable = {d for d in needed if not self._has(d)}
        for d in sorted(unavailable):
            affected = [f for f in requested if d in FEATURE_DATASETS.get(f, ())]
            warnings.append(
                f"{d} unavailable: provider '{self._provider_name}' lacks capability '{_CAPABILITY[d].value}'; "
                f"NaN for {', '.join(affected)}"
            )
        fetch = needed - unavailable

        def get(dataset: str, call: Callable[[], pd.DataFrame | pd.Series]):
            if dataset not in fetch:
                return None
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - optional vendor feeds degrade to NaN
                if dataset in _REQUIRED:
                    if isinstance(exc, ProviderError):
                        raise
                    raise ProviderError(f"{dataset} request failed for provider '{self._provider_name}': {exc}") from exc
                affected = [f for f in requested if dataset in FEATURE_DATASETS.get(f, ())]
                warnings.append(
                    f"{dataset} request failed ({type(exc).__name__}: {exc}); NaN for {', '.join(affected)}"
                )
                return None

        start = as_of - timedelta(days=LOOKBACK_CALENDAR_DAYS)
        prices: PricePanel | None = get(PRICES, lambda: self.provider.get_price_history(tickers, start, as_of))
        benchmark = get(BENCHMARK, lambda: self.provider.get_benchmark_history(start, as_of)) if prices is not None else None
        fundamentals = _by_ticker(get(FUNDAMENTALS, lambda: self.provider.get_fundamentals(tickers, as_of)))
        estimates = _by_ticker(get(ESTIMATES, lambda: self.provider.get_estimates(tickers, as_of)))
        short_interest = _by_ticker(get(SHORT_INTEREST, lambda: self.provider.get_short_interest(tickers, as_of)))
        options = _by_ticker(get(OPTIONS, lambda: self.provider.get_options_summary(tickers, as_of)))

        # technical
        if prices is not None:
            panel = prices.subset(tickers)
            tech = compute_technical_features(panel, benchmark, as_of).reindex(index)
            avg_volume = average_volume_shares(panel, as_of).reindex(index)
        else:
            tech = pd.DataFrame(np.nan, index=index, columns=TECHNICAL_FEATURES, dtype="float64")
            avg_volume = None

        # fundamental
        price = tech["price"] if prices is not None else None
        if fundamentals is None and estimates is None:
            fund = pd.DataFrame(np.nan, index=index, columns=FUNDAMENTAL_FEATURES, dtype="float64")
        else:
            fund = compute_fundamental_features(universe, fundamentals, estimates, price, as_of).reindex(index)

        # positioning
        if short_interest is None and options is None:
            pos = pd.DataFrame(np.nan, index=index, columns=POSITIONING_FEATURES, dtype="float64")
        else:
            rv = tech["volatility_20d_pct"] if prices is not None else None
            pos = compute_positioning_features(short_interest, options, avg_volume, rv, tickers).reindex(index)

        frame = self._assemble(universe, index, tech, fund, pos)
        n = len(frame)
        coverage = {f: (float(frame[f].notna().sum()) / n if n else 0.0) for f in requested}
        return FeatureFrame(frame=frame, coverage=coverage, warnings=warnings, universe=universe)

    # -- helpers -----------------------------------------------------------------------------

    @property
    def _provider_name(self) -> str:
        return str(getattr(self.provider, "name", type(self.provider).__name__))

    def _assemble(
        self,
        universe: pd.DataFrame,
        index: pd.Index,
        tech: pd.DataFrame,
        fund: pd.DataFrame,
        pos: pd.DataFrame,
    ) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for c in RAW_REFERENCE_COLUMNS:
            cols[c] = self._label(universe, c, index)
        mcap = pd.to_numeric(universe[fields.MARKET_CAP], errors="coerce") if fields.MARKET_CAP in universe else None
        reference = {
            "gics_sector": self._label(universe, fields.GICS_SECTOR, index),
            "gics_industry": self._label(universe, fields.GICS_INDUSTRY, index),
            "exchange": self._label(universe, fields.EXCHANGE, index),
            "market_cap_usd_bn": (
                (mcap.astype("float64").where(lambda s: np.isfinite(s) & (s > 0)) / 1e9)
                if mcap is not None
                else pd.Series(np.nan, index=index, dtype="float64")
            ),
        }
        for f in self.catalog.names():
            if f in reference:
                cols[f] = reference[f]
            elif f in tech.columns:
                cols[f] = tech[f]
            elif f in fund.columns:
                cols[f] = fund[f]
            elif f in pos.columns:
                cols[f] = pos[f]
            elif self.catalog[f].dtype == "category":
                cols[f] = pd.Series(None, index=index, dtype=object)
            else:
                cols[f] = pd.Series(np.nan, index=index, dtype="float64")
        frame = pd.DataFrame(cols, index=index)
        numeric = [f for f in self.catalog.names() if self.catalog[f].dtype != "category"]
        frame[numeric] = frame[numeric].astype("float64")
        return frame

    @staticmethod
    def _label(universe: pd.DataFrame, column: str, index: pd.Index) -> pd.Series:
        """Text column as object dtype with None for missing / blank labels."""
        if column not in universe.columns:
            return pd.Series(None, index=index, dtype=object)
        out: list[str | None] = []
        for v in universe[column].tolist():
            text = None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
            out.append(text if text and text.strip() else None)
        return pd.Series(out, index=index, dtype=object)
