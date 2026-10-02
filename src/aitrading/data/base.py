"""Vendor-neutral market-data provider interface.

Adapters (Bloomberg, LSEG, S&P Capital IQ, synthetic) implement ``MarketDataProvider`` and return
frames using the canonical column names in ``aitrading.core.fields``. Every method is point-in-time:
it returns only information that was public on or before ``as_of``.

Push-down: a provider that can evaluate (part of) a ScreenSpec inside the vendor's compute
(BQL, LSEG SCREEN) implements ``ScreenPushdown``. The pipeline uses push-down only to shrink the
universe; the local engine re-evaluates every condition on the narrowed set, so a compiler bug can
drop names but can never admit a name that fails the screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.spec import ScreenSpec, UniverseSpec


class ProviderError(RuntimeError):
    """Raised for vendor connectivity / entitlement / query errors."""


class ProviderUnavailable(ProviderError):
    """Raised when a vendor SDK or session is not available in this environment."""


class Capability(str, Enum):
    PRICES = "prices"
    FUNDAMENTALS = "fundamentals"
    ESTIMATES = "estimates"
    SHORT_INTEREST = "short_interest"
    OPTIONS = "options"
    TRANSCRIPTS = "transcripts"
    NEWS = "news"
    FILINGS = "filings"
    RESEARCH = "research"
    SCREEN_PUSHDOWN = "screen_pushdown"


@dataclass
class PricePanel:
    """Wide OHLCV frames: DatetimeIndex (ascending trading days) x ticker columns.

    Prices are split- and dividend-adjusted; volume is split-adjusted shares. All five frames share
    the same index and columns (missing values are NaN).
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        idx, cols = self.close.index, self.close.columns
        for name in ("open", "high", "low", "volume"):
            frame = getattr(self, name)
            if not frame.index.equals(idx) or not frame.columns.equals(cols):
                setattr(self, name, frame.reindex(index=idx, columns=cols))

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def truncate(self, as_of: date) -> "PricePanel":
        """Drop rows after ``as_of`` (guards against look-ahead)."""
        ts = pd.Timestamp(as_of)
        sl = lambda f: f.loc[f.index <= ts]  # noqa: E731
        return PricePanel(sl(self.open), sl(self.high), sl(self.low), sl(self.close), sl(self.volume))

    def subset(self, tickers: list[str]) -> "PricePanel":
        cols = [t for t in tickers if t in self.close.columns]
        return PricePanel(self.open[cols], self.high[cols], self.low[cols], self.close[cols], self.volume[cols])


@dataclass
class PushdownResult:
    tickers: list[str]
    query: str  # the vendor query that was executed, for the audit trail
    pushed_conditions: list[str] = field(default_factory=list)  # Condition.describe() of conditions evaluated vendor-side
    residual_conditions: list[str] = field(default_factory=list)  # conditions only the local engine evaluates


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str
    capabilities: set[Capability]
    boundary: DataBoundary

    def get_universe(self, spec: "UniverseSpec", as_of: date) -> pd.DataFrame:
        """Securities eligible for screening, indexed by ticker, columns ``fields.UNIVERSE_COLUMNS``."""
        ...

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel: ...

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        """Adjusted close of the benchmark (default: the provider's broad US index), DatetimeIndex."""
        ...

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Indexed by ticker, columns ``fields.FUNDAMENTAL_COLUMNS`` (point-in-time on report date)."""
        ...

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Indexed by ticker, columns ``fields.ESTIMATE_COLUMNS``."""
        ...

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Indexed by ticker, columns ``fields.SHORT_INTEREST_COLUMNS``."""
        ...

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Indexed by ticker, columns ``fields.OPTIONS_COLUMNS``."""
        ...

    def get_documents(
        self,
        ticker: str,
        kinds: set[DocumentKind],
        start: date,
        end: date,
        limit: int = 10,
    ) -> list[Document]:
        """Documents published in [start, end], newest first."""
        ...


@runtime_checkable
class ScreenPushdown(Protocol):
    def pushdown_screen(self, spec: "ScreenSpec", as_of: date) -> PushdownResult: ...


def empty_frame(columns: list[str], tickers: list[str] | None = None) -> pd.DataFrame:
    """All-NaN frame with the canonical columns, for providers lacking a dataset."""
    return pd.DataFrame(index=pd.Index(tickers or [], name="ticker"), columns=columns, dtype="float64")
