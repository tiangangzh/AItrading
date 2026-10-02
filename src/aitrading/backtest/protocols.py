"""Interface of the strategy runner, so replication, paper trading and reporting can be built and
tested independently of how the runner gathers data.

The concrete runner (``aitrading.backtest.runner.StrategyRunner``) turns a StrategySpec into
point-in-time signals, portfolios and a BacktestResult. Paper trading MUST call
``target_portfolio`` - the same code path the backtest uses at each rebalance - so live (simulated)
positions can never drift from what was tested.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from aitrading.backtest.models import BacktestResult
from aitrading.strategy.spec import StrategySpec


@runtime_checkable
class BacktestRunner(Protocol):
    provider_name: str

    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        """Run the full backtest for ``spec`` (its own start/end/rebalance/costs)."""
        ...

    def target_portfolio(self, spec: StrategySpec, as_of: date) -> pd.Series:
        """Signed target weights (ticker -> weight) the strategy would hold after rebalancing on
        ``as_of``, using only information available at the close of ``as_of``."""
        ...

    def latest_prices(self, tickers: list[str], as_of: date) -> pd.Series:
        """Adjusted close on or before ``as_of`` for each ticker (NaN when unavailable)."""
        ...
