"""Simulated (paper) trading of saved strategies - there is no broker integration.

* :class:`StrategyStore` saves a strategy (spec + latest backtest + metadata) under
  ``$AITRADING_HOME/strategies/<slug>/`` (default ``~/.aitrading/strategies``).
* :class:`PaperAccount` paper-trades a saved strategy: on each run it rebalances to the runner's
  target portfolio when a rebalance is due, otherwise marks the book to market, and keeps a
  persisted ledger (cash, positions, trades, NAV history) next to the saved strategy.
"""

from aitrading.trading.paper import (
    ALREADY_REBALANCED,
    DEFAULT_MIN_TRADE_PCT,
    DEFAULT_STALE_PRICE_DAYS,
    LEDGER_SCHEMA_VERSION,
    MAX_UNPRICED_TARGET_WEIGHT,
    LedgerError,
    Order,
    PaperAccount,
    PaperLedger,
    PriceMark,
    RebalanceReport,
    Trade,
    backtest_credits_cash_interest,
    backtest_return_over_window,
    is_rebalance_day,
    next_rebalance_date,
    scheduled_rebalance_on_or_before,
)
from aitrading.trading.store import (
    SCHEMA_VERSION,
    StoreError,
    StrategyExistsError,
    StrategyNotFoundError,
    StrategyStore,
    default_store_root,
    slugify_strategy_name,
)

__all__ = [
    "ALREADY_REBALANCED",
    "DEFAULT_MIN_TRADE_PCT",
    "DEFAULT_STALE_PRICE_DAYS",
    "LEDGER_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "LedgerError",
    "MAX_UNPRICED_TARGET_WEIGHT",
    "Order",
    "PaperAccount",
    "PaperLedger",
    "PriceMark",
    "RebalanceReport",
    "StoreError",
    "StrategyExistsError",
    "StrategyNotFoundError",
    "StrategyStore",
    "Trade",
    "backtest_credits_cash_interest",
    "backtest_return_over_window",
    "default_store_root",
    "is_rebalance_day",
    "next_rebalance_date",
    "scheduled_rebalance_on_or_before",
    "slugify_strategy_name",
]
