"""Paper (simulated) trading of a saved strategy. There is NO broker: no order ever leaves the PC.

A :class:`PaperAccount` keeps a persisted ledger (``ledger.json`` in the strategy's store
directory) and, every time the trader runs it, either rebalances the book to the strategy's
current target portfolio or just marks it to market:

* The target comes from ``runner.target_portfolio(spec, as_of)`` - the same code path the backtest
  uses at each rebalance - so the simulated positions can never drift from what was tested.
* Fills are simulated at ``runner.latest_prices(...)`` (the adjusted close on or before ``as_of``).

Rebalance calendar
------------------
A strategy rebalances on the last business day (Mon-Fri; exchange holidays are not modelled) of
each week (Mon-Sun) / month / quarter / calendar year, or every business day for ``daily``.
:meth:`PaperAccount.rebalance` trades when

* the account has never been started - the initial rebalance happens immediately, even mid-period;
* a scheduled rebalance date has arrived since the last rebalance - if the PC was off on that day
  the rebalance happens late, on the next run, with a warning;
* ``force=True``.

Otherwise it only marks the book to market. A second call on a day that already rebalanced returns
``skipped_reason == 'already rebalanced today'`` without touching the ledger (idempotent). Runs
must move forward in time; an ``as_of`` before the last run raises ``ValueError``, and so does an
``as_of`` in the future (more than one day after the account clock's UTC date, the slack covering
time zones), since it would block every later real run.

A rebalance whose target cannot mostly be priced - more than half of the target's gross weight has
no price, as in a data-provider outage - trades nothing and is not recorded as a rebalance, so the
next run retries it (as ``initial`` / ``late``) instead of leaving the book in cash for a period.

Accounting convention
---------------------
* ``NAV = cash + sum_i shares_i * price_i`` at all times (the identity holds after every
  operation). Shorts are negative share counts: selling short credits the proceeds to cash and
  the short position is a negative market value; buying to cover debits cash. Cash earns 0% and
  there is no margin interest or borrow fee; leverage simply shows as negative cash. (The backtest
  of a long-only, screen or timing strategy credits the daily T-bill rate on idle cash and on
  short-sale proceeds - see ``StrategyRunner._cash_credit`` - so it is not the same convention:
  see the differences listed below.)
* Orders bring each name from its current shares to ``target_weight x NAV' / price``, where NAV'
  is the NAV after this rebalance's costs (solved by a short fixed-point iteration), so after a
  rebalance the holdings weights equal the target weights and a fully invested long-only book
  ends with ~0 cash rather than a small overdraft. With ``allow_fractional=False`` share counts
  are truncated toward zero. Orders smaller than ``min_trade_pct`` (0.1%) of NAV are skipped,
  except full exits (so no dust positions remain).
* Costs: ``|notional| x costs_bps / 1e4`` per order (one-way, as in the backtest), deducted from
  cash. ``costs_bps`` defaults to ``spec.costs_bps``.
* Missing prices: a target name without a price is not traded (warning); a held name without a
  price is carried at its last known price (warning) and cannot be traded until it has a price.
  The mark records the first run that found the price missing (``PriceMark.missing_since``,
  cleared when a price comes back); a held name still without a price more than
  ``stale_price_days`` business days after that run is treated as delisted and closed at
  ``last price x (1 + spec.delisting_return)`` (warning). The gap is measured from when the price
  went missing, not from the last priced run, so an account run once a month is not force-closed
  by a one-day data gap; it always takes at least two runs without a price.
* Corporate actions: prices are adjusted closes, whose history the provider rescales after a
  split or dividend. With every mark the account also records an *anchor* price two business days
  earlier (a final close even when the run happens intraday, before today's close is final). At
  the next run the anchor is re-read; if the provider's history has been re-adjusted since, the
  position is scaled by the adjustment factor (and its mark divided by it, so its value is
  unchanged) - splits do not show up as fake losses and dividends are reinvested (cash in lieu
  for whole-share accounts), matching the total-return convention of the backtest. The current
  prices, the new anchors and the re-read anchors come from ONE price download
  (``runner.prices_at``, when the runner has it), so they share one adjustment vintage: a past
  window served from the provider's cache would still show the pre-split history.

Paper results differ from the backtest because the backtest trades at the next session's close
(``execution_lag=1``) while paper fills use the ``as_of`` close, because of whole shares, skipped
small orders, late runs and data revisions - and because idle cash earns the T-bill rate in the
backtest (2x for a -1 short: capital plus proceeds) but 0% here, so a book that sits in cash or is
net short lags its backtest by about RF x its cash share (e.g. ~1.1% over a quarter in cash at a
4.5% RF). Long-short cross-sectional and factor-model backtests get no cash credit.
:meth:`PaperAccount.summary` puts both side by side and lists the applicable differences in
``backtest_comparison_notes``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from aitrading.backtest.metrics import drawdown_series
from aitrading.backtest.models import BacktestResult
from aitrading.backtest.protocols import BacktestRunner
from aitrading.strategy.spec import StrategySpec
from aitrading.trading.store import (
    LEDGER_FILE,
    StoreError,
    StrategyNotFoundError,
    StrategyStore,
    _retry_os,
    read_json,
    write_json_atomic,
)

__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "ALREADY_REBALANCED",
    "DEFAULT_MIN_TRADE_PCT",
    "DEFAULT_STALE_PRICE_DAYS",
    "ANCHOR_LAG_BDAYS",
    "MAX_UNPRICED_TARGET_WEIGHT",
    "LedgerError",
    "Order",
    "Trade",
    "PriceMark",
    "PaperLedger",
    "RebalanceReport",
    "PaperAccount",
    "is_rebalance_day",
    "scheduled_rebalance_on_or_before",
    "next_rebalance_date",
    "backtest_return_over_window",
    "backtest_credits_cash_interest",
]

log = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
ALREADY_REBALANCED = "already rebalanced today"
DEFAULT_MIN_TRADE_PCT = 0.001
DEFAULT_STALE_PRICE_DAYS = 10

_PERIOD_CODES = {"weekly": "W-SUN", "monthly": "M", "quarterly": "Q", "annual": "Y"}
_BDAY = pd.offsets.BDay()
_SPLIT_WARN = 0.05  # adjustment factors further than this from 1 are reported as warnings
_FACTOR_EPS = 1e-6  # ... closer than this are float noise
ANCHOR_LAG_BDAYS = 2  # anchor prices are this many business days before the mark date
MAX_UNPRICED_TARGET_WEIGHT = 0.5  # above this share of the target's gross weight unpriced: no rebalance
FUTURE_SLACK_DAYS = 1  # as_of may be this many days after the clock's UTC date (time zones)
_ARCHIVE_RE = re.compile(r"^ledger-(\d{8}T\d{6}Z)(?:-(\d+))?\.json$")


# ------------------------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------------------------


class LedgerError(StoreError):
    """The paper-trading ledger cannot be read (corrupt, invalid or from a newer version)."""


class Order(BaseModel):
    """A simulated fill. ``shares`` and ``notional`` are absolute; ``side`` gives the direction
    ("sell" also opens or adds to a short, "buy" also covers one)."""

    ticker: str
    side: Literal["buy", "sell"]
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    notional: float = Field(ge=0, description="shares x price")
    cost: float = Field(ge=0, description="notional x costs_bps / 1e4, deducted from cash")
    reason: Literal["rebalance", "delisting"] = "rebalance"

    @property
    def signed_shares(self) -> float:
        return self.shares if self.side == "buy" else -self.shares

    @property
    def signed_notional(self) -> float:
        return self.notional if self.side == "buy" else -self.notional


class Trade(Order):
    """An executed order as stored in the ledger."""

    as_of: date


class PriceMark(BaseModel):
    """Last valid price of a held name and the run date it was observed on, plus the anchor price
    (``anchor_date`` = two business days earlier) used to detect re-adjusted price history.
    ``missing_since`` is the first run since then that found no price (None while priced)."""

    price: float = Field(gt=0)
    as_of: date
    anchor_date: date | None = None
    anchor_price: float | None = Field(None, gt=0)
    missing_since: date | None = None


class PaperLedger(BaseModel):
    """Persisted state of a paper account (``ledger.json``)."""

    schema_version: int = LEDGER_SCHEMA_VERSION
    name: str
    initial_capital: float
    cash: float
    positions: dict[str, float] = Field(default_factory=dict, description="ticker -> shares (negative = short)")
    last_prices: dict[str, PriceMark] = Field(default_factory=dict, description="mark of every held name")
    last_target: dict[str, float] = Field(default_factory=dict, description="target weights of the last rebalance")
    trades: list[Trade] = Field(default_factory=list, description="oldest first")
    nav_history: list[tuple[date, float]] = Field(default_factory=list, description="one entry per run date")
    runs: list[tuple[date, bool]] = Field(default_factory=list, description="(run date, rebalanced that day)")
    started: date | None = None
    last_rebalance: date | None = None
    rebalance_frequency: str | None = None
    total_costs: float = 0.0
    last_warnings: list[str] = Field(default_factory=list)
    allow_fractional: bool = True
    costs_bps: float | None = None
    created_at: datetime
    updated_at: datetime


class RebalanceReport(BaseModel):
    """What one :meth:`PaperAccount.rebalance` call did.

    ``nav_before`` is the NAV at the ``as_of`` prices before any trade, ``nav_after`` after the
    trades (``nav_before - total_cost``; equal when nothing traded). ``skipped_reason`` is set when
    the book was only marked to market (or nothing was done at all).
    """

    as_of: date
    rebalanced: bool
    trigger: Literal["initial", "scheduled", "late", "forced"] | None = None
    orders: list[Order] = Field(default_factory=list)
    nav_before: float
    nav_after: float
    cash_after: float
    total_cost: float = 0.0
    target_weights: dict[str, float] = Field(default_factory=dict)
    small_orders_skipped: int = 0
    skipped_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="informational (e.g. dividends reinvested)")
    next_rebalance: date | None = None


# ------------------------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------------------------


def _as_date(x: Any) -> date:
    if isinstance(x, datetime):  # includes pd.Timestamp
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return date.fromisoformat(x.strip()[:10])
    raise TypeError(f"expected a date, got {type(x).__name__}")


def _period_code(frequency: str) -> str | None:
    key = str(frequency).lower()
    if key == "daily":
        return None
    if key not in _PERIOD_CODES:
        raise ValueError(f"unknown rebalance frequency {frequency!r}; expected daily/weekly/monthly/quarterly/annual")
    return _PERIOD_CODES[key]


def _last_bday(period: pd.Period) -> pd.Timestamp:
    return _BDAY.rollback(period.end_time.normalize())


def scheduled_rebalance_on_or_before(as_of: date, frequency: str) -> date:
    """The most recent scheduled rebalance date on or before ``as_of``: the last business day of
    the week (Mon-Sun) / month / quarter / year (``daily``: the last business day)."""
    ts = pd.Timestamp(_as_date(as_of))
    code = _period_code(frequency)
    if code is None:
        return _BDAY.rollback(ts).date()
    period = ts.to_period(code)
    end = _last_bday(period)
    if end > ts:
        end = _last_bday(period - 1)
    return end.date()


def next_rebalance_date(after: date, frequency: str) -> date:
    """The first scheduled rebalance date strictly after ``after``."""
    ts = pd.Timestamp(_as_date(after))
    code = _period_code(frequency)
    if code is None:
        return (ts + _BDAY).date()
    period = ts.to_period(code)
    end = _last_bday(period)
    if end <= ts:
        end = _last_bday(period + 1)
    return end.date()


def _anchor_date(as_of: date) -> date:
    return (pd.Timestamp(as_of) - ANCHOR_LAG_BDAYS * _BDAY).date()


def is_rebalance_day(as_of: date, frequency: str) -> bool:
    """True when ``as_of`` is itself a scheduled rebalance date (a business day ending its period)."""
    d = _as_date(as_of)
    return scheduled_rebalance_on_or_before(d, frequency) == d


# ------------------------------------------------------------------------------------------------
# Backtest comparison
# ------------------------------------------------------------------------------------------------


def backtest_credits_cash_interest(backtest: BacktestResult) -> bool:
    """True when ``backtest``'s strategy series includes risk-free interest on idle cash (and on
    short-sale proceeds), which the paper account does not earn: every strategy except long-short
    cross-sectional and factor-model ones, when an official RF series was available."""
    spec = backtest.spec if isinstance(backtest.spec, dict) else {}
    kind = spec.get("kind")
    style = (spec.get("portfolio") or {}).get("style") if isinstance(spec.get("portfolio"), dict) else None
    if kind == "factor_model" or (kind == "cross_sectional" and style == "long_short"):
        return False
    return any(u.dataset == "risk_free" and u.source != "none" for u in backtest.data_usage)


def backtest_return_over_window(
    backtest: BacktestResult, start: date, end: date, *, series: str = "strategy"
) -> dict[str, Any] | None:
    """Compounded backtest return over the paper-trading window (``start``, ``end``].

    Uses the backtest's periodic returns whose period-end date falls in the window, so the match
    is as fine as the backtest's return frequency. Returns ``None`` unless the backtest covers the
    window (a period ending on or before ``start`` and one ending on or after ``end``) and at least
    one period ends inside it. Missing returns count as 0. Result keys: ``return_pct``,
    ``start``/``end`` (first/last period end used, ISO), ``n_periods``.
    """
    start, end = _as_date(start), _as_date(end)
    values = backtest.returns.get(series)
    if end <= start or not values or not backtest.dates or len(values) != len(backtest.dates):
        return None
    r = pd.Series([np.nan if v is None else float(v) for v in values], index=pd.DatetimeIndex(backtest.dates)).sort_index()
    r = r.where(np.isfinite(r))
    if r.index[0].date() > start or r.index[-1].date() < end:
        return None
    window = r[(r.index > pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    if window.empty:
        return None
    total = float(np.prod(1.0 + window.fillna(0.0).to_numpy()) - 1.0)
    return {
        "return_pct": total * 100.0,
        "start": window.index[0].date().isoformat(),
        "end": window.index[-1].date().isoformat(),
        "n_periods": int(len(window)),
    }


# ------------------------------------------------------------------------------------------------
# Account
# ------------------------------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _valid_price(p: Any) -> bool:
    try:
        x = float(p)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and x > 0


def _fmt_shares(x: float) -> str:
    return f"{x:,.6f}".rstrip("0").rstrip(".")


class PaperAccount:
    """Simulated trading account of strategy ``name`` in ``store`` (see the module docstring).

    ``initial_capital`` is used when the ledger is created (and by :meth:`reset`); an existing
    ledger keeps its own. ``allow_fractional``, ``costs_bps`` (``None`` = ``spec.costs_bps``),
    ``min_trade_pct`` and ``stale_price_days`` are execution settings applied from now on. Every
    operation re-reads ``ledger.json`` first and writes it atomically at the end, so several
    ``PaperAccount`` objects (or CLI runs) on the same strategy see one consistent ledger and a
    failing runner leaves the ledger untouched. Runs are not locked against each other: do not
    run two operations on the same strategy at the same moment (the last write wins; the file
    itself can never be left half-written). The constructor does not read the ledger: an
    unreadable ``ledger.json`` raises :class:`LedgerError` from the operations that need it, so
    :meth:`reset` can always be called on a freshly created account. ``clock`` (default: UTC now)
    stamps the ledger and bounds ``as_of`` (no runs in the future).
    """

    def __init__(
        self,
        store: StrategyStore,
        name: str,
        *,
        initial_capital: float = 100_000.0,
        allow_fractional: bool = True,
        costs_bps: float | None = None,
        min_trade_pct: float = DEFAULT_MIN_TRADE_PCT,
        stale_price_days: int = DEFAULT_STALE_PRICE_DAYS,
        clock: Callable[[], datetime] | None = None,
    ):
        if not _valid_price(initial_capital):
            raise ValueError(f"initial_capital must be a positive number, got {initial_capital!r}")
        if costs_bps is not None and not (math.isfinite(float(costs_bps)) and float(costs_bps) >= 0):
            raise ValueError(f"costs_bps must be >= 0, got {costs_bps!r}")
        if not (0 <= float(min_trade_pct) < 1):
            raise ValueError(f"min_trade_pct must be in [0, 1), got {min_trade_pct!r}")
        if int(stale_price_days) < 0:
            raise ValueError("stale_price_days must be >= 0")
        self.path: Path = store.path(name)  # validates the name (TypeError / ValueError)
        self.store = store
        self.name = name.strip()
        self.ledger_path: Path = self.path / LEDGER_FILE
        self.initial_capital = float(initial_capital)
        self.allow_fractional = bool(allow_fractional)
        self.costs_bps = None if costs_bps is None else float(costs_bps)
        self.min_trade_pct = float(min_trade_pct)
        self.stale_price_days = int(stale_price_days)
        self._clock = clock or _utcnow
        # Not read here (see the class docstring): a corrupt ledger must not prevent reset().
        self._ledger: PaperLedger | None = None

    def __repr__(self) -> str:
        return f"PaperAccount(name={self.name!r}, ledger={str(self.ledger_path)!r})"

    # ------------------------------------------------------------------ persistence
    def _new_ledger(self, capital: float) -> PaperLedger:
        now = self._clock()
        return PaperLedger(
            name=self.name,
            initial_capital=capital,
            cash=capital,
            allow_fractional=self.allow_fractional,
            costs_bps=self.costs_bps,
            created_at=now,
            updated_at=now,
        )

    def _read_ledger(self) -> PaperLedger:
        try:
            data = read_json(self.ledger_path)
        except FileNotFoundError:
            return self._new_ledger(self.initial_capital)
        except StoreError as e:
            raise LedgerError(f"{e} - restore it from a ledger-*.json archive or call reset()") from e
        if not isinstance(data, dict):
            raise LedgerError(f"{self.ledger_path}: expected a JSON object, got {type(data).__name__}")
        version = data.get("schema_version", 1)
        if not isinstance(version, int) or version > LEDGER_SCHEMA_VERSION:
            raise LedgerError(
                f"{self.ledger_path} has schema_version {version!r}; this version of aitrading reads <= {LEDGER_SCHEMA_VERSION}"
            )
        try:
            led = PaperLedger.model_validate(data)
        except ValidationError as e:
            raise LedgerError(f"{self.ledger_path} is not a valid paper ledger: {e}") from e
        unmarked = sorted(t for t in led.positions if t not in led.last_prices)
        if unmarked:
            raise LedgerError(f"{self.ledger_path}: held positions without a price mark: {', '.join(unmarked)}")
        return led

    def _reload(self) -> PaperLedger:
        led = self._read_ledger()
        self._ledger = led
        return led.model_copy(deep=True)

    def _save(self, led: PaperLedger) -> None:
        led.positions = {t: s for t, s in sorted(led.positions.items()) if s != 0.0}
        led.last_prices = {t: m for t, m in sorted(led.last_prices.items()) if t in led.positions}
        led.allow_fractional = self.allow_fractional
        led.costs_bps = self.costs_bps
        led.updated_at = self._clock()
        write_json_atomic(self.ledger_path, led.model_dump(mode="json"))
        self._ledger = led

    # ------------------------------------------------------------------ state
    @property
    def ledger(self) -> PaperLedger:
        """A copy of the current ledger (re-read from disk)."""
        return self._reload()

    @property
    def started(self) -> date | None:
        return self._reload().started

    @property
    def cash(self) -> float:
        return self._reload().cash

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._reload().positions)

    @property
    def nav(self) -> float:
        """NAV at the last known prices."""
        return self._nav(self._reload())

    @staticmethod
    def _nav(led: PaperLedger) -> float:
        return float(led.cash + sum(sh * led.last_prices[t].price for t, sh in led.positions.items()))

    def archived_ledgers(self) -> list[Path]:
        """Ledgers archived by :meth:`reset`, oldest first: by archive time, then by the counter
        that same-second archives get (``ledger-<stamp>.json``, ``-1``, ``-2``, ... ``-10``).
        Other ``ledger-*.json`` files (e.g. hand-made copies) come first, by name."""
        if not self.path.is_dir():
            return []

        def key(p: Path) -> tuple[int, str, int, str]:
            m = _ARCHIVE_RE.match(p.name)
            if m is None:
                return (0, "", 0, p.name)
            return (1, m.group(1), int(m.group(2) or 0), p.name)

        return sorted(self.path.glob("ledger-*.json"), key=key)

    def _stored_spec(self) -> StrategySpec | None:
        try:
            return self.store.load_spec(self.name)
        except StrategyNotFoundError:
            return None
        except StoreError as e:
            log.warning("cannot read the saved spec of %r: %s", self.name, e)
            return None

    def _effective_costs_bps(self) -> float | None:
        """The configured cost, else the saved spec's ``costs_bps`` (None if unknown)."""
        if self.costs_bps is not None:
            return self.costs_bps
        stored = self._stored_spec()
        return float(stored.costs_bps) if stored is not None else None

    # ------------------------------------------------------------------ helpers
    def _today(self) -> date:
        """The clock's date in UTC (a naive clock is taken as is)."""
        now = self._clock()
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc)
        return now.date()

    def _check_not_future(self, as_of: date) -> None:
        today = self._today()
        if as_of > today + timedelta(days=FUTURE_SLACK_DAYS):
            raise ValueError(
                f"as_of {as_of} is in the future (today is {today} UTC); paper trading only fills at prices that "
                "exist, and a run recorded in the future would block every real run until that date - check the date"
            )

    @staticmethod
    def _check_chronology(led: PaperLedger, as_of: date) -> None:
        last = led.runs[-1][0] if led.runs else None
        if last is not None and as_of < last:
            raise ValueError(
                f"as_of {as_of} is before the last paper-trading run on {last}; the ledger only moves "
                "forward in time (use reset() to start over)"
            )

    @staticmethod
    def _clean_prices(raw: Any, tickers: list[str]) -> pd.Series:
        """``raw`` prices as a float Series over ``tickers`` in that order; NaN where missing / invalid."""
        s = raw if isinstance(raw, pd.Series) else pd.Series(raw, dtype=object)
        s = pd.to_numeric(s, errors="coerce").astype(float)
        s.index = [str(i) for i in s.index]
        s = s[~s.index.duplicated(keep="last")].reindex(tickers)
        return s.where(np.isfinite(s) & (s > 0))

    @staticmethod
    def _fetch_prices(runner: BacktestRunner, tickers: list[str], as_of: date) -> pd.Series:
        """Prices of ``tickers`` as a float Series in that order; NaN where missing / invalid."""
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            return pd.Series(dtype=float)
        return PaperAccount._clean_prices(runner.latest_prices(tickers, as_of), tickers)

    @staticmethod
    def _price_table(runner: BacktestRunner, tickers: list[str], dates: list[date]) -> dict[date, pd.Series]:
        """Prices of ``tickers`` on or before each of ``dates`` (as :meth:`_fetch_prices`).

        A runner with ``prices_at(tickers, dates)`` (``StrategyRunner``) serves every date from ONE
        price download, so the current price and the re-read anchor prices share one adjustment
        vintage. Separate ``latest_prices`` calls per date would let a provider's cache of a past
        window (adjusted before a split or dividend) hide the re-adjustment, booking the split as a
        loss; they are only the fallback for runners without ``prices_at``.
        """
        tickers = list(dict.fromkeys(tickers))
        days = sorted(set(dates))
        if not tickers:
            return {d: pd.Series(dtype=float) for d in days}
        prices_at = getattr(runner, "prices_at", None)
        if not callable(prices_at):
            return {d: PaperAccount._fetch_prices(runner, tickers, d) for d in days}
        raw = prices_at(tickers, days)
        frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        rows = {_as_date(i): frame.iloc[k] for k, i in enumerate(frame.index)}
        return {d: PaperAccount._clean_prices(rows.get(d, pd.Series(dtype=float)), tickers) for d in days}

    @staticmethod
    def _clean_target(raw: Any, warnings: list[str]) -> pd.Series:
        if raw is None:
            return pd.Series(dtype=float)
        s = raw if isinstance(raw, pd.Series) else pd.Series(raw, dtype=object)
        s = pd.to_numeric(s, errors="coerce").astype(float)
        s.index = [str(i).strip() for i in s.index]
        if s.index.has_duplicates:
            dups = sorted(set(s.index[s.index.duplicated()]))
            warnings.append(f"target portfolio lists {', '.join(dups)} more than once; their weights were summed")
            s = s.groupby(level=0).sum(min_count=1)
        bad = ~np.isfinite(s.to_numpy())
        if bad.any():
            warnings.append(f"target weights that are not numbers were ignored: {', '.join(sorted(s.index[bad]))}")
            s = s[~bad]
        return s[s != 0.0].sort_index()

    @staticmethod
    def _anchors_to_reread(led: PaperLedger, as_of: date) -> list[date]:
        """Anchor dates of held marks from earlier runs (re-read to detect corporate actions)."""
        return sorted(
            {
                m.anchor_date
                for t, m in led.last_prices.items()
                if t in led.positions and m.as_of < as_of and m.anchor_date is not None and m.anchor_price is not None
            }
        )

    def _apply_corporate_actions(
        self,
        led: PaperLedger,
        as_of: date,
        table: dict[date, pd.Series],
        warnings: list[str],
        notes: list[str],
    ) -> None:
        """Scale held positions whose adjusted price history changed since their last mark
        (detected by re-reading the mark's anchor price, see the module docstring). ``table`` holds
        the ``as_of`` prices and the re-read anchor prices, from one download (:meth:`_price_table`)."""
        prices = table[as_of]
        groups: dict[date, list[str]] = {}
        for t in sorted(led.positions):
            mark = led.last_prices[t]
            if (
                mark.as_of < as_of
                and mark.anchor_date is not None
                and mark.anchor_price is not None
                and _valid_price(prices.get(t))
            ):
                groups.setdefault(mark.anchor_date, []).append(t)
        for anchor_date, names in sorted(groups.items()):
            reread = table.get(anchor_date, pd.Series(dtype=float))
            for t in names:
                pv = reread.get(t)
                if not _valid_price(pv):
                    continue
                pv = float(pv)
                mark = led.last_prices[t]
                factor = float(mark.anchor_price) / pv
                if abs(factor - 1.0) <= _FACTOR_EPS:
                    continue
                if not (0.01 <= factor <= 100.0):
                    warnings.append(
                        f"{t}: the provider's price for {anchor_date} changed by a factor of {factor:.4g} since it was "
                        "recorded - too large for a split or dividend adjustment; position left unchanged, check the data"
                    )
                    continue
                old = led.positions[t]
                new = old * factor
                adj_price = mark.price / factor
                if self.allow_fractional:
                    new = round(new, 6)
                else:
                    whole = float(math.trunc(new))
                    led.cash += (new - whole) * adj_price  # cash in lieu of the fractional share
                    new = whole
                led.positions[t] = new
                led.last_prices[t] = PriceMark(price=adj_price, as_of=mark.as_of, anchor_date=anchor_date, anchor_price=pv)
                msg = (
                    f"{t}: price history re-adjusted since {mark.as_of} (factor {factor:.4f}, a split or dividend); "
                    f"position scaled from {_fmt_shares(old)} to {_fmt_shares(new)} shares"
                )
                (warnings if abs(factor - 1.0) > _SPLIT_WARN else notes).append(msg)

    def _mark(
        self,
        led: PaperLedger,
        runner: BacktestRunner,
        as_of: date,
        extra_tickers: list[str],
        delisting_return: float | None,
        warnings: list[str],
        notes: list[str],
    ) -> tuple[pd.Series, list[Order]]:
        """Fetch prices for held + ``extra_tickers``, adjust for corporate actions, update marks,
        close long-stale positions as delisted. Returns ``(prices, delisting orders)``."""
        tickers = sorted(set(led.positions) | set(extra_tickers))
        anchor_d = _anchor_date(as_of)
        # One download for the current prices, the new anchors and the re-read old anchors: they must
        # share one adjustment vintage, or a cached pre-split anchor would hide the split.
        table = self._price_table(runner, tickers, [as_of, anchor_d, *self._anchors_to_reread(led, as_of)])
        prices = table[as_of]
        self._apply_corporate_actions(led, as_of, table, warnings, notes)
        valid = [t for t, p in prices.items() if _valid_price(p)]
        anchors = table[anchor_d]
        for t in valid:
            a = anchors.get(t)
            ok = _valid_price(a)
            led.last_prices[t] = PriceMark(
                price=float(prices[t]),
                as_of=as_of,
                anchor_date=anchor_d if ok else None,
                anchor_price=float(a) if ok else None,
            )
        delisted: list[Order] = []
        for t in sorted(led.positions):
            if _valid_price(prices.get(t)):
                continue
            mark = led.last_prices[t]
            if mark.missing_since is None or mark.missing_since > as_of:
                mark = mark.model_copy(update={"missing_since": as_of})
                led.last_prices[t] = mark
            since = mark.missing_since
            assert since is not None
            gap = int(np.busday_count(since, as_of))  # business days since the first run without a price
            if delisting_return is not None and gap > self.stale_price_days:
                shares = led.positions.pop(t)
                px = mark.price * (1.0 + float(delisting_return))
                led.cash += shares * px
                order = Order(
                    ticker=t,
                    side="sell" if shares > 0 else "buy",
                    shares=abs(shares),
                    price=px,
                    notional=abs(shares) * px,
                    cost=0.0,
                    reason="delisting",
                )
                led.trades.append(Trade(as_of=as_of, **order.model_dump()))
                delisted.append(order)
                warnings.append(
                    f"{t} has had no price for {gap} business days (missing since the run of {since}; last price "
                    f"{mark.price:,.4f} from {mark.as_of}); treated as delisted and closed at its last price x "
                    f"(1 + delisting_return {float(delisting_return):+.0%}) = {px:,.4f}"
                )
            else:
                missing = f", missing since the run of {since}" if since < as_of else ""
                warnings.append(
                    f"no price for held position {t} on {as_of}{missing}: carried at its last known price "
                    f"{mark.price:,.4f} (from {mark.as_of}); it cannot be traded until a price is available"
                )
        return prices, delisted

    @staticmethod
    def _record_run(led: PaperLedger, as_of: date, nav: float, rebalanced: bool, warnings: list[str]) -> None:
        if led.nav_history and led.nav_history[-1][0] == as_of:
            led.nav_history[-1] = (as_of, nav)
        else:
            led.nav_history.append((as_of, nav))
        if led.runs and led.runs[-1][0] == as_of:
            led.runs[-1] = (as_of, led.runs[-1][1] or rebalanced)
        else:
            led.runs.append((as_of, rebalanced))
        led.last_warnings = list(warnings)

    def _size_orders(
        self,
        led: PaperLedger,
        target: pd.Series,
        prices: pd.Series,
        tradable: list[str],
        sizing_nav: float,
        min_notional: float,
        rate: float,
    ) -> tuple[list[tuple[str, float, float, float, float, float]], int]:
        plan: list[tuple[str, float, float, float, float, float]] = []
        n_small = 0
        for t in tradable:
            price = float(prices[t])
            weight = float(target.get(t, 0.0))
            desired = weight * sizing_nav / price
            desired = round(desired, 6) if self.allow_fractional else float(math.trunc(desired))
            current = float(led.positions.get(t, 0.0))
            delta = desired - current
            if abs(delta) <= 1e-9:
                continue
            notional = abs(delta) * price
            if notional < min_notional and desired != 0.0:
                n_small += 1
                continue
            plan.append((t, desired, delta, price, notional, notional * rate))
        return plan, n_small

    # ------------------------------------------------------------------ operations
    def rebalance(self, runner: BacktestRunner, spec: StrategySpec, as_of: date, *, force: bool = False) -> RebalanceReport:
        """Rebalance to ``runner.target_portfolio(spec, as_of)`` if due (see module docstring),
        otherwise mark to market; persists the ledger and returns what happened."""
        as_of = _as_date(as_of)
        self._check_not_future(as_of)
        led = self._reload()
        self._check_chronology(led, as_of)
        nxt = next_rebalance_date(as_of, spec.rebalance)
        if led.runs and led.runs[-1][0] == as_of and led.runs[-1][1] and not force:
            nav = self._nav(led)
            return RebalanceReport(
                as_of=as_of,
                rebalanced=False,
                nav_before=nav,
                nav_after=nav,
                cash_after=led.cash,
                skipped_reason=ALREADY_REBALANCED,
                next_rebalance=nxt,
            )

        warnings: list[str] = []
        notes: list[str] = []
        stored = self._stored_spec()
        if stored is not None and stored.model_dump(mode="json") != spec.model_dump(mode="json"):
            warnings.append(
                f"the spec being traded differs from the spec saved as {self.name!r}; paper results will not be "
                "comparable with the saved backtest"
            )

        scheduled = scheduled_rebalance_on_or_before(as_of, spec.rebalance)
        trigger: Literal["initial", "scheduled", "late", "forced"] | None = None
        if led.started is None:
            trigger = "initial"
        elif led.last_rebalance is None or scheduled > led.last_rebalance:
            trigger = "scheduled"
            if pd.Timestamp(scheduled) < _BDAY.rollback(pd.Timestamp(as_of)):
                trigger = "late"
                warnings.append(
                    f"the scheduled {spec.rebalance} rebalance on {scheduled} was missed (no run that day); "
                    f"rebalancing late on {as_of}"
                )
        elif force:
            trigger = "forced"

        target: pd.Series | None = None
        if trigger is not None:
            target = self._clean_target(runner.target_portfolio(spec, as_of), warnings)
        prices, delisted = self._mark(
            led, runner, as_of, list(target.index) if target is not None else [], spec.delisting_return, warnings, notes
        )
        nav_before = self._nav(led)

        if trigger is None or nav_before <= 0:
            if trigger is None:
                reason = (
                    f"not a rebalance day for a {spec.rebalance} strategy (next scheduled rebalance on {nxt}); "
                    "positions marked to market only - use force=True to rebalance now"
                )
            else:
                reason = f"NAV is not positive ({nav_before:,.2f}); nothing to rebalance, positions marked to market only"
                warnings.append(reason)
            self._record_run(led, as_of, nav_before, False, warnings)
            self._save(led)
            return RebalanceReport(
                as_of=as_of,
                rebalanced=False,
                orders=delisted,
                nav_before=nav_before,
                nav_after=nav_before,
                cash_after=led.cash,
                skipped_reason=reason,
                warnings=warnings,
                notes=notes,
                next_rebalance=nxt,
            )

        assert target is not None
        unpriced = [t for t in target.index if not _valid_price(prices.get(t))]
        gross = float(target.abs().sum())
        unpriced_gross = float(target[unpriced].abs().sum()) if unpriced else 0.0
        if gross > 0 and unpriced_gross > MAX_UNPRICED_TARGET_WEIGHT * gross:
            # Most of the target cannot be priced (a data outage): trading the rest would leave the
            # book mostly in cash for a whole period. Trade nothing and do not count this run as a
            # rebalance, so the next run retries it (as "initial" / "late").
            retry = (
                "run again with force=True once prices are available"
                if trigger == "forced"
                else f"the {trigger} rebalance will be retried on the next run"
            )
            reason = (
                f"{len(unpriced)} of {len(target)} target names ({unpriced_gross / gross:.0%} of the target's gross "
                f"weight) have no price on {as_of} - probably a data outage; nothing was traded and {retry}"
            )
            warnings.append(reason)
            self._record_run(led, as_of, nav_before, False, warnings)
            self._save(led)
            return RebalanceReport(
                as_of=as_of,
                rebalanced=False,
                orders=delisted,
                nav_before=nav_before,
                nav_after=nav_before,
                cash_after=led.cash,
                target_weights={t: float(w) for t, w in target.items()},
                skipped_reason=reason,
                warnings=warnings,
                notes=notes,
                next_rebalance=nxt,
            )
        for t in unpriced:
            w = float(target[t])
            kept = " (the current position is kept)" if t in led.positions else ""
            warnings.append(f"no price for {t} on {as_of}; its target weight {w:+.2%} was not traded{kept}")
        tradable = [t for t in sorted(set(led.positions) | set(target.index)) if _valid_price(prices.get(t))]
        costs_bps = self.costs_bps if self.costs_bps is not None else float(spec.costs_bps)
        rate = costs_bps / 1e4
        min_notional = self.min_trade_pct * nav_before
        sizing_nav = nav_before
        plan, n_small = self._size_orders(led, target, prices, tradable, sizing_nav, min_notional, rate)
        for _ in range(8):  # size on the post-cost NAV: N' = N - costs(N')
            new_sizing = nav_before - sum(p[5] for p in plan)
            if abs(new_sizing - sizing_nav) <= 1e-9 * abs(nav_before):
                break
            sizing_nav = new_sizing
            plan, n_small = self._size_orders(led, target, prices, tradable, sizing_nav, min_notional, rate)

        orders: list[Order] = []
        for t, desired, delta, price, notional, cost in sorted(plan, key=lambda p: (p[2] > 0, p[0])):
            led.cash -= delta * price + cost
            led.positions[t] = desired
            order = Order(
                ticker=t,
                side="buy" if delta > 0 else "sell",
                shares=abs(delta),
                price=price,
                notional=notional,
                cost=cost,
            )
            led.trades.append(Trade(as_of=as_of, **order.model_dump()))
            led.total_costs += cost
            orders.append(order)
        led.positions = {t: s for t, s in led.positions.items() if s != 0.0}
        if n_small:
            notes.append(
                f"{n_small} order(s) smaller than {self.min_trade_pct:.1%} of NAV ({min_notional:,.2f}) were skipped"
            )
        nav_after = self._nav(led)
        total_cost = float(sum(o.cost for o in orders))
        led.last_target = {t: float(w) for t, w in target.items()}
        led.last_rebalance = as_of
        led.rebalance_frequency = spec.rebalance
        if led.started is None:
            led.started = as_of
        self._record_run(led, as_of, nav_after, True, warnings)
        self._save(led)
        return RebalanceReport(
            as_of=as_of,
            rebalanced=True,
            trigger=trigger,
            orders=delisted + orders,
            nav_before=nav_before,
            nav_after=nav_after,
            cash_after=led.cash,
            total_cost=total_cost,
            target_weights=dict(led.last_target),
            small_orders_skipped=n_small,
            warnings=warnings,
            notes=notes,
            next_rebalance=nxt,
        )

    def mark_to_market(self, runner: BacktestRunner, as_of: date, *, spec: StrategySpec | None = None) -> float:
        """Value the book at the ``as_of`` prices, record it and return the NAV (no trading).

        Before the first rebalance the account holds only cash: the initial capital is returned
        and nothing is written. ``spec`` (default: the saved spec) supplies ``delisting_return``
        for long-stale positions; without one, stale positions are just carried. Raises
        ``ValueError`` for an ``as_of`` before the last run or in the future.
        """
        as_of = _as_date(as_of)
        self._check_not_future(as_of)
        led = self._reload()
        if led.started is None:
            return float(led.cash)
        self._check_chronology(led, as_of)
        if spec is None:
            spec = self._stored_spec()
        warnings: list[str] = []
        notes: list[str] = []
        self._mark(led, runner, as_of, [], spec.delisting_return if spec is not None else None, warnings, notes)
        nav = self._nav(led)
        self._record_run(led, as_of, nav, False, warnings)
        self._save(led)
        return nav

    def reset(self, initial_capital: float | None = None) -> Path | None:
        """Start the account over (with ``initial_capital``, default: the current ledger's).

        The old ledger is kept as ``ledger-<UTC timestamp>.json`` next to it; returns that path
        (``None`` if there was no ledger yet). Works on a corrupt or unreadable ledger too (its
        capital then defaults to this object's ``initial_capital``), also from a newly created
        account object - the constructor never reads the ledger.
        """
        try:
            old_capital = self._read_ledger().initial_capital
        except Exception:  # noqa: BLE001 - best effort: the old ledger is only archived, never needed
            old_capital = self.initial_capital
        capital = old_capital if initial_capital is None else initial_capital
        if not _valid_price(capital):
            raise ValueError(f"initial_capital must be a positive number, got {capital!r}")
        archived: Path | None = None
        if self.ledger_path.exists():
            stamp = self._clock().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived = self.path / f"ledger-{stamp}.json"
            i = 1
            while archived.exists():
                archived = self.path / f"ledger-{stamp}-{i}.json"
                i += 1
            src, dst = self.ledger_path, archived
            _retry_os(lambda: os.replace(src, dst))
        self.initial_capital = float(capital)
        self._save(self._new_ledger(float(capital)))
        return archived

    # ------------------------------------------------------------------ reporting
    def holdings(self) -> list[dict[str, Any]]:
        """Current positions at their last marks, largest absolute market value first."""
        led = self._reload()
        return self._holdings(led, self._nav(led))

    @staticmethod
    def _holdings(led: PaperLedger, nav: float) -> list[dict[str, Any]]:
        rows = []
        for t, shares in led.positions.items():
            mark = led.last_prices[t]
            value = shares * mark.price
            rows.append(
                {
                    "ticker": t,
                    "shares": shares,
                    "side": "long" if shares > 0 else "short",
                    "price": mark.price,
                    "price_date": mark.as_of.isoformat(),
                    "market_value": value,
                    "weight": value / nav if nav > 0 else None,
                    "target_weight": led.last_target.get(t),
                    "price_missing_since": mark.missing_since.isoformat() if mark.missing_since else None,
                }
            )
        rows.sort(key=lambda r: (-abs(r["market_value"]), r["ticker"]))
        return rows

    def summary(self, *, backtest: BacktestResult | None = None) -> dict[str, Any]:
        """Everything the strategy page shows about the paper account (JSON-serialisable).

        Keys
        ----
        name, slug, status ("not_started" | "active"), started, last_run, last_rebalance,
        next_rebalance, rebalance (ISO dates or None; frequency of the last rebalance) where
        next_rebalance is the first scheduled rebalance date after ``last_rebalance``,
        rebalance_due (True when next_rebalance is on or before ``last_run``: a run on or after it
        only marked to market, so the next ``rebalance()`` trades whatever its date; None before
        the start),
        initial_capital, cash, nav, long_value, short_value, gross_exposure_pct, net_exposure_pct,
        since_start_return_pct (NAV / initial capital - 1, %, None before the start),
        backtest_expected_return_pct (compounded backtest strategy return over (started,
        last_run] when ``backtest`` - default: the saved one - covers that window, else None),
        backtest_window ({start, end, n_periods} of the periods used, or None),
        max_drawdown_pct (of the NAV path, <= 0, None before the start), total_costs, n_trades,
        nav_history [(date_iso, nav), ...] oldest first,
        holdings [{ticker, shares, side, price, price_date, market_value, weight, target_weight,
        price_missing_since}] largest first (``price_missing_since``: ISO date of the first run
        without a price while the name is carried at a stale mark, else None),
        trades [{date, ticker, side, shares, price, notional, cost, reason}] latest first,
        runs [(date_iso, rebalanced)], warnings (of the last run),
        backtest_comparison_notes (conventions that make the paper return differ from
        backtest_expected_return_pct even with identical holdings, e.g. the backtest's interest on
        idle cash; empty without a backtest),
        costs_bps (configured, else the saved spec's), allow_fractional, ledger_path.
        """
        led = self._reload()
        nav = self._nav(led)
        holdings = self._holdings(led, nav)
        long_value = float(sum(h["market_value"] for h in holdings if h["market_value"] > 0))
        short_value = float(sum(h["market_value"] for h in holdings if h["market_value"] < 0))
        last_run = led.runs[-1][0] if led.runs else None
        started = led.started

        since_start = (nav / led.initial_capital - 1.0) * 100.0 if started is not None else None
        max_dd = None
        if started is not None and led.nav_history:
            navs = np.array([led.initial_capital] + [v for _, v in led.nav_history], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = pd.Series(navs[1:] / navs[:-1] - 1.0)
            max_dd = float(min(drawdown_series(rets.replace([np.inf, -np.inf], np.nan)).min(), 0.0)) * 100.0

        window = None
        if started is not None and last_run is not None:
            if backtest is None:
                try:
                    backtest = self.store.load_backtest(self.name)
                except StrategyNotFoundError:
                    backtest = None
            if backtest is not None:
                window = backtest_return_over_window(backtest, started, last_run)
        comparison_notes: list[str] = []
        if backtest is not None and backtest_credits_cash_interest(backtest):
            comparison_notes.append(
                "The backtest credits the 1-month T-bill rate on idle cash and on short-sale proceeds (2x for a -1 "
                "short); paper cash earns 0%, so while the book holds cash or is net short the paper return lags the "
                "backtest by about RF x the cash share."
            )

        # The next scheduled rebalance after the last one actually made. When a run on or after that
        # date only marked to market (mark_to_market, or an outage), it is still pending: the next
        # rebalance() trades whatever its date, so the dashboard must show it as due, not the
        # following period's date.
        nxt = None
        due = None
        if led.rebalance_frequency and led.last_rebalance is not None:
            nxt_d = next_rebalance_date(led.last_rebalance, led.rebalance_frequency)
            nxt = nxt_d.isoformat()
            due = last_run is not None and nxt_d <= last_run

        def _iso(d: date | None) -> str | None:
            return d.isoformat() if d is not None else None

        return {
            "name": self.name,
            "slug": self.path.name,
            "status": "active" if started is not None else "not_started",
            "started": _iso(started),
            "last_run": _iso(last_run),
            "last_rebalance": _iso(led.last_rebalance),
            "next_rebalance": nxt,
            "rebalance_due": due,
            "rebalance": led.rebalance_frequency,
            "initial_capital": led.initial_capital,
            "cash": led.cash,
            "nav": nav,
            "long_value": long_value,
            "short_value": short_value,
            "gross_exposure_pct": (long_value - short_value) / nav * 100.0 if nav > 0 else None,
            "net_exposure_pct": (long_value + short_value) / nav * 100.0 if nav > 0 else None,
            "since_start_return_pct": since_start,
            "backtest_expected_return_pct": window["return_pct"] if window else None,
            "backtest_window": (
                {"start": window["start"], "end": window["end"], "n_periods": window["n_periods"]} if window else None
            ),
            "max_drawdown_pct": max_dd,
            "total_costs": led.total_costs,
            "n_trades": len(led.trades),
            "nav_history": [(d.isoformat(), float(v)) for d, v in led.nav_history],
            "holdings": holdings,
            "trades": [
                {
                    "date": tr.as_of.isoformat(),
                    "ticker": tr.ticker,
                    "side": tr.side,
                    "shares": tr.shares,
                    "price": tr.price,
                    "notional": tr.notional,
                    "cost": tr.cost,
                    "reason": tr.reason,
                }
                for tr in reversed(led.trades)
            ],
            "runs": [(d.isoformat(), bool(r)) for d, r in led.runs],
            "warnings": list(led.last_warnings),
            "backtest_comparison_notes": comparison_notes,
            "costs_bps": self._effective_costs_bps(),
            "allow_fractional": self.allow_fractional,
            "ledger_path": str(self.ledger_path),
        }
