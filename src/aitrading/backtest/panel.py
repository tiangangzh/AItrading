"""Preloaded-data provider wrapper for backtests: download the price history once, serve it many times.

A backtest evaluates features at every rebalance date, and each ``FeatureEngine.build`` asks the
provider for ~420 calendar days of prices (plus the benchmark) ending on that date. Against a
real data source that would mean one download per rebalance. :class:`PreloadedProvider` wraps a
``MarketDataProvider`` and

* serves ``get_price_history`` / ``get_benchmark_history`` from panels preloaded once for the whole
  backtest window, sliced to the requested ``[start, end]`` (inclusive, like the providers);
* fetches what the preload lacks once over the whole window and keeps it (tickers missing from the
  panel, a benchmark symbol requested for the first time), so later requests are served locally;
* delegates any request that reaches outside the preloaded window unchanged (never truncating
  what the provider would have returned);
* memoises the point-in-time snapshot calls (``get_universe``, ``get_fundamentals``,
  ``get_estimates``, ``get_short_interest``, ``get_options_summary``) per (method, tickers, as_of)
  in a bounded LRU memo, which a runner may share between several backtests on the same provider
  (e.g. the replication suite's re-runs);
* passes every other attribute through to the wrapped provider (``name``, ``capabilities``,
  ``boundary``, ``warnings``, ``is_stale``, ...), so it is a drop-in ``MarketDataProvider``.

Look-ahead: slicing returns only rows dated within the requested range, exactly as the wrapped
provider would, so a consumer asking for data up to ``as_of`` never sees a later row. Memoised
frames are returned as copies so a consumer cannot alter what the next date sees.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from typing import Any, Hashable

import numpy as np
import pandas as pd

from aitrading.data.base import PricePanel

__all__ = ["PreloadedProvider", "slice_panel"]

_FIELDS = ("open", "high", "low", "close", "volume")


def _d(x: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(x, str):
        x = pd.Timestamp(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date()
    return x


def slice_panel(prices: PricePanel, tickers: list[str], start: date, end: date) -> PricePanel:
    """Rows dated in ``[start, end]`` and the columns ``tickers`` (in that order) of ``prices``."""
    idx = prices.close.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    cols = pd.Index([str(t) for t in tickers], name=prices.close.columns.name)
    frames = {f: getattr(prices, f).loc[mask].reindex(columns=cols) for f in _FIELDS}
    return PricePanel(frames["open"], frames["high"], frames["low"], frames["close"], frames["volume"])


def _merge_panels(a: PricePanel, b: PricePanel) -> PricePanel:
    """Columns of ``b`` appended to ``a`` (outer join on dates; ``a`` wins on duplicate tickers)."""
    new = [c for c in b.close.columns if c not in a.close.columns]
    if not new:
        return a
    frames = {}
    for f in _FIELDS:
        fa, fb = getattr(a, f), getattr(b, f)[new]
        frames[f] = pd.concat([fa, fb], axis=1).sort_index()
    return PricePanel(frames["open"], frames["high"], frames["low"], frames["close"], frames["volume"])


class PreloadedProvider:
    """``MarketDataProvider`` wrapper serving prices / benchmark from preloaded panels (module docstring).

    Args:
        provider: the wrapped provider.
        prices: the preloaded panel (dates x tickers).
        benchmark: optional preloaded benchmark close for ``benchmark_symbol`` (None = the provider's
            default benchmark, i.e. ``get_benchmark_history(..., symbol=None)``).
        benchmark_symbol: the symbol ``benchmark`` represents (None = the provider default).
        window: ``(start, end)`` the preload covers; requests inside it are served locally. Defaults
            to the first / last date of ``prices``.
        memo: an (ordered) dict to memoise snapshot calls in; pass the same dict to several wrappers
            of the same provider to share it. A private one is created when omitted.
        max_memo: maximum number of memoised snapshot results (least recently used dropped first).
    """

    def __init__(
        self,
        provider: Any,
        prices: PricePanel,
        benchmark: pd.Series | None = None,
        *,
        benchmark_symbol: str | None = None,
        window: tuple[date, date] | None = None,
        memo: "OrderedDict[Hashable, Any] | None" = None,
        max_memo: int = 4096,
    ) -> None:
        self._provider = provider
        self._prices = prices
        if window is None:
            idx = prices.close.index
            if len(idx):
                window = (idx[0].date(), idx[-1].date())
            else:
                raise ValueError("PreloadedProvider needs a window when the preloaded panel is empty")
        self.window: tuple[date, date] = (_d(window[0]), _d(window[1]))
        self._benchmarks: dict[str | None, pd.Series] = {}
        if benchmark is not None:
            self._benchmarks[benchmark_symbol] = benchmark.sort_index()
        self._memo: OrderedDict[Hashable, Any] = memo if memo is not None else OrderedDict()
        self.max_memo = int(max_memo)
        self.delegated_price_requests = 0  # requests outside the window (for diagnostics / tests)
        self.served_price_requests = 0

    # ------------------------------------------------------------------ pass-through attributes
    @property
    def wrapped(self) -> Any:
        return self._provider

    @property
    def prices(self) -> PricePanel:
        return self._prices

    @property
    def name(self) -> str:
        return str(getattr(self._provider, "name", type(self._provider).__name__))

    @property
    def capabilities(self) -> Any:
        return getattr(self._provider, "capabilities", None)

    @property
    def boundary(self) -> Any:
        return getattr(self._provider, "boundary", None)

    def __getattr__(self, item: str) -> Any:  # only called when normal lookup fails
        if item.startswith("__") or item in ("_provider", "_prices", "_benchmarks", "_memo"):
            raise AttributeError(item)
        return getattr(self._provider, item)

    # ------------------------------------------------------------------ helpers
    def _inside(self, start: date, end: date) -> bool:
        return self.window[0] <= _d(start) and _d(end) <= self.window[1]

    def _remember(self, key: Hashable, value: Any) -> Any:
        self._memo[key] = value
        self._memo.move_to_end(key)
        while len(self._memo) > self.max_memo:
            self._memo.popitem(last=False)
        return value

    def _memoised(self, key: Hashable, call) -> Any:
        if key in self._memo:
            self._memo.move_to_end(key)
            value = self._memo[key]
        else:
            value = self._remember(key, call())
        return value.copy() if isinstance(value, (pd.DataFrame, pd.Series)) else value

    @staticmethod
    def _tickers_key(tickers: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(t) for t in tickers))

    # ------------------------------------------------------------------ prices
    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        req = list(self._tickers_key(tickers))
        if not self._inside(start, end):
            self.delegated_price_requests += 1
            return self._provider.get_price_history(req, start, end)
        missing = [t for t in req if t not in self._prices.close.columns]
        if missing:  # fetch the missing tickers once over the whole window and keep them
            extra = self._provider.get_price_history(missing, self.window[0], self.window[1])
            self._prices = _merge_panels(self._prices, extra)
            for t in missing:  # a ticker the provider has nothing for stays an all-NaN column
                if t not in self._prices.close.columns:
                    self._prices = _merge_panels(self._prices, _nan_panel(self._prices.close.index, [t]))
        self.served_price_requests += 1
        return slice_panel(self._prices, req, _d(start), _d(end))

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        if not self._inside(start, end):
            return self._provider.get_benchmark_history(start, end, symbol)
        if symbol not in self._benchmarks:
            full = self._provider.get_benchmark_history(self.window[0], self.window[1], symbol)
            self._benchmarks[symbol] = pd.Series(full).sort_index()
        ser = self._benchmarks[symbol]
        idx = ser.index
        return ser.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].copy()

    # ------------------------------------------------------------------ memoised snapshots
    def get_universe(self, spec: Any, as_of: date) -> pd.DataFrame:
        try:
            spec_key: Hashable = spec.model_dump_json() if spec is not None else None
        except AttributeError:
            spec_key = repr(spec)
        key = ("get_universe", spec_key, _d(as_of))
        return self._memoised(key, lambda: self._provider.get_universe(spec, as_of))

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        key = ("get_fundamentals", self._tickers_key(tickers), _d(as_of))
        return self._memoised(key, lambda: self._provider.get_fundamentals(list(tickers), as_of))

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        key = ("get_estimates", self._tickers_key(tickers), _d(as_of))
        return self._memoised(key, lambda: self._provider.get_estimates(list(tickers), as_of))

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        key = ("get_short_interest", self._tickers_key(tickers), _d(as_of))
        return self._memoised(key, lambda: self._provider.get_short_interest(list(tickers), as_of))

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        key = ("get_options_summary", self._tickers_key(tickers), _d(as_of))
        return self._memoised(key, lambda: self._provider.get_options_summary(list(tickers), as_of))

    def get_documents(self, ticker: str, kinds: Any, start: date, end: date, limit: int = 10) -> list:
        return self._provider.get_documents(ticker, kinds, start, end, limit)


def _nan_panel(index: pd.Index, tickers: list[str]) -> PricePanel:
    frame = pd.DataFrame(np.nan, index=index, columns=pd.Index(tickers), dtype="float64")
    return PricePanel(frame, frame.copy(), frame.copy(), frame.copy(), frame.copy())
