"""Free-data provider: run the whole pipeline on a personal computer (Windows / macOS / Linux).

Sources
-------
* **Prices** - Yahoo Finance via ``yfinance`` (split- and dividend-adjusted, ``auto_adjust=True``).
* **Fundamentals** - SEC EDGAR XBRL ``companyfacts``, point-in-time on the *filing* date
  (see :mod:`aitrading.data.sec_edgar`).
* **Documents** - SEC EDGAR: 8-K item 2.02 earnings press releases (``DocumentKind.NEWS``) and
  10-Q / 10-K MD&A (``DocumentKind.FILING``). Earnings-call transcripts are not available from any
  official free source, so they are not offered (no ``Capability.TRANSCRIPTS``); the press release
  plus MD&A is the free narrative proxy. Yahoo news is deliberately not used (not point-in-time,
  unclear licensing).
* **Snapshots** - short interest (Yahoo ``info``), consensus estimates (Yahoo analysis modules) and
  option chains are *current only*. For an ``as_of`` more than ``snapshot_staleness_days`` before
  today they are returned as NaN with a warning instead of silently leaking today's values into a
  historical run. (``last_earnings_date`` is still filled from SEC 8-K filings, which are
  point-in-time.)

Everything degrades gracefully: a failed ticker or dataset becomes NaN plus a message in
``provider.warnings``; only a price request in which *no* ticker returned data raises
``ProviderError``. All responses are cached on disk (``~/.aitrading/cache`` by default).

Approximations (documented, deliberate)
--------------------------------------
* ``revenue_ntm_est`` / ``eps_ntm_est``: Yahoo gives current-fiscal-year (``0y``) and next-fiscal-year
  (``+1y``) consensus. NTM = w x FY0 + (1 - w) x FY1, where w = fraction of the current fiscal year
  still ahead of ``as_of`` (from ``info['nextFiscalYearEnd']``; 0.5 if unknown).
  ``eps_ntm_est_3m_ago`` applies the same w to the ``90daysAgo`` column of ``eps_trend`` (so the
  revision reflects analyst changes, not the calendar roll). ``revenue_ntm_est_3m_ago`` is NaN
  (Yahoo has no revenue-estimate history).
* ``iv_30d_atm``: implied volatility of the expiry nearest 30 days within 20-45 days (else the nearest
  expiry >= 7 days out), averaging the call and put at the strike nearest spot. Put/call volume and
  open interest are summed over that same single expiry (one option-chain request per ticker).
  ``iv_30d_atm_1y_high`` / ``_low`` are NaN: there is no free implied-volatility history.
* ``gics_industry`` is Yahoo's industry label (close to, but not exactly, GICS industries).
* ``market_cap`` = last adjusted close on/before ``as_of`` x SEC cover-page shares (point-in-time);
  falls back to Yahoo ``sharesOutstanding`` (current) with a warning.

Yahoo data via ``yfinance`` is for personal research use; respect Yahoo's terms. SEC EDGAR is public.
"""

from __future__ import annotations

import csv
import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data.base import Capability, PricePanel, ProviderError, ProviderUnavailable
from aitrading.data.cache import DAY, HOUR, DiskCache
from aitrading.data.sec_edgar import US_EXCHANGES, SecEdgarClient, SecNotFound
from aitrading.data.universes import load_starter_universe

PRICE_BATCH = 100
_YF_FIELDS = {F.OPEN: "Open", F.HIGH: "High", F.LOW: "Low", F.CLOSE: "Close", F.VOLUME: "Volume"}

YAHOO_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}
GICS_SECTORS = set(YAHOO_SECTOR_TO_GICS.values())

YAHOO_EXCHANGE = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ", "NASDAQ": "NASDAQ",
    "NYQ": "NYSE", "NYS": "NYSE", "NYSE": "NYSE", "ASE": "NYSE American", "AMEX": "NYSE American",
    "PCX": "NYSE Arca", "BTS": "CBOE", "CXI": "CBOE", "PNK": "OTC", "OQB": "OTC", "OQX": "OTC",
}

QUOTE_TYPE_TO_SECURITY = {"EQUITY": "common_stock", "ETF": "etf", "MUTUALFUND": "fund", "INDEX": "index"}

COUNTRY_TO_ISO2 = {
    "United States": "US", "Canada": "CA", "United Kingdom": "GB", "Ireland": "IE", "Netherlands": "NL",
    "Switzerland": "CH", "Bermuda": "BM", "Cayman Islands": "KY", "Israel": "IL", "China": "CN",
    "Germany": "DE", "France": "FR", "Japan": "JP", "Luxembourg": "LU", "Jersey": "JE", "Puerto Rico": "PR",
    "Taiwan": "TW", "Brazil": "BR", "Mexico": "MX", "Australia": "AU", "India": "IN", "Singapore": "SG",
    "Hong Kong": "HK", "Sweden": "SE", "Denmark": "DK", "Norway": "NO", "Spain": "ES", "Italy": "IT",
    "Belgium": "BE", "Argentina": "AR", "Chile": "CL", "South Korea": "KR", "Uruguay": "UY", "Greece": "GR",
    "Monaco": "MC", "Panama": "PA", "Bahamas": "BS", "Guernsey": "GG", "Isle of Man": "IM", "Finland": "FI",
    "Austria": "AT", "South Africa": "ZA", "Peru": "PE", "Colombia": "CO", "Cyprus": "CY", "Macau": "MO",
    "Kazakhstan": "KZ", "Indonesia": "ID", "Philippines": "PH", "Thailand": "TH", "Malaysia": "MY",
}

INFO_KEYS = [
    "longName", "shortName", "displayName", "quoteType", "exchange", "fullExchangeName", "currency",
    "financialCurrency", "sector", "industry", "country", "sharesOutstanding", "impliedSharesOutstanding",
    "floatShares", "sharesShort", "sharesShortPriorMonth", "dateShortInterest", "sharesShortPreviousMonthDate",
    "shortPercentOfFloat", "trailingEps", "epsTrailingTwelveMonths", "forwardEps", "targetMeanPrice",
    "numberOfAnalystOpinions", "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter",
    "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd", "currentPrice",
    "regularMarketPrice", "previousClose", "marketCap",
]

TTL_INFO = DAY
TTL_ESTIMATES = 12 * HOUR
TTL_OPTIONS = 6 * HOUR

BOUNDARY_NOTE = "Yahoo data via yfinance is for personal research use; respect Yahoo's terms. SEC EDGAR is public."
YFINANCE_MISSING = (
    'yfinance is not installed. Install the free-data extra:  pip install -e ".[free]"   '
    "(or simply: pip install yfinance)"
)


# =============================================================================================
# Small helpers (pure; unit-tested)
# =============================================================================================


def normalize_ticker(t: str) -> str:
    """Upper-case and use Yahoo/SEC share-class notation: 'brk.b' -> 'BRK-B' ('SHOP.TO' is kept)."""
    s = str(t).strip().upper().replace("/", "-")
    if "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 1 and head:
            s = f"{head}-{tail}"
    return s


def read_ticker_file(path: str | Path) -> list[str]:
    """One ticker per line (``#`` comments allowed) or a CSV with a ``ticker`` / ``symbol`` column."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"universe file {p} contains no tickers")
    first = lines[0]
    if p.suffix.lower() in (".csv", ".tsv") or any(d in first for d in (",", ";", "\t")):
        delim = "\t" if "\t" in first else (";" if ";" in first and "," not in first else ",")
        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
        col = cols.get("ticker") or cols.get("symbol")
        if col is None:
            raise ValueError(f"universe file {p}: CSV needs a 'ticker' (or 'symbol') column, found {reader.fieldnames}")
        out = [str(r.get(col) or "").strip() for r in reader]
    else:
        out = [ln.split()[0].strip() for ln in lines]
    seen: dict[str, None] = {}
    for t in out:
        if t:
            seen.setdefault(normalize_ticker(t), None)
    return list(seen)


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _epoch_to_ts(x: Any) -> pd.Timestamp:
    """Yahoo epoch seconds -> New York calendar date (tz-naive Timestamp)."""
    v = _num(x)
    if math.isnan(v) or v <= 0:
        return pd.NaT
    try:
        return pd.Timestamp(int(v), unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
    except (ValueError, OverflowError, TypeError):
        return pd.NaT


def _to_ts(x: Any) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    try:
        ts = pd.Timestamp(x)
    except (ValueError, TypeError):
        return pd.NaT
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def extract_ohlcv(raw: pd.DataFrame | None, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a ``yfinance.download`` result into canonical wide frames (dates x symbols).

    Handles every layout yfinance has produced: MultiIndex ``(Price, Ticker)`` (``group_by='column'``,
    the default and what 1.x returns even for one ticker), ``(Ticker, Price)`` (``group_by='ticker'``)
    and flat single-ticker columns (yfinance < 0.2.48 or ``multi_level_index=False``).
    Index -> tz-naive, normalised, ascending, unique dates.
    """
    syms = [s.upper() for s in symbols]
    empty_idx = pd.DatetimeIndex([], name="date")
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return {f: pd.DataFrame(index=empty_idx, columns=syms, dtype="float64") for f in _YF_FIELDS}
    df = raw.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    out: dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        levels = [set(map(str, df.columns.get_level_values(i))) for i in range(df.columns.nlevels)]
        field_level = next((i for i, lv in enumerate(levels) if "Close" in lv or "Adj Close" in lv), 0)
        for f, yname in _YF_FIELDS.items():
            if yname in levels[field_level]:
                sub = df.xs(yname, axis=1, level=field_level)
                if isinstance(sub, pd.Series):
                    sub = sub.to_frame(name=syms[0] if len(syms) == 1 else sub.name)
                if isinstance(sub.columns, pd.MultiIndex):  # more than 2 levels: keep the ticker level
                    sub.columns = sub.columns.get_level_values(-1)
                sub.columns = [str(c).upper() for c in sub.columns]
                sub = sub.loc[:, ~sub.columns.duplicated()]
            else:
                sub = pd.DataFrame(index=df.index)
            out[f] = sub.reindex(columns=syms)
    else:
        for f, yname in _YF_FIELDS.items():
            if len(syms) == 1 and yname in df.columns:
                out[f] = df[[yname]].set_axis(syms, axis=1)
            else:
                out[f] = pd.DataFrame(index=df.index, columns=syms, dtype="float64")
    for f in out:
        frame = out[f].apply(pd.to_numeric, errors="coerce").astype("float64")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame.index = pd.DatetimeIndex(frame.index, name="date").astype("datetime64[ns]")
        out[f] = frame
    return out


def choose_expiry(expiries: Iterable[str], as_of: date) -> str | None:
    """Expiry nearest 30 days within 20-45 days out, else the nearest one >= 7 days out."""
    parsed: list[tuple[int, str]] = []
    for e in expiries or []:
        try:
            d = date.fromisoformat(str(e)[:10])
        except ValueError:
            continue
        parsed.append(((d - as_of).days, str(e)))
    window = [p for p in parsed if 20 <= p[0] <= 45]
    if window:
        return min(window, key=lambda p: (abs(p[0] - 30), p[0]))[1]
    later = [p for p in parsed if p[0] >= 7]
    return min(later)[1] if later else None


def _valid_iv(x: Any) -> float:
    v = _num(x)
    return v if 0.01 < v < 5.0 else math.nan


def options_summary(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float) -> dict[str, float]:
    """ATM IV (mean of the call and put IV at the strike nearest spot) + volume / OI sums for one expiry."""
    row = {c: math.nan for c in F.OPTIONS_COLUMNS}

    def clean(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "strike" not in df.columns:
            return pd.DataFrame(columns=["strike", "impliedVolatility", "volume", "openInterest"])
        out = df.copy()
        for c in ("strike", "impliedVolatility", "volume", "openInterest"):
            out[c] = pd.to_numeric(out[c], errors="coerce") if c in out.columns else np.nan
        return out

    c, p = clean(calls), clean(puts)
    if math.isfinite(_num(spot)) and spot > 0:
        common = sorted(set(c["strike"].dropna()) & set(p["strike"].dropna()))
        strikes = common or sorted(set(c["strike"].dropna()) | set(p["strike"].dropna()))
        if strikes:
            k = min(strikes, key=lambda s: (abs(s - spot), s))

            def iv_at(df: pd.DataFrame) -> float:
                hit = df.loc[df["strike"] == k, "impliedVolatility"]
                return _valid_iv(hit.iloc[0]) if len(hit) else math.nan

            ivs = [v for v in (iv_at(c), iv_at(p)) if math.isfinite(v)]
            if ivs:
                row[F.IV_30D_ATM] = float(np.mean(ivs))
    if len(c):
        row[F.CALL_VOLUME] = float(c["volume"].sum(min_count=1))
        row[F.CALL_OPEN_INTEREST] = float(c["openInterest"].sum(min_count=1))
    if len(p):
        row[F.PUT_VOLUME] = float(p["volume"].sum(min_count=1))
        row[F.PUT_OPEN_INTEREST] = float(p["openInterest"].sum(min_count=1))
    return row


def _cell(df: Any, row: str, col: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or row not in df.index or col not in df.columns:
        return math.nan
    val = df.loc[row, col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return _num(val)


def ntm_weight(info: dict, as_of: date) -> float:
    """Fraction of the current fiscal year (Yahoo's ``0y``) still ahead of ``as_of``, in [0, 1]."""
    end = _epoch_to_ts(info.get("nextFiscalYearEnd"))
    if end is pd.NaT:
        last = _epoch_to_ts(info.get("lastFiscalYearEnd"))
        end = last + pd.Timedelta(days=365) if last is not pd.NaT else pd.NaT
    if end is pd.NaT:
        return 0.5
    w = (end - pd.Timestamp(as_of)).days / 365.25
    return float(min(1.0, max(0.0, w)))


def _blend(fy0: float, fy1: float, w: float) -> float:
    if math.isfinite(fy0) and math.isfinite(fy1):
        return w * fy0 + (1.0 - w) * fy1
    if math.isfinite(fy0) and w >= 0.5:
        return fy0
    if math.isfinite(fy1) and w <= 0.5:
        return fy1
    return math.nan


def estimates_from_yahoo(info: dict, earnings_estimate: Any, revenue_estimate: Any, eps_trend: Any,
                         earnings_history: Any, calendar: Any, as_of: date,
                         last_earnings_date: date | None = None) -> dict[str, Any]:
    """One ``fields.ESTIMATE_COLUMNS`` row from yfinance's analysis objects (see module docstring)."""
    info = info or {}
    w = ntm_weight(info, as_of)
    row: dict[str, Any] = {c: math.nan for c in F.ESTIMATE_COLUMNS}

    row[F.REVENUE_NTM_EST] = _blend(_cell(revenue_estimate, "0y", "avg"), _cell(revenue_estimate, "+1y", "avg"), w)
    eps = _blend(_cell(earnings_estimate, "0y", "avg"), _cell(earnings_estimate, "+1y", "avg"), w)
    if not math.isfinite(eps):
        eps = _blend(_cell(eps_trend, "0y", "current"), _cell(eps_trend, "+1y", "current"), w)
    row[F.EPS_NTM_EST] = eps
    row[F.EPS_NTM_EST_3M_AGO] = _blend(_cell(eps_trend, "0y", "90daysAgo"), _cell(eps_trend, "+1y", "90daysAgo"), w)
    row[F.REVENUE_NTM_EST_3M_AGO] = math.nan  # no free revenue-estimate history

    eps_ttm = _num(info.get("trailingEps"))
    row[F.EPS_TTM] = eps_ttm if math.isfinite(eps_ttm) else _num(info.get("epsTrailingTwelveMonths"))
    n = _cell(earnings_estimate, "0y", "numberOfAnalysts")
    if not math.isfinite(n):
        n = _cell(earnings_estimate, "+1y", "numberOfAnalysts")
    row[F.NUM_ANALYSTS] = n if math.isfinite(n) else _num(info.get("numberOfAnalystOpinions"))
    row[F.TARGET_PRICE_MEAN] = _num(info.get("targetMeanPrice"))

    # Last surprise: latest reported quarter (index = fiscal quarter end) on/before as_of.
    if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
        eh = earnings_history.copy()
        try:
            eh.index = pd.DatetimeIndex(pd.to_datetime(eh.index)).tz_localize(None)
        except (TypeError, ValueError):
            eh.index = pd.DatetimeIndex(pd.to_datetime(eh.index, utc=True)).tz_localize(None)
        eh = eh.sort_index()
        eh = eh[eh.index <= pd.Timestamp(as_of)]
        for _, r in eh.iloc[::-1].iterrows():
            act, est = _num(r.get("epsActual")), _num(r.get("epsEstimate"))
            if math.isfinite(act) and math.isfinite(est):
                row[F.LAST_EPS_SURPRISE] = (act - est) / abs(est) if est != 0 else math.nan
                break
            sp = _num(r.get("surprisePercent"))
            if math.isfinite(sp):
                row[F.LAST_EPS_SURPRISE] = sp  # Yahoo's raw value is already a fraction
                break

    asof_ts = pd.Timestamp(as_of)
    last = _to_ts(last_earnings_date) if last_earnings_date is not None else pd.NaT
    nxt = pd.NaT
    cal_dates = []
    if isinstance(calendar, dict):
        cal_dates = [_to_ts(d) for d in (calendar.get("Earnings Date") or [])]
    future = sorted(d for d in cal_dates if d is not pd.NaT and d >= asof_ts)
    if future:
        nxt = future[0]
    for key in ("earningsTimestamp", "earningsTimestampStart"):
        ts = _epoch_to_ts(info.get(key))
        if ts is pd.NaT:
            continue
        if ts >= asof_ts and nxt is pd.NaT:
            nxt = ts
        elif ts < asof_ts and last is pd.NaT:
            last = ts
    row[F.LAST_EARNINGS_DATE] = last
    row[F.NEXT_EARNINGS_DATE] = nxt
    return row


def _frame(rows: dict[str, dict], columns: list[str], tickers: list[str], date_cols: Iterable[str] = ()) -> pd.DataFrame:
    """Canonical frame indexed by ticker (all requested tickers present; missing -> NaN / NaT)."""
    date_cols = set(date_cols)
    df = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    df = df.reindex(index=list(tickers), columns=columns)
    for c in columns:
        if c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df.index = pd.Index(list(tickers), name="ticker")
    return df


def _row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v is pd.NaT:
            out[k] = None
        elif isinstance(v, (pd.Timestamp, datetime, date)):
            out[k] = pd.Timestamp(v).isoformat()
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            fv = float(v)
            out[k] = fv if math.isfinite(fv) else None
        else:
            out[k] = v
    return out


def _row_from_json(d: dict[str, Any], date_cols: Iterable[str]) -> dict[str, Any]:
    date_cols = set(date_cols)
    return {k: (_to_ts(v) if k in date_cols else (math.nan if v is None else v)) for k, v in d.items()}


# =============================================================================================
# Provider
# =============================================================================================


class FreeDataProvider:
    """``MarketDataProvider`` over free sources (Yahoo via yfinance + SEC EDGAR) for running on a PC."""

    name = "free"

    def __init__(
        self,
        tickers: list[str] | None = None,
        *,
        universe_file: str | Path | None = None,
        cache: DiskCache | None = None,
        sec: SecEdgarClient | None = None,
        benchmark: str = "SPY",
        max_workers: int = 8,
        snapshot_staleness_days: int = 5,
        yf_module: Any | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        """
        Args:
            tickers: explicit universe (takes precedence).
            universe_file: text file with one ticker per line, or a CSV with a ``ticker`` column.
            cache: disk cache shared by the Yahoo and SEC layers (default ``~/.aitrading/cache``).
            sec: SEC client (default: reads ``SEC_USER_AGENT``, shares ``cache``).
            benchmark: symbol for ``get_benchmark_history`` (default SPY).
            max_workers: thread-pool size for per-ticker requests (SEC is additionally throttled to 8 req/s).
            snapshot_staleness_days: snapshot datasets are blank for ``as_of`` older than this.
            yf_module: inject a ``yfinance``-compatible module (tests / custom sessions).
            today: clock for the staleness rule (tests).
        """
        self.cache = cache if cache is not None else DiskCache()
        self.sec = sec if sec is not None else SecEdgarClient(cache=self.cache)
        self.benchmark = normalize_ticker(benchmark)
        self.max_workers = max(1, int(max_workers))
        self.snapshot_staleness_days = int(snapshot_staleness_days)
        self.capabilities: set[Capability] = {
            Capability.PRICES, Capability.FUNDAMENTALS, Capability.ESTIMATES, Capability.SHORT_INTEREST,
            Capability.OPTIONS, Capability.NEWS, Capability.FILINGS,
        }
        self.boundary = DataBoundary(provider="free", note=BOUNDARY_NOTE)
        self.warnings: list[str] = []
        self._warn_lock = threading.Lock()
        self._yf = yf_module
        self._today = today or date.today
        self._starter = load_starter_universe()
        if tickers:
            seen: dict[str, None] = {}
            for t in tickers:
                if t and str(t).strip():
                    seen.setdefault(normalize_ticker(t), None)
            self._tickers = list(seen)
        elif universe_file is not None:
            self._tickers = read_ticker_file(universe_file)
        else:
            self._tickers = list(self._starter.index)
        self._info_memo: dict[str, dict] = {}
        self._info_lock = threading.Lock()
        self._closes: dict[str, pd.Series] = {}
        self._close_cover: dict[str, list[tuple[date, date]]] = {}
        self._close_lock = threading.Lock()

    # ------------------------------------------------------------------ housekeeping
    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    def _warn(self, msg: str) -> None:
        with self._warn_lock:
            if msg not in self.warnings:
                self.warnings.append(msg)

    def _yfm(self) -> Any:
        if self._yf is None:
            try:
                import yfinance  # noqa: PLC0415 - optional dependency, imported lazily
            except ImportError as e:
                raise ProviderUnavailable(YFINANCE_MISSING) from e
            self._yf = yfinance
        return self._yf

    def diagnostics(self) -> list[str]:
        """Setup problems that would blank out data (empty list = ready)."""
        problems = []
        try:
            self._yfm()
        except ProviderUnavailable as e:
            problems.append(str(e))
        try:
            self.sec.check_user_agent()
        except ProviderUnavailable as e:
            problems.append(str(e))
        return problems

    def is_stale(self, as_of: date) -> bool:
        """True if snapshot-only datasets (estimates, short interest, options) cannot serve ``as_of``."""
        return as_of < self._today() - timedelta(days=self.snapshot_staleness_days)

    def _stale_warning(self, what: str, as_of: date) -> None:
        self._warn(
            f"as_of {as_of} is more than {self.snapshot_staleness_days} days before today ({self._today()}): the free "
            f"provider cannot give point-in-time {what} for historical dates (Yahoo only serves current "
            "snapshots), so these fields are left blank (NaN)."
        )

    def _parallel(self, fn: Callable[[str], Any], items: list[str]) -> dict[str, tuple[Any, BaseException | None]]:
        out: dict[str, tuple[Any, BaseException | None]] = {}
        if not items:
            return out
        if len(items) == 1 or self.max_workers == 1:
            for it in items:
                try:
                    out[it] = (fn(it), None)
                except Exception as e:  # noqa: BLE001
                    out[it] = (None, e)
            return out
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    out[it] = (fut.result(), None)
                except Exception as e:  # noqa: BLE001
                    out[it] = (None, e)
        return out

    def _summarise_errors(self, what: str, errors: dict[str, BaseException]) -> None:
        if not errors:
            return
        unavailable = next((e for e in errors.values() if isinstance(e, ProviderUnavailable)), None)
        if unavailable is not None:
            self._warn(str(unavailable))
        not_found = sorted(t for t, e in errors.items() if isinstance(e, SecNotFound))
        other = sorted(t for t, e in errors.items() if not isinstance(e, (SecNotFound, ProviderUnavailable)))
        if not_found:
            self._warn(f"{what}: no SEC data for {len(not_found)} ticker(s) ({', '.join(not_found[:20])}) - "
                       "foreign filer, fund, or no XBRL financials; left as NaN.")
        if other:
            sample = errors[other[0]]
            self._warn(f"{what}: failed for {len(other)} ticker(s) ({', '.join(other[:20])}); left as NaN. "
                       f"First error: {type(sample).__name__}: {sample}")

    # ------------------------------------------------------------------ Yahoo: info
    def _info(self, sym: str) -> dict:
        with self._info_lock:
            if sym in self._info_memo:
                return self._info_memo[sym]
        key = f"yf:info:v1:{sym}"
        info = self.cache.get_json(key)
        if not isinstance(info, dict):
            raw = self._yfm().Ticker(sym).info or {}
            info = {k: raw[k] for k in INFO_KEYS if k in raw and raw[k] is not None}
            if any(k in info for k in ("quoteType", "longName", "shortName", "sector")):
                self.cache.put_json(key, info, TTL_INFO)
            else:
                raise ProviderError(f"Yahoo returned no quote data for {sym}")
        with self._info_lock:
            self._info_memo[sym] = info
        return info

    def _infos(self, syms: list[str], what: str) -> dict[str, dict]:
        res = self._parallel(self._info, syms)
        errors = {s: e for s, (_, e) in res.items() if e is not None}
        if errors:
            self._summarise_errors(f"Yahoo quote info ({what})", errors)
        return {s: (v or {}) for s, (v, e) in res.items() if e is None}

    # ------------------------------------------------------------------ Yahoo: prices
    def _price_ttl(self, end: date) -> float:
        return 1 * HOUR if end >= self._today() - timedelta(days=1) else 7 * DAY

    def _call_download(self, batch: list[str], start: date, end: date) -> pd.DataFrame:
        yf = self._yfm()
        kwargs = dict(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                      auto_adjust=True, actions=False, threads=True, progress=False, group_by="column")
        try:
            return yf.download(batch, multi_level_index=True, **kwargs)
        except TypeError:  # yfinance < 0.2.48 has no multi_level_index
            return yf.download(batch, **kwargs)

    def _download_batch(self, batch: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        key = f"yf:ohlcv:v1:{start.isoformat()}:{end.isoformat()}:{','.join(batch)}"
        cached = self.cache.get_frame(key)
        if cached is not None:
            try:
                return {f: cached[f].reindex(columns=batch).astype("float64") for f in _YF_FIELDS}
            except (KeyError, ValueError, TypeError):
                pass
        try:
            raw = self._call_download(batch, start, end)
        except ProviderUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - yfinance raises many types; degrade to NaN
            self._warn(f"Yahoo price download failed for {len(batch)} ticker(s) ({type(e).__name__}: {e}).")
            raw = None
        fields = extract_ohlcv(raw, batch)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        fields = {f: df.loc[(df.index >= lo) & (df.index <= hi)] for f, df in fields.items()}
        n_ok = int(fields[F.CLOSE].notna().any().sum())
        if n_ok:
            ttl = self._price_ttl(end) if n_ok == len(batch) else min(self._price_ttl(end), HOUR)
            self.cache.put_frame(key, pd.concat(fields, axis=1), ttl)
        return fields

    def _download(self, syms: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        syms = sorted(set(syms))
        parts: dict[str, list[pd.DataFrame]] = {f: [] for f in _YF_FIELDS}
        for i in range(0, len(syms), PRICE_BATCH):
            res = self._download_batch(syms[i: i + PRICE_BATCH], start, end)
            for f in _YF_FIELDS:
                parts[f].append(res[f])
        out = {}
        for f in _YF_FIELDS:
            frame = pd.concat(parts[f], axis=1).sort_index() if parts[f] else pd.DataFrame(columns=syms)
            out[f] = frame.reindex(columns=syms).astype("float64")
        idx = out[F.CLOSE].index
        for f in out:
            out[f] = out[f].reindex(idx)
        self._remember_closes(out[F.CLOSE], start, end)
        return out

    def _remember_closes(self, close: pd.DataFrame, start: date, end: date) -> None:
        """Keep adjusted closes (merged across downloads) to answer 'last close on/before as_of'."""
        with self._close_lock:
            for s in close.columns:
                series = close[s].dropna()
                if series.empty:
                    continue
                old = self._closes.get(s)
                self._closes[s] = series if old is None else series.combine_first(old).sort_index()
                self._close_cover.setdefault(s, []).append((start, end))

    def _close_covered(self, sym: str, as_of: date) -> bool:
        return any(a <= as_of - timedelta(days=5) and b >= as_of for a, b in self._close_cover.get(sym, []))

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        req = list(dict.fromkeys(str(t) for t in tickers))
        if not req:
            empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), dtype="float64")
            return PricePanel(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
        sym_of = {t: normalize_ticker(t) for t in req}
        data = self._download(list(sym_of.values()), start, end)
        close = data[F.CLOSE]
        syms = sorted(set(sym_of.values()))
        failed = [s for s in syms if close[s].isna().all()]
        if len(failed) == len(syms):
            raise ProviderError(
                f"No price data from Yahoo Finance for any of {len(syms)} ticker(s) ({', '.join(syms[:10])}"
                f"{', ...' if len(syms) > 10 else ''}) between {start} and {end}. Check the internet connection and "
                "the symbols, or retry later (Yahoo rate-limits heavy use).")
        if failed:
            self._warn(f"No Yahoo price data for {len(failed)} ticker(s) ({', '.join(failed[:20])}"
                       f"{', ...' if len(failed) > 20 else ''}); their prices are NaN.")
        frames = {}
        for f, df in data.items():
            out = df.reindex(columns=[sym_of[t] for t in req])
            out.columns = pd.Index(req)
            frames[f] = out
        return PricePanel(frames[F.OPEN], frames[F.HIGH], frames[F.LOW], frames[F.CLOSE], frames[F.VOLUME])

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        sym = normalize_ticker(symbol or self.benchmark)
        data = self._download([sym], start, end)
        s = data[F.CLOSE][sym]
        if s.isna().all():
            raise ProviderError(f"No Yahoo price data for benchmark {sym} between {start} and {end}.")
        return s.rename(sym)

    def _last_closes(self, syms: list[str], as_of: date) -> dict[str, float]:
        with self._close_lock:
            need = [s for s in syms if not self._close_covered(s, as_of)]
        if need:
            try:
                self._download(need, as_of - timedelta(days=10), as_of)
            except ProviderUnavailable as e:
                self._warn(str(e))
        out = {}
        ts = pd.Timestamp(as_of)
        with self._close_lock:
            for s in syms:
                series = self._closes.get(s)
                if series is None:
                    out[s] = math.nan
                    continue
                series = series[series.index <= ts]
                out[s] = float(series.iloc[-1]) if len(series) else math.nan
        missing = [s for s in syms if not math.isfinite(out[s])]
        if missing:
            self._warn(f"No close on/before {as_of} for {len(missing)} ticker(s) ({', '.join(missing[:20])}); "
                       "market_cap is NaN.")
        return out

    # ------------------------------------------------------------------ universe
    def _sec_refs(self, syms: list[str]) -> dict[str, Any]:
        try:
            return {s: self.sec.lookup(s) for s in syms}
        except ProviderUnavailable as e:
            self._warn(str(e))
        except ProviderError as e:
            self._warn(f"SEC ticker map unavailable ({e}); exchange/country from Yahoo only.")
        return {}

    def _sec_shares(self, syms: list[str], as_of: date) -> dict[str, float]:
        try:
            self.sec.check_user_agent()
        except ProviderUnavailable:
            return {}  # already warned by _sec_refs
        res = self._parallel(lambda s: self.sec.shares_outstanding(s, as_of), syms)
        errors = {s: e for s, (_, e) in res.items() if e is not None and not isinstance(e, SecNotFound)}
        self._summarise_errors("SEC shares outstanding", errors)
        return {s: v for s, (v, e) in res.items() if e is None and v is not None and math.isfinite(v) and v > 0}

    def get_universe(self, spec: Any, as_of: date) -> pd.DataFrame:
        """Universe rows for ``self.tickers``, filtered by the spec's country / security types / excluded
        sectors (unknown values are kept). Price and liquidity floors are left to the feature engine."""
        syms = self.tickers
        infos = self._infos(syms, "universe")
        refs = self._sec_refs(syms)
        sec_shares = self._sec_shares(syms, as_of) if refs else {}
        closes = self._last_closes(syms, as_of)
        info_shares_used: list[str] = []
        assumed_common: list[str] = []
        rows: dict[str, dict] = {}
        for s in syms:
            info = infos.get(s, {})
            ref = refs.get(s)
            starter = self._starter.loc[s] if s in self._starter.index else None
            name = (starter["name"] if starter is not None else None) or info.get("longName") or info.get("shortName") \
                or (ref.name if ref else None) or s
            sector = starter["gics_sector"] if starter is not None and starter["gics_sector"] in GICS_SECTORS else None
            if sector is None:
                ys = info.get("sector")
                sector = YAHOO_SECTOR_TO_GICS.get(ys, ys if ys in GICS_SECTORS else None)
            exchange = (ref.exchange if ref and ref.exchange else None) or YAHOO_EXCHANGE.get(str(info.get("exchange", "")).upper()) \
                or info.get("fullExchangeName")
            if exchange in US_EXCHANGES:
                country = "US"
            elif info.get("country"):
                country = COUNTRY_TO_ISO2.get(info["country"], info["country"] if len(str(info["country"])) == 2 else None)
            else:
                country = "US" if starter is not None else None
            sec_type = QUOTE_TYPE_TO_SECURITY.get(str(info.get("quoteType", "")).upper())
            if sec_type is None:
                sec_type = "common_stock"
                if starter is None:
                    assumed_common.append(s)
            currency = info.get("currency") or ("USD" if country == "US" else None)
            shares = sec_shares.get(s, math.nan)
            if not math.isfinite(shares):
                shares = _num(info.get("sharesOutstanding"))
                if math.isfinite(shares) and shares > 0:
                    info_shares_used.append(s)
            mcap = closes.get(s, math.nan) * shares if math.isfinite(shares) else math.nan
            rows[s] = {
                F.NAME: name, F.GICS_SECTOR: sector, F.GICS_INDUSTRY: info.get("industry"), F.EXCHANGE: exchange,
                F.COUNTRY: country, F.CURRENCY: currency, F.SECURITY_TYPE: sec_type, F.MARKET_CAP: mcap, F.VENDOR_ID: s,
            }
        if info_shares_used:
            self._warn(f"market_cap for {len(info_shares_used)} ticker(s) ({', '.join(info_shares_used[:20])}) uses "
                       "Yahoo's current sharesOutstanding (SEC point-in-time share count unavailable).")
        if assumed_common:
            self._warn(f"Security type unknown for {', '.join(assumed_common[:20])}; assumed common stock.")
        df = pd.DataFrame.from_dict(rows, orient="index").reindex(index=syms, columns=F.UNIVERSE_COLUMNS)
        df[F.MARKET_CAP] = pd.to_numeric(df[F.MARKET_CAP], errors="coerce").astype("float64")
        for c in F.UNIVERSE_COLUMNS:
            if c != F.MARKET_CAP:
                df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        df.index = pd.Index(syms, name="ticker")
        return self._apply_universe_spec(df, spec)

    @staticmethod
    def _apply_universe_spec(df: pd.DataFrame, spec: Any) -> pd.DataFrame:
        if spec is None or df.empty:
            return df
        keep = pd.Series(True, index=df.index)
        country = getattr(spec, "country", None)
        if country:
            c = df[F.COUNTRY]
            keep &= c.isna() | (c.astype(str).str.upper() == str(country).upper())
        types = getattr(spec, "security_types", None)
        if types:
            t = df[F.SECURITY_TYPE]
            keep &= t.isna() | t.isin(list(types))
        excl = {str(x).strip().lower() for x in (getattr(spec, "exclude_sectors", None) or [])}
        if excl:
            keep &= ~df[F.GICS_SECTOR].astype(str).str.strip().str.lower().isin(excl)
        return df.loc[keep]

    # ------------------------------------------------------------------ SEC fundamentals
    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        date_cols = (F.PERIOD_END, F.REPORT_DATE)
        try:
            self.sec.check_user_agent()
        except ProviderUnavailable as e:
            self._warn(str(e))
            return _frame({}, F.FUNDAMENTAL_COLUMNS, tickers, date_cols)
        res = self._parallel(lambda t: self.sec.fundamentals(t, as_of), list(dict.fromkeys(tickers)))
        rows = {t: v for t, (v, e) in res.items() if e is None and v is not None}
        self._summarise_errors("SEC fundamentals", {t: e for t, (_, e) in res.items() if e is not None})
        return _frame(rows, F.FUNDAMENTAL_COLUMNS, tickers, date_cols)

    def _sec_last_earnings(self, tickers: list[str], as_of: date) -> dict[str, date]:
        try:
            self.sec.check_user_agent()
        except ProviderUnavailable:
            return {}
        res = self._parallel(lambda t: self.sec.last_earnings_release_date(t, as_of), tickers)
        return {t: v for t, (v, e) in res.items() if e is None and v is not None}

    # ------------------------------------------------------------------ Yahoo snapshots
    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        date_cols = (F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE)
        tickers = list(dict.fromkeys(tickers))
        sec_last = self._sec_last_earnings(tickers, as_of)
        if self.is_stale(as_of):
            self._stale_warning("consensus estimates", as_of)
            rows = {t: {F.LAST_EARNINGS_DATE: _to_ts(d)} for t, d in sec_last.items()}
            return _frame(rows, F.ESTIMATE_COLUMNS, tickers, date_cols)
        res = self._parallel(lambda t: self._estimates_one(t, as_of, sec_last.get(t)), tickers)
        self._summarise_errors("Yahoo consensus estimates", {t: e for t, (_, e) in res.items() if e is not None})
        rows = {t: v for t, (v, e) in res.items() if e is None and v is not None}
        for t, d in sec_last.items():  # SEC 8-K date is authoritative when Yahoo failed
            rows.setdefault(t, {F.LAST_EARNINGS_DATE: _to_ts(d)})
        return _frame(rows, F.ESTIMATE_COLUMNS, tickers, date_cols)

    def _estimates_one(self, ticker: str, as_of: date, sec_last: date | None) -> dict[str, Any]:
        sym = normalize_ticker(ticker)
        date_cols = (F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE)
        key = f"yf:estimates:v1:{sym}:{as_of.isoformat()}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            row = _row_from_json(cached, date_cols)
        else:
            info = self._info(sym)
            t = self._yfm().Ticker(sym)

            def grab(attr: str) -> Any:
                try:
                    return getattr(t, attr)
                except Exception:  # noqa: BLE001 - one missing module must not sink the others
                    return None

            row = estimates_from_yahoo(info, grab("earnings_estimate"), grab("revenue_estimate"), grab("eps_trend"),
                                       grab("earnings_history"), grab("calendar"), as_of)
            self.cache.put_json(key, _row_to_json(row), TTL_ESTIMATES)
        if sec_last is not None:
            row[F.LAST_EARNINGS_DATE] = _to_ts(sec_last)
        return row

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        date_cols = (F.SI_SETTLEMENT_DATE,)
        tickers = list(dict.fromkeys(tickers))
        if self.is_stale(as_of):
            self._stale_warning("short interest", as_of)
            return _frame({}, F.SHORT_INTEREST_COLUMNS, tickers, date_cols)
        sym_of = {t: normalize_ticker(t) for t in tickers}
        infos = self._infos(sorted(set(sym_of.values())), "short interest")
        rows = {}
        for t, s in sym_of.items():
            info = infos.get(s)
            if not info:
                continue
            rows[t] = {
                F.SHORT_INTEREST_SHARES: _num(info.get("sharesShort")),
                F.SHORT_INTEREST_SHARES_1M_AGO: _num(info.get("sharesShortPriorMonth")),
                F.FLOAT_SHARES: _num(info.get("floatShares")),
                F.SI_SETTLEMENT_DATE: _epoch_to_ts(info.get("dateShortInterest")),
            }
        return _frame(rows, F.SHORT_INTEREST_COLUMNS, tickers, date_cols)

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))
        if self.is_stale(as_of):
            self._stale_warning("option implied volatility / volume / open interest", as_of)
            return _frame({}, F.OPTIONS_COLUMNS, tickers)
        res = self._parallel(lambda t: self._options_one(t, as_of), tickers)
        self._summarise_errors("Yahoo option chains", {t: e for t, (_, e) in res.items() if e is not None})
        rows = {t: v for t, (v, e) in res.items() if e is None and v is not None}
        return _frame(rows, F.OPTIONS_COLUMNS, tickers)

    def _options_one(self, ticker: str, as_of: date) -> dict[str, float] | None:
        sym = normalize_ticker(ticker)
        key = f"yf:options:v1:{sym}:{as_of.isoformat()}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return _row_from_json(cached, ())
        t = self._yfm().Ticker(sym)
        expiry = choose_expiry(t.options, as_of)  # 1 request
        if expiry is None:
            return None  # no listed options
        chain = t.option_chain(expiry)  # 1 request
        underlying = getattr(chain, "underlying", None) or {}
        spot = _num(underlying.get("regularMarketPrice")) if isinstance(underlying, dict) else math.nan
        if not math.isfinite(spot):
            info = self._info_memo.get(sym, {})
            spot = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
        if not math.isfinite(spot):
            with self._close_lock:
                series = self._closes.get(sym)
            spot = float(series.iloc[-1]) if series is not None and len(series) else math.nan
        row = options_summary(getattr(chain, "calls", None), getattr(chain, "puts", None), spot)
        self.cache.put_json(key, _row_to_json(row), TTL_OPTIONS)
        return row

    # ------------------------------------------------------------------ documents
    def get_documents(self, ticker: str, kinds: set[DocumentKind], start: date, end: date, limit: int = 10) -> list[Document]:
        """NEWS -> SEC 8-K earnings press releases; FILING -> 10-Q/10-K MD&A. Newest first, filed in [start, end]."""
        kinds = set(kinds)
        if DocumentKind.TRANSCRIPT in kinds:
            self._warn("Earnings-call transcripts are not available from free sources; the free provider supplies "
                       "8-K earnings press releases (news) and 10-Q/10-K MD&A (filings) instead.")
        if not kinds & {DocumentKind.NEWS, DocumentKind.FILING}:
            return []
        try:
            return self.sec.documents(ticker, kinds, start, end, limit, warn=self._warn)
        except ProviderUnavailable as e:
            self._warn(str(e))
        except ProviderError as e:
            self._warn(f"{ticker}: SEC documents unavailable ({e}).")
        return []
