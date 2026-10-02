"""Free-data provider: run the whole pipeline on a personal computer (Windows / macOS / Linux).

Sources
-------
* **Prices** - Yahoo Finance via ``yfinance`` (``auto_adjust=False, actions=True``: Yahoo's split-adjusted
  ``Close``, its split- and dividend-adjusted ``Adj Close`` and the split events). The price panel is
  split- and dividend-adjusted exactly like yfinance's ``auto_adjust`` (OHLC x Adj Close / Close).
* **Fundamentals** - SEC EDGAR XBRL ``companyfacts``, point-in-time on the *filing* date
  (see :mod:`aitrading.data.sec_edgar`); the frame also carries the optional total-assets columns
  (``fields.FUNDAMENTAL_OPTIONAL_COLUMNS``, us-gaap ``Assets`` now and a year earlier).
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
``ProviderError``. All responses are cached on disk (``~/.aitrading/cache`` by default). Prices are
cached per ticker; a ticker Yahoo returns nothing for (rate limit, delisted, typo) is retried once after
a short pause and is never cached, so a rerun asks again. Rate-limited or incomplete estimate fetches
and degraded quotes (quoteSummary failed) are likewise not cached for long, and say so in a warning.

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
* ``market_cap`` = Yahoo ``Close`` on/before ``as_of`` (split-adjusted to today's share basis, *not*
  dividend-adjusted) x SEC cover-page shares put on the same basis: multiplied by every split with an
  ex-date after the cover-page date (so a 10-for-1 split after the 10-Q never makes the cap 10x too
  small). Exceptions, each with a warning: 20-F/40-F filers (the cover page counts ordinary shares,
  not the ADSs that trade) and multi-class filers whose classes trade at different prices (BRK-A /
  BRK-B) use Yahoo's implied share count; without SEC data Yahoo's ``sharesOutstanding`` is used.
  Yahoo's counts are today's: for a historical ``as_of`` those tickers are listed in the universe
  frame's ``attrs[MCAP_CURRENT_SHARES_ATTR]``, so a backtest reports their caps as not point-in-time.
* Snapshot fields for an ``as_of`` within ``snapshot_staleness_days`` of today are today's values (a
  warning says so); the last EPS surprise only uses quarters whose results were public on ``as_of``
  (SEC 8-K release date).
* Consensus / target / trailing-EPS figures that Yahoo reports in a currency other than USD (ADRs and
  foreign filers) are left NaN with a warning: the canonical fields are USD and there is no FX here.
* The universe is a fixed ticker list (the bundled starter list or yours), the same for every
  ``as_of``. For an ``as_of`` more than ``snapshot_staleness_days`` before today ``get_universe`` adds a
  SURVIVORSHIP BIAS warning: names delisted or acquired since then are not in the list.

Yahoo data via ``yfinance`` is for personal research use; respect Yahoo's terms. SEC EDGAR is public.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data.base import MCAP_CURRENT_SHARES_ATTR, Capability, PricePanel, ProviderError, ProviderUnavailable
from aitrading.data.cache import DAY, HOUR, MINUTE, DiskCache
from aitrading.data.sec_edgar import US_EXCHANGES, SecEdgarClient, SecNotFound, SecUnsupported, SharesInfo
from aitrading.data.universes import load_starter_universe

PRICE_BATCH = 100
CLOSE_SPLIT_ADJ = "close_split_adj"  # Yahoo 'Close': split-adjusted to today's basis, not dividend-adjusted
SPLITS = "stock_splits"  # split ratio on its ex-date (10.0 = 10-for-1, 0.1 = 1-for-10), else 0
PRICE_KEYS = [*F.PRICE_FIELDS, CLOSE_SPLIT_ADJ, SPLITS]
_YF_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Stock Splits")

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

# Keys only Yahoo's quoteSummary modules supply (the v7 quote endpoint has the rest): without any of them
# Ticker.info is a degraded quote (quoteSummary failed) and is cached for minutes only.
QUOTE_SUMMARY_KEYS = ("previousClose", "sector", "industry", "floatShares", "sharesShort", "impliedSharesOutstanding",
                      "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter")

TTL_INFO = DAY
TTL_PARTIAL = 15 * MINUTE
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


# One-letter Yahoo exchange suffixes (London, Tokyo, Frankfurt, TSX Venture): never US share classes.
_YAHOO_EXCHANGE_LETTERS = {"L", "T", "F", "V"}


def normalize_ticker(t: str) -> str:
    """Upper-case and use Yahoo/SEC share-class notation: 'brk.b' -> 'BRK-B'.

    Only a US share-class letter after an alphabetic root of up to 5 letters is converted; Yahoo
    exchange suffixes stay as they are ('SHOP.TO', 'VOD.L', '7203.T', 'BMW.F', 'ABC.V').
    """
    s = str(t).strip().upper().replace("/", "-")
    if "." in s:
        head, _, tail = s.rpartition(".")
        if (len(tail) == 1 and tail.isalpha() and tail not in _YAHOO_EXCHANGE_LETTERS
                and head.isalpha() and len(head) <= 5):
            s = f"{head}-{tail}"
    return s


def _read_text_any(p: Path) -> str:
    """Decode a user-supplied text/CSV file: UTF-16 (Excel 'Unicode Text'), UTF-8 (BOM optional),
    then Windows-1252 (Excel 'CSV (Comma delimited)' on Windows), then Latin-1 (never fails)."""
    raw = p.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw[:200].count(b"\x00") > len(raw[:200]) // 4:  # UTF-16 without a BOM
        try:
            return raw.decode("utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_ticker_file(path: str | Path) -> list[str]:
    """One ticker per line (``#`` comments allowed) or a CSV with a ``ticker`` / ``symbol`` column."""
    p = Path(path)
    text = _read_text_any(p)
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
    """Yahoo epoch seconds -> calendar date (tz-naive Timestamp).

    Date-like fields (fiscal year ends, short-interest settlement) are midnight UTC; earnings times
    (pre-market or ~16:05 New York) fall on the same UTC calendar day, so the UTC date is right for both.
    """
    v = _num(x)
    if math.isnan(v) or v <= 0:
        return pd.NaT
    try:
        return pd.Timestamp(int(v), unit="s").normalize()
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


def _empty_price_frames(syms: list[str]) -> dict[str, pd.DataFrame]:
    idx = pd.DatetimeIndex([], name="date").astype("datetime64[ns]")
    return {k: pd.DataFrame(index=idx, columns=syms, dtype="float64") for k in PRICE_KEYS}


def extract_ohlcv(raw: pd.DataFrame | None, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a ``yfinance.download`` result into canonical wide frames (dates x symbols).

    Returns the five ``fields.PRICE_FIELDS`` (split- and dividend-adjusted) plus ``CLOSE_SPLIT_ADJ``
    (Yahoo's ``Close``: split-adjusted, not dividend-adjusted - the right price for market cap) and
    ``SPLITS`` (split ratio on the ex-date, else 0).

    With ``auto_adjust=False`` (what the provider requests) OHLC are scaled by ``Adj Close / Close``,
    the same arithmetic as ``yfinance.utils.auto_adjust`` (a missing ratio - e.g. today's bar - counts
    as 1). Without an ``Adj Close`` column the frame is taken as already adjusted.

    Handles every layout yfinance has produced: MultiIndex ``(Price, Ticker)`` (``group_by='column'``,
    the default and what 1.x returns even for one ticker), ``(Ticker, Price)`` (``group_by='ticker'``)
    and flat single-ticker columns (yfinance < 0.2.48 or ``multi_level_index=False``).
    Index -> tz-naive, normalised, ascending, unique dates.
    """
    syms = [s.upper() for s in symbols]
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return _empty_price_frames(syms)
    df = raw.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    cols: dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        levels = [set(map(str, df.columns.get_level_values(i))) for i in range(df.columns.nlevels)]
        field_level = next((i for i, lv in enumerate(levels) if "Close" in lv or "Adj Close" in lv), 0)
        for yname in _YF_COLUMNS:
            if yname not in levels[field_level]:
                continue
            sub = df.xs(yname, axis=1, level=field_level)
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(name=syms[0] if len(syms) == 1 else sub.name)
            if isinstance(sub.columns, pd.MultiIndex):  # more than 2 levels: keep the ticker level
                sub.columns = sub.columns.get_level_values(-1)
            sub.columns = [str(c).upper() for c in sub.columns]
            cols[yname] = sub.loc[:, ~sub.columns.duplicated()].reindex(columns=syms)
    elif len(syms) == 1:
        for yname in _YF_COLUMNS:
            if yname in df.columns:
                cols[yname] = df[[yname]].set_axis(syms, axis=1)
    for yname in list(cols):
        frame = cols[yname].apply(pd.to_numeric, errors="coerce").astype("float64")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame.index = pd.DatetimeIndex(frame.index, name="date").astype("datetime64[ns]")
        cols[yname] = frame
    if not cols:
        return _empty_price_frames(syms)
    index = next(iter(cols.values())).index
    nan = pd.DataFrame(np.nan, index=index, columns=syms, dtype="float64")
    get = lambda name: cols[name].reindex(index=index) if name in cols else nan.copy()  # noqa: E731
    close = get("Close")
    if "Adj Close" in cols:
        ratio = (get("Adj Close") / close).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.bfill().fillna(1.0)
    else:
        ratio = pd.DataFrame(1.0, index=index, columns=syms)
    return {
        F.OPEN: get("Open") * ratio, F.HIGH: get("High") * ratio, F.LOW: get("Low") * ratio,
        F.CLOSE: close * ratio, F.VOLUME: get("Volume"),
        CLOSE_SPLIT_ADJ: close, SPLITS: get("Stock Splits").fillna(0.0),
    }


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


def _currency_of(df: Any) -> str | None:
    """The ``currency`` column yfinance attaches to its analysis frames (``revenueCurrency`` etc.)."""
    if not isinstance(df, pd.DataFrame) or "currency" not in df.columns:
        return None
    vals = [str(v).strip().upper() for v in df["currency"].dropna() if str(v).strip()]
    return vals[0] if vals else None


def _non_usd(cur: Any) -> str | None:
    c = str(cur).strip().upper() if cur is not None and not (isinstance(cur, float) and math.isnan(cur)) else ""
    return c if c and c != "USD" else None


# A quarter is reported 2-14 weeks after it ends; a release at time R covers a quarter ending after R - 100 days.
_REPORT_WINDOW = pd.Timedelta(days=100)


def estimates_from_yahoo(info: dict, earnings_estimate: Any, revenue_estimate: Any, eps_trend: Any,
                         earnings_history: Any, calendar: Any, as_of: date,
                         last_earnings_date: date | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    """One ``fields.ESTIMATE_COLUMNS`` row from yfinance's analysis objects (see module docstring).

    ``last_earnings_date``: date of the latest earnings release on/before ``as_of`` (SEC 8-K item
    2.02); the last EPS surprise only uses quarters that ended before it, i.e. results that were public
    on ``as_of`` (Yahoo indexes ``earnings_history`` by fiscal quarter end, not by report date).
    ``notes`` collects fields left NaN because Yahoo gave them in a currency other than USD.
    """
    info = info or {}
    notes = notes if notes is not None else []
    w = ntm_weight(info, as_of)
    row: dict[str, Any] = {c: math.nan for c in F.ESTIMATE_COLUMNS}

    rev_cur = _non_usd(_currency_of(revenue_estimate))
    eps_cur = _non_usd(_currency_of(earnings_estimate))
    trend_cur = _non_usd(_currency_of(eps_trend))
    if rev_cur:
        notes.append(f"revenue estimates in {rev_cur}")
    else:
        row[F.REVENUE_NTM_EST] = _blend(_cell(revenue_estimate, "0y", "avg"), _cell(revenue_estimate, "+1y", "avg"), w)
    eps = math.nan
    if eps_cur:
        notes.append(f"EPS estimates in {eps_cur}")
    else:
        eps = _blend(_cell(earnings_estimate, "0y", "avg"), _cell(earnings_estimate, "+1y", "avg"), w)
    if trend_cur:
        if not eps_cur:
            notes.append(f"EPS trend in {trend_cur}")
    else:
        if not math.isfinite(eps) and not eps_cur:
            eps = _blend(_cell(eps_trend, "0y", "current"), _cell(eps_trend, "+1y", "current"), w)
        row[F.EPS_NTM_EST_3M_AGO] = _blend(_cell(eps_trend, "0y", "90daysAgo"), _cell(eps_trend, "+1y", "90daysAgo"), w)
    row[F.EPS_NTM_EST] = eps
    row[F.REVENUE_NTM_EST_3M_AGO] = math.nan  # no free revenue-estimate history

    fin_cur = _non_usd(info.get("financialCurrency"))
    if fin_cur:
        notes.append(f"trailing EPS in {fin_cur}")
    else:
        eps_ttm = _num(info.get("trailingEps"))
        row[F.EPS_TTM] = eps_ttm if math.isfinite(eps_ttm) else _num(info.get("epsTrailingTwelveMonths"))
    n = _cell(earnings_estimate, "0y", "numberOfAnalysts")
    if not math.isfinite(n):
        n = _cell(earnings_estimate, "+1y", "numberOfAnalysts")
    row[F.NUM_ANALYSTS] = n if math.isfinite(n) else _num(info.get("numberOfAnalystOpinions"))
    quote_cur = _non_usd(info.get("currency"))
    if quote_cur:
        notes.append(f"target price in {quote_cur}")
    else:
        row[F.TARGET_PRICE_MEAN] = _num(info.get("targetMeanPrice"))

    asof_ts = pd.Timestamp(as_of)
    # Yahoo's earningsTimestamp* are UTC epochs: the next release window, or the latest past release.
    info_ts = {k: _epoch_to_ts(info.get(k)) for k in ("earningsTimestamp", "earningsTimestampStart")}

    # Last surprise: latest quarter (index = fiscal quarter end) whose results were public on as_of.
    if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
        eh = earnings_history.copy()
        try:
            eh.index = pd.DatetimeIndex(pd.to_datetime(eh.index)).tz_localize(None)
        except (TypeError, ValueError):
            eh.index = pd.DatetimeIndex(pd.to_datetime(eh.index, utc=True)).tz_localize(None)
        eh = eh.sort_index()
        eh = eh[eh.index <= asof_ts]
        released = _to_ts(last_earnings_date) if last_earnings_date is not None else pd.NaT
        if released is pd.NaT:
            past = [t for t in info_ts.values() if t is not pd.NaT and t <= asof_ts]
            released = max(past) if past else pd.NaT
        if released is not pd.NaT:
            eh = eh[eh.index < released]
        else:
            later = [t for t in info_ts.values() if t is not pd.NaT and t > asof_ts]
            if later:  # the quarter reported at that (later) release was not public on as_of
                eh = eh[eh.index < min(later) - _REPORT_WINDOW]
        for _, r in eh.iloc[::-1].iterrows():
            act, est = _num(r.get("epsActual")), _num(r.get("epsEstimate"))
            if math.isfinite(act) and math.isfinite(est):
                row[F.LAST_EPS_SURPRISE] = (act - est) / abs(est) if est != 0 else math.nan
                break
            sp = _num(r.get("surprisePercent"))
            if math.isfinite(sp):
                row[F.LAST_EPS_SURPRISE] = sp  # Yahoo's raw value is already a fraction
                break

    last = _to_ts(last_earnings_date) if last_earnings_date is not None else pd.NaT
    nxt = pd.NaT
    # The UTC epochs come first: yfinance builds calendar['Earnings Date'] with datetime.fromtimestamp(),
    # i.e. in the PC's local time zone, which moves an after-the-close release (~20:05 UTC) to the next
    # day for users east of ~UTC+4 (Asia, Australia).
    for ts in info_ts.values():
        if ts is pd.NaT:
            continue
        if ts >= asof_ts and nxt is pd.NaT:
            nxt = ts
        elif ts < asof_ts and last is pd.NaT:
            last = ts
    if nxt is pd.NaT and isinstance(calendar, dict):
        cal_dates = [_to_ts(d) for d in (calendar.get("Earnings Date") or [])]
        future = sorted(d for d in cal_dates if d is not pd.NaT and d >= asof_ts)
        if future:
            nxt = future[0]
    row[F.LAST_EARNINGS_DATE] = last
    row[F.NEXT_EARNINGS_DATE] = nxt
    return row


# SEC fundamentals carry the optional total-assets columns (us-gaap ``Assets``) after the required ones.
_FUNDAMENTAL_OUT_COLUMNS = F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS


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


#: Text of exceptions that mean the request never reached the server (no network, DNS failure, a proxy that
#: refused the tunnel, connection timeout) - as opposed to an HTTP error or a rate limit from Yahoo itself.
_NETWORK_ERROR_MARKERS = (
    "connectionerror", "connecterror", "connecttimeout", "proxyerror", "curl: (5)", "curl: (6)", "curl: (7)",
    "curl: (28)", "could not resolve", "name or service not known", "nodename nor servname", "getaddrinfo failed",
    "network is unreachable", "connection refused", "failed to establish a new connection",
)


def _is_network_error(e: BaseException) -> bool:
    text = f"{type(e).__name__}: {e}".lower()
    return isinstance(e, (ConnectionError, TimeoutError)) or any(m in text for m in _NETWORK_ERROR_MARKERS)


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
        sleep: Callable[[float], None] = time.sleep,
        retry_pause_s: float = 2.0,
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
            sleep / retry_pause_s: pause before retrying Yahoo after a rate limit or failed tickers (tests).
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
        self._sleep = sleep
        self.retry_pause_s = float(retry_pause_s)
        self._starter = load_starter_universe()
        if tickers:
            seen: dict[str, None] = {}
            for t in tickers:
                if t and str(t).strip():
                    seen.setdefault(normalize_ticker(t), None)
            self._tickers = list(seen)
            self._universe_source = "your ticker list"
        elif universe_file is not None:
            self._tickers = read_ticker_file(universe_file)
            self._universe_source = f"the ticker file {Path(universe_file).name}"
        else:
            self._tickers = list(self._starter.index)
            self._universe_source = "the bundled starter list, chosen in 2026"
        self._info_memo: dict[str, dict] = {}
        self._info_lock = threading.Lock()
        self._closes: dict[str, pd.Series] = {}
        self._close_lock = threading.Lock()
        self._price_failed: set[tuple[str, date, date]] = set()  # no data this session: not re-requested

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
            yf_logger = logging.getLogger("yfinance")
            if yf_logger.level == logging.NOTSET:  # failures are reported via self.warnings instead
                yf_logger.setLevel(logging.CRITICAL)
            self._yf = yfinance
        return self._yf

    def _retry_rate_limited(self, fn: Callable[[], Any], attempts: int = 3) -> Any:
        """Call ``fn``; on yfinance's rate-limit error (``YFRateLimitError``) wait and retry (other errors propagate)."""
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                if "RateLimit" not in type(e).__name__ or i == attempts - 1:
                    raise
                self._sleep(self.retry_pause_s * (i + 1))
        return None  # pragma: no cover

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

    def _survivorship_warning(self) -> None:
        # One message per provider (no as_of in it): a backtest asks for the universe at many past dates.
        self._warn(
            f"SURVIVORSHIP BIAS: for an as_of more than {self.snapshot_staleness_days} days before today "
            f"({self._today()}) the free provider still uses today's universe - {self._universe_source} "
            f"({len(self._tickers)} names) - not the companies listed on that date. Names delisted, acquired or "
            "bankrupt since then are missing, which flatters a historical screen (point-in-time index membership "
            "needs institutional data)."
        )

    def _snapshot_lag_warning(self, what: str, as_of: date) -> None:
        """Snapshot served for an as_of shortly before today: say that it is today's data."""
        lag = (self._today() - as_of).days
        if lag > 0:
            self._warn(f"as_of {as_of} is {lag} day(s) before today ({self._today()}): {what} are Yahoo's current "
                       f"values and may include up to {lag} day(s) of information published after as_of.")

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
        unsupported = {t: e for t, e in errors.items() if isinstance(e, SecUnsupported)}
        not_found = sorted(t for t, e in errors.items() if isinstance(e, SecNotFound) and t not in unsupported)
        other = sorted(t for t, e in errors.items() if not isinstance(e, (SecNotFound, ProviderUnavailable)))
        if unsupported:
            desc = "; ".join(f"{t} {getattr(e, 'reason', e)}" for t, e in sorted(unsupported.items())[:20])
            self._warn(f"{what}: not available for {len(unsupported)} ticker(s) - {desc}. The free provider reads "
                       "US-GAAP XBRL in USD only; left as NaN.")
        if not_found:
            self._warn(f"{what}: no SEC data for {len(not_found)} ticker(s) ({', '.join(not_found[:20])}) - "
                       "foreign filer, fund, or no XBRL financials; left as NaN.")
        if other:
            sample = errors[other[0]]
            self._warn(f"{what}: failed for {len(other)} ticker(s) ({', '.join(other[:20])}); left as NaN. "
                       f"First error: {type(sample).__name__}: {sample}")

    # ------------------------------------------------------------------ Yahoo: info
    @staticmethod
    def _complete_quote(info: dict) -> bool:
        return any(k in info for k in QUOTE_SUMMARY_KEYS)

    def _info(self, sym: str) -> dict:
        with self._info_lock:
            if sym in self._info_memo:
                return self._info_memo[sym]
        key = f"yf:info:v1:{sym}"
        info = self.cache.get_json(key)
        if not isinstance(info, dict):
            yf = self._yfm()
            raw = self._retry_rate_limited(lambda: yf.Ticker(sym).info) or {}
            info = {k: raw[k] for k in INFO_KEYS if k in raw and raw[k] is not None}
            if not any(k in info for k in ("quoteType", "longName", "shortName", "sector")):
                raise ProviderError(f"Yahoo returned no quote data for {sym}")
            # yfinance merges quoteSummary with the v7 quote; when quoteSummary fails (401 crumb, 5xx) only the
            # basic v7 fields remain - keep that degraded quote for minutes, not a day.
            self.cache.put_json(key, info, TTL_INFO if self._complete_quote(info) else TTL_PARTIAL)
        with self._info_lock:
            self._info_memo[sym] = info
        return info

    def _warn_partial_quotes(self, syms: Iterable[str], what: str) -> None:
        with self._info_lock:
            partial = sorted(s for s in syms if s in self._info_memo and not self._complete_quote(self._info_memo[s]))
        if partial:
            self._warn(f"Yahoo returned only a basic quote for {len(partial)} ticker(s) ({', '.join(partial[:20])}) "
                       f"- sector, short interest, fiscal-year dates and analyst data are missing for {what}; not "
                       "cached for long, rerun later.")

    def _infos(self, syms: list[str], what: str, *, fail_if_offline: bool = False) -> dict[str, dict]:
        """Quote info per ticker (failures are NaN plus a warning). With ``fail_if_offline``, a run in which
        EVERY request failed with a network-level error (no connection, DNS, refused proxy tunnel) raises
        ``ProviderUnavailable`` at once: the price downloads that follow would only time out and retry for
        minutes before failing the same way."""
        res = self._parallel(self._info, syms)
        errors = {s: e for s, (_, e) in res.items() if e is not None}
        if fail_if_offline and syms and len(errors) == len(syms) and all(_is_network_error(e) for e in errors.values()):
            first = errors[syms[0]] if syms[0] in errors else next(iter(errors.values()))
            raise ProviderUnavailable(
                f"Cannot reach Yahoo Finance: all {len(syms)} quote requests failed with a network error "
                f"({type(first).__name__}: {str(first)[:160]}). Check the internet connection; behind a proxy, set "
                "HTTPS_PROXY (pip, yfinance and the SEC client honour it). Then run again."
            )
        if errors:
            self._summarise_errors(f"Yahoo quote info ({what})", errors)
        out = {s: (v or {}) for s, (v, e) in res.items() if e is None}
        self._warn_partial_quotes(out, what)
        return out

    # ------------------------------------------------------------------ Yahoo: prices
    def _price_ttl(self, end: date) -> float:
        return 1 * HOUR if end >= self._today() - timedelta(days=1) else 7 * DAY

    def _call_download(self, batch: list[str], start: date, end: date) -> pd.DataFrame:
        yf = self._yfm()
        # auto_adjust=False: keep Yahoo's split-only 'Close' (market cap) next to 'Adj Close' (returns);
        # actions=True: the 'Stock Splits' column puts SEC share counts on the price basis.
        kwargs = dict(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                      auto_adjust=False, actions=True, threads=True, progress=False, group_by="column")
        try:
            return yf.download(batch, multi_level_index=True, **kwargs)
        except TypeError:  # yfinance < 0.2.48 has no multi_level_index
            return yf.download(batch, **kwargs)

    @staticmethod
    def _price_key(sym: str, start: date, end: date) -> str:
        return f"yf:px:v2:{sym}:{start.isoformat()}:{end.isoformat()}"

    def _fetch_prices(self, syms: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """One ``yfinance.download`` -> per-ticker frames (columns ``PRICE_KEYS``) for tickers that returned data."""
        try:
            raw = self._call_download(syms, start, end)
        except ProviderUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - yfinance raises many types; degrade to NaN
            self._warn(f"Yahoo price download failed for {len(syms)} ticker(s) ({type(e).__name__}: {e}).")
            return {}
        wide = extract_ohlcv(raw, syms)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        out: dict[str, pd.DataFrame] = {}
        for s in syms:
            f = pd.DataFrame({k: wide[k][s] for k in PRICE_KEYS})
            f = f.loc[(f.index >= lo) & (f.index <= hi)]
            f = f.loc[f[[F.OPEN, F.HIGH, F.LOW, F.CLOSE, CLOSE_SPLIT_ADJ]].notna().any(axis=1) | (f[SPLITS] > 0)]
            if f[F.CLOSE].notna().any():
                out[s] = f
        return out

    def _download_batch(self, batch: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """Per-ticker frames: disk cache per ticker, else download; failures are retried once and never cached.

        ``yfinance.download`` does not raise for a failed ticker (rate limit, delisted, typo): it returns an
        empty column. Those tickers get one more request after a short pause; still-missing ones are
        remembered for this session only, so a rerun after a Yahoo rate-limit burst asks again.
        """
        frames: dict[str, pd.DataFrame] = {}
        need: list[str] = []
        for s in batch:
            cached = self.cache.get_frame(self._price_key(s, start, end))
            if cached is not None and all(k in cached.columns for k in PRICE_KEYS):
                frames[s] = cached
            elif (s, start, end) not in self._price_failed:
                need.append(s)
        if not need:
            return frames
        got = self._fetch_prices(need, start, end)
        failed = [s for s in need if s not in got]
        if failed:
            self._sleep(self.retry_pause_s)
            got.update(self._fetch_prices(failed, start, end))
        ttl = self._price_ttl(end)
        for s in need:
            if s in got:
                self.cache.put_frame(self._price_key(s, start, end), got[s], ttl)
                frames[s] = got[s]
            else:
                with self._close_lock:
                    self._price_failed.add((s, start, end))
        return frames

    def _download(self, syms: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """Wide frames (dates x syms) for every key in ``PRICE_KEYS``; tickers without data are all-NaN columns."""
        syms = sorted(set(syms))
        per: dict[str, pd.DataFrame] = {}
        for i in range(0, len(syms), PRICE_BATCH):
            per.update(self._download_batch(syms[i: i + PRICE_BATCH], start, end))
        if not per:
            return _empty_price_frames(syms)
        idx = pd.DatetimeIndex(sorted(set().union(*(f.index for f in per.values()))), name="date").astype("datetime64[ns]")
        out = {}
        for k in PRICE_KEYS:
            frame = pd.DataFrame({s: f[k] for s, f in per.items()}, index=idx)
            out[k] = frame.reindex(columns=syms).astype("float64")
        self._remember_closes(out[CLOSE_SPLIT_ADJ])
        return out

    def _remember_closes(self, close: pd.DataFrame) -> None:
        """Keep the latest unadjusted-for-dividends closes (spot-price fallback for option chains)."""
        with self._close_lock:
            for s in close.columns:
                series = close[s].dropna()
                if series.empty:
                    continue
                old = self._closes.get(s)
                self._closes[s] = series if old is None else series.combine_first(old).sort_index()

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
        for f in F.PRICE_FIELDS:
            out = data[f].reindex(columns=[sym_of[t] for t in req])
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

    # ------------------------------------------------------------------ universe
    def _sec_refs(self, syms: list[str]) -> dict[str, Any]:
        try:
            return {s: self.sec.lookup(s) for s in syms}
        except ProviderUnavailable as e:
            self._warn(str(e))
        except ProviderError as e:
            self._warn(f"SEC ticker map unavailable ({e}); exchange/country from Yahoo only.")
        return {}

    def _sec_shares(self, syms: list[str], as_of: date) -> dict[str, SharesInfo]:
        try:
            self.sec.check_user_agent()
        except ProviderUnavailable:
            return {}  # already warned by _sec_refs
        res = self._parallel(lambda s: self.sec.shares_info(s, as_of), syms)
        errors = {s: e for s, (_, e) in res.items() if e is not None and not isinstance(e, SecNotFound)}
        self._summarise_errors("SEC shares outstanding", errors)
        return {s: v for s, (v, e) in res.items()
                if e is None and v is not None and math.isfinite(v.value) and v.value > 0}

    def _market_caps(self, syms: list[str], as_of: date, infos: dict[str, dict],
                     sec_shares: dict[str, SharesInfo], current_shares: list[str] | None = None) -> dict[str, float]:
        """market_cap = Yahoo split-adjusted (not dividend-adjusted) close on/before as_of x shares on the same basis.

        Yahoo's ``Close`` is restated for every split up to the download date (today), so the SEC
        cover-page count is multiplied by each split with an ex-date after its own date. Prices come
        from one download running from that date to today, which carries the splits.

        ``current_shares`` (if given) receives the tickers whose cap uses one of Yahoo's share counts,
        which are current (today's), not as of ``as_of``.
        """
        today = self._today()
        end = max(as_of, today)
        start = as_of - timedelta(days=10)
        for si in sec_shares.values():
            if si.basis_date is not None:
                start = min(start, si.basis_date + timedelta(days=1))
        try:
            px = self._download(syms, start, end)
        except ProviderUnavailable as e:
            self._warn(str(e))
            px = _empty_price_frames(syms)
        lo, hi = pd.Timestamp(as_of - timedelta(days=10)), pd.Timestamp(as_of)
        yahoo_used: list[str] = []
        adr: list[str] = []
        multi_yahoo: list[str] = []
        multi_summed: list[str] = []
        missing_close: list[str] = []
        out: dict[str, float] = {}
        for s in syms:
            col = px[CLOSE_SPLIT_ADJ][s] if s in px[CLOSE_SPLIT_ADJ].columns else pd.Series(dtype="float64")
            window = col[(col.index >= lo) & (col.index <= hi)].dropna()
            price = float(window.iloc[-1]) if len(window) else math.nan
            if not math.isfinite(price):
                missing_close.append(s)
            splits = px[SPLITS][s] if s in px[SPLITS].columns else pd.Series(dtype="float64")
            info = infos.get(s, {})
            implied, listed = _num(info.get("impliedSharesOutstanding")), _num(info.get("sharesOutstanding"))
            yahoo_any = next((v for v in (implied, listed) if math.isfinite(v) and v > 0), math.nan)
            si = sec_shares.get(s)
            shares = math.nan
            if si is not None and si.foreign_issuer:
                adr.append(s)  # 20-F/40-F cover page: ordinary shares, not the ADSs quoted in the US
                shares = yahoo_any
            elif si is not None:
                after = splits[(splits.index > pd.Timestamp(si.basis_date)) & (splits > 0)] if si.basis_date else splits[:0]
                shares = si.value * float(np.prod(after.to_numpy())) if len(after) else si.value
                if si.n_classes > 1:
                    if math.isfinite(implied) and implied > 0:
                        if abs(shares / implied - 1.0) > 0.10:  # classes trade at different prices (BRK-A / BRK-B)
                            shares = implied
                            multi_yahoo.append(s)
                    else:
                        multi_summed.append(s)
            if not (math.isfinite(shares) and shares > 0) and s not in adr:
                shares = listed if math.isfinite(listed) and listed > 0 else implied
                if math.isfinite(shares) and shares > 0:
                    yahoo_used.append(s)
            out[s] = price * shares if math.isfinite(shares) and shares > 0 else math.nan
            if current_shares is not None and math.isfinite(out[s]) and (s in yahoo_used or s in adr or s in multi_yahoo):
                current_shares.append(s)

        if missing_close:
            self._warn(f"No close on/before {as_of} for {len(missing_close)} ticker(s) ({', '.join(missing_close[:20])}); "
                       "market_cap is NaN.")
        if yahoo_used:
            self._warn(f"market_cap for {len(yahoo_used)} ticker(s) ({', '.join(yahoo_used[:20])}) uses Yahoo's current "
                       "sharesOutstanding (SEC point-in-time share count unavailable).")
        if adr:
            self._warn(f"market_cap for {len(adr)} ticker(s) ({', '.join(adr[:20])}) uses Yahoo's current share count: "
                       "they file 20-F/40-F, whose cover page counts ordinary shares rather than the ADSs that trade "
                       "in the US (their fundamentals are IFRS or non-USD and may be blank).")
        if multi_yahoo:
            self._warn(f"market_cap for {len(multi_yahoo)} ticker(s) ({', '.join(multi_yahoo[:20])}) uses Yahoo's "
                       "implied share count: the SEC cover page lists several share classes that trade at different "
                       "prices, so pricing their sum at this listing's price would misstate market cap.")
        if multi_summed:
            self._warn(f"market_cap for {len(multi_summed)} ticker(s) ({', '.join(multi_summed[:20])}) sums every share "
                       "class on the SEC cover page at this listing's price (Yahoo implied share count unavailable); "
                       "it is wrong if the classes trade at different prices.")
        return out

    def get_universe(self, spec: Any, as_of: date) -> pd.DataFrame:
        """Universe rows for ``self.tickers``, filtered by the spec's country / security types / excluded
        sectors (unknown values are kept). Price and liquidity floors are left to the feature engine.

        Membership is the same fixed list for every ``as_of``; for a historical ``as_of`` (stale by
        ``snapshot_staleness_days``) a survivorship-bias warning says so."""
        syms = self.tickers
        if self.is_stale(as_of):
            self._survivorship_warning()
        infos = self._infos(syms, "universe", fail_if_offline=True)
        refs = self._sec_refs(syms)
        sec_shares = self._sec_shares(syms, as_of) if refs else {}
        current_shares: list[str] = []
        mcaps = self._market_caps(syms, as_of, infos, sec_shares, current_shares)
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
            rows[s] = {
                F.NAME: name, F.GICS_SECTOR: sector, F.GICS_INDUSTRY: info.get("industry"), F.EXCHANGE: exchange,
                F.COUNTRY: country, F.CURRENCY: currency, F.SECURITY_TYPE: sec_type, F.MARKET_CAP: mcaps.get(s, math.nan),
                F.VENDOR_ID: s,
            }
        if assumed_common:
            self._warn(f"Security type unknown for {', '.join(assumed_common[:20])}; assumed common stock.")
        df = pd.DataFrame.from_dict(rows, orient="index").reindex(index=syms, columns=F.UNIVERSE_COLUMNS)
        df[F.MARKET_CAP] = pd.to_numeric(df[F.MARKET_CAP], errors="coerce").astype("float64")
        for c in F.UNIVERSE_COLUMNS:
            if c != F.MARKET_CAP:
                df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        df.index = pd.Index(syms, name="ticker")
        out = self._apply_universe_spec(df, spec)
        if current_shares and self.is_stale(as_of):
            # today's share count priced at a past date: not point-in-time (StrategyRunner reports it)
            out.attrs[MCAP_CURRENT_SHARES_ATTR] = [t for t in current_shares if t in out.index]
        return out

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
            return _frame({}, _FUNDAMENTAL_OUT_COLUMNS, tickers, date_cols)
        res = self._parallel(lambda t: self.sec.fundamentals(t, as_of), list(dict.fromkeys(tickers)))
        rows = {t: v for t, (v, e) in res.items() if e is None and v is not None}
        self._summarise_errors("SEC fundamentals", {t: e for t, (_, e) in res.items() if e is not None})
        return _frame(rows, _FUNDAMENTAL_OUT_COLUMNS, tickers, date_cols)

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
        self._snapshot_lag_warning("consensus estimates, target prices and next earnings dates", as_of)
        res = self._parallel(lambda t: self._estimates_one(t, as_of, sec_last.get(t)), tickers)
        self._summarise_errors("Yahoo consensus estimates", {t: e for t, (_, e) in res.items() if e is not None})
        rows: dict[str, dict] = {}
        incomplete: dict[str, str] = {}
        non_usd: dict[str, list[str]] = {}
        for t, (v, e) in res.items():
            if e is not None or v is None:
                continue
            row, fetch_errors, notes = v
            rows[t] = row
            if fetch_errors:
                incomplete[t] = fetch_errors[0]
            if notes:
                non_usd[t] = notes
        if incomplete:
            names = sorted(incomplete)
            self._warn(f"Yahoo consensus estimates incomplete for {len(names)} ticker(s) ({', '.join(names[:20])}): "
                       f"{incomplete[names[0]]}. Not cached - rerun later (Yahoo rate-limits heavy use).")
        if non_usd:
            desc = "; ".join(f"{t} ({', '.join(n)})" for t, n in sorted(non_usd.items())[:20])
            self._warn(f"Yahoo reports some figures in a currency other than USD - left as NaN (amounts must be USD; "
                       f"the free provider does no FX conversion): {desc}.")
        def is_equity(t: str) -> bool:
            with self._info_lock:
                info = self._info_memo.get(normalize_ticker(t))
            return info is None or str(info.get("quoteType", "EQUITY")).upper() == "EQUITY"

        no_cov = sorted(t for t, r in rows.items() if t not in incomplete and t not in non_usd and is_equity(t)
                        and not math.isfinite(_num(r.get(F.EPS_NTM_EST))) and not math.isfinite(_num(r.get(F.REVENUE_NTM_EST))))
        if no_cov:
            self._warn(f"No Yahoo consensus estimates for {len(no_cov)} ticker(s) ({', '.join(no_cov[:20])}): no analyst "
                       "coverage, or Yahoo returned nothing (kept for minutes only; rerun later if unexpected).")
        self._warn_partial_quotes([normalize_ticker(t) for t in tickers], "consensus estimates")
        for t, d in sec_last.items():  # SEC 8-K date is authoritative when Yahoo failed
            rows.setdefault(t, {F.LAST_EARNINGS_DATE: _to_ts(d)})
        return _frame(rows, F.ESTIMATE_COLUMNS, tickers, date_cols)

    def _estimates_one(self, ticker: str, as_of: date, sec_last: date | None) -> tuple[dict[str, Any], list[str], list[str]]:
        """(row, fetch errors, non-USD notes). Rows with fetch errors are not cached; empty ones briefly."""
        sym = normalize_ticker(ticker)
        date_cols = (F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE)
        key = f"yf:estimates:v2:{sym}:{as_of.isoformat()}:{sec_last.isoformat() if sec_last else '-'}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            notes = [str(n) for n in (cached.pop("_notes", None) or [])]
            row = _row_from_json(cached, date_cols)
            errors: list[str] = []
        else:
            info = self._info(sym)
            t = self._yfm().Ticker(sym)
            errors = []

            def grab(attr: str) -> Any:
                # yfinance raises YFRateLimitError (HTTP 429) from these properties; other HTTP errors come
                # back as empty frames. One failing module must not sink the others.
                try:
                    return self._retry_rate_limited(lambda: getattr(t, attr))
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{attr}: {type(e).__name__}: {e}".rstrip(": "))
                    return None

            parts = {a: grab(a) for a in ("earnings_estimate", "revenue_estimate", "eps_trend", "earnings_history",
                                          "calendar")}
            notes = []
            row = estimates_from_yahoo(info, parts["earnings_estimate"], parts["revenue_estimate"], parts["eps_trend"],
                                       parts["earnings_history"], parts["calendar"], as_of,
                                       last_earnings_date=sec_last, notes=notes)
            empty = all(not isinstance(parts[a], pd.DataFrame) or parts[a].empty
                        for a in ("earnings_estimate", "revenue_estimate", "eps_trend"))
            if not errors:  # a failed module must be fetched again next time
                ttl = TTL_PARTIAL if empty and str(info.get("quoteType", "EQUITY")).upper() == "EQUITY" else TTL_ESTIMATES
                self.cache.put_json(key, {**_row_to_json(row), "_notes": notes}, ttl)
        if sec_last is not None:
            row[F.LAST_EARNINGS_DATE] = _to_ts(sec_last)
        return row, errors, notes

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        date_cols = (F.SI_SETTLEMENT_DATE,)
        tickers = list(dict.fromkeys(tickers))
        if self.is_stale(as_of):
            self._stale_warning("short interest", as_of)
            return _frame({}, F.SHORT_INTEREST_COLUMNS, tickers, date_cols)
        self._snapshot_lag_warning("short interest figures", as_of)
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
        self._snapshot_lag_warning("option implied volatilities, volumes and open interest", as_of)
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
        expiry = choose_expiry(self._retry_rate_limited(lambda: t.options), as_of)  # 1 request
        if expiry is None:
            return None  # no listed options
        chain = self._retry_rate_limited(lambda: t.option_chain(expiry))  # 1 request
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
