"""Offline tests for the free-data provider (Yahoo via yfinance + SEC EDGAR).

No network: SEC HTTP goes through ``httpx.MockTransport`` serving fixtures in tests/fixtures/free, and
yfinance is replaced by fakes that return exactly the shapes the installed yfinance (1.x) returns:
``download`` -> MultiIndex columns (Price, Ticker) assembled the way ``yfinance.multi`` does it, the
analysis properties -> frames indexed by period ('0q', '+1q', '0y', '+1y'), ``option_chain`` -> a
namedtuple of calls/puts frames + the underlying quote dict. One test drives the *real*
``yfinance.download`` with a patched ``Ticker`` to pin the layout end to end.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, namedtuple
from datetime import date, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields as F
from aitrading.core.models import DocumentKind
from aitrading.data.base import Capability, MarketDataProvider, ProviderError, ProviderUnavailable
from aitrading.data.cache import DiskCache, safe_filename
from aitrading.data.free import (
    FreeDataProvider,
    choose_expiry,
    estimates_from_yahoo,
    extract_ohlcv,
    normalize_ticker,
    options_summary,
    read_ticker_file,
)
from aitrading.data.sec_edgar import (
    Fact,
    RateLimiter,
    SecEdgarClient,
    SecNotFound,
    _approx_quarter_end_before,
    extract_mdna,
    fiscal_label,
    fundamentals_from_companyfacts,
    html_to_text,
    pick_press_release,
    quarterly_series,
)
from aitrading.data.universes import load_starter_universe
from aitrading.screen.spec import UniverseSpec

FIX = Path(__file__).parent / "fixtures" / "free"
TODAY = date(2026, 10, 2)
M = 1_000_000
UA = "Test Runner test@example.com"


def _fx_json(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _fx_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# =============================================================================================
# Fake SEC HTTP layer
# =============================================================================================


class SecRouter:
    """URL -> fixture; counts every request. Unknown URLs -> 404 (like EDGAR)."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.overrides: dict[str, list[httpx.Response]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls[url] += 1
        assert request.headers.get("User-Agent") == UA
        if url in self.overrides and self.overrides[url]:
            return self.overrides[url].pop(0)
        if url == "https://www.sec.gov/files/company_tickers_exchange.json":
            return httpx.Response(200, json=_fx_json("company_tickers_exchange.json"))
        if url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json":
            return httpx.Response(200, json=_fx_json("companyfacts_CIK0001234567.json"))
        if url == "https://data.sec.gov/submissions/CIK0001234567.json":
            return httpx.Response(200, json=_fx_json("submissions_CIK0001234567.json"))
        base = "https://www.sec.gov/Archives/edgar/data/1234567/"
        if url.startswith(base):
            rest = url[len(base):]
            acc, _, name = rest.partition("/")
            if name == "index.json":
                if acc == "000123456726000031":
                    return httpx.Response(200, json=_fx_json("index_8k_000123456726000031.json"))
                return httpx.Response(200, json={"directory": {"item": [
                    {"name": f"acme-{acc}.htm", "size": "20000"}, {"name": "acme-ex991.htm", "size": "50000"}]}})
            if "ex991" in name:
                return httpx.Response(200, text=_fx_text("ex99_1_press_release.htm"))
            if name == "acme-20251231.htm":
                return httpx.Response(200, text=_fx_text("form10k_primary.htm"))
            if re.fullmatch(r"acme-20\d{6}\.htm", name):
                return httpx.Response(200, text=_fx_text("form10q_primary.htm"))
        return httpx.Response(404, text="Not Found")


def make_sec(tmp_path: Path, router: SecRouter | None = None, *, user_agent: str | None = UA,
             cache: DiskCache | None = None) -> tuple[SecEdgarClient, SecRouter]:
    router = router or SecRouter()
    client = SecEdgarClient(user_agent=user_agent, cache=cache or DiskCache(tmp_path / "cache"),
                            transport=httpx.MockTransport(router), sleep=lambda _s: None)
    return client, router


# =============================================================================================
# Fake yfinance (shapes copied from yfinance 1.x source: multi.py, ticker.py, scrapers/analysis.py)
# =============================================================================================


def price_frame(sym: str, start: str = "2025-06-02", end: str = "2026-10-02") -> pd.DataFrame:
    """Per-ticker frame like Ticker.history(auto_adjust=True): Open/High/Low/Close/Volume, index 'Date'."""
    idx = pd.bdate_range(start, end, name="Date")
    base = 20.0 + sum(map(ord, sym)) % 50
    close = base + np.arange(len(idx)) * 0.1
    return pd.DataFrame({"Open": close - 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close,
                         "Volume": (np.arange(len(idx)) % 7 + 1) * 1000}, index=idx)


def raw_history(df: pd.DataFrame) -> pd.DataFrame:
    """Ticker.history(auto_adjust=False, actions=True) columns: Yahoo's split-adjusted Close, its split- and
    dividend-adjusted Adj Close, and the split / dividend events (defaults: no dividends, no splits)."""
    out = df.copy()
    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]
    for c in ("Dividends", "Stock Splits"):
        if c not in out.columns:
            out[c] = 0.0
    return out[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]


def yf_like_download(frames: dict[str, pd.DataFrame | None], tickers: list[str], start: str, end: str,
                     fail: set[str] = frozenset()) -> pd.DataFrame:
    """Re-implementation of yfinance.multi._download_impl's assembly (group_by='column', actions=True)."""
    dfs: dict[str, pd.DataFrame] = {}
    for t in sorted({t.upper() for t in tickers}):
        df = frames.get(t)
        if df is None or t in fail:  # yfinance: failed ticker (incl. rate limit) -> utils.empty_df()
            empty = pd.DataFrame(index=[], data={"Open": np.nan, "High": np.nan, "Low": np.nan, "Close": np.nan,
                                                 "Adj Close": np.nan, "Volume": np.nan})
            empty.index.name = "Date"
            dfs[t] = empty
        else:
            df = raw_history(df)
            dfs[t] = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]  # end exclusive
    idx = None
    for df in dfs.values():
        if not df.empty:
            idx = df.index if idx is None else idx.union(df.index)
    idx = idx if idx is not None else pd.DatetimeIndex([])
    dfs = {k: v.reindex(idx) for k, v in dfs.items()}
    data = pd.concat(dfs.values(), axis=1, sort=True, keys=dfs.keys(), names=["Ticker", "Price"])
    data.columns = data.columns.swaplevel(0, 1)
    data.sort_index(level=0, axis=1, inplace=True)
    return data


Options = namedtuple("Options", ["calls", "puts", "underlying"])
OPT_COLS = ["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "change", "percentChange", "volume",
            "openInterest", "impliedVolatility", "inTheMoney", "contractSize", "currency"]


def option_frame(strikes: list[float], ivs: list[float], vols: list[float], ois: list[float]) -> pd.DataFrame:
    n = len(strikes)
    return pd.DataFrame({
        "contractSymbol": [f"X{i}" for i in range(n)], "lastTradeDate": pd.Timestamp("2026-10-01", tz="UTC"),
        "strike": strikes, "lastPrice": 1.0, "bid": 0.9, "ask": 1.1, "change": 0.0, "percentChange": 0.0,
        "volume": vols, "openInterest": ois, "impliedVolatility": ivs, "inTheMoney": False,
        "contractSize": "REGULAR", "currency": "USD"}, columns=OPT_COLS)


def periodic(rows: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"period": p, **v} for p, v in rows.items()]).set_index("period")
    df["currency"] = "USD"
    return df


class FakeTicker:
    def __init__(self, yf: "FakeYF", sym: str) -> None:
        self._yf = yf
        self.ticker = sym.upper()
        self._spec = yf.specs.get(self.ticker, {})
        yf.ticker_calls[self.ticker] += 1

    @property
    def info(self) -> dict:
        if "info" not in self._spec:
            raise RuntimeError("404 quote not found")
        self._yf.info_calls[self.ticker] += 1
        return dict(self._spec["info"])

    @property
    def options(self) -> tuple:
        return tuple(self._spec.get("expiries", ()))

    def option_chain(self, date=None, tz=None):  # noqa: A002 - yfinance's own signature
        self._yf.chain_calls.append((self.ticker, date))
        calls, puts, spot = self._spec["chain"]
        return Options(calls, puts, {"regularMarketPrice": spot, "symbol": self.ticker})

    def __getattr__(self, name: str):
        if name in ("earnings_estimate", "revenue_estimate", "eps_trend", "earnings_history", "calendar"):
            raises = self._spec.get("raises", {})
            if raises.get(name):  # list of exceptions to raise on successive calls, then normal behaviour
                raise raises[name].pop(0)
            if name not in self._spec:
                return pd.DataFrame() if name != "calendar" else {}
            return self._spec[name]
        raise AttributeError(name)


class FakeYF:
    """Stands in for the ``yfinance`` module (only the API the provider uses)."""

    def __init__(self, frames: dict[str, pd.DataFrame | None] | None = None, specs: dict[str, dict] | None = None,
                 fail_first: dict[str, int] | None = None) -> None:
        self.frames = frames or {}
        self.specs = specs or {}
        self.fail_first = dict(fail_first or {})  # ticker -> number of downloads that return nothing for it
        self.download_calls: list[dict] = []
        self.ticker_calls: Counter[str] = Counter()
        self.info_calls: Counter[str] = Counter()
        self.chain_calls: list[tuple] = []

    def download(self, tickers, start=None, end=None, **kwargs) -> pd.DataFrame:
        self.download_calls.append({"tickers": list(tickers), "start": start, "end": end, **kwargs})
        # The provider needs Yahoo's split-only Close and the split events, not auto-adjusted prices.
        assert kwargs.get("auto_adjust") is False and kwargs.get("actions") is True and kwargs.get("progress") is False
        fail = set()
        for t in tickers:
            if self.fail_first.get(t.upper(), 0) > 0:
                self.fail_first[t.upper()] -= 1
                fail.add(t.upper())
        return yf_like_download(self.frames, list(tickers), start, end, fail)

    def Ticker(self, sym: str) -> FakeTicker:  # noqa: N802 - mirrors yfinance
        return FakeTicker(self, sym)


def epoch(d: str) -> int:
    return int(pd.Timestamp(d).timestamp())


def make_provider(tmp_path: Path, *, yf: FakeYF | None = None, sec: SecEdgarClient | None = None,
                  tickers: list[str] | None = None, today: date = TODAY) -> FreeDataProvider:
    cache = DiskCache(tmp_path / "cache")
    if sec is None:
        sec, _ = make_sec(tmp_path, cache=cache)
    return FreeDataProvider(tickers or ["ACME"], cache=cache, sec=sec, yf_module=yf or FakeYF(),
                            today=lambda: today, max_workers=4, sleep=lambda _s: None)


# =============================================================================================
# Disk cache
# =============================================================================================


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_cache_roundtrip_ttl_and_disable(tmp_path, monkeypatch):
    clock = FakeClock()
    c = DiskCache(tmp_path, clock=clock)
    c.put_json("k", {"a": [1, 2]}, ttl_s=10)
    assert c.get_json("k") == {"a": [1, 2]}
    clock.t += 5
    assert c.get_json("k") == {"a": [1, 2]}
    assert c.get_json("k", ttl_s=3) is None  # caller demands fresher than 3 s
    clock.t += 6
    assert c.get_json("k") is None  # stored ttl expired
    c.put_json("forever", "x")
    clock.t += 10**9
    assert c.get_text("forever") == "x"

    df = pd.DataFrame({"a": [1.0, np.nan]}, index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
    c.put_frame("f", df, ttl_s=60)
    pd.testing.assert_frame_equal(c.get_frame("f"), df)

    c.path_for("bad", ".json").parent.mkdir(parents=True, exist_ok=True)
    c.path_for("bad", ".json").write_text("{not json", encoding="utf-8")
    assert c.get_json("bad") is None  # corrupt entry = miss, never an error

    off = DiskCache(tmp_path / "off", enabled=False)
    off.put_json("k", 1)
    assert off.get_json("k") is None and not (tmp_path / "off").exists()

    monkeypatch.setenv("AITRADING_CACHE_DIR", str(tmp_path / "envdir"))
    monkeypatch.setenv("AITRADING_CACHE_DISABLE", "1")
    env_cache = DiskCache()
    assert env_cache.root == tmp_path / "envdir" and env_cache.enabled is False
    monkeypatch.delenv("AITRADING_CACHE_DISABLE")
    monkeypatch.delenv("AITRADING_CACHE_DIR")
    assert DiskCache().root == Path.home() / ".aitrading" / "cache"


def test_cache_keys_are_windows_safe(tmp_path):
    keys = [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        'a<b>c:d"e/f\\g|h?i*j', "CON", "nul", "COM1", "lpt9", "trailing. ", "x" * 2000, "BRK.B", "brk.b", "",
        "yf:ohlcv:v1:2025-01-01:2026-10-02:AAPL,MSFT,BRK-B",
    ]
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}
    names = set()
    c = DiskCache(tmp_path)
    for k in keys:
        name = safe_filename(k, ".json")
        assert re.fullmatch(r"[A-Za-z0-9_-]+-[0-9a-f]{20}\.json", name), name
        assert len(name) <= 100
        assert name.split(".")[0].upper() not in reserved
        assert not name.endswith((" ", "."))
        names.add(name)
        c.put_json(k, {"key": k})
        assert c.get_json(k) == {"key": k}
        assert len(str(c.path_for(k, ".json").relative_to(tmp_path))) < 110
    assert len(names) == len(keys)  # 'BRK.B' vs 'brk.b' do not collide on case-insensitive file systems


# =============================================================================================
# XBRL -> point-in-time quarterly math
# =============================================================================================


def _f(start, end, val, filed, accn="a"):
    return Fact(date.fromisoformat(start) if start else None, date.fromisoformat(end), float(val),
                date.fromisoformat(filed), accn, "10-Q")


def test_ytd_cash_flow_is_differenced_into_quarters_and_q4_from_fy():
    facts = [
        _f("2025-01-01", "2025-03-31", 30, "2025-05-01", "q1"),
        _f("2025-01-01", "2025-06-30", 70, "2025-07-31", "q2"),
        _f("2025-01-01", "2025-09-30", 115, "2025-10-30", "q3"),
        _f("2025-01-01", "2025-12-31", 165, "2026-02-19", "k"),
    ]
    qs = quarterly_series(facts, date(2026, 3, 1))
    assert [(q.end.isoformat(), q.value) for q in qs.values()] == [
        ("2025-03-31", 30), ("2025-06-30", 40), ("2025-09-30", 45), ("2025-12-31", 50)]
    assert qs[date(2025, 12, 31)].derived and qs[date(2025, 12, 31)].start == date(2025, 10, 1)
    assert qs[date(2025, 12, 31)].first_filed == date(2026, 2, 19)
    # Before the 10-K is filed there is no Q4.
    assert date(2025, 12, 31) not in quarterly_series(facts, date(2026, 2, 18))


def test_q4_from_fy_minus_three_direct_quarters_when_no_nine_month_ytd():
    facts = [
        _f("2024-01-01", "2024-03-31", 10, "2024-05-01"),
        _f("2024-04-01", "2024-06-30", 11, "2024-08-01"),
        _f("2024-07-01", "2024-09-30", 12, "2024-11-01"),
        _f("2024-01-01", "2024-12-31", 50, "2025-02-20"),
    ]
    q4 = quarterly_series(facts, date(2025, 3, 1))[date(2024, 12, 31)]
    assert q4.value == 17 and q4.derived and q4.start == date(2024, 10, 1)


def test_restatement_uses_latest_filing_on_or_before_as_of():
    facts = [_f("2025-01-01", "2025-03-31", 125, "2025-05-01", "orig"),
             _f("2025-01-01", "2025-03-31", 127, "2026-04-30", "restated")]
    assert quarterly_series(facts, date(2026, 1, 1))[date(2025, 3, 31)].value == 125
    q = quarterly_series(facts, date(2026, 5, 1))[date(2025, 3, 31)]
    assert q.value == 127 and q.first_filed == date(2025, 5, 1)  # became public with the original filing
    assert quarterly_series(facts, date(2025, 4, 30)) == {}  # nothing filed yet


EXPECTED_2026_08_15 = {
    F.PERIOD_END: pd.Timestamp("2026-06-30"), F.REPORT_DATE: pd.Timestamp("2026-08-04"),
    # TTM = Q3'25 145 + Q4'25 (560 - 405 = 155) + Q1'26 150 + Q2'26 165
    F.REVENUE_TTM: 615 * M,
    # prior TTM = Q3'24 120 + Q4'24 (460 - 330 = 130) + Q1'25 127 (restated 2026-04-30) + Q2'25 135
    F.REVENUE_TTM_PRIOR_YEAR: 512 * M,
    F.REVENUE_LAST_Q: 165 * M, F.REVENUE_LAST_Q_PRIOR_YEAR: 135 * M,
    F.GROSS_PROFIT_TTM: (615 - (87 + 93 + 90 + 99)) * M,  # no GrossProfit tag -> revenue - CostOfRevenue
    F.GROSS_PROFIT_TTM_PRIOR_YEAR: (512 - (72 + 78 + 75 + 81)) * M,
    F.OPERATING_INCOME_TTM: (24 + 26 + 25 + 28) * M, F.OPERATING_INCOME_TTM_PRIOR_YEAR: (17 + 18 + 20 + 22) * M,
    F.EBITDA_TTM: (103 + 5 + 5 + 6 + 6) * M,  # + D&A differenced from YTD
    F.NET_INCOME_TTM: (16 + 17 + 18 + 19) * M,
    F.CFO_TTM: (45 + 50 + 35 + 45) * M,  # YTD: 9M'25 115-70, FY'25 165-115, Q1'26 35, 6M'26 80-35
    F.CAPEX_TTM: (11 + 17 + 12 + 13) * M,
    F.FCF_TTM: (175 - 53) * M,
    F.TOTAL_DEBT: (280 + 20) * M, F.CASH: 150 * M, F.INTEREST_EXPENSE_TTM: 8 * M, F.TOTAL_EQUITY: 500 * M,
    F.SHARES_OUTSTANDING: 49_500_000,
    F.TOTAL_ASSETS: 1210 * M, F.TOTAL_ASSETS_PRIOR_YEAR: 1100 * M,  # us-gaap Assets 2026-06-30 / 2025-06-30
}


def test_fundamentals_point_in_time_full_row():
    cf = _fx_json("companyfacts_CIK0001234567.json")
    row = fundamentals_from_companyfacts(cf, date(2026, 8, 15))
    assert list(row) == F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS
    for k, v in EXPECTED_2026_08_15.items():
        assert row[k] == pytest.approx(v) if isinstance(v, (int, float)) else row[k] == v, k
    assert row[F.FCF_TTM] == row[F.CFO_TTM] - row[F.CAPEX_TTM]


def test_fundamentals_ignore_filings_after_as_of():
    cf = _fx_json("companyfacts_CIK0001234567.json")
    row = fundamentals_from_companyfacts(cf, date(2026, 5, 20))  # Q2'26 10-Q (filed 2026-08-04) is in the future
    assert row[F.PERIOD_END] == pd.Timestamp("2026-03-31") and row[F.REPORT_DATE] == pd.Timestamp("2026-04-30")
    assert row[F.REVENUE_TTM] == (135 + 145 + 155 + 150) * M
    assert row[F.REVENUE_TTM_PRIOR_YEAR] == (110 + 120 + 130 + 127) * M  # restatement already public
    assert row[F.REVENUE_LAST_Q_PRIOR_YEAR] == 127 * M
    assert row[F.CFO_TTM] == (40 + 45 + 50 + 35) * M and row[F.FCF_TTM] == (170 - 52) * M
    assert row[F.TOTAL_DEBT] == 290 * M  # the 2026-06-30 short-term borrowings are not public yet
    assert row[F.SHARES_OUTSTANDING] == 50_000_000
    assert (row[F.TOTAL_ASSETS], row[F.TOTAL_ASSETS_PRIOR_YEAR]) == (1180 * M, 1080 * M)  # 2026-03-31 vs 2025-03-31

    row = fundamentals_from_companyfacts(cf, date(2026, 3, 1))  # just after the FY2025 10-K
    assert row[F.PERIOD_END] == pd.Timestamp("2025-12-31") and row[F.REPORT_DATE] == pd.Timestamp("2026-02-19")
    assert row[F.REVENUE_LAST_Q] == 155 * M  # Q4 = FY 560 - 9M 405
    assert row[F.REVENUE_TTM] == 560 * M  # original Q1'25 = 125 (restatement not yet filed)
    assert row[F.REVENUE_TTM_PRIOR_YEAR] == 460 * M
    assert (row[F.TOTAL_ASSETS], row[F.TOTAL_ASSETS_PRIOR_YEAR]) == (1150 * M, 1050 * M)  # FY-end vs prior FY-end


def test_fundamentals_missing_pieces_are_nan_not_errors():
    cf = _fx_json("companyfacts_CIK0001234567.json")
    row = fundamentals_from_companyfacts(cf, date(2024, 5, 10))  # one quarter of history
    assert row[F.REVENUE_LAST_Q] == 100 * M and math.isnan(row[F.REVENUE_TTM]) and math.isnan(row[F.FCF_TTM])
    assert row[F.TOTAL_ASSETS] == 1010 * M and math.isnan(row[F.TOTAL_ASSETS_PRIOR_YEAR])  # no year-ago balance sheet
    empty = fundamentals_from_companyfacts({"facts": {}}, date(2026, 1, 1))
    assert empty[F.PERIOD_END] is pd.NaT
    assert all(math.isnan(empty[c]) for c in F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS
               if c not in (F.PERIOD_END, F.REPORT_DATE))
    # EBITDA is NaN when D&A is not tagged
    no_da = json.loads(json.dumps(cf))
    del no_da["facts"]["us-gaap"]["DepreciationDepletionAndAmortization"]
    r = fundamentals_from_companyfacts(no_da, date(2026, 8, 15))
    assert math.isnan(r[F.EBITDA_TTM]) and r[F.OPERATING_INCOME_TTM] == 103 * M


# =============================================================================================
# SEC client: user agent, caching, retries, rate limiting
# =============================================================================================


def test_missing_sec_user_agent_raises_with_setup_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    client, router = make_sec(tmp_path, user_agent=None)
    with pytest.raises(ProviderUnavailable) as ei:
        client.company_facts(1234567)
    msg = str(ei.value)
    assert "SEC_USER_AGENT is not set" in msg
    assert '$env:SEC_USER_AGENT="Jane Doe jane@example.com"' in msg  # Windows PowerShell
    assert 'export SEC_USER_AGENT="Jane Doe jane@example.com"' in msg  # macOS / Linux
    assert sum(router.calls.values()) == 0

    bad, _ = make_sec(tmp_path, user_agent="justaname")
    with pytest.raises(ProviderUnavailable, match="Name email@domain"):
        bad.ticker_map()

    monkeypatch.setenv("SEC_USER_AGENT", "Env User env@example.com")
    assert SecEdgarClient(cache=DiskCache(tmp_path / "c2")).check_user_agent() == "Env User env@example.com"

    # The provider degrades to NaN + the same guidance as a warning (prices etc. still work).
    monkeypatch.delenv("SEC_USER_AGENT")
    p = make_provider(tmp_path, sec=client)
    out = p.get_fundamentals(["ACME"], date(2026, 8, 15))
    assert list(out.columns) == F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS and out.isna().all().all()
    assert any("$env:SEC_USER_AGENT" in w for w in p.warnings)
    assert any("SEC_USER_AGENT" in d for d in p.diagnostics())


def test_cache_hit_avoids_second_http_call(tmp_path):
    cache = DiskCache(tmp_path / "shared")
    c1, router = make_sec(tmp_path, cache=cache)
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001234567.json"
    cf1 = c1.company_facts(1234567)
    cf2 = c1.company_facts(1234567)  # in-memory memo
    assert router.calls[url] == 1 and cf1 is cf2
    c2, _ = make_sec(tmp_path, router, cache=cache)  # new process, same disk cache
    assert c2.company_facts(1234567)["entityName"] == "Acme Widgets Inc."
    assert router.calls[url] == 1
    # Only the concepts the client reads are stored on disk.
    assert set(cf1["facts"]["us-gaap"]) >= {"RevenueFromContractWithCustomerExcludingAssessedTax", "LongTermDebt", "Assets"}
    # Filing documents are cached forever as converted text.
    f_url = "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000031/acme-ex991_q22026.htm"
    t1 = c1.document_text(f_url)
    t2 = c2.document_text(f_url)
    assert t1 == t2 and router.calls[f_url] == 1


def test_retries_on_429_and_5xx_then_404_is_not_found(tmp_path):
    slept: list[float] = []
    router = SecRouter()
    url = "https://data.sec.gov/submissions/CIK0001234567.json"
    router.overrides[url] = [httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(503)]
    client = SecEdgarClient(user_agent=UA, cache=DiskCache(tmp_path), transport=httpx.MockTransport(router),
                            sleep=slept.append, max_requests_per_second=1000)
    sub = client.submissions(1234567)
    assert sub["name"] == "Acme Widgets Inc." and router.calls[url] == 3
    assert 2.0 in slept  # honoured Retry-After
    # Reduced submissions keep only 8-K/10-Q/10-K rows (the Form 4 is gone).
    assert "4" not in sub["filings"]["recent"]["form"]

    router.overrides["https://data.sec.gov/api/xbrl/companyfacts/CIK0007654321.json"] = [httpx.Response(500)] * 10
    with pytest.raises(ProviderError, match="HTTP 500"):
        client.company_facts(7654321)
    with pytest.raises(SecNotFound):
        client.company_facts(1111111)


def test_rate_limiter_spaces_requests():
    clock = FakeClock(0.0)
    waits: list[float] = []
    rl = RateLimiter(8.0, clock=clock, sleep=waits.append)
    for _ in range(3):
        rl.acquire()
    assert waits == pytest.approx([0.125, 0.25])  # first call free, then 1/8 s apart
    clock.t = 10.0
    rl.acquire()
    assert len(waits) == 2  # idle long enough -> no wait


# =============================================================================================
# Filing text: HTML -> text, exhibit choice, MD&A extraction, labels
# =============================================================================================


def test_html_to_text_and_mdna_extraction():
    pr = html_to_text(_fx_text("ex99_1_press_release.htm"))
    assert "said Jane Roe, Chief Executive Officer" in pr
    assert "freight costs in the Americas segment" in pr  # words split across inline tags are re-joined
    assert "CLEVELAND, Ohio, July 28, 2026" in pr  # &nbsp; -> space
    for junk in ("should never appear", "HIDDEN-TRACKING-TEXT", "font-family", "Cost of revenue 99.0"):
        assert junk not in pr
    assert "\n\n" in pr and "\n\n\n" not in pr

    q = html_to_text(_fx_text("form10q_primary.htm"))
    assert "HIDDEN-CONTEXT" not in q and "DocumentFiscalPeriodFocus" not in q
    mdna = extract_mdna(q, "10-Q")
    assert mdna.startswith("Item 2. Management’s Discussion and Analysis")
    assert "order backlog of $410 million" in mdna and "Ohio plant expansion" in mdna
    assert "STATEMENT-ONE-SENTINEL" not in mdna and "MARKET-RISK-SENTINEL" not in mdna  # not Item 1 / Item 3
    assert "Item 3." not in mdna  # the table of contents entry was not chosen
    # Cross-references ("see Item 2. Management's Discussion ...") in the notes and in Part II risk factors
    # are not headings; the open-ended Part II one used to win as the "longest" span.
    assert "RISK-FACTOR-SENTINEL" not in mdna and not mdna.startswith("Item 2. Management’s Discussion and Analysis for")
    assert extract_mdna(q, "10-Q", max_chars=700).endswith(".") and len(extract_mdna(q, "10-Q", max_chars=700)) <= 700

    k = extract_mdna(html_to_text(_fx_text("form10k_primary.htm")), "10-K")
    assert k.startswith("ITEM 7. MANAGEMENT'S DISCUSSION") and "record backlog of $380 million" in k
    assert "TENK-MARKET-RISK-SENTINEL" not in k and "TENK-NOTES-SENTINEL" not in k  # not the Item 8 cross-reference
    assert extract_mdna("no such section here " * 100, "10-Q") is None
    assert html_to_text("Plain text exhibit.\nSecond line.\n\nNew paragraph.") == \
        "Plain text exhibit. Second line.\n\nNew paragraph."


def test_pick_press_release_prefers_ex99_1():
    items = _fx_json("index_8k_000123456726000031.json")["directory"]["item"]
    assert pick_press_release(items, "acme-20260728.htm") == ("acme-ex991_q22026.htm", "Ex.99.1")
    assert pick_press_release([{"name": "d12345dex992.htm"}, {"name": "d12345dex99.htm"}]) == ("d12345dex99.htm", "Ex.99")
    assert pick_press_release([{"name": "exhibit99-1.htm"}, {"name": "ex99_01.txt"}])[0] == "exhibit99-1.htm"
    fallback = [{"name": "form8k.htm", "size": "9000"}, {"name": "q2release.htm", "size": "80000"},
                {"name": "0001234567-26-000031.txt", "size": "999999"}, {"name": "logo.jpg", "size": "99999"}]
    assert pick_press_release(fallback, "form8k.htm") == ("q2release.htm", "exhibit")
    assert pick_press_release([{"name": "logo.jpg"}]) is None


def test_fiscal_labels():
    assert fiscal_label(date(2026, 6, 30), "1231") == "Q2 FY2026"
    assert fiscal_label(date(2025, 12, 31), "1231") == "Q4 FY2025"
    assert fiscal_label(date(2025, 6, 28), "0927") == "Q3 FY2025"  # Apple-style 52/53-week year
    assert fiscal_label(date(2024, 12, 28), "0927") == "Q1 FY2025"
    assert fiscal_label(date(2025, 7, 27), "0126") == "Q2 FY2026"  # late-January year end
    assert fiscal_label(date(2026, 1, 3), "0103") == "Q4 FY2025"  # 53-week year ending in early January
    assert fiscal_label(date(2026, 3, 31), None) == "Q1 FY2026"
    assert _approx_quarter_end_before(date(2026, 7, 28), "1231") == date(2026, 6, 30)
    assert _approx_quarter_end_before(date(2026, 2, 5), "1231") == date(2025, 12, 31)


# =============================================================================================
# Provider: documents
# =============================================================================================


def test_documents_kinds_date_filter_and_order(tmp_path):
    p = make_provider(tmp_path)
    news = p.get_documents("ACME", {DocumentKind.NEWS}, date(2026, 1, 1), date(2026, 9, 30))
    assert [d.doc_id for d in news] == ["SEC-0001234567-26-000031-EX99", "SEC-0001234567-26-000018-EX99",
                                       "SEC-0001234567-26-000004-EX99"]  # 8-K 5.02 skipped; newest first
    d0 = news[0]
    assert d0.kind == DocumentKind.NEWS and d0.ticker == "ACME" and d0.source == "SEC EDGAR"
    assert d0.title == "Acme Widgets Inc. Q2 FY2026 earnings release (8-K Ex.99.1)"
    assert news[2].title.startswith("Acme Widgets Inc. Q4 FY2025 earnings release")
    assert d0.url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000031/acme-ex991_q22026.htm"
    assert d0.published_at == datetime(2026, 7, 28)
    assert d0.metadata["source_type"] == "earnings_release_8k" and d0.metadata["items"] == "2.02,9.01"
    assert "said Jane Roe" in d0.text and all(isinstance(v, str) for v in d0.metadata.values())

    filings = p.get_documents("ACME", {DocumentKind.FILING}, date(2026, 1, 1), date(2026, 9, 30))
    assert [(d.doc_id, d.title) for d in filings] == [
        ("SEC-0001234567-26-000034-MDA", "Acme Widgets Inc. Q2 FY2026 10-Q MD&A (Item 2)"),
        ("SEC-0001234567-26-000020-MDA", "Acme Widgets Inc. Q1 FY2026 10-Q MD&A (Item 2)"),
        ("SEC-0001234567-26-000007-MDA", "Acme Widgets Inc. FY2025 10-K MD&A (Item 7)"),
    ]
    assert all(d.kind == DocumentKind.FILING for d in filings)
    assert filings[0].text.startswith("Item 2.") and filings[2].metadata["source_type"] == "10k_mdna"
    assert filings[0].url.endswith("/000123456726000034/acme-20260630.htm")

    both = p.get_documents("ACME", {DocumentKind.NEWS, DocumentKind.FILING}, date(2026, 4, 1), date(2026, 7, 31), limit=3)
    assert [d.doc_id for d in both] == ["SEC-0001234567-26-000031-EX99", "SEC-0001234567-26-000020-MDA",
                                       "SEC-0001234567-26-000018-EX99"]  # 10-Q of 2026-08-04 is after `end`
    assert [d.published_at for d in both] == sorted((d.published_at for d in both), reverse=True)

    assert p.get_documents("ACME", {DocumentKind.TRANSCRIPT}, date(2026, 1, 1), date(2026, 9, 30)) == []
    assert any("transcripts are not available" in w for w in p.warnings)
    assert p.get_documents("NOPE", {DocumentKind.NEWS}, date(2026, 1, 1), date(2026, 9, 30)) == []
    assert any("NOPE" in w for w in p.warnings)


# =============================================================================================
# Provider: fundamentals / universe / prices
# =============================================================================================


def test_provider_fundamentals_frame(tmp_path):
    p = make_provider(tmp_path)
    df = p.get_fundamentals(["ACME", "BETA", "ZZZZ"], date(2026, 8, 15))
    assert list(df.columns) == F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS
    assert list(df.index) == ["ACME", "BETA", "ZZZZ"]
    assert df.loc["ACME", F.TOTAL_ASSETS] == 1210 * M and df.loc["ACME", F.TOTAL_ASSETS_PRIOR_YEAR] == 1100 * M
    assert df.index.name == "ticker"
    assert df.loc["ACME", F.FCF_TTM] == df.loc["ACME", F.CFO_TTM] - df.loc["ACME", F.CAPEX_TTM] == 122 * M
    assert df.loc["ACME", F.REVENUE_TTM] == 615 * M
    assert df[F.REPORT_DATE].dtype == "datetime64[ns]" and df.loc["ACME", F.REPORT_DATE] == pd.Timestamp("2026-08-04")
    assert df.loc[["BETA", "ZZZZ"]].drop(columns=[F.PERIOD_END, F.REPORT_DATE]).isna().all().all()
    assert df.loc[["BETA", "ZZZZ"], F.PERIOD_END].isna().all()
    assert any("BETA" in w and "ZZZZ" in w for w in p.warnings)


def universe_yf() -> FakeYF:
    frames = {s: price_frame(s) for s in ("ACME", "BETA", "GIDX", "AAPL")}
    specs = {
        "ACME": {"info": {"longName": "Acme Widgets Inc.", "quoteType": "EQUITY", "exchange": "NYQ",
                          "sector": "Industrials", "industry": "Specialty Industrial Machinery",
                          "country": "United States", "currency": "USD", "sharesOutstanding": 49_000_000}},
        "BETA": {"info": {"longName": "Beta Robotics Corp.", "quoteType": "EQUITY", "exchange": "NMS",
                          "sector": "Technology", "industry": "Software - Application", "country": "United States",
                          "currency": "USD", "sharesOutstanding": 20_000_000}},
        "GIDX": {"info": {"longName": "Gamma Index Trust", "quoteType": "ETF", "exchange": "PCX", "currency": "USD",
                          "previousClose": 50.0}},
        "AAPL": {"info": {"longName": "Apple Inc.", "quoteType": "EQUITY", "exchange": "NMS", "sector": "Technology",
                          "industry": "Consumer Electronics", "country": "United States",
                          "sharesOutstanding": 14_800_000_000}},
    }
    return FakeYF(frames, specs)


def test_universe_columns_market_cap_and_spec_filters(tmp_path):
    yf = universe_yf()
    p = make_provider(tmp_path, yf=yf, tickers=["ACME", "BETA", "GIDX", "AAPL"])
    as_of = date(2026, 8, 15)  # a Saturday -> last close is Friday 2026-08-14
    u = p.get_universe(UniverseSpec(), as_of)
    assert list(u.columns) == F.UNIVERSE_COLUMNS and u.index.name == "ticker"
    assert list(u.index) == ["ACME", "BETA", "AAPL"]  # the ETF is excluded by security_types=['common_stock']
    acme = u.loc["ACME"]
    assert acme[F.GICS_SECTOR] == "Industrials" and acme[F.GICS_INDUSTRY] == "Specialty Industrial Machinery"
    assert acme[F.EXCHANGE] == "NYSE" and acme[F.COUNTRY] == "US" and acme[F.CURRENCY] == "USD"
    assert acme[F.SECURITY_TYPE] == "common_stock" and acme[F.VENDOR_ID] == "ACME"
    last_close = price_frame("ACME").loc["2026-08-14", "Close"]
    assert acme[F.MARKET_CAP] == pytest.approx(last_close * 49_500_000)  # SEC dei shares as of 2026-08-15
    beta = u.loc["BETA"]
    assert beta[F.GICS_SECTOR] == "Information Technology" and beta[F.EXCHANGE] == "NASDAQ"  # Yahoo -> GICS naming
    assert beta[F.MARKET_CAP] == pytest.approx(price_frame("BETA").loc["2026-08-14", "Close"] * 20_000_000)
    assert any("BETA" in w and "sharesOutstanding" in w for w in p.warnings)  # fell back to Yahoo shares
    aapl = u.loc["AAPL"]
    assert aapl[F.NAME] == "Apple Inc." and aapl[F.GICS_SECTOR] == "Information Technology"  # starter CSV

    no_it = p.get_universe(UniverseSpec(exclude_sectors=["Information Technology"]), as_of)
    assert list(no_it.index) == ["ACME"]
    etfs = p.get_universe(UniverseSpec(security_types=["etf"]), as_of)
    assert list(etfs.index) == ["GIDX"] and etfs.loc["GIDX", F.EXCHANGE] == "NYSE"  # SEC map exchange wins
    assert sum(yf.info_calls.values()) == 4  # info fetched once per ticker, then memo / disk cache


def test_universe_fails_fast_when_yahoo_is_unreachable(tmp_path):
    """No network (or a proxy refusing the tunnel): every quote request fails with a network error. The
    universe raises ProviderUnavailable at once with connection guidance instead of degrading to NaN and
    letting the price downloads time out and retry for minutes before failing the same way."""

    class OfflineYF(FakeYF):
        def Ticker(self, sym):  # noqa: N802 - mirrors yfinance
            t = FakeTicker(self, sym)

            class _T:
                ticker = t.ticker

                @property
                def info(self):
                    raise ConnectionError("Failed to perform, curl: (7) CONNECT tunnel failed, response 403")

            return _T()

    yf = OfflineYF(universe_yf().frames, universe_yf().specs)
    p = make_provider(tmp_path, yf=yf, tickers=["ACME", "BETA"])
    with pytest.raises(ProviderUnavailable, match="Cannot reach Yahoo Finance.*HTTPS_PROXY"):
        p.get_universe(UniverseSpec(), TODAY)
    assert yf.download_calls == []  # no price download was attempted
    # a quote Yahoo itself rejects (not a network error) still degrades to NaN, as before
    p2 = make_provider(tmp_path / "b", yf=FakeYF({}, {}), tickers=["ACME", "BETA"])
    u = p2.get_universe(UniverseSpec(), TODAY)
    assert list(u.index) == ["ACME", "BETA"] and any("Yahoo quote info (universe): failed" in w for w in p2.warnings)


def test_historical_universe_warns_about_survivorship_bias(tmp_path):
    p = make_provider(tmp_path, yf=universe_yf(), tickers=["ACME", "BETA"])
    p.get_universe(UniverseSpec(), TODAY)
    p.get_universe(UniverseSpec(), date(2026, 9, 28))  # within the staleness window: today's list is fine
    assert not any("SURVIVORSHIP" in w for w in p.warnings)
    p.get_universe(UniverseSpec(), date(2022, 6, 1))
    p.get_universe(UniverseSpec(), date(2023, 6, 30))  # a backtest asks at many past dates: one warning
    surv = [w for w in p.warnings if "SURVIVORSHIP BIAS" in w]
    assert len(surv) == 1 and "your ticker list (2 names)" in surv[0] and "delisted" in surv[0]

    starter = FreeDataProvider(cache=DiskCache(tmp_path / "s"), sec=make_sec(tmp_path / "s")[0], yf_module=FakeYF(),
                               today=lambda: TODAY, sleep=lambda _s: None)
    starter._survivorship_warning()
    assert any("bundled starter list" in w for w in starter.warnings)


def test_price_panel_multi_and_single_ticker_layout(tmp_path):
    frames = {"ACME": price_frame("ACME"), "BETA": price_frame("BETA"), "BRK-B": price_frame("BRK-B"), "BAD": None}
    yf = FakeYF(frames)
    p = make_provider(tmp_path, yf=yf)
    start, end = date(2026, 1, 2), date(2026, 3, 31)
    panel = p.get_price_history(["BETA", "ACME", "BAD", "brk.b"], start, end)
    for frame in (panel.open, panel.high, panel.low, panel.close, panel.volume):
        assert list(frame.columns) == ["BETA", "ACME", "BAD", "brk.b"]  # caller's labels and order
        assert isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is None
        assert frame.index.is_monotonic_increasing and frame.index.is_unique
        assert frame.index.min() >= pd.Timestamp(start) and frame.index.max() <= pd.Timestamp(end)
        assert frame.dtypes.eq("float64").all()
    expected = price_frame("ACME").loc["2026-01-02":"2026-03-31", "Close"]
    np.testing.assert_allclose(panel.close["ACME"].to_numpy(), expected.to_numpy())
    np.testing.assert_allclose(panel.volume["BETA"].to_numpy(),
                               price_frame("BETA").loc["2026-01-02":"2026-03-31", "Volume"].to_numpy())
    assert panel.close["BAD"].isna().all() and panel.close["brk.b"].notna().all()
    assert any("BAD" in w for w in p.warnings)
    assert yf.download_calls[0]["tickers"] == ["ACME", "BAD", "BETA", "BRK-B"]
    assert yf.download_calls[0]["end"] == "2026-04-01"  # yfinance's end is exclusive

    single = p.get_price_history(["ACME"], start, end)
    assert list(single.close.columns) == ["ACME"] and len(single.close) == len(expected)

    n = len(yf.download_calls)
    again = p.get_price_history(["BETA", "ACME", "BAD", "brk.b"], start, end)
    assert len(yf.download_calls) == n  # disk cache hit
    pd.testing.assert_frame_equal(again.close, panel.close)


def test_benchmark_and_total_failure(tmp_path):
    yf = FakeYF({"SPY": price_frame("SPY")})
    p = make_provider(tmp_path, yf=yf)
    s = p.get_benchmark_history(date(2026, 1, 2), date(2026, 1, 30))
    assert isinstance(s, pd.Series) and s.name == "SPY" and s.notna().all() and len(s) == 21
    with pytest.raises(ProviderError, match="No price data"):
        p.get_price_history(["NOPE1", "NOPE2"], date(2026, 1, 2), date(2026, 1, 30))
    with pytest.raises(ProviderError):
        p.get_benchmark_history(date(2026, 1, 2), date(2026, 1, 30), symbol="QQQ")


def test_extract_ohlcv_handles_all_yfinance_layouts():
    single = price_frame("AAA", "2026-01-05", "2026-01-09")
    flat = extract_ohlcv(single, ["AAA"])  # yfinance < 0.2.48 / multi_level_index=False
    assert list(flat["close"].columns) == ["AAA"] and len(flat["close"]) == 5
    by_ticker = pd.concat({"AAA": single, "BBB": single * 2}, axis=1, names=["Ticker", "Price"])  # group_by='ticker'
    out = extract_ohlcv(by_ticker, ["AAA", "BBB"])
    np.testing.assert_allclose(out["close"]["BBB"].to_numpy(), (single["Close"] * 2).to_numpy())
    tz = single.copy()
    tz.index = tz.index.tz_localize("America/New_York")
    assert extract_ohlcv(tz, ["AAA"])["open"].index.tz is None
    assert extract_ohlcv(None, ["AAA"])["close"].empty


def test_price_layout_with_real_yfinance_download(tmp_path, monkeypatch):
    yfinance = pytest.importorskip("yfinance")
    import yfinance.multi as multi

    class HistTicker:  # only what multi._download_one touches
        def __init__(self, ticker, session=None):
            self.ticker = ticker.upper()
            self._price_history = None

        def history(self, start=None, end=None, auto_adjust=True, actions=True, **kw):
            assert auto_adjust is False and actions is True
            if self.ticker == "BAD":
                raise RuntimeError("delisted")
            df = raw_history(price_frame(self.ticker))
            df.index = df.index.tz_localize("America/New_York")  # Yahoo returns exchange-local timestamps
            return df.loc[(df.index >= pd.Timestamp(start, tz="America/New_York"))
                          & (df.index < pd.Timestamp(end, tz="America/New_York"))]

    monkeypatch.setattr(multi, "Ticker", HistTicker)
    p = FreeDataProvider(["AAA"], cache=DiskCache(tmp_path), sec=make_sec(tmp_path)[0], yf_module=yfinance,
                         today=lambda: TODAY, sleep=lambda _s: None)
    panel = p.get_price_history(["AAA", "BBB", "BAD"], date(2026, 2, 2), date(2026, 2, 27))
    assert panel.close.index.tz is None and len(panel.close) == 20
    np.testing.assert_allclose(panel.close["BBB"].to_numpy(),
                               price_frame("BBB").loc["2026-02-02":"2026-02-27", "Close"].to_numpy())
    assert panel.close["BAD"].isna().all()
    one = p.get_price_history(["AAA"], date(2026, 2, 2), date(2026, 2, 27))
    assert list(one.close.columns) == ["AAA"] and one.close["AAA"].notna().all()


# =============================================================================================
# Provider: snapshots (staleness, estimates, short interest, options)
# =============================================================================================


def snapshot_yf() -> FakeYF:
    calls = option_frame([90, 95, 100, 105, 110], [0.40, 0.35, 0.32, 0.30, 0.29], [10, 20, np.nan, 30, 5],
                         [100, 200, 300, 400, 500])
    puts = option_frame([90, 95, 100, 105, 110], [0.45, 0.40, 0.36, 0.33, 0.31], [15, 25, 35, np.nan, 5],
                        [150, 250, 350, 450, 50])
    info = {"quoteType": "EQUITY", "longName": "Acme Widgets Inc.", "sharesShort": 4_000_000,
            "sharesShortPriorMonth": 3_500_000, "floatShares": 48_000_000, "dateShortInterest": epoch("2026-09-15"),
            "trailingEps": 3.5, "targetMeanPrice": 120.0, "numberOfAnalystOpinions": 9,
            "nextFiscalYearEnd": epoch("2026-12-31"), "lastFiscalYearEnd": epoch("2025-12-31")}
    spec = {
        "info": info,
        "expiries": ("2026-10-09", "2026-10-16", "2026-10-23", "2026-10-30", "2026-11-20", "2026-12-18"),
        "chain": (calls, puts, 101.3),
        "earnings_estimate": periodic({
            "0q": {"avg": 1.15, "low": 1.1, "high": 1.2, "yearAgoEps": 0.9, "numberOfAnalysts": 11, "growth": 0.2},
            "+1q": {"avg": 1.2, "low": 1.1, "high": 1.3, "yearAgoEps": 1.0, "numberOfAnalysts": 10, "growth": 0.2},
            "0y": {"avg": 4.0, "low": 3.8, "high": 4.2, "yearAgoEps": 3.3, "numberOfAnalysts": 12, "growth": 0.21},
            "+1y": {"avg": 5.0, "low": 4.5, "high": 5.5, "yearAgoEps": 4.0, "numberOfAnalysts": 12, "growth": 0.25}}),
        "revenue_estimate": periodic({
            "0q": {"avg": 170e6, "low": 165e6, "high": 175e6, "numberOfAnalysts": 9, "yearAgoRevenue": 145e6, "growth": 0.17},
            "0y": {"avg": 660e6, "low": 650e6, "high": 680e6, "numberOfAnalysts": 10, "yearAgoRevenue": 560e6, "growth": 0.18},
            "+1y": {"avg": 760e6, "low": 720e6, "high": 800e6, "numberOfAnalysts": 10, "yearAgoRevenue": 660e6, "growth": 0.15}}),
        "eps_trend": periodic({
            "0q": {"current": 1.15, "7daysAgo": 1.15, "30daysAgo": 1.1, "60daysAgo": 1.1, "90daysAgo": 1.05},
            "0y": {"current": 4.0, "7daysAgo": 4.0, "30daysAgo": 3.9, "60daysAgo": 3.85, "90daysAgo": 3.8},
            "+1y": {"current": 5.0, "7daysAgo": 5.0, "30daysAgo": 4.8, "60daysAgo": 4.7, "90daysAgo": 4.6}}),
        "earnings_history": pd.DataFrame(
            {"epsActual": [0.9, 1.0, 1.0, 1.1], "epsEstimate": [0.85, 1.0, 0.95, 1.0],
             "epsDifference": [0.05, 0.0, 0.05, 0.1], "surprisePercent": [0.0588, 0.0, 0.0526, 0.1]},
            index=pd.DatetimeIndex(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"], name="quarter")),
        "calendar": {"Earnings Date": [date(2026, 10, 27)], "Earnings Average": 1.15, "Revenue Average": 170e6},
    }
    return FakeYF({"ACME": price_frame("ACME")}, {"ACME": spec})


def test_estimates_ntm_blend_revisions_surprise_and_dates(tmp_path):
    yf = snapshot_yf()
    p = make_provider(tmp_path, yf=yf)
    est = p.get_estimates(["ACME"], TODAY)
    assert list(est.columns) == F.ESTIMATE_COLUMNS
    r = est.loc["ACME"]
    w = (pd.Timestamp("2026-12-31") - pd.Timestamp(TODAY)).days / 365.25  # 90/365.25 of FY2026 still ahead
    assert r[F.EPS_NTM_EST] == pytest.approx(w * 4.0 + (1 - w) * 5.0)
    assert r[F.EPS_NTM_EST_3M_AGO] == pytest.approx(w * 3.8 + (1 - w) * 4.6)
    assert r[F.REVENUE_NTM_EST] == pytest.approx(w * 660e6 + (1 - w) * 760e6)
    assert math.isnan(r[F.REVENUE_NTM_EST_3M_AGO])
    assert r[F.EPS_TTM] == 3.5 and r[F.NUM_ANALYSTS] == 12 and r[F.TARGET_PRICE_MEAN] == 120.0
    assert r[F.LAST_EPS_SURPRISE] == pytest.approx(0.10)  # (1.10 - 1.00) / 1.00
    assert r[F.LAST_EARNINGS_DATE] == pd.Timestamp("2026-07-28")  # latest SEC 8-K item 2.02
    assert r[F.NEXT_EARNINGS_DATE] == pd.Timestamp("2026-10-27")
    assert est[F.NEXT_EARNINGS_DATE].dtype == "datetime64[ns]"
    n = yf.ticker_calls["ACME"]
    p2 = make_provider(tmp_path, yf=yf)  # same disk cache
    pd.testing.assert_frame_equal(p2.get_estimates(["ACME"], TODAY), est)
    assert yf.ticker_calls["ACME"] == n


def test_estimates_from_yahoo_handles_missing_pieces():
    row = estimates_from_yahoo({}, None, pd.DataFrame(), None, None, None, TODAY)
    assert set(row) == set(F.ESTIMATE_COLUMNS)
    assert all((v is pd.NaT) or math.isnan(v) for v in row.values())
    only_fy1 = pd.DataFrame({"avg": [5.0]}, index=pd.Index(["+1y"], name="period"))
    late = {"nextFiscalYearEnd": epoch("2026-10-20")}  # FY nearly over -> NTM ~ FY1
    assert estimates_from_yahoo(late, only_fy1, None, None, None, None, TODAY)[F.EPS_NTM_EST] == 5.0


def test_short_interest_current_snapshot(tmp_path):
    p = make_provider(tmp_path, yf=snapshot_yf())
    si = p.get_short_interest(["ACME"], TODAY)
    assert list(si.columns) == F.SHORT_INTEREST_COLUMNS
    assert si.loc["ACME", F.SHORT_INTEREST_SHARES] == 4_000_000
    assert si.loc["ACME", F.SHORT_INTEREST_SHARES_1M_AGO] == 3_500_000
    assert si.loc["ACME", F.FLOAT_SHARES] == 48_000_000
    assert si.loc["ACME", F.SI_SETTLEMENT_DATE] == pd.Timestamp("2026-09-15")


def test_options_expiry_choice_and_atm_iv(tmp_path):
    exps = ["2026-10-09", "2026-10-16", "2026-10-23", "2026-10-30", "2026-11-20"]
    assert choose_expiry(exps, TODAY) == "2026-10-30"  # 28 days: in 20-45 and nearest 30
    assert choose_expiry(["2026-10-05", "2026-10-12", "2026-12-18"], TODAY) == "2026-10-12"  # nearest >= 7 days
    assert choose_expiry(["2026-10-03"], TODAY) is None

    yf = snapshot_yf()
    p = make_provider(tmp_path, yf=yf)
    o = p.get_options_summary(["ACME"], TODAY)
    assert list(o.columns) == F.OPTIONS_COLUMNS
    r = o.loc["ACME"]
    assert r[F.IV_30D_ATM] == pytest.approx((0.32 + 0.36) / 2)  # strike 100 is nearest spot 101.3
    assert r[F.CALL_VOLUME] == 65 and r[F.PUT_VOLUME] == 80
    assert r[F.CALL_OPEN_INTEREST] == 1500 and r[F.PUT_OPEN_INTEREST] == 1250
    assert math.isnan(r[F.IV_30D_ATM_1Y_HIGH]) and math.isnan(r[F.IV_30D_ATM_1Y_LOW])
    assert yf.chain_calls == [("ACME", "2026-10-30")]  # one chain request

    bad_put = option_frame([100, 105], [1e-5, 0.3], [1, 1], [1, 1])  # Yahoo's junk IV on a stale quote
    calls = option_frame([100, 105], [0.31, 0.29], [1, 1], [1, 1])
    assert options_summary(calls, bad_put, 101.0)[F.IV_30D_ATM] == pytest.approx(0.31)
    assert math.isnan(options_summary(None, None, 100.0)[F.IV_30D_ATM])


def test_staleness_rule_blanks_snapshot_datasets_for_historical_as_of(tmp_path):
    yf = snapshot_yf()
    p = make_provider(tmp_path, yf=yf)
    as_of = date(2026, 6, 30)  # > 5 days before TODAY
    assert p.is_stale(as_of) and not p.is_stale(date(2026, 9, 28))
    si = p.get_short_interest(["ACME"], as_of)
    opt = p.get_options_summary(["ACME"], as_of)
    est = p.get_estimates(["ACME"], as_of)
    assert si.isna().all().all() and opt.isna().all().all()
    assert list(si.columns) == F.SHORT_INTEREST_COLUMNS and list(opt.columns) == F.OPTIONS_COLUMNS
    assert est.drop(columns=[F.LAST_EARNINGS_DATE]).isna().all().all()
    assert est.loc["ACME", F.LAST_EARNINGS_DATE] == pd.Timestamp("2026-04-23")  # SEC 8-K, point-in-time
    assert sum(yf.ticker_calls.values()) == 0  # Yahoo never asked for today's snapshot
    stale = [w for w in p.warnings if "cannot give point-in-time" in w]
    assert len(stale) == 3 and any("short interest" in w for w in stale) and any("estimates" in w for w in stale)


# =============================================================================================
# Universe resolution, starter list, protocol
# =============================================================================================


def test_ticker_files_and_normalisation(tmp_path):
    assert normalize_ticker(" brk.b ") == "BRK-B" and normalize_ticker("SHOP.TO") == "SHOP.TO"
    assert normalize_ticker("bf/b") == "BF-B" and normalize_ticker("hei.a") == "HEI-A"
    for foreign in ("VOD.L", "7203.T", "BMW.F", "ABC.V", "0700.HK"):  # Yahoo exchange suffixes are not share classes
        assert normalize_ticker(foreign.lower()) == foreign
    txt = tmp_path / "watch.txt"
    txt.write_text("# my list\nacme\n\nbrk.b  some comment\nACME\n", encoding="utf-8")
    assert read_ticker_file(txt) == ["ACME", "BRK-B"]
    csv_path = tmp_path / "u.csv"
    csv_path.write_text("Symbol,Name\nCROX,Crocs\nDECK,Deckers\n", encoding="utf-8")
    assert read_ticker_file(csv_path) == ["CROX", "DECK"]
    excel_ansi = tmp_path / "excel.csv"  # Excel "CSV (Comma delimited)" on Windows writes cp1252, not UTF-8
    excel_ansi.write_bytes("Symbol,Name\r\nNSRGY,Nestl\u00e9 SA\r\nBRK.B,Berkshire\r\n".encode("cp1252"))
    assert read_ticker_file(excel_ansi) == ["NSRGY", "BRK-B"]
    unicode_txt = tmp_path / "unicode.txt"  # Excel "Unicode Text": UTF-16 with BOM, tab-separated
    unicode_txt.write_bytes("Symbol\tName\r\nSAN\tSoci\u00e9t\u00e9\r\n".encode("utf-16"))
    assert read_ticker_file(unicode_txt) == ["SAN"]
    bad = tmp_path / "bad.csv"
    bad.write_text("name,sector\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ticker"):
        read_ticker_file(bad)
    p = FreeDataProvider(universe_file=csv_path, cache=DiskCache(tmp_path / "c"), yf_module=FakeYF())
    assert p.tickers == ["CROX", "DECK"]
    assert FreeDataProvider(["msft", "msft", "aapl"], cache=DiskCache(tmp_path / "c"), yf_module=FakeYF()).tickers == ["MSFT", "AAPL"]


def test_starter_universe_is_broad_and_mid_cap_tilted():
    u = load_starter_universe()
    gics = {"Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
            "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate", "Utilities"}
    assert 140 <= len(u) <= 160 and u.index.is_unique
    assert set(u["gics_sector"]) == gics
    assert u["gics_sector"].value_counts().min() >= 8
    assert all(re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", t) for t in u.index)
    assert u["name"].str.len().min() > 1
    default = FreeDataProvider(yf_module=FakeYF(), cache=DiskCache(enabled=False))
    assert default.tickers == list(u.index)


def test_provider_satisfies_protocol(tmp_path):
    p = make_provider(tmp_path)
    assert isinstance(p, MarketDataProvider)
    assert p.name == "free"
    assert p.capabilities == {Capability.PRICES, Capability.FUNDAMENTALS, Capability.ESTIMATES,
                              Capability.SHORT_INTEREST, Capability.OPTIONS, Capability.NEWS, Capability.FILINGS}
    assert p.boundary.provider == "free" and "personal research" in p.boundary.note
    assert p.diagnostics() == []


# =============================================================================================
# Concept fallbacks (balance sheet, shares, revenue tag switches)
# =============================================================================================


def _cf(gaap: dict[str, list[dict]], dei: list[dict] | None = None, unit: str = "USD") -> dict:
    out = {"facts": {"us-gaap": {k: {"units": {("shares" if "Shares" in k else unit): v}} for k, v in gaap.items()}}}
    if dei is not None:
        out["facts"]["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": dei}}}
    return out


def _inst(end, val, filed, accn="x"):
    return {"end": end, "val": val, "filed": filed, "accn": accn, "form": "10-Q"}


def _dur(start, end, val, filed, accn="x"):
    return {"start": start, "end": end, "val": val, "filed": filed, "accn": accn, "form": "10-Q"}


def test_balance_sheet_and_share_fallbacks():
    from aitrading.data.sec_edgar import shares_outstanding, total_debt

    as_of = date(2026, 6, 1)
    parts = _cf({"LongTermDebtNoncurrent": [_inst("2026-03-31", 200, "2026-05-01")],
                 "LongTermDebtCurrent": [_inst("2026-03-31", 15, "2026-05-01")],
                 "LongTermDebt": [_inst("2023-12-31", 999, "2024-02-20")],  # stale tag must not win
                 "CommercialPaper": [_inst("2026-03-31", 5, "2026-05-01")]})
    assert total_debt(parts, as_of) == 220
    # A balance sheet without any debt tag is unknown debt (NaN), not an invented 0 (which turned missing interest
    # into interest = 0 and capped interest coverage at 100 downstream).
    assert math.isnan(total_debt(_cf({"StockholdersEquity": [_inst("2026-03-31", 50, "2026-05-01")]}), as_of))
    assert total_debt(_cf({"StockholdersEquity": [_inst("2026-03-31", 50, "2026-05-01")],
                           "LongTermDebt": [_inst("2026-03-31", 0, "2026-05-01")]}), as_of) == 0.0  # explicit zero
    assert math.isnan(total_debt(_cf({}), as_of))

    two_classes = [_inst("2026-04-24", 300, "2026-05-01", "q1"), _inst("2026-04-24", 50, "2026-05-01", "q1"),
                   _inst("2026-01-30", 345, "2026-02-15", "k")]
    assert shares_outstanding(_cf({}, two_classes), as_of) == 350  # classes summed within the latest filing
    equal_classes = [_inst("2026-04-24", 300, "2026-05-01", "q1"), _inst("2026-04-24", 300, "2026-05-01", "q1")]
    assert shares_outstanding(_cf({}, equal_classes), as_of) == 600  # equal counts are two classes, not one
    assert shares_outstanding(_cf({}, two_classes), date(2026, 3, 1)) == 345
    diluted = _cf({"WeightedAverageNumberOfDilutedSharesOutstanding": [
        _dur("2026-01-01", "2026-03-31", 410, "2026-05-01"), _dur("2025-01-01", "2025-12-31", 400, "2026-02-15")]})
    assert shares_outstanding(diluted, as_of) == 410  # no dei -> latest quarterly diluted count


def test_total_assets_point_in_time_on_the_balance_sheet_date():
    from aitrading.data.sec_edgar import total_assets

    eq = [_inst("2026-03-28", 500, "2026-05-01"), _inst("2025-09-27", 480, "2025-11-01"), _inst("2025-03-29", 450, "2025-05-02")]
    assets = [_inst("2026-03-28", 1300, "2026-05-01"), _inst("2025-09-27", 1250, "2025-11-01"),
              _inst("2025-03-29", 1100, "2025-05-02"), _inst("2025-03-29", 1120, "2026-05-01", "restated")]
    cf = _cf({"StockholdersEquity": eq, "Assets": assets})
    # 52/53-week fiscal year: the year-ago quarter ended 364 days earlier; its restatement (filed with the
    # latest 10-Q) is already public and wins
    assert total_assets(cf, date(2026, 6, 1)) == (1300, 1120)
    now, prior = total_assets(cf, date(2026, 4, 1))  # the 2026-03-28 balance sheet is not public yet
    assert now == 1250 and math.isnan(prior)  # no balance sheet a year before 2025-09-27
    now, prior = total_assets(cf, date(2025, 6, 1))
    assert now == 1100 and math.isnan(prior)  # the original value: the restatement was filed in 2026
    # a stale Assets value is never carried onto a newer balance sheet
    stale = _cf({"StockholdersEquity": [_inst("2026-03-31", 500, "2026-05-01")],
                 "Assets": [_inst("2025-12-31", 1200, "2026-02-15")]})
    assert all(math.isnan(x) for x in total_assets(stale, date(2026, 6, 1)))
    # no equity / cash / debt tags: the Assets instant itself dates the balance sheet
    only_assets = _cf({"Assets": [_inst("2026-03-31", 900, "2026-05-01"), _inst("2025-03-31", 800, "2025-05-01")]})
    assert total_assets(only_assets, date(2026, 6, 1)) == (900, 800)
    assert all(math.isnan(x) for x in total_assets(_cf({}), date(2026, 6, 1)))


def test_revenue_tag_switch_and_sixteen_week_quarter():
    rows_new = [_dur(f"2025-{m:02d}-01", e, v, f) for m, e, v, f in
                ((1, "2025-03-31", 10, "2025-05-01"), (4, "2025-06-30", 11, "2025-08-01"),
                 (7, "2025-09-30", 12, "2025-11-01"), (10, "2025-12-31", 13, "2026-02-15"))]
    rows_old = [_dur("2016-01-01", "2016-03-31", 99, "2016-05-01")]  # obsolete tag
    cf = _cf({"Revenues": rows_new, "SalesRevenueNet": rows_old})
    row = fundamentals_from_companyfacts(cf, date(2026, 3, 1))
    assert row[F.REVENUE_TTM] == 46 and row[F.PERIOD_END] == pd.Timestamp("2025-12-31")

    # 12-12-12-16 week calendar: Q1 = 16 weeks (111 days), YTD reported cumulatively.
    sixteen = _cf({"NetCashProvidedByUsedInOperatingActivities": [
        _dur("2025-02-02", "2025-05-24", 40, "2025-06-20"), _dur("2025-02-02", "2025-08-16", 70, "2025-09-15"),
        _dur("2025-02-02", "2025-11-08", 100, "2025-12-10"), _dur("2025-02-02", "2026-01-31", 140, "2026-03-25")]})
    r = fundamentals_from_companyfacts(sixteen, date(2026, 4, 1))
    assert r[F.CFO_TTM] == 140 and r[F.PERIOD_END] == pd.Timestamp("2026-01-31")


# =============================================================================================
# Review regressions
# =============================================================================================


def split_frames(split_day: str = "2026-09-01", ratio: float = 10.0, div_exdate: str = "2026-07-15",
                 div_ratio: float = 0.99) -> tuple[pd.DataFrame, pd.Series]:
    """ACME as Yahoo serves it after a 10:1 split and a dividend: 'Close' restated for the split (pre-split
    prices / 10), 'Adj Close' additionally dividend-adjusted, 'Stock Splits' = 10 on the ex-date.
    Also returns the price actually traded each day (what market cap must use)."""
    idx = pd.bdate_range("2026-04-01", "2026-10-02", name="Date")
    traded = pd.Series(np.where(idx < pd.Timestamp(split_day), 500.0, 50.0) + np.arange(len(idx)) * 0.01, index=idx)
    close = traded.where(idx >= pd.Timestamp(split_day), traded / ratio)
    adj = close.where(idx >= pd.Timestamp(div_exdate), close * div_ratio)
    splits = pd.Series(0.0, index=idx)
    splits[pd.Timestamp(split_day)] = ratio
    df = pd.DataFrame({"Open": close - 0.1, "High": close + 0.2, "Low": close - 0.2, "Close": close, "Adj Close": adj,
                       "Volume": 1_000_000.0, "Dividends": 0.0, "Stock Splits": splits}, index=idx)
    return df, traded


def acme_info() -> dict:
    return {"longName": "Acme Widgets Inc.", "quoteType": "EQUITY", "exchange": "NYQ", "sector": "Industrials",
            "currency": "USD", "sharesOutstanding": 495_000_000}


def test_market_cap_puts_sec_shares_on_the_split_adjusted_price_basis(tmp_path):
    frame, traded = split_frames()
    yf = FakeYF({"ACME": frame}, {"ACME": {"info": acme_info()}})
    p = make_provider(tmp_path, yf=yf)
    # Split (2026-09-01) after as_of: Yahoo's close is 1/10 of the traded price, the SEC count (49.5m at 2026-07-31)
    # is pre-split -> scale the count by the split, not the cap down by 10x.
    u = p.get_universe(None, date(2026, 8, 15))
    assert u.loc["ACME", F.MARKET_CAP] == pytest.approx(traded["2026-08-14"] * 49_500_000)
    # Split between the cover-page date and as_of: 495m shares trade at the post-split price.
    u = p.get_universe(None, date(2026, 9, 15))
    assert u.loc["ACME", F.MARKET_CAP] == pytest.approx(traded["2026-09-15"] * 495_000_000)
    # Before a dividend ex-date the cap uses the traded (split-only) close, not the dividend-adjusted one;
    # 2026-06-30 uses the Q1 10-Q cover count (50.0m at 2026-04-24).
    u = p.get_universe(None, date(2026, 6, 30))
    assert u.loc["ACME", F.MARKET_CAP] == pytest.approx(traded["2026-06-30"] * 50_000_000)
    assert not any("sharesOutstanding" in w for w in p.warnings)  # SEC count used throughout


def test_price_panel_is_split_and_dividend_adjusted_from_raw_columns(tmp_path):
    frame, _ = split_frames()
    p = make_provider(tmp_path, yf=FakeYF({"ACME": frame}))
    panel = p.get_price_history(["ACME"], date(2026, 7, 1), date(2026, 7, 31))
    exp_close = frame.loc["2026-07-01":"2026-07-31", "Adj Close"]
    ratio = exp_close / frame.loc["2026-07-01":"2026-07-31", "Close"]
    np.testing.assert_allclose(panel.close["ACME"].to_numpy(), exp_close.to_numpy())
    np.testing.assert_allclose(panel.open["ACME"].to_numpy(), (frame.loc["2026-07-01":"2026-07-31", "Open"] * ratio).to_numpy())
    assert panel.close["ACME"]["2026-07-14"] / frame.loc["2026-07-14", "Close"] == pytest.approx(0.99)


def test_failed_price_tickers_are_retried_and_never_cached_as_nan(tmp_path):
    frames = {"ACME": price_frame("ACME"), "BETA": price_frame("BETA")}
    yf = FakeYF(frames, fail_first={"BETA": 1})  # e.g. a Yahoo 429 for BETA inside the batch
    p = make_provider(tmp_path, yf=yf)
    panel = p.get_price_history(["ACME", "BETA"], date(2026, 1, 2), date(2026, 1, 30))
    assert panel.close["BETA"].notna().all()
    assert [c["tickers"] for c in yf.download_calls] == [["ACME", "BETA"], ["BETA"]]  # one retry, failed only

    yf2 = FakeYF(frames, fail_first={"BETA": 2})  # still failing after the retry
    p2 = make_provider(tmp_path / "other", yf=yf2)
    panel = p2.get_price_history(["ACME", "BETA"], date(2026, 1, 2), date(2026, 1, 30))
    assert panel.close["BETA"].isna().all() and any("BETA" in w for w in p2.warnings)
    p2.get_price_history(["ACME", "BETA"], date(2026, 1, 2), date(2026, 1, 30))
    assert len(yf2.download_calls) == 2  # not re-requested within the session
    p3 = make_provider(tmp_path / "other", yf=yf2)  # a rerun: ACME from disk, BETA asked again (and now works)
    panel = p3.get_price_history(["ACME", "BETA"], date(2026, 1, 2), date(2026, 1, 30))
    assert yf2.download_calls[-1]["tickers"] == ["BETA"] and panel.close["BETA"].notna().all()


def _recast_facts() -> list[Fact]:
    """FY2025 = 400 (100/quarter); the 2026 10-Qs recast 2025 to 75/quarter (discontinued operations)."""
    rows = []
    for q, (s, e, f_orig, f_recast) in enumerate((("2025-01-01", "2025-03-31", "2025-05-01", "2026-05-01"),
                                                   ("2025-04-01", "2025-06-30", "2025-08-01", "2026-08-01"),
                                                   ("2025-07-01", "2025-09-30", "2025-11-01", "2026-11-01")), start=1):
        rows += [_f(s, e, 100, f_orig, f"o{q}"), _f(s, e, 75, f_recast, f"r{q}")]
        if q > 1:
            rows += [_f("2025-01-01", e, 100 * q, f_orig, f"o{q}"), _f("2025-01-01", e, 75 * q, f_recast, f"r{q}")]
    rows.append(_f("2025-01-01", "2025-12-31", 400, "2026-02-15", "k25"))
    for q, (s, e, filed) in enumerate((("2026-01-01", "2026-03-31", "2026-05-01"), ("2026-04-01", "2026-06-30", "2026-08-01"),
                                       ("2026-07-01", "2026-09-30", "2026-11-01")), start=1):
        rows.append(_f(s, e, 80, filed, f"r{q}"))
        if q > 1:
            rows.append(_f("2026-01-01", e, 80 * q, filed, f"r{q}"))
    return rows


def test_derived_q4_uses_one_restatement_vintage():
    facts = _recast_facts()
    qs = quarterly_series(facts, date(2026, 11, 5))
    q4 = qs[date(2025, 12, 31)]
    assert q4.derived and q4.value == 100  # FY 400 - 9M 300 as filed with the 10-K, not 400 - recast 225 = 175
    assert [qs[date(2025, m, d)].value for m, d in ((3, 31), (6, 30), (9, 30))] == [75, 75, 75]  # latest (recast)
    rev = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": f.start.isoformat(), "end": f.end.isoformat(), "val": f.val, "filed": f.filed.isoformat(),
         "accn": f.accn, "form": f.form} for f in facts]}}}}}
    assert fundamentals_from_companyfacts(rev, date(2026, 11, 5))[F.REVENUE_TTM] == 100 + 3 * 80  # not 175 + 240
    # FY - (Q1 + Q2 + Q3) when no 9M YTD is tagged: the quarters current when the FY was filed.
    no_ytd = [f for f in facts if not (f.start == date(2025, 1, 1) and f.end in (date(2025, 6, 30), date(2025, 9, 30)))]
    assert quarterly_series(no_ytd, date(2026, 11, 5))[date(2025, 12, 31)].value == 100


def test_revenue_prefers_total_revenues_over_asc606_subset():
    rows = lambda v: [_dur(f"2025-{m:02d}-01", e, v, f, a) for m, e, f, a in  # noqa: E731
                      ((1, "2025-03-31", "2025-05-01", "q1"), (4, "2025-06-30", "2025-08-01", "q2"),
                       (7, "2025-09-30", "2025-11-01", "q3"), (10, "2025-12-31", "2026-02-15", "k"))]
    reit = _cf({"Revenues": rows(100), "RevenueFromContractWithCustomerExcludingAssessedTax": rows(15)})  # lease income
    row = fundamentals_from_companyfacts(reit, date(2026, 3, 1))
    assert row[F.REVENUE_TTM] == 400 and row[F.REVENUE_LAST_Q] == 100
    only_606 = _cf({"RevenueFromContractWithCustomerExcludingAssessedTax": rows(15)})
    assert fundamentals_from_companyfacts(only_606, date(2026, 3, 1))[F.REVENUE_TTM] == 60


def test_total_debt_only_counts_the_current_balance_sheet():
    from aitrading.data.sec_edgar import total_debt

    as_of = date(2026, 9, 1)
    eq = {"StockholdersEquity": [_inst("2026-06-30", 900, "2026-08-01")]}
    stale = _cf({**eq, "LongTermDebt": [_inst("2020-12-31", 5e9, "2021-02-20")],
                 "ShortTermBorrowings": [_inst("2026-03-31", 7e8, "2026-05-01")]})
    assert math.isnan(total_debt(stale, as_of))  # 2020 debt and a repaid borrowing are not today's debt
    parts = _cf({**eq, "LongTermDebtNoncurrent": [_inst("2026-06-30", 1e9, "2026-08-01")],
                 "ShortTermBorrowings": [_inst("2026-03-31", 7e8, "2026-05-01")]})
    assert total_debt(parts, as_of) == 1e9
    convert = _cf({**eq, "ConvertibleNotesPayable": [_inst("2026-06-30", 2e9, "2026-08-01")]})
    assert total_debt(convert, as_of) == 2e9  # other standard debt tags are recognised
    notes = _cf({**eq, "LongTermNotesPayable": [_inst("2026-06-30", 3e8, "2026-08-01")],
                 "LongTermLineOfCredit": [_inst("2026-06-30", 5e7, "2026-08-01")],
                 "DebtCurrent": [_inst("2026-06-30", 4e7, "2026-08-01")]})
    assert total_debt(notes, as_of) == 3e8 + 5e7 + 4e7
    with_dc = _cf({**eq, "LongTermDebtNoncurrent": [_inst("2026-06-30", 1e9, "2026-08-01")],
                   "LongTermDebtCurrent": [_inst("2026-06-30", 1e8, "2026-08-01")],
                   "ShortTermBorrowings": [_inst("2026-06-30", 5e7, "2026-08-01")],
                   "DebtCurrent": [_inst("2026-06-30", 1.5e8, "2026-08-01")]})
    assert total_debt(with_dc, as_of) == 1.15e9  # DebtCurrent = current maturities + borrowings, not added twice


def test_extract_mdna_ignores_open_ended_cross_reference():
    real = "Item 2. Management's Discussion and Analysis of Financial Condition\n\n" + "Revenue grew on volume. " * 230
    text = (real + "\n\nItem 3. Quantitative and Qualitative Disclosures About Market Risk\n\nNone.\n\n"
            "PART II OTHER INFORMATION\n\nItem 1A. Risk Factors\n\nNo changes. See Item 2. Management's Discussion and "
            "Analysis for more.\n\n" + "Risk text. " * 3000)
    out = extract_mdna(text, "10-Q")
    assert out.startswith("Item 2. Management's Discussion and Analysis of Financial Condition")
    assert "Risk text" not in out and "Revenue grew on volume." in out


def test_shares_info_flags_foreign_issuers_and_share_classes():
    from aitrading.data.sec_edgar import shares_info, unsupported_reason

    adr = _cf({}, [{"end": "2026-03-31", "val": 25.9e9, "filed": "2026-04-15", "accn": "f", "form": "20-F"}])
    si = shares_info(adr, date(2026, 6, 1))
    assert si.foreign_issuer and si.basis_date == date(2026, 3, 31) and si.form == "20-F"
    brk = _cf({}, [_inst("2026-04-20", 570_000, "2026-05-01", "q"), _inst("2026-04-20", 1.3e9, "2026-05-01", "q")])
    si = shares_info(brk, date(2026, 6, 1))
    assert si.n_classes == 2 and si.value == 1.3e9 + 570_000 and not si.foreign_issuer
    assert unsupported_reason({"taxonomies": ["dei", "ifrs-full"], "facts": {"us-gaap": {}, "dei": {}}}).startswith("files IFRS")
    cny = {"facts": {"us-gaap": {"Revenues": {"units": {"CNY": [{"end": "2026-03-31", "val": 1}]}}}}}
    assert "CNY" in unsupported_reason(cny)
    assert unsupported_reason(_fx_json("companyfacts_CIK0001234567.json")) is None


def test_market_cap_for_adrs_and_multi_class_filers(tmp_path):
    router = SecRouter()
    adr_cf = {"cik": 7654321, "entityName": "Beta ADR plc", "facts": {
        "ifrs-full": {"Revenue": {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": 1e9,
                                                     "filed": "2026-04-15", "accn": "f", "form": "20-F"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-12-31", "val": 25.9e9, "filed": "2026-04-15", "accn": "f", "form": "20-F"}]}}}}}
    multi_cf = {"cik": 1111111, "entityName": "Gamma Holdings", "facts": {"us-gaap": {}, "dei": {
        "EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2026-07-20", "val": 570_000, "filed": "2026-08-03", "accn": "q", "form": "10-Q"},
            {"end": "2026-07-20", "val": 1.3e9, "filed": "2026-08-03", "accn": "q", "form": "10-Q"}]}}}}}
    router.overrides["https://data.sec.gov/api/xbrl/companyfacts/CIK0007654321.json"] = [httpx.Response(200, json=adr_cf)]
    router.overrides["https://data.sec.gov/api/xbrl/companyfacts/CIK0001111111.json"] = [httpx.Response(200, json=multi_cf)]
    cache = DiskCache(tmp_path / "cache")
    sec, _ = make_sec(tmp_path, router, cache=cache)
    specs = {"BETA": {"info": {"longName": "Beta ADR plc", "quoteType": "EQUITY", "sector": "Technology",
                               "currency": "USD", "sharesOutstanding": 5.18e9, "impliedSharesOutstanding": 5.18e9}},
             "GIDX": {"info": {"longName": "Gamma Holdings", "quoteType": "EQUITY", "sector": "Financial Services",
                               "currency": "USD", "sharesOutstanding": 1.3e9, "impliedSharesOutstanding": 2.16e9}}}
    yf = FakeYF({"BETA": price_frame("BETA"), "GIDX": price_frame("GIDX")}, specs)
    p = make_provider(tmp_path, yf=yf, sec=sec, tickers=["BETA", "GIDX"])
    u = p.get_universe(None, date(2026, 8, 15))
    assert u.loc["BETA", F.MARKET_CAP] == pytest.approx(price_frame("BETA").loc["2026-08-14", "Close"] * 5.18e9)
    assert u.loc["GIDX", F.MARKET_CAP] == pytest.approx(price_frame("GIDX").loc["2026-08-14", "Close"] * 2.16e9)
    assert any("BETA" in w and "20-F" in w for w in p.warnings)
    assert any("GIDX" in w and "share classes" in w for w in p.warnings)
    # Yahoo's share counts are today's: at a historical as_of the snapshot lists those names, so a
    # backtest reports their caps as not point-in-time; at today's date nothing is flagged.
    from aitrading.backtest.runner import MCAP_CURRENT_SHARES_ATTR
    assert sorted(u.attrs[MCAP_CURRENT_SHARES_ATTR]) == ["BETA", "GIDX"]
    assert MCAP_CURRENT_SHARES_ATTR not in p.get_universe(None, TODAY).attrs
    # Near-equal classes (GOOGL-like): Yahoo's implied count agrees, so the point-in-time SEC sum is kept.
    from aitrading.data.sec_edgar import SharesInfo
    caps = p._market_caps(["GIDX"], date(2026, 8, 15), {"GIDX": {"impliedSharesOutstanding": 1.31e9}},
                          {"GIDX": SharesInfo(1.3e9 + 570_000, date(2026, 7, 20), "dei", "10-Q", 2)})
    assert caps["GIDX"] == pytest.approx(price_frame("GIDX").loc["2026-08-14", "Close"] * (1.3e9 + 570_000))
    # IFRS filer: fundamentals are NaN with an explicit reason, not silently.
    fa = p.get_fundamentals(["BETA"], date(2026, 8, 15))
    assert fa.drop(columns=[F.PERIOD_END, F.REPORT_DATE]).isna().all().all()
    assert any("BETA" in w and "IFRS" in w for w in p.warnings)


def test_earnings_8k_amendments_are_not_release_events(tmp_path):
    sub = _fx_json("submissions_CIK0001234567.json")
    recent = sub["filings"]["recent"]
    for col, val in (("accessionNumber", "0001234567-26-000033"), ("filingDate", "2026-08-10"), ("reportDate", "2026-07-28"),
                     ("form", "8-K/A"), ("primaryDocument", "acme-20260728a.htm"), ("items", "2.02,9.01"),
                     ("acceptanceDateTime", "2026-08-10T16:05:00.000Z"), ("isXBRL", 0)):
        recent[col].insert(1, val)
    router = SecRouter()
    router.overrides["https://data.sec.gov/submissions/CIK0001234567.json"] = [httpx.Response(200, json=sub)]
    sec, _ = make_sec(tmp_path, router)
    assert sec.last_earnings_release_date("ACME", date(2026, 8, 15)) == date(2026, 7, 28)  # not the 8-K/A date
    p = make_provider(tmp_path, sec=sec)
    news = p.get_documents("ACME", {DocumentKind.NEWS}, date(2026, 7, 1), date(2026, 9, 30))
    assert [d.doc_id for d in news] == ["SEC-0001234567-26-000031-EX99"]  # no second document for the same quarter


def test_last_surprise_only_uses_results_public_on_as_of():
    # Fiscal quarters end Feb/May/Aug; the Aug quarter is reported 2026-09-30, after as_of 2026-09-28.
    eh = pd.DataFrame({"epsActual": [1.0, 1.1, 1.5], "epsEstimate": [0.95, 1.0, 1.0]},
                      index=pd.DatetimeIndex(["2026-02-28", "2026-05-31", "2026-08-31"], name="quarter"))
    as_of = date(2026, 9, 28)
    row = estimates_from_yahoo({}, None, None, None, eh, None, as_of, last_earnings_date=date(2026, 6, 30))
    assert row[F.LAST_EPS_SURPRISE] == pytest.approx(0.10)  # May quarter, not the Aug one (0.5)
    info = {"earningsTimestamp": epoch("2026-09-30 20:05")}  # no SEC date: Yahoo's release time says the same
    assert estimates_from_yahoo(info, None, None, None, eh, None, as_of)[F.LAST_EPS_SURPRISE] == pytest.approx(0.10)
    assert estimates_from_yahoo({}, None, None, None, eh, None, date(2026, 10, 2),
                                last_earnings_date=date(2026, 9, 30))[F.LAST_EPS_SURPRISE] == pytest.approx(0.5)


def test_snapshot_for_recent_as_of_warns_it_is_todays_data(tmp_path):
    p = make_provider(tmp_path, yf=snapshot_yf())
    p.get_short_interest(["ACME"], date(2026, 9, 28))
    assert any("4 day(s) before today" in w and "short interest" in w for w in p.warnings)
    q = make_provider(tmp_path / "q", yf=snapshot_yf())
    q.get_short_interest(["ACME"], TODAY)
    assert not any("day(s) before today" in w for w in q.warnings)


class YFRateLimitError(Exception):  # same class name as yfinance.exceptions.YFRateLimitError
    pass


def test_rate_limited_estimates_warn_and_are_not_cached(tmp_path):
    yf = snapshot_yf()
    yf.specs["ACME"]["raises"] = {"earnings_estimate": [YFRateLimitError("Too Many Requests")] * 3}
    slept: list[float] = []
    p = make_provider(tmp_path, yf=yf)
    p._sleep = slept.append
    r = p.get_estimates(["ACME"], TODAY).loc["ACME"]
    assert len(slept) == 2  # retried with back-off before giving up
    assert r[F.REVENUE_NTM_EST] > 0  # the other modules still count
    assert any("incomplete" in w and "YFRateLimitError" in w for w in p.warnings)
    n = yf.ticker_calls["ACME"]
    p2 = make_provider(tmp_path, yf=yf)  # same disk cache: nothing cached, Yahoo is asked again
    r2 = p2.get_estimates(["ACME"], TODAY).loc["ACME"]
    assert yf.ticker_calls["ACME"] > n and r2[F.EPS_NTM_EST] > 0 and not any("incomplete" in w for w in p2.warnings)

    once = snapshot_yf()  # a single 429 that clears on the retry is invisible
    once.specs["ACME"]["raises"] = {"eps_trend": [YFRateLimitError("Too Many Requests")]}
    p3 = make_provider(tmp_path / "once", yf=once)
    assert p3.get_estimates(["ACME"], TODAY).loc["ACME", F.EPS_NTM_EST_3M_AGO] > 0
    assert not any("incomplete" in w for w in p3.warnings)


def test_empty_estimates_are_flagged_and_cached_briefly(tmp_path):
    clock = FakeClock(time_now := 2_000_000_000.0)
    cache = DiskCache(tmp_path / "c", clock=clock)
    yf = FakeYF({}, {"ACME": {"info": {"quoteType": "EQUITY", "longName": "Acme", "sector": "Industrials"}}})
    mk = lambda: FreeDataProvider(["ACME"], cache=cache, sec=make_sec(tmp_path, cache=cache)[0], yf_module=yf,  # noqa: E731
                                  today=lambda: TODAY, sleep=lambda _s: None)
    p = mk()
    p.get_estimates(["ACME"], TODAY)
    assert any("No Yahoo consensus estimates" in w and "ACME" in w for w in p.warnings)
    n = yf.ticker_calls["ACME"]
    clock.t = time_now + 20 * 60  # 20 minutes later: asked again (an HTTP error also looks like "empty")
    mk().get_estimates(["ACME"], TODAY)
    assert yf.ticker_calls["ACME"] > n


def test_non_usd_estimates_are_blank_with_a_warning(tmp_path):
    yf = snapshot_yf()
    spec = yf.specs["ACME"]
    spec["revenue_estimate"] = spec["revenue_estimate"].assign(currency="TWD")
    spec["info"] = {**spec["info"], "financialCurrency": "TWD"}
    p = make_provider(tmp_path, yf=yf)
    r = p.get_estimates(["ACME"], TODAY).loc["ACME"]
    assert math.isnan(r[F.REVENUE_NTM_EST]) and math.isnan(r[F.EPS_TTM])
    assert r[F.EPS_NTM_EST] > 0  # EPS estimates are in USD here
    assert any("ACME" in w and "TWD" in w and "USD" in w for w in p.warnings)
    p2 = make_provider(tmp_path, yf=yf)  # served from cache: the warning is repeated
    p2.get_estimates(["ACME"], TODAY)
    assert any("TWD" in w for w in p2.warnings)


def test_degraded_yahoo_quote_is_cached_for_minutes_only(tmp_path):
    clock = FakeClock(time_now := 2_000_000_000.0)
    cache = DiskCache(tmp_path / "c", clock=clock)
    v7_only = {"quoteType": "EQUITY", "longName": "Acme Widgets Inc.", "exchange": "NYQ", "currency": "USD",
               "sharesOutstanding": 49_000_000}  # what Ticker.info holds when the quoteSummary call failed
    yf = FakeYF({}, {"ACME": {"info": v7_only}})
    mk = lambda: FreeDataProvider(["ACME"], cache=cache, sec=make_sec(tmp_path, cache=cache)[0], yf_module=yf,  # noqa: E731
                                  today=lambda: TODAY, sleep=lambda _s: None)
    p = mk()
    p.get_short_interest(["ACME"], TODAY)
    assert any("basic quote" in w and "ACME" in w for w in p.warnings)
    clock.t = time_now + 20 * 60
    yf.specs["ACME"]["info"] = {**v7_only, "sector": "Industrials", "sharesShort": 4_000_000}
    si = mk().get_short_interest(["ACME"], TODAY)
    assert yf.info_calls["ACME"] == 2 and si.loc["ACME", F.SHORT_INTEREST_SHARES] == 4_000_000
    clock.t = time_now + 3 * 3600  # a complete quote is kept for a day
    mk().get_short_interest(["ACME"], TODAY)
    assert yf.info_calls["ACME"] == 2


@pytest.mark.skipif(not hasattr(__import__("time"), "tzset"), reason="time.tzset is POSIX-only")
def test_next_earnings_date_is_not_shifted_by_the_pc_time_zone(monkeypatch):
    import time as _time

    release = epoch("2026-10-29 20:05")  # after the close in New York
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    _time.tzset()
    try:
        local_day = datetime.fromtimestamp(release).date()  # what yfinance puts in Ticker.calendar
        assert local_day == date(2026, 10, 30)
        info = {"earningsTimestamp": release, "earningsTimestampStart": release}
        row = estimates_from_yahoo(info, None, None, None, None, {"Earnings Date": [local_day]}, TODAY)
        assert row[F.NEXT_EARNINGS_DATE] == pd.Timestamp("2026-10-29")
    finally:
        monkeypatch.delenv("TZ")
        _time.tzset()
