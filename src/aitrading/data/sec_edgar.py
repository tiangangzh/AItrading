"""SEC EDGAR client for the free-data provider: point-in-time fundamentals and filing text.

Everything here is public, free and point-in-time:

* ``company_tickers_exchange.json`` maps tickers to CIKs and listing exchanges.
* XBRL ``companyfacts`` gives every reported value together with the date it was *filed*, so a
  fundamentals snapshot "as of D" only uses facts with ``filed <= D`` (no look-ahead, and the value
  of a period that was later restated is the one that was public on D).
* ``submissions`` lists filings; 8-K item 2.02 ("Results of Operations") filings carry the earnings
  press release as exhibit 99.1, and 10-Q / 10-K primary documents contain the MD&A.

Narrative proxy: earnings-call *transcripts* are not available from any official free source
(they are licensed content). The free build therefore feeds the narrative engine the 8-K earnings
press release (``DocumentKind.NEWS``, metadata ``source_type="earnings_release_8k"``) and the
10-Q Item 2 / 10-K Item 7 MD&A (``DocumentKind.FILING``) - management's own framing of the quarter.

Fair access: the SEC requires a ``User-Agent`` naming the requester ("Name email@domain") and at most
10 requests per second. Set ``SEC_USER_AGENT``; this client throttles itself to 8 req/s across
threads, retries 429/5xx with exponential back-off and caches responses on disk
(companyfacts 1 day, submissions 6 hours, ticker map 7 days, filing documents forever).
"""

from __future__ import annotations

import calendar
import math
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from html.parser import HTMLParser
from typing import Any, Callable, Iterable

import httpx
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.data.base import ProviderError, ProviderUnavailable
from aitrading.data.cache import DAY, HOUR, DiskCache

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"

TTL_TICKER_MAP = 7 * DAY
TTL_COMPANYFACTS = 1 * DAY
TTL_SUBMISSIONS = 6 * HOUR
TTL_SUBMISSIONS_PAGE = 7 * DAY  # older filing pages only grow at the "recent" end
TTL_DOCUMENT = None  # filed documents never change

MAX_DOC_CHARS = 60_000
RETRY_STATUSES = {429, 500, 502, 503, 504}

SOURCE = "SEC EDGAR"


def user_agent_help(problem: str = "SEC_USER_AGENT is not set.") -> str:
    return (
        f"{problem} The SEC requires every automated client to identify itself with a name and an "
        "email address (https://www.sec.gov/os/accessing-edgar-data). Set it and run again:\n"
        '  Windows PowerShell:  $env:SEC_USER_AGENT="Jane Doe jane@example.com"\n'
        "  Windows cmd.exe:     set SEC_USER_AGENT=Jane Doe jane@example.com\n"
        '  macOS / Linux:       export SEC_USER_AGENT="Jane Doe jane@example.com"\n'
        'To make it permanent: Windows -> Start -> "Edit environment variables for your account"; '
        "macOS/Linux -> add the export line to ~/.zshrc or ~/.bashrc."
    )


class SecNotFound(ProviderError):
    """The SEC has no such resource (HTTP 404), e.g. a company without XBRL financial data."""


# =============================================================================================
# Rate limiting
# =============================================================================================


class RateLimiter:
    """Spaces calls at least ``1 / rate`` seconds apart across all threads."""

    def __init__(self, rate_per_s: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
            wait = start - now
        if wait > 0:
            self._sleep(wait)


# =============================================================================================
# Reference data / filings
# =============================================================================================


@dataclass(frozen=True)
class CompanyRef:
    cik: int
    name: str
    ticker: str
    exchange: str | None


@dataclass(frozen=True)
class Filing:
    cik: int
    accession: str  # "0000320193-25-000071"
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str
    items: tuple[str, ...] = ()

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def base_url(self) -> str:
        return ARCHIVE_URL.format(cik=int(self.cik), acc=self.accession_nodash)

    @property
    def index_url(self) -> str:
        return self.base_url + "index.json"

    @property
    def primary_url(self) -> str:
        return self.base_url + self.primary_document

    @property
    def is_earnings_release(self) -> bool:
        return self.form.upper().startswith("8-K") and "2.02" in self.items


def normalize_exchange(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip()
    up = s.upper()
    table = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "CBOE": "CBOE", "OTC": "OTC", "NYSE AMERICAN": "NYSE American",
             "NYSE MKT": "NYSE American", "NYSE ARCA": "NYSE Arca", "BATS": "CBOE"}
    return table.get(up, s)


US_EXCHANGES = {"NYSE", "NASDAQ", "CBOE", "NYSE American", "NYSE Arca"}


def _parse_date(s: Any) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def parse_filings(cik: int, block: dict) -> list[Filing]:
    """Columnar ``filings.recent`` (or an older ``files`` page) -> list of Filing."""
    acc = block.get("accessionNumber") or []
    out: list[Filing] = []

    def col(name: str, i: int) -> Any:
        arr = block.get(name) or []
        return arr[i] if i < len(arr) else None

    for i in range(len(acc)):
        fd = _parse_date(col("filingDate", i))
        if fd is None or not acc[i]:
            continue
        items_raw = col("items", i) or ""
        items = tuple(x.strip() for x in str(items_raw).split(",") if x.strip())
        out.append(Filing(
            cik=int(cik), accession=str(acc[i]), form=str(col("form", i) or ""), filing_date=fd,
            report_date=_parse_date(col("reportDate", i)), primary_document=str(col("primaryDocument", i) or ""),
            items=items,
        ))
    return out


_KEEP_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A", "10-KT", "10-QT", "20-F", "40-F", "6-K"}
_SUBMISSION_COLS = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument", "items"]


def _reduce_submission_block(block: dict) -> dict:
    """Keep only the filing types this client reads (Form 4s etc. dominate large filers' lists)."""
    forms = block.get("form") or []
    keep = [i for i, f in enumerate(forms) if f in _KEEP_FORMS]
    out: dict[str, list] = {}
    for c in _SUBMISSION_COLS:
        arr = block.get(c) or []
        out[c] = [arr[i] if i < len(arr) else None for i in keep]
    return out


def _reduce_submissions(sub: dict) -> dict:
    filings = sub.get("filings") or {}
    return {
        "cik": sub.get("cik"),
        "name": sub.get("name"),
        "tickers": sub.get("tickers") or [],
        "exchanges": sub.get("exchanges") or [],
        "fiscalYearEnd": sub.get("fiscalYearEnd"),
        "filings": {
            "recent": _reduce_submission_block(filings.get("recent") or {}),
            "files": filings.get("files") or [],
        },
    }


# =============================================================================================
# XBRL companyfacts -> point-in-time quarterly series
# =============================================================================================

Q_MIN, Q_MAX = 80, 100  # quarter duration (end - start) in days; 13/14-week quarters included
FY_MIN, FY_MAX = 350, 380  # fiscal year (52/53-week years included)
_ONE = timedelta(days=1)

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
GROSS_PROFIT_CONCEPTS = ["GrossProfit"]
COST_OF_REVENUE_CONCEPTS = ["CostOfRevenue", "CostOfGoodsAndServicesSold"]
OPERATING_INCOME_CONCEPTS = ["OperatingIncomeLoss"]
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
CFO_CONCEPTS = ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX_CONCEPTS = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
DA_CONCEPTS = ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"]
INTEREST_CONCEPTS = ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"]
CASH_CONCEPTS = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
EQUITY_CONCEPTS = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
LTD_TOTAL = "LongTermDebt"  # includes current maturities
LTD_PARTS = [("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
             ("LongTermDebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsCurrent")]
SHORT_TERM_DEBT_CONCEPTS = ["ShortTermBorrowings", "CommercialPaper"]
DILUTED_SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding"
DEI_SHARES = "EntityCommonStockSharesOutstanding"

USGAAP_CONCEPTS = sorted({
    *REVENUE_CONCEPTS, *GROSS_PROFIT_CONCEPTS, *COST_OF_REVENUE_CONCEPTS, *OPERATING_INCOME_CONCEPTS,
    *NET_INCOME_CONCEPTS, *CFO_CONCEPTS, *CAPEX_CONCEPTS, *DA_CONCEPTS, *INTEREST_CONCEPTS, *CASH_CONCEPTS,
    *EQUITY_CONCEPTS, LTD_TOTAL, *(c for pair in LTD_PARTS for c in pair), *SHORT_TERM_DEBT_CONCEPTS, DILUTED_SHARES,
})
DEI_CONCEPTS = [DEI_SHARES]


def reduce_companyfacts(cf: dict) -> dict:
    """Keep only the concepts this module reads (companyfacts can be 10+ MB for large filers)."""
    facts = cf.get("facts") or {}
    gaap = facts.get("us-gaap") or {}
    dei = facts.get("dei") or {}
    return {
        "cik": cf.get("cik"),
        "entityName": cf.get("entityName"),
        "facts": {
            "us-gaap": {k: {"units": gaap[k].get("units", {})} for k in USGAAP_CONCEPTS if k in gaap},
            "dei": {k: {"units": dei[k].get("units", {})} for k in DEI_CONCEPTS if k in dei},
        },
    }


@dataclass(frozen=True)
class Fact:
    start: date | None
    end: date
    val: float
    filed: date
    accn: str
    form: str


@dataclass(frozen=True)
class Quarter:
    start: date
    end: date
    value: float
    filed: date  # filing date of the latest component used (<= as_of)
    first_filed: date  # when this quarter first became public
    derived: bool = False


def facts_for(cf: dict, concept: str, taxonomy: str = "us-gaap", unit: str = "USD") -> list[Fact]:
    node = ((cf.get("facts") or {}).get(taxonomy) or {}).get(concept) or {}
    rows = (node.get("units") or {}).get(unit) or []
    out: list[Fact] = []
    for r in rows:
        try:
            end = _parse_date(r.get("end"))
            filed = _parse_date(r.get("filed"))
            val = r.get("val")
            if end is None or filed is None or val is None:
                continue
            val = float(val)
            if not math.isfinite(val):
                continue
            out.append(Fact(_parse_date(r.get("start")), end, val, filed, str(r.get("accn") or ""), str(r.get("form") or "")))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _latest_by_period(facts: Iterable[Fact], as_of: date, *, durations: bool) -> tuple[dict, dict]:
    """(start, end) -> latest fact with filed <= as_of, plus (start, end) -> first filed date."""
    best: dict[tuple, Fact] = {}
    first: dict[tuple, date] = {}
    for f in facts:
        if f.filed > as_of or (f.start is None) == durations:
            continue
        key = (f.start, f.end)
        cur = best.get(key)
        if cur is None or (f.filed, f.accn) > (cur.filed, cur.accn):
            best[key] = f
        if key not in first or f.filed < first[key]:
            first[key] = f.filed
    return best, first


def _find(quarters: dict[date, Quarter], target: date, tol_days: int) -> Quarter | None:
    best: Quarter | None = None
    best_gap = tol_days + 1
    for end, q in quarters.items():
        gap = abs((end - target).days)
        if gap < best_gap:
            best, best_gap = q, gap
    return best


def quarterly_series(facts: Iterable[Fact], as_of: date) -> dict[date, Quarter]:
    """Point-in-time discrete quarterly values (keyed by quarter end) for one duration concept.

    1. Only facts filed on/before ``as_of``; for a period reported several times (original filing,
       comparatives, restatements) the latest filing wins.
    2. ~3-month facts are quarters directly.
    3. Year-to-date facts (3/6/9/12 months from the same fiscal-year start, typical for cash-flow
       items in 10-Qs) are differenced: quarter ending E = YTD(E) - YTD(previous quarter end). This
       also derives Q4 = FY - 9M.
    4. If a fiscal year has no 9-month YTD, Q4 = FY - (Q1 + Q2 + Q3).
    Directly reported quarters take precedence over derived ones.
    """
    best, first = _latest_by_period(facts, as_of, durations=True)
    out: dict[date, Quarter] = {}
    for (s, e), f in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if Q_MIN <= (e - s).days <= Q_MAX and _find(out, e, 3) is None:
            out[e] = Quarter(s, e, f.val, f.filed, first[(s, e)], False)

    by_start: dict[date, list[tuple[date, Fact, date]]] = defaultdict(list)
    for (s, e), f in best.items():
        if (e - s).days <= FY_MAX:
            by_start[s].append((e, f, first[(s, e)]))
    for s in sorted(by_start):
        items = sorted(by_start[s], key=lambda x: x[0])
        for (e0, f0, ff0), (e1, f1, ff1) in zip(items, items[1:]):
            if Q_MIN <= (e1 - e0).days <= Q_MAX and _find(out, e1, 3) is None:
                out[e1] = Quarter(e0 + _ONE, e1, f1.val - f0.val, max(f0.filed, f1.filed), max(ff0, ff1), True)

    for (s, e), f in sorted(best.items(), key=lambda kv: kv[0][1]):
        if not (FY_MIN <= (e - s).days <= FY_MAX) or _find(out, e, 3) is not None:
            continue
        chain: list[Quarter] = []
        cursor = s
        for _ in range(3):
            nxt = next((q for q in out.values() if abs((q.start - cursor).days) <= 5 and q.end < e), None)
            if nxt is None:
                break
            chain.append(nxt)
            cursor = nxt.end + _ONE
        if len(chain) == 3 and Q_MIN <= (e - chain[-1].end).days <= Q_MAX + 1:
            out[e] = Quarter(chain[-1].end + _ONE, e, f.val - sum(q.value for q in chain),
                             max([f.filed, *(q.filed for q in chain)]), max([first[(s, e)], *(q.first_filed for q in chain)]), True)
    return dict(sorted(out.items()))


def merged_quarters(cf: dict, concepts: list[str], as_of: date) -> dict[date, Quarter]:
    """Quarterly series using the first concept (in priority order) that is current, gaps filled from the others.

    "Current" = reaches the latest quarter end seen across the concepts, so a tag the company stopped
    using years ago (e.g. ``SalesRevenueNet``) never shadows the one it reports today, while its
    history still fills older quarters.
    """
    series = [quarterly_series(facts_for(cf, c), as_of) for c in concepts]
    series = [s for s in series if s]
    if not series:
        return {}
    latest = max(max(s) for s in series)
    primary = next(s for s in series if max(s) >= latest - timedelta(days=10))
    merged = dict(primary)
    for s in series:
        if s is primary:
            continue
        for e, q in s.items():
            if _find(merged, e, 10) is None:
                merged[e] = q
    return dict(sorted(merged.items()))


def ttm(quarters: dict[date, Quarter], anchor: date | None, n: int = 4, tol_days: int = 10) -> float:
    """Sum of ``n`` contiguous quarters ending at ``anchor`` (NaN if any is missing)."""
    if anchor is None or not quarters:
        return math.nan
    q = _find(quarters, anchor, tol_days)
    total = 0.0
    for i in range(n):
        if q is None or not math.isfinite(q.value):
            return math.nan
        total += q.value
        if i < n - 1:
            q = _find(quarters, q.start - _ONE, tol_days)
    return total


def quarter_value(quarters: dict[date, Quarter], anchor: date | None, tol_days: int = 10) -> float:
    if anchor is None:
        return math.nan
    q = _find(quarters, anchor, tol_days)
    return q.value if q is not None else math.nan


def latest_instant(facts: Iterable[Fact], as_of: date) -> tuple[date, float] | None:
    """Latest balance-sheet value (max period end) among facts filed on/before ``as_of``."""
    best, _ = _latest_by_period(facts, as_of, durations=False)
    if not best:
        return None
    (_, end), f = max(best.items(), key=lambda kv: kv[0][1])
    return end, f.val


def instant_at(facts: Iterable[Fact], as_of: date, end: date, tol_days: int = 7) -> float | None:
    best, _ = _latest_by_period(facts, as_of, durations=False)
    cands = [(abs((e - end).days), f.val) for (_, e), f in best.items() if abs((e - end).days) <= tol_days]
    return min(cands)[1] if cands else None


def _first_current_instant(cf: dict, concepts: list[str], as_of: date) -> tuple[date, float] | None:
    found = [x for x in (latest_instant(facts_for(cf, c), as_of) for c in concepts) if x is not None]
    if not found:
        return None
    latest = max(e for e, _ in found)
    return next(x for x in found if x[0] >= latest - timedelta(days=7))


def total_debt(cf: dict, as_of: date) -> float:
    """LongTermDebt (incl. current maturities; else noncurrent + current) + short-term borrowings.

    If the company has a balance sheet but never tagged any debt concept, debt is taken as 0.
    """
    cands: list[tuple[date, float]] = []
    lt = latest_instant(facts_for(cf, LTD_TOTAL), as_of)
    if lt:
        cands.append(lt)
    for noncur, cur in LTD_PARTS:
        nc = latest_instant(facts_for(cf, noncur), as_of)
        if nc:
            cur_val = instant_at(facts_for(cf, cur), as_of, nc[0]) or 0.0
            cands.append((nc[0], nc[1] + cur_val))
        else:
            c = latest_instant(facts_for(cf, cur), as_of)
            if c:
                cands.append(c)
    long_term: tuple[date, float] | None = None
    if cands:  # the most recent balance-sheet date wins; ties keep priority order
        latest = max(e for e, _ in cands)
        long_term = next(x for x in cands if x[0] == latest)
    ref_end = long_term[0] if long_term else None

    short = 0.0
    found_short = False
    for concept in SHORT_TERM_DEBT_CONCEPTS:
        st = latest_instant(facts_for(cf, concept), as_of)
        if st and (ref_end is None or st[0] >= ref_end - timedelta(days=100)):
            short = st[1]
            found_short = True
            break
    if long_term is None and not found_short:
        has_bs = any(_first_current_instant(cf, cs, as_of) for cs in (EQUITY_CONCEPTS, CASH_CONCEPTS))
        return 0.0 if has_bs else math.nan
    return (long_term[1] if long_term else 0.0) + short


def shares_outstanding(cf: dict, as_of: date) -> float:
    """dei cover-page shares from the latest filing on/before ``as_of`` (all classes summed);
    else the latest quarterly weighted-average diluted share count."""
    dei = [f for f in facts_for(cf, DEI_SHARES, "dei", "shares") if f.filed <= as_of and f.start is None]
    if dei:
        last = max((f.filed, f.accn) for f in dei)
        in_filing = [f for f in dei if (f.filed, f.accn) == last]
        end = max(f.end for f in in_filing)
        vals = {f.val for f in in_filing if f.end == end}
        total = float(sum(vals))
        if total > 0:
            return total
    dil = facts_for(cf, DILUTED_SHARES, "us-gaap", "shares")
    best, _ = _latest_by_period(dil, as_of, durations=True)
    if best:
        quarterly = {k: f for k, f in best.items() if Q_MIN <= (k[1] - k[0]).days <= Q_MAX}
        pool = quarterly or best
        (_, _), f = max(pool.items(), key=lambda kv: kv[0][1])
        return float(f.val)
    return math.nan


def _ts(d: date | None) -> pd.Timestamp:
    return pd.Timestamp(d) if d is not None else pd.NaT


def _nan_row() -> dict[str, Any]:
    row: dict[str, Any] = {c: math.nan for c in F.FUNDAMENTAL_COLUMNS}
    row[F.PERIOD_END] = pd.NaT
    row[F.REPORT_DATE] = pd.NaT
    return row


def fundamentals_from_companyfacts(cf: dict, as_of: date) -> dict[str, Any]:
    """One ``fields.FUNDAMENTAL_COLUMNS`` row, point-in-time as of ``as_of`` (missing -> NaN/NaT)."""
    row = _nan_row()
    rev = merged_quarters(cf, REVENUE_CONCEPTS, as_of)
    ni = merged_quarters(cf, NET_INCOME_CONCEPTS, as_of)
    oi = merged_quarters(cf, OPERATING_INCOME_CONCEPTS, as_of)
    cfo = merged_quarters(cf, CFO_CONCEPTS, as_of)

    anchor_series = [s for s in (rev, ni, oi, cfo) if s]
    anchor: date | None = max(max(s) for s in anchor_series) if anchor_series else None
    if anchor is not None:
        prior = anchor - timedelta(days=365)
        firsts = [q.first_filed for q in (_find(s, anchor, 3) for s in anchor_series) if q is not None]
        row[F.PERIOD_END] = _ts(anchor)
        row[F.REPORT_DATE] = _ts(min(firsts)) if firsts else pd.NaT

        row[F.REVENUE_TTM] = ttm(rev, anchor)
        row[F.REVENUE_TTM_PRIOR_YEAR] = ttm(rev, prior)
        row[F.REVENUE_LAST_Q] = quarter_value(rev, anchor, 3)
        row[F.REVENUE_LAST_Q_PRIOR_YEAR] = quarter_value(rev, prior)

        gp = merged_quarters(cf, GROSS_PROFIT_CONCEPTS, as_of)
        cost = merged_quarters(cf, COST_OF_REVENUE_CONCEPTS, as_of)
        for col, a, rev_ttm in ((F.GROSS_PROFIT_TTM, anchor, row[F.REVENUE_TTM]),
                                (F.GROSS_PROFIT_TTM_PRIOR_YEAR, prior, row[F.REVENUE_TTM_PRIOR_YEAR])):
            g = ttm(gp, a)
            if not math.isfinite(g):
                g = rev_ttm - ttm(cost, a)  # NaN propagates if either is missing
            row[col] = g

        row[F.OPERATING_INCOME_TTM] = ttm(oi, anchor)
        row[F.OPERATING_INCOME_TTM_PRIOR_YEAR] = ttm(oi, prior)
        da = ttm(merged_quarters(cf, DA_CONCEPTS, as_of), anchor)
        row[F.EBITDA_TTM] = row[F.OPERATING_INCOME_TTM] + da  # NaN if D&A missing
        row[F.NET_INCOME_TTM] = ttm(ni, anchor)
        row[F.CFO_TTM] = ttm(cfo, anchor)
        capex = ttm(merged_quarters(cf, CAPEX_CONCEPTS, as_of), anchor)
        row[F.CAPEX_TTM] = abs(capex) if math.isfinite(capex) else math.nan
        row[F.FCF_TTM] = row[F.CFO_TTM] - row[F.CAPEX_TTM]
        interest = ttm(merged_quarters(cf, INTEREST_CONCEPTS, as_of), anchor)
        row[F.INTEREST_EXPENSE_TTM] = abs(interest) if math.isfinite(interest) else math.nan

    row[F.TOTAL_DEBT] = total_debt(cf, as_of)
    cash = _first_current_instant(cf, CASH_CONCEPTS, as_of)
    row[F.CASH] = cash[1] if cash else math.nan
    eq = _first_current_instant(cf, EQUITY_CONCEPTS, as_of)
    row[F.TOTAL_EQUITY] = eq[1] if eq else math.nan
    row[F.SHARES_OUTSTANDING] = shares_outstanding(cf, as_of)
    for k, v in row.items():
        if k not in (F.PERIOD_END, F.REPORT_DATE) and v is not None:
            row[k] = float(v)
    return row


# =============================================================================================
# HTML -> text, exhibit selection, MD&A extraction
# =============================================================================================

_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section",
               "article", "blockquote", "pre", "hr", "center", "dd", "dt", "dl", "header", "footer", "caption"}
_SKIP_TAGS = {"script", "style", "head", "title", "noscript", "template", "ix:header", "xml", "svg"}
_VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source", "wbr"}
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
_NUMERIC_CELL_RE = re.compile(r"^[\s$€£¥()%,.\-–—−+*\d]*$")


class _Table:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def start_row(self) -> None:
        self.end_cell()
        self.rows.append([])

    def start_cell(self) -> None:
        self.end_cell()
        if not self.rows:
            self.rows.append([])
        self.cell = []

    def end_cell(self) -> None:
        if self.cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.cell = None

    def add(self, text: str) -> None:
        if self.cell is None:
            self.start_cell()
        assert self.cell is not None
        self.cell.append(text)

    def render(self) -> str:
        """Plain text for prose/layout tables; empty for tables of numbers (financial statements)."""
        self.end_cell()
        cells = [c for r in self.rows for c in r if c]
        if not cells:
            return ""
        numeric = [c for c in cells if _NUMERIC_CELL_RE.match(c)]
        text_cells = [c for c in cells if not _NUMERIC_CELL_RE.match(c)]
        mean_text = sum(len(c) for c in text_cells) / len(text_cells) if text_cells else 0.0
        if len(numeric) >= len(text_cells) and mean_text < 60:
            return ""
        lines = [" ".join(c for c in r if c) for r in self.rows]
        return "\n" + "\n".join(line for line in lines if line) + "\n"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip: list[list[Any]] = []  # [tag, depth]
        self.tables: list[_Table] = []

    def _emit(self, text: str) -> None:
        if self.tables:
            self.tables[-1].add(text)
        else:
            self.out.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.skip:
            if tag == self.skip[-1][0] and tag not in _VOID_TAGS:
                self.skip[-1][1] += 1
            return
        style = next((v or "" for k, v in attrs if k.lower() == "style"), "")
        if tag in _SKIP_TAGS or (tag not in _VOID_TAGS and _HIDDEN_RE.search(style)):
            self.skip.append([tag, 1])
            return
        if tag == "table":
            self.tables.append(_Table())
            return
        if self.tables:
            t = self.tables[-1]
            if tag == "tr":
                t.start_row()
                return
            if tag in ("td", "th"):
                t.start_cell()
                return
            if tag in _BLOCK_TAGS:
                t.add(" ")
            return
        if tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.skip:
            return
        if tag in _BLOCK_TAGS:
            self._emit(" " if self.tables else "\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip:
            if tag == self.skip[-1][0]:
                self.skip[-1][1] -= 1
                if self.skip[-1][1] <= 0:
                    self.skip.pop()
            return
        if tag == "table" and self.tables:
            rendered = self.tables.pop().render()
            if self.tables:
                self.tables[-1].add(" " + rendered.replace("\n", " ") + " ")
            else:
                self.out.append(rendered)
            return
        if self.tables:
            if tag in ("td", "th"):
                self.tables[-1].end_cell()
            return
        if tag in _BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip or not data:
            return
        self._emit(re.sub(r"\s+", " ", data.replace("\xa0", " ")))

    def text(self) -> str:
        while self.tables:  # unclosed tables in sloppy HTML
            self.out.append(self.tables.pop(0).render())
        return "".join(self.out)


_PAGE_NOISE_RE = re.compile(r"^(?:page\s*)?\d{1,3}$|^table of contents$|^\(?back to (?:top|contents)\)?$", re.I)


_HTML_HINT_RE = re.compile(r"<(?:html|body|p|div|br|table|font|span|td)\b", re.I)


def html_to_text(html: str) -> str:
    """Readable plain text from filing HTML: paragraphs separated by blank lines, scripts/styles,
    hidden inline-XBRL headers and tables of numbers dropped, prose tables kept as lines.
    Plain-text documents (.txt exhibits) keep their paragraph breaks."""
    if not html:
        return ""
    if not _HTML_HINT_RE.search(html[:20000]):
        paras = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", html.replace("\r", "")))
        return "\n\n".join(p for p in paras if p and not _PAGE_NOISE_RE.match(p))
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - html.parser is lenient; never fail the pipeline on bad markup
        pass
    lines = []
    for line in parser.text().replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v\xa0]+", " ", line).strip()
        if not line or _PAGE_NOISE_RE.match(line):
            continue
        lines.append(line)
    return "\n\n".join(lines)


_EX99_RE = re.compile(r"(?:ex|exh|exhibit)[-_. ]*99(?:[-_. ]*0?(\d{1,2}))?", re.I)
_DOC_EXT = (".htm", ".html", ".txt")


def pick_press_release(items: list[dict], primary_document: str = "") -> tuple[str, str] | None:
    """Choose the earnings press release from a filing index: (file name, exhibit label).

    Prefers exhibit 99.1 (``ex991``, ``ex99-1``, ``exhibit991`` ...), then a bare ex99, then other
    99.x; falls back to the largest non-primary HTML document.
    """
    scored: list[tuple[int, int, str, str]] = []
    others: list[tuple[int, str]] = []
    for it in items or []:
        name = str(it.get("name") or "")
        low = name.lower()
        if not low.endswith(_DOC_EXT) or "index" in low or re.match(r"^r\d+\.htm$", low):
            continue
        ext_rank = 0 if low.endswith((".htm", ".html")) else 1
        m = _EX99_RE.search(low)
        if m:
            sub = m.group(1)
            if sub is None:
                score, label = 1, "Ex.99"
            else:
                n = int(sub)
                score, label = (0 if n == 1 else 2 + n), f"Ex.99.{n}"
            scored.append((score, ext_rank, name, label))
        elif name != primary_document:
            try:
                size = int(str(it.get("size") or "0").strip() or 0)
            except ValueError:
                size = 0
            others.append((size, name))
    if scored:
        best = min(scored)
        return best[2], best[3]
    if others:
        return max(others)[1], "exhibit"
    return None


_SEP = r"\s*[\.:\-–—]?\s*"
_MGMT = r"management\W{0,3}s?\s+discussion\s+and\s+analysis"
_MDNA_PATTERNS = {
    "10-Q": (re.compile(rf"item\s*2{_SEP}{_MGMT}", re.I),
             re.compile(rf"item\s*3{_SEP}quantitative\s+and\s+qualitative|item\s*4{_SEP}controls\s+and\s+procedures|"
                        rf"part\s+ii\W{{0,3}}\s*other\s+information", re.I)),
    "10-K": (re.compile(rf"item\s*7{_SEP}{_MGMT}", re.I),
             re.compile(rf"item\s*7a{_SEP}quantitative\s+and\s+qualitative|item\s*8{_SEP}financial\s+statements", re.I)),
}


def extract_mdna(text: str, form: str, max_chars: int = MAX_DOC_CHARS) -> str | None:
    """MD&A section (10-Q Item 2 / 10-K Item 7) from a filing's plain text.

    Heuristic: among all "Item 2/7 ... Management's Discussion and Analysis" headings take the one that
    starts the longest section before the next Item 3/4 (10-Q) or Item 7A/8 (10-K) heading - the table
    of contents and cross-references produce short spans. Capped at ``max_chars`` on a paragraph
    boundary. None if no plausible section is found.
    """
    key = "10-K" if form.upper().startswith("10-K") else "10-Q"
    start_re, end_re = _MDNA_PATTERNS[key]
    best: tuple[int, int] | None = None
    for m in start_re.finditer(text):
        end_m = end_re.search(text, m.end())
        end = end_m.start() if end_m else min(len(text), m.start() + max_chars * 2)
        if best is None or end - m.start() > best[1] - best[0]:
            best = (m.start(), end)
    if best is None or best[1] - best[0] < 500:
        return None
    section = text[best[0]: best[1]].strip()
    if len(section) > max_chars:
        cut = section.rfind("\n\n", 0, max_chars)
        section = section[: cut if cut > max_chars // 2 else max_chars].rstrip()
    return section


def _fye(fye_mmdd: str | None) -> tuple[int, int]:
    """(month, day) of the fiscal year end from SEC ``fiscalYearEnd`` ("MMDD"), default Dec 31.

    52/53-week years that end in the first days of a month (e.g. "0103", "0201") are treated as
    ending on the last day of the previous month, which is how those companies' quarters line up.
    """
    try:
        m, d = int(str(fye_mmdd)[:2]), int(str(fye_mmdd)[2:4])
        if not (1 <= m <= 12 and 1 <= d <= 31):
            raise ValueError
    except (ValueError, TypeError):
        return 12, 31
    if d <= 10:
        m, d = (12 if m == 1 else m - 1), 31
    return m, d


def fiscal_label(period_end: date, fye_mmdd: str | None) -> str:
    """'Q2 FY2026' style label for the fiscal quarter ending ``period_end``.

    Derived from the fiscal year end on the SEC submission; the fiscal year is named after the
    calendar year in which it ends (companies with January year ends differ in their own naming -
    the label is a convenience, not data).
    """
    m, d = _fye(fye_mmdd)
    shifted = period_end - timedelta(days=10)  # tolerate 52/53-week quarter ends drifting into the next month
    q = ((shifted.month - m - 1) % 12) // 3 + 1
    fy = shifted.year + (1 if (shifted.month, shifted.day) > (m, d) else 0)
    return f"Q{q} FY{fy}"


def _approx_quarter_end_before(day: date, fye_mmdd: str | None, lag_days: int = 7) -> date:
    """Latest nominal fiscal quarter end at least ``lag_days`` before ``day`` (to label 8-K releases)."""
    m, d = _fye(fye_mmdd)
    limit = day - timedelta(days=lag_days)
    cands = []
    for y in (limit.year - 1, limit.year):
        for k in range(4):
            mm = (m - 1 + 3 * k) % 12 + 1
            cands.append(date(y, mm, min(d, calendar.monthrange(y, mm)[1])))
    return max(c for c in cands if c <= limit)


# =============================================================================================
# HTTP client
# =============================================================================================


class SecEdgarClient:
    """Throttled, retrying, caching SEC EDGAR client (sync httpx).

    ``user_agent`` defaults to ``$SEC_USER_AGENT``; it is checked on the first network request (cached
    responses are served without it) and a missing / malformed value raises ``ProviderUnavailable``
    with setup instructions.
    """

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        cache: DiskCache | None = None,
        transport: httpx.BaseTransport | None = None,
        max_requests_per_second: float = 8.0,
        max_retries: int = 4,
        backoff_s: float = 0.5,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        memo_size: int = 512,
    ) -> None:
        self._user_agent = user_agent
        self.cache = cache if cache is not None else DiskCache()
        self._transport = transport
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._sleep = sleep
        self.limiter = RateLimiter(min(max_requests_per_second, 10.0), sleep=sleep)
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        self._map_lock = threading.Lock()
        self._ticker_map: dict[str, CompanyRef] | None = None
        self._memo: OrderedDict[str, Any] = OrderedDict()
        self._memo_lock = threading.Lock()
        self._memo_size = memo_size
        self.requests_made = 0

    # ------------------------------------------------------------------ setup
    @property
    def user_agent(self) -> str | None:
        ua = self._user_agent if self._user_agent is not None else os.environ.get(SEC_USER_AGENT_ENV)
        return ua.strip() if ua and ua.strip() else None

    def check_user_agent(self) -> str:
        ua = self.user_agent
        if not ua:
            raise ProviderUnavailable(user_agent_help())
        if "@" not in ua or len(ua.split()) < 2:
            raise ProviderUnavailable(user_agent_help(
                f"SEC_USER_AGENT={ua!r} does not look like 'Name email@domain'."))
        return ua

    def _http(self) -> httpx.Client:
        ua = self.check_user_agent()
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
                    timeout=self.timeout_s, follow_redirects=True, transport=self._transport,
                )
            return self._client

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    # ------------------------------------------------------------------ raw HTTP
    def _request(self, url: str) -> httpx.Response:
        client = self._http()
        last_err: str = ""
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = client.get(url)
                self.requests_made += 1
            except httpx.TransportError as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    self._sleep(min(30.0, self.backoff_s * 2 ** attempt))
                    continue
                raise ProviderError(f"SEC EDGAR request failed ({last_err}): {url}") from e
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                raise SecNotFound(f"SEC EDGAR: not found: {url}")
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                delay = self.backoff_s * 2 ** attempt
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        delay = max(delay, float(ra))
                    except ValueError:
                        pass
                self._sleep(min(60.0, delay))
                last_err = f"HTTP {resp.status_code}"
                continue
            if resp.status_code == 403:
                raise ProviderError(
                    "SEC EDGAR refused the request (HTTP 403). The SEC blocks clients without a valid "
                    "'Name email@domain' User-Agent and clients exceeding 10 requests/second; check "
                    f"SEC_USER_AGENT (currently {self.user_agent!r}) and try again later: {url}")
            raise ProviderError(f"SEC EDGAR HTTP {resp.status_code}{' after retries' if attempt else ''}: {url}")
        raise ProviderError(f"SEC EDGAR request failed after {self.max_retries + 1} attempts ({last_err}): {url}")

    def _memo_get(self, key: str) -> Any:
        with self._memo_lock:
            hit = self._memo.get(key)
            if hit is None:
                return None
            expires_at, value = hit
            if expires_at is not None and time.monotonic() >= expires_at:
                del self._memo[key]
                return None
            self._memo.move_to_end(key)
            return value

    def _memo_put(self, key: str, value: Any, ttl_s: float | None) -> None:
        with self._memo_lock:
            self._memo[key] = (None if ttl_s is None else time.monotonic() + ttl_s, value)
            self._memo.move_to_end(key)
            while len(self._memo) > self._memo_size:
                self._memo.popitem(last=False)

    def get_json(self, url: str, ttl_s: float | None, *, reducer: Callable[[dict], dict] | None = None) -> Any:
        """GET a JSON resource through memory memo -> disk cache -> network (with ``reducer`` applied
        before caching, so only the parts this client reads are stored)."""
        key = f"sec:json:{url}"
        hit = self._memo_get(key)
        if hit is not None:
            return hit
        data = self.cache.get_json(key)
        if data is None:
            resp = self._request(url)
            try:
                data = resp.json()
            except ValueError as e:
                raise ProviderError(f"SEC EDGAR returned invalid JSON: {url}") from e
            if reducer is not None and isinstance(data, dict):
                data = reducer(data)
            self.cache.put_json(key, data, ttl_s)
        self._memo_put(key, data, ttl_s)
        return data

    def document_text(self, url: str) -> str:
        """Plain text of a filed document (HTML converted), cached forever."""
        key = f"sec:text:{url}"
        cached = self.cache.get_text(key)
        if cached is not None:
            return cached
        text = html_to_text(self._request(url).text)
        self.cache.put_text(key, text, TTL_DOCUMENT)
        return text

    # ------------------------------------------------------------------ reference
    def ticker_map(self) -> dict[str, CompanyRef]:
        with self._map_lock:
            if self._ticker_map is not None:
                return self._ticker_map
            data = self.get_json(TICKER_MAP_URL, TTL_TICKER_MAP)
            fields_ = data.get("fields") or ["cik", "name", "ticker", "exchange"]
            idx = {f: i for i, f in enumerate(fields_)}
            out: dict[str, CompanyRef] = {}
            for rec in data.get("data") or []:
                try:
                    tkr = str(rec[idx["ticker"]]).upper()
                    ref = CompanyRef(int(rec[idx["cik"]]), str(rec[idx["name"]]), tkr,
                                     normalize_exchange(rec[idx["exchange"]]) if "exchange" in idx else None)
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                out.setdefault(tkr, ref)  # the file lists the primary security first
            self._ticker_map = out
            return out

    def lookup(self, ticker: str) -> CompanyRef | None:
        m = self.ticker_map()
        t = ticker.strip().upper()
        for cand in (t, t.replace(".", "-"), t.replace("/", "-"), t.replace("-", ".")):
            if cand in m:
                return m[cand]
        return None

    def _require(self, ticker: str) -> CompanyRef:
        ref = self.lookup(ticker)
        if ref is None:
            raise SecNotFound(f"{ticker}: not in the SEC ticker map (foreign listing, fund, or delisted?)")
        return ref

    # ------------------------------------------------------------------ datasets
    def company_facts(self, cik: int) -> dict:
        return self.get_json(COMPANYFACTS_URL.format(cik=int(cik)), TTL_COMPANYFACTS, reducer=reduce_companyfacts)

    def submissions(self, cik: int) -> dict:
        return self.get_json(SUBMISSIONS_URL.format(cik=int(cik)), TTL_SUBMISSIONS, reducer=_reduce_submissions)

    def filings(self, cik: int, *, start: date | None = None, end: date | None = None,
                forms: set[str] | None = None) -> list[Filing]:
        """Filings in [start, end] (by filing date), newest first."""
        sub = self.submissions(cik)
        recent = parse_filings(cik, (sub.get("filings") or {}).get("recent") or {})
        out = list(recent)
        earliest = min((f.filing_date for f in recent), default=None)
        if start is not None and (earliest is None or earliest > start):
            for page in (sub.get("filings") or {}).get("files") or []:
                to = _parse_date(page.get("filingTo"))
                name = page.get("name")
                if not name or (to is not None and to < start):
                    continue
                try:
                    block = self.get_json(SUBMISSIONS_PAGE_URL.format(name=name), TTL_SUBMISSIONS_PAGE,
                                          reducer=_reduce_submission_block)
                except ProviderError:
                    continue
                out.extend(parse_filings(cik, block))
        seen: set[str] = set()
        res = []
        for f in sorted(out, key=lambda f: (f.filing_date, f.accession), reverse=True):
            if f.accession in seen:
                continue
            seen.add(f.accession)
            if start is not None and f.filing_date < start:
                continue
            if end is not None and f.filing_date > end:
                continue
            if forms is not None and f.form not in forms:
                continue
            res.append(f)
        return res

    def filing_index(self, filing: Filing) -> list[dict]:
        data = self.get_json(filing.index_url, TTL_DOCUMENT)
        return list(((data or {}).get("directory") or {}).get("item") or [])

    # ------------------------------------------------------------------ high level
    def fundamentals(self, ticker: str, as_of: date) -> dict[str, Any]:
        ref = self._require(ticker)
        return fundamentals_from_companyfacts(self.company_facts(ref.cik), as_of)

    def shares_outstanding(self, ticker: str, as_of: date) -> float:
        ref = self._require(ticker)
        return shares_outstanding(self.company_facts(ref.cik), as_of)

    def last_earnings_release_date(self, ticker: str, as_of: date) -> date | None:
        """Filing date of the latest 8-K item 2.02 filed on/before ``as_of`` (point-in-time)."""
        ref = self._require(ticker)
        for f in self.filings(ref.cik, start=as_of - timedelta(days=200), end=as_of, forms={"8-K", "8-K/A"}):
            if f.is_earnings_release:
                return f.filing_date
        return None

    def documents(self, ticker: str, kinds: set[DocumentKind], start: date, end: date, limit: int = 10,
                  warn: Callable[[str], None] | None = None) -> list[Document]:
        """8-K earnings releases (NEWS) and 10-Q/10-K MD&A (FILING) filed in [start, end], newest first."""
        warn = warn or (lambda _msg: None)
        want_news = DocumentKind.NEWS in kinds
        want_filing = DocumentKind.FILING in kinds
        if not (want_news or want_filing) or limit <= 0:
            return []
        ref = self._require(ticker)
        sub = self.submissions(ref.cik)
        company = str(sub.get("name") or ref.name)
        fye = sub.get("fiscalYearEnd")
        forms = set()
        if want_news:
            forms |= {"8-K", "8-K/A"}
        if want_filing:
            forms |= {"10-Q", "10-K", "10-KT"}
        docs: list[Document] = []
        for f in self.filings(ref.cik, start=start, end=end, forms=forms):
            if len(docs) >= limit:
                break
            try:
                if f.form.startswith("8-K"):
                    if not f.is_earnings_release:
                        continue
                    doc = self._earnings_release_doc(ticker, company, fye, f)
                else:
                    doc = self._mdna_doc(ticker, company, fye, f)
            except ProviderUnavailable:
                raise
            except ProviderError as e:
                warn(f"{ticker}: could not fetch {f.form} {f.accession}: {e}")
                continue
            if doc is None:
                warn(f"{ticker}: no usable text in {f.form} {f.accession} ({f.filing_date})")
                continue
            docs.append(doc)
        return docs

    def _earnings_release_doc(self, ticker: str, company: str, fye: str | None, f: Filing) -> Document | None:
        picked = pick_press_release(self.filing_index(f), f.primary_document)
        if picked is None:
            return None
        name, label = picked
        url = f.base_url + name
        text = self.document_text(url)
        if len(text) < 200:
            return None
        if len(text) > MAX_DOC_CHARS:
            cut = text.rfind("\n\n", 0, MAX_DOC_CHARS)
            text = text[: cut if cut > MAX_DOC_CHARS // 2 else MAX_DOC_CHARS]
        qend = _approx_quarter_end_before(f.report_date or f.filing_date, fye)
        period = fiscal_label(qend, fye)
        return Document(
            doc_id=f"SEC-{f.accession}-EX99",
            ticker=ticker,
            kind=DocumentKind.NEWS,
            title=f"{company} {period} earnings release (8-K {label})",
            published_at=datetime.combine(f.filing_date, dtime()),
            source=SOURCE,
            url=url,
            text=text,
            metadata={
                "source_type": "earnings_release_8k", "form": f.form, "accession": f.accession, "cik": str(f.cik),
                "items": ",".join(f.items), "exhibit": name, "fiscal_period": period,
                "fiscal_period_note": "derived from the fiscal year end; FY named by the calendar year it ends in",
            },
        )

    def _mdna_doc(self, ticker: str, company: str, fye: str | None, f: Filing) -> Document | None:
        if not f.primary_document:
            return None
        url = f.primary_url
        text = self.document_text(url)
        section = extract_mdna(text, f.form)
        if section is None:
            return None
        is_10k = f.form.upper().startswith("10-K")
        item = "Item 7" if is_10k else "Item 2"
        pe = f.report_date
        if pe is not None:
            lab = fiscal_label(pe, fye)
            period = lab.replace("Q4 ", "") if is_10k else lab
        else:
            period = ""
        title = f"{company} {period + ' ' if period else ''}{f.form} MD&A ({item})"
        return Document(
            doc_id=f"SEC-{f.accession}-MDA",
            ticker=ticker,
            kind=DocumentKind.FILING,
            title=title,
            published_at=datetime.combine(f.filing_date, dtime()),
            source=SOURCE,
            url=url,
            text=section,
            metadata={
                "source_type": "10k_mdna" if is_10k else "10q_mdna", "form": f.form, "accession": f.accession,
                "cik": str(f.cik), "section": item, "period_of_report": pe.isoformat() if pe else "",
                "fiscal_period": period,
            },
        )
