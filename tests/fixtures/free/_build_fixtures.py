"""Regenerates the SEC JSON fixtures for tests/test_free_provider.py (run: python _build_fixtures.py).

Fictional filer "Acme Widgets Inc." (CIK 1234567, ticker ACME), calendar fiscal year. Values are in
USD millions here and scaled to absolute USD in the JSON. The hand-computed expectations in the tests
depend on these numbers - change both together.

Design of the data:
* income-statement concepts: 10-Qs report the 3-month quarter and the year-to-date (YTD) figure for the
  current and prior year; 10-Ks report the fiscal year (current + prior) -> Q4 must be derived (FY - 9M).
* cash-flow concepts (CFO, capex, D&A, interest): YTD only (3M / 6M / 9M / FY) -> must be differenced.
* Q1 2025 revenue (125) is restated to 127 as a comparative in the Q1 2026 10-Q filed 2026-04-30.
* the Q2 2026 10-Q is filed 2026-08-04 (anything "as of" earlier must ignore it).
* an obsolete revenue tag (SalesRevenueNet, 2016-2017) must not shadow the current one.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
CIK = 1234567
M = 1_000_000

# accession, form, filed, fiscal year, fiscal period, period end
FILINGS = {
    "Q1-24": ("0001234567-24-000020", "10-Q", "2024-05-02", 2024, "Q1", "2024-03-31"),
    "Q2-24": ("0001234567-24-000035", "10-Q", "2024-08-01", 2024, "Q2", "2024-06-30"),
    "Q3-24": ("0001234567-24-000050", "10-Q", "2024-10-31", 2024, "Q3", "2024-09-30"),
    "K-24": ("0001234567-25-000008", "10-K", "2025-02-20", 2024, "FY", "2024-12-31"),
    "Q1-25": ("0001234567-25-000021", "10-Q", "2025-05-01", 2025, "Q1", "2025-03-31"),
    "Q2-25": ("0001234567-25-000036", "10-Q", "2025-07-31", 2025, "Q2", "2025-06-30"),
    "Q3-25": ("0001234567-25-000051", "10-Q", "2025-10-30", 2025, "Q3", "2025-09-30"),
    "K-25": ("0001234567-26-000007", "10-K", "2026-02-19", 2025, "FY", "2025-12-31"),
    "Q1-26": ("0001234567-26-000020", "10-Q", "2026-04-30", 2026, "Q1", "2026-03-31"),
    "Q2-26": ("0001234567-26-000034", "10-Q", "2026-08-04", 2026, "Q2", "2026-06-30"),
}

QEND = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
QSTART = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}

# quarterly values, USD mn
IS = {  # income statement: 3M + YTD in 10-Qs, FY in 10-Ks
    "RevenueFromContractWithCustomerExcludingAssessedTax": {2023: [85, 90, 95, 100], 2024: [100, 110, 120, 130],
                                                            2025: [125, 135, 145, 155], 2026: [150, 165]},
    "CostOfRevenue": {2023: [51, 54, 57, 60], 2024: [60, 66, 72, 78], 2025: [75, 81, 87, 93], 2026: [90, 99]},
    "OperatingIncomeLoss": {2023: [12, 13, 14, 15], 2024: [15, 16, 17, 18], 2025: [20, 22, 24, 26], 2026: [25, 28]},
    "NetIncomeLoss": {2023: [8, 9, 9, 10], 2024: [10, 11, 12, 13], 2025: [14, 15, 16, 17], 2026: [18, 19]},
}
CF = {  # cash-flow style: YTD only
    "NetCashProvidedByUsedInOperatingActivities": {2023: [20, 22, 24, 26], 2024: [25, 30, 35, 40],
                                                   2025: [30, 40, 45, 50], 2026: [35, 45]},
    "PaymentsToAcquirePropertyPlantAndEquipment": {2023: [8, 8, 9, 10], 2024: [9, 10, 10, 13],
                                                   2025: [10, 12, 11, 17], 2026: [12, 13]},
    "DepreciationDepletionAndAmortization": {2023: [3, 3, 3, 3], 2024: [4, 4, 4, 4], 2025: [5, 5, 5, 5], 2026: [6, 6]},
    "InterestExpenseNonoperating": {2023: [1, 1, 1, 1], 2024: [1, 1, 1, 1], 2025: [2, 2, 2, 2], 2026: [2, 2]},
}
RESTATED_Q1_2025_REVENUE = 127  # comparative in the Q1-26 10-Q

BS = {  # balance-sheet instants by period end
    "LongTermDebt": {"2023-12-31": 320, "2024-03-31": 318, "2024-06-30": 316, "2024-09-30": 314, "2024-12-31": 310,
                     "2025-03-31": 308, "2025-06-30": 306, "2025-09-30": 304, "2025-12-31": 300,
                     "2026-03-31": 290, "2026-06-30": 280},
    "CashAndCashEquivalentsAtCarryingValue": {"2023-12-31": 100, "2024-03-31": 105, "2024-06-30": 110,
                                              "2024-09-30": 112, "2024-12-31": 115, "2025-03-31": 118,
                                              "2025-06-30": 120, "2025-09-30": 125, "2025-12-31": 130,
                                              "2026-03-31": 140, "2026-06-30": 150},
    "StockholdersEquity": {"2023-12-31": 400, "2024-03-31": 405, "2024-06-30": 410, "2024-09-30": 415,
                           "2024-12-31": 420, "2025-03-31": 430, "2025-06-30": 440, "2025-09-30": 450,
                           "2025-12-31": 460, "2026-03-31": 480, "2026-06-30": 500},
    "ShortTermBorrowings": {"2026-06-30": 20},
    "Assets": {"2023-12-31": 1000, "2024-03-31": 1010, "2024-06-30": 1020, "2024-09-30": 1035, "2024-12-31": 1050,
               "2025-03-31": 1080, "2025-06-30": 1100, "2025-09-30": 1120, "2025-12-31": 1150,
               "2026-03-31": 1180, "2026-06-30": 1210},
}
DEI_SHARES = {  # filing -> (cover date, shares)
    "Q1-24": ("2024-04-26", 51_000_000), "Q2-24": ("2024-07-26", 50_900_000), "Q3-24": ("2024-10-25", 50_800_000),
    "K-24": ("2025-02-14", 50_700_000), "Q1-25": ("2025-04-25", 50_600_000), "Q2-25": ("2025-07-25", 50_500_000),
    "Q3-25": ("2025-10-24", 50_400_000), "K-25": ("2026-02-13", 50_200_000), "Q1-26": ("2026-04-24", 50_000_000),
    "Q2-26": ("2026-07-31", 49_500_000),
}


def d(y: int, md: tuple[int, int]) -> str:
    return date(y, *md).isoformat()


def fact(start: str | None, end: str, val: float, key: str) -> dict:
    accn, form, filed, fy, fp, _ = FILINGS[key]
    out = {"end": end, "val": int(round(val * M)) if val == int(val) else val * M, "accn": accn, "fy": fy, "fp": fp,
           "form": form, "filed": filed}
    if start:
        out = {"start": start, **out}
    return out


def q_key(year: int, q: int) -> str | None:
    k = f"Q{q}-{str(year)[2:]}" if q < 4 else f"K-{str(year)[2:]}"
    return k if k in FILINGS else None


def build() -> dict:
    gaap: dict[str, dict] = {}
    for concept, years in IS.items():
        rows = []
        for y, qs in years.items():
            for q in range(1, len(qs) + 1):
                key = q_key(y, q)
                if key is None:
                    continue
                for yy in (y, y - 1):  # current period + prior-year comparative
                    vals = years.get(yy)
                    if not vals or len(vals) < q:
                        continue
                    v3 = vals[q - 1]
                    if concept.startswith("Revenue") and yy == 2025 and q == 1 and key == "Q1-26":
                        v3 = RESTATED_Q1_2025_REVENUE
                    if q < 4:
                        rows.append(fact(d(yy, QSTART[q]), d(yy, QEND[q]), v3, key))
                        if q > 1:
                            ytd = sum(vals[:q])
                            if concept.startswith("Revenue") and yy == 2025 and key == "Q2-26":
                                ytd += RESTATED_Q1_2025_REVENUE - vals[0]
                            rows.append(fact(d(yy, (1, 1)), d(yy, QEND[q]), ytd, key))
                    else:
                        rows.append(fact(d(yy, (1, 1)), d(yy, (12, 31)), sum(vals), key))
        gaap[concept] = {"label": concept, "description": "", "units": {"USD": rows}}
    for concept, years in CF.items():
        rows = []
        for y, qs in years.items():
            for q in range(1, len(qs) + 1):
                key = q_key(y, q)
                if key is None:
                    continue
                for yy in (y, y - 1):
                    vals = years.get(yy)
                    if not vals or len(vals) < q:
                        continue
                    end = d(yy, QEND[q])
                    rows.append(fact(d(yy, (1, 1)), end, sum(vals[:q]), key))
        gaap[concept] = {"label": concept, "description": "", "units": {"USD": rows}}
    for concept, points in BS.items():
        rows = []
        for key, (accn, form, filed, fy, fp, pe) in FILINGS.items():
            prior_fye = f"{int(pe[:4]) - 1}-12-31"
            for end in (pe, prior_fye):
                if end in points:
                    rows.append(fact(None, end, points[end], key))
        gaap[concept] = {"label": concept, "description": "", "units": {"USD": rows}}
    # Obsolete tag from long ago: must never shadow the current revenue concept.
    gaap["SalesRevenueNet"] = {"label": "SalesRevenueNet", "description": "", "units": {"USD": [
        {"start": "2016-01-01", "end": "2016-03-31", "val": 50 * M, "accn": "0001234567-16-000010", "fy": 2016,
         "fp": "Q1", "form": "10-Q", "filed": "2016-05-05"},
        {"start": "2016-04-01", "end": "2016-06-30", "val": 52 * M, "accn": "0001234567-16-000020", "fy": 2016,
         "fp": "Q2", "form": "10-Q", "filed": "2016-08-04"},
    ]}}
    # Diluted shares (fallback only) - present so the dei preference is exercised.
    gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = {"label": "", "description": "", "units": {"shares": [
        fact("2026-04-01", "2026-06-30", 51.0, "Q2-26") | {"val": 51_000_000},
    ]}}
    dei_rows = [{"end": end, "val": val, "accn": FILINGS[k][0], "fy": FILINGS[k][3], "fp": FILINGS[k][4],
                 "form": FILINGS[k][1], "filed": FILINGS[k][2]} for k, (end, val) in DEI_SHARES.items()]
    return {
        "cik": CIK,
        "entityName": "Acme Widgets Inc.",
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"label": "", "description": "", "units": {"shares": dei_rows}}},
            "us-gaap": gaap,
        },
    }


def submissions() -> dict:
    rows = [  # accession, filingDate, reportDate, form, primaryDocument, items
        ("0001234567-26-000034", "2026-08-04", "2026-06-30", "10-Q", "acme-20260630.htm", ""),
        ("0001234567-26-000031", "2026-07-28", "2026-07-28", "8-K", "acme-20260728.htm", "2.02,9.01"),
        ("0001234567-26-000027", "2026-06-10", "2026-06-09", "8-K", "acme-20260610.htm", "5.02"),
        ("0001111111-26-000001", "2026-05-15", "2026-05-13", "4", "xslF345X05/wk-form4.xml", ""),
        ("0001234567-26-000020", "2026-04-30", "2026-03-31", "10-Q", "acme-20260331.htm", ""),
        ("0001234567-26-000018", "2026-04-23", "2026-04-23", "8-K", "acme-20260423.htm", "2.02,9.01"),
        ("0001234567-26-000007", "2026-02-19", "2025-12-31", "10-K", "acme-20251231.htm", ""),
        ("0001234567-26-000004", "2026-02-05", "2026-02-05", "8-K", "acme-20260205.htm", "2.02,7.01,9.01"),
        ("0001234567-25-000051", "2025-10-30", "2025-09-30", "10-Q", "acme-20250930.htm", ""),
        ("0001234567-25-000049", "2025-10-23", "2025-10-23", "8-K", "acme-20251023.htm", "2.02,9.01"),
    ]
    cols = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument", "items"]
    recent = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    recent["acceptanceDateTime"] = [r[1] + "T16:05:00.000Z" for r in rows]
    recent["isXBRL"] = [1 if r[3] in ("10-Q", "10-K") else 0 for r in rows]
    return {
        "cik": str(CIK), "entityType": "operating", "sic": "3560", "name": "Acme Widgets Inc.",
        "tickers": ["ACME"], "exchanges": ["NYSE"], "fiscalYearEnd": "1231",
        "filings": {"recent": recent, "files": []},
    }


def ticker_map() -> dict:
    return {"fields": ["cik", "name", "ticker", "exchange"],
            "data": [[CIK, "Acme Widgets Inc.", "ACME", "NYSE"], [7654321, "Beta Robotics Corp.", "BETA", "Nasdaq"],
                     [1111111, "Gamma Index Trust", "GIDX", "NYSE"]]}


def index_8k() -> dict:
    items = [
        {"last-modified": "2026-07-28 16:05:12", "name": "0001234567-26-000031-index-headers.html", "type": "text.gif", "size": ""},
        {"last-modified": "2026-07-28 16:05:12", "name": "0001234567-26-000031-index.html", "type": "text.gif", "size": ""},
        {"last-modified": "2026-07-28 16:05:12", "name": "0001234567-26-000031.txt", "type": "text.gif", "size": "250311"},
        {"last-modified": "2026-07-28 16:05:12", "name": "R1.htm", "type": "text.gif", "size": "3311"},
        {"last-modified": "2026-07-28 16:05:12", "name": "acme-20260728.htm", "type": "text.gif", "size": "25120"},
        {"last-modified": "2026-07-28 16:05:12", "name": "acme-20260728_htm.xml", "type": "text.gif", "size": "4120"},
        {"last-modified": "2026-07-28 16:05:12", "name": "acme-ex992_cfocommentary.htm", "type": "text.gif", "size": "41000"},
        {"last-modified": "2026-07-28 16:05:12", "name": "acme-ex991_q22026.htm", "type": "text.gif", "size": "90210"},
        {"last-modified": "2026-07-28 16:05:12", "name": "acmelogo.jpg", "type": "image2.gif", "size": "5120"},
    ]
    return {"directory": {"item": items, "name": "/Archives/edgar/data/1234567/000123456726000031",
                          "parent-dir": "/Archives/edgar/data/1234567/"}}


if __name__ == "__main__":
    for name, obj in (("companyfacts_CIK0001234567.json", build()), ("submissions_CIK0001234567.json", submissions()),
                      ("company_tickers_exchange.json", ticker_map()), ("index_8k_000123456726000031.json", index_8k())):
        (HERE / name).write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
        print("wrote", name)
