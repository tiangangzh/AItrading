"""Tests for the deterministic synthetic market-data provider."""

from __future__ import annotations

import math
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aitrading.core import fields as F
from aitrading.core.models import DocumentKind
from aitrading.data.base import Capability, MarketDataProvider, PricePanel, ProviderError
from aitrading.data.synthetic import ARCHETYPE_TO_DISLOCATION, ARCHETYPES, BENCHMARK_SYMBOL, SyntheticProvider
from aitrading.screen.spec import UniverseSpec

END = date(2026, 9, 30)
ALL_KINDS = {DocumentKind.TRANSCRIPT, DocumentKind.NEWS, DocumentKind.FILING, DocumentKind.RESEARCH}
LABELS = [a for a in ARCHETYPES if a != "normal"]
DISLOCATION_ARCHES = ["transitory_shock", "value_trap", "guidance_reset", "sector_contagion"]


@pytest.fixture(scope="module")
def prov() -> SyntheticProvider:
    return SyntheticProvider()


@pytest.fixture(scope="module")
def raw(prov):
    """Raw end-of-window data for the whole universe (computed once)."""
    t = prov.tickers
    return dict(
        panel=prov.get_price_history(t, prov.start, END),
        uni=prov.get_universe(None, END),
        fund=prov.get_fundamentals(t, END),
        est=prov.get_estimates(t, END),
        si=prov.get_short_interest(t, END),
        opt=prov.get_options_summary(t, END),
        arch=pd.Series(prov.archetypes()),
    )


def by_arch(prov, arch: str) -> list[str]:
    return [t for t, a in prov.archetypes().items() if a == arch]


def canonical_features(raw) -> pd.DataFrame:
    """The canonical demo conditions computed directly from raw provider data with plain pandas."""
    p = raw["panel"]
    c, h, v = p.close, p.high, p.volume
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    dn = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    u, fu, si = raw["uni"], raw["fund"], raw["si"]
    return pd.DataFrame({
        "price": c.iloc[-1],
        "adv_mn": (c * v).iloc[-20:].mean() / 1e6,
        "country": u[F.COUNTRY],
        "stype": u[F.SECURITY_TYPE],
        "cap_bn": u[F.MARKET_CAP] / 1e9,
        "trend": (c.iloc[-50:].mean() / c.iloc[-200:].mean() - 1) * 100,
        "mom_12_1": (c.iloc[-22] / c.iloc[-253] - 1) * 100,
        "dd": (c.iloc[-1] / h.iloc[-252:].max() - 1) * 100,
        "max_vol_ratio": v.iloc[-20:].max() / v.iloc[-120:].mean(),
        "rsi": 100 - 100 / (1 + up / dn),
        "fcf_yield": fu[F.FCF_TTM] / u[F.MARKET_CAP] * 100,
        "rev_growth": (fu[F.REVENUE_TTM] / fu[F.REVENUE_TTM_PRIOR_YEAR] - 1) * 100,
        "si_pct": si[F.SHORT_INTEREST_SHARES] / si[F.FLOAT_SHARES] * 100,
        "arch": raw["arch"],
    })


def canonical_mask(df: pd.DataFrame) -> pd.Series:
    universe = (df.country == "US") & (df.stype == "common_stock") & (df.price >= 5) & (df.adv_mn >= 5)
    return (universe & df.cap_bn.between(2, 20) & (df.trend > 0) & (df.mom_12_1 > 0) & df.dd.between(-40, -15)
            & (df.max_vol_ratio >= 2) & (df.rsi < 40) & (df.fcf_yield > 4) & (df.rev_growth > 8) & (df.si_pct > 6))


# ---------------------------------------------------------------------------------------------
# Construction, protocol, determinism, performance
# ---------------------------------------------------------------------------------------------


def test_protocol_conformance(prov):
    assert isinstance(prov, MarketDataProvider)
    assert prov.name == "synthetic"
    assert prov.capabilities == {c for c in Capability if c is not Capability.SCREEN_PUSHDOWN}
    assert prov.boundary.provider == "synthetic"
    assert prov.boundary.allowed_document_kinds == {DocumentKind.TRANSCRIPT, DocumentKind.NEWS, DocumentKind.FILING}


def test_construction_performance():
    t0 = time.perf_counter()
    SyntheticProvider(n_tickers=500, seed=123)
    assert time.perf_counter() - t0 < 3.0


def test_same_seed_is_byte_identical(prov):
    other = SyntheticProvider()
    t = prov.tickers
    assert other.tickers == t
    assert other.archetypes() == prov.archetypes()
    a, b = prov.get_price_history(t, prov.start, END), other.get_price_history(t, prov.start, END)
    for k in F.PRICE_FIELDS:
        assert getattr(a, k).to_numpy().tobytes() == getattr(b, k).to_numpy().tobytes()
    for fn in ("get_fundamentals", "get_estimates", "get_short_interest", "get_options_summary"):
        pd.testing.assert_frame_equal(getattr(prov, fn)(t, END), getattr(other, fn)(t, END), check_exact=True)
    pd.testing.assert_frame_equal(prov.get_universe(None, END), other.get_universe(None, END), check_exact=True)
    pd.testing.assert_series_equal(prov.get_benchmark_history(prov.start, END), other.get_benchmark_history(prov.start, END))
    for tick in t[:5] + by_arch(prov, "transitory_shock")[:2]:
        d1 = prov.get_documents(tick, ALL_KINDS, prov.start, END, limit=100)
        d2 = other.get_documents(tick, ALL_KINDS, prov.start, END, limit=100)
        assert [d.model_dump_json() for d in d1] == [d.model_dump_json() for d in d2]


_DIGEST = """
import hashlib, pandas as pd
from aitrading.data.synthetic import SyntheticProvider
p = SyntheticProvider(n_tickers=120, seed=5)
h = hashlib.sha256()
panel = p.get_price_history(p.tickers, p.start, p.end)
for k in ("open", "high", "low", "close", "volume"):
    h.update(getattr(panel, k).to_numpy().tobytes())
for fn in ("get_fundamentals", "get_estimates", "get_short_interest", "get_options_summary"):
    h.update(pd.util.hash_pandas_object(getattr(p, fn)(p.tickers, p.end)).to_numpy().tobytes())
for t in p.tickers[::10] + [x for x, a in p.archetypes().items() if a != "normal"][:4]:
    for d in p.get_documents(t, None, p.start, p.end, limit=1000):
        h.update(d.model_dump_json().encode())
print(h.hexdigest())
"""


def test_byte_identical_across_processes():
    """No dependence on Python's per-process string hashing (set / dict ordering)."""
    import os
    import subprocess
    import sys

    out = []
    for hash_seed in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        res = subprocess.run([sys.executable, "-c", _DIGEST], capture_output=True, text=True, env=env, check=True)
        out.append(res.stdout.strip())
    assert len(out[0]) == 64 and out[0] == out[1]


def test_different_seed_differs(prov):
    other = SyntheticProvider(seed=8)
    assert other.tickers != prov.tickers
    c1 = prov.get_price_history(prov.tickers[:3], prov.start, END).close.to_numpy()
    c2 = other.get_price_history(other.tickers[:3], other.start, END).close.to_numpy()
    assert not np.array_equal(c1, c2, equal_nan=True)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        SyntheticProvider(n_tickers=0)
    with pytest.raises(ValueError):
        SyntheticProvider(start=date(2026, 1, 10), end=date(2026, 1, 1))
    with pytest.raises(ValueError):
        SyntheticProvider(start=date(2026, 1, 3), end=date(2026, 1, 4))  # a weekend: no sessions


def test_tiny_and_short_configurations_work():
    p = SyntheticProvider(n_tickers=1, start=date(2026, 9, 1), end=date(2026, 9, 30))
    assert len(p.tickers) == 1 and set(p.archetypes().values()) == {"normal"}  # too short to plant stories
    t = p.tickers
    panel = p.get_price_history(t, p.start, p.end)
    assert len(panel.close) == len(pd.bdate_range(p.start, p.end))
    assert p.get_fundamentals(t, p.end).shape == (1, len(F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS))
    assert p.get_estimates(t, p.end).shape == (1, len(F.ESTIMATE_COLUMNS))
    assert p.get_short_interest(t, p.end).shape == (1, len(F.SHORT_INTEREST_COLUMNS))
    assert p.get_options_summary(t, p.end).shape == (1, len(F.OPTIONS_COLUMNS))
    assert isinstance(p.get_documents(t[0], ALL_KINDS, p.start, p.end), list)
    small = SyntheticProvider(n_tickers=40, seed=3)
    assert len(set(small.tickers)) == 40


# ---------------------------------------------------------------------------------------------
# Reference data / universe
# ---------------------------------------------------------------------------------------------


def test_securities_reference_data(prov, raw):
    u = raw["uni"]
    t = prov.tickers
    assert len(t) == 500 and len(set(t)) == 500
    assert all(re.fullmatch(r"[A-Z]{3,5}", x) for x in t)
    assert not {"AAPL", "MSFT", "NVDA", "SPY", "AMZN", "GOOGL", "META"} & set(t)
    assert u[F.NAME].is_unique
    assert list(u.columns) == F.UNIVERSE_COLUMNS
    assert (u[F.VENDOR_ID] == u.index + " SYN").all()
    stocks = u[u[F.SECURITY_TYPE].isin(["common_stock", "adr", "reit"])]
    assert stocks[F.GICS_SECTOR].nunique() == 11
    assert set(u[F.EXCHANGE]) == {"NYSE", "NASDAQ"}
    assert {"common_stock", "adr", "reit", "etf", "preferred"} <= set(u[F.SECURITY_TYPE])
    assert 0.94 <= (u[F.COUNTRY] == "US").mean() <= 0.99
    assert (u.loc[u[F.SECURITY_TYPE] == "adr", F.COUNTRY] != "US").all()
    caps = stocks[F.MARKET_CAP]
    assert caps.min() > 0.08e9 and caps.max() < 400e9
    assert 2e9 < caps.median() < 12e9
    c = raw["panel"].close.iloc[-1]
    adv = (raw["panel"].close * raw["panel"].volume).iloc[-20:].mean()
    common = u.index[u[F.SECURITY_TYPE] == "common_stock"]
    assert (c[common] < 5).sum() >= 3  # penny stocks
    assert (adv[common] < 5e6).sum() >= 5  # illiquid names


def test_universe_filters_and_market_cap(prov, raw):
    full = raw["uni"]
    us = prov.get_universe(UniverseSpec(), END)
    assert (us[F.COUNTRY] == "US").all() and (us[F.SECURITY_TYPE] == "common_stock").all()
    assert len(us) == ((full[F.COUNTRY] == "US") & (full[F.SECURITY_TYPE] == "common_stock")).sum()
    # Penny / illiquid names are NOT removed by the provider (the screen engine applies price/liquidity filters).
    assert (raw["panel"].close.iloc[-1][us.index] < 5).any()
    both = prov.get_universe(UniverseSpec(security_types=["common_stock", "reit"], country="US"), END)
    assert set(both[F.SECURITY_TYPE]) == {"common_stock", "reit"}
    # market cap = close x shares outstanding (latest public report)
    d = date(2025, 6, 13)
    u = prov.get_universe(None, d)
    fu = prov.get_fundamentals(list(u.index), d)
    close = prov.get_price_history(list(u.index), d, d).close.iloc[-1]
    has = fu[F.SHARES_OUTSTANDING].notna()
    np.testing.assert_allclose(u.loc[has, F.MARKET_CAP], (close * fu[F.SHARES_OUTSTANDING])[has], rtol=1e-12)


def test_universe_point_in_time_listing(prov):
    full = prov.get_universe(None, END)
    early = prov.get_universe(None, date(2024, 3, 1))
    ipos = set(full.index) - set(early.index)
    assert ipos  # recent IPOs are not in the early universe
    close = prov.get_price_history(sorted(ipos), prov.start, END).close
    assert close.iloc[0].isna().all() and close.iloc[-1].notna().all()
    before = prov.get_universe(None, date(2023, 6, 1))
    assert before[F.MARKET_CAP].isna().all() and not set(before.index) & ipos


# ---------------------------------------------------------------------------------------------
# Prices / benchmark
# ---------------------------------------------------------------------------------------------


def test_ohlc_sanity(raw):
    p = raw["panel"]
    o, h, l, c, v = p.open.to_numpy(), p.high.to_numpy(), p.low.to_numpy(), p.close.to_numpy(), p.volume.to_numpy()
    ok = np.isfinite(c)
    assert ok.mean() > 0.98
    assert (np.isfinite(o) == ok).all() and (np.isfinite(v) == ok).all()
    assert (h[ok] >= np.maximum(o[ok], c[ok])).all()
    assert (l[ok] <= np.minimum(o[ok], c[ok])).all()
    assert (l[ok] > 0).all() and (v[ok] > 0).all()
    assert p.close.index.is_monotonic_increasing and p.close.index.is_unique
    r = np.log(p.close).diff().iloc[1:]
    assert r.abs().max().max() < 1.0  # no absurd jumps
    vol = r.std() * np.sqrt(252)
    assert 0.08 < vol.median() < 0.6


def test_price_history_window_and_unknowns(prov):
    t = prov.tickers[:4]
    p = prov.get_price_history(t + ["NOPE", t[0]], date(2025, 3, 3), date(2027, 1, 1))
    assert isinstance(p, PricePanel)
    assert p.tickers == t + ["NOPE"]  # de-duplicated, order kept, unknown kept as NaN
    assert p.close.index.min() >= pd.Timestamp(2025, 3, 3)
    assert p.close.index.max() == pd.Timestamp(END)  # nothing after the provider's end
    assert p.close["NOPE"].isna().all()
    empty = prov.get_price_history(t, date(2025, 3, 10), date(2025, 3, 1))
    assert empty.close.empty and empty.tickers == t
    assert prov.get_price_history([], prov.start, END).close.shape[1] == 0
    full = prov.get_price_history(t, prov.start, END).close
    assert full.index[0] == pd.Timestamp(prov.start) and len(full) == len(pd.bdate_range(prov.start, END))


def test_benchmark(prov, raw):
    b = prov.get_benchmark_history(prov.start, END)
    assert b.name == BENCHMARK_SYMBOL
    assert b.iloc[0] == 1000.0 and b.index[-1] == pd.Timestamp(END) and b.notna().all()
    assert prov.get_benchmark_history(prov.start, END, BENCHMARK_SYMBOL).equals(b)
    sub = prov.get_benchmark_history(date(2025, 1, 6), date(2025, 2, 28))
    assert sub.index.min() >= pd.Timestamp(2025, 1, 6) and sub.index.max() <= pd.Timestamp(2025, 2, 28)
    with pytest.raises(ProviderError):
        prov.get_benchmark_history(prov.start, END, "NOT-A-SYMBOL")
    # Cap-weighted: one day's index return equals the cap-weighted average of constituent returns.
    d0, d1 = date(2025, 6, 12), date(2025, 6, 13)
    u = prov.get_universe(None, d0)
    u = u[~u[F.SECURITY_TYPE].isin(["etf", "preferred"])]
    close = prov.get_price_history(list(u.index), d0, d1).close
    w = u[F.MARKET_CAP]
    r = close.iloc[1] / close.iloc[0] - 1
    expected = (w * r).sum() / w.sum()
    got = b[pd.Timestamp(d1)] / b[pd.Timestamp(d0)] - 1
    assert got == pytest.approx(expected, abs=2e-6)
    # Broad market index tracks the market: positively correlated with the median stock.
    med = np.log(raw["panel"].close).diff().median(axis=1)
    assert np.corrcoef(np.log(b).diff().iloc[1:], med.iloc[1:])[0, 1] > 0.7


# ---------------------------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------------------------


def test_fundamental_columns_units_and_identity(raw):
    fu = raw["fund"]
    assert list(fu.columns) == F.FUNDAMENTAL_COLUMNS + F.FUNDAMENTAL_OPTIONAL_COLUMNS  # optional columns appended
    assert str(fu[F.REPORT_DATE].dtype).startswith("datetime64") and str(fu[F.PERIOD_END].dtype).startswith("datetime64")
    ok = fu[F.REVENUE_TTM].notna()
    assert ok.sum() >= 480  # everything except ETFs / preferreds
    np.testing.assert_array_equal(fu.loc[ok, F.FCF_TTM].to_numpy(), (fu[F.CFO_TTM] - fu[F.CAPEX_TTM])[ok].to_numpy())
    assert (fu.loc[ok, F.REVENUE_TTM] > 1e6).all()  # absolute USD, not millions
    assert (fu.loc[ok, F.CAPEX_TTM] > 0).all() and (fu.loc[ok, F.INTEREST_EXPENSE_TTM] >= 0).all()
    assert (fu.loc[ok, F.TOTAL_DEBT] >= 0).all() and (fu.loc[ok, F.CASH] > 0).all()
    assert (fu.loc[ok, F.EBITDA_TTM] >= fu.loc[ok, F.OPERATING_INCOME_TTM]).all()
    assert (fu.loc[ok, F.REPORT_DATE] <= pd.Timestamp(END)).all()
    lag = (fu.loc[ok, F.REPORT_DATE] - fu.loc[ok, F.PERIOD_END]).dt.days
    assert lag.between(25, 45).all()
    etfs = raw["uni"].index[raw["uni"][F.SECURITY_TYPE].isin(["etf", "preferred"])]
    assert fu.loc[etfs].isna().all().all()


def test_total_assets_consistent_with_balance_sheet(prov, raw):
    fu = raw["fund"]
    ok = fu[F.REVENUE_TTM].notna()
    ta, ta_py = fu.loc[ok, F.TOTAL_ASSETS], fu.loc[ok, F.TOTAL_ASSETS_PRIOR_YEAR]
    assert ta.notna().all() and (ta > 1e6).all()  # absolute USD
    # assets = equity + debt + other liabilities (> 0), never below cash
    assert (ta > fu.loc[ok, F.TOTAL_EQUITY] + fu.loc[ok, F.TOTAL_DEBT]).all()
    assert (ta > fu.loc[ok, F.CASH]).all()
    assert ta_py.notna().mean() > 0.95 and (ta_py.dropna() > 0).all()
    growth = (ta / ta_py - 1).dropna()
    assert -0.6 < growth.median() < 0.6 and growth.between(-0.9, 3.0).all()
    # banks / insurers carry far more assets per dollar of revenue than industrials
    sector = raw["uni"][F.GICS_SECTOR].reindex(ta.index)
    turn = fu.loc[ok, F.REVENUE_TTM] / ta
    assert turn[sector == "Financials"].median() < 0.5 * turn[sector == "Industrials"].median()
    etfs = raw["uni"].index[raw["uni"][F.SECURITY_TYPE].isin(["etf", "preferred"])]
    assert fu.loc[etfs, F.FUNDAMENTAL_OPTIONAL_COLUMNS].isna().all().all()


def test_total_assets_prior_year_is_the_point_in_time_value_four_quarters_back(prov):
    for t in prov.tickers[:12]:
        snap = prov.get_fundamentals([t], END).iloc[0]
        if pd.isna(snap[F.REPORT_DATE]):
            continue
        cur, row = snap[F.REPORT_DATE], None
        for _ in range(4):  # walk back four reported quarters through point-in-time snapshots
            row = prov.get_fundamentals([t], (cur - pd.Timedelta(days=1)).date()).iloc[0]
            cur = row[F.REPORT_DATE]
            if pd.isna(cur):  # recent IPO: pre-listing quarters became public together
                break
        if pd.isna(cur):
            continue
        assert row[F.PERIOD_END] < snap[F.PERIOD_END] - pd.Timedelta(days=330)
        assert row[F.TOTAL_ASSETS] == pytest.approx(snap[F.TOTAL_ASSETS_PRIOR_YEAR], rel=1e-12)


def test_fundamentals_point_in_time_and_ttm(prov):
    tick = prov.tickers[:60]
    snap = prov.get_fundamentals(tick, END)
    ipo_names = set(prov.get_universe(None, END).index) - set(prov.get_universe(None, date(2024, 3, 1)).index)
    for t in tick[:25]:
        rd = snap.at[t, F.REPORT_DATE]
        if pd.isna(rd):
            continue
        # the day before the report date, the latest quarter is not yet public
        prev = prov.get_fundamentals([t], (rd - pd.Timedelta(days=1)).date()).iloc[0]
        assert prev[F.REPORT_DATE] < rd and prev[F.PERIOD_END] < snap.at[t, F.PERIOD_END]
        on = prov.get_fundamentals([t], rd.date()).iloc[0]
        assert on[F.REPORT_DATE] == rd
        # TTM == sum of the last four reported quarters (walk back through point-in-time snapshots)
        last_q, cur = [], rd
        for _ in range(8):
            row = prov.get_fundamentals([t], cur.date()).iloc[0]
            if pd.isna(row[F.REPORT_DATE]):
                break
            last_q.append(row[F.REVENUE_LAST_Q])
            cur = row[F.REPORT_DATE] - pd.Timedelta(days=1)  # the day before: previous quarter is the latest
        if len(last_q) < 8:  # recent IPO: pre-listing quarters became public together with the S-1
            assert t in ipo_names
            continue
        assert snap.at[t, F.REVENUE_TTM] == pytest.approx(sum(last_q[:4]), rel=1e-12)
        assert snap.at[t, F.REVENUE_TTM_PRIOR_YEAR] == pytest.approx(sum(last_q[4:8]), rel=1e-12)
        assert snap.at[t, F.REVENUE_LAST_Q_PRIOR_YEAR] == pytest.approx(last_q[4], rel=1e-12)
    early = prov.get_fundamentals(tick, date(2019, 1, 1))
    assert early.drop(columns=[F.PERIOD_END, F.REPORT_DATE]).isna().all().all() and early[F.REPORT_DATE].isna().all()
    after = prov.get_fundamentals(tick, date(2030, 1, 1))  # clamped to the provider's end
    pd.testing.assert_frame_equal(after, snap)
    unk = prov.get_fundamentals(["ZZZZZ"], END)
    assert unk.index.tolist() == ["ZZZZZ"] and unk.isna().all().all()


# ---------------------------------------------------------------------------------------------
# Estimates / short interest / options
# ---------------------------------------------------------------------------------------------


def test_estimates_consistency(prov, raw):
    est, fu = raw["est"], raw["fund"]
    assert list(est.columns) == F.ESTIMATE_COLUMNS
    ok = fu[F.REPORT_DATE].notna()
    assert (est.loc[ok, F.LAST_EARNINGS_DATE] == fu.loc[ok, F.REPORT_DATE]).all()
    nxt = est.loc[ok, F.NEXT_EARNINGS_DATE]
    assert (nxt > pd.Timestamp(END)).all() and ((nxt - est.loc[ok, F.LAST_EARNINGS_DATE]).dt.days.between(70, 110)).all()
    # "3m ago" equals the consensus as of 91 days earlier
    then = prov.get_estimates(prov.tickers, END - timedelta(days=91))
    pd.testing.assert_series_equal(est[F.REVENUE_NTM_EST_3M_AGO], then[F.REVENUE_NTM_EST], check_names=False)
    pd.testing.assert_series_equal(est[F.EPS_NTM_EST_3M_AGO], then[F.EPS_NTM_EST], check_names=False)
    assert (est.loc[ok, F.NUM_ANALYSTS] >= 1).all()
    assert (est.loc[ok, F.TARGET_PRICE_MEAN] > 0).all()
    eps_ttm = fu[F.NET_INCOME_TTM] / fu[F.SHARES_OUTSTANDING]
    np.testing.assert_allclose(est.loc[ok, F.EPS_TTM], eps_ttm[ok], rtol=1e-9)


def test_short_interest(prov, raw):
    si, fu = raw["si"], raw["fund"]
    assert list(si.columns) == F.SHORT_INTEREST_COLUMNS
    ok = si[F.SHORT_INTEREST_SHARES].notna()
    sd = si.loc[ok, F.SI_SETTLEMENT_DATE]
    assert (sd <= pd.Timestamp(END)).all() and sd.nunique() == 1
    frac = (si[F.FLOAT_SHARES] / fu[F.SHARES_OUTSTANDING])[ok & fu[F.SHARES_OUTSTANDING].notna()]
    assert frac.between(0.849, 0.981).all()
    # settlement dates: semi-monthly (mid-month or month-end business days)
    dates = {prov.get_short_interest(prov.tickers[:1], d).iloc[0][F.SI_SETTLEMENT_DATE]
             for d in pd.bdate_range("2025-01-01", "2025-12-31")}
    in_2025 = {d for d in dates if d.year == 2025}
    assert len(in_2025) == 24 and dates - in_2025 == {pd.Timestamp(2024, 12, 31)}
    assert all(d.day >= 25 or 12 <= d.day <= 15 for d in dates)
    # point-in-time: as of a mid-month date we see the previous settlement, and 1m-ago is two settlements back
    mid = prov.get_short_interest(prov.tickers, date(2026, 9, 14))
    assert (mid.loc[ok, F.SI_SETTLEMENT_DATE] == pd.Timestamp(2026, 8, 31)).all()
    month_ago = prov.get_short_interest(prov.tickers, date(2026, 8, 31))
    pd.testing.assert_series_equal(si[F.SHORT_INTEREST_SHARES_1M_AGO], month_ago[F.SHORT_INTEREST_SHARES], check_names=False)


def test_options_summary(prov, raw):
    o = raw["opt"]
    assert list(o.columns) == F.OPTIONS_COLUMNS
    ok = o[F.IV_30D_ATM].notna()
    assert ok.mean() > 0.85
    assert ((o.loc[ok, F.IV_30D_ATM] >= o.loc[ok, F.IV_30D_ATM_1Y_LOW] - 1e-9)
            & (o.loc[ok, F.IV_30D_ATM] <= o.loc[ok, F.IV_30D_ATM_1Y_HIGH] + 1e-9)).all()
    assert o.loc[ok, F.IV_30D_ATM].between(0.05, 3.0).all()
    for col in (F.PUT_VOLUME, F.CALL_VOLUME, F.PUT_OPEN_INTEREST, F.CALL_OPEN_INTEREST):
        assert (o.loc[ok, col] >= 0).all()
    # names without listed options (illiquid, penny, preferred) are NaN, not zero
    assert o[F.IV_30D_ATM].isna().sum() >= 5
    assert prov.get_options_summary(prov.tickers[:3], date(2020, 1, 1)).isna().all().all()


# ---------------------------------------------------------------------------------------------
# Archetype stories
# ---------------------------------------------------------------------------------------------


def test_archetype_mix_and_ground_truth_api(prov):
    arch = prov.archetypes()
    counts = pd.Series(arch).value_counts()
    assert counts["transitory_shock"] == 15 and counts["value_trap"] == 15
    assert counts["guidance_reset"] == 10 and counts["sector_contagion"] == 10 and counts["momentum_leader"] == 25
    t = by_arch(prov, "value_trap")[0]
    assert prov.archetype(t) == "value_trap"
    st = prov.story(t)
    assert st["expected_dislocation_type"] == ARCHETYPE_TO_DISLOCATION["value_trap"]
    assert st["report_date"] <= END and st["shock_date"] <= END
    assert prov.story(by_arch(prov, "normal")[0]) == {}
    with pytest.raises(KeyError):
        prov.archetype("NOPE")
    for a in DISLOCATION_ARCHES:  # every planted name was calibrated to its design
        assert all(prov.story(x)["calibrated"] for x in by_arch(prov, a))


def test_canonical_screen_survivors_and_mix(prov, raw):
    df = canonical_features(raw)
    surv = df[canonical_mask(df)]
    mix = surv.arch.value_counts()
    assert 8 <= len(surv) <= 20, surv
    assert mix.get("transitory_shock", 0) >= 2 and mix.get("value_trap", 0) >= 2
    assert mix.get("guidance_reset", 0) >= 1 and mix.get("sector_contagion", 0) >= 1
    assert surv.arch.isin(DISLOCATION_ARCHES).mean() >= 0.8
    fit = [t for t, a in prov.archetypes().items() if a in DISLOCATION_ARCHES and prov.story(t)["canonical_fit"]]
    assert set(surv.index) >= set(fit) - set(fit[:1]) or set(fit) <= set(surv.index)
    # deliberate near misses fail the screen
    near = [t for t, a in prov.archetypes().items() if a in DISLOCATION_ARCHES and not prov.story(t)["canonical_fit"]]
    assert not set(near) & set(surv.index)


def test_canonical_screen_robust_across_seeds():
    for seed in (1, 2):
        p = SyntheticProvider(seed=seed)
        t = p.tickers
        raw = dict(panel=p.get_price_history(t, p.start, END), uni=p.get_universe(None, END), fund=p.get_fundamentals(t, END),
                   si=p.get_short_interest(t, END), arch=pd.Series(p.archetypes()))
        surv = canonical_features(raw)[canonical_mask(canonical_features(raw))]
        assert 6 <= len(surv) <= 24
        assert surv.arch.isin(DISLOCATION_ARCHES).sum() >= 6


def test_planted_price_stories(prov, raw):
    df = canonical_features(raw)
    c, v = raw["panel"].close, raw["panel"].volume
    for t in by_arch(prov, "transitory_shock") + by_arch(prov, "value_trap") + by_arch(prov, "guidance_reset"):
        st = prov.story(t)
        rd = pd.Timestamp(st["report_date"])
        i = c.index.get_loc(rd)
        assert 25 <= len(c) - 1 - i <= 60  # shock 25-60 sessions before the end
        assert v[t].iloc[i] / v[t].iloc[i - 120: i].mean() >= 3.0  # heavy volume on the report
        assert c[t].iloc[i] / c[t].iloc[i - 1] - 1 < -0.04  # gapped down
        if "drawdown" not in st["near_misses"]:
            assert -40 < df.at[t, "dd"] < -15
        if "trend" not in st["near_misses"]:
            assert df.at[t, "trend"] > 0 and df.at[t, "mom_12_1"] > 0
    for t in by_arch(prov, "momentum_leader"):
        assert df.at[t, "dd"] > -10 and df.at[t, "rsi"] > 50 and df.at[t, "si_pct"] < 3 and df.at[t, "mom_12_1"] > 0
    # sector contagion: the whole industry sold off on the headline day
    uni = raw["uni"]
    for t in by_arch(prov, "sector_contagion")[:3]:
        hd = pd.Timestamp(prov.story(t)["shock_date"])
        peers = uni.index[uni[F.GICS_INDUSTRY] == uni.at[t, F.GICS_INDUSTRY]]
        day = (c.loc[hd, peers] / c.shift(1).loc[hd, peers] - 1)
        assert day.median() < -0.04 and (day < 0).mean() > 0.8
        assert pd.Timestamp(prov.story(t)["report_date"]) > hd  # management addresses it on the next call


def test_planted_fundamental_stories(prov, raw):
    fu, est, si, opt = raw["fund"], raw["est"], raw["si"], raw["opt"]
    g_ttm = fu[F.REVENUE_TTM] / fu[F.REVENUE_TTM_PRIOR_YEAR] - 1
    g_q = fu[F.REVENUE_LAST_Q] / fu[F.REVENUE_LAST_Q_PRIOR_YEAR] - 1
    gm_chg = fu[F.GROSS_PROFIT_TTM] / fu[F.REVENUE_TTM] - fu[F.GROSS_PROFIT_TTM_PRIOR_YEAR] / fu[F.REVENUE_TTM_PRIOR_YEAR]
    om_chg = (fu[F.OPERATING_INCOME_TTM] / fu[F.REVENUE_TTM]
              - fu[F.OPERATING_INCOME_TTM_PRIOR_YEAR] / fu[F.REVENUE_TTM_PRIOR_YEAR])
    rev_rev = est[F.REVENUE_NTM_EST] / est[F.REVENUE_NTM_EST_3M_AGO] - 1
    eps_rev = est[F.EPS_NTM_EST] / est[F.EPS_NTM_EST_3M_AGO] - 1
    si_pct = si[F.SHORT_INTEREST_SHARES] / si[F.FLOAT_SHARES]
    si_chg = si[F.SHORT_INTEREST_SHARES] / si[F.SHORT_INTEREST_SHARES_1M_AGO] - 1
    fcf_y = fu[F.FCF_TTM] / raw["uni"][F.MARKET_CAP]
    for t in by_arch(prov, "transitory_shock"):
        miss = prov.story(t)["near_misses"]
        assert -0.081 <= rev_rev[t] <= -0.029 and -0.081 <= eps_rev[t] <= -0.034
        assert g_q[t] < g_ttm[t] and abs(gm_chg[t]) < 0.01
        if "si" not in miss:
            assert 0.079 <= si_pct[t] <= 0.201 and si_chg[t] > 0
        if "fcf" not in miss:
            assert 0.049 <= fcf_y[t] <= 0.091
        if "growth" not in miss:
            assert g_ttm[t] > 0.10
        assert est.at[t, F.LAST_EPS_SURPRISE] < 0.011
    for t in by_arch(prov, "value_trap"):
        miss = prov.story(t)["near_misses"]
        assert g_q[t] < 0.011 and g_q[t] < g_ttm[t] - 0.07  # latest quarter negative / sharply decelerating
        assert gm_chg[t] < 0 and om_chg[t] < 0
        assert -0.301 <= eps_rev[t] <= -0.159 and rev_rev[t] <= -0.099
        assert est.at[t, F.LAST_EPS_SURPRISE] < 0
        if "growth" not in miss:
            assert g_ttm[t] > 0.08  # trailing screens still pass
        if "fcf" not in miss:
            assert fcf_y[t] > 0.04
    for t in by_arch(prov, "guidance_reset"):
        assert est.at[t, F.LAST_EPS_SURPRISE] > 0.039
        assert fu.at[t, F.CASH] > fu.at[t, F.TOTAL_DEBT]  # net cash
        assert rev_rev[t] < 0
    for t in by_arch(prov, "sector_contagion"):
        assert abs(rev_rev[t]) < 0.03 and est.at[t, F.LAST_EPS_SURPRISE] > 0 and g_ttm[t] > 0.05
    for t in by_arch(prov, "momentum_leader"):
        assert rev_rev[t] > 0 and eps_rev[t] > 0 and si_pct[t] < 0.03
    shocked = by_arch(prov, "transitory_shock") + by_arch(prov, "value_trap")
    before = prov.get_options_summary(shocked, END - timedelta(days=100))
    assert (opt.loc[shocked, F.IV_30D_ATM] > before[F.IV_30D_ATM]).mean() > 0.9  # IV elevated after the shock
    pcr = opt[F.PUT_VOLUME] / opt[F.CALL_VOLUME]
    assert pcr[shocked].median() > pcr[by_arch(prov, "momentum_leader")].median()


def test_planted_names_look_healthy_before_the_shock(prov):
    """Point-in-time: before the story quarter, planted names were compounders with intact estimates."""
    names = by_arch(prov, "value_trap")
    d = min(prov.story(t)["report_date"] for t in names) - timedelta(days=1)
    fu = prov.get_fundamentals(names, d)
    g_q = fu[F.REVENUE_LAST_Q] / fu[F.REVENUE_LAST_Q_PRIOR_YEAR] - 1
    assert (g_q > 0.05).all()
    est = prov.get_estimates(names, d)
    assert (est[F.LAST_EPS_SURPRISE] > 0).all()
    docs = prov.get_documents(names[0], {DocumentKind.TRANSCRIPT}, prov.start, d)
    assert docs and all(doc.published_at.date() <= d for doc in docs)


# ---------------------------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------------------------

_MONEY = re.compile(r"\$([\d,]+\.\d+) (million|billion)")


def _amount(m: re.Match) -> float:
    return float(m.group(1).replace(",", "")) * (1e6 if m.group(2) == "million" else 1e9)


def _sample_tickers(prov) -> list[str]:
    arch = prov.archetypes()
    planted = [t for t, a in arch.items() if a != "normal"]
    normal = [t for t, a in arch.items() if a == "normal" and prov.get_fundamentals([t], END)[F.REVENUE_TTM].notna().iloc[0]]
    return planted + normal[:30]


def test_documents_ordering_window_and_ids(prov):
    t = by_arch(prov, "transitory_shock")[0]
    docs = prov.get_documents(t, ALL_KINDS, prov.start, END, limit=1000)
    assert len(docs) >= 11 * 4
    ts = [d.published_at for d in docs]
    assert ts == sorted(ts, reverse=True)
    assert len({d.doc_id for d in docs}) == len(docs)
    assert all(d.ticker == t and d.published_at.date() <= END for d in docs)
    assert {d.kind for d in docs} == ALL_KINDS
    for d in docs:
        if d.kind == DocumentKind.TRANSCRIPT:
            assert d.doc_id == f"SYN-TR-{t}-{d.published_at:%Y%m%d}"
        assert re.fullmatch(r"SYN-(TR|NW|FL|RS)-[A-Z]{3,5}-\d{8}(-[A-Z0-9]+)?", d.doc_id)
    # [start, end] inclusive, limit, kinds filter
    day = docs[0].published_at.date()
    same_day = prov.get_documents(t, ALL_KINDS, day, day, limit=100)
    assert same_day and all(d.published_at.date() == day for d in same_day)
    assert len(prov.get_documents(t, ALL_KINDS, prov.start, END, limit=3)) == 3
    news = prov.get_documents(t, {DocumentKind.NEWS}, prov.start, END, limit=100)
    assert news and all(d.kind == DocumentKind.NEWS for d in news)
    assert prov.get_documents(t, set(), prov.start, END) == []
    assert prov.get_documents("NOPE", ALL_KINDS, prov.start, END) == []
    assert prov.get_documents(t, ALL_KINDS, date(2030, 1, 1), date(2031, 1, 1)) == []
    # returned documents are copies: mutating one does not leak into later calls
    docs[0].text = "mutated"
    assert prov.get_documents(t, ALL_KINDS, prov.start, END, limit=1)[0].text != "mutated"


def test_documents_point_in_time(prov):
    for t in by_arch(prov, "transitory_shock")[:3]:
        rd = prov.story(t)["report_date"]
        before = prov.get_documents(t, ALL_KINDS, prov.start, rd - timedelta(days=1), limit=1000)
        assert before and all(d.published_at.date() < rd for d in before)
        assert f"SYN-TR-{t}-{rd:%Y%m%d}" not in {d.doc_id for d in before}
        upto = prov.get_documents(t, {DocumentKind.TRANSCRIPT}, prov.start, rd, limit=1)
        assert upto[0].doc_id == f"SYN-TR-{t}-{rd:%Y%m%d}"


def test_transcript_structure(prov):
    t = by_arch(prov, "value_trap")[0]
    rd = prov.story(t)["report_date"]
    tr = prov.get_documents(t, {DocumentKind.TRANSCRIPT}, rd, rd)[0]
    segs = tr.segments
    assert segs[0].role == "Operator" and segs[1].role == "IR"
    prepared = [s for s in segs if s.section == "prepared_remarks"]
    assert [s.role for s in prepared] == ["Operator", "IR", "CEO", "CFO"]
    analysts = {s.speaker for s in segs if s.role == "Analyst"}
    assert 3 <= len(analysts) <= 9
    assert all(s.section == "qa" for s in segs if s.role == "Analyst")
    assert tr.text == "\n\n".join(f"{s.speaker} ({s.role}): {s.text}" for s in segs)
    assert tr.metadata["event"] == "earnings_call"


def test_transcript_numbers_match_fundamentals(prov):
    checked = 0
    for t in _sample_tickers(prov):
        for tr in prov.get_documents(t, {DocumentKind.TRANSCRIPT}, date(2025, 6, 1), END, limit=4):
            snap = prov.get_fundamentals([t], tr.published_at.date()).iloc[0]
            assert snap[F.REPORT_DATE] == pd.Timestamp(tr.published_at.date())
            cfo = next(s for s in tr.segments if s.role == "CFO")
            m = re.search(r"Revenue for the \w+ quarter was " + _MONEY.pattern + r", (up|down) ([\d.]+)%", cfo.text)
            assert m, cfo.text[:200]
            rev = _amount(m)
            tol = 0.051e6 if m.group(2) == "million" else 0.0051e9
            assert abs(rev - snap[F.REVENUE_LAST_Q]) <= tol
            growth = snap[F.REVENUE_LAST_Q] / snap[F.REVENUE_LAST_Q_PRIOR_YEAR] - 1
            sign = 1 if m.group(3) == "up" else -1
            assert sign * float(m.group(4)) == pytest.approx(growth * 100, abs=0.051)
            m2 = re.search(r"Over the trailing twelve months, (?:we generated|free cash flow was an outflow of) "
                           + _MONEY.pattern, cfo.text)
            assert m2
            assert abs(_amount(m2) - abs(snap[F.FCF_TTM])) <= (0.051e6 if m2.group(2) == "million" else 0.0051e9)
            m3 = re.search(r"of cash and equivalents and " + _MONEY.pattern + " of total debt", cfo.text)
            assert m3 and abs(_amount(m3) - snap[F.TOTAL_DEBT]) <= (0.051e6 if m3.group(2) == "million" else 0.0051e9)
            checked += 1
    assert checked >= 200


def test_transcript_lengths(prov):
    for a in DISLOCATION_ARCHES:
        for t in by_arch(prov, a):
            rd = prov.story(t)["report_date"]
            words = len(prov.get_documents(t, {DocumentKind.TRANSCRIPT}, rd, rd)[0].text.split())
            assert 1500 <= words <= 3000, (t, a, words)
    planted_other = by_arch(prov, "transitory_shock")[0]
    older = prov.get_documents(planted_other, {DocumentKind.TRANSCRIPT}, prov.start, date(2026, 6, 30), limit=20)
    assert all(1200 <= len(d.text.split()) <= 3000 for d in older)
    normal = by_arch(prov, "normal")[:20]
    lens = [len(d.text.split()) for t in normal for d in prov.get_documents(t, {DocumentKind.TRANSCRIPT}, prov.start, END)]
    assert lens and max(lens) < 1200 and min(lens) > 400


def test_archetype_specific_narratives(prov):
    def story_call(t):
        rd = prov.story(t)["report_date"]
        return prov.get_documents(t, {DocumentKind.TRANSCRIPT}, rd, rd)[0]

    for t in by_arch(prov, "transitory_shock"):
        tr = story_call(t)
        ceo = next(s for s in tr.segments if s.role == "CEO").text
        assert "Excluding that item, revenue would have grown approximately" in ceo
        assert "book-to-bill" in ceo and "orders are running up" in ceo
        assert any("push back" in s.text for s in tr.segments if s.role == "Analyst")
        fu = prov.get_fundamentals([t], prov.story(t)["report_date"]).iloc[0]
        m = re.search(r"headwind of approximately " + _MONEY.pattern, ceo)
        target = prov.story(t)["targets"]["oneoff_pts"] * fu[F.REVENUE_LAST_Q_PRIOR_YEAR]
        assert m and abs(_amount(m) - target) <= 0.051e6 * (1 if m.group(2) == "million" else 100)
    for t in by_arch(prov, "value_trap"):
        txt = story_call(t).text
        assert "I don't think it's productive to parse it" in txt or "I'd be careful about isolating" in txt
        assert "no longer reaffirming the medium-term financial framework" in txt
    for t in by_arch(prov, "guidance_reset"):
        txt = story_call(t).text
        assert "embedded a meaningful degree of conservatism" in txt and "share repurchase" in txt
        assert "net cash position" in txt
        filings = prov.get_documents(t, {DocumentKind.FILING}, prov.story(t)["report_date"], END, limit=10)
        assert any("Item 8.01" in d.text for d in filings)
    for t in by_arch(prov, "sector_contagion"):
        txt = story_call(t).text
        assert "Sales tied to" in txt and "less than" in txt
        news = prov.get_documents(t, {DocumentKind.NEWS}, prov.story(t)["shock_date"], prov.story(t)["shock_date"])
        assert any(d.metadata.get("category") == "industry" for d in news)
    for t in by_arch(prov, "momentum_leader")[:5]:
        rd = prov.story(t)["report_date"]
        assert "raising" in prov.get_documents(t, {DocumentKind.TRANSCRIPT}, rd, rd)[0].text


def test_no_document_reveals_the_label(prov):
    pats = [re.compile(lbl.replace("_", "[ _-]?"), re.I) for lbl in LABELS] + [re.compile("contagion", re.I)]
    for t in _sample_tickers(prov):
        for d in prov.get_documents(t, ALL_KINDS, prov.start, END, limit=10_000):
            blob = " ".join([d.title, d.text, *d.metadata.values()])
            assert not any(p.search(blob) for p in pats), (t, d.doc_id)


def test_news_consistent_with_prices(prov):
    t = by_arch(prov, "transitory_shock")[1]
    c = prov.get_price_history([t], prov.start, END).close[t]
    moves = [d for d in prov.get_documents(t, {DocumentKind.NEWS}, prov.start, END, limit=200)
             if d.doc_id.endswith("-MOVE")]
    assert len(moves) >= 8
    for d in moves:
        day = pd.Timestamp(d.published_at.date())
        r = c.loc[day] / c.shift(1).loc[day] - 1
        m = re.search(r"closed (up|down) ([\d.]+)% at \$([\d.]+)", d.text)
        assert m
        assert (1 if m.group(1) == "up" else -1) * float(m.group(2)) == pytest.approx(r * 100, abs=0.051)
        assert float(m.group(3)) == pytest.approx(c.loc[day], abs=0.006)
    rd = prov.story(t)["report_date"]
    pr = [d for d in prov.get_documents(t, {DocumentKind.NEWS}, rd, rd) if d.doc_id.endswith("-PR")][0]
    assert pr.published_at.hour < 9  # pre-market release, before the gap


def test_filings_and_research(prov):
    t = by_arch(prov, "value_trap")[2]
    filings = prov.get_documents(t, {DocumentKind.FILING}, prov.start, END, limit=100)
    forms = {d.metadata.get("form") for d in filings}
    assert {"10-Q", "10-K", "8-K"} <= forms
    q = [d for d in filings if d.metadata.get("form") == "10-Q"][0]
    snap = prov.get_fundamentals([t], date.fromisoformat(q.metadata["period_end"]) + timedelta(days=46)).iloc[0]
    m = re.search(r"Net sales for the three months ended [A-Za-z]+ \d+, \d{4} were " + _MONEY.pattern, q.text)
    assert m and abs(_amount(m) - snap[F.REVENUE_LAST_Q]) <= 0.051e6 * (1 if m.group(2) == "million" else 100)
    research = prov.get_documents(t, {DocumentKind.RESEARCH}, prov.start, END, limit=100)
    assert research and all(d.kind == DocumentKind.RESEARCH for d in research)
    kept, withheld = prov.boundary.filter_documents(research)
    assert not kept and withheld  # broker research is withheld from the LLM by the default boundary


# ---------------------------------------------------------------------------------------------
# Long histories (factor research over 15 years)
# ---------------------------------------------------------------------------------------------

LONG_START = date(2012, 1, 2)


@pytest.fixture(scope="module")
def long_prov() -> SyntheticProvider:
    t0 = time.perf_counter()
    p = SyntheticProvider(n_tickers=400, seed=7, start=LONG_START, end=END)
    p.build_seconds = time.perf_counter() - t0  # type: ignore[attr-defined]
    return p


def _valuations(p: SyntheticProvider, d: date) -> pd.DataFrame:
    u = p.get_universe(None, d)
    u = u[u[F.SECURITY_TYPE].isin(["common_stock", "adr", "reit"])]
    fu = p.get_fundamentals(list(u.index), d)
    cap = u[F.MARKET_CAP]
    return pd.DataFrame({
        "fcf_yield": fu[F.FCF_TTM] / cap,
        "log_ps": np.log(cap / fu[F.REVENUE_TTM]),
        "growth": fu[F.REVENUE_TTM] / fu[F.REVENUE_TTM_PRIOR_YEAR] - 1,
        "asset_growth": fu[F.TOTAL_ASSETS] / fu[F.TOTAL_ASSETS_PRIOR_YEAR] - 1,
    })


def _spread(x: pd.Series) -> float:
    return float(x.quantile(0.9) - x.quantile(0.1))


def test_long_history_builds_fast_and_covers_the_window(long_prov):
    assert long_prov.build_seconds < 15.0
    t = long_prov.tickers
    panel = long_prov.get_price_history(t, LONG_START, END)
    assert len(panel.close) == len(pd.bdate_range(LONG_START, END)) and panel.close.iloc[0].notna().any()
    c = panel.close.to_numpy()
    assert (c[np.isfinite(c)] > 0).all()  # no price rounds down to 0 however far back the window goes
    early = long_prov.get_fundamentals(t, date(2012, 3, 1))  # quarters from 2009 are available
    assert early[F.REVENUE_TTM].notna().mean() > 0.9 and early[F.TOTAL_ASSETS].notna().mean() > 0.9


def test_long_history_valuations_stay_realistic(long_prov):
    """Multiples must not random-walk away over 15 years: relative valuations stay as dispersed as at the
    start of the last three years (where fundamentals evolve freely), medians stay plausible."""
    ref = _spread(_valuations(long_prov, date(2023, 6, 30)).log_ps)
    for d in (date(2014, 6, 30), date(2018, 6, 29), date(2021, 12, 31)):
        v = _valuations(long_prov, d)
        assert _spread(v.log_ps) < ref + 0.75, d
        assert 0.03 < v.fcf_yield.median() < 0.25 and v.fcf_yield.quantile(0.9) < 1.0, d
        assert 0.0 < v.growth.median() < 0.25 and v.growth.quantile(0.95) < 1.2, d
        assert abs(v.asset_growth.median() - v.growth.median()) < 0.05, d


def test_long_history_market_earns_the_market_factor(long_prov):
    """End-anchored cap weights would otherwise make the cap-weighted market trail its constituents."""
    b = long_prov.get_benchmark_history(LONG_START, date(2022, 12, 30))
    years = (b.index[-1] - b.index[0]).days / 365.25
    bench = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
    mret = long_prov._mret[: len(b)]  # the simulated market factor (log returns), same sessions
    market = math.exp(mret[1:].sum() / years) - 1
    assert abs(bench - market) < 0.01


def test_long_history_keeps_the_planted_end_of_sample_stories(long_prov):
    t = long_prov.tickers
    counts = pd.Series(long_prov.archetypes()).value_counts()
    assert counts["transitory_shock"] == 12 and counts["value_trap"] == 12 and counts["momentum_leader"] == 20
    raw = dict(panel=long_prov.get_price_history(t, date(2025, 1, 1), END), uni=long_prov.get_universe(None, END),
               fund=long_prov.get_fundamentals(t, END), si=long_prov.get_short_interest(t, END),
               arch=pd.Series(long_prov.archetypes()))
    df = canonical_features(raw)
    surv = df[canonical_mask(df)]
    assert 6 <= len(surv) <= 24
    assert surv.arch.isin(DISLOCATION_ARCHES).sum() >= 6


def test_long_history_factor_characteristics_have_coverage(long_prov):
    from aitrading.screen.features import FeatureEngine

    d = date(2015, 6, 30)
    uni = long_prov.get_universe(None, d)
    uni = uni[uni[F.SECURITY_TYPE] == "common_stock"]
    feats = ["book_to_market", "operating_profitability_pct", "asset_growth_yoy_pct", "gross_profitability_pct",
             "earnings_yield_ttm_pct"]
    ff = FeatureEngine(long_prov).build(uni, d, set(feats))
    assert all(ff.coverage[f] > 0.9 for f in feats), ff.coverage
    f = ff.frame
    assert 0.1 < f.book_to_market.median() < 3 and 5 < f.gross_profitability_pct.median() < 60
