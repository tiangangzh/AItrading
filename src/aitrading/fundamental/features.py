"""Fundamental features: fundamentals + consensus estimates snapshots -> catalog fundamental features.

Inputs are the provider snapshots indexed by ticker (``fields.FUNDAMENTAL_COLUMNS`` /
``fields.ESTIMATE_COLUMNS``, currency in USD absolute, ratios as fractions), the universe frame
(market cap from ``fields.MARKET_CAP``, USD absolute) and the latest adjusted close per ticker.

Conventions
-----------
* Output units follow the catalog: %, percentage points, multiples (x), USD bn, calendar days.
* Ratios are NaN when the denominator is missing, zero or negative (market cap, price, revenue,
  EBITDA, net income, equity, consensus revenue); numerators may be negative (negative FCF gives a
  negative FCF yield, a loss gives a negative earnings yield / operating margin).
* Market cap, price, target price, revenue estimates, capex, debt, cash, interest expense and the
  analyst count must be >= 0 (> 0 where used as a denominator); a negative value is a data error
  and is treated as missing. Non-numeric or infinite inputs are missing too.
* FCF is ``fcf_ttm``; when that is missing it falls back to ``cfo_ttm - capex_ttm``.
* Enterprise value = market cap + total debt - cash; it may be negative (net cash above market
  cap). ev_to_ebitda / ev_to_sales keep that sign: only the catalog's denominator guards apply.
* interest_coverage = operating income / interest expense, capped at 100. With no interest expense
  (exactly 0, or missing while total debt is exactly 0) it is 100 if operating income > 0, else NaN.
  Negative coverage (operating loss) is kept.
* eps_revision_3m_pct = (EPS now - EPS 3m ago) / |EPS 3m ago| x 100 (NaN when EPS 3m ago is 0), so
  an upward revision is positive even when the estimate was negative.
* Dates may be ``date``, ``Timestamp`` or ISO strings (unparseable -> missing; tz-aware values are
  taken in UTC). days_since_last_earnings uses ``last_earnings_date`` and falls back to the
  fundamentals ``report_date``; a date after as_of (not yet public) gives NaN. days_to_next_earnings
  is NaN when the date is missing or already before as_of.
* Tickers absent from an input get NaN for every feature that depends on it (never a KeyError);
  missing input columns behave like all-NaN columns. Duplicate input rows: the last one wins.

Factor characteristics (the academic factor sort variables)
-----------------------------------------------------------
* book_to_market = total equity / market cap (x); NaN when equity <= 0 (Fama-French exclude
  negative book equity from the B/M sort). Equity is the latest public balance sheet and market cap
  the current one (the monthly-formation variant of HML's B/M, not the June / December timing).
* operating_profitability_pct = (operating income TTM - interest expense TTM) / total equity x 100,
  NaN when equity <= 0. Fama-French (2015) define OP as (revenue - COGS - SG&A - interest expense)
  / book equity; operating income is revenue - COGS - SG&A - other operating items (incl. D&A and
  R&D), so this is the closest catalog approximation. A missing interest expense counts as 0
  (filers with no debt usually do not tag it), so coverage is not lost for debt-free companies.
* asset_growth_yoy_pct = (total assets / total assets four quarters earlier - 1) x 100 (the CMA
  investment variable; lower = conservative). gross_profitability_pct = gross profit TTM / total
  assets x 100 (Novy-Marx 2013). Both read the optional ``fields.TOTAL_ASSETS`` /
  ``fields.TOTAL_ASSETS_PRIOR_YEAR`` columns (``fields.FUNDAMENTAL_OPTIONAL_COLUMNS``) and are
  NaN when a provider does not supply them; total assets must be > 0.
* earnings_yield_ttm_pct = net income TTM / market cap x 100 (negative for loss-makers).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from aitrading.core import fields
from aitrading.screen.catalog import FUNDAMENTAL_FEATURES

AsOf = date | pd.Timestamp | str
COVERAGE_CAP = 100.0
_BN = 1e9


def _rows(data: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    """``data`` keyed by ticker (index or ``ticker`` column), deduplicated, re-indexed to ``index``."""
    if data is None or len(data) == 0:
        return pd.DataFrame(index=index)
    if fields.TICKER in data.columns and data.index.name != fields.TICKER:
        data = data.set_index(fields.TICKER)
    data = data[~data.index.duplicated(keep="last")]
    return data.reindex(index)


def _num(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    s = pd.to_numeric(data[column], errors="coerce").astype("float64")
    return s.where(np.isfinite(s))


def _dates(data: pd.DataFrame, column: str) -> pd.Series:
    """Column parsed to tz-naive midnight timestamps (NaT where missing / unparseable)."""
    if column not in data.columns or len(data) == 0:
        return pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    s = pd.to_datetime(data[column], errors="coerce", utc=True, format="mixed")
    return s.dt.tz_convert(None).dt.normalize()


def _nonneg(s: pd.Series) -> pd.Series:
    return s.where(s >= 0)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den where den > 0 and both are finite, else NaN."""
    ok = num.notna() & den.notna() & (den > 0)
    return (num / den.where(ok)).where(ok)


def _growth_pct(now: pd.Series, before: pd.Series) -> pd.Series:
    """(now / before - 1) x 100 with before > 0."""
    return (_ratio(now, before) - 1.0) * 100.0


def _as_of(as_of: AsOf) -> pd.Timestamp:
    ts = pd.Timestamp(as_of)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def compute_fundamental_features(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    estimates: pd.DataFrame | None,
    price: pd.Series | None,
    as_of: AsOf,
) -> pd.DataFrame:
    """Catalog fundamental features for every ticker in ``universe.index`` (order kept).

    Returns a float frame indexed by ``ticker`` with exactly the columns
    ``aitrading.screen.catalog.FUNDAMENTAL_FEATURES``, in that order.
    """
    index = pd.Index(list(dict.fromkeys(universe.index)), name=fields.TICKER)
    uni = _rows(universe, index)
    fa = _rows(fundamentals, index)
    est = _rows(estimates, index)
    if price is None or len(price) == 0:
        px = pd.Series(np.nan, index=index, dtype="float64")
    else:
        px = pd.to_numeric(price[~price.index.duplicated(keep="last")].reindex(index), errors="coerce").astype("float64")
    px = px.where(np.isfinite(px) & (px > 0))

    mcap = _num(uni, fields.MARKET_CAP)
    mcap = mcap.where(mcap > 0)

    revenue = _num(fa, fields.REVENUE_TTM)
    revenue_py = _num(fa, fields.REVENUE_TTM_PRIOR_YEAR)
    gross = _num(fa, fields.GROSS_PROFIT_TTM)
    gross_py = _num(fa, fields.GROSS_PROFIT_TTM_PRIOR_YEAR)
    op_inc = _num(fa, fields.OPERATING_INCOME_TTM)
    op_inc_py = _num(fa, fields.OPERATING_INCOME_TTM_PRIOR_YEAR)
    ebitda = _num(fa, fields.EBITDA_TTM)
    net_inc = _num(fa, fields.NET_INCOME_TTM)
    fcf = _num(fa, fields.FCF_TTM).fillna(_num(fa, fields.CFO_TTM) - _nonneg(_num(fa, fields.CAPEX_TTM)))
    debt = _nonneg(_num(fa, fields.TOTAL_DEBT))
    cash = _nonneg(_num(fa, fields.CASH))
    equity = _num(fa, fields.TOTAL_EQUITY)

    net_debt = debt - cash
    ev = mcap + net_debt

    interest_raw = _nonneg(_num(fa, fields.INTEREST_EXPENSE_TTM))
    interest = interest_raw.where(interest_raw.notna() | (debt != 0), 0.0)
    coverage = _ratio(op_inc, interest).clip(upper=COVERAGE_CAP)
    coverage = coverage.mask((interest == 0) & (op_inc > 0), COVERAGE_CAP)

    eps_ntm = _num(est, fields.EPS_NTM_EST)
    eps_ntm_3m = _num(est, fields.EPS_NTM_EST_3M_AGO)
    eps_ttm = _num(est, fields.EPS_TTM)
    rev_ntm = _nonneg(_num(est, fields.REVENUE_NTM_EST))
    rev_ntm_3m = _nonneg(_num(est, fields.REVENUE_NTM_EST_3M_AGO))
    target = _num(est, fields.TARGET_PRICE_MEAN)
    target = target.where(target > 0)

    eps_3m_abs = eps_ntm_3m.abs()
    eps_revision = _ratio(eps_ntm - eps_ntm_3m, eps_3m_abs) * 100.0

    def margin(num: pd.Series, rev: pd.Series = revenue) -> pd.Series:
        return _ratio(num, rev) * 100.0

    # factor characteristics (optional total-assets columns: absent -> NaN via _num)
    assets = _num(fa, fields.TOTAL_ASSETS)
    assets_py = _num(fa, fields.TOTAL_ASSETS_PRIOR_YEAR)
    op_profit = op_inc - interest_raw.fillna(0.0)

    ts = _as_of(as_of)
    last_er = _dates(est, fields.LAST_EARNINGS_DATE).fillna(_dates(fa, fields.REPORT_DATE))
    next_er = _dates(est, fields.NEXT_EARNINGS_DATE)
    since_last = (ts - last_er).dt.days.astype("float64")
    to_next = (next_er - ts).dt.days.astype("float64")

    out = pd.DataFrame(
        {
            "enterprise_value_usd_bn": ev / _BN,
            "fcf_yield_pct": _ratio(fcf, mcap) * 100.0,
            "ev_to_ebitda": _ratio(ev, ebitda),
            "ev_to_sales": _ratio(ev, revenue),
            "pe_ntm": _ratio(px, eps_ntm),
            "earnings_yield_ntm_pct": _ratio(eps_ntm, px) * 100.0,
            "target_price_upside_pct": _growth_pct(target, px),
            "revenue_growth_yoy_pct": _growth_pct(revenue, revenue_py),
            "revenue_growth_last_q_yoy_pct": _growth_pct(_num(fa, fields.REVENUE_LAST_Q), _num(fa, fields.REVENUE_LAST_Q_PRIOR_YEAR)),
            "revenue_growth_ntm_est_pct": _growth_pct(rev_ntm, revenue),
            "eps_growth_ntm_est_pct": _growth_pct(eps_ntm, eps_ttm),
            "gross_margin_pct": margin(gross),
            "operating_margin_pct": margin(op_inc),
            "ebitda_margin_pct": margin(ebitda),
            "fcf_margin_pct": margin(fcf),
            "net_margin_pct": margin(net_inc),
            "gross_margin_change_yoy_pp": margin(gross) - margin(gross_py, revenue_py),
            "operating_margin_change_yoy_pp": margin(op_inc) - margin(op_inc_py, revenue_py),
            "fcf_conversion_pct": _ratio(fcf, net_inc) * 100.0,
            "roe_pct": _ratio(net_inc, equity) * 100.0,
            "net_debt_usd_bn": net_debt / _BN,
            "net_debt_to_ebitda": _ratio(net_debt, ebitda),
            "interest_coverage": coverage,
            "cash_pct_market_cap": _ratio(cash, mcap) * 100.0,
            "eps_revision_3m_pct": eps_revision,
            "revenue_revision_3m_pct": _growth_pct(rev_ntm, rev_ntm_3m),
            "num_analysts": _nonneg(_num(est, fields.NUM_ANALYSTS)),
            "last_eps_surprise_pct": _num(est, fields.LAST_EPS_SURPRISE) * 100.0,
            "days_since_last_earnings": since_last.where(since_last >= 0),
            "days_to_next_earnings": to_next.where(to_next >= 0),
            "book_to_market": _ratio(equity, mcap).where(equity > 0),
            "operating_profitability_pct": _ratio(op_profit, equity) * 100.0,
            "asset_growth_yoy_pct": _growth_pct(assets.where(assets > 0), assets_py),
            "gross_profitability_pct": _ratio(gross, assets) * 100.0,
            "earnings_yield_ttm_pct": _ratio(net_inc, mcap) * 100.0,
        },
        index=index,
    )
    return out[FUNDAMENTAL_FEATURES].astype("float64")
