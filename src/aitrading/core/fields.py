"""Canonical column names for the raw datasets every data provider must return.

Vendor adapters (Bloomberg, LSEG, S&P Capital IQ, synthetic) normalise their native field codes to
these names so that everything downstream (feature engine, screen, ranking) is vendor-neutral.

Conventions
-----------
* Currency amounts are in USD, absolute units (not thousands / millions).
* Ratios are fractions (0.35 == 35%) in the raw data. The *feature* layer converts to the
  units declared in the feature catalog (mostly percentages, named ``*_pct``).
* Point-in-time: a snapshot "as of" date D contains only information that was public on or
  before D (fundamentals keyed off the report/filing date, not the fiscal period end).
* All frames returned by providers are indexed by ``ticker`` (str) unless stated otherwise.
"""

from __future__ import annotations

# --- Reference / universe (provider.get_universe) -------------------------------------------
TICKER = "ticker"
NAME = "name"
GICS_SECTOR = "gics_sector"
GICS_INDUSTRY = "gics_industry"
EXCHANGE = "exchange"
COUNTRY = "country"  # ISO-3166 alpha-2 of primary listing, e.g. "US"
CURRENCY = "currency"
SECURITY_TYPE = "security_type"  # "common_stock", "adr", "reit", "etf", "preferred", ...
MARKET_CAP = "market_cap"  # USD, as of the snapshot date
VENDOR_ID = "vendor_id"  # the provider's native identifier (e.g. "AAPL US Equity", "AAPL.O", "IQ24937")

UNIVERSE_COLUMNS = [NAME, GICS_SECTOR, GICS_INDUSTRY, EXCHANGE, COUNTRY, CURRENCY, SECURITY_TYPE, MARKET_CAP, VENDOR_ID]

# --- Price panel (provider.get_price_history -> PricePanel) ------------------------------------
# Wide frames: DatetimeIndex (trading days, ascending) x tickers. Prices are split- and
# dividend-adjusted; volume is split-adjusted shares.
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
PRICE_FIELDS = [OPEN, HIGH, LOW, CLOSE, VOLUME]

# --- Fundamentals snapshot (provider.get_fundamentals) -----------------------------------------
PERIOD_END = "period_end"  # fiscal period end of the latest reported quarter (date)
REPORT_DATE = "report_date"  # date the latest quarter became public (date)
REVENUE_TTM = "revenue_ttm"
REVENUE_TTM_PRIOR_YEAR = "revenue_ttm_prior_year"  # TTM ending one year before PERIOD_END
REVENUE_LAST_Q = "revenue_last_q"
REVENUE_LAST_Q_PRIOR_YEAR = "revenue_last_q_prior_year"
GROSS_PROFIT_TTM = "gross_profit_ttm"
GROSS_PROFIT_TTM_PRIOR_YEAR = "gross_profit_ttm_prior_year"
OPERATING_INCOME_TTM = "operating_income_ttm"
OPERATING_INCOME_TTM_PRIOR_YEAR = "operating_income_ttm_prior_year"
EBITDA_TTM = "ebitda_ttm"
NET_INCOME_TTM = "net_income_ttm"
CFO_TTM = "cfo_ttm"  # cash flow from operations
CAPEX_TTM = "capex_ttm"  # positive number = cash spent
FCF_TTM = "fcf_ttm"  # CFO - capex
TOTAL_DEBT = "total_debt"
CASH = "cash_and_equivalents"
INTEREST_EXPENSE_TTM = "interest_expense_ttm"  # positive number
TOTAL_EQUITY = "total_equity"
SHARES_OUTSTANDING = "shares_outstanding"  # absolute share count

FUNDAMENTAL_COLUMNS = [
    PERIOD_END, REPORT_DATE, REVENUE_TTM, REVENUE_TTM_PRIOR_YEAR, REVENUE_LAST_Q, REVENUE_LAST_Q_PRIOR_YEAR,
    GROSS_PROFIT_TTM, GROSS_PROFIT_TTM_PRIOR_YEAR, OPERATING_INCOME_TTM, OPERATING_INCOME_TTM_PRIOR_YEAR,
    EBITDA_TTM, NET_INCOME_TTM, CFO_TTM, CAPEX_TTM, FCF_TTM, TOTAL_DEBT, CASH, INTEREST_EXPENSE_TTM,
    TOTAL_EQUITY, SHARES_OUTSTANDING,
]

# --- Consensus estimates snapshot (provider.get_estimates) -------------------------------------
REVENUE_NTM_EST = "revenue_ntm_est"  # next-twelve-months consensus revenue
REVENUE_NTM_EST_3M_AGO = "revenue_ntm_est_3m_ago"
EPS_NTM_EST = "eps_ntm_est"
EPS_NTM_EST_3M_AGO = "eps_ntm_est_3m_ago"
EPS_TTM = "eps_ttm"  # trailing diluted EPS (actuals) for growth comparisons
NUM_ANALYSTS = "num_analysts"
TARGET_PRICE_MEAN = "target_price_mean"
LAST_EPS_SURPRISE = "last_eps_surprise"  # fraction: (actual - est) / |est|
LAST_EARNINGS_DATE = "last_earnings_date"  # date
NEXT_EARNINGS_DATE = "next_earnings_date"  # date (may be NaT)

ESTIMATE_COLUMNS = [
    REVENUE_NTM_EST, REVENUE_NTM_EST_3M_AGO, EPS_NTM_EST, EPS_NTM_EST_3M_AGO, EPS_TTM, NUM_ANALYSTS,
    TARGET_PRICE_MEAN, LAST_EPS_SURPRISE, LAST_EARNINGS_DATE, NEXT_EARNINGS_DATE,
]

# --- Short interest snapshot (provider.get_short_interest) -------------------------------------
SHORT_INTEREST_SHARES = "short_interest_shares"
SHORT_INTEREST_SHARES_1M_AGO = "short_interest_shares_1m_ago"
FLOAT_SHARES = "float_shares"
SI_SETTLEMENT_DATE = "settlement_date"  # date of the latest short-interest settlement report

SHORT_INTEREST_COLUMNS = [SHORT_INTEREST_SHARES, SHORT_INTEREST_SHARES_1M_AGO, FLOAT_SHARES, SI_SETTLEMENT_DATE]

# --- Options summary snapshot (provider.get_options_summary) -----------------------------------
IV_30D_ATM = "iv_30d_atm"  # fraction, annualised (0.35 == 35 vol)
IV_30D_ATM_1Y_HIGH = "iv_30d_atm_1y_high"
IV_30D_ATM_1Y_LOW = "iv_30d_atm_1y_low"
PUT_VOLUME = "put_volume"  # contracts, latest session (or 5d avg if the vendor only gives that)
CALL_VOLUME = "call_volume"
PUT_OPEN_INTEREST = "put_open_interest"
CALL_OPEN_INTEREST = "call_open_interest"

OPTIONS_COLUMNS = [IV_30D_ATM, IV_30D_ATM_1Y_HIGH, IV_30D_ATM_1Y_LOW, PUT_VOLUME, CALL_VOLUME, PUT_OPEN_INTEREST, CALL_OPEN_INTEREST]
