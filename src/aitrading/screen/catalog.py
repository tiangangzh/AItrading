"""Feature catalog: the closed vocabulary the screen DSL, the NL translator and the ranker share.

Each feature has a precise definition (implemented by the technical / fundamental / positioning
modules), a unit, and a source. The NL layer may only reference features listed here; anything it
cannot express goes into ``ScreenSpec.unsupported_requests`` rather than being silently dropped.

Window conventions (trading sessions): 1m = 21, 3m = 63, 6m = 126, 12m / 52w = 252.
"Price" is the latest split- and dividend-adjusted close on or before the as-of date.
A feature whose inputs are insufficient (short history, non-positive denominator) is NaN.
The ``factor_characteristics`` features are the sort variables of the academic factors (HML, RMW,
CMA, Novy-Marx gross profitability, trailing E/P); the asset-based ones need the optional
``total_assets`` fundamentals (``fields.FUNDAMENTAL_OPTIONAL_COLUMNS``) and are NaN without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

Source = Literal["reference", "technical", "fundamental", "positioning"]
DType = Literal["number", "bool", "category"]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    source: Source
    category: str
    dtype: DType
    unit: str
    description: str
    higher_is_better: bool | None = None  # default ranking direction hint; None = no natural direction

    def prompt_line(self) -> str:
        unit = f" [{self.unit}]" if self.unit else ""
        return f"- {self.name} ({self.dtype}){unit}: {self.description}"


def _f(name, source, category, dtype, unit, description, higher_is_better=None) -> FeatureDef:
    return FeatureDef(name, source, category, dtype, unit, description, higher_is_better)


_FEATURES: list[FeatureDef] = [
    # ---------------- classification (reference) ----------------
    _f("gics_sector", "reference", "classification", "category", "label", "GICS sector name, e.g. 'Information Technology', 'Health Care', 'Industrials'."),
    _f("gics_industry", "reference", "classification", "category", "label", "GICS industry name."),
    _f("exchange", "reference", "classification", "category", "label", "Primary listing exchange code, e.g. 'NYSE', 'NASDAQ'."),
    # ---------------- size & liquidity ----------------
    _f("market_cap_usd_bn", "reference", "size", "number", "USD bn", "Equity market capitalisation in USD billions. Small cap ~0.3-2, mid cap ~2-10, large cap >10.", None),
    _f("enterprise_value_usd_bn", "fundamental", "size", "number", "USD bn", "Market cap + total debt - cash, USD billions.", None),
    _f("price", "technical", "size", "number", "USD", "Latest adjusted close price."),
    _f("avg_dollar_volume_20d_usd_mn", "technical", "liquidity", "number", "USD mn", "Mean of close x volume over the last 20 sessions, USD millions.", True),
    # ---------------- trend / moving-average structure ----------------
    _f("sma_20", "technical", "trend", "number", "USD", "20-session simple moving average of close."),
    _f("sma_50", "technical", "trend", "number", "USD", "50-session simple moving average of close."),
    _f("sma_200", "technical", "trend", "number", "USD", "200-session simple moving average of close."),
    _f("price_vs_sma_50_pct", "technical", "trend", "number", "%", "(price / sma_50 - 1) x 100. Negative = trading below the 50-day."),
    _f("price_vs_sma_200_pct", "technical", "trend", "number", "%", "(price / sma_200 - 1) x 100. Positive = above the 200-day (long-term uptrend)."),
    _f("sma_50_vs_sma_200_pct", "technical", "trend", "number", "%", "(sma_50 / sma_200 - 1) x 100. Positive = 50-day above 200-day (golden-cross structure)."),
    _f("sma_200_slope_1m_pct", "technical", "trend", "number", "%", "(sma_200 today / sma_200 21 sessions ago - 1) x 100. Positive = rising 200-day."),
    _f("golden_cross_20d", "technical", "trend", "bool", "bool", "1 if sma_50 crossed above sma_200 within the last 20 sessions, else 0."),
    # ---------------- momentum & relative strength ----------------
    _f("return_1m_pct", "technical", "momentum", "number", "%", "Total return over the last 21 sessions, %.", True),
    _f("return_3m_pct", "technical", "momentum", "number", "%", "Total return over the last 63 sessions, %.", True),
    _f("return_6m_pct", "technical", "momentum", "number", "%", "Total return over the last 126 sessions, %.", True),
    _f("return_12m_pct", "technical", "momentum", "number", "%", "Total return over the last 252 sessions, %.", True),
    _f("return_12m_ex_1m_pct", "technical", "momentum", "number", "%", "12-1 momentum: return from 252 to 21 sessions ago, %.", True),
    _f("rel_strength_3m_pp", "technical", "momentum", "number", "pp", "return_3m_pct minus the benchmark's 3m return, percentage points.", True),
    _f("rel_strength_6m_pp", "technical", "momentum", "number", "pp", "return_6m_pct minus the benchmark's 6m return, percentage points.", True),
    _f("rel_strength_12m_pp", "technical", "momentum", "number", "pp", "return_12m_pct minus the benchmark's 12m return, percentage points.", True),
    _f("return_6m_percentile", "technical", "momentum", "number", "0-100", "Cross-sectional percentile (0-100) of return_6m_pct within the screened universe.", True),
    # ---------------- oscillators ----------------
    _f("rsi_14", "technical", "oscillator", "number", "0-100", "Wilder 14-session RSI. <30 oversold, >70 overbought."),
    _f("macd_line", "technical", "oscillator", "number", "USD", "EMA(12) - EMA(26) of close."),
    _f("macd_signal", "technical", "oscillator", "number", "USD", "EMA(9) of macd_line."),
    _f("macd_histogram", "technical", "oscillator", "number", "USD", "macd_line - macd_signal."),
    _f("macd_histogram_pct_price", "technical", "oscillator", "number", "%", "macd_histogram / price x 100 (comparable across stocks)."),
    _f("macd_bullish_cross_10d", "technical", "oscillator", "bool", "bool", "1 if macd_line crossed above macd_signal within the last 10 sessions, else 0."),
    # ---------------- range / drawdown ----------------
    _f("high_52w", "technical", "range", "number", "USD", "Highest high over the last 252 sessions."),
    _f("low_52w", "technical", "range", "number", "USD", "Lowest low over the last 252 sessions."),
    _f("drawdown_from_52w_high_pct", "technical", "range", "number", "%", "(price / high_52w - 1) x 100. Always <= 0; -25 means 25% below the 52-week high."),
    _f("above_52w_low_pct", "technical", "range", "number", "%", "(price / low_52w - 1) x 100. Always >= 0."),
    _f("days_since_52w_high", "technical", "range", "number", "sessions", "Trading sessions since the 52-week high was set."),
    # ---------------- volatility ----------------
    _f("volatility_20d_pct", "technical", "volatility", "number", "%", "Annualised stdev of daily log returns over 20 sessions, %."),
    _f("volatility_60d_pct", "technical", "volatility", "number", "%", "Annualised stdev of daily log returns over 60 sessions, %."),
    _f("atr_14_pct", "technical", "volatility", "number", "%", "Wilder 14-session average true range / price x 100."),
    _f("beta_1y", "technical", "volatility", "number", "x", "OLS beta of daily returns vs the benchmark over 252 sessions."),
    # ---------------- volume ----------------
    _f("rel_volume_5d", "technical", "volume", "number", "x", "Mean volume last 5 sessions / mean volume last 60 sessions. >1.5 = volume surge."),
    _f("rel_volume_20d", "technical", "volume", "number", "x", "Mean volume last 20 sessions / mean volume last 120 sessions."),
    _f("max_volume_ratio_20d", "technical", "volume", "number", "x", "Max single-session volume in the last 20 sessions / mean volume over the last 120 sessions (detects capitulation / event days)."),
    _f("up_down_volume_ratio_50d", "technical", "volume", "number", "x", "Sum of volume on up-close sessions / sum on down-close sessions over 50 sessions. >1 = accumulation."),
    # ---------------- options ----------------
    _f("iv_30d_pct", "positioning", "options", "number", "%", "30-day at-the-money implied volatility, annualised %."),
    _f("iv_rank_1y", "positioning", "options", "number", "0-100", "(iv - 1y low) / (1y high - 1y low) x 100."),
    _f("iv_to_realized_vol_ratio", "positioning", "options", "number", "x", "iv_30d_pct / volatility_20d_pct. >1 = options price more risk than realised."),
    _f("put_call_volume_ratio", "positioning", "options", "number", "x", "Put volume / call volume."),
    _f("put_call_oi_ratio", "positioning", "options", "number", "x", "Put open interest / call open interest."),
    # ---------------- short interest ----------------
    _f("short_interest_pct_float", "positioning", "short_interest", "number", "%", "Short interest / float shares x 100."),
    _f("days_to_cover", "positioning", "short_interest", "number", "days", "Short interest / 20-session average daily volume (shares)."),
    _f("short_interest_change_1m_pct", "positioning", "short_interest", "number", "%", "(short interest / short interest one month earlier - 1) x 100."),
    # ---------------- valuation ----------------
    _f("fcf_yield_pct", "fundamental", "valuation", "number", "%", "Trailing-12m free cash flow (CFO - capex) / market cap x 100.", True),
    _f("ev_to_ebitda", "fundamental", "valuation", "number", "x", "Enterprise value / TTM EBITDA (NaN if EBITDA <= 0).", False),
    _f("ev_to_sales", "fundamental", "valuation", "number", "x", "Enterprise value / TTM revenue.", False),
    _f("pe_ntm", "fundamental", "valuation", "number", "x", "Price / next-12m consensus EPS (NaN if EPS <= 0).", False),
    _f("earnings_yield_ntm_pct", "fundamental", "valuation", "number", "%", "Next-12m consensus EPS / price x 100.", True),
    _f("target_price_upside_pct", "fundamental", "valuation", "number", "%", "(mean analyst target price / price - 1) x 100.", True),
    # ---------------- growth ----------------
    _f("revenue_growth_yoy_pct", "fundamental", "growth", "number", "%", "TTM revenue vs TTM revenue one year earlier, %.", True),
    _f("revenue_growth_last_q_yoy_pct", "fundamental", "growth", "number", "%", "Latest quarter revenue vs same quarter a year earlier, %.", True),
    _f("revenue_growth_ntm_est_pct", "fundamental", "growth", "number", "%", "Next-12m consensus revenue vs TTM revenue, %.", True),
    _f("eps_growth_ntm_est_pct", "fundamental", "growth", "number", "%", "Next-12m consensus EPS vs TTM EPS, % (NaN if TTM EPS <= 0).", True),
    # ---------------- profitability ----------------
    _f("gross_margin_pct", "fundamental", "profitability", "number", "%", "TTM gross profit / revenue x 100.", True),
    _f("operating_margin_pct", "fundamental", "profitability", "number", "%", "TTM operating income / revenue x 100.", True),
    _f("ebitda_margin_pct", "fundamental", "profitability", "number", "%", "TTM EBITDA / revenue x 100.", True),
    _f("fcf_margin_pct", "fundamental", "profitability", "number", "%", "TTM FCF / revenue x 100.", True),
    _f("net_margin_pct", "fundamental", "profitability", "number", "%", "TTM net income / revenue x 100.", True),
    _f("gross_margin_change_yoy_pp", "fundamental", "profitability", "number", "pp", "TTM gross margin minus TTM gross margin one year earlier, percentage points.", True),
    _f("operating_margin_change_yoy_pp", "fundamental", "profitability", "number", "pp", "TTM operating margin minus the year-earlier TTM operating margin, percentage points.", True),
    _f("fcf_conversion_pct", "fundamental", "profitability", "number", "%", "TTM FCF / TTM net income x 100 (NaN if net income <= 0).", True),
    _f("roe_pct", "fundamental", "profitability", "number", "%", "TTM net income / total equity x 100 (NaN if equity <= 0).", True),
    # ---------------- balance sheet ----------------
    _f("net_debt_usd_bn", "fundamental", "balance_sheet", "number", "USD bn", "Total debt - cash, USD billions (negative = net cash).", False),
    _f("net_debt_to_ebitda", "fundamental", "balance_sheet", "number", "x", "Net debt / TTM EBITDA (NaN if EBITDA <= 0; negative = net cash).", False),
    _f("interest_coverage", "fundamental", "balance_sheet", "number", "x", "TTM operating income / TTM interest expense; capped at 100 (100 when there is no interest expense and operating income > 0).", True),
    _f("cash_pct_market_cap", "fundamental", "balance_sheet", "number", "%", "Cash and equivalents / market cap x 100.", None),
    # ---------------- estimates ----------------
    _f("eps_revision_3m_pct", "fundamental", "estimates", "number", "%", "Change in next-12m consensus EPS over the last 3 months, % (relative to |EPS 3m ago|).", True),
    _f("revenue_revision_3m_pct", "fundamental", "estimates", "number", "%", "Change in next-12m consensus revenue over the last 3 months, %.", True),
    _f("num_analysts", "fundamental", "estimates", "number", "count", "Number of analysts in the EPS consensus.", None),
    _f("last_eps_surprise_pct", "fundamental", "estimates", "number", "%", "Last reported EPS surprise vs consensus, %.", True),
    # ---------------- events ----------------
    _f("days_since_last_earnings", "fundamental", "events", "number", "days", "Calendar days since the last earnings report."),
    _f("days_to_next_earnings", "fundamental", "events", "number", "days", "Calendar days until the next expected earnings report (NaN if unknown)."),
    # ---------------- factor characteristics (academic factor sort variables) ----------------
    _f("book_to_market", "fundamental", "factor_characteristics", "number", "x", "Book equity (total stockholders' equity, latest balance sheet) / market cap: the Fama-French HML sort variable on the current market cap (NaN if equity <= 0).", True),
    _f("operating_profitability_pct", "fundamental", "factor_characteristics", "number", "%", "(TTM operating income - TTM interest expense) / total equity x 100: approximates Fama-French (2015) operating profitability, the RMW sort variable (NaN if equity <= 0).", True),
    _f("asset_growth_yoy_pct", "fundamental", "factor_characteristics", "number", "%", "(total assets / total assets four quarters earlier - 1) x 100: the CMA investment variable; low = conservative, high = aggressive.", False),
    _f("gross_profitability_pct", "fundamental", "factor_characteristics", "number", "%", "TTM gross profit / total assets x 100 (Novy-Marx 2013 gross profits-to-assets).", True),
    _f("earnings_yield_ttm_pct", "fundamental", "factor_characteristics", "number", "%", "TTM net income / market cap x 100 (trailing E/P from reported earnings; negative for loss-makers).", True),
]


class FeatureCatalog:
    def __init__(self, features: list[FeatureDef]):
        names = [f.name for f in features]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate feature names: {sorted(dupes)}")
        self._by_name = {f.name: f for f in features}

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> FeatureDef:
        return self._by_name[name]

    def __iter__(self) -> Iterator[FeatureDef]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> list[str]:
        return list(self._by_name)

    def by_source(self, source: Source) -> list[FeatureDef]:
        return [f for f in self if f.source == source]

    def to_prompt(self) -> str:
        """Render the catalog for an LLM prompt, grouped by category, in a stable order."""
        lines: list[str] = []
        seen: list[str] = []
        for f in self:
            if f.category not in seen:
                seen.append(f.category)
        for cat in seen:
            lines.append(f"## {cat}")
            lines.extend(f.prompt_line() for f in self if f.category == cat)
        return "\n".join(lines)


_DEFAULT = FeatureCatalog(_FEATURES)


def default_catalog() -> FeatureCatalog:
    return _DEFAULT


TECHNICAL_FEATURES = [f.name for f in _DEFAULT.by_source("technical")]
FUNDAMENTAL_FEATURES = [f.name for f in _DEFAULT.by_source("fundamental")]
POSITIONING_FEATURES = [f.name for f in _DEFAULT.by_source("positioning")]
REFERENCE_FEATURES = [f.name for f in _DEFAULT.by_source("reference")]
