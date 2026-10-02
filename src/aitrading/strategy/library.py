"""Curated research-idea templates for the idea lab.

Each :class:`IdeaTemplate` is a well-known, published idea expressed as a ready-to-run
:class:`~aitrading.strategy.spec.StrategySpec`. The templates serve three purposes:

* the offline :class:`~aitrading.strategy.nl.HeuristicStrategyTranslator` maps an idea to the
  closest template and then applies parameter overrides parsed from the text;
* the LLM translator sees every template (key, title, description and spec JSON) in its system
  prompt and starts from the closest one, so common ideas are translated consistently;
* the discovery pipeline de-duplicates newly found ideas against the library.

Template specs use ``name == key`` (so a translation can be traced back to its template), carry
methodology notes in ``assumptions`` and leave run parameters (dates, costs) at the platform
defaults: monthly rebalance, 10 bps one-way costs, the maximum available history, attribution to
the Fama-French 3-factor model.

Data caveats (free edition): prices come with full history, fundamentals are point-in-time from
SEC filings (as of the filing date), but short interest and consensus-estimate features have no
point-in-time history in the free edition - their backtests need institutional data, and the
descriptions below say so.

Every built spec is checked against the default feature catalog by the test suite.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Callable

from aitrading.screen.spec import Condition
from aitrading.strategy.spec import (
    PortfolioConstruction,
    SignalComponent,
    StrategySpec,
    TimeSeriesRule,
)

__all__ = [
    "IdeaTemplate",
    "TEMPLATES",
    "get_template",
    "template_keys",
    "suggest_templates",
    "templates_prompt",
    "INSTITUTIONAL_DATA_NOTE",
]

INSTITUTIONAL_DATA_NOTE = (
    "Free-data caveat: {what} has no point-in-time history in the free edition (only today's snapshot), "
    "so a historical backtest of this idea needs institutional data (e.g. Bloomberg / LSEG / Compustat)."
)


@dataclass
class IdeaTemplate:
    key: str
    title: str
    aliases: list[str]
    description: str
    references: list[str]
    build: Callable[[], StrategySpec]
    needs_institutional_data: bool = field(default=False)

    def spec(self) -> StrategySpec:
        """A fresh spec (templates are immutable recipes; callers may mutate the returned copy)."""
        return self.build()

    def one_line(self) -> str:
        """First sentence of the description (used in prompts and suggestion lists)."""
        text = self.description.strip()
        cut = text.find(". ")
        return text if cut < 0 else text[: cut + 1]

    def prompt_block(self) -> str:
        """Byte-stable rendering for the translator's system prompt."""
        spec_json = json.dumps(
            self.spec().model_dump(mode="json", exclude_defaults=True, exclude={"idea"}),
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"### {self.key} - {self.title}\n{self.one_line()}\nspec: {spec_json}"


# ------------------------------------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------------------------------------


def _sig(feature: str, direction: str = "higher_is_better", weight: float = 1.0) -> SignalComponent:
    return SignalComponent(feature=feature, direction=direction, weight=weight)  # type: ignore[arg-type]


def _cross_sectional(
    key: str,
    title: str,
    signal: list[SignalComponent],
    *,
    style: str = "long_short",
    n_quantiles: int = 5,
    rebalance: str = "monthly",
    assumptions: list[str] | None = None,
    filters: list[Condition] | None = None,
    attribution: str = "ff3",
) -> StrategySpec:
    return StrategySpec(
        name=key,
        idea=title,
        kind="cross_sectional",
        rebalance=rebalance,  # type: ignore[arg-type]
        signal=signal,
        filters=list(filters or []),
        portfolio=PortfolioConstruction(style=style, selection="quantile", n_quantiles=n_quantiles, weighting="equal"),  # type: ignore[arg-type]
        attribution_model=attribution,  # type: ignore[arg-type]
        assumptions=list(assumptions or []),
    )


def _factor_model(key: str, title: str, model: str, assumptions: list[str]) -> StrategySpec:
    return StrategySpec(
        name=key,
        idea=title,
        kind="factor_model",
        rebalance="monthly",
        factor_model=model,  # type: ignore[arg-type]
        attribution_model=model,  # type: ignore[arg-type]
        assumptions=assumptions,
    )


def _timing(key: str, title: str, entry: list[Condition], assumptions: list[str], asset: str = "SPY") -> StrategySpec:
    return StrategySpec(
        name=key,
        idea=title,
        kind="time_series",
        rebalance="monthly",
        time_series=TimeSeriesRule(assets=[asset], entry=entry, when_flat="cash"),
        attribution_model="capm",
        benchmark=asset,
        assumptions=assumptions,
    )


_FF_CONSTRUCTION = (
    "Factors are constructed from the universe with the Fama-French methodology (independent 2x3 "
    "size x characteristic sorts, value-weighted legs, annual June formation, momentum re-formed monthly) "
    "and compared with the official Kenneth French Data Library series."
)
_FF_UNIVERSE = (
    "The universe is the provider's investable list (mostly large caps), so 'Small' is closer to mid cap "
    "than to the CRSP small-cap segment; expect a weaker SMB than the official series."
)


def _capm() -> StrategySpec:
    return _factor_model(
        "capm",
        "CAPM (single-factor market model)",
        "capm",
        [
            "Mkt-RF = value-weighted universe return minus the 1-month T-bill rate.",
            "Compared with the official Kenneth French Mkt-RF series.",
        ],
    )


def _ff3() -> StrategySpec:
    return _factor_model("ff3", "Fama-French 3-factor model", "ff3", [_FF_CONSTRUCTION, _FF_UNIVERSE, "Factors: Mkt-RF, SMB, HML."])


def _carhart4() -> StrategySpec:
    return _factor_model(
        "carhart4",
        "Carhart 4-factor model",
        "carhart4",
        [_FF_CONSTRUCTION, _FF_UNIVERSE, "Factors: Mkt-RF, SMB, HML, Mom (12-1 momentum, re-formed monthly)."],
    )


def _ff5() -> StrategySpec:
    return _factor_model(
        "ff5",
        "Fama-French 5-factor model",
        "ff5",
        [_FF_CONSTRUCTION, _FF_UNIVERSE, "Factors: Mkt-RF, SMB, HML, RMW (operating profitability), CMA (investment)."],
    )


def _momentum_12_1() -> StrategySpec:
    return _cross_sectional(
        "momentum_12_1",
        "12-1 cross-sectional momentum",
        [_sig("return_12m_ex_1m_pct")],
        n_quantiles=10,
        assumptions=[
            "Signal = return from 12 months to 1 month ago (skip the latest month to avoid short-term reversal).",
            "Decile sort, long top decile / short bottom decile, equal weighted (Jegadeesh-Titman style).",
        ],
    )


def _short_term_reversal() -> StrategySpec:
    return _cross_sectional(
        "short_term_reversal",
        "Short-term (1-month) reversal",
        [_sig("return_1m_pct", "lower_is_better")],
        assumptions=["Signal = last month's return; losers are expected to outperform next month."],
    )


def _low_volatility() -> StrategySpec:
    return _cross_sectional(
        "low_volatility",
        "Low volatility anomaly",
        [_sig("volatility_60d_pct", "lower_is_better")],
        style="long_only",
        assumptions=[
            "Signal = 60-day realised volatility (lower is better).",
            "Long-only lowest-volatility quintile, equal weighted (the anomaly is usually harvested long-only).",
        ],
    )


def _low_beta() -> StrategySpec:
    return _cross_sectional(
        "low_beta",
        "Low beta / betting against beta",
        [_sig("beta_1y", "lower_is_better")],
        assumptions=[
            "Signal = 1-year daily beta to the benchmark (lower is better).",
            "Quintile long-short, equal weighted; the leverage that makes Frazzini-Pedersen's BAB beta-neutral is "
            "not replicated, so the long-short portfolio has a negative market beta.",
        ],
    )


def _value_fcf() -> StrategySpec:
    return _cross_sectional(
        "value_fcf",
        "Value (free-cash-flow yield)",
        [_sig("fcf_yield_pct")],
        assumptions=["Signal = trailing-12m free-cash-flow yield (FCF / market cap); point-in-time from filings."],
    )


def _value_composite() -> StrategySpec:
    return _cross_sectional(
        "value_composite",
        "Value composite (FCF yield, earnings yield, EV/EBITDA)",
        [
            _sig("fcf_yield_pct"),
            _sig("earnings_yield_ntm_pct"),
            _sig("ev_to_ebitda", "lower_is_better"),
        ],
        assumptions=["Equal-weighted average of cross-sectional ranks of the three valuation ratios."],
    )


def _quality_signal() -> list[SignalComponent]:
    return [
        _sig("roe_pct"),
        _sig("gross_margin_pct"),
        _sig("fcf_conversion_pct"),
        _sig("net_debt_to_ebitda", "lower_is_better"),
    ]


def _quality() -> StrategySpec:
    return _cross_sectional(
        "quality",
        "Quality (profitability, cash conversion, low leverage)",
        _quality_signal(),
        assumptions=[
            "Quality composite = equal-weighted ranks of ROE, gross margin, FCF conversion and (low) net debt / EBITDA.",
            "Gross margin stands in for Novy-Marx's gross profits / assets (not in the catalog).",
        ],
    )


def _qarp() -> StrategySpec:
    value = [
        _sig("fcf_yield_pct", weight=1.0),
        _sig("earnings_yield_ntm_pct", weight=1.0),
        _sig("ev_to_ebitda", "lower_is_better", weight=1.0),
    ]
    quality = [SignalComponent(feature=s.feature, direction=s.direction, weight=0.75) for s in _quality_signal()]
    return _cross_sectional(
        "qarp",
        "Quality at a reasonable price (QARP)",
        quality + value,
        style="long_only",
        assumptions=[
            "Composite = 50% quality (ROE, gross margin, FCF conversion, low net debt / EBITDA) + 50% value "
            "(FCF yield, NTM earnings yield, low EV/EBITDA); weights 0.75 x 4 and 1.0 x 3.",
            "Long-only top quintile (QARP is a stock-selection approach); use long_short to test it as a factor.",
        ],
    )


def _size() -> StrategySpec:
    return _cross_sectional(
        "size",
        "Size (small minus big)",
        [_sig("market_cap_usd_bn", "lower_is_better")],
        assumptions=[
            "Signal = market capitalisation (smaller is better); quintile long-short, equal weighted.",
            "The investable universe excludes micro caps, where most of the historical size premium lived.",
        ],
    )


def _estimate_revisions() -> StrategySpec:
    return _cross_sectional(
        "estimate_revisions",
        "Analyst earnings-estimate revisions",
        [_sig("eps_revision_3m_pct")],
        assumptions=["Signal = 3-month change in next-12m consensus EPS."],
    )


def _short_interest() -> StrategySpec:
    return _cross_sectional(
        "short_interest",
        "Short interest (avoid heavily shorted stocks)",
        [_sig("short_interest_pct_float", "lower_is_better")],
        assumptions=["Signal = short interest as % of float (lower is better): heavily shorted names are expected to underperform."],
    )


def _rsi_reversal() -> StrategySpec:
    return _cross_sectional(
        "rsi_reversal",
        "RSI oversold reversal",
        [_sig("rsi_14", "lower_is_better")],
        rebalance="weekly",
        assumptions=["Signal = 14-day Wilder RSI (lower = more oversold = better); weekly rebalance for a short-horizon effect."],
    )


def _trend_200dma_spy() -> StrategySpec:
    return _timing(
        "trend_200dma_spy",
        "SPY 200-day moving-average trend filter",
        [Condition(feature="price_vs_sma_200_pct", op=">", value=0.0, rationale="price above its 200-day moving average")],
        [
            "Long SPY while it closes above its 200-day simple moving average, otherwise in cash (T-bills).",
            "Signal checked at each month-end, as in Faber (2007), to limit whipsaw trades.",
            "Benchmark = buy-and-hold SPY; attribution to the CAPM market factor.",
        ],
    )


def _golden_cross_spy() -> StrategySpec:
    return _timing(
        "golden_cross_spy",
        "SPY golden cross (50-day above 200-day)",
        [Condition(feature="sma_50_vs_sma_200_pct", op=">", value=0.0, rationale="50-day moving average above the 200-day")],
        [
            "Long SPY while its 50-day SMA is above its 200-day SMA, otherwise in cash.",
            "Signal checked at each month-end.",
            "Benchmark = buy-and-hold SPY; attribution to the CAPM market factor.",
        ],
    )


def _dislocation_screen() -> StrategySpec:
    def c(feature: str, op: str, value: float, high: float | None = None, why: str = "") -> Condition:
        return Condition(feature=feature, op=op, value=value, value_high=high, rationale=why)  # type: ignore[arg-type]

    return StrategySpec(
        name="dislocation_screen",
        idea="Quality mid caps in a long-term uptrend after a sharp, high-volume sell-off",
        kind="screen",
        rebalance="monthly",
        filters=[
            c("market_cap_usd_bn", "between", 2, 20, "mid caps"),
            c("sma_50_vs_sma_200_pct", ">", 0, why="long-term uptrend intact (50-day above 200-day)"),
            c("return_12m_ex_1m_pct", ">", 0, why="positive 12-1 momentum"),
            c("drawdown_from_52w_high_pct", "between", -40, -15, "sharp pullback from the 52-week high"),
            c("max_volume_ratio_20d", ">=", 2, why="capitulation / event-day volume"),
            c("rsi_14", "<", 40, why="oversold"),
            c("fcf_yield_pct", ">", 4, why="cash-generative and cheap"),
            c("revenue_growth_yoy_pct", ">", 8, why="fundamentals still growing"),
            c("short_interest_pct_float", ">", 6, why="crowded short positioning"),
        ],
        portfolio=PortfolioConstruction(style="long_only", selection="quantile", weighting="equal"),
        assumptions=[
            "Hold every name passing all nine conditions, equal weighted, re-screened monthly (cash when nothing passes).",
        ],
    )


TEMPLATES: dict[str, IdeaTemplate] = {
    t.key: t
    for t in [
        IdeaTemplate(
            key="capm",
            title="CAPM (single-factor market model)",
            aliases=["capm", "capital asset pricing model", "single factor model", "one factor model", "market model", "market factor"],
            description=(
                "Build the market factor (Mkt-RF) from the universe and compare it with the official series. "
                "The baseline every other strategy's alpha is measured against."
            ),
            references=[
                "Sharpe (1964), Capital asset prices: a theory of market equilibrium under conditions of risk, JF 19.",
                "Lintner (1965), The valuation of risk assets and the selection of risky investments, REStat 47.",
            ],
            build=_capm,
        ),
        IdeaTemplate(
            key="ff3",
            title="Fama-French 3-factor model",
            aliases=["fama french 3 factor model", "fama-french three-factor model", "three factor model", "3 factor model", "ff3", "smb hml"],
            description=(
                "Build Mkt-RF, SMB (size) and HML (value) from the universe with the Fama-French 2x3 sorts and compare "
                "them with the official Kenneth French factors. Book equity comes from SEC filings (point-in-time by "
                "filing date), not Compustat, so expect differences from the official HML."
            ),
            references=["Fama & French (1993), Common risk factors in the returns on stocks and bonds, JFE 33."],
            build=_ff3,
        ),
        IdeaTemplate(
            key="carhart4",
            title="Carhart 4-factor model",
            aliases=["carhart 4 factor model", "carhart four-factor model", "four factor model", "4 factor model", "ff3 plus momentum", "umd"],
            description=(
                "Fama-French 3 factors plus the momentum factor (Mom / UMD), built from the universe and compared with "
                "the official series."
            ),
            references=["Carhart (1997), On persistence in mutual fund performance, JF 52."],
            build=_carhart4,
        ),
        IdeaTemplate(
            key="ff5",
            title="Fama-French 5-factor model",
            aliases=["fama french 5 factor model", "fama-french five-factor model", "five factor model", "5 factor model", "ff5", "rmw cma"],
            description=(
                "Mkt-RF, SMB, HML plus RMW (profitability) and CMA (investment), built from the universe and compared "
                "with the official Kenneth French 2x3 five-factor series."
            ),
            references=["Fama & French (2015), A five-factor asset pricing model, JFE 116."],
            build=_ff5,
        ),
        IdeaTemplate(
            key="momentum_12_1",
            title="12-1 cross-sectional momentum",
            aliases=["momentum", "12-1 momentum", "price momentum", "cross-sectional momentum", "winners minus losers", "relative strength"],
            description=(
                "Rank stocks on their return from 12 months to 1 month ago and go long the top decile, short the bottom "
                "decile, rebalanced monthly. Price data only, so it is fully testable with free data."
            ),
            references=[
                "Jegadeesh & Titman (1993), Returns to buying winners and selling losers, JF 48.",
                "Asness, Moskowitz & Pedersen (2013), Value and momentum everywhere, JF 68.",
            ],
            build=_momentum_12_1,
        ),
        IdeaTemplate(
            key="short_term_reversal",
            title="Short-term (1-month) reversal",
            aliases=["short-term reversal", "short term reversal", "1-month reversal", "monthly reversal", "reversal"],
            description=(
                "Buy last month's losers and sell last month's winners, rebalanced monthly. Highly sensitive to "
                "transaction costs because turnover is very high."
            ),
            references=[
                "Jegadeesh (1990), Evidence of predictable behavior of security returns, JF 45.",
                "Lehmann (1990), Fads, martingales, and market efficiency, QJE 105.",
            ],
            build=_short_term_reversal,
        ),
        IdeaTemplate(
            key="low_volatility",
            title="Low volatility anomaly",
            aliases=["low volatility", "low vol", "low-volatility anomaly", "minimum volatility", "min vol", "low risk anomaly"],
            description=(
                "Hold the lowest-volatility quintile (60-day realised volatility), long only, rebalanced monthly. "
                "Price data only, so it is fully testable with free data."
            ),
            references=[
                "Ang, Hodrick, Xing & Zhang (2006), The cross-section of volatility and expected returns, JF 61.",
                "Baker, Bradley & Wurgler (2011), Benchmarks as limits to arbitrage: understanding the low-volatility anomaly, FAJ 67.",
            ],
            build=_low_volatility,
        ),
        IdeaTemplate(
            key="low_beta",
            title="Low beta / betting against beta",
            aliases=["low beta", "betting against beta", "bab", "low-beta anomaly"],
            description=(
                "Long low-beta, short high-beta stocks (quintiles of 1-year beta), rebalanced monthly. The BAB leverage "
                "to beta neutrality is not replicated."
            ),
            references=["Frazzini & Pedersen (2014), Betting against beta, JFE 111."],
            build=_low_beta,
        ),
        IdeaTemplate(
            key="value_fcf",
            title="Value (free-cash-flow yield)",
            aliases=["value", "value factor", "fcf yield", "free cash flow yield", "cheap stocks", "high fcf yield"],
            description=(
                "Long the highest free-cash-flow-yield quintile, short the lowest, rebalanced monthly. Uses filings-based "
                "fundamentals that are point-in-time by filing date."
            ),
            references=[
                "Fama & French (1992), The cross-section of expected stock returns, JF 47.",
                "Lakonishok, Shleifer & Vishny (1994), Contrarian investment, extrapolation, and risk, JF 49.",
            ],
            build=_value_fcf,
        ),
        IdeaTemplate(
            key="value_composite",
            title="Value composite (FCF yield, earnings yield, EV/EBITDA)",
            aliases=["value composite", "composite value", "multi-factor value", "blended value", "ev/ebitda value"],
            description=(
                "Rank stocks on an equal blend of FCF yield, NTM earnings yield and (low) EV/EBITDA; quintile long-short. "
                + INSTITUTIONAL_DATA_NOTE.format(what="the NTM consensus earnings yield")
            ),
            references=[
                "Asness, Moskowitz & Pedersen (2013), Value and momentum everywhere, JF 68.",
                "Loughran & Wellman (2011), New evidence on the relation between the enterprise multiple and average stock returns, JFQA 46.",
            ],
            build=_value_composite,
            needs_institutional_data=True,
        ),
        IdeaTemplate(
            key="quality",
            title="Quality (profitability, cash conversion, low leverage)",
            aliases=["quality", "quality factor", "quality minus junk", "qmj", "profitability", "high quality stocks"],
            description=(
                "Rank stocks on ROE, gross margin, FCF conversion and low leverage; quintile long-short. Uses filings-based "
                "fundamentals that are point-in-time by filing date."
            ),
            references=[
                "Asness, Frazzini & Pedersen (2019), Quality minus junk, Review of Accounting Studies 24.",
                "Novy-Marx (2013), The other side of value: the gross profitability premium, JFE 108.",
            ],
            build=_quality,
        ),
        IdeaTemplate(
            key="qarp",
            title="Quality at a reasonable price (QARP)",
            aliases=["quality at a reasonable price", "qarp", "quality value", "quality and value", "cheap quality"],
            description=(
                "Long-only top quintile of a 50/50 quality + value composite, rebalanced monthly. "
                + INSTITUTIONAL_DATA_NOTE.format(what="the NTM consensus earnings yield")
            ),
            references=[
                "Asness, Frazzini & Pedersen (2019), Quality minus junk, Review of Accounting Studies 24.",
                "Novy-Marx (2013), The other side of value: the gross profitability premium, JFE 108.",
            ],
            build=_qarp,
            needs_institutional_data=True,
        ),
        IdeaTemplate(
            key="size",
            title="Size (small minus big)",
            aliases=["size", "size factor", "small cap premium", "small minus big", "small caps outperform"],
            description=(
                "Long the smallest-market-cap quintile, short the largest, rebalanced monthly. The investable universe "
                "excludes micro caps, so this is a weak test of the classic size effect."
            ),
            references=[
                "Banz (1981), The relationship between return and market value of common stocks, JFE 9.",
                "Fama & French (1993), Common risk factors in the returns on stocks and bonds, JFE 33.",
            ],
            build=_size,
        ),
        IdeaTemplate(
            key="estimate_revisions",
            title="Analyst earnings-estimate revisions",
            aliases=["estimate revisions", "earnings revisions", "eps revisions", "analyst revisions", "earnings momentum"],
            description=(
                "Long stocks with the largest 3-month upward revisions to NTM consensus EPS, short the largest downward "
                "revisions; quintile long-short, monthly. "
                + INSTITUTIONAL_DATA_NOTE.format(what="consensus EPS")
            ),
            references=[
                "Chan, Jegadeesh & Lakonishok (1996), Momentum strategies, JF 51.",
                "Hawkins, Chamberlin & Daniel (1984), Earnings expectations and security prices, FAJ 40.",
            ],
            build=_estimate_revisions,
            needs_institutional_data=True,
        ),
        IdeaTemplate(
            key="short_interest",
            title="Short interest (avoid heavily shorted stocks)",
            aliases=["short interest", "heavily shorted stocks", "short sellers", "low short interest"],
            description=(
                "Long the least-shorted quintile, short the most-shorted, rebalanced monthly. "
                + INSTITUTIONAL_DATA_NOTE.format(what="short interest")
            ),
            references=[
                "Asquith, Pathak & Ritter (2005), Short interest, institutional ownership, and stock returns, JFE 78.",
                "Boehmer, Jones & Zhang (2008), Which shorts are informed?, JF 63.",
            ],
            build=_short_interest,
            needs_institutional_data=True,
        ),
        IdeaTemplate(
            key="rsi_reversal",
            title="RSI oversold reversal",
            aliases=["rsi reversal", "rsi oversold", "oversold stocks", "rsi mean reversion", "buy oversold"],
            description=(
                "Rank stocks on 14-day RSI (most oversold first), quintile long-short, rebalanced weekly. Price data only; "
                "very high turnover, so costs matter."
            ),
            references=["Wilder (1978), New Concepts in Technical Trading Systems."],
            build=_rsi_reversal,
        ),
        IdeaTemplate(
            key="trend_200dma_spy",
            title="SPY 200-day moving-average trend filter",
            aliases=["200 day moving average", "200-day trend filter", "spy above 200 day", "market timing", "faber trend following", "10 month sma timing"],
            description=(
                "Long SPY when it is above its 200-day moving average, otherwise cash, checked monthly. "
                "Price data only, so it is fully testable with free data."
            ),
            references=["Faber (2007), A quantitative approach to tactical asset allocation, Journal of Wealth Management 9."],
            build=_trend_200dma_spy,
        ),
        IdeaTemplate(
            key="golden_cross_spy",
            title="SPY golden cross (50-day above 200-day)",
            aliases=["golden cross", "death cross", "50 day above 200 day", "moving average crossover", "dual moving average"],
            description=(
                "Long SPY while its 50-day moving average is above its 200-day, otherwise cash, checked monthly. "
                "Price data only."
            ),
            references=["Brock, Lakonishok & LeBaron (1992), Simple technical trading rules and the stochastic properties of stock returns, JF 47."],
            build=_golden_cross_spy,
        ),
        IdeaTemplate(
            key="dislocation_screen",
            title="Dislocation screen (quality mid caps after a high-volume sell-off)",
            aliases=["dislocation screen", "dislocation", "oversold quality mid caps", "capitulation screen", "buy the dip in uptrends"],
            description=(
                "Equal-weight every mid cap in a long-term uptrend that sold off 15-40% on capitulation volume while "
                "cash-generative, growing and heavily shorted; re-screened monthly. "
                + INSTITUTIONAL_DATA_NOTE.format(what="short interest")
            ),
            references=[
                "De Bondt & Thaler (1985), Does the stock market overreact?, JF 40.",
                "Jegadeesh & Titman (1993), Returns to buying winners and selling losers, JF 48.",
            ],
            build=_dislocation_screen,
            needs_institutional_data=True,
        ),
    ]
}


def template_keys() -> list[str]:
    return list(TEMPLATES)


def get_template(key: str) -> IdeaTemplate:
    try:
        return TEMPLATES[key]
    except KeyError:
        raise KeyError(f"unknown idea template '{key}'; known: {', '.join(TEMPLATES)}") from None


def suggest_templates(text: str, n: int = 5) -> list[str]:
    """Titles of the ``n`` templates closest to ``text`` (difflib ratio over titles, keys and aliases).

    Deterministic: ties are broken by library order.
    """
    q = " ".join(text.lower().split())
    scored: list[tuple[float, int, str]] = []
    for i, t in enumerate(TEMPLATES.values()):
        names = [t.title, t.key.replace("_", " "), *t.aliases]
        score = max(difflib.SequenceMatcher(None, q, s.lower()).ratio() for s in names)
        scored.append((-score, i, t.title))
    scored.sort()
    return [title for _, _, title in scored[: max(0, n)]]


def templates_prompt() -> str:
    """All templates rendered for an LLM prompt, in library order (byte-stable)."""
    return "\n\n".join(t.prompt_block() for t in TEMPLATES.values())
