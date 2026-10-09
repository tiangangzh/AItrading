"""What a strategy *claims*: hypothesis identity, plain-English statement and a similarity geometry.

A backtest tests a hypothesis ("stocks with the strongest 3-day returns keep outperforming"), not a
spec file. This module maps a :class:`~aitrading.strategy.spec.StrategySpec` to

* :func:`hypothesis_key` - the identity of the economic claim. Two specs that differ only in how
  the claim is implemented or measured (quintiles vs deciles, weighting, costs, dates, attribution
  model, benchmark, name) are the SAME hypothesis; their results are correlated measurements of one
  effect. Signal features and directions, filters, the timing rule, the universe, the rebalance
  frequency and long-only vs long-short define it.
* :func:`hypothesis_statement` - the claim in words, so people and language models reason about the
  idea, not about JSON.
* :func:`similarity` - a positive semi-definite kernel in [-1, 1] over hypotheses: how much
  evidence about one should move beliefs about the other. It is what lets "3-day momentum works"
  raise the prior on 5-day momentum, lower it on 3-day reversal (the opposite claim: negative
  similarity), and barely touch 12-1 momentum.

Geometry
--------
Every catalog feature has a *family* (the economic concept: past returns, value, profitability,
risk, ...), a sign that orients it inside the family (``ev_to_ebitda`` is value with sign -1: a high
multiple is expensive) and, for families where it matters, a horizon in trading sessions. A signal
component (feature, direction, weight) is embedded as

* a family part (share ``FAMILY_SHARE[family]``): the signed weight spread over a log-spaced
  horizon grid with a Gaussian of width ``HORIZON_BANDWIDTH`` (in log-horizon units), or a single
  coordinate for families without a horizon;
* a feature-specific part (the rest): the signed weight on the feature itself.

The signal embedding is the sum over components, normalised; the cosine of two embeddings is a PSD
kernel. With the defaults (same kind, style and rebalance), 3- vs 5-day momentum have similarity
~0.7, 3- vs 10-day ~0.3, 1-month vs 12-1 month ~0, 6-month vs 12-1 month ~0.6, two different value
ratios 0.5, and momentum vs reversal at the same horizon -1. The full kernel multiplies it by PSD factors for
the strategy kind, long-only vs long-short, the universe, the rebalance frequency, the filters and
the traded assets of timing rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from aitrading.strategy.spec import StrategySpec

__all__ = [
    "FeatureSemantics",
    "FEATURE_SEMANTICS",
    "FAMILY_LABELS",
    "HypothesisShape",
    "hypothesis_key",
    "hypothesis_id",
    "hypothesis_shape",
    "hypothesis_statement",
    "primary_concept",
    "similarity",
    "spec_from_any",
]


@dataclass(frozen=True)
class FeatureSemantics:
    family: str
    sign: float = 1.0  # orientation inside the family (+1: higher = more of the family's concept)
    horizon: float | None = None  # trading sessions, for families where the look-back matters
    label: str = ""


def _fs(family: str, sign: float = 1.0, horizon: float | None = None, label: str = "") -> FeatureSemantics:
    return FeatureSemantics(family, sign, horizon, label)


# Families: past_return (continuation when higher is better, reversal when lower), trend, risk,
# value, growth, profitability, investment, balance_sheet, earnings_momentum, size, liquidity,
# attention (volume), options_sentiment, short_interest, analyst, event_timing.
FEATURE_SEMANTICS: dict[str, FeatureSemantics] = {
    # past returns (horizon = look-back in sessions)
    "return_3d_pct": _fs("past_return", 1, 3, "3-day return"),
    "return_5d_pct": _fs("past_return", 1, 5, "1-week return"),
    "return_10d_pct": _fs("past_return", 1, 10, "2-week return"),
    "return_1m_pct": _fs("past_return", 1, 21, "1-month return"),
    "return_3m_pct": _fs("past_return", 1, 63, "3-month return"),
    "return_6m_pct": _fs("past_return", 1, 126, "6-month return"),
    "return_12m_pct": _fs("past_return", 1, 252, "12-month return"),
    "return_12m_ex_1m_pct": _fs("past_return", 1, 231, "12-1 month return"),
    "rel_strength_3m_pp": _fs("past_return", 1, 63, "3-month relative strength"),
    "rel_strength_6m_pp": _fs("past_return", 1, 126, "6-month relative strength"),
    "rel_strength_12m_pp": _fs("past_return", 1, 252, "12-month relative strength"),
    "return_6m_percentile": _fs("past_return", 1, 126, "6-month return percentile"),
    "rsi_14": _fs("past_return", 1, 14, "14-day RSI"),
    "drawdown_from_52w_high_pct": _fs("past_return", 1, 252, "distance from the 52-week high"),
    "above_52w_low_pct": _fs("past_return", 1, 252, "distance above the 52-week low"),
    "days_since_52w_high": _fs("past_return", -1, 252, "days since the 52-week high"),
    # trend / moving-average structure
    "price_vs_sma_50_pct": _fs("trend", 1, 50, "price vs 50-day average"),
    "price_vs_sma_200_pct": _fs("trend", 1, 200, "price vs 200-day average"),
    "sma_50_vs_sma_200_pct": _fs("trend", 1, 125, "50-day vs 200-day average"),
    "sma_200_slope_1m_pct": _fs("trend", 1, 200, "200-day average slope"),
    "golden_cross_20d": _fs("trend", 1, 125, "recent golden cross"),
    "macd_line": _fs("trend", 1, 26, "MACD line"),
    "macd_signal": _fs("trend", 1, 34, "MACD signal"),
    "macd_histogram": _fs("trend", 1, 26, "MACD histogram"),
    "macd_histogram_pct_price": _fs("trend", 1, 26, "MACD histogram / price"),
    "macd_bullish_cross_10d": _fs("trend", 1, 26, "recent MACD bullish cross"),
    "sma_20": _fs("price_level", 1, None, "20-day average price"),
    "sma_50": _fs("price_level", 1, None, "50-day average price"),
    "sma_200": _fs("price_level", 1, None, "200-day average price"),
    "high_52w": _fs("price_level", 1, None, "52-week high"),
    "low_52w": _fs("price_level", 1, None, "52-week low"),
    "price": _fs("price_level", 1, None, "share price"),
    # risk (higher = riskier)
    "volatility_20d_pct": _fs("risk", 1, 20, "20-day volatility"),
    "volatility_60d_pct": _fs("risk", 1, 60, "60-day volatility"),
    "atr_14_pct": _fs("risk", 1, 14, "average true range"),
    "beta_1y": _fs("risk", 1, 252, "market beta"),
    "iv_30d_pct": _fs("risk", 1, 21, "implied volatility"),
    # size / liquidity / attention
    "market_cap_usd_bn": _fs("size", 1, None, "market capitalisation"),
    "enterprise_value_usd_bn": _fs("size", 1, None, "enterprise value"),
    "avg_dollar_volume_20d_usd_mn": _fs("liquidity", 1, None, "dollar volume"),
    "rel_volume_5d": _fs("attention", 1, None, "5-day relative volume"),
    "rel_volume_20d": _fs("attention", 1, None, "20-day relative volume"),
    "max_volume_ratio_20d": _fs("attention", 1, None, "volume spike"),
    "up_down_volume_ratio_50d": _fs("attention", 1, None, "up/down volume ratio"),
    # options / short interest
    "iv_rank_1y": _fs("options_sentiment", 1, None, "IV rank"),
    "iv_to_realized_vol_ratio": _fs("options_sentiment", 1, None, "implied vs realised volatility"),
    "put_call_volume_ratio": _fs("options_sentiment", 1, None, "put/call volume"),
    "put_call_oi_ratio": _fs("options_sentiment", 1, None, "put/call open interest"),
    "short_interest_pct_float": _fs("short_interest", 1, None, "short interest"),
    "days_to_cover": _fs("short_interest", 1, None, "days to cover"),
    "short_interest_change_1m_pct": _fs("short_interest", 1, None, "change in short interest"),
    # value (higher = cheaper)
    "fcf_yield_pct": _fs("value", 1, None, "free-cash-flow yield"),
    "earnings_yield_ntm_pct": _fs("value", 1, None, "forward earnings yield"),
    "earnings_yield_ttm_pct": _fs("value", 1, None, "trailing earnings yield"),
    "book_to_market": _fs("value", 1, None, "book-to-market"),
    "ev_to_ebitda": _fs("value", -1, None, "EV/EBITDA"),
    "ev_to_sales": _fs("value", -1, None, "EV/sales"),
    "pe_ntm": _fs("value", -1, None, "forward P/E"),
    "cash_pct_market_cap": _fs("value", 1, None, "cash / market cap"),
    "target_price_upside_pct": _fs("analyst", 1, None, "analyst target upside"),
    "num_analysts": _fs("analyst", 1, None, "analyst coverage"),
    # growth
    "revenue_growth_yoy_pct": _fs("growth", 1, None, "revenue growth"),
    "revenue_growth_last_q_yoy_pct": _fs("growth", 1, None, "latest-quarter revenue growth"),
    "revenue_growth_ntm_est_pct": _fs("growth", 1, None, "expected revenue growth"),
    "eps_growth_ntm_est_pct": _fs("growth", 1, None, "expected EPS growth"),
    # profitability / quality
    "gross_margin_pct": _fs("profitability", 1, None, "gross margin"),
    "operating_margin_pct": _fs("profitability", 1, None, "operating margin"),
    "ebitda_margin_pct": _fs("profitability", 1, None, "EBITDA margin"),
    "fcf_margin_pct": _fs("profitability", 1, None, "FCF margin"),
    "net_margin_pct": _fs("profitability", 1, None, "net margin"),
    "roe_pct": _fs("profitability", 1, None, "return on equity"),
    "operating_profitability_pct": _fs("profitability", 1, None, "operating profitability"),
    "gross_profitability_pct": _fs("profitability", 1, None, "gross profitability"),
    "fcf_conversion_pct": _fs("profitability", 1, None, "FCF conversion"),
    "gross_margin_change_yoy_pp": _fs("profitability", 1, None, "gross-margin change"),
    "operating_margin_change_yoy_pp": _fs("profitability", 1, None, "operating-margin change"),
    # investment / balance sheet
    "asset_growth_yoy_pct": _fs("investment", 1, None, "asset growth"),
    "net_debt_usd_bn": _fs("balance_sheet", -1, None, "net debt"),
    "net_debt_to_ebitda": _fs("balance_sheet", -1, None, "net debt / EBITDA"),
    "interest_coverage": _fs("balance_sheet", 1, None, "interest coverage"),
    # estimates / events
    "eps_revision_3m_pct": _fs("earnings_momentum", 1, None, "EPS estimate revisions"),
    "revenue_revision_3m_pct": _fs("earnings_momentum", 1, None, "revenue estimate revisions"),
    "last_eps_surprise_pct": _fs("earnings_momentum", 1, None, "earnings surprise"),
    "days_since_last_earnings": _fs("event_timing", 1, None, "days since earnings"),
    "days_to_next_earnings": _fs("event_timing", 1, None, "days to earnings"),
}

#: How the family reads with sign +1 / -1 (used in statements and lesson titles).
FAMILY_LABELS: dict[str, tuple[str, str]] = {
    "past_return": ("momentum (past winners keep winning)", "reversal (past losers rebound)"),
    "trend": ("trend following", "counter-trend"),
    "risk": ("high risk earns more", "low-risk anomaly (calmer stocks earn more per unit of risk)"),
    "value": ("value (cheap beats expensive)", "glamour (expensive beats cheap)"),
    "growth": ("growth (faster growers outperform)", "slow growers outperform"),
    "profitability": ("quality / profitability", "unprofitable firms outperform"),
    "investment": ("aggressive investors outperform", "conservative investment (low asset growth) outperforms"),
    "balance_sheet": ("strong balance sheets outperform", "levered firms outperform"),
    "earnings_momentum": ("earnings momentum (upgrades, beats)", "earnings disappointment rebounds"),
    "size": ("large caps outperform", "size effect (small caps outperform)"),
    "liquidity": ("liquid stocks outperform", "illiquidity premium"),
    "attention": ("high-attention stocks outperform", "neglected stocks outperform"),
    "options_sentiment": ("option-market fear predicts gains", "option-market fear predicts losses"),
    "short_interest": ("heavily shorted stocks outperform", "heavily shorted stocks underperform"),
    "analyst": ("analyst optimism pays", "analyst optimism is priced in"),
    "event_timing": ("event timing (later)", "event timing (earlier)"),
    "price_level": ("high-priced stocks outperform", "low-priced stocks outperform"),
}

#: Share of a component's embedding on its family (the rest is feature-specific).
FAMILY_SHARE: dict[str, float] = {"past_return": 0.8, "trend": 0.7, "risk": 0.6}
DEFAULT_FAMILY_SHARE = 0.5
HORIZON_BANDWIDTH = 0.6  # log-horizon units (Gaussian width of the horizon bumps)
_HORIZON_GRID = tuple(1.3 ** k for k in range(0, 25))  # 1 ... ~540 sessions

# Kind-level similarity (PSD: verified in the tests).
_KIND_SIM = {
    ("cross_sectional", "screen"): 0.5,
    ("cross_sectional", "time_series"): 0.25,
    ("cross_sectional", "factor_model"): 0.5,
    ("screen", "time_series"): 0.25,
    ("screen", "factor_model"): 0.25,
    ("time_series", "factor_model"): 0.1,
}
_REBALANCE_DAYS = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63, "annual": 252}
# Factor-model factors as (feature-like key, family, sign, horizon) for the embedding.
_FACTORS = {
    "capm": [("factor:mkt", "market", 1.0, None)],
    "ff3": [("factor:smb", "size", -1.0, None), ("factor:hml", "value", 1.0, None)],
    "carhart4": [("factor:smb", "size", -1.0, None), ("factor:hml", "value", 1.0, None),
                 ("factor:mom", "past_return", 1.0, 231.0)],
    "ff5": [("factor:smb", "size", -1.0, None), ("factor:hml", "value", 1.0, None),
            ("factor:rmw", "profitability", 1.0, None), ("factor:cma", "investment", -1.0, None)],
}


def _semantics(feature: str) -> FeatureSemantics:
    return FEATURE_SEMANTICS.get(feature) or FeatureSemantics(f"feature:{feature}", 1.0, None, feature.replace("_", " "))


def spec_from_any(spec: StrategySpec | dict[str, Any]) -> StrategySpec:
    return spec if isinstance(spec, StrategySpec) else StrategySpec.model_validate(spec)


# ------------------------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------------------------


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def hypothesis_key(spec: StrategySpec | dict[str, Any]) -> dict[str, Any]:
    """The fields that define the claim (see the module docstring); JSON-serialisable and canonical."""
    s = spec_from_any(spec)
    total = sum(c.weight for c in s.signal) or 1.0
    signal = sorted(
        [c.feature, c.direction, round(c.weight / total, 4), bool(c.sector_neutral)] for c in s.signal
    )
    key: dict[str, Any] = {
        "kind": s.kind,
        "signal": signal,
        "filters": sorted(c.describe() for c in s.filters),
        "universe": s.universe.model_dump(mode="json"),
        "rebalance": s.rebalance,
        "style": s.portfolio.style if s.kind in ("cross_sectional", "screen") else None,
    }
    if s.kind == "factor_model":
        key["factor_model"] = s.factor_model
        key["style"] = None
    if s.time_series is not None and s.kind == "time_series":
        ts = s.time_series
        key["time_series"] = {
            "assets": sorted(a.upper() for a in ts.assets),
            "entry": sorted(c.describe() for c in ts.entry),
            "exit": sorted(c.describe() for c in ts.exit),
            "when_flat": ts.when_flat,
        }
    return key


def hypothesis_id(spec: StrategySpec | dict[str, Any]) -> str:
    """Short stable id of the claim: ``h-`` + 12 hex characters of the canonical key's SHA-1."""
    raw = json.dumps(hypothesis_key(spec), sort_keys=True, separators=(",", ":"))
    return "h-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------
# Shape (embedding) and similarity
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisShape:
    """Everything :func:`similarity` needs, precomputed (hashable, so kernels can be cached)."""

    kind: str
    style: str | None
    rebalance: str
    universe: str
    signal: tuple[tuple[str, float], ...]  # normalised sparse embedding (coordinate, value)
    filters: frozenset[str]
    assets: frozenset[str]


def _component(vec: dict[str, float], feature: str, family: str, sign: float, horizon: float | None, weight: float) -> None:
    share = FAMILY_SHARE.get(family, DEFAULT_FAMILY_SHARE)
    fam_w, feat_w = math.sqrt(share) * weight * sign, math.sqrt(1.0 - share) * weight * sign
    if horizon is None:
        vec[f"fam:{family}"] = vec.get(f"fam:{family}", 0.0) + fam_w
    else:
        lh = math.log(max(horizon, 1.0))
        bumps = [math.exp(-((lh - math.log(g)) ** 2) / (2 * HORIZON_BANDWIDTH ** 2)) for g in _HORIZON_GRID]
        norm = math.sqrt(sum(b * b for b in bumps)) or 1.0
        for g, b in zip(_HORIZON_GRID, bumps):
            if b / norm > 1e-4:
                k = f"fam:{family}@{g:.3f}"
                vec[k] = vec.get(k, 0.0) + fam_w * b / norm
    vec[f"feat:{feature}"] = vec.get(f"feat:{feature}", 0.0) + feat_w


def _condition_sign(op: str) -> float:
    return -1.0 if op in ("<", "<=") else 1.0


def hypothesis_shape(spec: StrategySpec | dict[str, Any]) -> HypothesisShape:
    s = spec_from_any(spec)
    vec: dict[str, float] = {}
    if s.kind == "cross_sectional":
        total = sum(c.weight for c in s.signal) or 1.0
        for c in s.signal:
            sem = _semantics(c.feature)
            d = 1.0 if c.direction == "higher_is_better" else -1.0
            _component(vec, c.feature, sem.family, sem.sign * d, sem.horizon, c.weight / total)
    elif s.kind == "screen":
        conds = [c for c in s.filters if c.op not in ("in", "not_in", "==", "!=", "between")]
        for c in conds or s.filters:
            sem = _semantics(c.feature)
            _component(vec, c.feature, sem.family, sem.sign * _condition_sign(c.op), sem.horizon, 1.0 / max(len(conds), 1))
    elif s.kind == "time_series" and s.time_series is not None:
        conds = s.time_series.entry
        for c in conds:
            sem = _semantics(c.feature)
            _component(vec, f"ts:{c.feature}", "timing:" + sem.family, sem.sign * _condition_sign(c.op), sem.horizon,
                       1.0 / max(len(conds), 1))
    elif s.kind == "factor_model" and s.factor_model:
        facs = _FACTORS.get(s.factor_model, [])
        for key, fam, sign, hor in facs:
            _component(vec, key, fam, sign, hor, 1.0 / max(len(facs), 1))
    norm = math.sqrt(sum(v * v for v in vec.values()))
    signal = tuple(sorted((k, v / norm) for k, v in vec.items() if v != 0.0)) if norm > 0 else ()
    if s.kind in ("cross_sectional",):
        filters = frozenset(c.describe() for c in s.filters)
    else:
        filters = frozenset()
    assets = frozenset(a.upper() for a in (s.time_series.assets if s.time_series and s.kind == "time_series" else []))
    return HypothesisShape(
        kind=s.kind,
        style=s.portfolio.style if s.kind in ("cross_sectional", "screen") else None,
        rebalance=s.rebalance,
        universe=json.dumps(s.universe.model_dump(mode="json"), sort_keys=True),
        signal=signal,
        filters=filters,
        assets=assets,
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@lru_cache(maxsize=200_000)
def similarity(a: HypothesisShape, b: HypothesisShape) -> float:
    """PSD similarity kernel in [-1, 1] (1 = the same claim, negative = opposite claims)."""
    if a == b:
        return 1.0
    da = dict(a.signal)
    cos = sum(v * da.get(k, 0.0) for k, v in b.signal)
    if cos == 0.0:
        return 0.0
    if a.kind == b.kind:
        kind = 1.0
    else:
        kind = _KIND_SIM.get((a.kind, b.kind)) or _KIND_SIM.get((b.kind, a.kind)) or 0.1
    style = 1.0 if a.style == b.style else 0.7
    universe = 1.0 if a.universe == b.universe else 0.85
    rebalance = 1.0
    if a.rebalance != b.rebalance:
        gap = abs(math.log(_REBALANCE_DAYS.get(a.rebalance, 21) / _REBALANCE_DAYS.get(b.rebalance, 21)))
        rebalance = 0.6 + 0.4 * math.exp(-gap)
    filters = 0.8 + 0.2 * _jaccard(a.filters, b.filters)
    assets = 0.6 + 0.4 * _jaccard(a.assets, b.assets)
    return max(-1.0, min(1.0, cos * kind * style * universe * rebalance * filters * assets))


# ------------------------------------------------------------------------------------------------
# Words
# ------------------------------------------------------------------------------------------------


def primary_concept(spec: StrategySpec | dict[str, Any]) -> tuple[str, float, float | None]:
    """(family, oriented sign, horizon) of the component that carries the most weight."""
    s = spec_from_any(spec)
    comps: list[tuple[float, str, float, float | None]] = []
    if s.kind == "cross_sectional":
        for c in s.signal:
            sem = _semantics(c.feature)
            comps.append((c.weight, sem.family, sem.sign * (1.0 if c.direction == "higher_is_better" else -1.0), sem.horizon))
    elif s.kind == "screen":
        for c in s.filters:
            sem = _semantics(c.feature)
            comps.append((1.0, sem.family, sem.sign * _condition_sign(c.op), sem.horizon))
    elif s.kind == "time_series" and s.time_series is not None:
        for c in s.time_series.entry:
            sem = _semantics(c.feature)
            comps.append((1.0, "timing:" + sem.family, sem.sign * _condition_sign(c.op), sem.horizon))
    elif s.kind == "factor_model" and s.factor_model:
        return f"factor_model:{s.factor_model}", 1.0, None
    if not comps:
        return "unknown", 1.0, None
    _, fam, sign, hor = max(comps, key=lambda c: c[0])
    return fam, sign, hor


def _horizon_words(h: float | None) -> str:
    if h is None:
        return ""
    if h <= 15:
        return f"{h:g}-day"
    months = h / 21.0
    if abs(h - 231) < 1:
        return "12-1 month"
    return f"{months:.0f}-month" if months < 11.5 else f"{months / 12:.0f}-year"


def _feature_label(feature: str) -> str:
    return _semantics(feature).label or feature.replace("_", " ")


def hypothesis_statement(spec: StrategySpec | dict[str, Any]) -> str:
    """The claim in one plain-English sentence (deterministic)."""
    s = spec_from_any(spec)
    when = {"daily": "each day", "weekly": "each week", "monthly": "each month", "quarterly": "each quarter",
            "annual": "each year"}[s.rebalance]
    if s.kind == "cross_sectional" and s.signal:
        parts = []
        for c in s.signal:
            hi = c.direction == "higher_is_better"
            parts.append(f"{'high' if hi else 'low'} {_feature_label(c.feature)}")
        fam, sign, hor = primary_concept(s)
        label = FAMILY_LABELS.get(fam, (fam, fam))[0 if sign >= 0 else 1]
        hz = _horizon_words(hor)
        concept = f"{hz} {label}".strip() if fam == "past_return" and hz else label
        side = ("buying the best-ranked stocks" if s.portfolio.style == "long_only"
                else "buying the best-ranked and shorting the worst-ranked stocks")
        filt = f" among stocks with {', '.join(c.describe() for c in s.filters)}" if s.filters else ""
        return (f"Stocks with {' and '.join(parts)} outperform{filt}: {side}, rebalanced {when}, earns a positive "
                f"return net of costs [{concept}].")
    if s.kind == "screen" and s.filters:
        return (f"Holding the stocks that pass {'; '.join(c.describe() for c in s.filters)}, rebalanced {when}, "
                "beats cash on a risk-adjusted basis net of costs.")
    if s.kind == "time_series" and s.time_series is not None:
        ts = s.time_series
        flat = "short" if ts.when_flat == "short" else "in cash"
        return (f"Holding {', '.join(ts.assets)} only while {' and '.join(c.describe() for c in ts.entry)} "
                f"(otherwise {flat}), checked {when}, earns a better risk-adjusted return than cash.")
    if s.kind == "factor_model":
        return f"The {str(s.factor_model).upper()} factors built from this universe earn positive premia net of costs."
    return s.idea or s.name
