"""Backtest results -> ``BacktestInterpretation`` (Claude as a skeptical reviewer, plus an offline heuristic).

Grounding
---------
Every number an interpretation cites is a :class:`~aitrading.backtest.models.CitedMetric` with a
dotted path into the :class:`~aitrading.backtest.models.BacktestResult` (``stats.strategy.sharpe``,
``regression.betas.HML``, ``quantiles.annual_return_by_quantile_pct.0``,
``factor_checks.HML.correlation_with_official``). :func:`resolve_metric_path` resolves a path and
:func:`verify_interpretation` checks each cited value against it, returning one
:class:`~aitrading.core.models.EvidenceCheck` per citation.

Path syntax: model fields and dict keys joined by dots; list items by integer index (negative
indices count from the end) or - for lists of named records - by name (``factor`` for
factor_checks, ``dataset`` for data_usage). ``a[0]`` and ``a['HML']`` are accepted as aliases of
``a.0`` / ``a.HML``; dict keys and record names also match case- and punctuation-insensitively
(``betas.mktrf`` -> ``Mkt-RF``; ``UMD`` / ``WML`` -> ``Mom``).

Heuristic verdict rules (documented thresholds, :class:`HeuristicInterpreter`)
------------------------------------------------------------------------------
Let ``years`` = strategy periods / periods-per-year (else the calendar span), ``SR`` = strategy
Sharpe, ``t`` = factor-attribution alpha Newey-West t-stat, ``mono`` = quantile monotonicity.

* inconclusive    - fewer than 3 years of data (or no strategy statistics);
* robust          - SR >= 0.8 and t >= 3 and (no quantile analysis or mono >= 0.8);
* promising       - t >= 2, or (SR >= 0.5 and mono >= 0.6);
* likely_spurious - t < 1 and mono < 0.3;
* weak            - everything else.

A missing statistic never satisfies a threshold (e.g. without a regression a strategy cannot be
"robust"). The t > 3 hurdle for a new factor follows Harvey, Liu & Zhu (2016), "... and the
cross-section of expected returns", RFS 29; t > 2 is weak evidence.

``factor_model`` runs are judged on the factors themselves: with ``T`` the Newey-West t-stats of
the mean returns of the non-market factors (all factors for CAPM) and ``C`` the correlations of
the constructed factors with the official Kenneth French series:

* inconclusive    - fewer than 3 years, or no factor t-stat;
* robust          - min(T) >= 3 and (no C or min(C) >= 0.7);
* promising       - max(T) >= 2 and (no C or min(C) >= 0.5);
* likely_spurious - max(T) < 1;
* weak            - everything else.

Data-quality caps (:func:`verdict_caps`, applied after the rules above to both interpreters):

* a dataset that is not point-in-time (other than the risk-free rate, official factor series or
  the benchmark), or an engine warning of look-ahead bias -> the verdict is ``inconclusive``: the
  backtest used information that was not available at the time, so it cannot be judged;
* an engine warning (or data note) of survivorship bias -> at most ``promising``.

Annual cost drag (percent) = 2 x avg_turnover_pct x rebalances per year x costs_bps / 10 000:
``avg_turnover_pct`` is ONE-WAY turnover per rebalance while the engine charges ``costs_bps`` on
every unit bought or sold (traded notional = 2 x one-way turnover); rebalances per year come from
the run's rebalance frequency (daily 252, weekly 52, monthly 12, quarterly 4, annual 1), not from
the return series' periods per year.
"""

from __future__ import annotations

import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from aitrading.backtest.models import BacktestInterpretation, BacktestResult, CitedMetric
from aitrading.core.models import EvidenceCheck
from aitrading.llm.base import LLMError, StructuredLLM

__all__ = [
    "INTERPRET_SYSTEM_PROMPT",
    "THRESHOLDS",
    "BacktestInterpreter",
    "HeuristicInterpreter",
    "annual_cost_drag_pct",
    "apply_verdict_caps",
    "interpretation_payload",
    "interpret_result",
    "rebalances_per_year",
    "resolve_metric_path",
    "verdict_caps",
    "verify_interpretation",
]

#: Thresholds used by the heuristic interpreter (and quoted in the LLM system prompt).
THRESHOLDS: dict[str, float] = {
    "min_years": 3.0,
    "robust_sharpe": 0.8,
    "robust_alpha_t": 3.0,
    "robust_monotonicity": 0.8,
    "promising_alpha_t": 2.0,
    "promising_sharpe": 0.5,
    "promising_monotonicity": 0.6,
    "spurious_alpha_t": 1.0,
    "spurious_monotonicity": 0.3,
    "factor_robust_corr": 0.7,
    "factor_promising_corr": 0.5,
    "short_sample_years": 5.0,
    "high_r_squared": 0.8,
    "significant_beta_t": 3.0,
}

_MARKET_FACTORS = {"mktrf", "mkt", "market"}
_MODEL_FACTORS: dict[str, list[str]] = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "carhart4": ["Mkt-RF", "SMB", "HML", "Mom"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


# ------------------------------------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------------------------------------

_MISSING = object()
_ALIASES = {"umd": "mom", "wml": "mom", "momentum": "mom", "mkt": "mktrf", "market": "mktrf", "mktexcess": "mktrf"}
_RECORD_NAME_FIELDS = ("factor", "dataset", "label", "path", "name")


def _canon(s: str) -> str:
    c = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return _ALIASES.get(c, c)


def _lookup_key(d: dict, part: str) -> Any:
    if part in d:
        return d[part]
    lowered = [k for k in d if isinstance(k, str) and k.lower() == part.lower()]
    if len(lowered) == 1:
        return d[lowered[0]]
    canon = [k for k in d if isinstance(k, str) and _canon(k) == _canon(part)]
    if len(canon) == 1:
        return d[canon[0]]
    return _MISSING


def _step(obj: Any, part: str) -> Any:
    if isinstance(obj, BaseModel):
        if part in type(obj).model_fields:
            return getattr(obj, part)
        return _MISSING
    if isinstance(obj, dict):
        return _lookup_key(obj, part)
    if isinstance(obj, (list, tuple)):
        if re.fullmatch(r"-?\d+", part):
            i = int(part)
            return obj[i] if -len(obj) <= i < len(obj) else _MISSING
        hits = []
        for item in obj:
            for f in _RECORD_NAME_FIELDS:
                name = item.get(f) if isinstance(item, dict) else getattr(item, f, None)
                if isinstance(name, str) and _canon(name) == _canon(part):
                    hits.append(item)
                    break
        return hits[0] if len(hits) == 1 else _MISSING
    return _MISSING


def _as_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    try:  # numpy scalars
        import numpy as np

        if isinstance(x, np.generic) and np.issubdtype(type(x), np.number):
            v = float(x)
            return v if math.isfinite(v) else None
    except Exception:  # pragma: no cover
        pass
    return None


def resolve_metric_path(result: BacktestResult, path: str) -> float | None:
    """Resolve a dotted path into ``result`` to a finite float, or None if it does not resolve.

    Non-numeric leaves (strings, dates, whole records) and None / NaN values resolve to None.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    p = path.strip()
    p = re.sub(r"\[\s*(-?\d+)\s*\]", r".\1", p)
    p = re.sub(r"\[\s*['\"]([^'\"\]]+)['\"]\s*\]", r".\1", p)
    if p.startswith("result."):
        p = p[len("result."):]
    parts = [x for x in p.split(".") if x != ""]
    if not parts:
        return None
    obj: Any = result
    for part in parts:
        obj = _step(obj, part)
        if obj is _MISSING or obj is None:
            return None
    return _as_float(obj)


# ------------------------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------------------------


def _decimals_written(x: float) -> int:
    """Decimals in the shortest representation of ``x`` (at least 1: 3.0 and 3 are read as '3.0')."""
    try:
        exp = Decimal(repr(float(x))).normalize().as_tuple().exponent
    except InvalidOperation:  # pragma: no cover
        return 1
    return max(1, -int(exp)) if isinstance(exp, int) else 1


def _rounds_to(actual: float, cited: float) -> bool:
    d = _decimals_written(cited)
    if d > 12:
        return False
    q = Decimal(1).scaleb(-d)
    try:
        return Decimal(repr(float(actual))).quantize(q, rounding=ROUND_HALF_UP) == Decimal(repr(float(cited))).quantize(
            q, rounding=ROUND_HALF_UP
        )
    except InvalidOperation:  # pragma: no cover - absurd magnitudes
        return False


def _value_matches(cited: float, actual: float, rel_tol: float, abs_tol: float) -> bool:
    if not math.isfinite(cited):
        return False
    return math.isclose(cited, actual, rel_tol=rel_tol, abs_tol=abs_tol) or _rounds_to(actual, cited)


#: Precision of the numbers Claude is shown (``interpretation_payload`` rounds floats to 4 decimals).
_PAYLOAD_DECIMALS = 4


def _threshold_gates(path: str) -> tuple[tuple[float, ...], bool]:
    """(verdict thresholds that apply to the metric at ``path``, compare |value|?).

    t-stats are gated on |t| at 1 / 2 / 3 (spurious, promising, robust / significant beta), Sharpe
    at 0.5 / 0.8, quantile monotonicity at 0.3 / 0.6 / 0.8, correlations with the official factors
    at 0.5 / 0.7 and R-squared at 0.8 - the values in ``THRESHOLDS``.
    """
    p = re.sub(r"\[\s*['\"]?([^'\"\]]+?)['\"]?\s*\]", r".\1", path or "")
    parts = [_canon(x) for x in p.split(".") if x]
    th = THRESHOLDS
    if any(x.endswith("tstat") or x.endswith("tstats") for x in parts):
        gates = {th["spurious_alpha_t"], th["promising_alpha_t"], th["robust_alpha_t"], th["significant_beta_t"]}
        return tuple(sorted(gates)), True
    if "sharpe" in parts:
        return (th["promising_sharpe"], th["robust_sharpe"]), False
    if "monotonicity" in parts:
        return (th["spurious_monotonicity"], th["promising_monotonicity"], th["robust_monotonicity"]), False
    if "correlationwithofficial" in parts:
        return (th["factor_promising_corr"], th["factor_robust_corr"]), False
    if "rsquared" in parts:
        return (th["high_r_squared"],), False
    return (), False


def _sign_problem(cited: float, actual: float) -> str | None:
    a = round(actual, _PAYLOAD_DECIMALS)
    if a != 0 and (cited == 0 or math.copysign(1.0, cited) != math.copysign(1.0, a)):
        return "the sign differs" if cited != 0 else "a zero citation of a non-zero value"
    return None


def _citation_problem(path: str, cited: float, actual: float) -> str | None:
    """Why a citation whose value is numerically close is still misleading, or None.

    * sign: when the actual value is non-zero at payload precision, the citation must have the
      same sign (a zero citation of a non-zero value counts as a different sign);
    * thresholds: the cited value must lie on the same side of every verdict threshold as the
      actual value (rounded to payload precision), so rounding 2.95 up to "3.0" cannot clear the
      t >= 3 hurdle and 0.76 cannot become a "monotonic" 0.8.
    """
    sign = _sign_problem(cited, actual)
    if sign:
        return sign
    a = round(actual, _PAYLOAD_DECIMALS)
    gates, use_abs = _threshold_gates(path)
    f = abs if use_abs else (lambda x: x)
    for g in gates:
        if (f(a) >= g) != (f(cited) >= g):
            return f"the citation lands on the other side of the {g:g} {'|t|' if use_abs else 'verdict'} threshold"
    return None


def verify_interpretation(
    interp: BacktestInterpretation,
    result: BacktestResult,
    rel_tol: float = 0.01,
    abs_tol: float = 0.0,
) -> list[EvidenceCheck]:
    """One ``EvidenceCheck`` (kind 'quant', ref = path) per cited metric.

    * ``verified`` - the cited value is within ``math.isclose(rel_tol, abs_tol)`` of the resolved
      value (default: 1% relative, no blanket absolute tolerance, so small statistics such as rank
      ICs, betas and correlations get no free slack), or equals the resolved value rounded (half
      away from zero) to the decimals written in the citation (at least one decimal: an
      integer-valued citation is read as "x.0"); AND it has the actual value's sign (when that is
      non-zero at the 4-decimal payload precision) AND it lies on the same side of every verdict
      threshold (|t| 1 / 2 / 3, Sharpe 0.5 / 0.8, monotonicity 0.3 / 0.6 / 0.8, correlation with
      the official factor 0.5 / 0.7, R-squared 0.8) as the actual value;
    * ``mismatch`` - the path resolves but the value differs, or fails the sign / threshold test;
    * ``not_found`` - the path does not resolve to a number.
    """
    checks: list[EvidenceCheck] = []
    for cm in interp.cited_metrics:
        actual = resolve_metric_path(result, cm.path)
        claim = f"{cm.path}={cm.value:g}"
        if actual is None:
            checks.append(EvidenceCheck(kind="quant", ref=cm.path, claim=claim, status="not_found",
                                        detail="path does not resolve to a number in the backtest result"))
            continue
        if not _value_matches(cm.value, actual, rel_tol, abs_tol):
            sign = _sign_problem(cm.value, actual) if math.isfinite(cm.value) else None
            checks.append(EvidenceCheck(kind="quant", ref=cm.path, claim=claim, status="mismatch",
                                        detail=f"cited {cm.value:g} but the result has {actual:.6g}" + (f": {sign}" if sign else "")))
            continue
        problem = _citation_problem(cm.path, cm.value, actual)
        if problem:
            checks.append(EvidenceCheck(kind="quant", ref=cm.path, claim=claim, status="mismatch",
                                        detail=f"cited {cm.value:g} but the result has {actual:.6g}: {problem}"))
        else:
            checks.append(EvidenceCheck(kind="quant", ref=cm.path, claim=claim, status="verified", detail=f"actual={actual:.6g}"))
    return checks


def _score(checks: list[EvidenceCheck]) -> tuple[float, int]:
    n_ok = sum(c.status == "verified" for c in checks)
    return (n_ok / len(checks) if checks else 0.0, n_ok)


# ------------------------------------------------------------------------------------------------
# Prompt payload
# ------------------------------------------------------------------------------------------------


def _round_floats(x: Any, nd: int = 4) -> Any:
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round_floats(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round_floats(v, nd) for v in x]
    return x


def _years(result: BacktestResult) -> float:
    st = result.stats.get("strategy")
    if st is not None and st.periods_per_year > 0:
        return st.n_periods / st.periods_per_year
    return max(0.0, (result.end - result.start).days / 365.25)


_REBALANCES_PER_YEAR: dict[str, float] = {
    "daily": 252.0, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0, "annual": 1.0, "annually": 1.0, "yearly": 1.0,
}


def rebalances_per_year(result: BacktestResult) -> float | None:
    """Rebalances a year from ``result.rebalance`` (else ``spec.rebalance``); None if unknown.

    daily 252, weekly 52, monthly 12, quarterly 4, annual 1. This - not the return series'
    ``periods_per_year`` (252 for daily returns) - is what one-way turnover per rebalance scales by.
    """
    for r in (result.rebalance, (result.spec or {}).get("rebalance")):
        if isinstance(r, str) and r.strip().lower() in _REBALANCES_PER_YEAR:
            return _REBALANCES_PER_YEAR[r.strip().lower()]
    return None


def annual_cost_drag_pct(avg_turnover_pct: float, rebalances: float, costs_bps: float) -> float:
    """Annual trading-cost drag in percent: 2 x one-way turnover x rebalances a year x bps / 10 000.

    The engine charges ``costs_bps`` on traded notional (buys + sells) = 2 x the reported one-way
    turnover, e.g. 50% one-way turnover x 12 rebalances x 10 bps = 100% traded x 12 x 0.10% = 1.20%.
    """
    return 2.0 * float(avg_turnover_pct) * float(rebalances) * float(costs_bps) / 10_000.0


_SPEC_KEYS = (
    "name", "kind", "universe", "start", "end", "rebalance", "signal", "filters", "portfolio", "factor_model",
    "time_series", "attribution_model", "costs_bps", "benchmark", "assumptions", "unsupported_requests",
)
_MAX_WARNINGS = 40


def interpretation_payload(result: BacktestResult) -> dict:
    """Everything the reviewer needs except the return arrays (floats rounded to 4 decimals)."""
    spec = result.spec or {}
    warnings = [w[:400] for w in result.warnings[:_MAX_WARNINGS]]
    if len(result.warnings) > _MAX_WARNINGS:
        warnings.append(f"... {len(result.warnings) - _MAX_WARNINGS} more warnings omitted")
    payload = {
        "idea": result.idea,
        "run": {
            "provider": result.provider,
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "rebalance": result.rebalance,
            "years": round(_years(result), 2),
            "n_latest_holdings": len(result.latest_holdings),
        },
        "spec": {k: spec[k] for k in _SPEC_KEYS if k in spec},
        "stats": {k: v.model_dump(mode="json") for k, v in result.stats.items()},
        "regression": result.regression.model_dump(mode="json") if result.regression else None,
        "quantiles": result.quantiles.model_dump(mode="json") if result.quantiles else None,
        "factor_checks": [c.model_dump(mode="json") for c in result.factor_checks],
        "data_usage": [d.model_dump(mode="json") for d in result.data_usage],
        "warnings": warnings,
    }
    return _round_floats(payload)


# ------------------------------------------------------------------------------------------------
# Data-quality verdict caps
# ------------------------------------------------------------------------------------------------

#: Verdict order for capping; ``inconclusive`` is the floor (a cap at inconclusive overrides all).
_VERDICT_RANK = {"inconclusive": 0, "likely_spurious": 1, "weak": 2, "promising": 3, "robust": 4}
#: Datasets that do not feed the signal: their point-in-time status cannot create look-ahead.
_NON_SIGNAL_DATASET = re.compile(r"risk[\s_-]*free|\brf\b|t[\s_-]*bills?|official|benchmark", re.I)
_LOOKAHEAD = re.compile(r"look[\s_-]*ahead|not\s+point[\s_-]*in[\s_-]*time|non[\s_-]*point[\s_-]*in[\s_-]*time|\bnon[\s_-]*pit\b", re.I)
_SURVIVOR = re.compile(r"survivor\w*", re.I)
_NEGATED_BEFORE = re.compile(
    r"\b(?:no|without|avoid\w*|prevent\w*|free\s+of|guard\w*\s+against|eliminat\w*|remov\w*|correct\w*\s+for|"
    r"adjust\w*\s+for)\s+(?:[\w-]+\s+){0,2}$",
    re.I,
)
_NEGATED_AFTER = re.compile(r"^[\s-]*(?:bias[\s-]*)?(?:free|corrected|adjusted)\b", re.I)


def _mentions(text: str, pat: re.Pattern) -> bool:
    """True if ``pat`` occurs in ``text`` without an obvious negation ("no look-ahead", "survivorship-free")."""
    for m in pat.finditer(text or ""):
        if _NEGATED_BEFORE.search(text[max(0, m.start() - 40): m.start()]) or _NEGATED_AFTER.search(text[m.end(): m.end() + 20]):
            continue
        return True
    return False


def _has_survivorship_flag(result: BacktestResult) -> bool:
    texts = [*result.warnings, *(d.notes for d in result.data_usage)]
    return any(_mentions(t, _SURVIVOR) for t in texts)


def verdict_caps(result: BacktestResult) -> list[tuple[str, str]]:
    """``(cap, reason)`` for every data problem that limits the verdict, strictest first.

    * a dataset with ``point_in_time=False`` (other than the risk-free rate, official factor
      series or benchmark, which do not feed the signal) -> ``inconclusive``;
    * an engine warning of look-ahead bias -> ``inconclusive``;
    * an engine warning or data note of survivorship bias -> ``promising``.
    """
    caps: list[tuple[str, str]] = []
    non_pit = [d.dataset for d in result.data_usage if not d.point_in_time and not _NON_SIGNAL_DATASET.search(d.dataset)]
    if non_pit:
        names = ", ".join(f"'{n}'" for n in non_pit)
        caps.append(("inconclusive", f"dataset{'s' if len(non_pit) > 1 else ''} {names} {'are' if len(non_pit) > 1 else 'is'} "
                                     "not point-in-time (look-ahead bias), so the backtest cannot be judged"))
    if any(_mentions(w, _LOOKAHEAD) for w in result.warnings):
        caps.append(("inconclusive", "the engine warns of look-ahead bias, so the backtest cannot be judged"))
    if _has_survivorship_flag(result):
        caps.append(("promising", "the universe has survivorship bias (delisted names missing), which caps the verdict at promising"))
    caps.sort(key=lambda c: _VERDICT_RANK[c[0]])
    return caps


def _capped(verdict: str, result: BacktestResult) -> tuple[str, list[str]]:
    """(capped verdict, reasons of every cap that binds) for ``verdict``."""
    binding = [(cap, why) for cap, why in verdict_caps(result) if _VERDICT_RANK.get(verdict, 0) > _VERDICT_RANK[cap]]
    if not binding:
        return verdict, []
    return binding[0][0], [why for _, why in binding]  # verdict_caps is sorted strictest first


def apply_verdict_caps(interp: BacktestInterpretation, result: BacktestResult) -> BacktestInterpretation:
    """Enforce :func:`verdict_caps` on an interpretation (e.g. Claude's): returns a copy with the
    verdict lowered, the summary prefixed and a caveat added when a cap binds, else ``interp``."""
    new, reasons = _capped(interp.verdict, result)
    if new == interp.verdict:
        return interp
    note = f"Verdict capped at {new} (the reviewer said {interp.verdict}): " + "; ".join(reasons) + "."
    return interp.model_copy(update={
        "verdict": new,
        "summary": note + " " + interp.summary,
        "biases_and_caveats": [*interp.biases_and_caveats, note],
    })


# ------------------------------------------------------------------------------------------------
# Claude interpreter
# ------------------------------------------------------------------------------------------------

INTERPRET_SYSTEM_PROMPT = """\
You are a skeptical senior quantitative researcher reviewing a backtest produced by an automated research \
platform. A portfolio manager typed a research idea; the platform translated it into a strategy spec, gathered \
point-in-time data, ran the backtest and computed the statistics you are given. Your job is to judge how credible \
the result is as evidence that the idea works out of sample, and to say so plainly. You are not selling the \
strategy; an honest "this is probably noise" is a good answer.

# What you receive
A <result> JSON object with:
- idea, run (provider, start, end, rebalance, years, number of latest holdings);
- spec: the strategy definition, including its assumptions and unsupported_requests (parts of the idea that were \
NOT tested);
- stats: performance statistics per series (strategy, benchmark, long, short, factor series ...). Units: fields \
ending in _pct are percent; Sharpe, Sortino, Calmar, information ratio, betas and t-stats are unitless; t-stats \
are Newey-West;
- regression: factor attribution of the strategy (alpha_annual_pct, Newey-West alpha_t_stat, betas, beta_t_stats, \
r_squared, the model and where its factors came from);
- quantiles: cross-sectional bucket analysis (bucket 1 = worst signal ... bucket n = best), spread_annual_pct, \
monotonicity (Spearman of bucket number vs bucket return, -1..1), rank IC mean / t-stat / hit rate;
- factor_checks: constructed factors vs the official Kenneth French series (correlation, annual premia);
- data_usage: datasets, sources, coverage and whether each is point-in-time;
- warnings: messages from the engine.
The return series themselves are deliberately not included.

# How to judge
- Significance. A new factor or signal needs an alpha t-stat above 3 (Harvey, Liu & Zhu 2016: hundreds of factors \
have been tried, so t > 2 is no longer enough); t between 2 and 3 is weak evidence; below 2 the alpha is not \
distinguishable from zero. Be equally skeptical of Sharpe ratios from short samples: the standard error of an \
annualised Sharpe ratio is roughly sqrt((1 + SR^2 / 2) / years).
- Known-factor exposure. Large, significant betas with an insignificant alpha mean the strategy repackages known \
factors (market, size, value, momentum, profitability, investment) rather than adding an edge; a high r_squared \
says the same.
- Quantile monotonicity. A genuine cross-sectional signal orders returns across buckets (monotonicity near 1) with \
a positive, significant IC. A spread that comes from one extreme bucket, or an inverted ordering, is fragile.
- Turnover x costs. avg_turnover_pct is ONE-WAY turnover per rebalance, but costs_bps is charged on every unit \
bought or sold, i.e. on traded notional = 2 x avg_turnover_pct. The annual cost drag in percent is therefore \
2 x avg_turnover_pct x rebalances per year x costs_bps / 10000, with rebalances per year taken from run.rebalance \
(daily 252, weekly 52, monthly 12, quarterly 4, annual 1) - not from periods_per_year, which describes the return \
series (e.g. 50% one-way turnover, monthly, 10 bps: 2 x 50 x 12 x 10 / 10000 = 1.2% a year). Ask whether the edge \
survives realistic and doubled costs, and whether capacity is limited (illiquid or small names, high turnover).
- Biases. Survivorship bias when the universe is today's constituents (delisted losers are missing); look-ahead \
when fundamentals or estimates are not point-in-time; data snooping (well-known published anomalies, parameters \
picked after the fact); short samples and regime dependence (a single bull market or a single crash can drive \
everything); post-publication decay for published anomalies.
- Data caveats. Read data_usage and warnings and carry every material caveat into biases_and_caveats. If the spec \
lists unsupported_requests, say that part of the idea was not tested.
- Disqualifying data problems. If any dataset in data_usage other than the risk-free rate, official factor series \
or the benchmark has point_in_time = false, or a warning reports look-ahead bias, the backtest used information \
that was not available at the time: the verdict must be inconclusive, however strong the statistics look. A \
survivorship-bias warning caps the verdict at promising. These caps are enforced programmatically.
- Factor-model runs (spec.kind = factor_model): for each factor report its premium, its significance and its \
correlation with the official series; low correlation means the construction (universe, data) does not reproduce \
the academic factor, which limits what the run can say.
- Use only the numbers provided. Do not use outside knowledge of market events after the sample end, and do not \
claim things about the sample period that the data does not show.

# Verdict (pick one)
- robust: economically meaningful and statistically strong (Sharpe >= 0.8 and alpha t >= 3, quantiles monotonic \
where available), survives costs, no disqualifying bias.
- promising: encouraging but not yet convincing (e.g. alpha t between 2 and 3, or a decent Sharpe with fairly \
monotonic quantiles); worth more testing.
- weak: little evidence either way.
- likely_spurious: no significant alpha with non-monotonic or inverted quantiles, or results explained by known \
factors or by a bias.
- inconclusive: too little data (under about 3 years), too few observations, or data problems that prevent a \
judgment.

# Citations
Every number you state in summary or key_findings must also appear in cited_metrics with the exact dotted path into \
the <result> JSON and the value exactly as given there (do not round differently, do not recompute, do not convert \
units). Path syntax: field names and dict keys joined by dots; list items by index (quantiles.\
annual_return_by_quantile_pct.0 is bucket 1); factor_checks items by factor name. Examples: stats.strategy.sharpe, \
stats.benchmark.cagr_pct, stats.strategy.avg_turnover_pct, regression.alpha_t_stat, regression.betas.HML, \
regression.beta_t_stats.Mkt-RF, quantiles.monotonicity, factor_checks.HML.correlation_with_official. Cited values \
are verified programmatically and a citation that does not match is an error; so is a citation with the wrong \
sign, or one rounded across a verdict threshold (e.g. a t-stat of 2.95 cited as 3.0, a monotonicity of 0.76 cited \
as 0.8). Numbers you derive (cost drag, Sharpe standard error) may be described in words but are not cited.

# Output
- summary: 2-4 plain sentences that lead with the verdict and the single most important reason.
- key_findings: 3-6 findings, each with its evidence.
- biases_and_caveats: every material bias, data caveat and untested part of the idea.
- next_experiments: 2-5 concrete follow-up tests (sub-period split, doubled costs, sector-neutral version, another \
attribution model, a point-in-time universe with delisted names, ...).
"""


class BacktestInterpreter:
    """Claude reads the result as a skeptical reviewer; cited numbers are verified, with a repair round."""

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        effort: str = "high",
        max_repair_rounds: int = 1,
        rel_tol: float = 0.01,
        abs_tol: float = 0.0,
    ):
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be >= 0")
        self.llm = llm
        self.effort = effort
        self.max_repair_rounds = max_repair_rounds
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    system_prompt = INTERPRET_SYSTEM_PROMPT

    @staticmethod
    def user_prompt(result: BacktestResult) -> str:
        body = json.dumps(interpretation_payload(result), separators=(",", ":"), ensure_ascii=False)
        return f"<result>\n{body}\n</result>\n\nReview this backtest and return your interpretation."

    def repair_prompt(self, result: BacktestResult, previous: BacktestInterpretation, checks: list[EvidenceCheck]) -> str:
        failures = [c for c in checks if c.status != "verified"]
        lines = [f"- {c.ref}: {c.status} ({c.detail})" for c in failures]
        if not previous.cited_metrics:
            lines.append("- no metrics were cited: cite the metrics behind every number you state")
        prev = json.dumps(previous.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False)
        return (
            self.user_prompt(result)
            + "\n\nYour previous interpretation failed citation checks:\n"
            + f"<previous_interpretation>\n{prev}\n</previous_interpretation>\n"
            + "<verification_failures>\n" + "\n".join(lines) + "\n</verification_failures>\n\n"
            + "Return the complete corrected interpretation. Cite only paths that exist in <result>, with values copied "
            "exactly as given; correct or remove every failed citation and any statement that relied on it."
        )

    def _verify(self, interp: BacktestInterpretation, result: BacktestResult) -> list[EvidenceCheck]:
        return verify_interpretation(interp, result, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def interpret(self, result: BacktestResult) -> tuple[BacktestInterpretation, list[EvidenceCheck]]:
        interp = self.llm.structured(
            purpose="interpret", system=self.system_prompt, user=self.user_prompt(result),
            output_model=BacktestInterpretation, effort=self.effort,
        )
        checks = self._verify(interp, result)
        best = (interp, checks)
        for _ in range(self.max_repair_rounds):
            cur_interp, cur_checks = best
            if cur_checks and all(c.status == "verified" for c in cur_checks):
                break
            fixed = self.llm.structured(
                purpose="interpret:repair", system=self.system_prompt,
                user=self.repair_prompt(result, cur_interp, cur_checks),
                output_model=BacktestInterpretation, effort=self.effort,
            )
            fixed_checks = self._verify(fixed, result)
            if _score(fixed_checks) > _score(cur_checks):
                best = (fixed, fixed_checks)
        interp, checks = best
        return apply_verdict_caps(interp, result), checks


# ------------------------------------------------------------------------------------------------
# Heuristic interpreter
# ------------------------------------------------------------------------------------------------


def _fmt(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


class HeuristicInterpreter:
    """Deterministic, rule-based reading of a result (thresholds in the module docstring / ``THRESHOLDS``)."""

    name = "heuristic"

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.th = {**THRESHOLDS, **(thresholds or {})}

    # -- helpers ----------------------------------------------------------------------------

    @staticmethod
    def _cite(result: BacktestResult, out: list[CitedMetric], path: str, meaning: str) -> float | None:
        v = resolve_metric_path(result, path)
        if v is not None and all(c.path != path for c in out):
            out.append(CitedMetric(path=path, value=v, meaning=meaning))
        return v

    # -- verdicts ---------------------------------------------------------------------------

    def verdict(self, result: BacktestResult) -> tuple[str, str]:
        """(verdict, one-line reason) for any result: the statistical rules, then the data caps."""
        base, why = self._base_verdict(result)
        capped, reasons = _capped(base, result)
        if capped == base:
            return base, why
        return capped, "; ".join(reasons) + f" (the statistics alone read {base}: {why})"

    def _base_verdict(self, result: BacktestResult) -> tuple[str, str]:
        kind = (result.spec or {}).get("kind")
        years = _years(result)
        th = self.th
        if years < th["min_years"]:
            return "inconclusive", f"only {years:.1f} years of data (< {th['min_years']:g})"
        if kind == "factor_model":
            return self._factor_verdict(result)
        st = result.stats.get("strategy")
        if st is None:
            return "inconclusive", "no strategy statistics"
        sr = st.sharpe
        t = result.regression.alpha_t_stat if result.regression else None
        mono = result.quantiles.monotonicity if result.quantiles else None
        has_q = result.quantiles is not None
        ge = lambda x, k: x is not None and x >= th[k]  # noqa: E731
        if ge(sr, "robust_sharpe") and ge(t, "robust_alpha_t") and (not has_q or ge(mono, "robust_monotonicity")):
            return "robust", f"Sharpe {_fmt(sr)} with alpha t-stat {_fmt(t)} >= {th['robust_alpha_t']:g}" + (
                f" and monotonic quantiles ({_fmt(mono)})" if has_q else ""
            )
        if ge(t, "promising_alpha_t"):
            if ge(sr, "robust_sharpe") and ge(t, "robust_alpha_t"):
                why = f"Sharpe {_fmt(sr)} and alpha t-stat {_fmt(t)} are strong, but quantile monotonicity {_fmt(mono)} is below {th['robust_monotonicity']:g}"
            elif ge(t, "robust_alpha_t"):
                why = f"alpha t-stat {_fmt(t)} clears {th['robust_alpha_t']:g}, but the Sharpe ratio {_fmt(sr)} is below {th['robust_sharpe']:g}"
            else:
                why = f"alpha t-stat {_fmt(t)} clears {th['promising_alpha_t']:g} but not the {th['robust_alpha_t']:g} hurdle for a new factor"
            return "promising", why
        if ge(sr, "promising_sharpe") and ge(mono, "promising_monotonicity"):
            return "promising", f"Sharpe {_fmt(sr)} with fairly monotonic quantiles ({_fmt(mono)}), but alpha t-stat only {_fmt(t)}"
        if t is not None and t < th["spurious_alpha_t"] and mono is not None and mono < th["spurious_monotonicity"]:
            return "likely_spurious", f"alpha t-stat {_fmt(t)} < {th['spurious_alpha_t']:g} and quantile monotonicity {_fmt(mono)} < {th['spurious_monotonicity']:g}"
        return "weak", f"Sharpe {_fmt(sr)}, alpha t-stat {_fmt(t)}" + (f", monotonicity {_fmt(mono)}" if has_q else "") + ": little evidence either way"

    def _factor_names(self, result: BacktestResult) -> list[str]:
        model = (result.spec or {}).get("factor_model") or ""
        names = list(_MODEL_FACTORS.get(model, []))
        for c in result.factor_checks:
            if all(_canon(c.factor) != _canon(n) for n in names):
                names.append(c.factor)
        return names

    def _factor_evidence(self, result: BacktestResult) -> list[dict]:
        rows = []
        for f in self._factor_names(result):
            st = next((v for k, v in result.stats.items() if _canon(k) == _canon(f)), None)
            chk = next((c for c in result.factor_checks if _canon(c.factor) == _canon(f)), None)
            if st is None and chk is None:
                continue
            rows.append({
                "factor": f,
                "stats_key": next((k for k in result.stats if _canon(k) == _canon(f)), None),
                "t": st.mean_return_t_stat if st else None,
                "corr": chk.correlation_with_official if chk else None,
                "premium": chk.annual_premium_constructed_pct if chk else None,
                "official": chk.annual_premium_official_pct if chk else None,
                "market": _canon(f) in _MARKET_FACTORS,
            })
        return rows

    def _factor_verdict(self, result: BacktestResult) -> tuple[str, str]:
        th = self.th
        rows = self._factor_evidence(result)
        pool = [r for r in rows if not r["market"]] or rows
        T = [r["t"] for r in pool if r["t"] is not None]
        C = [r["corr"] for r in rows if r["corr"] is not None]
        if not T:
            return "inconclusive", "no factor t-statistics available"
        if min(T) >= th["robust_alpha_t"] and (not C or min(C) >= th["factor_robust_corr"]):
            return "robust", f"every factor premium has t >= {th['robust_alpha_t']:g} and the construction tracks the official factors"
        if max(T) >= th["promising_alpha_t"] and (not C or min(C) >= th["factor_promising_corr"]):
            return "promising", f"at least one factor premium has t >= {th['promising_alpha_t']:g}, but not all clear {th['robust_alpha_t']:g}"
        if max(T) < th["spurious_alpha_t"]:
            return "likely_spurious", f"no factor premium has t >= {th['spurious_alpha_t']:g} in this sample"
        if C and min(C) < th["factor_promising_corr"]:
            return "weak", f"the constructed factors track the official series poorly (lowest correlation {min(C):.2f})"
        return "weak", "factor premia are not statistically reliable in this sample"

    # -- main -------------------------------------------------------------------------------

    def interpret(self, result: BacktestResult) -> tuple[BacktestInterpretation, list[EvidenceCheck]]:
        th = self.th
        spec = result.spec or {}
        kind = spec.get("kind")
        years = _years(result)
        verdict, reason = self.verdict(result)
        cites: list[CitedMetric] = []
        findings: list[str] = []
        caveats: list[str] = []
        nexts: list[str] = []
        c = lambda path, meaning: self._cite(result, cites, path, meaning)  # noqa: E731

        # strategy performance
        st = result.stats.get("strategy")
        if st is not None:
            sr = c("stats.strategy.sharpe", "strategy annualised Sharpe ratio")
            cagr = c("stats.strategy.cagr_pct", "strategy CAGR, %")
            vol = c("stats.strategy.volatility_pct", "strategy annualised volatility, %")
            mdd = c("stats.strategy.max_drawdown_pct", "strategy maximum drawdown, %")
            tm = c("stats.strategy.mean_return_t_stat", "Newey-West t-stat of the mean periodic return")
            findings.append(
                f"Over {years:.1f} years ({result.start} to {result.end}) the strategy returned {_fmt(cagr)}% a year with "
                f"{_fmt(vol)}% volatility (Sharpe {_fmt(sr)}, max drawdown {_fmt(mdd)}%, mean-return t-stat {_fmt(tm)})."
            )
            if "benchmark" in result.stats:
                bsr = c("stats.benchmark.sharpe", "benchmark Sharpe ratio")
                bcagr = c("stats.benchmark.cagr_pct", "benchmark CAGR, %")
                findings.append(f"Benchmark over the same period: {_fmt(bcagr)}% a year, Sharpe {_fmt(bsr)}.")
            turn = c("stats.strategy.avg_turnover_pct", "average one-way turnover per rebalance, % of book")
            costs = spec.get("costs_bps")
            n_reb = rebalances_per_year(result)
            if turn is not None and isinstance(costs, (int, float)) and not isinstance(costs, bool) and n_reb:
                drag = annual_cost_drag_pct(turn, n_reb, float(costs))
                txt = (
                    f"Average one-way turnover of {_fmt(turn, 1)}% per rebalance ({_fmt(2 * turn, 1)}% of the book traded) "
                    f"x {n_reb:g} rebalances a year at {float(costs):g} bps per unit traded implies about "
                    f"{drag:.2f}% a year of trading costs (already deducted); doubling costs would take roughly another {drag:.2f}%."
                )
                if cagr is not None and drag > 0 and (drag >= 1.0 or (cagr > 0 and drag >= 0.25 * cagr)):
                    caveats.append(txt + " The result is cost-sensitive.")
                else:
                    findings.append(txt)

        # attribution
        reg = result.regression
        if reg is not None:
            a = c("regression.alpha_annual_pct", f"annualised alpha vs {reg.model}, %")
            at = c("regression.alpha_t_stat", f"Newey-West t-stat of the {reg.model} alpha")
            r2 = c("regression.r_squared", f"R-squared of the {reg.model} regression")
            for f in reg.betas:
                c(f"regression.betas.{f}", f"{f} beta")
            sig = [(f, b, reg.beta_t_stats.get(f)) for f, b in reg.betas.items()
                   if reg.beta_t_stats.get(f) is not None and abs(reg.beta_t_stats[f]) >= th["significant_beta_t"]]
            for f, _, _ in sig:
                c(f"regression.beta_t_stats.{f}", f"t-stat of the {f} beta")
            hurdle = (
                "clears the t > 3 hurdle for a new factor (Harvey-Liu-Zhu 2016)" if at is not None and at >= th["robust_alpha_t"]
                else "is weak evidence (2 <= t < 3)" if at is not None and at >= th["promising_alpha_t"]
                else "is not distinguishable from zero (t < 2)"
            )
            findings.append(
                f"{reg.model.upper()} attribution ({reg.factor_source}, n={reg.n}): alpha {_fmt(a)}% a year with t-stat "
                f"{_fmt(at)}, which {hurdle}; R-squared {_fmt(r2)}."
            )
            if sig and (at is None or at < th["promising_alpha_t"]):
                expo = ", ".join(f"{f} beta {b:.2f} (t {t:.1f})" for f, b, t in sig)
                caveats.append(f"Returns are largely explained by known factor exposures ({expo}) with no significant alpha: "
                               "the strategy looks like a repackaging of known factors.")
            elif sig:
                findings.append("Significant factor exposures: " + ", ".join(f"{f} beta {b:.2f} (t {t:.1f})" for f, b, t in sig) + ".")
            if r2 is not None and r2 >= th["high_r_squared"]:
                caveats.append(f"R-squared of {r2:.2f} against {reg.model}: most of the return variation is known-factor exposure.")
        elif kind != "factor_model":
            caveats.append("No factor attribution is available, so the return cannot be separated from known factor exposures.")

        # quantiles
        q = result.quantiles
        if q is not None:
            mono = c("quantiles.monotonicity", "Spearman correlation of bucket number vs bucket return")
            spread = c("quantiles.spread_annual_pct", "best minus worst bucket, annualised %")
            ic = c("quantiles.ic_mean", "mean cross-sectional rank IC")
            ict = c("quantiles.ic_t_stat", "t-stat of the mean rank IC")
            if q.annual_return_by_quantile_pct:
                lo = c("quantiles.annual_return_by_quantile_pct.0", "worst-signal bucket annual return, %")
                hi = c(f"quantiles.annual_return_by_quantile_pct.{len(q.annual_return_by_quantile_pct) - 1}",
                       "best-signal bucket annual return, %")
            else:
                lo = hi = None
            shape = ("monotonic" if mono is not None and mono >= th["robust_monotonicity"]
                     else "roughly monotonic" if mono is not None and mono >= th["promising_monotonicity"]
                     else "not monotonic" if mono is not None and mono >= 0 else "inverted")
            findings.append(
                f"{q.n_quantiles} signal buckets: worst {_fmt(lo)}% vs best {_fmt(hi)}% a year (spread {_fmt(spread)}%), "
                f"monotonicity {_fmt(mono)} ({shape}); rank IC {_fmt(ic, 3)} (t {_fmt(ict)})."
            )

        # factor-model runs
        if kind == "factor_model":
            rows = self._factor_evidence(result)
            for r in rows:
                f = r["factor"]
                parts = []
                if r["premium"] is not None:
                    c(f"factor_checks.{f}.annual_premium_constructed_pct", f"constructed {f} annual premium, %")
                    parts.append(f"premium {r['premium']:.2f}% a year")
                if r["official"] is not None:
                    c(f"factor_checks.{f}.annual_premium_official_pct", f"official {f} annual premium over the overlap, %")
                    parts.append(f"official {r['official']:.2f}%")
                if r["t"] is not None and r["stats_key"] is not None:
                    c(f"stats.{r['stats_key']}.mean_return_t_stat", f"Newey-West t-stat of the mean {f} return")
                    parts.append(f"t-stat {r['t']:.2f}")
                if r["corr"] is not None:
                    c(f"factor_checks.{f}.correlation_with_official", f"correlation of constructed {f} with the official series")
                    parts.append(f"correlation with official {r['corr']:.2f}")
                findings.append(f"{f}: " + (", ".join(parts) if parts else "no statistics available") + ".")
                if r["corr"] is not None and r["corr"] < th["factor_promising_corr"]:
                    caveats.append(f"Constructed {f} correlates only {r['corr']:.2f} with the official factor: the universe / data do "
                                   "not reproduce the academic construction.")
            if not rows:
                caveats.append("No factor statistics or official-factor comparisons are available for this run.")

        # sample length / regime
        if years < th["min_years"]:
            caveats.append(f"Only {years:.1f} years of data: far too short to separate skill from noise.")
        elif years < th["short_sample_years"]:
            caveats.append(f"Short sample ({years:.1f} years): results may reflect a single market regime.")

        # data caveats
        for d in result.data_usage:
            if not d.point_in_time:
                caveats.append(f"Dataset '{d.dataset}' ({d.source}, {d.coverage}) is not point-in-time: possible look-ahead bias."
                               + (f" {d.notes}" if d.notes else ""))
        priority = re.compile(r"survivor|point[\s-]*in[\s-]*time|look[\s-]*ahead|delist|coverage|missing|snapshot|stale", re.I)
        ordered = [w for w in result.warnings if priority.search(w)] + [w for w in result.warnings if not priority.search(w)]
        seen: set[str] = set()
        shown = 0
        for w in ordered:
            if w in seen:
                continue
            seen.add(w)
            if shown < 12 or priority.search(w):
                caveats.append(f"Engine warning: {w}")
                shown += 1
        if len(seen) > shown:
            caveats.append(f"{len(seen) - shown} further engine warnings not listed.")
        for u in spec.get("unsupported_requests") or []:
            caveats.append(f"Not tested (could not be expressed): {u}")
        if spec.get("name") and kind != "factor_model":
            from aitrading.strategy.library import TEMPLATES

            if spec["name"] in TEMPLATES:
                caveats.append("This is a published anomaly: much of the sample may overlap the original study, and published "
                               "anomalies tend to weaken after publication (McLean & Pontiff 2016).")

        # next experiments (most important first; at most five)
        nexts.append("Split the sample into halves (and by market regime) and check the sign and t-stat of alpha in each.")
        survivor = _has_survivorship_flag(result)
        if survivor or any(not d.point_in_time for d in result.data_usage):
            nexts.append("Re-run on a point-in-time universe that includes delisted names (institutional data) to remove survivorship / look-ahead bias.")
        if kind == "factor_model":
            nexts.append("Regress the constructed factors on the official ones over the overlap and inspect the periods with the largest gaps.")
        if isinstance(spec.get("costs_bps"), (int, float)) and kind != "factor_model":
            nexts.append(f"Re-run with costs doubled to {2 * float(spec['costs_bps']):g} bps one-way.")
        if kind == "cross_sectional":
            if not any(s.get("sector_neutral") for s in spec.get("signal") or [] if isinstance(s, dict)):
                nexts.append("Re-run sector-neutral to check the result is not a sector bet.")
            nexts.append("Vary the number of quantiles (5 vs 10) and the rebalance frequency to test parameter sensitivity.")
        if reg is not None and reg.model in ("capm", "ff3"):
            nexts.append("Attribute against carhart4 and ff5 to see whether momentum, profitability or investment explain the alpha.")
        nexts = nexts[:5]

        summary = f"Verdict: {verdict} - {reason}."
        if verdict in ("weak", "likely_spurious", "inconclusive"):
            summary += " Treat this result as unproven."
        elif verdict == "promising":
            summary += " Worth further testing before relying on it."
        if survivor:
            summary += " Note the survivorship-bias warning on the universe."

        interp = BacktestInterpretation(
            summary=summary,
            verdict=verdict,  # type: ignore[arg-type]
            key_findings=findings,
            cited_metrics=cites,
            biases_and_caveats=caveats,
            next_experiments=nexts,
        )
        return interp, verify_interpretation(interp, result)


# ------------------------------------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------------------------------------


def interpret_result(
    result: BacktestResult, llm: StructuredLLM | None = None, *, effort: str = "high"
) -> tuple[BacktestInterpretation, list[EvidenceCheck], str]:
    """Claude when ``llm`` is given (heuristic fallback on ``LLMError``); returns (interp, checks, interpreter name)."""
    if llm is not None:
        try:
            interp, checks = BacktestInterpreter(llm, effort=effort).interpret(result)
            return interp, checks, llm.name
        except LLMError as e:
            interp, checks = HeuristicInterpreter().interpret(result)
            interp.biases_and_caveats.append(f"Claude interpretation failed ({type(e).__name__}); this is the rule-based reading.")
            return interp, checks, "heuristic"
    interp, checks = HeuristicInterpreter().interpret(result)
    return interp, checks, "heuristic"
