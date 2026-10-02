"""Replication / robustness suite for strategy ideas (from the auto strategy creator, or any spec).

``replicate(candidate, spec, runner)`` runs the base backtest and a battery of robustness checks,
each a modified copy of the spec pushed through the same ``BacktestRunner.backtest`` code path:

* ``first_half`` / ``second_half`` - the window the base run's strategy returns actually cover
  (the result's [start, end] narrowed to its strategy statistics when the returns start or end
  materially inside it) split at its calendar midpoint; skipped when a half would be shorter than
  ``min_years_per_split`` years (the window length is the smaller of the calendar span and
  n_periods / periods_per_year);
* ``post_publication`` - only when the idea comes from a dated source: the strategy from the
  publication date to the base end, compared with the in-sample (pre-publication) Sharpe from a
  separate ``pre_publication`` reference run over [start, publication - 1 day] (McLean & Pontiff
  2016 measure post-publication decay against the in-sample period: published anomalies lose
  about a third of their returns out of sample, so a strong decay is a red flag). The full-sample
  Sharpe is the fallback reference when the pre-publication window is shorter than
  ``min_years_per_split`` or its run cannot be used, and the note says so;
* ``costs_<x>bps`` - the same strategy at each one-way transaction-cost level. Only levels ABOVE
  the base run's ``costs_bps`` are robustness checks; levels at or below it (the 0 bps gross run,
  the base level itself - which reuses the base result) cannot be harder than the base, so they
  are *informational*: reported (and used in the gross-vs-net caveat) but not counted;
* parameter perturbations relevant to the kind - cross_sectional: quintiles <-> deciles (or
  top_n doubled / halved) and monthly <-> quarterly rebalancing; screen: monthly <-> quarterly
  rebalancing; time_series: a stricter entry threshold when the rule is a single numeric
  comparison (e.g. ``price_vs_sma_200_pct > 1`` instead of ``> 0``); factor_model: none (the
  formation scheme is not a spec parameter).

Pass rules (see the ``*_RULE`` constants)
-----------------------------------------
The effect under test is a positive premium, so keeping a negative sign is never a pass.

* sub-period: Sharpe > 0 and the full-sample (base) Sharpe > 0;
* post-publication: Sharpe > 0 and >= 50% of the pre-publication Sharpe (or the full-sample
  Sharpe in the fallback case);
* costs: Sharpe > 0 at that cost level;
* perturbation: Sharpe > 0, base Sharpe > 0 and Sharpe >= 50% of the base Sharpe.
A check that could not be run or evaluated (window too short, runner error, no Sharpe, or a run
whose returns are not inside the requested window or cover fewer than ``min_years_per_split``
years, give or take one rebalance period) has ``passed=None`` and does not count in the pass
share. Informational cost checks keep their pass / fail flag but are not counted either.

Verdict
-------
* ``inconclusive`` - the base run has no Sharpe ratio (the runner returned no usable strategy
  returns), covers fewer than ``MIN_BASE_YEARS`` years, or no counted robustness check could be
  run;
* ``replicates`` - base Sharpe > 0 AND base alpha t-stat >= 2 (Sharpe >= 0.5 when there is no
  factor regression) AND >= 75% of the counted runnable checks pass AND (no usable claim, or
  replicated / claimed Sharpe >= 0.5);
* ``partially_replicates`` - base Sharpe > 0 and >= 50% of the counted runnable checks pass;
* ``fails_to_replicate`` - otherwise.

A runner failure on a check is recorded on that check (``passed=None``, note = the error) and the
suite continues; a failure of the base run (an exception, or a return value that is not a
``BacktestResult``) propagates to the caller.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Collection, Literal, Sequence

from aitrading.backtest.models import BacktestResult, PerformanceStats
from aitrading.backtest.protocols import BacktestRunner
from aitrading.discovery.models import IdeaCandidate, ReplicationReport, RobustnessCheck
from aitrading.strategy.spec import StrategySpec

__all__ = [
    "replicate",
    "replication_verdict",
    "subperiod_passed",
    "post_publication_passed",
    "cost_passed",
    "perturbation_passed",
    "Verdict",
    "MIN_BASE_YEARS",
    "SIGNIFICANT_ALPHA_T",
    "MIN_SHARPE_WITHOUT_REGRESSION",
    "REPLICATES_MIN_PASS_SHARE",
    "PARTIAL_MIN_PASS_SHARE",
    "MIN_REPLICATION_RATIO",
    "POST_PUBLICATION_MIN_SHARE",
    "PERTURBATION_MIN_SHARE",
    "SUBPERIOD_RULE",
    "POST_PUBLICATION_RULE",
    "COST_RULE",
    "COST_INFORMATIONAL_RULE",
    "PERTURBATION_RULE",
    "YEAR_TOLERANCE",
]

Verdict = Literal["replicates", "partially_replicates", "fails_to_replicate", "inconclusive"]

# ------------------------------------------------------------------------------------------------
# Rules (documented constants)
# ------------------------------------------------------------------------------------------------

#: The base backtest needs at least this many years of returns for any verdict but "inconclusive".
MIN_BASE_YEARS = 3.0
#: "Replicates" needs a factor-regression alpha t-stat of at least this (Newey-West) ...
SIGNIFICANT_ALPHA_T = 2.0
#: ... or, when the run has no factor regression, an annualised Sharpe ratio of at least this.
MIN_SHARPE_WITHOUT_REGRESSION = 0.5
#: Share of the runnable robustness checks that must pass for "replicates".
REPLICATES_MIN_PASS_SHARE = 0.75
#: Share of the runnable robustness checks that must pass for "partially_replicates".
PARTIAL_MIN_PASS_SHARE = 0.5
#: Replicated / claimed Sharpe needed for "replicates" when the source reports a Sharpe.
MIN_REPLICATION_RATIO = 0.5
#: Post-publication Sharpe must be positive and at least this share of the base Sharpe.
POST_PUBLICATION_MIN_SHARE = 0.5
#: Perturbed Sharpe must keep the base's sign and at least this share of its magnitude.
PERTURBATION_MIN_SHARE = 0.5
#: Windows this much shorter than a year threshold still count (calendar rounding, ~1 week).
YEAR_TOLERANCE = 0.02

SUBPERIOD_RULE = "passes if its Sharpe is > 0 and the full-sample Sharpe is > 0 too"
POST_PUBLICATION_RULE = (
    f"passes if its Sharpe is > 0 and at least {POST_PUBLICATION_MIN_SHARE:.0%} of the pre-publication (in-sample) "
    "Sharpe, or of the full-sample Sharpe when the pre-publication window is too short "
    "(McLean & Pontiff 2016 post-publication decay check)"
)
COST_RULE = "passes if its Sharpe is still > 0 at that cost"
COST_INFORMATIONAL_RULE = (
    "informational only: a cost at or below the base run's cannot make the test harder than the base, "
    "so it is not counted in the verdict"
)
PERTURBATION_RULE = (
    f"passes if its Sharpe is > 0 and at least {PERTURBATION_MIN_SHARE:.0%} of the base Sharpe (which must be > 0 too)"
)

_DAYS_PER_YEAR = 365.25
_MAX_NOTE = 300
_MAX_OTHER_WARNINGS = 8
_EPS = 1e-9
#: Days of slack when comparing a run's return dates with the window it was asked for.
_WINDOW_SLACK_DAYS = 7
#: Length of one rebalance period in years: the first period of a window can be spent forming the book.
_REBALANCE_YEARS = {"daily": 1 / 252, "weekly": 1 / 52, "monthly": 1 / 12, "quarterly": 0.25, "annual": 1.0}

_QUANTILE_NAMES = {2: "halves", 3: "terciles", 4: "quartiles", 5: "quintiles", 10: "deciles", 20: "vigintiles"}
_REBALANCE_ALTERNATIVE = {
    "monthly": "quarterly",
    "quarterly": "monthly",
    "weekly": "monthly",
    "daily": "weekly",
    "annual": "quarterly",
}

_SURVIVOR = re.compile(r"survivor\w*", re.I)
_NEGATED_BEFORE = re.compile(
    r"\b(?:no|without|free\s+of|avoid\w*|prevent\w*|eliminat\w*|remov\w*|correct\w*\s+for|adjust\w*\s+for)"
    r"\s+(?:[\w-]+\s+){0,2}$",
    re.I,
)
_NEGATED_AFTER = re.compile(r"^[\s-]*(?:bias[\s-]*)?(?:free|corrected|adjusted)\b", re.I)
_DATA_WARNING = re.compile(
    r"survivor|point[\s_-]*in[\s_-]*time|look[\s_-]*ahead|delist|coverage|missing|snapshot|stale|restat", re.I
)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


# ------------------------------------------------------------------------------------------------
# Pass rules
# ------------------------------------------------------------------------------------------------


def subperiod_passed(sharpe: float | None, base_sharpe: float | None) -> bool | None:
    """Sub-period rule: Sharpe > 0 and the base Sharpe > 0 too. ``None`` when either is missing."""
    if sharpe is None or base_sharpe is None:
        return None
    return sharpe > 0 and base_sharpe > 0


def post_publication_passed(sharpe: float | None, reference_sharpe: float | None) -> bool | None:
    """Post-publication rule: Sharpe > 0 and >= ``POST_PUBLICATION_MIN_SHARE`` x the reference
    Sharpe - the pre-publication (in-sample) Sharpe, or the full-sample one as a fallback. A
    positive post-publication Sharpe after a non-positive in-sample one is not a decay, so it
    passes. ``None`` when either Sharpe is missing."""
    if sharpe is None or reference_sharpe is None:
        return None
    return sharpe > 0 and sharpe >= POST_PUBLICATION_MIN_SHARE * reference_sharpe - _EPS


def cost_passed(sharpe: float | None, base_sharpe: float | None = None) -> bool | None:
    """Cost rule: Sharpe > 0 at that cost level (the base Sharpe is not used). ``None`` if missing."""
    if sharpe is None:
        return None
    return sharpe > 0


def perturbation_passed(sharpe: float | None, base_sharpe: float | None) -> bool | None:
    """Perturbation rule: Sharpe > 0, base Sharpe > 0 and Sharpe >= ``PERTURBATION_MIN_SHARE`` x
    base. The effect under test is a positive premium, so a perturbed run that keeps a negative
    (or zero) base Sharpe's sign is a failure, not robustness. ``None`` when either is missing."""
    if sharpe is None or base_sharpe is None:
        return None
    return sharpe > 0 and base_sharpe > 0 and sharpe >= PERTURBATION_MIN_SHARE * base_sharpe - _EPS


def replication_verdict(
    *,
    base_sharpe: float | None,
    base_alpha_t: float | None,
    base_years: float,
    checks: Sequence[RobustnessCheck],
    replication_ratio: float | None,
    informational: Collection[str] = (),
) -> tuple[Verdict, str]:
    """``(verdict, reason)`` from the base statistics, the checks and the replication ratio.

    ``base_alpha_t`` is ``None`` when the run has no factor regression (the Sharpe criterion is
    used instead). Checks named in ``informational`` (cost levels at or below the base run's
    costs, which cannot be harder than the base) are left out of the pass share. See the module
    docstring for the rules.
    """
    if base_sharpe is None:
        return "inconclusive", "the base backtest produced no Sharpe ratio (no usable strategy returns)"
    if base_years < MIN_BASE_YEARS - YEAR_TOLERANCE:
        return "inconclusive", f"the base backtest covers only {base_years:.1f} years (at least {MIN_BASE_YEARS:g} needed)"
    info = set(informational)
    runnable = [c for c in checks if c.passed is not None and c.name not in info]
    if not runnable:
        if any(c.passed is not None for c in checks if c.name in info):
            return "inconclusive", ("no robustness check harder than the base run could be run (cost checks at or below "
                                    "the base run's costs are informational only)")
        return "inconclusive", "none of the robustness checks could be run"
    n_pass = sum(1 for c in runnable if c.passed)
    share = n_pass / len(runnable)
    if base_alpha_t is not None:
        # t >= 2 implies a positive alpha; a negative alpha can never count as significant here
        strong = base_alpha_t >= SIGNIFICANT_ALPHA_T
        strength = (f"the alpha is significant (t = {base_alpha_t:.2f})" if strong
                    else f"the alpha is not significant (t = {base_alpha_t:.2f} < {SIGNIFICANT_ALPHA_T:g})")
    else:
        strong = base_sharpe >= MIN_SHARPE_WITHOUT_REGRESSION
        strength = (f"the Sharpe ratio of {base_sharpe:.2f} is strong (no factor regression available)" if strong
                    else f"the Sharpe ratio of {base_sharpe:.2f} is below {MIN_SHARPE_WITHOUT_REGRESSION:g} (no factor regression available)")
    claim_ok = replication_ratio is None or replication_ratio >= MIN_REPLICATION_RATIO
    tally = f"{n_pass} of {len(runnable)} checks pass"
    if base_sharpe > 0 and strong and share >= REPLICATES_MIN_PASS_SHARE - _EPS and claim_ok:
        reason = f"{strength}, {tally}"
        if replication_ratio is not None:
            reason += f" and it reaches {replication_ratio:.0%} of the claimed Sharpe"
        return "replicates", reason
    if base_sharpe > 0 and share >= PARTIAL_MIN_PASS_SHARE - _EPS:
        missing = []
        if not strong:
            missing.append(strength)
        if share < REPLICATES_MIN_PASS_SHARE - _EPS:
            missing.append(f"only {tally} (needs {REPLICATES_MIN_PASS_SHARE:.0%})")
        if not claim_ok:
            missing.append(f"it reaches only {replication_ratio:.0%} of the claimed Sharpe (needs {MIN_REPLICATION_RATIO:.0%})")
        return "partially_replicates", f"the Sharpe is positive and {tally}, but " + "; ".join(missing)
    reasons = []
    if base_sharpe <= 0:
        reasons.append(f"the full-sample Sharpe is {base_sharpe:.2f} (not positive)")
    if share < PARTIAL_MIN_PASS_SHARE - _EPS:
        reasons.append(f"only {tally}")
    return "fails_to_replicate", "; ".join(reasons)


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


@dataclass
class _Plan:
    name: str
    description: str
    rule: Callable[[float | None, float | None], bool | None]
    spec: StrategySpec | None = None
    reuse_base: bool = False
    skip_note: str = ""
    rule_kind: str = ""
    #: Not counted in the verdict (a cost level at or below the base run's costs).
    informational: bool = False
    #: One-way cost level of a cost check, in bps.
    cost_bps: float | None = None
    #: Requested [start, end] of a window check; the run must cover it (see ``_coverage_problem``).
    window: tuple[date, date] | None = None
    min_years: float = 0.0
    #: post_publication only: the pre-publication reference run, or why the full sample is used.
    ref_spec: StrategySpec | None = None
    ref_window: tuple[date, date] | None = None
    ref_skip_note: str = ""


def _finite(x) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _strategy_stats(result: BacktestResult) -> PerformanceStats | None:
    return (result.stats or {}).get("strategy")


def _calendar_years(start: date, end: date) -> float:
    return max((end - start).days, 0) / _DAYS_PER_YEAR


def _sample_years(result: BacktestResult) -> float:
    """Years of strategy returns (n_periods / periods_per_year), else the calendar window."""
    st = _strategy_stats(result)
    if st is not None and st.periods_per_year and st.periods_per_year > 0:
        return st.n_periods / st.periods_per_year
    return _calendar_years(result.start, result.end)


def _period_days(st: PerformanceStats | None) -> float:
    ppy = st.periods_per_year if st is not None else None
    return _DAYS_PER_YEAR / ppy if ppy and ppy > 0 else 31.0


def _realised_window(result: BacktestResult) -> tuple[date, date]:
    """The [start, end] the strategy returns actually cover: the result's window, narrowed to its
    strategy statistics when the returns start (or end) more than about one period inside it - data
    availability or a warm-up - so that sub-period splits split the returns, not an empty stretch.
    The start is placed one period before the first return date (that return covers the period
    ending on it)."""
    start, end = result.start, result.end
    st = _strategy_stats(result)
    if st is None:
        return start, end
    period = timedelta(days=math.ceil(_period_days(st)))
    slack = period + timedelta(days=_WINDOW_SLACK_DAYS)
    if st.start - start > slack:
        start = st.start - period
    if end - st.end > slack:
        end = st.end
    return min(start, end), end


def _coverage_problem(result: BacktestResult, window: tuple[date, date], min_years: float, rebalance: str,
                      what: str = "the run") -> str | None:
    """Why a window run does not test its window, or ``None`` when it does: its strategy returns
    must lie inside [start, end] (give or take a week, plus one period at the end for period-end
    labels) and cover at least ``min_years`` years, give or take one rebalance period (the first
    period of a window can be spent forming the book)."""
    st = _strategy_stats(result)
    if st is None:
        return None  # evaluated as "no Sharpe ratio" by the rule note
    req_start, req_end = window
    end_slack = timedelta(days=math.ceil(_period_days(st)) + _WINDOW_SLACK_DAYS)
    if st.start < req_start - timedelta(days=_WINDOW_SLACK_DAYS) or st.end > req_end + end_slack:
        return (f"{what}'s returns ({st.start.isoformat()} to {st.end.isoformat()}) are not inside the requested "
                f"window ({req_start.isoformat()} to {req_end.isoformat()}), so it does not test that period")
    years = _sample_years(result)
    period_years = 1.0 / st.periods_per_year if st.periods_per_year and st.periods_per_year > 0 else 1 / 12
    slack = max(_REBALANCE_YEARS.get(rebalance, 1 / 12), period_years)
    if years < min_years - slack - YEAR_TOLERANCE:
        return f"{what} has only {years:.1f} years of returns in the requested window ({min_years:g} required)"
    return None


def _n_returns(result: BacktestResult) -> int:
    st = _strategy_stats(result)
    if st is not None:
        return int(st.n_periods)
    return sum(1 for r in (result.returns or {}).get("strategy", []) if r is not None)


def _alpha_t(result: BacktestResult) -> float | None:
    return _finite(result.regression.alpha_t_stat) if result.regression is not None else None


def _fmt(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _bps(x: float) -> str:
    return f"{x:g}"


def _qname(n: int) -> str:
    return _QUANTILE_NAMES.get(n, f"{n}_quantiles")


def _trim(text: str, limit: int = _MAX_NOTE) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _mentions_survivorship(text: str) -> bool:
    for m in _SURVIVOR.finditer(text or ""):
        before = text[max(0, m.start() - 40): m.start()]
        after = text[m.end(): m.end() + 20]
        if _NEGATED_BEFORE.search(before) or _NEGATED_AFTER.search(after):
            continue
        return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = it.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ------------------------------------------------------------------------------------------------
# Check planning
# ------------------------------------------------------------------------------------------------


def _split_plans(spec: StrategySpec, base: BacktestResult, min_years: float) -> list[_Plan]:
    start, end = _realised_window(base)
    # the smaller of the calendar span and the sample length: a sparse sample must not pass the test
    window_years = min(_calendar_years(start, end), _sample_years(base))
    half_years = window_years / 2.0
    mid = start + timedelta(days=max((end - start).days, 0) // 2)
    halves = [
        ("first_half", "First half", start, mid),
        ("second_half", "Second half", mid + timedelta(days=1), end),
    ]
    plans = []
    for name, label, s, e in halves:
        desc = f"{label} of the base run's return window ({s.isoformat()} to {e.isoformat()}); {SUBPERIOD_RULE}."
        if half_years < min_years - YEAR_TOLERANCE:
            plans.append(_Plan(name, desc, subperiod_passed, rule_kind="subperiod",
                               skip_note=f"not run: each half of the {window_years:.1f}-year return window would cover only "
                                         f"{half_years:.1f} years (< {min_years:g} required)"))
        else:
            plans.append(_Plan(name, desc, subperiod_passed, spec=spec.model_copy(update={"start": s, "end": e}),
                               rule_kind="subperiod", window=(s, e), min_years=min_years))
    return plans


def _post_publication_plan(candidate: IdeaCandidate | None, spec: StrategySpec, base: BacktestResult,
                           min_years: float) -> _Plan | None:
    if candidate is None:
        return None
    published = candidate.source.published
    if published is None:
        return _Plan("post_publication", f"Strategy from the source's publication date to the end of the backtest; {POST_PUBLICATION_RULE}.",
                     post_publication_passed, rule_kind="post_publication",
                     skip_note="not run: the source has no publication date")
    start, end = _realised_window(base)
    desc = (f"Strategy from the source's publication date ({published.isoformat()}) to {end.isoformat()}, "
            f"compared with the period before publication; {POST_PUBLICATION_RULE}.")
    if published <= start:
        return _Plan("post_publication", desc, post_publication_passed, rule_kind="post_publication",
                     skip_note=f"not run separately: the whole backtest ({start.isoformat()} on) is already after "
                               f"publication ({published.isoformat()}), so the base run is itself out of sample")
    years_after = _calendar_years(published, end)
    if years_after < min_years - YEAR_TOLERANCE:
        return _Plan("post_publication", desc, post_publication_passed, rule_kind="post_publication",
                     skip_note=f"not run: only {years_after:.1f} years of data after publication "
                               f"({published.isoformat()}); {min_years:g} required")
    plan = _Plan("post_publication", desc, post_publication_passed, rule_kind="post_publication",
                 spec=spec.model_copy(update={"start": published, "end": end}), window=(published, end),
                 min_years=min_years)
    ref_end = published - timedelta(days=1)
    years_before = _calendar_years(start, published)
    if years_before < min_years - YEAR_TOLERANCE:
        plan.ref_skip_note = (f"the pre-publication window ({start.isoformat()} to {ref_end.isoformat()}) covers only "
                              f"{years_before:.1f} years (< {min_years:g} required)")
    else:
        plan.ref_spec = spec.model_copy(update={"start": start, "end": ref_end})
        plan.ref_window = (start, ref_end)
    return plan


def _cost_plans(spec: StrategySpec, levels: Sequence[float]) -> list[_Plan]:
    plans = []
    base_bps = float(spec.costs_bps)
    for lvl in levels:
        same = math.isclose(lvl, base_bps, abs_tol=1e-12)
        informational = same or lvl < base_bps
        rule = COST_INFORMATIONAL_RULE if informational else COST_RULE
        desc = f"Same strategy with one-way transaction costs of {_bps(lvl)} bps (base: {_bps(base_bps)} bps); {rule}."
        plans.append(_Plan(f"costs_{_bps(lvl)}bps", desc, cost_passed, rule_kind="costs",
                           spec=None if same else spec.model_copy(update={"costs_bps": float(lvl)}), reuse_base=same,
                           informational=informational, cost_bps=float(lvl)))
    return plans


def _threshold_alternative(spec: StrategySpec) -> tuple[str, str, StrategySpec] | None:
    """A stricter entry threshold for a time-series rule with a single numeric comparison."""
    ts = spec.time_series
    if ts is None or len(ts.entry) != 1:
        return None
    cond = ts.entry[0]
    if cond.op not in (">", ">=", "<", "<=") or cond.other_feature or cond.value is None:
        return None
    value = float(cond.value)
    if value == 0.0:
        if not cond.feature.endswith("_pct"):
            return None
        step = 1.0  # a 1-percentage-point band around zero (a common whipsaw filter)
    else:
        step = 0.1 * abs(value)
    alt = value + step if cond.op in (">", ">=") else value - step
    new_cond = cond.model_copy(update={"value": alt})
    new_ts = ts.model_copy(update={"entry": [new_cond]})
    desc = (f"Stricter entry rule: {cond.feature} {cond.op} {alt:g} instead of {cond.op} {value:g}; "
            f"{PERTURBATION_RULE}.")
    return "stricter_entry_threshold", desc, spec.model_copy(update={"time_series": new_ts})


def _perturbation_plans(spec: StrategySpec) -> list[_Plan]:
    out: list[tuple[str, str, StrategySpec]] = []
    if spec.kind == "cross_sectional":
        pc = spec.portfolio
        if pc.selection == "quantile":
            n = pc.n_quantiles
            alt = 10 if n == 5 else (5 if n <= 10 else 10)
            side = "per side" if pc.style == "long_short" else "in the long book"
            out.append((
                f"{_qname(alt)}_instead_of_{_qname(n)}",
                f"{alt} signal buckets ({_qname(alt)}) instead of {n} ({_qname(n)}): holds the extreme "
                f"{100 / alt:.0f}% of names {side} instead of {100 / n:.0f}%; {PERTURBATION_RULE}.",
                spec.model_copy(update={"portfolio": pc.model_copy(update={"n_quantiles": alt})}),
            ))
        elif pc.selection == "top_n" and pc.top_n:
            alt = pc.top_n * 2 if pc.top_n < 50 else max(1, pc.top_n // 2)
            out.append((
                f"top_{alt}_instead_of_top_{pc.top_n}",
                f"Hold the top {alt} names instead of the top {pc.top_n}; {PERTURBATION_RULE}.",
                spec.model_copy(update={"portfolio": pc.model_copy(update={"top_n": alt})}),
            ))
    if spec.kind in ("cross_sectional", "screen"):
        alt_rb = _REBALANCE_ALTERNATIVE.get(spec.rebalance)
        if alt_rb:
            out.append((
                f"{alt_rb}_rebalance",
                f"Rebalance {alt_rb} instead of {spec.rebalance}; {PERTURBATION_RULE}.",
                spec.model_copy(update={"rebalance": alt_rb}),
            ))
    if spec.kind == "time_series":
        alt_ts = _threshold_alternative(spec)
        if alt_ts is not None:
            out.append(alt_ts)
    return [_Plan(name, desc, perturbation_passed, spec=s, rule_kind="perturbation") for name, desc, s in out]


# ------------------------------------------------------------------------------------------------
# Check evaluation
# ------------------------------------------------------------------------------------------------


def _rule_note(kind: str, sharpe: float | None, base_sharpe: float | None, passed: bool | None,
               ref_label: str = "base") -> str:
    if sharpe is None:
        return "no Sharpe ratio in the result (no usable strategy returns), so the check cannot be evaluated"
    if base_sharpe is None and kind != "costs":
        return f"Sharpe {sharpe:.2f}; the {ref_label} run has no Sharpe to compare with"
    if kind == "costs":
        return f"Sharpe {sharpe:.2f} at this cost" + ("" if passed else " (not positive)")
    if kind == "post_publication" and base_sharpe <= 0:
        return (f"Sharpe {sharpe:.2f}; the {ref_label} Sharpe {base_sharpe:.2f} is not positive, so there is no decay "
                "to measure" + ("" if passed else " (not positive)"))
    if kind == "perturbation" and base_sharpe <= 0:
        return (f"Sharpe {sharpe:.2f}; the base Sharpe {base_sharpe:.2f} is not positive, so there is no positive "
                "premium to keep")
    ratio = sharpe / base_sharpe if base_sharpe else None
    rel = (f" = {ratio:.0%} of the {ref_label} Sharpe {base_sharpe:.2f}" if ratio is not None
           else f" ({ref_label} Sharpe {base_sharpe:.2f})")
    if kind == "subperiod":
        why = "" if passed else (" (not positive)" if sharpe <= 0 else " (the full-sample Sharpe is not positive)")
        return f"Sharpe {sharpe:.2f}{rel}{why}"
    if kind == "post_publication":
        why = ""
        if not passed:
            why = (" (not positive)" if sharpe <= 0
                   else f" (needs >= {POST_PUBLICATION_MIN_SHARE:.0%} of the {ref_label} Sharpe)")
        return f"Sharpe {sharpe:.2f}{rel}{why}"
    why = "" if passed else (" (not positive)" if sharpe <= 0
                             else f" (needs >= {PERTURBATION_MIN_SHARE:.0%} of the base Sharpe)")
    return f"Sharpe {sharpe:.2f}{rel}{why}"


def _evaluate(plan: _Plan, result: BacktestResult, base_sharpe: float | None, *, reused: bool = False,
              ref_label: str = "base", ref_note: str = "", base_costs_bps: float | None = None) -> RobustnessCheck:
    """Apply ``plan.rule`` to ``result``; ``base_sharpe`` is the reference Sharpe (for
    post_publication: the pre-publication Sharpe, or the full-sample one, as ``ref_label`` says)."""
    st = _strategy_stats(result)
    sharpe = _finite(st.sharpe) if st is not None else None
    cagr = _finite(st.cagr_pct) if st is not None else None
    passed = plan.rule(sharpe, base_sharpe)
    problem = (_coverage_problem(result, plan.window, plan.min_years, plan.spec.rebalance if plan.spec else "monthly")
               if plan.window is not None and not reused else None)
    if problem is not None:
        passed = None
    s, e = (st.start, st.end) if st is not None else (result.start, result.end)
    parts = []
    if plan.informational:
        at = f" {_bps(base_costs_bps)} bps" if base_costs_bps is not None else ""
        parts.append(f"informational, not counted in the verdict (a cost at or below the base run's{at} "
                     "cannot make the test harder)")
    if reused:
        parts.append("same costs as the base run, so the base result is reused")
    parts.append(f"{s.isoformat()} to {e.isoformat()}, {_n_returns(result)} periods")
    if problem is not None:
        parts.append(f"not evaluated: {problem}")
        parts.append(f"Sharpe {_fmt(sharpe)}")
    else:
        parts.append(_rule_note(plan.rule_kind, sharpe, base_sharpe, passed, ref_label))
    if ref_note:
        parts.append(ref_note)
    return RobustnessCheck(
        name=plan.name,
        description=plan.description,
        sharpe=sharpe,
        cagr_pct=cagr,
        alpha_t_stat=_alpha_t(result),
        n_periods=_n_returns(result),
        passed=passed,
        note=_trim("; ".join(parts)),
    )


def _skipped(plan: _Plan, note: str) -> RobustnessCheck:
    return RobustnessCheck(name=plan.name, description=plan.description, sharpe=None, cagr_pct=None,
                           alpha_t_stat=None, n_periods=0, passed=None, note=_trim(note))


# ------------------------------------------------------------------------------------------------
# Caveats and summary
# ------------------------------------------------------------------------------------------------


def _claim_caveats(candidate: IdeaCandidate | None, spec: StrategySpec, base: BacktestResult,
                   base_sharpe: float | None, base_alpha_t: float | None, ratio: float | None,
                   checks: list[RobustnessCheck]) -> list[str]:
    out: list[str] = []
    if candidate is None:
        return out
    x = candidate.extraction
    claimed = _finite(x.reported_sharpe)
    claimed_t = _finite(x.reported_t_stat)
    claimed_ret = _finite(x.reported_annual_return_pct)
    base_st = _strategy_stats(base)
    base_cagr = _finite(base_st.cagr_pct) if base_st is not None else None
    gross = next((c for c in checks if c.name == f"costs_{_bps(0.0)}bps" and c.sharpe is not None), None)

    if claimed is not None or claimed_t is not None or claimed_ret is not None:
        out.append(
            "Claimed figures are not directly comparable: papers often report gross (pre-cost), long-short, "
            f"monthly or in-sample numbers, while this replication is net of {_bps(spec.costs_bps)} bps one-way costs, "
            f"annualised, on this platform's universe and data ({base.provider})."
        )
    if claimed is not None:
        if claimed <= 0:
            out.append(f"The source's Sharpe ratio ({claimed:g}) is not positive, so no replication ratio is computed.")
        elif ratio is not None and base_sharpe is not None:
            out.append(f"Replicated Sharpe {base_sharpe:.2f} vs claimed {claimed:.2f}: replication ratio {ratio:.2f}.")
            if 2.5 <= ratio <= 4.5:
                out.append(f"The claimed Sharpe {claimed:g} may be a monthly (non-annualised) figure: annualised it "
                           f"would be about {claimed * math.sqrt(12):.2f}.")
        if gross is not None and claimed > 0 and gross.sharpe is not None and not math.isclose(spec.costs_bps, 0.0):
            out.append(f"Before costs (0 bps) the replicated Sharpe is {gross.sharpe:.2f}, a gross replication ratio "
                       f"of {gross.sharpe / claimed:.2f}.")
    if claimed_ret is not None:
        if base_cagr is not None:
            gross_txt = (f" ({gross.cagr_pct:.1f}% before costs)" if gross is not None and gross.cagr_pct is not None
                         and not math.isclose(spec.costs_bps, 0.0) else "")
            out.append(f"The source claims about {claimed_ret:.1f}% a year; the replication compounded {base_cagr:.1f}% "
                       f"a year (CAGR, net of costs){gross_txt}. A claimed long-short spread or arithmetic mean is "
                       "usually higher than a net CAGR.")
        else:
            out.append(f"The source claims about {claimed_ret:.1f}% a year; the replication has no CAGR to compare.")
    if claimed_t is not None:
        rep_t = f"the replicated alpha t-stat is {base_alpha_t:.2f}" if base_alpha_t is not None else "the replication has no factor regression"
        out.append(f"The source reports a t-stat of {claimed_t:.2f}; {rep_t}. t-stats grow with the square root of the "
                   "sample length, so a shorter sample gives a smaller t-stat for the same effect.")
    if (claimed is not None or claimed_ret is not None) and spec.kind == "cross_sectional" and spec.portfolio.style == "long_only":
        out.append("This replication holds a long-only portfolio, while papers usually report the long-short spread; "
                   "the two measure different portfolios.")
    years = [int(y) for y in _YEAR.findall(x.sample_period or "")]
    if len(years) >= 2:
        s0, s1 = min(years), max(years)
        b0, b1 = base.start.year, base.end.year
        lo, hi = max(s0, b0), min(s1, b1)
        if lo > hi:
            out.append(f"The backtest ({b0}-{b1}) does not overlap the source's sample ({s0}-{s1}): it is an out-of-sample "
                       "test, so some decay versus the claim is expected.")
        else:
            out.append(f"The backtest ({b0}-{b1}) overlaps the source's sample ({s0}-{s1}) in {lo}-{hi}; that part is "
                       "in-sample for the source.")
    return out


def _cost_caveats(spec: StrategySpec, base_sharpe: float | None, checks: list[RobustnessCheck],
                  cost_levels: dict[str, float]) -> list[str]:
    """Gross-vs-net table plus a caveat per kind of cost failure: a failure at 0 bps means there is
    no gross effect, one at or below the base costs means the effect is missing at the base costs
    already, and only failures above the base costs mean the effect is killed by trading costs."""
    out: list[str] = []
    cost_checks = [c for c in checks if c.name in cost_levels and c.sharpe is not None]
    base_bps = float(spec.costs_bps)
    if math.isclose(base_bps, 0.0):
        out.append("The base run assumes zero transaction costs (gross returns); see the cost checks for net figures.")
    if not cost_checks or base_sharpe is None:
        return out
    parts = [f"{c.name.removeprefix('costs_')}: {c.sharpe:.2f}" for c in cost_checks]
    out.append(f"Gross vs net: the base Sharpe is {base_sharpe:.2f} at {_bps(base_bps)} bps one-way costs; "
               f"Sharpe by cost level - {', '.join(parts)}.")
    failed = [c for c in cost_checks if c.passed is False]
    gross = [c.name for c in failed if math.isclose(cost_levels[c.name], 0.0, abs_tol=1e-12)]
    low = [c.name for c in failed if c.name not in gross and cost_levels[c.name] <= base_bps + 1e-12]
    above = [c.name for c in failed if cost_levels[c.name] > base_bps + 1e-12]
    if gross:
        out.append(f"The Sharpe is not positive even before trading costs ({', '.join(gross)}): there is no gross "
                   "effect, so trading costs are not the reason it fails.")
    if low:
        out.append(f"The Sharpe is not positive even at or below the base run's {_bps(base_bps)} bps costs "
                   f"({', '.join(low)}), so the effect is already missing before higher costs are considered.")
    if above:
        out.append(f"The effect does not survive realistic trading costs ({', '.join(above)}).")
    return out


def _data_caveats(runs: list[tuple[str, BacktestResult]]) -> list[str]:
    out: list[str] = []
    survivor_in: list[str] = []
    non_pit: dict[tuple[str, str], str] = {}
    notes: list[str] = []
    seen_warn: dict[str, list[str]] = {}
    for label, res in runs:
        texts = [*res.warnings, *(d.notes for d in res.data_usage)]
        if any(_mentions_survivorship(t) for t in texts):
            survivor_in.append(label)
        for d in res.data_usage:
            if not d.point_in_time:
                key = (d.dataset, d.source)
                if key not in non_pit:
                    non_pit[key] = (f"Dataset '{d.dataset}' ({d.source}, {d.coverage}) is not point-in-time: possible "
                                    "look-ahead bias." + (f" {d.notes}" if d.notes else ""))
            elif d.notes and _DATA_WARNING.search(d.notes):
                notes.append(f"Data note ({d.dataset}, {d.source}): {d.notes}")
        for w in res.warnings:
            w = " ".join(str(w).split())
            if w:
                seen_warn.setdefault(w, [])
                if label not in seen_warn[w]:
                    seen_warn[w].append(label)
    if survivor_in:
        out.append("Survivorship bias: the universe misses delisted names in "
                   + ("every run" if len(survivor_in) == len(runs) else ", ".join(survivor_in))
                   + ", so returns (and this replication) are likely overstated.")
    out.extend(non_pit.values())
    out.extend(notes)

    def fmt(w: str, labels: list[str]) -> str:
        where = "" if "base" in labels else f" [check run{'s' if len(labels) > 1 else ''}: {', '.join(labels)}]"
        return _trim(f"Backtest warning: {w}{where}", 400)

    priority = [(w, ls) for w, ls in seen_warn.items() if _DATA_WARNING.search(w)]
    other = [(w, ls) for w, ls in seen_warn.items() if not _DATA_WARNING.search(w)]
    out.extend(fmt(w, ls) for w, ls in priority)
    out.extend(fmt(w, ls) for w, ls in other[:_MAX_OTHER_WARNINGS])
    if len(other) > _MAX_OTHER_WARNINGS:
        out.append(f"{len(other) - _MAX_OTHER_WARNINGS} further backtest warnings not listed.")
    return out


def _summary(spec: StrategySpec, base: BacktestResult, base_sharpe: float | None, base_alpha_t: float | None,
             base_years: float, claimed: float | None, ratio: float | None, checks: list[RobustnessCheck],
             verdict: str, reason: str, data_flag: bool, informational: Collection[str] = (),
             post_reference: tuple[float | None, str] | None = None) -> str:
    st = _strategy_stats(base)
    s, e = (st.start, st.end) if st is not None else (base.start, base.end)
    first = f"The replication earned a Sharpe ratio of {_fmt(base_sharpe)}"
    if base_alpha_t is not None and base.regression is not None:
        first += f" (alpha t-stat {base_alpha_t:.2f} against {base.regression.model})"
    first += (f" from {s.isoformat()} to {e.isoformat()} ({base_years:.1f} years), net of "
              f"{_bps(spec.costs_bps)} bps one-way costs")
    if ratio is not None:
        first += f"; the source claimed {claimed:.2f}, a replication ratio of {ratio:.2f}"
    elif claimed is not None:
        first += f"; the source claimed a Sharpe of {claimed:.2f}"
    sentences = [first + "."]

    info_names = set(informational)
    counted = [c for c in checks if c.name not in info_names]
    info = [c for c in checks if c.name in info_names]
    runnable = [c for c in counted if c.passed is not None]
    skipped = len(counted) - len(runnable)
    info_txt = ""
    if info and any(c.passed is not None for c in info):
        info_txt = (f"{', '.join(c.name for c in info)} {'is' if len(info) == 1 else 'are'} informational (costs at or "
                    f"below the base run's {_bps(spec.costs_bps)} bps) and not counted")
    if runnable:
        n_pass = sum(1 for c in runnable if c.passed)
        failed = [c.name for c in runnable if not c.passed]
        txt = f"{n_pass} of {len(runnable)} robustness checks passed"
        if failed:
            txt += f" (failed: {', '.join(failed)})"
        if skipped:
            txt += f", {skipped} could not be run"
        if info_txt:
            txt += f"; {info_txt}"
        sentences.append(txt + ".")
    elif info_txt:
        sentences.append(f"None of the robustness checks that count towards the verdict could be run; {info_txt}.")
    elif checks:
        sentences.append("None of the robustness checks could be run.")

    label = verdict.replace("_", " ")
    sentences.append(f"Verdict: {label}, because {reason}.")

    post = next((c for c in checks if c.name == "post_publication" and c.passed is not None), None)
    if post is not None and post.sharpe is not None:
        ref, ref_label = post_reference if post_reference is not None else (base_sharpe, "full-sample")
        if ref is not None and ref > 0:
            share = f" ({post.sharpe / ref:.0%} of the {ref_label} figure of {ref:.2f})"
        elif ref is not None:
            share = f" (the {ref_label} Sharpe was {ref:.2f}, so there is no decay to measure)"
        else:
            share = ""
        sentences.append(f"After publication the Sharpe ratio was {post.sharpe:.2f}{share}.")
    elif data_flag:
        sentences.append("Data caveats (survivorship or point-in-time) apply, see the caveats.")
    return " ".join(sentences)


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------


def replicate(
    candidate: IdeaCandidate | None,
    spec: StrategySpec,
    runner: BacktestRunner,
    *,
    min_years_per_split: float = 3.0,
    cost_levels_bps: tuple = (0.0, 25.0),
    progress: Callable[[str], None] | None = None,
) -> tuple[BacktestResult, ReplicationReport]:
    """Backtest ``spec`` and run the replication / robustness suite on it.

    Returns ``(base_result, report)``. ``candidate`` (the inbox idea the spec came from) supplies
    the publication date for the post-publication check and the claimed Sharpe / t-stat / annual
    return to compare with; pass ``None`` for a spec that did not come from a source document.
    Every check is a modified copy of ``spec`` run through ``runner.backtest(..., label=<check>)``;
    ``spec`` itself is never mutated. ``progress`` receives one human-readable line per step.

    Raises whatever ``runner.backtest`` raises for the base run, ``TypeError`` when the base run
    returns anything but a ``BacktestResult``, and ``ValueError`` for a non-positive
    ``min_years_per_split`` or a negative / non-finite cost level. When the post-publication check
    runs, an extra ``runner.backtest(..., label="pre_publication")`` reference run supplies the
    in-sample Sharpe it is compared with (it is not a check of its own).
    """
    if not (isinstance(min_years_per_split, (int, float)) and math.isfinite(min_years_per_split) and min_years_per_split > 0):
        raise ValueError("min_years_per_split must be a positive number of years")
    levels: list[float] = []
    for lvl in cost_levels_bps:
        v = _finite(lvl)
        if v is None or v < 0:
            raise ValueError(f"cost levels must be finite and >= 0 bps, got {lvl!r}")
        if not any(math.isclose(v, u, abs_tol=1e-12) for u in levels):
            levels.append(v)

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    say(f"Replication of '{spec.name}': running the base backtest")
    base = runner.backtest(spec, label="base")
    if not isinstance(base, BacktestResult):
        raise TypeError(f"runner returned {type(base).__name__} for the base run, not a BacktestResult")
    base_st = _strategy_stats(base)
    base_sharpe = _finite(base_st.sharpe) if base_st is not None else None
    base_alpha_t = _alpha_t(base)
    base_years = _sample_years(base)
    min_years = float(min_years_per_split)

    plans: list[_Plan] = _split_plans(spec, base, min_years)
    post = _post_publication_plan(candidate, spec, base, min_years)
    if post is not None:
        plans.append(post)
    plans.extend(_cost_plans(spec, levels))
    plans.extend(_perturbation_plans(spec))
    informational = {p.name for p in plans if p.informational}
    cost_levels = {p.name: p.cost_bps for p in plans if p.cost_bps is not None}

    checks: list[RobustnessCheck] = []
    runs: list[tuple[str, BacktestResult]] = [("base", base)]
    post_reference: tuple[float | None, str] | None = None
    n = len(plans)

    def pre_publication_reference(plan: _Plan, i: int) -> tuple[float | None, str, str]:
        """(reference Sharpe, its label, note) for the post-publication rule: the Sharpe of a
        pre-publication run, or the full-sample Sharpe (with the reason) when it cannot be used."""
        fallback = "compared with the full-sample Sharpe instead of the pre-publication one, because "
        if plan.ref_spec is None or plan.ref_window is None:
            return base_sharpe, "full-sample", fallback + plan.ref_skip_note
        say(f"Replication check {i}/{n}: {plan.name} - pre-publication reference run")
        try:
            ref = runner.backtest(plan.ref_spec, label="pre_publication")
            if not isinstance(ref, BacktestResult):
                raise TypeError(f"runner returned {type(ref).__name__}, not a BacktestResult")
        except Exception as exc:  # noqa: BLE001 - fall back to the full sample, keep going
            say(f"Replication reference run pre_publication failed: {_trim(str(exc), 200)}")
            return base_sharpe, "full-sample", fallback + f"the pre-publication run failed ({type(exc).__name__}: {_trim(str(exc), 100)})"
        runs.append(("pre_publication", ref))
        st = _strategy_stats(ref)
        sharpe = _finite(st.sharpe) if st is not None else None
        problem = _coverage_problem(ref, plan.ref_window, plan.min_years, plan.ref_spec.rebalance,
                                    what="the pre-publication run")
        if problem is None and (st is None or sharpe is None):
            problem = "the pre-publication run has no Sharpe ratio"
        if problem is not None:
            return base_sharpe, "full-sample", fallback + problem
        return sharpe, "pre-publication", (f"pre-publication reference {st.start.isoformat()} to {st.end.isoformat()}, "
                                           f"{_n_returns(ref)} periods")

    for i, plan in enumerate(plans, start=1):
        if plan.reuse_base:
            say(f"Replication check {i}/{n}: {plan.name} (same as the base run, reused)")
            checks.append(_evaluate(plan, base, base_sharpe, reused=True, base_costs_bps=float(spec.costs_bps)))
            continue
        if plan.spec is None:
            say(f"Replication check {i}/{n}: {plan.name} skipped - {plan.skip_note}")
            checks.append(_skipped(plan, plan.skip_note))
            continue
        ref_sharpe, ref_label, ref_note = base_sharpe, "base", ""
        if plan.rule_kind == "post_publication":
            ref_sharpe, ref_label, ref_note = pre_publication_reference(plan, i)
        say(f"Replication check {i}/{n}: {plan.name}")
        try:
            result = runner.backtest(plan.spec, label=plan.name)
            if not isinstance(result, BacktestResult):
                raise TypeError(f"runner returned {type(result).__name__}, not a BacktestResult")
            check = _evaluate(plan, result, ref_sharpe, ref_label=ref_label, ref_note=ref_note,
                              base_costs_bps=float(spec.costs_bps))
        except Exception as exc:  # noqa: BLE001 - a failed check must not stop the suite
            err = f"backtest failed: {type(exc).__name__}: {exc}"
            say(f"Replication check {plan.name} failed: {_trim(str(exc), 200)}")
            checks.append(_skipped(plan, err))
            continue
        if plan.rule_kind == "post_publication":
            post_reference = (ref_sharpe, ref_label)
        runs.append((plan.name, result))
        checks.append(check)

    claimed = _finite(candidate.extraction.reported_sharpe) if candidate is not None else None
    claimed_t = _finite(candidate.extraction.reported_t_stat) if candidate is not None else None
    ratio = base_sharpe / claimed if base_sharpe is not None and claimed is not None and claimed > 0 else None

    verdict, reason = replication_verdict(
        base_sharpe=base_sharpe,
        base_alpha_t=base_alpha_t,
        base_years=base_years,
        checks=checks,
        replication_ratio=ratio,
        informational=informational,
    )

    caveats: list[str] = []
    if base_years < MIN_BASE_YEARS - YEAR_TOLERANCE:
        caveats.append(f"Only {base_years:.1f} years of base returns: too short to separate skill from noise "
                       f"(at least {MIN_BASE_YEARS:g} needed for a verdict).")
    caveats.extend(_claim_caveats(candidate, spec, base, base_sharpe, base_alpha_t, ratio, checks))
    caveats.extend(_cost_caveats(spec, base_sharpe, checks, cost_levels))
    post_check = next((c for c in checks if c.name == "post_publication"), None)
    if post_check is not None and post_check.passed is None and post_check.note:
        caveats.append(f"Post-publication decay check: {post_check.note}.")
    data = _data_caveats(runs)
    caveats.extend(data)
    for u in spec.unsupported_requests:
        caveats.append(f"Not tested (could not be expressed with the platform's features): {u}")
    caveats = _dedupe(caveats)

    data_flag = any(c.startswith(("Survivorship bias", "Dataset '")) for c in data)
    summary = _summary(spec, base, base_sharpe, base_alpha_t, base_years, claimed, ratio, checks, verdict, reason,
                       data_flag, informational, post_reference)

    report = ReplicationReport(
        idea_id=candidate.idea_id if candidate is not None else spec.name,
        backtest_run_id=base.run_id,
        base_sharpe=base_sharpe,
        base_alpha_t_stat=base_alpha_t,
        claimed_sharpe=claimed,
        claimed_t_stat=claimed_t,
        replication_ratio=ratio,
        checks=checks,
        verdict=verdict,
        summary=summary,
        caveats=caveats,
    )
    n_run = sum(1 for c in checks if c.passed is not None and c.name not in informational)
    n_pass = sum(1 for c in checks if c.passed and c.name not in informational)
    say(f"Replication verdict: {verdict.replace('_', ' ')} ({n_pass}/{n_run} checks passed)")
    return base, report
