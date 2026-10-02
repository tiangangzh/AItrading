"""Backtest interpretation: metric paths, citation verification, Claude reviewer (ScriptedLLM), heuristic rules."""

from __future__ import annotations

import json
import math
from datetime import date, datetime

import pytest

from aitrading.backtest.models import (
    BacktestInterpretation,
    BacktestResult,
    CitedMetric,
    DataUsage,
    FactorConstructionCheck,
    FactorRegression,
    PerformanceStats,
    QuantileAnalysis,
)
from aitrading.llm.base import LLMError, ScriptedLLM
from aitrading.strategy.interpret import (
    INTERPRET_SYSTEM_PROMPT,
    THRESHOLDS,
    BacktestInterpreter,
    HeuristicInterpreter,
    annual_cost_drag_pct,
    apply_verdict_caps,
    interpret_result,
    interpretation_payload,
    rebalances_per_year,
    resolve_metric_path,
    verdict_caps,
    verify_interpretation,
)

SENTINEL_RETURN = 0.0123456789


def _stats(label: str, *, sharpe=0.9, n=120, ppy=12.0, cagr=6.0, turnover=30.0, t=2.1) -> PerformanceStats:
    return PerformanceStats(
        label=label, start=date(2015, 1, 31), end=date(2024, 12, 31), n_periods=n, periods_per_year=ppy,
        total_return_pct=79.08, cagr_pct=cagr, volatility_pct=10.0, sharpe=sharpe, sortino=1.2, max_drawdown_pct=-18.5,
        max_drawdown_duration_periods=14, calmar=0.32, hit_rate_pct=56.0, best_period_pct=6.1, worst_period_pct=-7.3,
        skew=-0.2, excess_kurtosis=1.1, mean_return_t_stat=t, avg_turnover_pct=turnover,
    )


def make_result(
    *,
    sharpe: float | None = 0.9,
    alpha_t: float | None = 3.4,
    mono: float | None = 0.9,
    n_periods: int = 120,
    kind: str = "cross_sectional",
    name: str = "momentum_12_1",
    regression: bool = True,
    quantiles: bool = True,
    warnings: list[str] | None = None,
    betas: dict[str, float] | None = None,
    beta_t: dict[str, float] | None = None,
    r2: float = 0.3,
    turnover: float = 30.0,
    cagr: float = 6.0,
    extra_stats: dict[str, PerformanceStats] | None = None,
    factor_checks: list[FactorConstructionCheck] | None = None,
    data_usage: list[DataUsage] | None = None,
    spec_extra: dict | None = None,
) -> BacktestResult:
    stats = {"strategy": _stats("strategy", sharpe=sharpe, n=n_periods, cagr=cagr, turnover=turnover),
             "benchmark": _stats("benchmark", sharpe=0.6, n=n_periods, cagr=9.0, turnover=None)}
    stats.update(extra_stats or {})
    spec = {"name": name, "kind": kind, "costs_bps": 10.0, "rebalance": "monthly",
            "signal": [{"feature": "return_12m_ex_1m_pct", "direction": "higher_is_better", "sector_neutral": False}],
            "unsupported_requests": []}
    spec.update(spec_extra or {})
    return BacktestResult(
        run_id="run-1", idea="12-1 momentum", spec=spec, provider="synthetic", llm="none",
        start=date(2015, 1, 31), end=date(2024, 12, 31), rebalance="monthly",
        returns={"strategy": [SENTINEL_RETURN] * 3, "benchmark": [0.01, None, 0.02]},
        dates=[date(2024, 10, 31), date(2024, 11, 30), date(2024, 12, 31)],
        stats=stats,
        regression=FactorRegression(
            model="ff3", factor_source="Kenneth French Data Library", n=n_periods, alpha_annual_pct=3.1,
            alpha_t_stat=alpha_t if alpha_t is not None else 0.0,
            betas=betas or {"Mkt-RF": 0.12, "SMB": 0.21, "HML": -0.34},
            beta_t_stats=beta_t or {"Mkt-RF": 1.1, "SMB": 1.9, "HML": -2.2}, r_squared=r2,
        ) if regression else None,
        quantiles=QuantileAnalysis(
            n_quantiles=5, annual_return_by_quantile_pct=[1.0, 3.0, 5.0, 6.0, 9.0], spread_annual_pct=8.0,
            monotonicity=mono if mono is not None else 0.0, ic_mean=0.031, ic_t_stat=2.5, ic_hit_rate_pct=58.0,
        ) if quantiles else None,
        factor_checks=factor_checks or [],
        data_usage=data_usage or [DataUsage(dataset="prices", source="synthetic", coverage="150/150 tickers", point_in_time=True)],
        warnings=warnings or [],
        started_at=datetime(2026, 10, 2, 9, 0, 0),
    )


# ------------------------------------------------------------------------------------------------
# resolve_metric_path
# ------------------------------------------------------------------------------------------------


@pytest.fixture
def res() -> BacktestResult:
    return make_result(
        betas={"Mkt-RF": 0.12, "SMB": 0.21, "HML": -0.34, "Mom": 0.5},
        beta_t={"Mkt-RF": 1.1, "SMB": 1.9, "HML": -2.2, "Mom": 4.0},
        factor_checks=[
            FactorConstructionCheck(factor="SMB", correlation_with_official=0.62, annual_premium_constructed_pct=1.1,
                                    annual_premium_official_pct=1.9, n_overlap_periods=118),
            FactorConstructionCheck(factor="HML", correlation_with_official=0.81, annual_premium_constructed_pct=None,
                                    annual_premium_official_pct=2.4, n_overlap_periods=118),
        ],
        data_usage=[DataUsage(dataset="prices", source="yahoo", coverage="x", point_in_time=True),
                    DataUsage(dataset="fundamentals", source="sec", coverage="y", point_in_time=False)],
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        ("stats.strategy.sharpe", 0.9),
        ("stats.benchmark.cagr_pct", 9.0),
        ("stats.strategy.n_periods", 120.0),
        ("regression.alpha_t_stat", 3.4),
        ("regression.betas.HML", -0.34),
        ("regression.betas.hml", -0.34),            # case-insensitive key
        ("regression.betas.MktRF", 0.12),           # punctuation-insensitive key
        ("regression.betas.UMD", 0.5),              # UMD alias of Mom
        ("regression.beta_t_stats.Mkt-RF", 1.1),
        ("quantiles.monotonicity", 0.9),
        ("quantiles.annual_return_by_quantile_pct.0", 1.0),
        ("quantiles.annual_return_by_quantile_pct.4", 9.0),
        ("quantiles.annual_return_by_quantile_pct.-1", 9.0),
        ("quantiles.annual_return_by_quantile_pct[1]", 3.0),
        ("factor_checks.HML.correlation_with_official", 0.81),
        ("factor_checks.smb.annual_premium_official_pct", 1.9),
        ("factor_checks.0.n_overlap_periods", 118.0),
        ("factor_checks['SMB'].correlation_with_official", 0.62),
        ("data_usage.fundamentals.point_in_time", 0.0),
        ("result.stats.strategy.sharpe", 0.9),
        ("returns.strategy.0", SENTINEL_RETURN),
    ],
)
def test_resolve_metric_path(res, path, expected):
    assert resolve_metric_path(res, path) == pytest.approx(expected)


@pytest.mark.parametrize(
    "path",
    [
        "", "stats", "stats.strategy", "stats.nope.sharpe", "stats.strategy.nope", "regression.model",
        "quantiles.annual_return_by_quantile_pct.5", "quantiles.annual_return_by_quantile_pct.x",
        "factor_checks.CMA.correlation_with_official", "factor_checks.HML.annual_premium_constructed_pct",
        "factor_checks.HML", "start", "returns.benchmark.1", "regression.betas.HML.x", "stats.strategy.avg_turnover_pct.0",
    ],
)
def test_resolve_metric_path_unresolvable(res, path):
    assert resolve_metric_path(res, path) is None


def test_resolve_none_sharpe_and_missing_sections():
    r = make_result(sharpe=None, regression=False, quantiles=False)
    assert resolve_metric_path(r, "stats.strategy.sharpe") is None
    assert resolve_metric_path(r, "regression.alpha_t_stat") is None
    assert resolve_metric_path(r, "quantiles.monotonicity") is None


# ------------------------------------------------------------------------------------------------
# verify_interpretation
# ------------------------------------------------------------------------------------------------


def _interp(*cites: tuple[str, float], verdict="promising") -> BacktestInterpretation:
    return BacktestInterpretation(
        summary="s", verdict=verdict, key_findings=["k"],
        cited_metrics=[CitedMetric(path=p, value=v, meaning="m") for p, v in cites],
        biases_and_caveats=[], next_experiments=[],
    )


def _status(result, path, value, **kw):
    return verify_interpretation(_interp((path, value)), result, **kw)[0].status


def test_verification_check_fields(res):
    [chk] = verify_interpretation(_interp(("stats.strategy.sharpe", 0.9)), res)
    assert chk.kind == "quant" and chk.ref == "stats.strategy.sharpe" and chk.claim == "stats.strategy.sharpe=0.9"
    assert chk.status == "verified" and "actual=0.9" in chk.detail


def test_verification_tolerances():
    r = make_result(cagr=10.0)  # stats.strategy.cagr_pct = 10.0
    # default rel_tol 1%: |10.08 - 10| = 0.08 <= 0.1 -> verified; |10.15 - 10| = 0.15 > 0.1 -> mismatch
    assert _status(r, "stats.strategy.cagr_pct", 10.08) == "verified"
    assert _status(r, "stats.strategy.cagr_pct", 9.91) == "verified"
    assert _status(r, "stats.strategy.cagr_pct", 10.15) == "mismatch"
    assert _status(r, "stats.strategy.cagr_pct", 10.3) == "mismatch"
    # no blanket abs_tol: ic_mean = 0.031 only verifies as itself or rounded to the written decimals
    assert _status(r, "quantiles.ic_mean", 0.031) == "verified"
    assert _status(r, "quantiles.ic_mean", 0.03) == "verified"    # 0.031 rounds to 0.03
    assert _status(r, "quantiles.ic_mean", 0.04) == "mismatch"
    assert _status(r, "quantiles.ic_mean", 0.07) == "mismatch"    # was 'verified' under abs_tol 0.05
    # a materially wrong t-stat is caught: actual 2.6 cited as 3.0
    r2 = make_result(alpha_t=2.6)
    assert _status(r2, "regression.alpha_t_stat", 3.0) == "mismatch"
    assert _status(r2, "regression.alpha_t_stat", 2.6) == "verified"
    # an explicit abs_tol is still honoured, but never across a sign flip
    assert _status(r, "quantiles.ic_mean", 0.07, abs_tol=0.05) == "verified"
    assert _status(r, "quantiles.ic_mean", -0.01, abs_tol=0.05) == "mismatch"


def test_verification_reviewer_cases_sign_and_thresholds():
    """Regression: under the old abs_tol=0.05 / rel_tol=2% every one of these was 'verified'."""
    r = make_result(alpha_t=2.95, mono=0.76, betas={"Mkt-RF": 0.12, "SMB": 0.21, "HML": 0.03})
    r.quantiles.ic_mean = 0.02
    # sign flips on small statistics
    [chk] = verify_interpretation(_interp(("quantiles.ic_mean", -0.03)), r)
    assert chk.status == "mismatch" and "sign differs" in chk.detail
    assert _status(r, "regression.betas.HML", -0.02) == "mismatch"
    # a zero citation of a non-zero value (0.02 'rounds' to 0.0 at one decimal) is not accepted either
    [chk] = verify_interpretation(_interp(("quantiles.ic_mean", 0.0)), r)
    assert chk.status == "mismatch" and "zero citation" in chk.detail
    # rounding across a verdict threshold: t 2.95 -> "3.0" would clear the Harvey-Liu-Zhu hurdle
    [chk] = verify_interpretation(_interp(("regression.alpha_t_stat", 3.0)), r)
    assert chk.status == "mismatch" and "other side of the 3 |t| threshold" in chk.detail
    assert chk.detail.startswith("cited 3 but the result has 2.95")
    assert _status(r, "regression.alpha_t_stat", 2.95) == "verified"
    assert _status(r, "regression.alpha_t_stat", 2.9) == "mismatch"   # rounds to 3.0, not 2.9 -> plain mismatch
    # monotonicity 0.76 -> "0.8" would read as robustly monotonic
    [chk] = verify_interpretation(_interp(("quantiles.monotonicity", 0.8)), r)
    assert chk.status == "mismatch" and "0.8 verdict threshold" in chk.detail
    assert _status(r, "quantiles.monotonicity", 0.76) == "verified"
    # the same rounding is fine when no threshold is crossed
    assert _status(make_result(alpha_t=3.04), "regression.alpha_t_stat", 3.0) == "verified"
    assert _status(make_result(mono=0.84), "quantiles.monotonicity", 0.8) == "verified"
    # Sharpe 0.79 vs 0.8, correlation 0.69 vs 0.7, R-squared 0.79 vs 0.8
    assert _status(make_result(sharpe=0.79), "stats.strategy.sharpe", 0.8) == "mismatch"
    assert _status(make_result(sharpe=0.83), "stats.strategy.sharpe", 0.8) == "verified"
    rf = make_result(factor_checks=[FactorConstructionCheck(factor="HML", correlation_with_official=0.69, annual_premium_constructed_pct=1.0,
                                                            annual_premium_official_pct=1.0, n_overlap_periods=100)], r2=0.79)
    assert _status(rf, "factor_checks.HML.correlation_with_official", 0.7) == "mismatch"
    assert _status(rf, "regression.r_squared", 0.8) == "mismatch"
    # negative t-stats are gated on |t|: -2.96 cited as -3.0 is a mismatch, -3.04 as -3.0 is fine
    rn = make_result(beta_t={"Mkt-RF": 1.1, "SMB": -3.04, "HML": -2.96})
    assert _status(rn, "regression.beta_t_stats.HML", -3.0) == "mismatch"
    assert _status(rn, "regression.beta_t_stats['SMB']", -3.0) == "verified"
    # payload precision: a t-stat of 2.99999 is shown to Claude as 3.0, so citing 3.0 is faithful
    assert _status(make_result(alpha_t=2.99999), "regression.alpha_t_stat", 3.0) == "verified"
    # tiny values that are zero at payload precision may be cited as 0
    r0 = make_result()
    r0.quantiles.ic_mean = 0.00003
    assert _status(r0, "quantiles.ic_mean", 0.0) == "verified"


def test_verification_rounding_rule_with_zero_tolerance():
    r = make_result(alpha_t=1.2345)
    kw = dict(rel_tol=0.0, abs_tol=0.0)
    assert _status(r, "regression.alpha_t_stat", 1.2345, **kw) == "verified"
    assert _status(r, "regression.alpha_t_stat", 1.23, **kw) == "verified"    # rounded to 2 decimals written
    assert _status(r, "regression.alpha_t_stat", 1.235, **kw) == "verified"   # half-up at 3 decimals
    assert _status(r, "regression.alpha_t_stat", 1.2, **kw) == "verified"
    assert _status(r, "regression.alpha_t_stat", 1.24, **kw) == "mismatch"
    assert _status(r, "regression.alpha_t_stat", 1.0, **kw) == "mismatch"    # integer citations read as "1.0"
    r3 = make_result(alpha_t=3.04)
    assert _status(r3, "regression.alpha_t_stat", 3.0, **kw) == "verified"
    r4 = make_result(alpha_t=2.6)
    assert _status(r4, "regression.alpha_t_stat", 3.0, **kw) == "mismatch"
    r5 = make_result(alpha_t=-1.25)
    assert _status(r5, "regression.alpha_t_stat", -1.3, **kw) == "verified"   # half away from zero


def test_verification_not_found_and_nan(res):
    checks = verify_interpretation(
        _interp(("stats.strategy.made_up", 1.0), ("regression.betas.RMW", 0.1), ("stats.strategy.sharpe", math.nan)), res
    )
    assert [c.status for c in checks] == ["not_found", "not_found", "mismatch"]
    assert "does not resolve" in checks[0].detail


# ------------------------------------------------------------------------------------------------
# Claude interpreter (ScriptedLLM)
# ------------------------------------------------------------------------------------------------


def _seq_responder(*outs):
    seq = list(outs)

    def responder(purpose, system, user, output_model):
        assert output_model is BacktestInterpretation
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return responder


def test_user_prompt_excludes_returns_and_includes_sections():
    r = make_result(warnings=["Universe is today's constituents: survivorship bias."])
    user = BacktestInterpreter.user_prompt(r)
    assert str(SENTINEL_RETURN) not in user and '"returns"' not in user and '"dates"' not in user
    body = user.split("<result>\n", 1)[1].split("\n</result>", 1)[0]
    payload = json.loads(body)
    assert set(payload) == {"idea", "run", "spec", "stats", "regression", "quantiles", "factor_checks", "data_usage", "warnings"}
    assert payload["stats"]["strategy"]["sharpe"] == 0.9
    assert payload["regression"]["betas"]["HML"] == -0.34
    assert payload["run"]["years"] == 10.0
    assert payload["warnings"] == ["Universe is today's constituents: survivorship bias."]
    assert payload == interpretation_payload(r)


def test_payload_rounds_floats_to_4_decimals():
    r = make_result(alpha_t=3.123456789)
    assert interpretation_payload(r)["regression"]["alpha_t_stat"] == 3.1235


def test_interpreter_no_repair_when_all_verified():
    r = make_result()
    good = _interp(("stats.strategy.sharpe", 0.9), ("regression.alpha_t_stat", 3.4), verdict="robust")
    llm = ScriptedLLM({"interpret": _seq_responder(good)})
    interp, checks = BacktestInterpreter(llm).interpret(r)
    assert interp == good and [c.status for c in checks] == ["verified", "verified"]
    assert [c.purpose for c in llm.calls] == ["interpret"]
    assert llm.prompts[0]["system"] == INTERPRET_SYSTEM_PROMPT


def test_interpreter_repair_loop_fixes_bad_citation():
    r = make_result()
    bad = _interp(("stats.strategy.sharpe", 1.4), ("regression.alpha_t_stat", 3.4), ("regression.betas.UMDX", 0.2))
    good = _interp(("stats.strategy.sharpe", 0.9), ("regression.alpha_t_stat", 3.4))
    llm = ScriptedLLM({"interpret": _seq_responder(bad, good)})
    interp, checks = BacktestInterpreter(llm).interpret(r)
    assert [c.purpose for c in llm.calls] == ["interpret", "interpret:repair"]
    assert interp == good and all(c.status == "verified" for c in checks)
    repair_user = llm.prompts[1]["user"]
    assert "stats.strategy.sharpe: mismatch (cited 1.4 but the result has 0.9)" in repair_user
    assert "regression.betas.UMDX: not_found" in repair_user
    assert "<previous_interpretation>" in repair_user and "<result>" in repair_user
    assert llm.prompts[0]["system"] == llm.prompts[1]["system"]


def test_interpreter_keeps_better_verified_result():
    r = make_result()
    first = _interp(("stats.strategy.sharpe", 0.9), ("regression.alpha_t_stat", 9.9))       # 1/2 verified
    worse = _interp(("stats.strategy.sharpe", 5.0), ("regression.alpha_t_stat", 9.9))       # 0/2 verified
    llm = ScriptedLLM({"interpret": _seq_responder(first, worse)})
    interp, checks = BacktestInterpreter(llm).interpret(r)
    assert len(llm.calls) == 2
    assert interp == first and [c.status for c in checks] == ["verified", "mismatch"]


def test_interpreter_respects_max_repair_rounds():
    r = make_result()
    bad = _interp(("stats.strategy.sharpe", 5.0))
    good = _interp(("stats.strategy.sharpe", 0.9))
    llm = ScriptedLLM({"interpret": _seq_responder(bad, good)})
    interp, checks = BacktestInterpreter(llm, max_repair_rounds=0).interpret(r)
    assert len(llm.calls) == 1 and checks[0].status == "mismatch"
    llm2 = ScriptedLLM({"interpret": _seq_responder(bad, bad, good)})
    interp, checks = BacktestInterpreter(llm2, max_repair_rounds=2).interpret(r)
    assert [c.purpose for c in llm2.calls] == ["interpret", "interpret:repair", "interpret:repair"]
    assert interp == good


def test_interpreter_repairs_when_nothing_is_cited():
    r = make_result()
    empty = _interp()
    good = _interp(("quantiles.monotonicity", 0.9))
    llm = ScriptedLLM({"interpret": _seq_responder(empty, good)})
    interp, checks = BacktestInterpreter(llm).interpret(r)
    assert "no metrics were cited" in llm.prompts[1]["user"]
    assert interp == good and checks[0].status == "verified"


def test_interpreter_system_prompt_byte_stable_and_skeptical():
    llm = ScriptedLLM({"interpret": _seq_responder(_interp(("stats.strategy.sharpe", 0.9)))})
    BacktestInterpreter(llm).interpret(make_result())
    BacktestInterpreter(llm).interpret(make_result(sharpe=0.1, n_periods=36, warnings=["w"]))
    assert llm.prompts[0]["system"] == llm.prompts[1]["system"] == INTERPRET_SYSTEM_PROMPT
    for needle in ["Harvey, Liu & Zhu 2016", "t-stat above 3", "monotonicity", "Survivorship", "look-ahead",
                   "costs_bps", "regime", "outside knowledge", "dotted path", "factor_checks.HML.correlation_with_official",
                   "inconclusive", "likely_spurious"]:
        assert needle in INTERPRET_SYSTEM_PROMPT


def test_interpreter_llm_error_propagates_and_fallback():
    def boom(*a):
        raise LLMError("down")

    with pytest.raises(LLMError):
        BacktestInterpreter(ScriptedLLM({"interpret": boom})).interpret(make_result())
    interp, checks, who = interpret_result(make_result(), ScriptedLLM({"interpret": boom}))
    assert who == "heuristic" and interp.verdict == "robust"
    assert any("Claude interpretation failed" in c for c in interp.biases_and_caveats)
    assert interpret_result(make_result())[2] == "heuristic"


# ------------------------------------------------------------------------------------------------
# Heuristic interpreter: verdict rules
# ------------------------------------------------------------------------------------------------

H = HeuristicInterpreter()


@pytest.mark.parametrize(
    "kw,verdict",
    [
        (dict(n_periods=24, sharpe=2.0, alpha_t=6.0, mono=1.0), "inconclusive"),          # 2 years < 3
        (dict(n_periods=35, sharpe=2.0, alpha_t=6.0, mono=1.0), "inconclusive"),          # 2.92 years
        (dict(n_periods=36, sharpe=0.8, alpha_t=3.0, mono=0.8), "robust"),                # boundaries inclusive
        (dict(sharpe=0.9, alpha_t=3.4, mono=0.9), "robust"),
        (dict(sharpe=0.8, alpha_t=3.0, quantiles=False), "robust"),                       # quantiles absent allowed
        (dict(sharpe=0.9, alpha_t=3.5, mono=0.5), "promising"),                           # robust blocked by mono
        (dict(sharpe=0.79, alpha_t=3.5, mono=0.9), "promising"),                          # robust blocked by Sharpe
        (dict(sharpe=0.3, alpha_t=2.5, mono=0.1), "promising"),                           # alpha t >= 2
        (dict(sharpe=0.6, alpha_t=1.5, mono=0.7), "promising"),                           # Sharpe & mono
        (dict(sharpe=0.9, alpha_t=1.5, quantiles=False), "weak"),                         # no mono -> no 2nd route
        (dict(sharpe=0.3, alpha_t=1.5, mono=0.5), "weak"),
        (dict(sharpe=0.2, alpha_t=0.5, mono=0.1), "likely_spurious"),
        (dict(sharpe=0.2, alpha_t=0.5, mono=-0.6), "likely_spurious"),
        (dict(sharpe=0.2, alpha_t=0.5, quantiles=False), "weak"),                         # spurious needs mono
        (dict(sharpe=1.5, regression=False, quantiles=False), "weak"),                    # no alpha -> not robust
        (dict(sharpe=None, alpha_t=2.0, mono=0.2), "promising"),
    ],
)
def test_heuristic_verdict_rules(kw, verdict):
    r = make_result(**kw)
    interp, checks = H.interpret(r)
    assert interp.verdict == verdict
    assert H.verdict(r)[0] == verdict
    assert interp.summary.startswith(f"Verdict: {verdict}")
    assert checks and all(c.status == "verified" for c in checks)


def test_thresholds_documented_values():
    assert THRESHOLDS["min_years"] == 3 and THRESHOLDS["robust_alpha_t"] == 3 and THRESHOLDS["promising_alpha_t"] == 2
    assert THRESHOLDS["robust_sharpe"] == 0.8 and THRESHOLDS["robust_monotonicity"] == 0.8
    assert THRESHOLDS["promising_sharpe"] == 0.5 and THRESHOLDS["promising_monotonicity"] == 0.6
    assert THRESHOLDS["spurious_alpha_t"] == 1 and THRESHOLDS["spurious_monotonicity"] == 0.3


def test_heuristic_cites_exact_resolved_values():
    r = make_result()
    interp, checks = H.interpret(r)
    paths = {c.path for c in interp.cited_metrics}
    assert {"stats.strategy.sharpe", "stats.strategy.cagr_pct", "regression.alpha_t_stat", "regression.alpha_annual_pct",
            "quantiles.monotonicity", "quantiles.annual_return_by_quantile_pct.0",
            "quantiles.annual_return_by_quantile_pct.4", "regression.betas.HML"} <= paths
    for c in interp.cited_metrics:
        assert c.value == resolve_metric_path(r, c.path)
    assert len(checks) == len(interp.cited_metrics) and all(c.status == "verified" for c in checks)
    # tight tolerance still verifies (values are exact)
    assert all(c.status == "verified" for c in verify_interpretation(interp, r, rel_tol=0.0, abs_tol=0.0))


def test_heuristic_skips_none_metrics():
    r = make_result(sharpe=None)
    interp, _ = H.interpret(r)
    assert "stats.strategy.sharpe" not in {c.path for c in interp.cited_metrics}


def test_heuristic_caveats_survivorship_data_and_unsupported():
    r = make_result(
        warnings=["Universe is today's constituents: survivorship bias (delisted names missing).", "3 names lacked prices."],
        data_usage=[DataUsage(dataset="estimates", source="snapshot", coverage="today only", point_in_time=False, notes="no history")],
        spec_extra={"unsupported_requests": ["stop-loss exits"]},
    )
    interp, _ = H.interpret(r)
    cav = " | ".join(interp.biases_and_caveats)
    assert "Engine warning: Universe is today's constituents: survivorship bias" in cav
    assert "Engine warning: 3 names lacked prices." in cav
    assert "Dataset 'estimates' (snapshot, today only) is not point-in-time" in cav and "no history" in cav
    assert "Not tested (could not be expressed): stop-loss exits" in cav
    assert "McLean & Pontiff" in cav  # published-anomaly decay for library templates
    assert "survivorship-bias warning" in interp.summary
    assert any("delisted names" in n for n in interp.next_experiments)


def test_heuristic_cost_drag_hand_checked():
    # The engine charges costs on traded notional = 2 x one-way turnover:
    # 2 x 50% one-way turnover x 12 rebalances x 10 bps = 100% x 12 x 0.10% = 1.20% a year;
    # CAGR 2% -> drag >= 1% (and >= 25% of CAGR) -> cost caveat
    interp, _ = H.interpret(make_result(turnover=50.0, cagr=2.0))
    assert any("about 1.20% a year of trading costs" in c and "cost-sensitive" in c for c in interp.biases_and_caveats)
    assert any("(100.0% of the book traded) x 12 rebalances a year" in c for c in interp.biases_and_caveats)
    # 10% turnover -> 2 x 10 x 12 x 10 / 10000 = 0.24% a year; CAGR 6% -> a finding, not a caveat
    interp, _ = H.interpret(make_result(turnover=10.0, cagr=6.0))
    assert any("about 0.24% a year of trading costs" in f for f in interp.key_findings)
    assert not any("trading costs" in c for c in interp.biases_and_caveats)
    # 30% turnover -> 0.72% a year; CAGR 3%: 0.72 < 1 and 0.72 < 0.25 x 3 = 0.75 -> a finding
    interp, _ = H.interpret(make_result(turnover=30.0, cagr=3.0))
    assert any("about 0.72% a year of trading costs" in f for f in interp.key_findings)
    # ... while the old (halved) formula put a 0.6% drag at 50% turnover below the 1% caveat line
    interp, _ = H.interpret(make_result(turnover=50.0, cagr=6.0))
    assert any("about 1.20% a year" in c and "cost-sensitive" in c for c in interp.biases_and_caveats)


def test_cost_drag_helpers():
    assert annual_cost_drag_pct(50.0, 12, 10.0) == pytest.approx(1.2)
    assert annual_cost_drag_pct(100.0, 52, 5.0) == pytest.approx(5.2)   # 2 x 100% x 52 x 0.05%
    assert annual_cost_drag_pct(25.0, 4, 20.0) == pytest.approx(0.4)    # 2 x 25% x 4 x 0.20%
    r = make_result()
    assert rebalances_per_year(r) == 12
    for freq, n in [("daily", 252), ("weekly", 52), ("quarterly", 4), ("annual", 1), ("Monthly", 12)]:
        assert rebalances_per_year(r.model_copy(update={"rebalance": freq})) == n
    # result.rebalance unknown -> falls back to spec.rebalance; both unknown -> None
    assert rebalances_per_year(r.model_copy(update={"rebalance": "?"})) == 12
    assert rebalances_per_year(r.model_copy(update={"rebalance": "?", "spec": {**r.spec, "rebalance": "weekly"}})) == 52
    assert rebalances_per_year(r.model_copy(update={"rebalance": "?", "spec": {}})) is None


def test_cost_drag_uses_rebalance_frequency_not_return_frequency():
    # Stats computed on DAILY returns (periods_per_year 252) of a MONTHLY strategy: the drag must
    # scale by 12 rebalances, not by 252 return periods (the old formula gave 50 x 252 x 10 / 1e4 = 12.6%).
    r = make_result(turnover=50.0, cagr=2.0)
    r.stats["strategy"] = _stats("strategy", ppy=252.0, n=2520, cagr=2.0, turnover=50.0)
    interp, _ = H.interpret(r)
    text = " | ".join(interp.biases_and_caveats + interp.key_findings)
    assert "about 1.20% a year of trading costs" in text and "12.60%" not in text
    # quarterly rebalancing: 2 x 50 x 4 x 10 / 1e4 = 0.40% (CAGR 6% -> a finding)
    rq = make_result(turnover=50.0, cagr=6.0).model_copy(update={"rebalance": "quarterly"})
    interp, _ = H.interpret(rq)
    assert any("about 0.40% a year of trading costs" in f and "x 4 rebalances a year" in f for f in interp.key_findings)
    # unknown rebalance frequency -> no cost-drag statement rather than a wrong one
    ru = make_result(turnover=50.0).model_copy(update={"rebalance": "?", "spec": {**make_result().spec, "rebalance": None}})
    interp, _ = H.interpret(ru)
    assert not any("trading costs" in x for x in interp.key_findings + interp.biases_and_caveats)


def test_system_prompt_cost_drag_formula():
    flat = " ".join(INTERPRET_SYSTEM_PROMPT.split())
    assert "2 x avg_turnover_pct x rebalances per year x costs_bps / 10000" in flat
    assert "not from periods_per_year" in flat
    assert "avg_turnover_pct x periods_per_year" not in flat


def test_heuristic_flags_known_factor_exposure():
    r = make_result(alpha_t=0.8, mono=0.5, betas={"Mkt-RF": 0.9, "SMB": 0.1, "HML": 0.55},
                    beta_t={"Mkt-RF": 12.0, "SMB": 0.5, "HML": 4.1}, r2=0.85)
    interp, checks = H.interpret(r)
    cav = " | ".join(interp.biases_and_caveats)
    assert "repackaging of known factors" in cav and "HML beta 0.55 (t 4.1)" in cav and "Mkt-RF beta 0.90 (t 12.0)" in cav
    assert "R-squared of 0.85" in cav
    assert "regression.beta_t_stats.HML" in {c.path for c in interp.cited_metrics}
    assert all(c.status == "verified" for c in checks)


def test_heuristic_short_sample_caveat():
    interp, _ = H.interpret(make_result(n_periods=48))
    assert any("Short sample (4.0 years)" in c for c in interp.biases_and_caveats)
    interp, _ = H.interpret(make_result(n_periods=24))
    assert interp.verdict == "inconclusive"
    assert any("Only 2.0 years" in c for c in interp.biases_and_caveats)


# ------------------------------------------------------------------------------------------------
# Heuristic interpreter: data-quality caps (look-ahead / survivorship)
# ------------------------------------------------------------------------------------------------

_PIT_PRICES = DataUsage(dataset="prices", source="yahoo", coverage="150/150 tickers", point_in_time=True)
_SNAPSHOT_SI = DataUsage(dataset="short_interest", source="free snapshot", coverage="today only", point_in_time=False,
                         notes="no point-in-time history in the free edition")
_STRONG = dict(sharpe=1.2, alpha_t=3.5, mono=0.9)


def test_non_point_in_time_signal_data_caps_verdict_at_inconclusive():
    """Regression: a pure look-ahead backtest (today's short-interest snapshot) used to be 'robust'."""
    r = make_result(**_STRONG, name="short_interest", data_usage=[_PIT_PRICES, _SNAPSHOT_SI],
                    warnings=["Universe is today's constituents: survivorship bias."])
    v, why = H.verdict(r)
    assert v == "inconclusive"
    assert "'short_interest' is not point-in-time" in why and "cannot be judged" in why
    assert "the statistics alone read robust" in why and "Sharpe 1.20" in why
    interp, checks = H.interpret(r)
    assert interp.verdict == "inconclusive" and interp.summary.startswith("Verdict: inconclusive - dataset 'short_interest'")
    assert "Treat this result as unproven" in interp.summary
    assert all(c.status == "verified" for c in checks)
    # without the bad dataset the same statistics are robust
    assert H.verdict(make_result(**_STRONG, data_usage=[_PIT_PRICES]))[0] == "robust"


@pytest.mark.parametrize("kw", [dict(sharpe=0.9, alpha_t=2.5, mono=0.9), dict(sharpe=0.3, alpha_t=1.5, mono=0.5),
                                dict(sharpe=0.2, alpha_t=0.5, mono=0.1)])
def test_non_point_in_time_caps_every_statistical_verdict(kw):
    r = make_result(**kw, data_usage=[_PIT_PRICES, _SNAPSHOT_SI])
    assert H.verdict(make_result(**kw))[0] != "inconclusive"
    assert H.verdict(r)[0] == "inconclusive"


def test_lookahead_warning_caps_verdict_at_inconclusive():
    r = make_result(**_STRONG, warnings=["fundamentals are restated values: possible look-ahead bias"])
    v, why = H.verdict(r)
    assert v == "inconclusive" and "warns of look-ahead bias" in why
    # negated mentions do not trigger the cap
    for w in ["No look-ahead: signals are lagged one session.", "Signals lagged to avoid look-ahead bias."]:
        assert H.verdict(make_result(**_STRONG, warnings=[w]))[0] == "robust"


def test_survivorship_caps_verdict_at_promising():
    r = make_result(**_STRONG, warnings=["Universe is today's S&P 500 constituents: survivorship bias (delisted names missing)."])
    v, why = H.verdict(r)
    assert v == "promising"
    assert "survivorship bias" in why and "the statistics alone read robust" in why
    interp, _ = H.interpret(r)
    assert interp.verdict == "promising" and "survivorship-bias warning" in interp.summary
    # a survivorship note on a dataset counts too
    du = DataUsage(dataset="universe", source="free", coverage="503 names", point_in_time=True, notes="today's constituents, survivorship bias")
    assert H.verdict(make_result(**_STRONG, data_usage=[_PIT_PRICES, du]))[0] == "promising"
    # the cap only lowers: weaker verdicts are unchanged
    assert H.verdict(make_result(sharpe=0.3, alpha_t=1.5, mono=0.5, warnings=["survivorship bias"]))[0] == "weak"
    # negated mentions do not cap
    for w in ["Point-in-time universe including delisted names: no survivorship bias.", "survivorship-bias-free universe"]:
        assert H.verdict(make_result(**_STRONG, warnings=[w]))[0] == "robust"


def test_non_signal_datasets_do_not_cap():
    benign = [DataUsage(dataset="risk_free", source="FRED", coverage="x", point_in_time=False),
              DataUsage(dataset="ff_factors_official", source="Kenneth French", coverage="x", point_in_time=False)]
    r = make_result(**_STRONG, data_usage=[_PIT_PRICES, *benign])
    assert verdict_caps(r) == [] and H.verdict(r)[0] == "robust"


def test_verdict_caps_order_and_factor_model_runs():
    r = make_result(**_STRONG, data_usage=[_PIT_PRICES, _SNAPSHOT_SI],
                    warnings=["survivorship bias", "look-ahead bias in estimates"])
    assert [c for c, _ in verdict_caps(r)] == ["inconclusive", "inconclusive", "promising"]
    v, why = H.verdict(r)
    assert v == "inconclusive" and "'short_interest'" in why and "look-ahead" in why
    # factor-model runs are capped as well
    fr = _factor_result(3.2, 3.5, 0.8, 0.9)
    assert H.verdict(fr)[0] == "robust"
    fr.data_usage = [_PIT_PRICES, DataUsage(dataset="fundamentals", source="sec", coverage="x", point_in_time=False)]
    assert H.verdict(fr)[0] == "inconclusive"


def test_claude_verdict_is_capped_too():
    r = make_result(**_STRONG, data_usage=[_PIT_PRICES, _SNAPSHOT_SI])
    claude = _interp(("stats.strategy.sharpe", 1.2), verdict="robust")
    llm = ScriptedLLM({"interpret": _seq_responder(claude)})
    interp, checks = BacktestInterpreter(llm).interpret(r)
    assert interp.verdict == "inconclusive" and [c.status for c in checks] == ["verified"]
    assert interp.summary.startswith("Verdict capped at inconclusive (the reviewer said robust): dataset 'short_interest'")
    assert any("Verdict capped at inconclusive" in c for c in interp.biases_and_caveats)
    assert interp.cited_metrics == claude.cited_metrics
    # nothing binds -> the very same object comes back
    ok = _interp(verdict="weak")
    assert apply_verdict_caps(ok, r.model_copy(update={"data_usage": [_PIT_PRICES]})) is ok
    assert apply_verdict_caps(_interp(verdict="inconclusive"), r).verdict == "inconclusive"
    for needle in ["point_in_time = false", "must be inconclusive", "caps the verdict at promising"]:
        assert needle in " ".join(INTERPRET_SYSTEM_PROMPT.split())


# ------------------------------------------------------------------------------------------------
# Heuristic interpreter: factor-model runs
# ------------------------------------------------------------------------------------------------


def _factor_result(t_smb, t_hml, corr_smb, corr_hml, n=120):
    def fs(label, t):
        return _stats(label, sharpe=0.3, n=n, t=t, turnover=None)

    return make_result(
        kind="factor_model", name="ff3", regression=False, quantiles=False, n_periods=n,
        spec_extra={"factor_model": "ff3", "signal": []},
        extra_stats={"Mkt-RF": fs("Mkt-RF", 2.5), "SMB": fs("SMB", t_smb), "HML": fs("HML", t_hml)},
        factor_checks=[
            FactorConstructionCheck(factor="Mkt-RF", correlation_with_official=0.99, annual_premium_constructed_pct=8.1,
                                    annual_premium_official_pct=8.4, n_overlap_periods=n),
            FactorConstructionCheck(factor="SMB", correlation_with_official=corr_smb, annual_premium_constructed_pct=0.9,
                                    annual_premium_official_pct=1.5, n_overlap_periods=n),
            FactorConstructionCheck(factor="HML", correlation_with_official=corr_hml, annual_premium_constructed_pct=2.2,
                                    annual_premium_official_pct=2.0, n_overlap_periods=n),
        ],
    )


@pytest.mark.parametrize(
    "t_smb,t_hml,c_smb,c_hml,verdict",
    [
        (3.2, 3.5, 0.8, 0.9, "robust"),
        (3.2, 3.5, 0.6, 0.9, "promising"),     # correlation below 0.7 blocks robust
        (1.2, 2.4, 0.8, 0.9, "promising"),
        (1.2, 2.4, 0.4, 0.9, "weak"),          # poor replication
        (0.4, 0.9, 0.8, 0.9, "likely_spurious"),
        (1.2, 1.5, 0.8, 0.9, "weak"),
    ],
)
def test_factor_model_verdicts(t_smb, t_hml, c_smb, c_hml, verdict):
    assert H.interpret(_factor_result(t_smb, t_hml, c_smb, c_hml))[0].verdict == verdict


def test_factor_model_summary_per_factor():
    r = _factor_result(1.2, 2.4, 0.45, 0.9)
    interp, checks = H.interpret(r)
    text = " | ".join(interp.key_findings)
    assert "SMB: premium 0.90% a year, official 1.50%, t-stat 1.20, correlation with official 0.45." in text
    assert "HML: premium 2.20% a year, official 2.00%, t-stat 2.40, correlation with official 0.90." in text
    assert "Mkt-RF:" in text
    paths = {c.path for c in interp.cited_metrics}
    assert {"factor_checks.SMB.correlation_with_official", "factor_checks.HML.annual_premium_constructed_pct",
            "stats.HML.mean_return_t_stat"} <= paths
    assert any("Constructed SMB correlates only 0.45" in c for c in interp.biases_and_caveats)
    assert all(c.status == "verified" for c in checks)


def test_factor_model_without_evidence_is_inconclusive():
    r = make_result(kind="factor_model", name="ff3", regression=False, quantiles=False, spec_extra={"factor_model": "ff3"})
    r.stats.pop("strategy")
    assert H.interpret(r)[0].verdict == "inconclusive"
