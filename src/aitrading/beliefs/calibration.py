"""Is the intuition any good? Score every pre-registered prediction against what happened.

A judge's :class:`~aitrading.beliefs.models.Prediction` is about the TRUE Sharpe ratio; what we
observe is a noisy estimate of it. So each prediction is turned into a forecast of the observation:
N(mean, sd^2 + the observation's sampling / regime / implementation noise), and scored with

* the Brier score of P(observed Sharpe > 0) - and its skill against a coin flip (0.25): positive
  skill = better than guessing;
* the mean log predictive density of the observed Sharpe, against a naive forecast
  N(0, 0.5^2 + noise) (``log_score_gain`` > 0 = the judge knew something);
* the coverage of the 80% predictive interval (about 80% when calibrated, much lower when
  overconfident).

Rows are per judge and per kind of target (backtest / holdout / paper). Out-of-sample targets are
the ones that matter: in-sample backtests can be "predicted" by anyone who has read the papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aitrading.beliefs.bayes import DEFAULT_PRIOR_SD, normal_cdf, predictive_sd
from aitrading.beliefs.ledger import BeliefLedger

__all__ = ["CalibrationRow", "calibration_report", "Z80"]

Z80 = 1.2815515655446004


@dataclass
class CalibrationRow:
    judge: str
    target_kind: str
    n: int
    brier: float
    brier_skill: float
    log_score: float
    log_score_gain: float
    coverage_80: float
    mean_z: float


def _log_pdf(x: float, mu: float, sd: float) -> float:
    return -0.5 * math.log(2 * math.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2


def calibration_report(ledger: BeliefLedger) -> list[CalibrationRow]:
    groups: dict[tuple[str, str], list[tuple[float, float, float, float, float]]] = {}
    for p in ledger.predictions():
        if not p.target_evidence_id:
            continue
        ev = ledger.data.evidence.get(p.target_evidence_id)
        if ev is None or not math.isfinite(ev.sharpe):
            continue
        sd = predictive_sd(p.sd, ev)
        naive_sd = predictive_sd(DEFAULT_PRIOR_SD, ev)
        p_obs = normal_cdf(p.mean / sd)
        outcome = 1.0 if ev.sharpe > 0 else 0.0
        z = (ev.sharpe - p.mean) / sd
        lp = _log_pdf(ev.sharpe, p.mean, sd)
        lp0 = _log_pdf(ev.sharpe, 0.0, naive_sd)
        for kind in (ev.kind, "all"):
            groups.setdefault((p.judge, kind), []).append(((p_obs - outcome) ** 2, lp, lp0, abs(z), z))
    rows: list[CalibrationRow] = []
    for (judge, kind), vals in sorted(groups.items()):
        n = len(vals)
        brier = sum(v[0] for v in vals) / n
        rows.append(CalibrationRow(
            judge=judge,
            target_kind=kind,
            n=n,
            brier=brier,
            brier_skill=1.0 - brier / 0.25,
            log_score=sum(v[1] for v in vals) / n,
            log_score_gain=sum(v[1] - v[2] for v in vals) / n,
            coverage_80=sum(1 for v in vals if v[3] <= Z80) / n,
            mean_z=sum(v[4] for v in vals) / n,
        ))
    return rows
