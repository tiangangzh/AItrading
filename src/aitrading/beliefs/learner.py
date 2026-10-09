"""Predict -> test -> record -> update: the step every test goes through when learning is on.

:class:`Learner.before` registers the hypothesis and records the judge's forecast BEFORE the test
(pre-registration: the forecast cannot be adjusted after the fact, and it is what calibration scores).
:class:`Learner.after` records the result as evidence, links the forecast to it and returns the
:class:`~aitrading.beliefs.models.BeliefUpdate` (what the test taught: prediction, observation,
surprise, posterior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aitrading.backtest.models import BacktestResult
from aitrading.beliefs.bayes import DEFAULT_HURDLE, predictive_sd
from aitrading.beliefs.judge import BayesianJudge, Judge
from aitrading.beliefs.ledger import BeliefLedger
from aitrading.beliefs.models import Assessment, BeliefUpdate, EvidenceKind, Prediction
from aitrading.beliefs.semantics import spec_from_any

__all__ = ["Learner", "PreRegistration"]


@dataclass
class PreRegistration:
    assessment: Assessment
    prediction: Prediction


class Learner:
    def __init__(self, ledger: BeliefLedger, judge: Judge | None = None, *, hurdle: float = DEFAULT_HURDLE):
        self.ledger = ledger
        self.judge = judge if judge is not None else BayesianJudge(ledger, hurdle=hurdle)
        self.hurdle = hurdle

    def before(self, spec: Any, *, purpose: EvidenceKind = "backtest", statement: str | None = None,
               mechanism: str = "", source: str = "user", assessment: Assessment | None = None) -> PreRegistration:
        """Register the hypothesis and record the judge's forecast (``assessment``: one the judge just
        made for this spec against the current ledger, so it is not asked twice)."""
        s = spec_from_any(spec)
        self.ledger.add_hypothesis(s, statement=statement, mechanism=mechanism, source=source, save=False)
        a = assessment if assessment is not None else self.judge.assess(s, statement=statement, mechanism=mechanism)
        pred = self.ledger.add_prediction(hypothesis_id=a.hypothesis_id, judge=a.judge, purpose=purpose, mean=a.mean,
                                          sd=a.sd, p_positive=a.p_positive, note=a.verdict)
        return PreRegistration(a, pred)

    def after(self, spec: Any, result: BacktestResult, pre: PreRegistration | None, *, kind: EvidenceKind = "backtest",
              out_of_sample: bool | None = None, statement: str | None = None, mechanism: str = "",
              source: str = "user") -> BeliefUpdate | None:
        """Record ``result``; None when the run has no usable Sharpe ratio (nothing was learned)."""
        s = spec_from_any(spec)
        ev = self.ledger.record_backtest(s, result, kind=kind, out_of_sample=out_of_sample, statement=statement,
                                         mechanism=mechanism, source=source, save=False)
        if ev is None:
            self.ledger.save()
            return None
        pred_id = None
        if pre is not None and ev.evidence_id not in pre.prediction.context_evidence:
            self.ledger.resolve_prediction(pre.prediction.prediction_id, ev.evidence_id, save=False)
            pred_id = pre.prediction.prediction_id
        self.ledger.save()
        judge = BayesianJudge(self.ledger, hurdle=self.hurdle)
        belief = judge.model().belief(ev.hypothesis_id)
        before = pre.assessment if pre is not None else judge.assess(s)
        z = (ev.sharpe - before.mean) / predictive_sd(before.sd, ev)
        return BeliefUpdate(hypothesis_id=ev.hypothesis_id, statement=belief.statement, before=before, after=belief,
                            evidence_id=ev.evidence_id, observed_sharpe=ev.sharpe, surprise_z=z, prediction_id=pred_id)
