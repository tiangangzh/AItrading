"""Serialisable records of the hypothesis ledger: hypotheses, evidence, pre-registered predictions,
and the belief / assessment summaries the judges return."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceKind = Literal["backtest", "holdout", "paper"]
BeliefStatus = Literal["untested", "supported", "refuted", "uncertain"]
Verdict = Literal["novel", "consistent_with_experience", "contradicts_experience", "uncertain"]

SCHEMA_VERSION = 1


class LLMPrior(BaseModel):
    """A language model's prior on a hypothesis from economic reasoning and the literature, given
    BEFORE it looked at the experience log (so it is information the log does not already hold)."""

    mean: float = Field(description="Expected annualised Sharpe ratio net of costs.")
    sd: float = Field(gt=0, description="Uncertainty of that expectation (annualised Sharpe units).")
    model: str
    made_at: datetime
    reasoning: str = ""
    info_horizon: date | None = Field(
        None, description="Last date of the evidence data the model was shown when it formed this prior (None = unknown). "
                          "A view whose holdout starts on or before it does not use the prior.")


class Hypothesis(BaseModel):
    hypothesis_id: str
    statement: str = Field(description="The claim in plain English.")
    mechanism: str = Field("", description="Why it might work (who is on the other side of the trade).")
    concept: str = Field(description="Family of the dominant signal, '+' / '-' for its orientation, e.g. 'past_return-'.")
    horizon_days: float | None = None
    spec: dict = Field(description="StrategySpec JSON of the first test (a representative implementation).")
    key: dict = Field(description="The identity fields (semantics.hypothesis_key).")
    source: str = Field("user", description="user | generator:<name> | discovery | template ...")
    names: list[str] = Field(default_factory=list, description="Spec names this hypothesis was tested under.")
    created_at: datetime
    llm_prior: LLMPrior | None = None
    text_horizon: date | None = Field(
        None, description="Set when a language model wrote the statement / mechanism: the last date of the evidence data it "
                          "had seen. A view whose holdout starts on or before it shows the deterministic statement instead.")


class Evidence(BaseModel):
    """One measurement of a hypothesis: a backtest, a holdout test or a paper-trading track record."""

    evidence_id: str
    hypothesis_id: str
    kind: EvidenceKind
    run_id: str | None = None
    start: date
    end: date
    years: float
    sharpe: float = Field(description="Annualised Sharpe ratio net of costs (excess of the risk-free rate when the run had it).")
    se: float = Field(description="Sampling standard error of the Sharpe ratio (Lo 2002, iid).")
    n_periods: int
    periods_per_year: float
    mean_t: float | None = None
    alpha_t: float | None = None
    costs_bps: float | None = None
    cost_sharpe: float | None = Field(
        None, description="Trading-cost drag in Sharpe units (annual cost / annual volatility); sharpe + cost_sharpe is the "
                          "gross Sharpe. None when the run does not report turnover (paper trading).")
    monthly: dict[str, float] = Field(default_factory=dict, description="'YYYY-MM' -> monthly return (overlap / correlation).")
    out_of_sample: bool = Field(False, description="Data that did not exist (paper) or was not looked at (holdout) when the hypothesis was chosen.")
    recorded_at: datetime
    caveats: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    """A judge's forecast recorded BEFORE the evidence it predicts (pre-registration)."""

    prediction_id: str
    hypothesis_id: str
    judge: str
    made_at: datetime
    purpose: EvidenceKind
    mean: float = Field(description="Predicted true Sharpe ratio.")
    sd: float = Field(description="Uncertainty about the true Sharpe ratio.")
    p_positive: float = Field(description="P(true Sharpe > 0).")
    context_evidence: list[str] = Field(default_factory=list, description="Evidence ids in the ledger when the forecast was "
                                                                           "recorded (none of them can resolve it).")
    target_evidence_id: str | None = None
    note: str = ""


class HoldoutUse(BaseModel):
    start: date
    end: date
    uses: int = 0
    last_used: datetime | None = None


class LedgerData(BaseModel):
    schema_version: int = SCHEMA_VERSION
    market: str
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    predictions: list[Prediction] = Field(default_factory=list)
    holdouts: dict[str, HoldoutUse] = Field(default_factory=dict)


class Belief(BaseModel):
    """Posterior belief about a hypothesis given everything in the ledger."""

    hypothesis_id: str
    statement: str
    mean: float = Field(description="Posterior mean of the true annualised Sharpe ratio net of costs.")
    sd: float
    p_positive: float
    p_hurdle: float
    hurdle: float
    gross_mean: float = Field(0.0, description="Posterior mean before trading costs.")
    cost: float = Field(0.0, description="Estimated cost drag in Sharpe units at the hypothesis's cost level.")
    n_evidence: int
    years: float = Field(description="Years of data behind its own evidence (overlapping runs counted once).")
    oos_years: float
    oos_sharpe: float | None = Field(None, description="Precision-weighted Sharpe of its out-of-sample evidence.")
    last_sharpe: float | None = None
    status: BeliefStatus


class Neighbor(BaseModel):
    hypothesis_id: str
    statement: str
    similarity: float
    mean: float
    sd: float
    p_positive: float
    n_evidence: int


class Assessment(BaseModel):
    """What a judge thinks of a hypothesis before (or without) testing it."""

    hypothesis_id: str
    statement: str
    judge: str
    mean: float
    sd: float
    p_positive: float
    p_hurdle: float
    hurdle: float
    verdict: Verdict
    already_tested: int = Field(description="Runs of this very hypothesis in the ledger.")
    related_tested: int = Field(description="Tested hypotheses with |similarity| >= 0.3.")
    neighbors: list[Neighbor] = Field(default_factory=list)
    critique: str = ""
    acquisition: float = Field(0.0, description="Upper confidence bound mean + sd (what the research loop tests first).")
    details: dict[str, Any] = Field(default_factory=dict)


class BeliefUpdate(BaseModel):
    """Before / after a new piece of evidence (what a backtest taught)."""

    hypothesis_id: str
    statement: str
    before: Assessment
    after: Belief
    evidence_id: str
    observed_sharpe: float
    surprise_z: float = Field(description="(observed - predicted) / predictive sd: |z| > 2 is a real surprise.")
    prediction_id: str | None = None
