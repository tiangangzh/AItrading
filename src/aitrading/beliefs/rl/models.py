"""Persisted and exchanged models of the RL research loop (every string and list is bounded)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
HorizonBucket = Literal["short", "medium", "long"]
DirectiveKind = Literal["explore", "exploit", "avoid", "formulate"]


class RegionRef(BaseModel):
    concept: str = Field(max_length=60, description="A concept code such as 'value+' or 'past_return-'.")
    horizon_bucket: HorizonBucket | None = None


class Directive(BaseModel):
    directive_id: str = Field(max_length=12)
    kind: DirectiveKind
    region: RegionRef | None = None
    text: str = Field(max_length=200)
    cites: list[str] = Field(default_factory=list, max_length=4)


class PolicyVersion(BaseModel):
    version_id: str = Field(max_length=12)
    parent: str | None = None
    directives: list[Directive] = Field(default_factory=list, max_length=12)
    info_horizon: date | None = None
    created_at: datetime
    author: str = Field(max_length=80)
    episode_id: str | None = None
    edit_summary: list[str] = Field(default_factory=list, max_length=3)


class PolicyEvent(BaseModel):
    seq: int
    key: str = Field(max_length=120)
    kind: Literal["challenger", "pair", "terminal", "paper", "rejected_edit", "manual_reset", "manual_rollback"]
    at: datetime
    info_horizon: date
    episode_id: str | None = None
    champion: str | None = None
    challenger: str | None = None
    version: str | None = None
    outcome: Literal["win", "loss", "tie"] | None = None
    s_champion: float | None = None
    s_challenger: float | None = None
    finalist: str | None = None
    z: float | None = None
    reasons: list[str] = Field(default_factory=list, max_length=5)
    text_hash: str | None = None


class PolicyFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    market: str
    reward_version: str
    versions: dict[str, PolicyVersion] = Field(default_factory=dict)
    events: list[PolicyEvent] = Field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# LLM outputs (actor / reflector)
# ------------------------------------------------------------------------------------------------


class SketchComponent(BaseModel):
    feature: str = Field(max_length=60, description="Exact catalog feature name.")
    direction: Literal["higher_is_better", "lower_is_better"]
    weight: float = Field(1.0, ge=0.1, le=10.0)


class ActorProposal(BaseModel):
    statement: str = Field(max_length=300, description="The claim in one sentence; no result numbers.")
    mechanism: str = Field(max_length=400, description="Who is on the other side of the trade and why it persists.")
    signal: list[SketchComponent] = Field(min_length=1, description="1-3 ranking components.")
    rebalance: Literal["weekly", "monthly", "quarterly"]
    filters: list[str] = Field(default_factory=list, description="Names from the filter menu (at most 2).")
    follows: list[str] = Field(default_factory=list, description="Ids (d-...) of policy heuristics that shaped this.")
    explore: bool = Field(description="True only for a concept with no tested hypothesis in the experience log.")
    expected_sharpe: float = Field(description="Your forecast of the net Sharpe ratio (recorded for calibration only).")


class ActorBatch(BaseModel):
    plan: str = Field("", max_length=300, description="One or two sentences on this batch (audit only).")
    proposals: list[ActorProposal]


class EditOp(BaseModel):
    op: Literal["add", "replace", "delete"]
    target: str | None = Field(None, description="Directive id (d-...) for replace / delete.")
    kind: DirectiveKind | None = None
    region: RegionRef | None = None
    text: str = Field("", max_length=200)
    cites: list[str] = Field(default_factory=list, max_length=4)


class PolicyEdit(BaseModel):
    diagnosis: str = Field("", max_length=600, description="At most 4 sentences; stored for audit, never shown to a model again.")
    ops: list[EditOp] = Field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# Episode records
# ------------------------------------------------------------------------------------------------


class CreditBreakdown(BaseModel):
    kg_plain: float = 0.0
    kappa: float = 0.0
    rho: float | None = None
    rho_imputed: bool = False
    rho_all: float | None = None
    tau: float = 1.0
    complexity_excess: int = 0
    penalty: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    advantage_std: float = 0.0
    terminal_credit: float | None = None
    paper_credit: float | None = None


ProposalStatus = Literal["tested", "not_selected", "below_test_cost", "unparseable", "unknown_feature", "filter_not_in_menu",
                         "too_many_components", "style_not_allowed", "duplicate_in_arm", "retest_same_window", "near_duplicate"]


class ProposalRecord(BaseModel):
    idx: int
    arm: Literal["champion", "challenger", "heuristic"]
    policy_version: str | None = None
    statement: str = Field("", max_length=300)
    mechanism: str = Field("", max_length=400)
    sketch: dict = Field(default_factory=dict)
    spec: dict | None = None
    hypothesis_id: str | None = None
    concept: str | None = None
    status: ProposalStatus
    reason: str | None = Field(None, max_length=200)
    follows: list[str] = Field(default_factory=list, max_length=12)
    explore_claimed: bool = False
    explore_verified: bool = False
    max_sim_tested: float | None = None
    expected_sharpe: float = 0.0
    predicted: dict | None = None
    observed_sharpe: float | None = Field(None, description="Logged only; never an input of any reward.")
    evidence_id: str | None = None
    credit: CreditBreakdown = Field(default_factory=CreditBreakdown)
    flagged: list[str] = Field(default_factory=list, max_length=10)


class ArmPrompt(BaseModel):
    arm: str
    policy_version: str | None = None
    purpose: str
    system_sha: str
    user_sha: str
    user_text: str | None = Field(None, max_length=12000)
    truncated: bool = False


class RoundRecord(BaseModel):
    round: int
    started_at: datetime
    info_horizon: date
    view_evidence_hash: str
    n_view_evidence: int
    hyper: dict
    tau: dict[str, float] = Field(default_factory=dict)
    value_before: float = 0.0
    value_after: float = 0.0
    baseline_score: float | None = None
    arms: list[ArmPrompt] = Field(default_factory=list, max_length=2)
    proposals: list[ProposalRecord] = Field(default_factory=list, max_length=16)
    selected: list[int] = Field(default_factory=list)
    pair: dict | None = None
    feedback_out: list[str] = Field(default_factory=list, max_length=20)
    llm_calls: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, max_length=10)


class FinalistRecord(BaseModel):
    hypothesis_id: str
    origin: str | None = None
    origin_version: str | None = None
    mu: float
    sd: float
    rho_all: float | None = None
    clean: bool = False
    censor_reason: str | None = None
    evidence_id: str | None = None
    holdout_sharpe: float | None = None
    z: float | None = None
    shares: dict[str, float] = Field(default_factory=dict)
    external_share: float = 0.0
    paper: dict | None = None


class EpisodeRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    episode_id: str
    status: Literal["running", "finished", "aborted"]
    market: str
    theme: str | None = None
    reward_version: str
    bayes_hash: str
    actor_model: str
    reflector_model: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    research_window: list[Any]
    holdout_window: list[Any]
    cutoff: date
    holdout_uses_before: int = 0
    hyper_frozen: dict
    config: dict = Field(default_factory=dict, description="costs_bps, style, k, per_round, rounds of the run.")
    champion_at_start: str
    challenger_at_start: str | None = None
    after_llm_cutoff: bool | None = None
    rounds: list[RoundRecord] = Field(default_factory=list, max_length=10)
    finalists: list[FinalistRecord] = Field(default_factory=list, max_length=5)
    r_oos: float | None = None
    llm_calls_total: int = 0
    budget_exhausted: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=20)


class IndexRow(BaseModel):
    episode_id: str
    market: str
    cutoff: date
    holdout_window: list[Any]
    status: str
    n_tested: int
    n_finalists: int
    r_oos: float | None = None
    finished_at: datetime | None = None
