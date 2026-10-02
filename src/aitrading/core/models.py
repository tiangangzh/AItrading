"""Domain models shared across the pipeline (documents, candidates, theses, audit records)."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------------
# Documents (narrative engine inputs)
# --------------------------------------------------------------------------------------------


class DocumentKind(str, Enum):
    TRANSCRIPT = "transcript"  # earnings call / investor day transcript
    NEWS = "news"
    FILING = "filing"  # 10-K / 10-Q / 8-K text
    RESEARCH = "research"  # broker / independent research


class TranscriptSegment(BaseModel):
    speaker: str
    role: str = Field(description='e.g. "CEO", "CFO", "Analyst", "Operator", "IR"')
    section: Literal["prepared_remarks", "qa"]
    text: str


class Document(BaseModel):
    doc_id: str = Field(description="Stable, provider-unique identifier; used for citations.")
    ticker: str
    kind: DocumentKind
    title: str
    published_at: datetime
    source: str = Field(description="Provider / publisher name, for provenance.")
    url: str | None = None
    text: str = Field(description="Full plain text. For transcripts, the concatenation of segments.")
    segments: list[TranscriptSegment] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Screening / ranking outputs
# --------------------------------------------------------------------------------------------


class FunnelStep(BaseModel):
    """How many names survived each screen condition (applied cumulatively, in spec order)."""

    label: str
    passed_alone: int = Field(description="Names satisfying this condition on its own.")
    remaining: int = Field(description="Names left after applying this and all previous conditions.")
    missing_data: int = Field(0, description="Names excluded because the feature was NaN.")


class RankedCandidate(BaseModel):
    ticker: str
    name: str
    rank: int
    score: float = Field(description="Composite rank score in [0, 1]; higher is better.")
    factor_scores: dict[str, float] = Field(default_factory=dict, description="Per-factor percentile in [0, 1].")
    features: dict[str, float | str | None] = Field(default_factory=dict, description="Feature values used for the screen/ranking.")


# --------------------------------------------------------------------------------------------
# Explanation agent outputs (LLM structured output) + grounding
# --------------------------------------------------------------------------------------------

DislocationType = Literal[
    "transitory_fundamental_shock",  # one-off / timing issue the market extrapolates
    "guidance_reset_overreaction",  # conservative guide or reset that the price over-discounts
    "sector_or_macro_contagion",  # sold with the group / macro factor, idiosyncratics intact
    "technical_or_flow_driven",  # index deletion, forced selling, crowded short, liquidity
    "misunderstood_change",  # mix shift, accounting change, transition the market misreads
    "structural_decline_value_trap",  # the market is probably right; not a dislocation
    "insufficient_evidence",  # cannot tell from the evidence provided
]


class QuantEvidence(BaseModel):
    feature: str = Field(description="Exact feature name from the candidate's feature table.")
    value: float | None = Field(description="The value exactly as given in the feature table.")
    interpretation: str


class QuoteEvidence(BaseModel):
    doc_id: str = Field(description="doc_id of the source document the quote was taken from.")
    speaker: str | None = Field(description="Speaker for transcript quotes, else null.")
    quote: str = Field(description="Verbatim excerpt copied character-for-character from the document (<= 400 chars).")
    interpretation: str


class DislocationThesis(BaseModel):
    ticker: str
    headline: str = Field(description="One-sentence thesis.")
    dislocation_type: DislocationType
    market_narrative: str = Field(description="What the price action implies the market currently believes.")
    variant_view: str = Field(description="Where and why the evidence disagrees with the market narrative (or agrees, for value traps).")
    why_dislocation_exists: str = Field(description="Mechanism: why the mispricing exists and why it may persist or close.")
    quant_evidence: list[QuantEvidence]
    narrative_evidence: list[QuoteEvidence]
    catalysts: list[str]
    risks: list[str]
    invalidation_triggers: list[str] = Field(description="Observable events/data that would prove the thesis wrong.")
    conviction: Literal["low", "medium", "high"]
    is_actionable: bool = Field(description="False for value traps or insufficient evidence.")
    data_gaps: list[str] = Field(description="Information that was missing and would change the view.")


class EvidenceCheck(BaseModel):
    kind: Literal["quote", "quant"]
    ref: str = Field(description="doc_id for quotes, feature name for quant evidence.")
    claim: str = Field(description="The quote text or 'feature=value' that was checked.")
    status: Literal["verified", "mismatch", "not_found"]
    detail: str = ""


class GroundingReport(BaseModel):
    ticker: str
    checks: list[EvidenceCheck] = Field(default_factory=list)

    @property
    def n_verified(self) -> int:
        return sum(c.status == "verified" for c in self.checks)

    @property
    def verified_ratio(self) -> float:
        return self.n_verified / len(self.checks) if self.checks else 0.0

    @property
    def is_fully_grounded(self) -> bool:
        return bool(self.checks) and all(c.status == "verified" for c in self.checks)


class InvestmentIdea(BaseModel):
    candidate: RankedCandidate
    thesis: DislocationThesis | None = None
    grounding: GroundingReport | None = None
    documents_used: list[str] = Field(default_factory=list, description="doc_ids shown to the explainer.")
    error: str | None = None


# --------------------------------------------------------------------------------------------
# Audit records
# --------------------------------------------------------------------------------------------


class LLMCallRecord(BaseModel):
    purpose: str = Field(description='e.g. "nl_screen", "explain:ACME"')
    model: str
    request_id: str | None = None
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_s: float = 0.0
    served_by_fallback: bool = False
    error: str | None = None


class PipelineResult(BaseModel):
    run_id: str
    observation: str
    as_of: date
    provider: str
    llm: str
    spec: dict = Field(description="The ScreenSpec as JSON (kept untyped here to avoid an import cycle).")
    universe_size: int
    feature_coverage: dict[str, float] = Field(default_factory=dict, description="Fraction of universe with a non-NaN value per feature used.")
    funnel: list[FunnelStep] = Field(default_factory=list)
    survivors: int = 0
    ideas: list[InvestmentIdea] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    pushdown_query: str | None = Field(None, description="Vendor-side screen query, if the screen was pushed down.")
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
