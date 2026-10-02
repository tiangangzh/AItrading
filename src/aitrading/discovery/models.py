"""Auto strategy discovery: find ideas in papers / websites, verify them, ask the trader, test them.

Flow
----
1. Sources fetch candidate documents (arXiv q-fin API, RSS/Atom feeds of research blogs, Claude's
   server-side web search, or a URL / PDF the trader supplies).
2. An extractor (Claude, or an offline heuristic) reads each document and returns an
   ``IdeaCandidate``: what the paper claims, the signal, the reported evidence, verbatim quotes that
   support the summary (checked programmatically), the data it needs and whether the platform can
   test it with the data available.
3. Candidates are de-duplicated against the inbox and the built-in idea library, ranked, and stored
   in the ``IdeaInbox`` with status "new".
4. The trader is asked, idea by idea, whether to try it. Accepted ideas are translated into a
   ``StrategySpec`` and backtested; a replication suite then checks robustness (sub-periods,
   post-publication decay, transaction costs, parameter perturbations) and compares the replicated
   result with what the source claimed.
5. Results land back in the inbox (status "tested") and in the dashboard; the trader can save a
   strategy and paper-trade it.

Web and paper text is untrusted input: it is only ever *read* into structured fields, never
executed, and instructions inside it are ignored.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from aitrading.core.models import EvidenceCheck

SourceType = Literal["arxiv", "rss", "web_search", "url", "pdf"]
Testability = Literal["testable_now", "partially_testable", "needs_institutional_data", "not_testable"]
IdeaStatus = Literal["new", "proposed", "accepted", "rejected", "deferred", "tested", "saved", "failed"]


class SourceDocument(BaseModel):
    """A raw document fetched from a source, before extraction."""

    source_type: SourceType
    url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: date | None = None
    text: str = Field(description="Abstract or extracted article text (plain text).")
    fetched_at: datetime
    source_name: str = Field("", description='e.g. "arXiv q-fin.PM", a blog name, "Claude web search".')
    text_origin: Literal["page", "pdf", "abstract", "feed_item", "cited_passages", "not_read"] | None = Field(
        None, description="Where `text` came from; 'not_read' means the page itself was never fetched."
    )
    journal_ref: str | None = Field(None, description="Journal reference when the source states one (e.g. arXiv journal_ref).")
    doi: str | None = None

    @property
    def doc_key(self) -> str:
        return hashlib.sha1(f"{self.url.strip().lower()}|{self.title.strip().lower()}".encode()).hexdigest()[:12]


class IdeaExtraction(BaseModel):
    """Structured output the extractor (LLM) produces for one SourceDocument."""

    is_trading_idea: bool = Field(description="False for documents that do not propose a testable return-predicting signal or strategy.")
    title: str = Field(description="Short name for the idea, e.g. 'Industry-adjusted 12-1 momentum'.")
    summary: str = Field(description="Two to four plain-English sentences a trader can understand.")
    claimed_effect: str = Field(description="What the source claims (direction, magnitude, universe, period).")
    signal_description: str = Field(description="Precise description of the signal / rule as the source defines it.")
    asset_class: str = Field(description='e.g. "us_equities", "global_equities", "futures", "fx", "crypto", "options".')
    holding_period: str | None = Field(description='e.g. "1 month", "weekly", "1 year".')
    reported_sharpe: float | None = None
    reported_annual_return_pct: float | None = None
    reported_t_stat: float | None = None
    sample_period: str | None = Field(None, description='e.g. "1963-2019".')
    evidence_quotes: list[str] = Field(description="1-4 verbatim sentences copied from the document that support the summary.")
    data_requirements: list[str] = Field(description="Data needed to replicate (prices, book equity, analyst estimates, options, ...).")
    testability: Testability
    missing_data: list[str] = Field(description="Required data the platform's catalog / providers do not have.")
    proposed_strategy_idea: str = Field(
        description="One self-contained sentence describing the strategy to backtest with the platform's features, "
        "e.g. 'Long-short quintiles on 12-1 momentum, monthly rebalance, US stocks'. Empty if not testable."
    )
    closest_library_template: str | None = Field(None, description="Key of the closest built-in idea template, if any.")
    credibility_notes: list[str] = Field(description="Peer review status, sample length, data-mining risk, replication evidence.")


class IdeaCandidate(BaseModel):
    """An idea in the trader's inbox."""

    idea_id: str
    source: SourceDocument
    extraction: IdeaExtraction
    quote_checks: list[EvidenceCheck] = Field(default_factory=list, description="Verification of evidence_quotes against the source text.")
    score: float = Field(0.0, description="Ranking score in [0, 1] (testability, credibility, novelty, evidence).")
    novelty: Literal["new", "variant_of_library", "duplicate"] = "new"
    status: IdeaStatus = "new"
    security_flags: list[str] = Field(
        default_factory=list,
        description="Prompt-injection or integrity concerns found in the source; flagged ideas need explicit trader confirmation.",
    )
    discovered_at: datetime
    decided_at: datetime | None = None
    strategy_spec: dict | None = Field(None, description="StrategySpec JSON once translated.")
    backtest_run_id: str | None = None
    replication: "ReplicationReport | None" = None
    notes: list[str] = Field(default_factory=list)

    @property
    def quotes_verified(self) -> bool:
        return bool(self.quote_checks) and all(c.status == "verified" for c in self.quote_checks)


class RobustnessCheck(BaseModel):
    name: str = Field(description='e.g. "first_half", "second_half", "post_publication", "costs_25bps", "deciles_instead_of_quintiles".')
    description: str
    sharpe: float | None
    cagr_pct: float | None
    alpha_t_stat: float | None
    n_periods: int
    passed: bool | None = Field(description="Whether the effect survives this check (None if it could not be run).")
    note: str = ""


class ReplicationReport(BaseModel):
    idea_id: str
    backtest_run_id: str
    base_sharpe: float | None
    base_alpha_t_stat: float | None
    claimed_sharpe: float | None
    claimed_t_stat: float | None
    replication_ratio: float | None = Field(description="Replicated Sharpe / claimed Sharpe (None when no claim).")
    checks: list[RobustnessCheck]
    verdict: Literal["replicates", "partially_replicates", "fails_to_replicate", "inconclusive"]
    summary: str
    caveats: list[str]


IdeaCandidate.model_rebuild()
