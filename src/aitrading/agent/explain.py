"""Explanation agent: one ranked candidate in, a grounded ``DislocationThesis`` out.

Flow
----
1. Build the user prompt: as-of date, observation, screen conditions and ranking factors, the
   candidate, a markdown FEATURE TABLE, an optional narrative-signal tally, the rendered documents
   inside ``<documents>...</documents>``, then the instructions. The system prompt
   (``EXPLAINER_SYSTEM_PROMPT``) is constant, so it is cached across candidates.
2. One structured call (purpose ``explain:<TICKER>``) returns a ``DislocationThesis``; its
   ``ticker`` is forced to the candidate's.
3. Every quote and cited number is verified (``aitrading.narrative.grounding.verify_thesis`` by
   default) against the documents and against the feature values *as shown in the table*, so the
   table's rounding never causes a false mismatch.
4. If any check failed and repair rounds remain, the model is called again (purpose
   ``explain:<TICKER>:repair``) with its previous thesis and the failed checks. The result with the
   higher ``verified_ratio`` is kept (ties go to the later round).

Displayed values
----------------
Numbers are rounded to 4 significant digits (integer digits are never dropped; at most 6
decimals), with trailing zeros removed: 6.81349 -> "6.813", 1234.56 -> "1235", 0.012345 ->
"0.01235". Booleans show as 1 / 0; None, NaN and +-inf show as "n/a" (verified as missing);
strings (category labels) are shown verbatim with whitespace collapsed.

Errors
------
``LLMError`` / ``LLMRefusalError`` from the first call propagate to the caller. A failure in a
repair call (an ``LLMError`` such as a truncated or schema-violating reply, or a raw pydantic
``ValidationError`` from a custom ``StructuredLLM``) does not discard the first-round thesis: the
best result so far is returned and the error is reported in ``ExplanationResult.repair_error``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from pydantic import ValidationError

from aitrading.agent.prompts import EXPLAINER_SYSTEM_PROMPT, FINAL_INSTRUCTIONS, NO_DOCUMENTS_NOTE, REPAIR_INSTRUCTIONS
from aitrading.core.models import DislocationThesis, Document, GroundingReport, RankedCandidate
from aitrading.llm.base import LLMError, StructuredLLM
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, ScreenSpec

__all__ = [
    "MISSING",
    "ExplanationResult",
    "Explainer",
    "Verifier",
    "display_value",
    "render_feature_table",
    "build_repair_prompt",
]

Verifier = Callable[[DislocationThesis, list[Document], dict], GroundingReport]

MISSING = "n/a"
TABLE_HEADER = "| feature | value | unit | sector median |"
_SIG_DIGITS = 4
_MAX_DECIMALS = 6
_WRAPPER_CLOSE = re.compile(r"</(?=documents\s*>)", re.IGNORECASE)


@dataclass
class ExplanationResult:
    thesis: DislocationThesis
    grounding: GroundingReport
    rounds: int  # LLM calls made for this candidate: 1 + repair calls attempted
    repair_error: str | None = None  # "<ErrorType>: message" if a repair call failed


# --------------------------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------------------------


def _format_number(x: float) -> str:
    """``x`` (finite) rounded to 4 significant digits, plain notation, trailing zeros removed."""
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = min(_MAX_DECIMALS, max(0, _SIG_DIGITS - 1 - magnitude))
    try:
        d = Decimal(repr(x)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
        text = format(d, "f")
    except InvalidOperation:  # beyond Decimal context precision (|x| >~ 1e28)
        return repr(x)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def display_value(value: Any) -> tuple[str, float | str | None]:
    """``(text shown in the feature table, value the verifier compares against)``.

    The two always agree: a number's verification value is ``float(text)``.
    """
    if value is None:
        return MISSING, None
    if isinstance(value, str):
        text = " ".join(value.split())
        return (text, text) if text else (MISSING, None)
    try:
        if pd.isna(value):
            return MISSING, None
    except (TypeError, ValueError):  # non-scalar
        pass
    if isinstance(value, (bool, np.bool_)):
        text = "1" if value else "0"
        return text, float(text)
    try:
        x = float(value)
    except (TypeError, ValueError):
        text = " ".join(str(value).split())
        return (text, text) if text else (MISSING, None)
    if not math.isfinite(x):
        return MISSING, None
    text = _format_number(x)
    return text, float(text)


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _spec_feature_order(spec: ScreenSpec | None) -> list[str]:
    """Features referenced by the spec, in order of first appearance (conditions, any_of, ranking)."""
    if spec is None:
        return []
    names: list[str] = []
    for c in spec.all_conditions():
        names.append(c.feature)
        if c.other_feature:
            names.append(c.other_feature)
    names.extend(f.feature for f in spec.ranking)
    return list(dict.fromkeys(names))


def _ordered_features(features: Mapping[str, Any], catalog: FeatureCatalog, priority: Iterable[str]) -> list[str]:
    """Screen features first, then the rest in catalog order, then non-catalog names in input order."""
    order = [n for n in priority if n in features]
    order += [n for n in catalog.names() if n in features]
    order += list(features)
    return list(dict.fromkeys(order))


def render_feature_table(
    features: Mapping[str, Any],
    catalog: FeatureCatalog | None = None,
    *,
    sector_context: Mapping[str, Any] | None = None,
    spec: ScreenSpec | None = None,
) -> tuple[str, dict[str, float | str | None]]:
    """Markdown ``| feature | value | unit | sector median |`` table and the values as shown.

    Every key of ``features`` gets a row: the spec's features first, then catalog order, then any
    feature the catalog does not know (blank unit). The returned dict maps each feature to the
    value in its VALUE cell (see ``display_value``) and is what the verifier checks against.
    """
    catalog = catalog or default_catalog()
    sector_context = sector_context or {}
    rows = [TABLE_HEADER, "|---|---|---|---|"]
    shown: dict[str, float | str | None] = {}
    for name in _ordered_features(features, catalog, _spec_feature_order(spec)):
        text, value = display_value(features[name])
        shown[name] = value
        unit = catalog[name].unit if name in catalog else ""
        median = display_value(sector_context[name])[0] if name in sector_context else MISSING
        rows.append(f"| {name} | {_cell(text)} | {_cell(unit)} | {_cell(median)} |")
    return "\n".join(rows), shown


# --------------------------------------------------------------------------------------------
# Prompt sections
# --------------------------------------------------------------------------------------------


def _as_date(d: date) -> date:
    return d.date() if isinstance(d, datetime) else d


def _condition_line(c: Condition) -> str:
    rationale = " ".join(c.rationale.split())
    return f"- {c.describe()}" + (f"  ({rationale})" if rationale else "")


def _screen_section(spec: ScreenSpec) -> str:
    u = spec.universe
    universe = [f"country {u.country}", "security types " + (", ".join(u.security_types) or "any")]
    if u.min_price is not None:
        universe.append(f"price >= {display_value(u.min_price)[0]} USD")
    if u.min_avg_dollar_volume_usd_mn is not None:
        universe.append(f"20-day average dollar volume >= {display_value(u.min_avg_dollar_volume_usd_mn)[0]} USD mn")
    if u.exclude_sectors:
        universe.append("excluding sectors " + ", ".join(u.exclude_sectors))
    lines = ["# Screen", "", "Universe: " + "; ".join(universe) + ".", "", "Conditions (all must hold):"]
    lines += [_condition_line(c) for c in spec.conditions] or ["- (none)"]
    if spec.any_of:
        lines += ["", "Alternative groups (at least one condition in each group must hold):"]
        lines += ["- " + " OR ".join(c.describe() for c in group) for group in spec.any_of]
    total = sum(f.weight for f in spec.ranking)
    lines += ["", "Ranking factors (weights normalised to sum to 1):"]
    for f in spec.ranking:
        w = display_value(f.weight / total if total > 0 else 0.0)[0]
        lines.append(f"- {f.feature}: {f.direction.replace('_', ' ')}, weight {w}")
    if not spec.ranking:
        lines.append("- (none)")
    if spec.assumptions:
        lines += ["", "Assumptions made translating the observation into the screen:"]
        lines += [f"- {a}" for a in spec.assumptions]
    if spec.unsupported_requests:
        lines += ["", "Parts of the observation the screen could not express (not checked by the screen):"]
        lines += [f"- {r}" for r in spec.unsupported_requests]
    return "\n".join(lines)


def _candidate_section(candidate: RankedCandidate, spec: ScreenSpec) -> str:
    lines = [
        "# Candidate",
        "",
        f"Ticker: {candidate.ticker}",
        f"Name: {candidate.name}",
        f"Rank: {candidate.rank}",
        f"Composite rank score: {display_value(candidate.score)[0]} (0 to 1, higher is better)",
    ]
    if candidate.factor_scores:
        order = list(dict.fromkeys([f.feature for f in spec.ranking if f.feature in candidate.factor_scores] + list(candidate.factor_scores)))
        scores = ", ".join(f"{k} {display_value(candidate.factor_scores[k])[0]}" for k in order)
        lines.append(f"Factor percentile scores (0 to 1, higher is better; not feature values): {scores}")
    return "\n".join(lines)


def _signals_section(signal_summary: Mapping[str, int]) -> str:
    lines = [
        "# Narrative signal tally",
        "",
        "Counts of phrase patterns found by a deterministic tagger over the documents below. A hint about "
        "where to look, not evidence; do not cite it.",
        "",
    ]
    lines += [f"- {tag}: {count}" for tag, count in signal_summary.items()] or ["- (no signals detected)"]
    return "\n".join(lines)


def _documents_section(documents_prompt: str) -> str:
    body = _WRAPPER_CLOSE.sub("&lt;/", documents_prompt.strip()) if documents_prompt and documents_prompt.strip() else NO_DOCUMENTS_NOTE
    return f"# Documents\n\n<documents>\n{body}\n</documents>"


def build_repair_prompt(previous: DislocationThesis, report: GroundingReport) -> str:
    """The repair section: the previous thesis as JSON plus every check that did not verify."""
    failed: list[str] = []
    for c in report.checks:
        if c.status == "verified":
            continue
        detail = f": {c.detail}" if c.detail else ""
        if c.kind == "quote":
            failed.append(f'- narrative_evidence (doc_id={c.ref}) quote "{c.claim}" -> {c.status}{detail}')
        else:
            failed.append(f"- quant_evidence ({c.ref}) {c.claim} -> {c.status}{detail}")
    return REPAIR_INSTRUCTIONS.format(
        previous_thesis_json=previous.model_dump_json(indent=2),
        failed_checks="\n".join(failed) or "- (none)",
    )


def _has_failures(report: GroundingReport) -> bool:
    return any(c.status != "verified" for c in report.checks)


# --------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------


class Explainer:
    """Explains why a screened candidate's price dislocation exists, grounded in evidence."""

    def __init__(
        self,
        llm: StructuredLLM,
        catalog: FeatureCatalog | None = None,
        *,
        effort: str = "high",
        max_repair_rounds: int = 1,
        verifier: Verifier | None = None,
        max_tokens: int = 16_000,
    ):
        if max_repair_rounds < 0:
            raise ValueError("max_repair_rounds must be >= 0")
        self.llm = llm
        self.catalog = catalog or default_catalog()
        self.effort = effort
        self.max_repair_rounds = max_repair_rounds
        self.verifier = verifier
        self.max_tokens = max_tokens

    # -- prompts ---------------------------------------------------------------------------

    def _build(
        self,
        candidate: RankedCandidate,
        features: Mapping[str, Any],
        documents_prompt: str,
        spec: ScreenSpec,
        as_of: date,
        sector_context: Mapping[str, Any] | None,
        signal_summary: Mapping[str, int] | None,
    ) -> tuple[str, dict[str, float | str | None]]:
        as_of_s = _as_date(as_of).isoformat()
        table, shown = render_feature_table(features, self.catalog, sector_context=sector_context, spec=spec)
        if not features:
            table = "(No feature values were provided for this candidate.)"
        sections = [
            f"As-of date: {as_of_s}. All data is point-in-time as of this date; nothing after it exists for this analysis.",
            "# Investment observation\n\n" + spec.observation.strip(),
            _screen_section(spec),
            _candidate_section(candidate, spec),
            "# Feature table\n\n"
            f"Candidate values as of {as_of_s}. Copy feature names and values exactly as shown in the value column; "
            f'"{MISSING}" means missing. Sector medians are context only and are not citable as quant_evidence.\n\n' + table,
        ]
        if signal_summary is not None:
            sections.append(_signals_section(signal_summary))
        sections.append(_documents_section(documents_prompt))
        sections.append(FINAL_INSTRUCTIONS.format(ticker=candidate.ticker, name=candidate.name, as_of=as_of_s))
        return "\n\n".join(sections), shown

    def build_user_prompt(
        self,
        candidate: RankedCandidate,
        features: Mapping[str, Any],
        documents_prompt: str,
        spec: ScreenSpec,
        as_of: date,
        sector_context: Mapping[str, Any] | None = None,
        signal_summary: Mapping[str, int] | None = None,
    ) -> str:
        """The user turn for ``explain`` (deterministic for identical inputs)."""
        return self._build(candidate, features, documents_prompt, spec, as_of, sector_context, signal_summary)[0]

    # -- calls -----------------------------------------------------------------------------

    def _ask(self, purpose: str, user: str, ticker: str) -> DislocationThesis:
        thesis = self.llm.structured(
            purpose=purpose,
            system=EXPLAINER_SYSTEM_PROMPT,
            user=user,
            output_model=DislocationThesis,
            effort=self.effort,
            max_tokens=self.max_tokens,
        )
        if thesis.ticker != ticker:
            thesis = thesis.model_copy(update={"ticker": ticker})
        return thesis

    def _verify(self, thesis: DislocationThesis, documents: list[Document], features: dict) -> GroundingReport:
        verifier = self.verifier
        if verifier is None:
            from aitrading.narrative.grounding import verify_thesis

            verifier = verify_thesis
        return verifier(thesis, documents, features)

    def explain(
        self,
        candidate: RankedCandidate,
        features: dict[str, float | str | None],
        documents: list[Document],
        documents_prompt: str,
        spec: ScreenSpec,
        as_of: date,
        sector_context: dict[str, float | None] | None = None,
        signal_summary: dict[str, int] | None = None,
    ) -> ExplanationResult:
        """Explain one candidate; raises ``LLMError`` / ``LLMRefusalError`` if the first call fails."""
        ticker = candidate.ticker
        user, shown = self._build(candidate, features, documents_prompt, spec, as_of, sector_context, signal_summary)
        thesis = self._ask(f"explain:{ticker}", user, ticker)
        report = self._verify(thesis, documents, shown)
        rounds, repair_error = 1, None
        for _ in range(self.max_repair_rounds):
            if not _has_failures(report):
                break
            rounds += 1
            try:
                retry = self._ask(f"explain:{ticker}:repair", user + "\n\n" + build_repair_prompt(thesis, report), ticker)
            except (LLMError, ValidationError) as e:  # keep the first-round thesis whatever the repair call did
                repair_error = f"{type(e).__name__}: {e}"
                break
            retry_report = self._verify(retry, documents, shown)
            if retry_report.verified_ratio >= report.verified_ratio:
                thesis, report = retry, retry_report
        return ExplanationResult(thesis=thesis, grounding=report, rounds=rounds, repair_error=repair_error)
