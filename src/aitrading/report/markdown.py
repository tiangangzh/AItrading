"""Markdown research report for a ``PipelineResult`` (GitHub-flavoured Markdown, analyst-facing).

Sections, in order
------------------
* Title, the observation, and a run header (as-of, provider, LLM, run id, timings, counts).
* Summary: one paragraph plus an at-a-glance table of the explained ideas.
* 1 Screen specification: universe, conditions with rationale, ``any_of`` groups, ranking factors
  (weights normalised), assumptions, unsupported requests.
* 2 Screen funnel, 3 Feature coverage (low-coverage flags and the warnings naming those features),
  4 Ranked candidates (rank, ticker, name, score, key features).
* 5 Investment ideas: per explained (or failed) candidate - headline, dislocation type,
  actionable / conviction, market narrative vs variant view, mechanism, a quant evidence table and
  quotes as blockquotes, each with a grounding mark, catalysts, risks, invalidation triggers, data
  gaps, grounding score; errors are shown as a callout.
* Appendix A audit trail: LLM calls (purpose, model, request id, tokens, cache reads / writes, stop
  reason, fallback, error), the vendor push-down query, every warning.

Conventions
-----------
* Numbers are shown with ``aitrading.agent.explain.display_value`` (4 significant digits): the same
  text the explainer saw and the grounding verifier checked, so report and evidence agree.
* Grounding marks: ``✓`` verified, ``✗`` mismatch / not found, ``–`` no matching check. Checks are
  matched to evidence items by position when ``verify_thesis``'s order lines up (quant evidence
  first, then quotes), otherwise by feature / quote text.
* Table cells collapse whitespace and escape ``|`` as ``\\|``. Free text has raw HTML tags
  neutralised (``<tag`` -> ``&lt;tag``) so document text cannot inject markup into a rendered page.
* The output is a pure function of the result (no clock, no randomness).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from aitrading.agent.explain import display_value
from aitrading.core.models import (
    DislocationThesis,
    EvidenceCheck,
    GroundingReport,
    InvestmentIdea,
    LLMCallRecord,
    PipelineResult,
)
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, ScreenSpec

__all__ = [
    "render_markdown",
    "escape_cell",
    "match_grounding",
    "LOW_COVERAGE",
    "MAX_KEY_FEATURES",
    "CHECK_MARKS",
]

LOW_COVERAGE = 0.9  # coverage below this is flagged
MAX_KEY_FEATURES = 10  # feature columns in the ranked-candidates table
CHECK_MARKS = {"verified": "✓", "mismatch": "✗", "not_found": "✗"}
NOT_CHECKED = "–"
MISSING = "n/a"

_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
_BACKTICKS = re.compile(r"`+")
_THEMATIC_BREAK = re.compile(r"([-*_])(?:\s*\1){2,}\s*")
_BLOCK_MARKER = re.compile(r"^(?:([#>+*=-])|(\d+)([.)]))(?=\s|$)")


# --------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------


def _inline(text: Any) -> str:
    """Single-line free text: whitespace collapsed, raw HTML tags neutralised."""
    return _TAG_START.sub("&lt;", " ".join(str(text).split()))


def _block(text: Any) -> str:
    """Like ``_inline``, and a leading block marker (``#``, ``-``, ``>``, ``1.``) is escaped so the
    text stays a paragraph when it starts a line (bullet item, blockquote line)."""
    s = _inline(text)
    if _THEMATIC_BREAK.fullmatch(s):
        return "\\" + s
    m = _BLOCK_MARKER.match(s)
    if m is None:
        return s
    if m.group(1):
        return "\\" + s
    return f"{m.group(2)}\\{s[len(m.group(2)):]}"


def escape_cell(text: Any) -> str:
    """Text for a Markdown table cell: one line, ``|`` escaped, HTML tags neutralised."""
    return _inline(text).replace("|", "\\|")


def _code(text: Any) -> str:
    """Inline code span that survives backticks in ``text`` (whitespace collapsed)."""
    s = " ".join(str(text).split())
    if not s:
        return ""
    run = max((len(m.group(0)) for m in _BACKTICKS.finditer(s)), default=0)
    fence = "`" * (run + 1)
    pad = " " if run or s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def _fence(text: str, lang: str = "text") -> str:
    run = max((len(m.group(0)) for m in _BACKTICKS.finditer(text)), default=0)
    fence = "`" * max(3, run + 1)
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def _blockquote(text: Any) -> list[str]:
    """Multi-line text as blockquote lines (paragraph breaks kept, HTML tags neutralised)."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", str(text))]
    lines: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        if lines:
            lines.append(">")
        lines.append("> " + _block(p))
    return lines or [">"]


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, right: Iterable[int] = ()) -> list[str]:
    """Markdown table; every cell is escaped (headers included). ``right`` = right-aligned columns."""
    right = set(right)
    out = ["| " + " | ".join(escape_cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---:" if i in right else "---" for i in range(len(headers))) + "|")
    for row in rows:
        cells = [escape_cell(c) for c in row]
        cells += [""] * (len(headers) - len(cells))
        out.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return out


def _bullets(items: Iterable[Any], empty: str = "None.") -> list[str]:
    lines = [f"- {_block(i)}" for i in items if str(i).strip()]
    return lines or [f"_{empty}_"]


def _num(value: Any) -> str:
    return display_value(value)[0]


def _pct(fraction: float, decimals: int = 1) -> str:
    return f"{fraction * 100:.{decimals}f}%"


def _int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return MISSING


def _when(dt: datetime | None) -> str:
    if dt is None:
        return MISSING
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _duration(start: datetime, end: datetime | None) -> str | None:
    if end is None:
        return None
    try:
        secs = (end - start).total_seconds()
    except TypeError:  # naive vs aware
        return None
    if secs < 0:
        return None
    return f"{secs:.1f} s" if secs < 120 else f"{secs / 60:.1f} min"


def _label(code: str) -> str:
    """``transitory_fundamental_shock`` -> ``Transitory fundamental shock``."""
    text = code.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


# --------------------------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------------------------


def _norm(s: str | None) -> str:
    return " ".join((s or "").split()).casefold()


def match_grounding(
    thesis: DislocationThesis, report: GroundingReport | None
) -> tuple[list[EvidenceCheck | None], list[EvidenceCheck | None]]:
    """Grounding check for each quant evidence item and each quote (``None`` = no matching check)."""
    quant_ev, quote_ev = thesis.quant_evidence, thesis.narrative_evidence
    if report is None:
        return [None] * len(quant_ev), [None] * len(quote_ev)
    q_checks = [c for c in report.checks if c.kind == "quant"]
    n_checks = [c for c in report.checks if c.kind == "quote"]

    if len(q_checks) == len(quant_ev) and all(_norm(c.ref) == _norm(e.feature) for c, e in zip(q_checks, quant_ev)):
        quant: list[EvidenceCheck | None] = list(q_checks)
    else:
        used: set[int] = set()
        quant = []
        for e in quant_ev:
            j = next((j for j, c in enumerate(q_checks) if j not in used and c.ref == e.feature), None)
            if j is None:
                j = next((j for j, c in enumerate(q_checks) if j not in used and _norm(c.ref) == _norm(e.feature)), None)
            if j is not None:
                used.add(j)
            quant.append(q_checks[j] if j is not None else None)

    if len(n_checks) == len(quote_ev) and all(_norm(c.claim) == _norm(e.quote) for c, e in zip(n_checks, quote_ev)):
        quotes: list[EvidenceCheck | None] = list(n_checks)
    else:
        used = set()
        quotes = []
        for e in quote_ev:
            j = next(
                (j for j, c in enumerate(n_checks) if j not in used and _norm(c.claim) == _norm(e.quote) and c.ref == e.doc_id),
                None,
            )
            if j is None:
                j = next((j for j, c in enumerate(n_checks) if j not in used and _norm(c.claim) == _norm(e.quote)), None)
            if j is not None:
                used.add(j)
            quotes.append(n_checks[j] if j is not None else None)
    return quant, quotes


def _mark(check: EvidenceCheck | None) -> str:
    if check is None:
        return f"{NOT_CHECKED} not checked"
    return f"{CHECK_MARKS.get(check.status, '?')} {check.status.replace('_', ' ')}"


def _failure_detail(check: EvidenceCheck | None) -> str:
    if check is None or check.status == "verified" or not check.detail.strip():
        return ""
    return f": {check.detail}"


def _grounding_line(report: GroundingReport | None) -> str:
    if report is None:
        return "not run"
    if not report.checks:
        return "no evidence items to check"
    n, total = report.n_verified, len(report.checks)
    tail = "fully grounded" if report.is_fully_grounded else f"{total - n} failed"
    return f"{n}/{total} evidence items verified ({_pct(report.verified_ratio, 0)}) - {tail}"


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------


def _parse_spec(spec: Any) -> ScreenSpec | None:
    try:
        return ScreenSpec.model_validate(spec)
    except (ValidationError, TypeError, ValueError):
        return None


def _ranked(result: PipelineResult) -> list[InvestmentIdea]:
    return sorted(result.ideas, key=lambda i: (i.candidate.rank, i.candidate.ticker))


def _header(result: PipelineResult, spec: ScreenSpec | None) -> list[str]:
    title = spec.name if spec is not None and spec.name.strip() else "screen"
    ranked = _ranked(result)
    explained = sum(i.thesis is not None for i in ranked)
    finished = _when(result.finished_at)
    duration = _duration(result.started_at, result.finished_at)
    rows = [
        ("As of", result.as_of.isoformat()),
        ("Provider", result.provider),
        ("LLM", result.llm),
        ("Run id", _code(result.run_id)),
        ("Started", _when(result.started_at)),
        ("Finished", f"{finished} ({duration})" if duration else finished),
        ("Universe", f"{_int(result.universe_size)} names"),
        ("Passed the screen", _int(result.survivors)),
        ("Ranked / explained", f"{len(ranked)} / {explained}"),
    ]
    lines = [f"# Equity research report: {_inline(title)}", "", "**Investment observation**", ""]
    lines += _blockquote(result.observation or "(none)")
    lines += ["", *_table(["Run", "Value"], rows)]
    return lines


def _summary(result: PipelineResult, spec: ScreenSpec | None) -> list[str]:
    ranked = _ranked(result)
    explained = [i for i in ranked if i.thesis is not None]
    failed = [i for i in ranked if i.error]
    parts = [f"{_int(result.universe_size)} names were in the universe as of {result.as_of.isoformat()}"]
    if spec is not None:
        groups = f" and {len(spec.any_of)} any-of group(s)" if spec.any_of else ""
        parts.append(f"{_int(result.survivors)} passed all {len(spec.conditions)} condition(s){groups}")
    else:
        parts.append(f"{_int(result.survivors)} passed the screen")
    if spec is not None and spec.ranking:
        parts.append(f"the top {len(ranked)} were ranked by " + ", ".join(_code(f.feature) for f in spec.ranking))
    text = "; ".join(parts) + "."
    if explained:
        actionable = sum(i.thesis.is_actionable for i in explained)
        grounded = sum(bool(i.grounding and i.grounding.is_fully_grounded) for i in explained)
        text += (
            f" {len(explained)} candidate(s) were explained: {actionable} actionable, "
            f"{grounded} with every evidence item verified against the source data."
        )
    if failed:
        text += f" {len(failed)} explanation(s) failed (see section 5)."
    if result.warnings:
        text += f" {len(result.warnings)} warning(s) are listed in appendix A.3."
    lines = ["## Summary", "", text]
    if explained or failed:
        rows = []
        for i in ranked:
            if i.thesis is None and not i.error:
                continue
            c, t = i.candidate, i.thesis
            if t is None:
                rows.append([f"#{c.rank}", c.ticker, c.name, "explanation failed", MISSING, MISSING, MISSING])
                continue
            rows.append([
                f"#{c.rank}", c.ticker, c.name, _label(t.dislocation_type), t.conviction, _yes_no(t.is_actionable),
                _grounding_cell(i.grounding),
            ])
        lines += ["", *_table(["Rank", "Ticker", "Name", "Dislocation type", "Conviction", "Actionable", "Grounding"], rows)]
    return lines


def _grounding_cell(report: GroundingReport | None) -> str:
    if report is None:
        return MISSING
    if not report.checks:
        return "no checks"
    mark = CHECK_MARKS["verified"] if report.is_fully_grounded else CHECK_MARKS["mismatch"]
    return f"{mark} {report.n_verified}/{len(report.checks)}"


def _condition_rows(conds: Sequence[Condition], start: int = 1) -> list[list[str]]:
    return [[str(k), _code(c.describe()), c.rationale or MISSING] for k, c in enumerate(conds, start)]


def _spec_section(result: PipelineResult, spec: ScreenSpec | None) -> list[str]:
    lines = ["## 1. Screen specification", ""]
    if spec is None:
        lines += ["_The screen specification could not be parsed; raw JSON:_", ""]
        lines.append(_fence(json.dumps(result.spec, indent=2, default=str, ensure_ascii=False), "json"))
        return lines
    u = spec.universe
    universe = [
        ("Country", u.country),
        ("Security types", ", ".join(u.security_types) or "any"),
        ("Minimum price", f"${u.min_price:g}" if u.min_price is not None else "none"),
        ("Minimum 20-day avg dollar volume", f"${u.min_avg_dollar_volume_usd_mn:g}mn" if u.min_avg_dollar_volume_usd_mn is not None else "none"),
        ("Excluded sectors", ", ".join(u.exclude_sectors) or "none"),
    ]
    lines += [f"Screen {_code(spec.name)}; top {spec.top_n} survivors are ranked.", "", "### Universe", ""]
    lines += _table(["Filter", "Setting"], universe)
    lines += ["", "### Conditions (all must hold)", ""]
    if spec.conditions:
        lines += _table(["#", "Condition", "Rationale"], _condition_rows(spec.conditions), right=[0])
    else:
        lines.append("_No conditions: every name in the universe passes._")
    for g, group in enumerate(spec.any_of, 1):
        lines += ["", f"### Any-of group {g} (at least one must hold)", ""]
        lines += _table(["#", "Condition", "Rationale"], _condition_rows(group), right=[0])
    lines += ["", "### Ranking", ""]
    total = sum(f.weight for f in spec.ranking) or 1.0
    rows = [
        [_code(f.feature), f.direction.replace("_", " "), _pct(f.weight / total, 0), f.rationale or MISSING]
        for f in spec.ranking
    ]
    lines += _table(["Factor", "Direction", "Weight", "Rationale"], rows, right=[2]) if rows else ["_No ranking factors._"]
    lines += ["", "### Assumptions", "", *_bullets(spec.assumptions)]
    lines += ["", "### Unsupported requests", ""]
    if spec.unsupported_requests:
        lines += ["Parts of the observation that no catalog feature can express (not screened):", ""]
        lines += _bullets(spec.unsupported_requests)
    else:
        lines.append("_None: every part of the observation was expressed as a condition, ranking factor or assumption._")
    return lines


def _funnel_section(result: PipelineResult) -> list[str]:
    lines = ["## 2. Screen funnel", ""]
    if not result.funnel:
        return lines + ["_No funnel recorded._"]
    lines += [
        "Universe filters and conditions are applied cumulatively, in order. _Passed alone_ counts names "
        "satisfying the step on its own; _missing data_ counts names excluded because the feature was missing.",
        "",
    ]
    rows: list[list[str]] = [["0", "Universe", _int(result.universe_size), _int(result.universe_size), ""]]
    for k, step in enumerate(result.funnel, 1):
        rows.append([str(k), step.label, _int(step.passed_alone), _int(step.remaining), _int(step.missing_data)])
    lines += _table(["#", "Step", "Passed alone", "Remaining", "Missing data"], rows, right=[0, 2, 3, 4])
    lines += ["", f"**{_int(result.funnel[-1].remaining)} name(s) passed every step.**"]
    return lines


def _coverage_section(result: PipelineResult, catalog: FeatureCatalog) -> list[str]:
    lines = ["## 3. Feature coverage", ""]
    cov = result.feature_coverage
    if not cov:
        return lines + ["_No coverage recorded._"]
    lines += [
        "Share of the universe with a value for each feature the screen uses. A missing value never "
        "satisfies a condition, so low coverage silently shrinks the candidate set.",
        "",
    ]
    rows = []
    for f, v in cov.items():
        unit = catalog[f].unit if f in catalog else ""
        flag = f"LOW (< {_pct(LOW_COVERAGE, 0)})" if v < LOW_COVERAGE else "ok"
        rows.append([_code(f), unit, _pct(v), flag])
    lines += _table(["Feature", "Unit", "Coverage", "Status"], rows, right=[2])
    low = [f for f, v in cov.items() if v < LOW_COVERAGE]
    lines.append("")
    if low:
        lines.append(
            f"**Coverage warning:** {len(low)} feature(s) below {_pct(LOW_COVERAGE, 0)}: "
            + ", ".join(f"{_code(f)} ({_pct(cov[f])})" for f in low)
            + "."
        )
    else:
        lines.append(f"All screen features have at least {_pct(LOW_COVERAGE, 0)} coverage.")
    related = [w for w in result.warnings if any(re.search(rf"\b{re.escape(f)}\b", w) for f in cov)]
    if related:
        lines += ["", "Data warnings naming these features:", "", *_bullets(related)]
    return lines


def _key_features(result: PipelineResult, spec: ScreenSpec | None) -> list[str]:
    order: list[str] = []
    if spec is not None:
        order += [f.feature for f in spec.ranking]
        for c in spec.all_conditions():
            order.append(c.feature)
            if c.other_feature:
                order.append(c.other_feature)
    for i in _ranked(result):
        order += list(i.candidate.features)
    present = {k for i in result.ideas for k in i.candidate.features}
    return [f for f in dict.fromkeys(order) if f in present][:MAX_KEY_FEATURES]


def _candidates_section(result: PipelineResult, spec: ScreenSpec | None, catalog: FeatureCatalog) -> list[str]:
    lines = ["## 4. Ranked candidates", ""]
    ranked = _ranked(result)
    if not ranked:
        return lines + ["_No names passed the screen._"]
    keys = _key_features(result, spec)
    lines += [
        "Score = weighted mean of each ranking factor's percentile among the survivors (0-1, higher is better).",
        "",
    ]
    headers = ["Rank", "Ticker", "Name", "Score", *(f"{k} ({catalog[k].unit})" if k in catalog and catalog[k].unit else k for k in keys), "Explained"]
    rows = []
    for i in ranked:
        c = i.candidate
        status = "yes" if i.thesis is not None else ("failed" if i.error else "no")
        rows.append([str(c.rank), c.ticker, c.name, f"{c.score:.3f}", *(_num(c.features.get(k)) for k in keys), status])
    lines += _table(headers, rows, right=[0, 3, *range(4, 4 + len(keys))])
    return lines


def _idea_section(n: int, idea: InvestmentIdea) -> list[str]:
    c, t = idea.candidate, idea.thesis
    lines = [f"### 5.{n} #{c.rank} {_inline(c.ticker)} - {_inline(c.name)}", ""]
    if idea.error:
        lines += [f"> **Explanation failed:** {_code(idea.error)}", ""]
    if t is not None:
        lines += [f"**Thesis:** {_inline(t.headline)}", ""]
        lines.append(f"- **Dislocation type:** {_label(t.dislocation_type)} ({_code(t.dislocation_type)})")
        lines.append(f"- **Actionable:** {_yes_no(t.is_actionable)} · **Conviction:** {t.conviction}")
    factors = ", ".join(f"{_code(k)} {v:.2f}" for k, v in c.factor_scores.items())
    lines.append(f"- **Rank score:** {c.score:.3f}" + (f" ({factors})" if factors else ""))
    if t is not None:
        lines.append(f"- **Grounding:** {_grounding_line(idea.grounding)}")
    if idea.documents_used:
        lines.append("- **Documents shown to the explainer:** " + ", ".join(_code(d) for d in idea.documents_used))
    if t is None:
        return lines
    quant_checks, quote_checks = match_grounding(t, idea.grounding)
    lines += [
        "",
        "#### Market narrative vs variant view",
        "",
        f"**Market narrative.** {_inline(t.market_narrative)}",
        "",
        f"**Variant view.** {_inline(t.variant_view)}",
        "",
        "#### Why the dislocation exists",
        "",
        _inline(t.why_dislocation_exists),
        "",
        "#### Quantitative evidence",
        "",
    ]
    if t.quant_evidence:
        rows = []
        for ev, chk in zip(t.quant_evidence, quant_checks):
            rows.append([_mark(chk) + _failure_detail(chk), _code(ev.feature), _num(ev.value), ev.interpretation])
        lines += _table(["Check", "Feature", "Value", "Interpretation"], rows, right=[2])
    else:
        lines.append("_No quantitative evidence cited._")
    lines += ["", "#### Narrative evidence", ""]
    if t.narrative_evidence:
        for k, (ev, chk) in enumerate(zip(t.narrative_evidence, quote_checks)):
            if k:
                lines.append("")
            mark = CHECK_MARKS.get(chk.status, "?") if chk is not None else NOT_CHECKED
            source = ([_inline(ev.speaker)] if ev.speaker and ev.speaker.strip() else []) + [_code(ev.doc_id)]
            source.append("grounding: " + _inline(_mark(chk) + _failure_detail(chk)))
            lines += _blockquote(f"{mark} “{ev.quote}”")
            lines += [">", "> — " + " · ".join(source)]
            interpretation = _inline(ev.interpretation)
            lines += [">", f"> _{interpretation}_" if interpretation else "> _(no interpretation given)_"]
    else:
        lines.append("_No quotes cited._")
    for title, items, empty in (
        ("Catalysts", t.catalysts, "None stated."),
        ("Risks", t.risks, "None stated."),
        ("Invalidation triggers", t.invalidation_triggers, "None stated."),
        ("Data gaps", t.data_gaps, "None stated."),
    ):
        lines += ["", f"#### {title}", "", *_bullets(items, empty)]
    lines += ["", f"**Grounding score:** {_grounding_line(idea.grounding)}."]
    return lines


def _ideas_section(result: PipelineResult) -> list[str]:
    lines = ["## 5. Investment ideas", ""]
    ranked = _ranked(result)
    shown = [i for i in ranked if i.thesis is not None or i.error]
    rest = [i for i in ranked if i.thesis is None and not i.error]
    if not ranked:
        return lines + ["_No candidates to explain._"]
    if not shown:
        lines.append("_No candidates were explained in this run (screen only)._")
    else:
        lines += [
            "Every quoted sentence and every cited number below was checked programmatically against the "
            "source documents and the feature table (✓ verified, ✗ mismatch or not found, – not checked).",
        ]
        for n, idea in enumerate(shown, 1):
            lines += ["", *_idea_section(n, idea)]
    if rest and shown:
        lines += ["", "Not explained (ranked below the explanation cut-off): " + ", ".join(f"#{i.candidate.rank} {_inline(i.candidate.ticker)}" for i in rest) + "."]
    return lines


def _llm_rows(calls: Sequence[LLMCallRecord]) -> list[list[str]]:
    rows = []
    for k, c in enumerate(calls, 1):
        rows.append([
            str(k), c.purpose, c.model, _code(c.request_id) if c.request_id else MISSING, c.stop_reason or MISSING,
            _int(c.input_tokens), _int(c.output_tokens), _int(c.cache_read_input_tokens),
            _int(c.cache_creation_input_tokens), f"{c.latency_s:.2f}", _yes_no(c.served_by_fallback), c.error or "",
        ])
    if len(calls) > 1:
        rows.append([
            "", "**total**", "", "", "",
            _int(sum(c.input_tokens for c in calls)), _int(sum(c.output_tokens for c in calls)),
            _int(sum(c.cache_read_input_tokens for c in calls)), _int(sum(c.cache_creation_input_tokens for c in calls)),
            f"{sum(c.latency_s for c in calls):.2f}", str(sum(c.served_by_fallback for c in calls)),
            str(sum(bool(c.error) for c in calls)),
        ])
    return rows


def _audit_section(result: PipelineResult) -> list[str]:
    lines = ["## Appendix A. Audit trail", "", "### A.1 LLM calls", ""]
    if result.llm_calls:
        headers = ["#", "Purpose", "Model", "Request id", "Stop reason", "Input tokens", "Output tokens",
                   "Cache read", "Cache write", "Latency (s)", "Fallback", "Error"]
        lines += _table(headers, _llm_rows(result.llm_calls), right=[0, 5, 6, 7, 8, 9])
    else:
        lines.append(f"_No LLM calls were made (LLM: {_inline(result.llm)})._")
    lines += ["", "### A.2 Vendor push-down query", ""]
    if result.pushdown_query:
        lines += ["The screen was pushed down to the vendor; the local engine re-evaluated every condition on the returned names.", ""]
        lines.append(_fence(result.pushdown_query))
    else:
        lines.append("_Not used: every condition was evaluated locally by the deterministic screen engine._")
    lines += ["", "### A.3 Warnings", ""]
    if result.warnings:
        lines += [f"{k}. {_block(w)}" for k, w in enumerate(result.warnings, 1)]
    else:
        lines.append("_None._")
    return lines


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def render_markdown(result: PipelineResult, *, catalog: FeatureCatalog | None = None) -> str:
    """The full research report for ``result`` as GitHub-flavoured Markdown (ends with a newline)."""
    catalog = catalog or default_catalog()
    spec = _parse_spec(result.spec)
    sections = [
        _header(result, spec),
        _summary(result, spec),
        _spec_section(result, spec),
        _funnel_section(result),
        _coverage_section(result, catalog),
        _candidates_section(result, spec, catalog),
        _ideas_section(result),
        _audit_section(result),
    ]
    return "\n\n".join("\n".join(s) for s in sections).rstrip() + "\n"
