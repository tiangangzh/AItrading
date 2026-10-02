"""Tests for the Markdown report (aitrading.report.markdown), the CLI (aitrading.cli) and the
representative example script. All offline: the live path runs on ScriptedLLM or a fake client."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from aitrading import cli
from aitrading.agent.offline import HeuristicExplainer
from aitrading.core.models import (
    DislocationThesis,
    EvidenceCheck,
    FunnelStep,
    GroundingReport,
    InvestmentIdea,
    LLMCallRecord,
    PipelineResult,
    QuantEvidence,
    QuoteEvidence,
    RankedCandidate,
)
from aitrading.data.base import ProviderError
from aitrading.data.synthetic import SyntheticProvider
from aitrading.llm.base import LLMError, LLMRefusalError, ScriptedLLM
from aitrading.pipeline import NO_LLM, ResearchPipeline
from aitrading.report.markdown import CHECK_MARKS, escape_cell, match_grounding, render_markdown
from aitrading.screen.catalog import default_catalog
from aitrading.screen.nl import HeuristicScreenTranslator
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec

AS_OF = date(2026, 9, 30)
OBS = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but "
    "have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI "
    "under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where "
    "short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then "
    "read the latest earnings calls and explain the dislocation."
)
CANONICAL_CONDITIONS = [
    "market_cap_usd_bn between 2 and 20",
    "sma_50_vs_sma_200_pct > 0",
    "return_12m_ex_1m_pct > 0",
    "drawdown_from_52w_high_pct between -40 and -15",
    "max_volume_ratio_20d >= 2",
    "rsi_14 < 40",
    "fcf_yield_pct > 4",
    "revenue_growth_yoy_pct > 8",
    "short_interest_pct_float > 6",
]
SECTION_HEADINGS = [
    "# Equity research report: ",
    "## Summary",
    "## 1. Screen specification",
    "### Universe",
    "### Conditions (all must hold)",
    "### Ranking",
    "### Assumptions",
    "### Unsupported requests",
    "## 2. Screen funnel",
    "## 3. Feature coverage",
    "## 4. Ranked candidates",
    "## 5. Investment ideas",
    "## Appendix A. Audit trail",
    "### A.1 LLM calls",
    "### A.2 Vendor push-down query",
    "### A.3 Warnings",
]
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "representative_task.py"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def canonical_spec(**kw) -> ScreenSpec:
    c = Condition
    fields = dict(
        name="midcap_quality_pullback",
        observation=OBS,
        conditions=[
            c(feature="market_cap_usd_bn", op="between", value=2, value_high=20),
            c(feature="sma_50_vs_sma_200_pct", op=">", value=0),
            c(feature="return_12m_ex_1m_pct", op=">", value=0),
            c(feature="drawdown_from_52w_high_pct", op="between", value=-40, value_high=-15),
            c(feature="max_volume_ratio_20d", op=">=", value=2),
            c(feature="rsi_14", op="<", value=40),
            c(feature="fcf_yield_pct", op=">", value=4),
            c(feature="revenue_growth_yoy_pct", op=">", value=8),
            c(feature="short_interest_pct_float", op=">", value=6),
        ],
        ranking=[
            RankFactor(feature="fcf_yield_pct", direction="higher_is_better"),
            RankFactor(feature="revenue_growth_yoy_pct", direction="higher_is_better"),
            RankFactor(feature="drawdown_from_52w_high_pct", direction="lower_is_better"),
        ],
    )
    fields.update(kw)
    return ScreenSpec(**fields)


def _unescaped_pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def tables(md: str) -> list[list[str]]:
    """Blocks of consecutive lines starting with '|' (Markdown tables)."""
    out: list[list[str]] = []
    block: list[str] = []
    for line in md.splitlines():
        if line.startswith("|"):
            block.append(line)
        elif block:
            out.append(block)
            block = []
    if block:
        out.append(block)
    return out


def assert_tables_well_formed(md: str) -> None:
    blocks = tables(md)
    assert blocks, "no tables rendered"
    for block in blocks:
        assert len(block) >= 2 and re.fullmatch(r"\|(?:-{3}:?\|)+|\|(?:(?:---|---:)\|)+", block[1]), block[:2]
        width = _unescaped_pipes(block[0])
        for row in block:
            assert _unescaped_pipes(row) == width, (block[0], row)


def thesis(ticker: str = "ACME", **kw) -> DislocationThesis:
    base = dict(
        ticker=ticker,
        headline=f"{ticker} sold off on a one-off",
        dislocation_type="transitory_fundamental_shock",
        market_narrative="the market extrapolates the miss",
        variant_view="the miss is a one-off",
        why_dislocation_exists="forced selling after the print",
        quant_evidence=[QuantEvidence(feature="fcf_yield_pct", value=6.5, interpretation="cheap")],
        narrative_evidence=[QuoteEvidence(doc_id="D1", speaker="Jane Roe", quote="We expect the headwind to reverse.", interpretation="transitory")],
        catalysts=["next print"],
        risks=["recurrence"],
        invalidation_triggers=["guide cut"],
        conviction="medium",
        is_actionable=True,
        data_gaps=["no channel checks"],
    )
    base.update(kw)
    return DislocationThesis(**base)


def make_result(**kw) -> PipelineResult:
    base = dict(
        run_id="20260930-abcdef123456",
        observation="obs",
        as_of=AS_OF,
        provider="synthetic",
        llm=NO_LLM,
        spec=canonical_spec().model_dump(mode="json"),
        universe_size=100,
        started_at=datetime(2026, 10, 2, 9, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 10, 2, 9, 0, 3, tzinfo=timezone.utc),
    )
    base.update(kw)
    return PipelineResult(**base)


def candidate(ticker: str = "ACME", rank: int = 1, name: str = "Acme Corp", **kw) -> RankedCandidate:
    return RankedCandidate(
        ticker=ticker, name=name, rank=rank, score=kw.pop("score", 0.75),
        factor_scores=kw.pop("factor_scores", {"fcf_yield_pct": 0.8}),
        features=kw.pop("features", {"fcf_yield_pct": 6.5, "rsi_14": 31.234}), **kw,
    )


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider()


@pytest.fixture(scope="module")
def offline_result(provider) -> PipelineResult:
    pipe = ResearchPipeline(provider, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=None)
    return pipe.run(OBS, AS_OF)


@pytest.fixture(scope="module")
def offline_md(offline_result) -> str:
    return render_markdown(offline_result)


@pytest.fixture
def shared_provider(provider, monkeypatch):
    """Route the CLI's synthetic provider to the module's instance (generation is the slow part)."""
    real = cli.make_provider
    monkeypatch.setattr(cli, "make_provider", lambda name, **kw: provider if name == "synthetic" else real(name, **kw))
    return provider


@pytest.fixture
def no_credentials(monkeypatch):
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# --------------------------------------------------------------------------------------------
# Report: content on the canonical offline run
# --------------------------------------------------------------------------------------------


def test_report_has_every_section_in_order(offline_md):
    positions = [offline_md.find(h) for h in SECTION_HEADINGS]
    assert all(p >= 0 for p in positions), [h for h, p in zip(SECTION_HEADINGS, positions) if p < 0]
    assert positions == sorted(positions)
    assert offline_md.endswith("\n") and not offline_md.endswith("\n\n")


def test_report_header(offline_result, offline_md):
    r = offline_result
    assert offline_md.startswith("# Equity research report: " + r.spec["name"])
    assert "> " + OBS in offline_md
    for cell in (f"| As of | {AS_OF.isoformat()} |", "| Provider | synthetic |", f"| LLM | {NO_LLM} |",
                 f"| Run id | `{r.run_id}` |", f"| Universe | {r.universe_size} names |", f"| Passed the screen | {r.survivors} |"):
        assert cell in offline_md, cell
    assert re.search(r"\| Started \| \d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC \|", offline_md)


def test_report_spec_table_rationale_assumptions(offline_result, offline_md):
    spec = ScreenSpec.model_validate(offline_result.spec)
    for k, c in enumerate(spec.conditions, 1):
        assert f"| {k} | `{c.describe()}` | {escape_cell(c.rationale)} |" in offline_md
    assert [c.describe() for c in spec.conditions] == CANONICAL_CONDITIONS
    for f in spec.ranking:
        assert f"| `{f.feature}` | {f.direction.replace('_', ' ')} | 33% |" in offline_md
    for a in spec.assumptions:
        assert f"- {a}" in offline_md
    assert "| Minimum price | $5 |" in offline_md
    assert "_None: every part of the observation was expressed" in offline_md


def test_report_funnel_and_coverage(offline_result, offline_md):
    r = offline_result
    assert f"| 0 | Universe | {r.universe_size} | {r.universe_size} |  |" in offline_md
    for k, s in enumerate(r.funnel, 1):
        assert f"| {k} | {s.label} | {s.passed_alone:,} | {s.remaining:,} | {s.missing_data:,} |" in offline_md
    assert f"**{r.survivors} name(s) passed every step.**" in offline_md
    for f, v in r.feature_coverage.items():
        assert re.search(rf"\| `{f}` \| [^|]* \| {v * 100:.1f}% \| ok \|", offline_md), f
    assert "All screen features have at least 90% coverage." in offline_md


def test_report_candidates_table(offline_result, offline_md):
    block = next(b for b in tables(offline_md) if b[0].startswith("| Rank | Ticker | Name | Score |"))
    assert "fcf_yield_pct (%)" in block[0] and block[0].endswith("| Explained |")
    assert len(block) == 2 + len(offline_result.ideas)
    for row, idea in zip(block[2:], offline_result.ideas):
        c = idea.candidate
        assert row.startswith(f"| {c.rank} | {c.ticker} | {c.name} | {c.score:.3f} |")
        assert row.endswith("| yes |") if idea.thesis is not None else row.endswith("| no |")


def test_report_idea_sections(offline_result, offline_md):
    explained = [i for i in offline_result.ideas if i.thesis is not None]
    assert explained
    for n, idea in enumerate(explained, 1):
        c, t, g = idea.candidate, idea.thesis, idea.grounding
        start = offline_md.index(f"### 5.{n} #{c.rank} {c.ticker} - {c.name}")
        end = offline_md.find("\n### 5.", start + 1)
        sec = offline_md[start:end if end > 0 else len(offline_md)]
        assert f"**Thesis:** {t.headline}" in sec
        assert f"(`{t.dislocation_type}`)" in sec
        assert f"**Conviction:** {t.conviction}" in sec
        assert f"**Actionable:** {'yes' if t.is_actionable else 'no'}" in sec
        for h in ("#### Market narrative vs variant view", "**Market narrative.**", "**Variant view.**", "#### Why the dislocation exists",
                  "#### Quantitative evidence", "#### Narrative evidence", "#### Catalysts", "#### Risks",
                  "#### Invalidation triggers", "#### Data gaps"):
            assert h in sec, h
        for q in t.quant_evidence:
            assert re.search(rf"\| ✓ verified \| `{q.feature}` \| [^|]+ \|", sec), q.feature
        for q in t.narrative_evidence:
            assert f"> ✓ “{q.quote}”" in sec
            assert f"`{q.doc_id}`" in sec and (q.speaker is None or q.speaker in sec)
        for item in [*t.catalysts, *t.risks, *t.invalidation_triggers, *t.data_gaps]:
            assert f"- {' '.join(item.split())}" in sec
        assert f"**Grounding score:** {g.n_verified}/{len(g.checks)} evidence items verified (100%) - fully grounded." in sec
    rest = [i for i in offline_result.ideas if i.thesis is None]
    assert "Not explained (ranked below the explanation cut-off): " + ", ".join(f"#{i.candidate.rank} {i.candidate.ticker}" for i in rest) in offline_md


def test_report_summary_and_audit(offline_result, offline_md):
    explained = [i for i in offline_result.ideas if i.thesis is not None]
    for i in explained:
        assert re.search(rf"\| #{i.candidate.rank} \| {i.candidate.ticker} \| .* \| {i.thesis.conviction} \| (yes|no) \| ✓ \d+/\d+ \|", offline_md)
    assert f"_No LLM calls were made (LLM: {NO_LLM})._" in offline_md
    assert "_Not used: every condition was evaluated locally" in offline_md
    for k, w in enumerate(offline_result.warnings, 1):
        assert f"{k}. {w}" in offline_md


def test_report_is_deterministic_and_well_formed(offline_result, offline_md):
    assert render_markdown(offline_result) == offline_md
    assert render_markdown(PipelineResult.model_validate_json(offline_result.model_dump_json())) == offline_md
    assert_tables_well_formed(offline_md)


# --------------------------------------------------------------------------------------------
# Report: escaping and edge cases
# --------------------------------------------------------------------------------------------


def test_escape_cell():
    assert escape_cell("a | b") == "a \\| b"
    assert escape_cell("line1\n\nline2\t x") == "line1 line2 x"
    assert escape_cell("<script>alert(1)</script> x < 3") == "&lt;script>alert(1)&lt;/script> x < 3"
    assert escape_cell(4.5) == "4.5"


def test_pipes_are_escaped_everywhere():
    spec = canonical_spec(assumptions=["A | B assumption"], unsupported_requests=["C | D request"])
    spec.conditions[0].rationale = "'$2-20B' | mid caps"
    spec.ranking[0].rationale = "rank | by fcf"
    t = thesis(
        headline="pipe | headline",
        quant_evidence=[QuantEvidence(feature="fcf_yield_pct", value=6.5, interpretation="x | y")],
        narrative_evidence=[QuoteEvidence(doc_id="D|1", speaker="A | B", quote="We | they said it.", interpretation="i | j")],
        catalysts=["cat | 1"],
    )
    g = GroundingReport(ticker="ACME", checks=[
        EvidenceCheck(kind="quant", ref="fcf_yield_pct", claim="fcf_yield_pct=6.5", status="mismatch", detail="table 7.1; |diff|=0.6 > tol"),
        EvidenceCheck(kind="quote", ref="D|1", claim="We | they said it.", status="verified"),
    ])
    r = make_result(
        observation="obs | with pipe",
        spec=spec.model_dump(mode="json"),
        funnel=[FunnelStep(label="weird | label", passed_alone=10, remaining=5)],
        feature_coverage={"fcf_yield_pct": 0.5},
        survivors=1,
        ideas=[InvestmentIdea(candidate=candidate(name="Pipe | Co", features={"fcf_yield_pct": 6.5, "custom|feat": "x|y"}), thesis=t, grounding=g)],
        llm_calls=[LLMCallRecord(purpose="explain:A|B", model="m|1", error="bad | thing")],
        warnings=["fcf_yield_pct request | failed"],
    )
    md = render_markdown(r)
    assert_tables_well_formed(md)
    for text in ("'$2-20B' \\| mid caps", "rank \\| by fcf", "weird \\| label", "Pipe \\| Co", "x \\| y", "explain:A\\|B",
                 "m\\|1", "bad \\| thing", "|diff|=0.6 > tol".replace("|", "\\|"), "x\\|y"):
        assert text in md, text
    # outside tables pipes stay as typed
    assert "- A | B assumption" in md and "- C | D request" in md and "> obs | with pipe" in md


def test_errors_are_shown_clearly():
    r = make_result(
        survivors=2,
        ideas=[
            InvestmentIdea(candidate=candidate("ACME", 1), documents_used=["D1"], error="LLMRefusalError: [explain:ACME] model declined (category=test)"),
            InvestmentIdea(candidate=candidate("BETA", 2, "Beta Inc"), thesis=thesis("BETA"), grounding=GroundingReport(ticker="BETA")),
        ],
    )
    md = render_markdown(r)
    assert "> **Explanation failed:** `LLMRefusalError: [explain:ACME] model declined (category=test)`" in md
    assert "| #1 | ACME | Acme Corp | explanation failed | n/a | n/a | n/a |" in md
    assert re.search(r"\| 1 \| ACME \| Acme Corp \| 0\.750 \| .*\| failed \|", md)
    assert "1 explanation(s) failed (see section 5)." in md
    assert "### 5.1 #1 ACME - Acme Corp" in md and "### 5.2 #2 BETA - Beta Inc" in md
    assert "**Grounding:** no evidence items to check" in md
    assert_tables_well_formed(md)


def test_grounding_marks_for_failures_and_unchecked_items():
    t = thesis(
        quant_evidence=[
            QuantEvidence(feature="fcf_yield_pct", value=6.5, interpretation="a"),
            QuantEvidence(feature="rsi_14", value=31.23, interpretation="b"),
            QuantEvidence(feature="beta_1y", value=1.2, interpretation="never checked"),
        ],
        narrative_evidence=[
            QuoteEvidence(doc_id="D1", speaker="Jane Roe", quote="We expect the headwind to reverse.", interpretation="ok"),
            QuoteEvidence(doc_id="D2", speaker=None, quote="Invented sentence.", interpretation="bad"),
        ],
    )
    g = GroundingReport(ticker="ACME", checks=[
        EvidenceCheck(kind="quant", ref="rsi_14", claim="rsi_14=31.23", status="verified"),
        EvidenceCheck(kind="quant", ref="fcf_yield_pct", claim="fcf_yield_pct=6.5", status="mismatch", detail="table 7.1"),
        EvidenceCheck(kind="quote", ref="D2", claim="Invented sentence.", status="not_found", detail="no fragment found"),
        EvidenceCheck(kind="quote", ref="D1", claim="We expect the headwind to reverse.", status="verified"),
    ])
    quant, quotes = match_grounding(t, g)
    assert [c.ref if c else None for c in quant] == ["fcf_yield_pct", "rsi_14", None]
    assert [c.status for c in quotes] == ["verified", "not_found"]
    md = render_markdown(make_result(survivors=1, ideas=[InvestmentIdea(candidate=candidate(), thesis=t, grounding=g)]))
    assert "| ✗ mismatch: table 7.1 | `fcf_yield_pct` | 6.5 | a |" in md
    assert "| ✓ verified | `rsi_14` | 31.23 | b |" in md
    assert "| – not checked | `beta_1y` | 1.2 | never checked |" in md
    assert "> ✗ “Invented sentence.”" in md and "grounding: ✗ not found: no fragment found" in md
    assert "> ✓ “We expect the headwind to reverse.”" in md and "— Jane Roe · `D1` · grounding: ✓ verified" in md
    assert "**Grounding score:** 2/4 evidence items verified (50%) - 2 failed." in md
    assert "| ✗ 2/4 |" in md
    assert CHECK_MARKS == {"verified": "✓", "mismatch": "✗", "not_found": "✗"}


def test_match_grounding_positional_and_missing_report():
    t = thesis(quant_evidence=[QuantEvidence(feature="fcf_yield_pct", value=1, interpretation=""),
                               QuantEvidence(feature="fcf_yield_pct", value=2, interpretation="")])
    checks = [EvidenceCheck(kind="quant", ref="fcf_yield_pct", claim=f"fcf_yield_pct={v}", status=s) for v, s in ((1, "verified"), (2, "mismatch"))]
    quant, quotes = match_grounding(t, GroundingReport(ticker="ACME", checks=checks))
    assert [c.status for c in quant] == ["verified", "mismatch"] and quotes == [None]
    assert match_grounding(t, None) == ([None, None], [None])


def test_llm_calls_table_and_pushdown_query():
    calls = [
        LLMCallRecord(purpose="nl_screen", model="claude-opus-5-5", request_id="req_1", stop_reason="end_turn", input_tokens=12000,
                      output_tokens=800, cache_read_input_tokens=0, cache_creation_input_tokens=5000, latency_s=3.25),
        LLMCallRecord(purpose="explain:ACME", model="claude-fallback", request_id="req_2", stop_reason="end_turn", input_tokens=20000,
                      output_tokens=1500, cache_read_input_tokens=4800, latency_s=10.5, served_by_fallback=True),
        LLMCallRecord(purpose="explain:BETA", model="claude-opus-5-5", error="refusal:cyber", stop_reason="refusal"),
    ]
    query = "get(px_last) for(filter(equitiesuniv(['ACTIVE']), ```weird```))"
    md = render_markdown(make_result(llm="claude-opus-5-5", llm_calls=calls, pushdown_query=query))
    assert "| 1 | nl_screen | claude-opus-5-5 | `req_1` | end_turn | 12,000 | 800 | 0 | 5,000 | 3.25 | no |  |" in md
    assert "| 2 | explain:ACME | claude-fallback | `req_2` | end_turn | 20,000 | 1,500 | 4,800 | 0 | 10.50 | yes |  |" in md
    assert "| 3 | explain:BETA | claude-opus-5-5 | n/a | refusal | 0 | 0 | 0 | 0 | 0.00 | no | refusal:cyber |" in md
    assert "|  | **total** |  |  |  | 32,000 | 2,300 | 4,800 | 5,000 | 13.75 | 1 | 1 |" in md
    assert "````text\n" + query + "\n````" in md
    assert_tables_well_formed(md)


def test_low_coverage_flag_and_related_warnings():
    r = make_result(
        feature_coverage={"fcf_yield_pct": 0.42, "rsi_14": 0.995},
        warnings=["fundamentals request failed (Boom); NaN for fcf_yield_pct", "unrelated warning"],
    )
    md = render_markdown(r)
    assert "| `fcf_yield_pct` | % | 42.0% | LOW (< 90%) |" in md
    assert "| `rsi_14` | 0-100 | 99.5% | ok |" in md
    assert "**Coverage warning:** 1 feature(s) below 90%: `fcf_yield_pct` (42.0%)." in md
    cov = md[md.index("## 3. Feature coverage"):md.index("## 4. Ranked candidates")]
    assert "- fundamentals request failed (Boom); NaN for fcf_yield_pct" in cov and "unrelated warning" not in cov
    assert "1. fundamentals request failed" in md and "2. unrelated warning" in md


def test_empty_result_renders_every_section():
    r = make_result(spec=canonical_spec(conditions=[], ranking=[RankFactor(feature="price", direction="higher_is_better")]).model_dump(mode="json"),
                    universe_size=0, finished_at=None)
    md = render_markdown(r)
    for h in SECTION_HEADINGS:
        assert h in md, h
    for text in ("_No conditions: every name in the universe passes._", "_No funnel recorded._", "_No coverage recorded._",
                 "_No names passed the screen._", "_No candidates to explain._", "_None._", "| Finished | n/a |"):
        assert text in md, text


def test_screen_only_result_says_so():
    r = make_result(survivors=2, ideas=[InvestmentIdea(candidate=candidate("ACME", 1)), InvestmentIdea(candidate=candidate("BETA", 2))])
    md = render_markdown(r)
    assert "_No candidates were explained in this run (screen only)._" in md
    assert "Not explained" not in md and "## Summary" in md


def test_unparseable_spec_falls_back_to_raw_json():
    md = render_markdown(make_result(spec={"name": "broken", "conditions": "nope"}))
    assert "_The screen specification could not be parsed; raw JSON:_" in md
    assert '```json\n{\n  "name": "broken"' in md
    assert md.startswith("# Equity research report: screen")


def test_html_is_neutralised_and_multiline_text_stays_quoted():
    t = thesis(headline="<b>bold</b> claim", narrative_evidence=[
        QuoteEvidence(doc_id="D1", speaker=None, quote="<script>alert(1)</script> we grew", interpretation="")])
    r = make_result(observation="First paragraph.\n\nSecond <img src=x> paragraph.", survivors=1,
                    ideas=[InvestmentIdea(candidate=candidate(), thesis=t, grounding=None)])
    md = render_markdown(r)
    assert "<script" not in md and "<img" not in md and "<b>" not in md
    assert "&lt;script>alert(1)&lt;/script> we grew" in md
    assert "> First paragraph.\n>\n> Second &lt;img src=x> paragraph." in md
    assert "> _(no interpretation given)_" in md
    assert "**Grounding:** not run" in md


def test_leading_block_markers_are_escaped():
    r = make_result(
        observation="# Not a heading\n\n- not a list\n\n---",
        spec=canonical_spec(assumptions=["1. not an ordered list", "> not a quote"]).model_dump(mode="json"),
        warnings=["- dash first", "plain - warning"],
    )
    md = render_markdown(r)
    assert "> \\# Not a heading\n>\n> \\- not a list\n>\n> \\---" in md
    assert "- 1\\. not an ordered list" in md and "- \\> not a quote" in md
    assert "1. \\- dash first" in md and "2. plain - warning" in md


def test_naive_timestamps_and_long_runs():
    r = make_result(started_at=datetime(2026, 10, 2, 9, 0, 0), finished_at=datetime(2026, 10, 2, 9, 5, 0))
    md = render_markdown(r)
    assert "| Started | 2026-10-02 09:00:00 |" in md and "| Finished | 2026-10-02 09:05:00 (5.0 min) |" in md
    md = render_markdown(make_result(finished_at=datetime(2026, 10, 2, 9, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=1)))
    assert "| Finished | 2026-10-02 08:59:59 UTC |" in md


# --------------------------------------------------------------------------------------------
# CLI: parsing, catalog, spec
# --------------------------------------------------------------------------------------------


def test_cli_requires_a_command(capsys):
    code, out, err = run_cli(capsys)
    assert code == 2 and "a command is required" in err and out == ""


def test_cli_help_and_version(capsys):
    code, out, _ = run_cli(capsys, "--help")
    assert code == 0 and all(c in out for c in ("run", "screen", "spec", "catalog"))
    code, out, _ = run_cli(capsys, "run", "--help")
    assert code == 0 and "--observation-file" in out and "--offline" in out and "--explain" in out
    code, out, _ = run_cli(capsys, "--version")
    assert code == 0 and out.startswith("aitrading ")


def test_cli_catalog_markdown(capsys):
    code, out, _ = run_cli(capsys, "catalog")
    assert code == 0 and out.startswith("# Feature catalog (80 features)")
    for f in default_catalog():
        assert f"| `{f.name}` | {f.source} | {f.dtype} |" in out
    assert "| `fcf_yield_pct` | fundamental | number | % | higher is better |" in out
    assert_tables_well_formed(out)


@pytest.mark.parametrize("category, expected", [
    ("trend", lambda f: f.category == "trend"),
    ("VALUATION", lambda f: f.category == "valuation"),
    ("positioning", lambda f: f.source == "positioning"),
])
def test_cli_catalog_filter(capsys, category, expected):
    code, out, _ = run_cli(capsys, "catalog", "--category", category, "--format", "json")
    assert code == 0
    names = [d["name"] for d in json.loads(out)]
    assert names == [f.name for f in default_catalog() if expected(f)] and names


def test_cli_catalog_unknown_category(capsys):
    code, out, err = run_cli(capsys, "catalog", "--category", "astrology")
    assert code == 2 and out == "" and "unknown category 'astrology'" in err and "valuation" in err


def test_cli_spec_offline(capsys):
    code, out, err = run_cli(capsys, "spec", "--offline", OBS)
    assert code == 0
    spec = ScreenSpec.model_validate_json(out)
    assert [c.describe() for c in spec.conditions] == CANONICAL_CONDITIONS
    assert [(f.feature, f.direction) for f in spec.ranking] == [
        ("fcf_yield_pct", "higher_is_better"), ("revenue_growth_yoy_pct", "higher_is_better"),
        ("drawdown_from_52w_high_pct", "lower_is_better")]
    assert spec.observation == OBS and "translator: heuristic" in err


def test_cli_spec_from_file_and_stdin(capsys, tmp_path, monkeypatch):
    path = tmp_path / "obs.txt"
    path.write_text("\ufeff" + OBS + "\n", encoding="utf-8")
    code, out, _ = run_cli(capsys, "spec", "--offline", "-f", str(path), "--top", "3")
    spec = ScreenSpec.model_validate_json(out)
    assert code == 0 and spec.top_n == 3 and spec.observation == OBS
    monkeypatch.setattr(sys, "stdin", io.StringIO(OBS))
    code, out, _ = run_cli(capsys, "spec", "--offline", "--observation-file", "-")
    assert code == 0 and ScreenSpec.model_validate_json(out).observation == OBS


def test_cli_spec_accepts_unquoted_words(capsys):
    code, out, _ = run_cli(capsys, "spec", "--offline", "mid", "caps", "with", "RSI", "under", "30")
    spec = ScreenSpec.model_validate_json(out)
    assert code == 0 and spec.observation == "mid caps with RSI under 30"
    assert "rsi_14 < 30" in [c.describe() for c in spec.conditions]


@pytest.mark.parametrize("argv, message", [
    (["run", "--offline"], "an observation is required"),
    (["spec", "--offline", "   "], "an observation is required"),
    (["run", "--offline", "x", "--observation-file", "f.txt"], "not both"),
    (["spec", "--offline", "-f", "/definitely/missing/file.txt"], "cannot read observation file"),
    (["run", "--offline", "x", "--as-of", "2026-13-01"], "invalid date"),
    (["run", "--offline", "x", "--top", "0"], "between 1 and 100"),
    (["screen", "--offline", "x", "--top", "101"], "between 1 and 100"),
    (["run", "--offline", "x", "--explain", "-1"], "must be >= 0"),
    (["run", "--offline", "x", "--explain", "two"], "invalid integer"),
    (["run", "--offline", "x", "--provider", "yahoo"], "invalid choice"),
    (["run", "--offline", "x", "--format", "pdf"], "invalid choice"),
    (["spec", "--offline", "x", "--effort", "extreme"], "invalid choice"),
    (["catalog", "--bogus"], "unrecognized arguments"),
])
def test_cli_usage_errors_exit_2(capsys, tmp_path, monkeypatch, argv, message):
    monkeypatch.chdir(tmp_path)
    code, out, err = run_cli(capsys, *argv)
    assert code == 2 and message in err and out == ""
    assert not (tmp_path / "runs").exists()


# --------------------------------------------------------------------------------------------
# CLI: run / screen offline
# --------------------------------------------------------------------------------------------


def test_cli_run_offline_markdown(capsys, tmp_path, shared_provider):
    out_dir = tmp_path / "out"
    code, out, err = run_cli(capsys, "run", "--offline", OBS, "--out", str(out_dir))
    assert code == 0
    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert f"run directory: {run_dir}" in err and "names passed the screen" in err
    assert {p.name for p in run_dir.iterdir()} >= {"result.json", "spec.json", "features.csv", "documents.json", "report.md"}
    assert out == (run_dir / "report.md").read_text(encoding="utf-8")
    result = PipelineResult.model_validate_json((run_dir / "result.json").read_text(encoding="utf-8"))
    assert out == render_markdown(result)
    assert result.run_id == run_dir.name and result.llm == NO_LLM
    for h in SECTION_HEADINGS:
        assert h in out
    assert sum(i.thesis is not None for i in result.ideas) == 5


def test_cli_run_offline_json_no_save(capsys, tmp_path, monkeypatch, shared_provider):
    monkeypatch.chdir(tmp_path)
    code, out, err = run_cli(capsys, "run", "--offline", OBS, "--format", "json", "--no-save", "--top", "3", "--explain", "1")
    assert code == 0 and "run directory: none (--no-save)" in err
    result = PipelineResult.model_validate_json(out)
    assert len(result.ideas) == 3 and [i.thesis is not None for i in result.ideas] == [True, False, False]
    assert ScreenSpec.model_validate(result.spec).top_n == 3
    assert not (tmp_path / "runs").exists()


def test_cli_screen_offline(capsys, tmp_path, shared_provider):
    code, out, err = run_cli(capsys, "screen", "--offline", OBS, "--out", str(tmp_path), "--as-of", "2026-09-30")
    assert code == 0
    assert "## 4. Ranked candidates" in out and "_No candidates were explained in this run (screen only)._" in out
    assert "0 explained" in err
    (run_dir,) = list(tmp_path.iterdir())
    result = PipelineResult.model_validate_json((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result.survivors >= 5 and all(i.thesis is None and i.error is None for i in result.ideas)


def test_cli_run_with_real_synthetic_factory(capsys, tmp_path):
    code, out, err = run_cli(capsys, "screen", "--offline", OBS, "--no-save", "--format", "json", "--provider", "synthetic")
    assert code == 0 and PipelineResult.model_validate_json(out).provider == "synthetic"


# --------------------------------------------------------------------------------------------
# CLI: providers and runtime errors
# --------------------------------------------------------------------------------------------


def test_make_provider_factory(monkeypatch):
    assert isinstance(cli.make_provider("SYNTHETIC", n_tickers=40), SyntheticProvider)
    with pytest.raises(cli.UsageError, match="unknown provider"):
        cli.make_provider("yahoo")
    monkeypatch.setitem(cli.PROVIDERS, "vendorx", ("aitrading.data.no_such_adapter", "VendorX", "ask IT"))
    with pytest.raises(cli.ProviderUnavailableError, match="aitrading.data.no_such_adapter is not part of this installation"):
        cli.make_provider("vendorx")
    with pytest.raises(cli.UsageError, match="provider 'synthetic' does not accept: tickers"):
        cli.make_provider("synthetic", tickers=["AAA"])
    monkeypatch.setitem(cli.PROVIDERS, "vendorx", ("aitrading.data.synthetic", "NoSuchProvider", "ask IT"))
    with pytest.raises(cli.ProviderUnavailableError, match="has no class NoSuchProvider. To use it: ask IT"):
        cli.make_provider("vendorx")
    with pytest.raises(cli.UsageError, match="does not accept: bogus_kwarg"):
        cli.make_provider("synthetic", bogus_kwarg=1)

    class Exploding:
        def __init__(self, **kw):
            raise RuntimeError("session refused")

    real_import = cli.importlib.import_module
    monkeypatch.setitem(cli.PROVIDERS, "vendorx", ("vendorx_module", "VendorX", "ask IT"))
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(VendorX=Exploding) if name == "vendorx_module" else real_import(name))
    with pytest.raises(cli.ProviderUnavailableError, match=r"could not be started \(RuntimeError: session refused\)"):
        cli.make_provider("vendorx", tickers=["A"])

    def missing_sdk(name):
        raise ModuleNotFoundError("No module named 'vendor_sdk'", name="vendor_sdk")

    monkeypatch.setattr(cli.importlib, "import_module", missing_sdk)
    with pytest.raises(cli.ProviderUnavailableError, match="needs the 'vendor_sdk' package"):
        cli.make_provider("bloomberg")


def test_cli_unavailable_provider_exits_1(capsys, tmp_path, monkeypatch):
    monkeypatch.setitem(cli.PROVIDERS, "vendorx", ("aitrading.data.no_such_adapter", "VendorX", "install the VendorX adapter"))
    code, out, err = run_cli(capsys, "run", "--offline", OBS, "--provider", "vendorx", "--out", str(tmp_path))
    assert code == 1 and out == ""
    assert "provider 'vendorx' is not available" in err and "install the VendorX adapter" in err and "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def test_cli_tickers_option(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_make_provider(name, **kw):
        seen.update(name=name, **kw)
        return _BrokenProvider()

    monkeypatch.setattr(cli, "make_provider", fake_make_provider)
    path = tmp_path / "tickers.txt"
    path.write_text("aapl  # Apple\nMSFT\n\n# comment\naapl\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "screen", "--offline", OBS, "--provider", "free", "--tickers", f"@{path}", "--no-save")
    assert code == 1 and seen == {"name": "free", "tickers": ["AAPL", "MSFT"]} and "vendor session expired" in err
    code, _, _ = run_cli(capsys, "screen", "--offline", OBS, "--tickers", "ibm, ge f", "--no-save")
    assert seen["tickers"] == ["IBM", "GE", "F"]
    code, _, err = run_cli(capsys, "screen", "--offline", OBS, "--tickers", " , ", "--no-save")
    assert code == 2 and "--tickers is empty" in err
    code, _, err = run_cli(capsys, "screen", "--offline", OBS, "--tickers", "@/no/such/file", "--no-save")
    assert code == 2 and "cannot read tickers file" in err


def test_cli_tickers_rejected_by_synthetic(capsys):
    code, out, err = run_cli(capsys, "screen", "--offline", OBS, "--tickers", "AAA", "--no-save")
    assert code == 2 and out == "" and "provider 'synthetic' does not accept: tickers" in err


@pytest.mark.skipif(importlib.util.find_spec("lseg") is not None, reason="the LSEG library is installed here")
def test_cli_vendor_without_sdk_exits_1(capsys, tmp_path):
    code, out, err = run_cli(capsys, "screen", "--offline", OBS, "--provider", "lseg", "--out", str(tmp_path))
    assert code == 1 and out == "" and "data provider error" in err and "--provider synthetic" in err
    assert list(tmp_path.iterdir()) == []


class _BrokenProvider:
    name = "broken"
    capabilities: set = set()

    def get_universe(self, spec, as_of):
        raise ProviderError("vendor session expired")


def test_cli_runtime_error_exit_1_and_debug_traceback(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "make_provider", lambda name, **kw: _BrokenProvider())
    code, out, err = run_cli(capsys, "run", "--offline", OBS, "--out", str(tmp_path))
    assert code == 1 and out == "" and "data provider error: vendor session expired" in err and "Traceback" not in err
    code, _, err = run_cli(capsys, "run", "--offline", OBS, "--out", str(tmp_path), "--debug")
    assert code == 1 and "Traceback" in err
    assert list(tmp_path.iterdir()) == []


def test_cli_keyboard_interrupt(capsys, monkeypatch):
    def interrupted(name, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_provider", interrupted)
    code, _, err = run_cli(capsys, "run", "--offline", OBS, "--no-save")
    assert code == 130 and "interrupted" in err


# --------------------------------------------------------------------------------------------
# CLI: live path (Claude) without network
# --------------------------------------------------------------------------------------------


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic; ``mode`` decides how construction / calls fail."""

    mode = "no_credentials"

    def __init__(self, *args, **kwargs):
        if self.mode == "construct":
            raise RuntimeError("profile 'default' is malformed")
        self.beta = self
        self.messages = self

    def parse(self, **kwargs):
        if self.mode == "no_credentials":
            raise TypeError("Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set.")
        import anthropic

        class _Response:
            status_code = 401
            request = None
            headers: dict = {}

        raise anthropic.AuthenticationError("invalid x-api-key", response=_Response(), body=None)


@pytest.fixture
def fake_anthropic(monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    return _FakeAnthropic


@pytest.mark.parametrize("command", ["spec", "run", "screen"])
def test_cli_without_credentials_suggests_offline(capsys, tmp_path, no_credentials, fake_anthropic, monkeypatch, shared_provider, command):
    monkeypatch.setattr(fake_anthropic, "mode", "no_credentials")
    argv = [command, OBS] + (["--out", str(tmp_path)] if command != "spec" else [])
    code, out, err = run_cli(capsys, *argv)
    assert code == 1 and out == ""
    assert "--offline" in err and "ANTHROPIC_API_KEY" in err and "are not set" in err and "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def test_cli_client_construction_failure_suggests_offline(capsys, no_credentials, fake_anthropic, monkeypatch):
    monkeypatch.setattr(fake_anthropic, "mode", "construct")
    code, out, err = run_cli(capsys, "spec", OBS)
    assert code == 1 and "could not create the Anthropic client" in err and "--offline" in err


def test_cli_rejected_key_suggests_checking_it(capsys, fake_anthropic, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-invalid")
    monkeypatch.setattr(fake_anthropic, "mode", "rejected")
    code, out, err = run_cli(capsys, "spec", OBS)
    assert code == 1 and "authentication" in err and "Check that the API key" in err and "--offline" in err


def test_is_auth_error_and_credentials_configured(monkeypatch):
    for k in cli.CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    assert not cli.credentials_configured()
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert cli.credentials_configured()
    wrapped = LLMError("[nl_screen] authentication/permission error: bad key")
    assert cli.is_auth_error(wrapped)
    try:
        try:
            raise TypeError("Could not resolve authentication method.")
        except TypeError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        assert cli.is_auth_error(outer)
    assert not cli.is_auth_error(LLMError("[explain:X] rate limited after SDK retries"))
    assert not cli.is_auth_error(ValueError("nope"))


def _scripted_llm(refuse: set[str] = frozenset()) -> ScriptedLLM:
    def nl(purpose, system, user, model):
        return canonical_spec()

    def explain(purpose, system, user, model):
        ticker = re.search(r"^Ticker: (\S+)$", user, re.M).group(1)
        if ticker in refuse:
            raise LLMRefusalError(f"[{purpose}] model declined (category=test)", category="test")
        value = re.search(r"^\| fcf_yield_pct \| ([^|]+?) \|", user, re.M).group(1)
        return thesis(ticker, quant_evidence=[QuantEvidence(feature="fcf_yield_pct", value=float(value), interpretation="from the table")],
                      narrative_evidence=[])

    return ScriptedLLM({"nl_screen": nl, "explain": explain})


def test_cli_live_path_uses_the_llm(capsys, tmp_path, monkeypatch, shared_provider):
    llm = _scripted_llm(refuse={"MARM"})
    seen = {}

    def fake_make_llm(model=None, effort="high"):
        seen.update(model=model, effort=effort)
        return llm

    monkeypatch.setattr(cli, "make_llm", fake_make_llm)
    code, out, err = run_cli(capsys, "run", OBS, "--out", str(tmp_path), "--model", "claude-test", "--effort", "medium", "--explain", "3")
    assert code == 0 and seen == {"model": "claude-test", "effort": "medium"}
    (run_dir,) = list(tmp_path.iterdir())
    result = PipelineResult.model_validate_json((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result.llm == "scripted" and [c.purpose for c in result.llm_calls][0] == "nl_screen"
    # the refused call raises inside ScriptedLLM before it is recorded: nl_screen + 2 successful explains
    assert [c.purpose for c in result.llm_calls] == ["nl_screen", "explain:DLCR", "explain:GOLF"]
    assert "10 ranked, 2 explained, 1 failed; 3 LLM call(s)" in err
    assert "| 1 | nl_screen | scripted |" in out and "| 2 | explain:DLCR | scripted |" in out
    assert "> **Explanation failed:** `LLMRefusalError: [explain:MARM] model declined (category=test)`" in out
    assert "| ✓ verified | `fcf_yield_pct` |" in out
    assert [p["purpose"] for p in llm.prompts][:2] == ["nl_screen", "explain:DLCR"]


def test_cli_live_translation_failure_exits_1(capsys, monkeypatch, tmp_path):
    bad = canonical_spec(conditions=[Condition(feature="moon_phase", op=">", value=1)])
    monkeypatch.setattr(cli, "make_llm", lambda model=None, effort="high": ScriptedLLM({"nl_screen": lambda *a: bad}))
    code, out, err = run_cli(capsys, "run", OBS, "--out", str(tmp_path))
    assert code == 1 and out == "" and "could not produce a valid screen after 3 attempt(s):" in err
    assert "  - unknown feature 'moon_phase'" in err
    assert list(tmp_path.iterdir()) == []


def test_cli_spec_live_with_scripted_llm(capsys, monkeypatch):
    monkeypatch.setattr(cli, "make_llm", lambda model=None, effort="high": _scripted_llm())
    code, out, err = run_cli(capsys, "spec", OBS)
    assert code == 0 and [c.describe() for c in ScreenSpec.model_validate_json(out).conditions] == CANONICAL_CONDITIONS
    assert "translator: scripted; attempts: 1" in err


# --------------------------------------------------------------------------------------------
# Example script
# --------------------------------------------------------------------------------------------


def _load_example():
    spec = importlib.util.spec_from_file_location("representative_task", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_script_offline(capsys, tmp_path):
    example = _load_example()
    report = tmp_path / "nested" / "report.md"
    assert example.main(["--report", str(report), "--save-run", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert "matches the canonical demo spec: yes" in out
    assert re.search(r"Funnel: \d+ names -> \d+ passed -> top 10 ranked -> 5 explained", out)
    assert out.count("[planted: ") == 5 and f"Report: {report}" in out
    md = report.read_text(encoding="utf-8")
    for h in SECTION_HEADINGS:
        assert h in md
    assert len(list((tmp_path / "runs").iterdir())) == 1
    assert example.OBSERVATION == OBS


def test_example_matches_canonical_helper():
    example = _load_example()
    assert example.matches_canonical(canonical_spec())
    assert not example.matches_canonical(canonical_spec(conditions=canonical_spec().conditions[:-1]))


def test_example_live_without_credentials(capsys, tmp_path, no_credentials, fake_anthropic, monkeypatch):
    monkeypatch.setattr(fake_anthropic, "mode", "no_credentials")
    example = _load_example()
    assert example.main(["--live", "--report", str(tmp_path / "r.md")]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "without --live" in err
    assert not (tmp_path / "r.md").exists()
