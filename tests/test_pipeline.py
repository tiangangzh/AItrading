"""End-to-end tests for aitrading.pipeline and the offline HeuristicExplainer (all offline)."""

from __future__ import annotations

import copy
import inspect
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from aitrading.agent.explain import Explainer, ExplanationResult
from aitrading.agent.offline import HEURISTIC_NOTE, HeuristicExplainer, score_dislocation
from aitrading.core.models import (
    DislocationThesis,
    Document,
    DocumentKind,
    GroundingReport,
    PipelineResult,
    QuantEvidence,
    QuoteEvidence,
    RankedCandidate,
    TranscriptSegment,
)
from aitrading.core.policy import DataBoundary
from aitrading.data.base import PushdownResult
from aitrading.data.synthetic import ARCHETYPE_TO_DISLOCATION, SyntheticProvider
from aitrading.llm.base import LLMError, LLMRefusalError, ScriptedLLM
from aitrading.narrative.excerpts import build_bundle_excerpts, render_documents_for_prompt
from aitrading.narrative.grounding import normalize_text
from aitrading.narrative.retrieval import NarrativeBundle
from aitrading.pipeline import NO_LLM, ResearchPipeline, make_run_id
from aitrading.screen.catalog import default_catalog
from aitrading.screen.engine import ScreenValidationError
from aitrading.screen.features import FeatureEngine
from aitrading.screen.nl import HeuristicScreenTranslator, NLScreenTranslator
from aitrading.screen.spec import Condition, RankFactor, ScreenSpec, UniverseSpec

AS_OF = date(2026, 9, 30)
OBS = (
    "Find US mid-caps ($2-20B) that were in established uptrends (50-day above 200-day, positive 12-1 momentum) but "
    "have pulled back 15-40% from their 52-week highs over the past few months on heavy volume, now oversold (RSI "
    "under 40), still generating strong free cash flow (FCF yield above 4%) with revenue growth above 8%, and where "
    "short interest is elevated (above 6% of float). Rank by FCF yield, growth and the size of the drawdown, then "
    "read the latest earnings calls and explain the dislocation."
)
CATALOG = default_catalog()
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
SPEC_FEATURES = [
    "market_cap_usd_bn", "sma_50_vs_sma_200_pct", "return_12m_ex_1m_pct", "drawdown_from_52w_high_pct",
    "max_volume_ratio_20d", "rsi_14", "fcf_yield_pct", "revenue_growth_yoy_pct", "short_interest_pct_float",
]


def canonical_spec(**kw) -> ScreenSpec:
    c = Condition
    return ScreenSpec(
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
        **kw,
    )


# --------------------------------------------------------------------------------------------
# A scripted "LLM" explainer that builds its thesis from the prompt it is shown
# --------------------------------------------------------------------------------------------

_DOC_RX = re.compile(r'<document doc_id="([^"]+)" kind="([^"]+)"[^>]*>\n(.*?)\n</document>', re.S)
_ROW_RX = re.compile(r"^\| ([a-z0-9_]+) \| ([^|]+?) \| ", re.M)
_TURN_RX = re.compile(r"^([^():\n]{2,80}) \(([^()\n]{1,60})\): (.+)$", re.M)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip())[0]


def thesis_from_prompt(user: str) -> DislocationThesis:
    ticker = re.search(r"^Ticker: (\S+)$", user, re.M).group(1)
    table = {m.group(1): m.group(2).strip() for m in _ROW_RX.finditer(user)}
    quant = [
        QuantEvidence(feature=f, value=float(table[f]), interpretation="copied from the table")
        for f in ("fcf_yield_pct", "drawdown_from_52w_high_pct", "rsi_14")
        if table.get(f, "n/a") != "n/a"
    ]
    quotes: list[QuoteEvidence] = []
    for doc_id, kind, text in _DOC_RX.findall(user):
        if kind == "transcript":
            for name, role, body in _TURN_RX.findall(text):
                sentence = _first_sentence(body)
                if role.lower() not in {"analyst", "operator"} and len(sentence) >= 30:
                    quotes.append(QuoteEvidence(doc_id=doc_id, speaker=name, quote=sentence, interpretation="management"))
                    break
        else:
            line = next((ln for ln in text.splitlines() if len(ln.strip()) >= 30 and ln.strip() != "[...]"), None)
            if line:
                quotes.append(QuoteEvidence(doc_id=doc_id, speaker=None, quote=_first_sentence(line), interpretation="news"))
        if len(quotes) >= 2:
            break
    return DislocationThesis(
        ticker=ticker,
        headline=f"{ticker} scripted thesis",
        dislocation_type="insufficient_evidence",
        market_narrative="m",
        variant_view="v",
        why_dislocation_exists="w",
        quant_evidence=quant,
        narrative_evidence=quotes,
        catalysts=[],
        risks=[],
        invalidation_triggers=[],
        conviction="low",
        is_actionable=False,
        data_gaps=[],
    )


def scripted_llm(*, refuse: set[str] = frozenset(), error: set[str] = frozenset()) -> ScriptedLLM:
    def nl(purpose, system, user, model):
        return canonical_spec()

    def explain(purpose, system, user, model):
        thesis = thesis_from_prompt(user)
        if thesis.ticker in refuse:
            raise LLMRefusalError(f"[{purpose}] model declined (category=test)", category="test")
        if thesis.ticker in error:
            raise LLMError(f"[{purpose}] output truncated at max_tokens=16000")
        return thesis

    return ScriptedLLM({"nl_screen": nl, "explain": explain})


def explain_prompts(llm: ScriptedLLM) -> list[str]:
    return [p["user"] for p in llm.prompts if p["purpose"].startswith("explain")]


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider()


@pytest.fixture(scope="module")
def offline(provider, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    pipe = ResearchPipeline(provider, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=out)
    return pipe, pipe.run(OBS, AS_OF), out


@pytest.fixture(scope="module")
def llm_run(provider, tmp_path_factory):
    llm = scripted_llm()
    pipe = ResearchPipeline(provider, NLScreenTranslator(llm), Explainer(llm), out_dir=tmp_path_factory.mktemp("llm"))
    return llm, pipe.run(OBS, AS_OF)


# --------------------------------------------------------------------------------------------
# Offline end to end: canonical demo
# --------------------------------------------------------------------------------------------


def test_offline_screen_and_funnel(provider, offline):
    _, r, _ = offline
    assert [c["feature"] for c in r.spec["conditions"]] == SPEC_FEATURES
    assert [ScreenSpec.model_validate(r.spec).conditions[i].describe() for i in range(9)] == CANONICAL_CONDITIONS
    assert r.provider == "synthetic" and r.llm == NO_LLM and r.llm_calls == []
    assert r.as_of == AS_OF and r.observation == OBS
    assert r.universe_size == len(provider.get_universe(UniverseSpec(), AS_OF))
    assert r.survivors >= 5
    assert r.funnel[-1].remaining == r.survivors
    labels = [s.label for s in r.funnel]
    assert labels[:4] == ["country == US", "security_type in [common_stock]", "price >= 5", "avg_dollar_volume_20d_usd_mn >= 5"]
    assert labels[4:] == CANONICAL_CONDITIONS
    assert list(r.feature_coverage) == SPEC_FEATURES
    assert all(v > 0.9 for v in r.feature_coverage.values())
    assert r.pushdown_query is None
    assert r.started_at <= r.finished_at


def test_offline_ideas_explained_and_fully_grounded(offline):
    _, r, _ = offline
    assert len(r.ideas) == min(10, r.survivors)
    assert [i.candidate.rank for i in r.ideas] == list(range(1, len(r.ideas) + 1))
    explained, rest = r.ideas[:5], r.ideas[5:]
    for idea in explained:
        assert idea.error is None
        t, g = idea.thesis, idea.grounding
        assert t is not None and g is not None
        assert t.ticker == idea.candidate.ticker
        assert g.is_fully_grounded, [c for c in g.checks if c.status != "verified"]
        assert len(t.quant_evidence) >= 9 and len(t.narrative_evidence) >= 2
        assert t.data_gaps[0] == HEURISTIC_NOTE
        assert any(d.startswith("SYN-TR-") for d in idea.documents_used)
        assert {q.doc_id for q in t.narrative_evidence} <= set(idea.documents_used)
        assert set(idea.candidate.features) == set(SPEC_FEATURES)
    for idea in rest:
        assert idea.thesis is None and idea.grounding is None and idea.error is None


def test_offline_classification_matches_planted_archetypes(provider, offline):
    _, r, _ = offline
    for idea in r.ideas[:5]:
        expected = ARCHETYPE_TO_DISLOCATION[provider.archetype(idea.candidate.ticker)]
        assert idea.thesis.dislocation_type == expected, idea.candidate.ticker
        if expected == "structural_decline_value_trap":
            assert idea.thesis.is_actionable is False


def test_offline_artifacts_written(offline):
    _, r, out = offline
    run_dir = Path(out) / r.run_id
    assert sorted(p.name for p in run_dir.iterdir()) == ["documents.json", "features.csv", "result.json", "spec.json"]
    assert PipelineResult.model_validate_json((run_dir / "result.json").read_text()) == r
    assert ScreenSpec.model_validate_json((run_dir / "spec.json").read_text()).model_dump(mode="json") == r.spec
    feats = pd.read_csv(run_dir / "features.csv", index_col="ticker")
    assert len(feats) == r.survivors
    assert {i.candidate.ticker for i in r.ideas} <= set(feats.index)
    assert set(CATALOG.names()) <= set(feats.columns) and "name" in feats.columns
    docs = json.loads((run_dir / "documents.json").read_text())
    assert [d["ticker"] for d in docs] == [i.candidate.ticker for i in r.ideas[:5]]
    for entry, idea in zip(docs, r.ideas):
        assert [d["doc_id"] for d in entry["documents"]] == idea.documents_used
        for d in entry["documents"]:
            assert set(d) == {"doc_id", "kind", "title", "published_at", "source"}
            datetime.fromisoformat(d["published_at"])


def test_offline_warnings_summarise_capped_documents(offline):
    _, r, _ = offline
    capped = [w for w in r.warnings if "not shown to the explainer" in w and "per ticker" in w]
    assert len(capped) == 1  # one summary line, not one warning per routine cap
    assert not any("over max_documents_per_ticker" in w for w in r.warnings)
    assert len(r.warnings) == len(set(r.warnings))


def test_run_id_is_deterministic_and_suffixed(provider, offline, tmp_path):
    _, r, out = offline
    base = make_run_id(OBS, AS_OF, "synthetic")
    assert r.run_id == base and re.fullmatch(r"20260930-[0-9a-f]{12}", base)
    assert make_run_id(OBS, datetime(2026, 9, 30, 15), "synthetic") == base
    assert make_run_id(OBS + " ", AS_OF, "synthetic") != base
    assert make_run_id(OBS, AS_OF, "other") != base
    pipe = ResearchPipeline(provider, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=out, explain_top_k=0)
    again = pipe.run(OBS, AS_OF)
    assert again.run_id == f"{base}-2" and (Path(out) / again.run_id / "result.json").exists()
    assert [i.candidate.ticker for i in again.ideas] == [i.candidate.ticker for i in r.ideas]
    assert all(i.thesis is None for i in again.ideas)
    none = ResearchPipeline(provider, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=None, explain_top_k=0)
    assert none.run(OBS, AS_OF).run_id == base


def test_injected_clock(provider, tmp_path):
    ticks = iter([datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc), datetime(2026, 10, 2, 9, 1, tzinfo=timezone.utc)])
    pipe = ResearchPipeline(provider, None, None, out_dir=tmp_path, explain_top_k=0, clock=lambda: next(ticks))
    r = pipe.run(OBS, AS_OF, spec=canonical_spec())
    assert r.started_at == datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)
    assert r.finished_at == datetime(2026, 10, 2, 9, 1, tzinfo=timezone.utc)


class SpyExplainer:
    """Records what the pipeline hands an explainer, then delegates to the heuristic one."""

    def __init__(self):
        self.inner = HeuristicExplainer()
        self.calls: list[dict] = []

    def explain(self, candidate, features, documents, documents_prompt, spec, as_of, sector_context=None, signal_summary=None):
        self.calls.append(dict(candidate=candidate, features=features, documents=documents, documents_prompt=documents_prompt,
                               spec=spec, as_of=as_of, sector_context=sector_context, signal_summary=signal_summary))
        return self.inner.explain(candidate, features, documents, documents_prompt, spec, as_of, sector_context, signal_summary)


def test_explainer_inputs(provider):
    spy = SpyExplainer()
    pipe = ResearchPipeline(provider, None, spy, out_dir=None, explain_top_k=2)
    r = pipe.run(OBS, AS_OF, spec=canonical_spec(), top_n=3)
    assert len(spy.calls) == 2 and len(r.ideas) == 3
    universe = provider.get_universe(UniverseSpec(), AS_OF)
    screen = FeatureEngine(provider).build(universe, AS_OF, None).frame
    for call, idea in zip(spy.calls, r.ideas):
        t = idea.candidate.ticker
        feats = call["features"]
        assert list(feats) == CATALOG.names()
        assert feats["return_6m_percentile"] == pytest.approx(screen.at[t, "return_6m_percentile"])  # universe-relative
        assert feats["fcf_yield_pct"] == pytest.approx(screen.at[t, "fcf_yield_pct"])
        assert all(v is None or isinstance(v, (float, str)) for v in feats.values())
        # sector medians over the screen-pass rows of the same sector that pass the universe filters
        sector = feats["gics_sector"]
        pool = screen[(screen["gics_sector"] == sector) & (screen["price"] >= 5) & (screen["avg_dollar_volume_20d_usd_mn"] >= 5)]
        assert call["sector_context"]["rsi_14"] == pytest.approx(pool["rsi_14"].median())
        assert call["sector_context"]["fcf_yield_pct"] == pytest.approx(pool["fcf_yield_pct"].median())
        # documents shown are exactly those rendered into the prompt, all permitted by the boundary
        ids = re.findall(r'<document doc_id="([^"]+)"', call["documents_prompt"])
        assert ids == [d.doc_id for d in call["documents"]] == idea.documents_used
        assert all(provider.boundary.permits(d) for d in call["documents"])
        assert len(call["documents"]) <= provider.boundary.max_documents_per_ticker
        assert isinstance(call["signal_summary"], dict) and call["signal_summary"]
        assert call["as_of"] == AS_OF


def test_spec_given_skips_translator_and_top_n_override(provider):
    pipe = ResearchPipeline(provider, None, HeuristicExplainer(), out_dir=None, explain_top_k=1)
    r = pipe.run("my own words", AS_OF, spec=canonical_spec(), top_n=2)
    assert len(r.ideas) == 2 and r.spec["top_n"] == 2
    assert r.ideas[0].thesis is not None and r.ideas[1].thesis is None
    assert r.observation == "my own words"


@pytest.mark.parametrize("top_n", [0, 101])
def test_invalid_top_n(provider, top_n):
    pipe = ResearchPipeline(provider, None, None, out_dir=None)
    with pytest.raises(ValueError):
        pipe.run(OBS, AS_OF, spec=canonical_spec(), top_n=top_n)


class CountingProvider:
    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.name, self.capabilities, self.boundary = inner.name, inner.capabilities, inner.boundary

    def __getattr__(self, item):
        attr = getattr(self.inner, item)
        if callable(attr) and item.startswith("get_"):
            def wrapped(*a, **k):
                self.calls.append(item)
                return attr(*a, **k)
            return wrapped
        return attr


def test_invalid_spec_fails_before_any_data_and_leaves_no_run_dir(provider, tmp_path):
    counting = CountingProvider(provider)
    bad = canonical_spec()
    bad.conditions.append(Condition(feature="moon_phase", op=">", value=1))
    pipe = ResearchPipeline(counting, None, HeuristicExplainer(), out_dir=tmp_path)
    with pytest.raises(ScreenValidationError, match="moon_phase"):
        pipe.run(OBS, AS_OF, spec=bad)
    assert counting.calls == []
    assert list(tmp_path.iterdir()) == []


def test_translator_required_without_spec(provider):
    with pytest.raises(ValueError, match="translator"):
        ResearchPipeline(provider, None, None, out_dir=None).run(OBS, AS_OF)


@pytest.mark.parametrize("kw", [{"explain_top_k": -1}, {"documents_lookback_days": -1}, {"max_doc_chars_total": -5}])
def test_constructor_validation(provider, kw):
    with pytest.raises(ValueError):
        ResearchPipeline(provider, None, None, out_dir=None, **kw)


def test_no_survivors(provider, tmp_path):
    spec = canonical_spec()
    spec.conditions.append(Condition(feature="rsi_14", op=">", value=99))
    r = ResearchPipeline(provider, None, HeuristicExplainer(), out_dir=tmp_path).run(OBS, AS_OF, spec=spec)
    assert r.survivors == 0 and r.ideas == [] and "no names passed the screen" in r.warnings
    run_dir = tmp_path / r.run_id
    assert json.loads((run_dir / "documents.json").read_text()) == []
    assert len(pd.read_csv(run_dir / "features.csv")) == 0


def test_unsupported_requests_become_warnings(provider):
    spec = canonical_spec(unsupported_requests=["insider buying"])
    r = ResearchPipeline(provider, None, None, out_dir=None, explain_top_k=0).run(OBS, AS_OF, spec=spec)
    assert any("insider buying" in w for w in r.warnings)


# --------------------------------------------------------------------------------------------
# LLM path (ScriptedLLM)
# --------------------------------------------------------------------------------------------


def test_llm_path(offline, llm_run):
    llm, r = llm_run
    _, offline_result, _ = offline
    assert r.llm == "scripted"
    assert [c.describe() for c in ScreenSpec.model_validate(r.spec).conditions] == CANONICAL_CONDITIONS
    tickers = [i.candidate.ticker for i in r.ideas]
    assert tickers == [i.candidate.ticker for i in offline_result.ideas]
    assert [c.purpose for c in r.llm_calls] == ["nl_screen"] + [f"explain:{t}" for t in tickers[:5]]
    assert all(c.model == "scripted" for c in r.llm_calls)
    for idea in r.ideas[:5]:
        assert idea.error is None
        assert idea.grounding.is_fully_grounded, [c for c in idea.grounding.checks if c.status != "verified"]
        assert {c.kind for c in idea.grounding.checks} == {"quant", "quote"}
    prompts = explain_prompts(llm)
    assert len(prompts) == 5
    for p in prompts:
        assert "<documents>" in p and "| fcf_yield_pct |" in p and 'kind="transcript"' in p


def test_llm_calls_are_per_run(provider, tmp_path):
    llm = scripted_llm()
    pipe = ResearchPipeline(provider, NLScreenTranslator(llm), Explainer(llm), out_dir=None, explain_top_k=2)
    first = pipe.run(OBS, AS_OF)
    second = pipe.run(OBS, AS_OF)
    assert len(first.llm_calls) == len(second.llm_calls) == 3
    assert len(llm.calls) == 6


@pytest.mark.parametrize("mode, error_type", [("refuse", "LLMRefusalError"), ("error", "LLMError")])
def test_llm_failure_on_one_candidate_is_recorded(provider, offline, tmp_path, mode, error_type):
    _, ref, _ = offline
    victim = ref.ideas[1].candidate.ticker
    llm = scripted_llm(**{mode: {victim}})
    r = ResearchPipeline(provider, NLScreenTranslator(llm), Explainer(llm), out_dir=tmp_path).run(OBS, AS_OF)
    bad = r.ideas[1]
    assert bad.candidate.ticker == victim
    assert bad.thesis is None and bad.grounding is None
    assert bad.error.startswith(f"{error_type}:") and victim in bad.error
    assert bad.documents_used
    others = [i for k, i in enumerate(r.ideas[:5]) if k != 1]
    assert all(i.thesis is not None and i.error is None for i in others)
    assert (tmp_path / r.run_id / "result.json").exists()


def test_boundary_forbidding_transcripts_keeps_them_out_of_prompts(provider):
    restricted = copy.copy(provider)
    restricted.boundary = DataBoundary(provider="synthetic", allowed_document_kinds={DocumentKind.NEWS, DocumentKind.FILING})
    llm = scripted_llm()
    r = ResearchPipeline(restricted, NLScreenTranslator(llm), Explainer(llm), out_dir=None).run(OBS, AS_OF)
    prompts = explain_prompts(llm)
    assert len(prompts) == 5
    for idea, prompt in zip(r.ideas, prompts):
        t = idea.candidate.ticker
        transcripts = provider.get_documents(t, {DocumentKind.TRANSCRIPT}, date(2026, 1, 1), AS_OF)
        assert transcripts
        assert 'kind="transcript"' not in prompt
        for doc in transcripts:
            assert doc.doc_id not in prompt
            for seg in doc.segments:
                if seg.role not in ("Operator", "Analyst") and len(seg.text) > 120:
                    assert seg.text[:120] not in prompt
        assert not any(d.startswith("SYN-TR-") for d in idea.documents_used)
        assert any(f"{t}: withheld SYN-TR-" in w and "transcript text not permitted" in w for w in r.warnings)


def test_boundary_forbidding_numeric_features(provider):
    restricted = copy.copy(provider)
    restricted.boundary = DataBoundary(provider="synthetic", allow_numeric_features=False)
    llm = scripted_llm()
    r = ResearchPipeline(restricted, NLScreenTranslator(llm), Explainer(llm), out_dir=None, explain_top_k=2).run(OBS, AS_OF)
    for prompt in explain_prompts(llm):
        assert "(No feature values were provided for this candidate.)" in prompt
        assert "| fcf_yield_pct |" not in prompt
    assert any("feature values withheld" in w for w in r.warnings)


# --------------------------------------------------------------------------------------------
# Push-down
# --------------------------------------------------------------------------------------------


class PushdownProvider:
    """Synthetic provider that also 'compiles' the screen to a vendor query returning ``tickers``."""

    def __init__(self, inner, tickers=None, exc=None):
        self.inner, self.tickers, self.exc = inner, tickers, exc
        self.name, self.capabilities, self.boundary = inner.name, inner.capabilities, inner.boundary

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def pushdown_screen(self, spec, as_of):
        if self.exc is not None:
            raise self.exc
        return PushdownResult(tickers=list(self.tickers), query="SCREEN(test)", pushed_conditions=[c.describe() for c in spec.conditions])


def test_pushdown_narrows_universe(provider, offline):
    _, ref, out = offline
    survivors = list(pd.read_csv(Path(out) / ref.run_id / "features.csv", index_col="ticker").index)
    assert len(survivors) == ref.survivors
    extra = [t for t in provider.get_universe(UniverseSpec(), AS_OF).index if t not in survivors][:7]
    pp = PushdownProvider(provider, tickers=survivors + extra + ["NOPE1"])
    r = ResearchPipeline(pp, HeuristicScreenTranslator(), HeuristicExplainer(), out_dir=None, explain_top_k=1).run(OBS, AS_OF)
    assert r.pushdown_query == "SCREEN(test)"
    assert r.funnel[0].label == "vendor push-down (synthetic)" and r.funnel[0].remaining == len(survivors) + 7
    assert r.survivors == ref.survivors
    assert [i.candidate.ticker for i in r.ideas] == [i.candidate.ticker for i in ref.ideas]
    assert any("outside the universe" in w for w in r.warnings)
    assert any("sector medians omitted" in w for w in r.warnings)
    assert r.universe_size == ref.universe_size
    assert r.ideas[0].grounding.is_fully_grounded


def test_pushdown_failure_falls_back_to_local_screen(provider, offline):
    _, ref, _ = offline
    pp = PushdownProvider(provider, exc=RuntimeError("BQL syntax error"))
    r = ResearchPipeline(pp, HeuristicScreenTranslator(), None, out_dir=None).run(OBS, AS_OF)
    assert r.pushdown_query is None
    assert any("push-down failed" in w and "BQL syntax error" in w for w in r.warnings)
    assert [i.candidate.ticker for i in r.ideas] == [i.candidate.ticker for i in ref.ideas]
    assert r.funnel == ref.funnel


# --------------------------------------------------------------------------------------------
# HeuristicExplainer on hand-written inputs
# --------------------------------------------------------------------------------------------


def test_heuristic_explain_signature_matches_explainer():
    a = inspect.signature(HeuristicExplainer.explain).parameters
    b = inspect.signature(Explainer.explain).parameters
    assert list(a) == list(b)
    assert all(a[k].default == b[k].default for k in a)
    assert not hasattr(HeuristicExplainer(), "llm")


@pytest.mark.parametrize(
    "features, tags, expected, conviction",
    [
        (dict(revenue_growth_yoy_pct=10, revenue_growth_last_q_yoy_pct=-3, eps_revision_3m_pct=-25,
              revenue_revision_3m_pct=-12, operating_margin_change_yoy_pp=-1.5),
         {"structural_concern": 2, "evasive_answer": 3}, "structural_decline_value_trap", "high"),
        (dict(revenue_growth_last_q_yoy_pct=7, eps_revision_3m_pct=-6), {"transitory_language": 5, "demand_resilience": 2},
         "transitory_fundamental_shock", "high"),
        (dict(last_eps_surprise_pct=8, net_debt_usd_bn=-1.2), {"guidance_conservative": 4, "results_beat": 2},
         "guidance_reset_overreaction", "high"),
        (dict(eps_revision_3m_pct=0.5, revenue_growth_last_q_yoy_pct=10), {"peer_contagion": 1, "limited_exposure": 3},
         "sector_or_macro_contagion", "high"),
        (dict(short_interest_pct_float=22, days_to_cover=8, eps_revision_3m_pct=1, revenue_growth_last_q_yoy_pct=5), {},
         "technical_or_flow_driven", "low"),  # no narrative support: thin evidence
        ({}, {}, "insufficient_evidence", "low"),
        (dict(revenue_growth_last_q_yoy_pct=-2, eps_revision_3m_pct=-20), {}, "structural_decline_value_trap", "medium"),
        (dict(revenue_growth_last_q_yoy_pct=None, eps_revision_3m_pct=float("nan")), {"transitory_language": 1},
         "insufficient_evidence", "low"),
    ],
)
def test_score_dislocation_rules(features, tags, expected, conviction):
    s = score_dislocation(features, tags)
    assert s.dislocation_type == expected
    assert s.conviction == conviction


def test_score_dislocation_ties_go_to_the_sceptical_type():
    # value trap: 2 (quant) + 1 (narrative) = 3; transitory: 0 + 3 = 3 -> value trap wins the tie
    s = score_dislocation({"revenue_growth_last_q_yoy_pct": -1.0}, {"structural_concern": 1, "transitory_language": 3})
    assert s.total("structural_decline_value_trap") == s.total("transitory_fundamental_shock") == 3
    assert s.dislocation_type == "structural_decline_value_trap"


def test_score_dislocation_conviction_needs_a_transcript():
    f = dict(revenue_growth_last_q_yoy_pct=7, eps_revision_3m_pct=-6)
    assert score_dislocation(f, {"transitory_language": 5}, has_transcript=False).conviction == "medium"


def test_score_dislocation_uses_sector_context():
    s = score_dislocation({"eps_revision_3m_pct": 0.0}, {"limited_exposure": 1}, {"return_3m_pct": -12.0})
    assert "return_3m_pct" in s.flags["sector_or_macro_contagion"]
    assert s.dislocation_type == "sector_or_macro_contagion"


CEO_TEXT = (
    "Revenue grew 9% in the quarter despite a one-time headwind from the ERP cut-over. "
    "We expect the shipments to recover as volumes normalize in the next quarter."
)
CFO_TEXT = (
    "Order intake was unaffected and grew 8% in the quarter. "
    "We are not seeing any share loss with our top customers."
)
NEWS_TEXT = "Shares of Acme fell 15% after the second-quarter report. Analysts said the miss was a timing issue."


def _docs() -> list[Document]:
    segs = [
        TranscriptSegment(speaker="Operator", role="Operator", section="prepared_remarks", text="Welcome to the call."),
        TranscriptSegment(speaker="Jane Roe", role="CEO", section="prepared_remarks", text=CEO_TEXT),
        TranscriptSegment(speaker="Sam Poe", role="CFO", section="prepared_remarks", text=CFO_TEXT),
        TranscriptSegment(speaker="Ann Lee", role="Analyst", section="qa", text="Is the issue really temporary?"),
    ]
    tr = Document(
        doc_id="TR-1", ticker="ACME", kind=DocumentKind.TRANSCRIPT, title="Acme Q2 call", published_at=datetime(2026, 8, 3, 8, 30),
        source="test", text="\n\n".join(f"{s.speaker}: {s.text}" for s in segs), segments=segs,
    )
    news = Document(
        doc_id="NW-1", ticker="ACME", kind=DocumentKind.NEWS, title="Acme slides", published_at=datetime(2026, 8, 3, 16),
        source="test", text=NEWS_TEXT,
    )
    return [tr, news]


def _features(**over) -> dict:
    f = {name: None for name in CATALOG.names()}
    f.update(
        market_cap_usd_bn=8.0, sma_50_vs_sma_200_pct=3.0, return_12m_ex_1m_pct=25.0, drawdown_from_52w_high_pct=-24.0,
        max_volume_ratio_20d=3.1, rsi_14=31.0, fcf_yield_pct=6.81349, revenue_growth_yoy_pct=12.0, short_interest_pct_float=8.0,
        revenue_growth_last_q_yoy_pct=9.0, eps_revision_3m_pct=-6.0, gics_sector="Industrials", days_to_next_earnings=33.0,
    )
    f.update(over)
    return f


def _candidate() -> RankedCandidate:
    return RankedCandidate(ticker="ACME", name="Acme Corp", rank=1, score=0.9)


def _prompt(docs: list[Document]) -> str:
    return render_documents_for_prompt(build_bundle_excerpts(NarrativeBundle(ticker="ACME", documents=docs)))


def test_heuristic_explain_handwritten_documents():
    docs = _docs()
    res = HeuristicExplainer().explain(_candidate(), _features(), docs, _prompt(docs), canonical_spec(), AS_OF,
                                       sector_context={"rsi_14": 55.0})
    assert isinstance(res, ExplanationResult) and res.rounds == 0
    t = res.thesis
    assert t.ticker == "ACME" and t.dislocation_type == "transitory_fundamental_shock"
    assert res.grounding.is_fully_grounded, [c for c in res.grounding.checks if c.status != "verified"]
    fcf = next(q for q in t.quant_evidence if q.feature == "fcf_yield_pct")
    assert fcf.value == 6.813  # as the feature table displays it
    rsi = next(q for q in t.quant_evidence if q.feature == "rsi_14")
    assert "passes" in rsi.interpretation and "sector median 55" in rsi.interpretation
    speakers = {q.speaker for q in t.narrative_evidence if q.doc_id == "TR-1"}
    assert speakers <= {"Jane Roe", "Sam Poe"} and speakers
    for q in t.narrative_evidence:
        source = next(d for d in docs if d.doc_id == q.doc_id)
        assert q.quote in source.text
        assert "Ann Lee" != q.speaker
    assert any("33" in c for c in t.catalysts)
    assert t.data_gaps[0] == HEURISTIC_NOTE
    assert any(g.startswith("Missing features:") for g in t.data_gaps)
    assert t.is_actionable


def test_heuristic_explain_without_documents():
    res = HeuristicExplainer().explain(_candidate(), _features(), [], "", canonical_spec(), AS_OF)
    t = res.thesis
    assert t.narrative_evidence == []
    assert t.dislocation_type == "insufficient_evidence" and t.conviction == "low" and not t.is_actionable
    assert any("No documents" in g for g in t.data_gaps)
    assert res.grounding.is_fully_grounded  # quant evidence only, all copied from the table


def test_heuristic_value_trap_from_numbers_alone():
    f = _features(revenue_growth_last_q_yoy_pct=-4.0, eps_revision_3m_pct=-22.0, revenue_revision_3m_pct=-11.0)
    res = HeuristicExplainer().explain(_candidate(), f, [], "", canonical_spec(), AS_OF)
    assert res.thesis.dislocation_type == "structural_decline_value_trap" and not res.thesis.is_actionable
    cited = [q.feature for q in res.thesis.quant_evidence]
    assert cited[:9] == SPEC_FEATURES and "eps_revision_3m_pct" in cited
    assert res.grounding.is_fully_grounded


def test_heuristic_skips_missing_features_and_lists_them():
    f = _features(rsi_14=None, short_interest_pct_float=float("nan"))
    res = HeuristicExplainer().explain(_candidate(), f, [], "", canonical_spec(), AS_OF)
    cited = {q.feature for q in res.thesis.quant_evidence}
    assert "rsi_14" not in cited and "short_interest_pct_float" not in cited
    gaps = next(g for g in res.thesis.data_gaps if g.startswith("Missing features:"))
    assert "rsi_14" in gaps and "short_interest_pct_float" in gaps


def test_heuristic_quotes_only_text_that_was_shown():
    docs = _docs()
    shown = _prompt([docs[1]])  # only the news item was rendered into the prompt
    res = HeuristicExplainer().explain(_candidate(), _features(), docs, shown, canonical_spec(), AS_OF)
    assert {q.doc_id for q in res.thesis.narrative_evidence} <= {"NW-1"}


def test_heuristic_truncates_long_sentences_verbatim():
    long = "This was a one-time issue " + "and it affected several plants across the network " * 12 + "in the quarter."
    doc = Document(doc_id="NW-2", ticker="ACME", kind=DocumentKind.NEWS, title="t", published_at=datetime(2026, 8, 3),
                   source="test", text=long)
    res = HeuristicExplainer().explain(_candidate(), _features(), [doc], _prompt([doc]), canonical_spec(), AS_OF)
    (q,) = [q for q in res.thesis.narrative_evidence if q.doc_id == "NW-2"]
    assert len(q.quote) <= 400 and long.startswith(q.quote)
    assert res.grounding.is_fully_grounded


def test_heuristic_uses_custom_verifier():
    seen = []

    def verifier(thesis, documents, features):
        seen.append((thesis.ticker, len(documents), features["fcf_yield_pct"]))
        return GroundingReport(ticker=thesis.ticker, checks=[])

    res = HeuristicExplainer(verifier=verifier).explain(_candidate(), _features(), _docs(), "", canonical_spec(), AS_OF)
    assert seen == [("ACME", 2, 6.813)]
    assert res.grounding.checks == []


def test_heuristic_quote_text_is_normalisable_in_prompt():
    docs = _docs()
    prompt = _prompt(docs)
    res = HeuristicExplainer().explain(_candidate(), _features(), docs, prompt, canonical_spec(), AS_OF)
    for q in res.thesis.narrative_evidence:
        assert normalize_text(q.quote) in normalize_text(prompt)


# --------------------------------------------------------------------------------------------
# Integration with adapter conventions
# --------------------------------------------------------------------------------------------


class WarningProvider:
    """Adapter-style provider that appends caveats to ``self.warnings`` (as free / Bloomberg / LSEG do)."""

    def __init__(self, inner):
        self.inner, self.warnings = inner, ["stale caveat from an earlier run"]
        self.name, self.capabilities, self.boundary = inner.name, inner.capabilities, inner.boundary

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def get_universe(self, spec, as_of):
        self.warnings.append("listing reflects today's constituents (survivorship bias)")
        return self.inner.get_universe(spec, as_of)


def test_provider_warnings_from_this_run_are_kept(provider):
    wp = WarningProvider(provider)
    r = ResearchPipeline(wp, None, None, out_dir=None).run(OBS, AS_OF, spec=canonical_spec())
    assert "provider synthetic: listing reflects today's constituents (survivorship bias)" in r.warnings
    assert not any("stale caveat" in w for w in r.warnings)


def test_invalid_structured_output_is_recorded_per_idea(provider):
    class BadExplainer:
        def explain(self, candidate, features, documents, documents_prompt, spec, as_of, sector_context=None, signal_summary=None):
            DislocationThesis.model_validate({"ticker": candidate.ticker})  # raises pydantic.ValidationError

    r = ResearchPipeline(provider, None, BadExplainer(), out_dir=None, explain_top_k=2).run(OBS, AS_OF, spec=canonical_spec())
    assert all(i.error.startswith("ValidationError:") for i in r.ideas[:2])
    assert all(i.error is None and i.thesis is None for i in r.ideas[2:])


def test_malformed_pushdown_result_falls_back(provider, offline):
    _, ref, _ = offline
    pp = PushdownProvider(provider, tickers=None)  # iterating None fails inside the push-down step
    r = ResearchPipeline(pp, None, None, out_dir=None).run(OBS, AS_OF, spec=canonical_spec())
    assert r.pushdown_query is None
    assert any("push-down failed" in w for w in r.warnings)
    assert r.survivors == ref.survivors and r.funnel == ref.funnel
