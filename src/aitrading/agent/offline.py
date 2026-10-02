"""Offline heuristic explainer: a deterministic, rule-based ``DislocationThesis`` without an LLM.

``HeuristicExplainer.explain`` has the same signature and return type as
``aitrading.agent.explain.Explainer.explain`` so the pipeline can run end to end without an API
key, and so LLM explanations have a transparent baseline to be compared against. It never invents
evidence: every quant_evidence value is copied from the feature table exactly as the LLM explainer
would display it, and every quote is a verbatim sentence (or a verbatim leading part of one) found
by the narrative tagger (``aitrading.narrative.signals``) in the text that was shown. The result is
checked by the same grounding verifier as the LLM's output.

Classification
--------------
Each dislocation type gets a score = narrative points + quant points; the highest *eligible* score
wins (ties: value trap, guidance reset, contagion, transitory, flow - the most sceptical first).

Narrative points: ``weight x min(count, 3)`` per signal tag in the shown documents.

====================================  ==========================================================
type                                  tags (weight)
====================================  ==========================================================
structural_decline_value_trap         structural_concern, share_loss, pricing_pressure,
                                      customer_churn, guidance_withdrawn, capital_return_cut (1);
                                      evasive_answer, demand_weakness, margin_pressure,
                                      results_miss, guidance_lowered (0.5)
transitory_fundamental_shock          transitory_language, inventory_destocking (1); no_share_loss,
                                      no_demand_weakness, no_customer_churn, no_pricing_pressure,
                                      no_structural_concern, demand_resilience, demand_recovery,
                                      backlog_strength, fx_headwind (0.5)
guidance_reset_overreaction           guidance_conservative, results_beat (1); guidance_lowered (0.5)
sector_or_macro_contagion             peer_contagion (1.5); limited_exposure (1)
====================================  ==========================================================

Quant points (a flag needs its inputs present):

* value trap: revenue_growth_last_q_yoy_pct < 0 (+2); last-quarter growth more than 10 pp below
  TTM growth (+1); eps_revision_3m_pct <= -15 (+2); revenue_revision_3m_pct <= -10 (+1);
  operating_margin_change_yoy_pp <= -1 (+1).
* transitory: revenue_growth_last_q_yoy_pct > 0 (+1); -15 < eps_revision_3m_pct < 0 (+0.5).
* guidance reset: last_eps_surprise_pct > 0 (+1); net_debt_usd_bn < 0, i.e. net cash (+1).
* contagion: eps_revision_3m_pct > -3 (+1); revenue_growth_last_q_yoy_pct > 0 (+0.5); sector
  median return_3m_pct < -5 (+1).
* flow-driven (technical_or_flow_driven): short_interest_pct_float >= 15 and days_to_cover >= 5
  (+1.5, required); eps_revision_3m_pct > -3 (+0.5); revenue_growth_last_q_yoy_pct > 0 (+0.5).

Eligibility: value trap needs a score >= 3 with quant points >= 2 or narrative points >= 3;
transitory, guidance reset and contagion need narrative points >= 1 and a score >= 2; flow-driven
needs the crowded-short flag, a score >= 2 and less than 1.5 narrative points for each of transitory,
guidance reset and contagion (the documents offer no fundamental explanation). When
nothing is eligible the thesis is ``insufficient_evidence``. ``misunderstood_change`` needs
judgement the rules cannot supply and is never assigned.

Conviction (evidence agreement): high when quant points >= 1, narrative points >= 2, the runner-up
scores at most half the winner and a transcript was read; medium when quant and narrative both
support the winner and the runner-up scores at most 75% of it, or the runner-up scores at most half;
low otherwise (always low for insufficient_evidence). ``is_actionable`` is true for a genuine
dislocation type with medium or high conviction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

import pandas as pd

from aitrading.agent.explain import ExplanationResult, Verifier, display_value, render_feature_table
from aitrading.core.models import (
    DislocationThesis,
    Document,
    DocumentKind,
    GroundingReport,
    QuantEvidence,
    QuoteEvidence,
    RankedCandidate,
)
from aitrading.narrative.grounding import normalize_text, verify_thesis
from aitrading.narrative.signals import NarrativeSignal, extract_signals_from_documents
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.engine import evaluate_condition
from aitrading.screen.spec import Condition, ScreenSpec

__all__ = [
    "HeuristicExplainer",
    "HEURISTIC_NOTE",
    "NARRATIVE_WEIGHTS",
    "TAG_DESCRIPTIONS",
    "score_dislocation",
    "DislocationScores",
]

VT = "structural_decline_value_trap"
TS = "transitory_fundamental_shock"
GR = "guidance_reset_overreaction"
SC = "sector_or_macro_contagion"
TF = "technical_or_flow_driven"
IE = "insufficient_evidence"
_PRIORITY = (VT, GR, SC, TS, TF)  # tie-break order: most sceptical first
_GENUINE = frozenset({TS, GR, SC, TF, "misunderstood_change"})

HEURISTIC_NOTE = (
    "Heuristic baseline: dislocation type, conviction and text come from transparent rules over feature "
    "thresholds and phrase-matched narrative signals (aitrading.agent.offline), not from an analyst's or "
    "an LLM's judgement; use it as a starting point."
)

NARRATIVE_WEIGHTS: dict[str, dict[str, float]] = {
    VT: {
        "structural_concern": 1.0, "share_loss": 1.0, "pricing_pressure": 1.0, "customer_churn": 1.0,
        "guidance_withdrawn": 1.0, "capital_return_cut": 1.0, "evasive_answer": 0.5, "demand_weakness": 0.5,
        "margin_pressure": 0.5, "results_miss": 0.5, "guidance_lowered": 0.5,
    },
    TS: {
        "transitory_language": 1.0, "inventory_destocking": 1.0, "no_share_loss": 0.5, "no_demand_weakness": 0.5,
        "no_customer_churn": 0.5, "no_pricing_pressure": 0.5, "no_structural_concern": 0.5,
        "demand_resilience": 0.5, "demand_recovery": 0.5, "backlog_strength": 0.5, "fx_headwind": 0.5,
    },
    GR: {"guidance_conservative": 1.0, "results_beat": 1.0, "guidance_lowered": 0.5},
    SC: {"peer_contagion": 1.5, "limited_exposure": 1.0},
    TF: {},
}
_COUNT_CAP = 3

TAG_DESCRIPTIONS: dict[str, str] = {
    "transitory_language": "the hit is framed as one-off or temporary",
    "inventory_destocking": "the shortfall is attributed to customer destocking, a timing effect",
    "demand_resilience": "underlying demand is described as intact",
    "demand_recovery": "demand is described as recovering",
    "backlog_strength": "backlog or orders are described as strong",
    "no_share_loss": "management says it is not losing share",
    "no_demand_weakness": "management says demand is not weakening",
    "no_pricing_pressure": "management says pricing is not under pressure",
    "no_customer_churn": "management says customers are not leaving",
    "no_structural_concern": "management rejects a structural explanation",
    "no_margin_pressure": "management says margins are not under pressure",
    "guidance_conservative": "guidance is described as deliberately conservative",
    "guidance_raised": "guidance was raised",
    "guidance_reaffirmed": "guidance was reaffirmed",
    "guidance_lowered": "guidance was lowered",
    "guidance_withdrawn": "guidance was withdrawn or no longer reaffirmed",
    "results_beat": "results beat expectations",
    "results_miss": "results fell short of expectations",
    "capital_return": "capital is being returned to shareholders",
    "capital_return_cut": "capital returns were cut",
    "balance_sheet_strength": "the balance sheet is described as strong",
    "peer_contagion": "the sell-off is tied to a peer or industry event",
    "limited_exposure": "management describes limited exposure to the industry issue",
    "structural_concern": "a structural or competitive deterioration is acknowledged",
    "share_loss": "market share is being lost",
    "pricing_pressure": "pricing pressure is acknowledged",
    "pricing_power": "pricing is holding up",
    "customer_churn": "customer losses or churn are acknowledged",
    "evasive_answer": "management deflects or declines to quantify",
    "demand_weakness": "demand is described as weak",
    "margin_pressure": "margins are under pressure",
    "fx_headwind": "currency is cited as a headwind",
    "management_change": "a management change is mentioned",
}

# Features cited after the screen's own, as the metrics that separate a dislocation from a trap.
_DECISIVE = (
    "revenue_growth_last_q_yoy_pct", "eps_revision_3m_pct", "revenue_revision_3m_pct",
    "operating_margin_change_yoy_pp", "last_eps_surprise_pct", "net_debt_usd_bn",
    "short_interest_change_1m_pct", "days_to_next_earnings",
)
_MAX_DECISIVE = 5
_MAX_SUPPORT_QUOTES = 4
_MAX_QUOTE_CHARS = 400
_MIN_QUOTE_CHARS = 20
_MAX_LISTED_GAPS = 25


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def _num(features: Mapping[str, Any], name: str) -> float | None:
    """Finite float value of ``name``, else None (missing, NaN, text)."""
    v = features.get(name)
    if v is None or isinstance(v, str):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass
class DislocationScores:
    """Per-type points and the flags behind them (for the thesis text and for audit)."""

    narrative: dict[str, float]
    quant: dict[str, float]
    flags: dict[str, list[str]]  # type -> feature names whose flag fired (in rule order)
    tag_counts: dict[str, int]
    crowded_short: bool = False
    dislocation_type: str = IE
    runner_up: str | None = None
    conviction: str = "low"
    reasons: list[str] = field(default_factory=list)

    def total(self, t: str) -> float:
        return self.narrative.get(t, 0.0) + self.quant.get(t, 0.0)


def score_dislocation(
    features: Mapping[str, Any],
    tag_counts: Mapping[str, int],
    sector_context: Mapping[str, Any] | None = None,
    *,
    has_transcript: bool = True,
) -> DislocationScores:
    """Apply the rule table in the module docstring; returns scores, flags and the decision."""
    sector_context = sector_context or {}
    narrative = {t: sum(w * min(int(tag_counts.get(tag, 0)), _COUNT_CAP) for tag, w in ws.items()) for t, ws in NARRATIVE_WEIGHTS.items()}
    quant = {t: 0.0 for t in NARRATIVE_WEIGHTS}
    flags: dict[str, list[str]] = {t: [] for t in NARRATIVE_WEIGHTS}

    def add(t: str, points: float, feature: str) -> None:
        quant[t] += points
        if feature not in flags[t]:
            flags[t].append(feature)

    g_ttm = _num(features, "revenue_growth_yoy_pct")
    g_q = _num(features, "revenue_growth_last_q_yoy_pct")
    eps_rev = _num(features, "eps_revision_3m_pct")
    rev_rev = _num(features, "revenue_revision_3m_pct")
    om_chg = _num(features, "operating_margin_change_yoy_pp")
    surprise = _num(features, "last_eps_surprise_pct")
    net_debt = _num(features, "net_debt_usd_bn")
    si = _num(features, "short_interest_pct_float")
    dtc = _num(features, "days_to_cover")
    sector_r3m = _num(sector_context, "return_3m_pct")

    if g_q is not None and g_q < 0:
        add(VT, 2.0, "revenue_growth_last_q_yoy_pct")
    if g_q is not None and g_ttm is not None and g_q < g_ttm - 10:
        add(VT, 1.0, "revenue_growth_last_q_yoy_pct")
    if eps_rev is not None and eps_rev <= -15:
        add(VT, 2.0, "eps_revision_3m_pct")
    if rev_rev is not None and rev_rev <= -10:
        add(VT, 1.0, "revenue_revision_3m_pct")
    if om_chg is not None and om_chg <= -1:
        add(VT, 1.0, "operating_margin_change_yoy_pp")

    if g_q is not None and g_q > 0:
        add(TS, 1.0, "revenue_growth_last_q_yoy_pct")
        add(SC, 0.5, "revenue_growth_last_q_yoy_pct")
    if eps_rev is not None and -15 < eps_rev < 0:
        add(TS, 0.5, "eps_revision_3m_pct")
    if surprise is not None and surprise > 0:
        add(GR, 1.0, "last_eps_surprise_pct")
    if net_debt is not None and net_debt < 0:
        add(GR, 1.0, "net_debt_usd_bn")
    if eps_rev is not None and eps_rev > -3:
        add(SC, 1.0, "eps_revision_3m_pct")
    if sector_r3m is not None and sector_r3m < -5:
        add(SC, 1.0, "return_3m_pct")

    crowded = si is not None and dtc is not None and si >= 15 and dtc >= 5
    if crowded:
        add(TF, 1.5, "short_interest_pct_float")
        add(TF, 0.0, "days_to_cover")
        if eps_rev is not None and eps_rev > -3:
            add(TF, 0.5, "eps_revision_3m_pct")
        if g_q is not None and g_q > 0:
            add(TF, 0.5, "revenue_growth_last_q_yoy_pct")

    out = DislocationScores(narrative=narrative, quant=quant, flags=flags, tag_counts=dict(tag_counts), crowded_short=crowded)

    def eligible(t: str) -> bool:
        total = out.total(t)
        if t == VT:
            return total >= 3 and (quant[VT] >= 2 or narrative[VT] >= 3)
        if t == TF:
            return crowded and total >= 2 and all(narrative[o] < 1.5 for o in (TS, GR, SC))
        return narrative[t] >= 1 and total >= 2

    candidates = [t for t in _PRIORITY if eligible(t)]
    if not candidates:
        out.dislocation_type = IE
        out.conviction = "low"
        out.reasons.append("no rule reached its eligibility threshold")
        return out
    winner = max(candidates, key=lambda t: (out.total(t), -_PRIORITY.index(t)))
    others = [t for t in _PRIORITY if t != winner]
    runner = max(others, key=lambda t: (out.total(t), -_PRIORITY.index(t)))
    out.dislocation_type = winner
    out.runner_up = runner if out.total(runner) > 0 else None
    w, r = out.total(winner), out.total(runner)
    ratio = r / w if w > 0 else 1.0
    q, n = quant[winner], narrative[winner]
    if q >= 1 and n >= 2 and ratio <= 0.5 and has_transcript:
        out.conviction = "high"
    elif (q > 0 and n > 0 and ratio <= 0.75) or ratio <= 0.5:
        out.conviction = "medium"
    else:
        out.conviction = "low"
    out.reasons.append(f"{winner} scored {w:g} (narrative {n:g}, quant {q:g}); runner-up {runner} scored {r:g}")
    return out


# --------------------------------------------------------------------------------------------
# Explainer
# --------------------------------------------------------------------------------------------


def _quote_text(sentence: str) -> str | None:
    """The sentence, or its longest leading part of <= 400 chars ending at a word boundary."""
    s = sentence.strip()
    if len(s) > _MAX_QUOTE_CHARS:
        cut = s.rfind(" ", 0, _MAX_QUOTE_CHARS)
        s = s[: cut if cut > 0 else _MAX_QUOTE_CHARS].rstrip(" ,;:-")
    return s if len(s) >= _MIN_QUOTE_CHARS else None


class HeuristicExplainer:
    """Deterministic stand-in for ``Explainer`` (no LLM). See the module docstring for the rules."""

    name = "heuristic"

    def __init__(self, catalog: FeatureCatalog | None = None, *, verifier: Verifier | None = None):
        self.catalog = catalog or default_catalog()
        self.verifier: Verifier = verifier or verify_thesis

    # -- public API (same as Explainer.explain) ---------------------------------------------

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
        """Build, then verify, a rule-based thesis. ``signal_summary`` is accepted for signature parity;
        signals are recomputed from ``documents`` so each one keeps its verbatim sentence."""
        _, shown = render_feature_table(features, self.catalog, spec=spec)
        signals = self._shown_signals(documents, documents_prompt)
        counts: dict[str, int] = {}
        for s in signals:
            counts[s.tag] = counts.get(s.tag, 0) + 1
        has_transcript = any(d.kind == DocumentKind.TRANSCRIPT for d in documents)
        scores = score_dislocation(features, counts, sector_context, has_transcript=has_transcript)
        thesis = self._thesis(candidate, features, shown, documents, signals, scores, spec, sector_context or {})
        report: GroundingReport = self.verifier(thesis, documents, shown)
        return ExplanationResult(thesis=thesis, grounding=report, rounds=0)

    # -- evidence ----------------------------------------------------------------------------

    @staticmethod
    def _shown_signals(documents: list[Document], documents_prompt: str) -> list[NarrativeSignal]:
        """Signals whose sentence appears in the prompt text (all signals when no prompt was rendered)."""
        signals = extract_signals_from_documents(documents)
        if not documents_prompt or not documents_prompt.strip():
            return signals
        shown = normalize_text(documents_prompt)
        return [s for s in signals if normalize_text(s.sentence) in shown]

    def _quant_evidence(
        self,
        features: Mapping[str, Any],
        shown: Mapping[str, Any],
        spec: ScreenSpec,
        scores: DislocationScores,
        sector_context: Mapping[str, Any],
    ) -> list[QuantEvidence]:
        out: list[QuantEvidence] = []
        used: set[str] = set()

        def cite(name: str, interpretation: str) -> None:
            v = shown.get(name)
            if name in used or not isinstance(v, float) or not math.isfinite(v):
                return
            if name in self.catalog and self.catalog[name].dtype == "category":
                return
            med = sector_context.get(name)
            if med is not None and display_value(med)[1] is not None:
                interpretation += f"; sector median {display_value(med)[0]}"
            used.add(name)
            out.append(QuantEvidence(feature=name, value=v, interpretation=interpretation))

        for cond in spec.all_conditions():
            cite(cond.feature, self._condition_note(cond, features))
        for f in spec.ranking:
            cite(f.feature, f"ranking factor ({f.direction.replace('_', ' ')})")
        winner_flags = scores.flags.get(scores.dislocation_type, [])
        decisive = [f for f in winner_flags if f in _DECISIVE] + [f for f in _DECISIVE if f not in winner_flags]
        n_before = len(out)
        for name in decisive:
            if len(out) - n_before >= _MAX_DECISIVE:
                break
            cite(name, self._decisive_note(name, features))
        return out

    def _condition_note(self, cond: Condition, features: Mapping[str, Any]) -> str:
        names = sorted(cond.features())
        if any(features.get(c) is None for c in names):
            return f"screen condition {cond.describe()}: input missing"
        try:
            row = pd.DataFrame([{c: features[c] for c in names}])
            ok = bool(evaluate_condition(cond, row, catalog=self.catalog).iloc[0])
            verdict = "passes" if ok else "fails"
        except Exception:  # noqa: BLE001 - a note must never break the explanation
            verdict = "cannot be evaluated"
        return f"screen condition {cond.describe()}: {verdict}"

    @staticmethod
    def _decisive_note(name: str, features: Mapping[str, Any]) -> str:
        v = _num(features, name)
        g_ttm = _num(features, "revenue_growth_yoy_pct")
        if name == "revenue_growth_last_q_yoy_pct" and v is not None:
            if v < 0:
                return "latest quarter revenue is shrinking year on year: trailing growth overstates the current trend"
            if g_ttm is not None and v < g_ttm - 10:
                return "latest quarter growth is well below trailing growth: sharp deceleration"
            return "latest quarter revenue still growing year on year"
        if name == "eps_revision_3m_pct" and v is not None:
            if v <= -15:
                return "hard cut to next-12m EPS consensus over 3 months"
            if v < 0:
                return "modest trim to next-12m EPS consensus over 3 months"
            return "next-12m EPS consensus flat to up over 3 months"
        if name == "revenue_revision_3m_pct" and v is not None:
            return "revenue consensus cut sharply" if v <= -10 else ("revenue consensus trimmed" if v < 0 else "revenue consensus holding")
        if name == "operating_margin_change_yoy_pp" and v is not None:
            return "operating margin compressing year on year" if v <= -1 else "operating margin broadly stable year on year"
        if name == "last_eps_surprise_pct" and v is not None:
            return "last quarter beat EPS consensus" if v > 0 else "last quarter missed EPS consensus"
        if name == "net_debt_usd_bn" and v is not None:
            return "net cash balance sheet" if v < 0 else "net debt position"
        if name == "short_interest_change_1m_pct" and v is not None:
            return "short interest rising over the last month" if v > 0 else "short interest falling over the last month"
        if name == "days_to_next_earnings" and v is not None:
            return "calendar days until the next earnings report"
        return "context"

    def _quotes(self, signals: list[NarrativeSignal], scores: DislocationScores, documents: list[Document]) -> list[QuoteEvidence]:
        kinds = {d.doc_id: d.kind for d in documents}
        used: set[str] = set()
        out: list[QuoteEvidence] = []

        def by_tag(tag: str) -> list[NarrativeSignal]:
            hits = [s for s in signals if s.tag == tag]
            # management transcript sentences first, then document order
            return sorted(hits, key=lambda s: 0 if kinds.get(s.doc_id) == DocumentKind.TRANSCRIPT else 1)

        def take(tag: str, prefix: str = "") -> bool:
            for s in by_tag(tag):
                text = _quote_text(s.sentence)
                key = normalize_text(text or "")
                if text is None or key in used:
                    continue
                used.add(key)
                kind = kinds.get(s.doc_id)
                where = (kind.value if kind else "document") + (f", {s.speaker}" if s.speaker else "")
                desc = TAG_DESCRIPTIONS.get(tag, tag.replace("_", " "))
                out.append(QuoteEvidence(doc_id=s.doc_id, speaker=s.speaker, quote=text, interpretation=f"{prefix}{desc} ({where}; tag {tag})"))
                return True
            return False

        winner = scores.dislocation_type
        weights = NARRATIVE_WEIGHTS.get(winner, {})
        ranked = sorted((t for t in weights if scores.tag_counts.get(t)), key=lambda t: (-weights[t], -scores.tag_counts[t], t))
        if winner == IE:  # show the strongest signals of any kind
            ranked = sorted(scores.tag_counts, key=lambda t: (-scores.tag_counts[t], t))
        for tag in ranked:
            if len(out) >= _MAX_SUPPORT_QUOTES:
                break
            take(tag)
        # one item that cuts against the conclusion, when the documents contain one
        if winner == VT:
            against = [t for t in sorted(NARRATIVE_WEIGHTS[TS], key=lambda t: -NARRATIVE_WEIGHTS[TS][t]) if scores.tag_counts.get(t)]
            against += [t for t in ("guidance_conservative", "results_beat", "limited_exposure") if scores.tag_counts.get(t)]
        elif winner == IE:
            against = []
        else:
            against = [t for t in sorted(NARRATIVE_WEIGHTS[VT], key=lambda t: -NARRATIVE_WEIGHTS[VT][t]) if scores.tag_counts.get(t)]
        for tag in against:
            if take(tag, prefix="Cuts against the thesis: "):
                break
        return out

    # -- text --------------------------------------------------------------------------------

    def _thesis(
        self,
        candidate: RankedCandidate,
        features: Mapping[str, Any],
        shown: Mapping[str, Any],
        documents: list[Document],
        signals: list[NarrativeSignal],
        scores: DislocationScores,
        spec: ScreenSpec,
        sector_context: Mapping[str, Any],
    ) -> DislocationThesis:
        t = scores.dislocation_type
        ticker, name = candidate.ticker, candidate.name or candidate.ticker
        who = f"{name} ({ticker})" if name != ticker else ticker

        def v(feature: str) -> str | None:
            x = shown.get(feature)
            return display_value(x)[0] if isinstance(x, float) and math.isfinite(x) else None

        dd, rsi, fcf = v("drawdown_from_52w_high_pct"), v("rsi_14"), v("fcf_yield_pct")
        g_ttm, g_q = v("revenue_growth_yoy_pct"), v("revenue_growth_last_q_yoy_pct")
        eps_rev, surprise = v("eps_revision_3m_pct"), v("last_eps_surprise_pct")
        si, si_chg, dtc, nde = v("short_interest_pct_float"), v("short_interest_change_1m_pct"), v("days_to_cover"), v("days_to_next_earnings")
        tags = scores.tag_counts

        def tag_list(type_: str) -> str:
            present = [x for x in NARRATIVE_WEIGHTS.get(type_, {}) if tags.get(x)]
            return ", ".join(f"{x} x{tags[x]}" for x in present) or "none"

        # Prose quotes numbers only as the feature table shows them.
        price_txt = f"a {dd}% drawdown from the 52-week high" if dd else "a sell-off"
        if rsi:
            price_txt += f" and an RSI of {rsi}"
        si_txt = ""
        if si:
            si_txt = f", and short interest at {si}% of float" + (f" ({si_chg}% change over one month)" if si_chg else "")

        beliefs = {
            TS: "the problem behind the last report to persist",
            VT: "a lasting deterioration in the business",
            GR: "the lowered outlook to be the new run-rate",
            SC: "the industry headwind to hit this company as hard as its peers",
            TF: "a fundamental problem that the reported numbers do not show",
            IE: "a deterioration that the evidence provided neither confirms nor rules out",
        }
        market_narrative = f"With {price_txt}{si_txt}, the price implies the market expects {beliefs[t]}."

        growth_txt = ""
        if g_q:
            growth_txt = f"latest-quarter revenue growth of {g_q}%" + (f" ({g_ttm}% trailing)" if g_ttm else "")
        revision_txt = f"a {eps_rev}% revision to NTM EPS consensus over 3 months" if eps_rev else ""
        beat = (_num(features, "last_eps_surprise_pct") or 0.0) > 0

        if t == VT:
            bits = " and ".join(x for x in (growth_txt, revision_txt) if x) or "deteriorating forward indicators"
            lead = f"its {fcf}% trailing FCF yield sits on " if fcf else "its trailing metrics sit on "
            headline = f"{who} looks like a value trap: {lead}{bits}."
            variant = (
                "The evidence agrees with the market. Bearish quant flags: "
                + (", ".join(scores.flags[VT]) or "none")
                + f"; bearish narrative signals: {tag_list(VT)}."
            )
            why = (
                "Trailing metrics lag: the screen sees a high trailing FCF yield and TTM growth, while the latest quarter, "
                "estimate revisions and management commentary point down. The low price reflects falling forward numbers, "
                "so the gap is unlikely to close until revisions stop."
            )
        elif t == TS:
            headline = (
                f"{who} looks like a transitory shock: the shares show {price_txt}, yet management frames the hit as temporary"
                + (f" and {growth_txt} is still positive." if growth_txt and (_num(features, "revenue_growth_last_q_yoy_pct") or 0) > 0 else ".")
            )
            variant = (
                "The evidence points to a one-off rather than a broken business: narrative signals "
                + tag_list(TS)
                + (f"; {revision_txt} is a trim, not a collapse" if revision_txt and "eps_revision_3m_pct" in scores.flags[TS] else "")
                + "."
            )
            why = (
                "Holders of a former uptrend sold the miss on heavy volume and trailing-growth screens extrapolated it. "
                "If the next report shows the one-off reversing, the gap can close; if the issue recurs, the market was right."
            )
        elif t == GR:
            headline = (
                f"{who} looks like a guidance-reset overreaction: "
                + (f"the quarter beat consensus (EPS surprise {surprise}%), but " if beat and surprise else "")
                + f"management guided conservatively and the shares show {price_txt}."
            )
            variant = (
                "Management describes the outlook as deliberately conservative: narrative signals "
                + tag_list(GR)
                + ("; the company has net cash" if "net_debt_usd_bn" in scores.flags[GR] else "")
                + "."
            )
            why = (
                "The market priced the lower outlook as a run-rate cut. If quarterly results land at or above the "
                "conservative guide, estimates stabilise and the discount can unwind."
            )
        elif t == SC:
            held = revision_txt and "eps_revision_3m_pct" in scores.flags[SC]
            headline = (
                f"{who} looks like sector contagion: it sold off with its industry"
                + (f" while estimates held ({revision_txt})" if held else "")
                + " and management describes limited exposure."
            )
            variant = (
                "The company-specific evidence is better than the group move implies: narrative signals "
                + tag_list(SC)
                + "."
            )
            why = (
                "Investors sold the industry on a peer or macro headline without separating exposures. The gap can close "
                "as company results show limited impact, or widen if the headwind spreads."
            )
        elif t == TF:
            headline = (
                f"{who} looks flow-driven: short interest of {si}% of float"
                + (f" and {dtc} days to cover" if dtc else "")
                + " against intact reported fundamentals."
            )
            variant = "The documents give no fundamental explanation, while positioning is crowded on the short side."
            why = "Crowded short positioning can push the price below fundamentals and reverse sharply on short covering."
        else:
            strongest = sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))
            headline = f"{who}: the evidence provided does not explain the sell-off ({price_txt})."
            variant = (
                "Neither the quantitative flags nor the narrative signals reach the thresholds for a call; "
                f"strongest narrative signals: {', '.join(f'{k} x{c}' for k, c in strongest[:4]) or 'none'}."
            )
            why = "Cannot tell from the inputs; the next report or a transcript covering the sell-off would decide it."

        catalysts: list[str] = []
        if nde:
            catalysts.append(f"Next earnings report in about {nde} days (days_to_next_earnings).")
        if t == TS:
            catalysts.append("Next quarter's results showing the one-off item reversing, as management said it would.")
        elif t == GR:
            catalysts.append("Quarterly delivery at or above the conservative guidance.")
        elif t == SC:
            catalysts.append("Industry news stabilising and company results confirming limited exposure.")
        elif t == TF:
            catalysts.append("Short covering after a neutral or positive data point.")
        elif t == VT:
            catalysts.append("None identified for a re-rating: estimate revisions would first need to stop falling.")
        if tags.get("capital_return") and t != VT:
            catalysts.append("Share repurchases or dividends that management cited.")

        risks: list[str] = []
        bear = [x for x in NARRATIVE_WEIGHTS[VT] if tags.get(x)]
        if t != VT and bear:
            risks.append("Bearish signals in the documents: " + ", ".join(f"{x} x{tags[x]}" for x in bear) + ".")
        if si and t != TF:
            risks.append(f"Short interest at {si}% of float can extend the decline if the next print disappoints.")
        if t == VT:
            risks.append("Low expectations and a crowded short can cause sharp rallies even if the decline is structural.")
        if t in (TS, GR, SC):
            risks.append("The issue management calls temporary or limited proves larger or recurring.")
        if not risks:
            risks.append("Evidence is thin; unidentified fundamental problems may explain the sell-off.")

        invalidation: list[str] = []
        if t in (TS, GR, SC, TF):
            invalidation.append("revenue_growth_last_q_yoy_pct turns negative in the next report.")
            invalidation.append("eps_revision_3m_pct falls to -15 or below.")
            invalidation.append("Management lowers or withdraws guidance, or stops describing the issue as temporary.")
        if t == VT:
            invalidation.append(
                "Latest-quarter revenue growth returns to the trailing rate"
                + (f" ({g_ttm}%)" if g_ttm else "")
                + " with eps_revision_3m_pct turning positive."
            )
            invalidation.append("Management quantifies the problem and shows it reversing (margins and orders stabilising).")
        if dd:
            invalidation.append(f"Price makes new lows below the current drawdown of {dd}% while short interest keeps rising.")
        if t == IE:
            invalidation.append("A transcript or filing explaining the sell-off becomes available and settles the question.")

        gaps = [HEURISTIC_NOTE]
        if not documents:
            gaps.append("No documents were available, so there is no narrative evidence.")
        elif not any(d.kind == DocumentKind.TRANSCRIPT for d in documents):
            gaps.append("No earnings-call transcript was available in the lookback window.")
        if documents and not signals:
            gaps.append("The narrative tagger found no signals in the documents provided.")
        missing = [k for k, x in shown.items() if x is None]
        if missing:
            listed = ", ".join(missing[:_MAX_LISTED_GAPS]) + (f" (+{len(missing) - _MAX_LISTED_GAPS} more)" if len(missing) > _MAX_LISTED_GAPS else "")
            gaps.append(f"Missing features: {listed}.")
        gaps.append("Decision trace: " + "; ".join(scores.reasons))

        return DislocationThesis(
            ticker=ticker,
            headline=headline,
            dislocation_type=t,
            market_narrative=market_narrative,
            variant_view=variant,
            why_dislocation_exists=why,
            quant_evidence=self._quant_evidence(features, shown, spec, scores, sector_context),
            narrative_evidence=self._quotes(signals, scores, documents),
            catalysts=catalysts,
            risks=risks,
            invalidation_triggers=invalidation,
            conviction=scores.conviction,  # type: ignore[arg-type]
            is_actionable=t in _GENUINE and scores.conviction != "low",
            data_gaps=gaps,
        )
