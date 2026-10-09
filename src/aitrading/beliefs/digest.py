"""Experience as text: the ledger summarised for people and for language models.

* :func:`lessons` - family-level regularities ("momentum at horizons of 2 weeks or less: 3 variants
  tested, 0 supported, 2 refuted"), computed deterministically from the posterior. These are the
  "intuitions" the platform has earned in a market, each backed by numbers from the ledger.
* :func:`experience_digest` - the most relevant tested hypotheses (to a target idea, or overall)
  with their beliefs, plus the lessons. It is what the hypothesis generator and the LLM judge read,
  and the context of the judge-training dataset (:mod:`aitrading.beliefs.dataset`), so a model is
  trained on exactly what it sees in use.
"""

from __future__ import annotations

from dataclasses import dataclass

from aitrading.beliefs.bayes import BeliefModel
from aitrading.beliefs.grounding import render_safe
from aitrading.beliefs.models import Belief
from aitrading.beliefs.semantics import FAMILY_LABELS, HypothesisShape, similarity

__all__ = ["Lesson", "lessons", "experience_digest", "belief_line", "horizon_bucket"]


@dataclass(frozen=True)
class Lesson:
    concept: str
    title: str
    text: str
    n_tested: int
    n_supported: int
    n_refuted: int
    mean_sharpe: float
    hypothesis_ids: tuple[str, ...]


def horizon_bucket(h: float | None) -> str:
    if h is None:
        return ""
    if h <= 10:
        return "horizons of 2 weeks or less"
    if h <= 70:
        return "1-3 month horizons"
    return "6-12 month horizons"


def _concept_title(concept: str, horizon: float | None) -> str:
    fam, sign = concept[:-1], concept[-1:]
    label = FAMILY_LABELS.get(fam, (fam, f"inverse {fam}"))[0 if sign == "+" else 1]
    if fam.startswith("factor_model:"):
        label = f"{fam.split(':', 1)[1].upper()} factor model"
    elif fam.startswith("timing:"):
        base = fam.split(":", 1)[1]
        label = f"market timing on {base.replace('_', ' ')}" + (" (long when high)" if sign == "+" else " (long when low)")
    hb = horizon_bucket(horizon)
    return f"{label}, {hb}" if hb else label


def lessons(model: BeliefModel, *, min_tested: int = 1) -> list[Lesson]:
    """One lesson per (concept, horizon bucket) with at least ``min_tested`` tested hypotheses."""
    beliefs = {b.hypothesis_id: b for b in model.beliefs()}
    groups: dict[tuple[str, str], list[Belief]] = {}
    titles: dict[tuple[str, str], str] = {}
    for hid, b in beliefs.items():
        h = model.hypothesis(hid)
        if h is None:
            continue
        key = (h.concept, horizon_bucket(h.horizon_days))
        groups.setdefault(key, []).append(b)
        titles.setdefault(key, _concept_title(h.concept, h.horizon_days))
    out: list[Lesson] = []
    for key, bs in groups.items():
        if len(bs) < min_tested:
            continue
        sup = [b for b in bs if b.status == "supported"]
        ref = [b for b in bs if b.status == "refuted"]
        mean = sum(b.mean for b in bs) / len(bs)
        best = max(bs, key=lambda b: b.mean)
        worst = min(bs, key=lambda b: b.mean)
        if sup and not ref:
            tone = "has worked here"
        elif ref and not sup:
            tone = "has not worked here"
        elif sup and ref:
            tone = "works only in some variants"
        else:
            tone = "is unproven here"
        text = (f"{titles[key][0].upper()}{titles[key][1:]} {tone}: {len(bs)} variant(s) tested, {len(sup)} supported, "
                f"{len(ref)} refuted; average posterior Sharpe {mean:+.2f}")
        if len(bs) > 1:
            text += f" (best {best.mean:+.2f} [{best.hypothesis_id}], worst {worst.mean:+.2f} [{worst.hypothesis_id}])"
        out.append(Lesson(concept=key[0], title=titles[key], text=text + ".", n_tested=len(bs), n_supported=len(sup),
                          n_refuted=len(ref), mean_sharpe=mean, hypothesis_ids=tuple(b.hypothesis_id for b in bs)))
    out.sort(key=lambda les: (-les.n_tested, -abs(les.mean_sharpe), les.title))
    return out


def belief_line(b: Belief, *, similarity_to_target: float | None = None) -> str:
    sim = f" | similarity {similarity_to_target:+.2f}" if similarity_to_target is not None else ""
    oos = (f"out-of-sample {b.oos_sharpe:+.2f} over {b.oos_years:.1f}y" if b.oos_sharpe is not None
           else "no out-of-sample evidence yet")
    return (f"- [{b.hypothesis_id}] {render_safe(b.statement)}{sim} | belief: net Sharpe {b.mean:+.2f} ± {b.sd:.2f} "
            f"(gross {b.gross_mean:+.2f}, costs {b.cost:.2f}), "
            f"P(>0) = {b.p_positive:.0%}, {b.status}; {b.n_evidence} run(s) over {b.years:.1f}y; {oos}")


def experience_digest(model: BeliefModel, *, target: HypothesisShape | None = None, target_id: str | None = None,
                      max_items: int = 12, max_lessons: int = 8) -> str:
    """Text summary of what the ledger has learned (most relevant to ``target`` when given; the
    target's own earlier runs, when ``target_id`` has some, are listed first and labelled)."""
    beliefs = model.beliefs()
    hyp = model.hyper
    head = (f"Experience log for market '{model.market}': {len(beliefs)} tested hypothesis(es), "
            f"{len(model.view.evidence())} run(s). Learned prior for a new idea: gross Sharpe (before costs) {hyp.m0:+.2f} ± {hyp.s0:.2f} "
            f"({'learned from the log' if hyp.source == 'empirical_bayes' else 'default, no experience yet'}).")
    if not beliefs:
        return head + "\nNo hypothesis has been tested in this market yet."
    lines = [head]
    if target is not None:
        own = [b for b in beliefs if b.hypothesis_id == target_id]
        if own:
            lines.append("This hypothesis has been tested before:")
            lines.append(belief_line(own[0]))
        scored = []
        for b in beliefs:
            if b.hypothesis_id == target_id:
                continue
            sim = similarity(target, model.shape(b.hypothesis_id))
            scored.append((abs(sim), sim, b))
        scored.sort(key=lambda t: (-t[0], -abs(t[2].mean) / max(t[2].sd, 1e-6)))
        rel = [t for t in scored if t[0] >= 0.05][:max_items]
        if rel:
            lines.append("Most related tested hypotheses (similarity 1 = same claim, -1 = the opposite claim):")
            lines += [belief_line(b, similarity_to_target=sim) for _, sim, b in rel]
        elif not own:
            lines.append("No tested hypothesis is related to this one: the learned prior is all the experience there is.")
    else:
        beliefs.sort(key=lambda b: (-abs(b.mean) / max(b.sd, 1e-6), b.hypothesis_id))
        lines.append("Tested hypotheses with the strongest evidence:")
        lines += [belief_line(b) for b in beliefs[:max_items]]
    les = lessons(model)[:max_lessons]
    if les:
        lines.append("Lessons (per concept):")
        lines += [f"- {x.text}" for x in les]
    return "\n".join(lines)
