"""The research loop: propose hypotheses, let experience judge them, test, learn, repeat - then give the
survivors one exam on data the loop never saw.

Per round:

1. a generator proposes candidate hypotheses - :class:`LLMGenerator` (Claude reads the experience
   digest and the previous round's feedback and writes new claims with mechanisms, which the strategy
   translator turns into testable specs) or :class:`HeuristicGenerator` (offline: the idea library
   plus horizon / direction variants of return signals);
2. the judge assesses each candidate against the ledger (prediction, related experience, critique);
3. the loop tests the ``per_round`` candidates with the highest upper confidence bound (mean + sd):
   promising OR still very uncertain ideas first, ideas that contradict experience last - a
   Bayesian-optimisation search over hypothesis space;
4. each test runs on the RESEARCH window only, after its forecast was pre-registered; the result is
   recorded, the beliefs update, and "predicted X, observed Y, belief now Z" plus the judge's
   critique become the next round's feedback.

After the rounds, the best ``holdout_top`` hypotheses (by posterior mean, if positive) are tested
once on the HOLDOUT window (the last ``holdout_years`` before ``end``), which no step of the loop
looked at. That evidence is out of sample - unless this holdout window was used before, in which case
the ledger knows (it counts holdout uses) and the report says the holdout is burnt.

Searching many hypotheses on one data set is data mining; what keeps the loop honest is that every
trial is in the ledger (the posterior conditions on the failures too), related trials share evidence,
the forecasts are pre-registered, and the final word goes to data the search did not touch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from pydantic import BaseModel, Field

from aitrading.beliefs.digest import experience_digest, lessons
from aitrading.beliefs.grounding import sanitize_llm_text
from aitrading.beliefs.judge import BayesianJudge, Judge
from aitrading.beliefs.learner import Learner
from aitrading.beliefs.ledger import BeliefLedger, CutoffLedger
from aitrading.beliefs.models import Assessment, Belief, BeliefUpdate
from aitrading.beliefs.semantics import FEATURE_SEMANTICS, hypothesis_id, hypothesis_shape, similarity
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.strategy.library import TEMPLATES
from aitrading.strategy.spec import PortfolioConstruction, SignalComponent, StrategySpec

__all__ = [
    "Proposal",
    "HeuristicGenerator",
    "LLMGenerator",
    "HypothesisProposal",
    "ProposalBatch",
    "ResearchLoop",
    "LoopConfig",
    "LoopReport",
    "TestRecord",
    "candidate_pool",
]


@dataclass
class Proposal:
    spec: StrategySpec
    statement: str | None = None
    mechanism: str = ""
    source: str = "heuristic"


# ------------------------------------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------------------------------------

_RETURN_SIGNALS = [("return_3d_pct", "3d"), ("return_5d_pct", "5d"), ("return_10d_pct", "10d"), ("return_1m_pct", "1m"),
                   ("return_3m_pct", "3m"), ("return_6m_pct", "6m"), ("return_12m_ex_1m_pct", "12_1")]
_THEMES = {
    "past_return": r"momentum|reversal|revert|winner|loser|past return|short[\s-]*term|trend",
    "value": r"value|cheap|valuation|yield|book",
    "profitability": r"quality|profit|margin|roe",
    "risk": r"volatil|risk|beta",
    "size": r"size|small[\s-]*cap",
    "investment": r"invest|asset growth",
    "earnings_momentum": r"revision|surprise|earnings momentum",
    "short_interest": r"short interest|shorted",
}


def _theme_families(theme: str | None) -> set[str] | None:
    import re

    if not theme or not theme.strip():
        return None
    t = theme.lower()
    fams = {fam for fam, pat in _THEMES.items() if re.search(pat, t)}
    return fams or None


def _spec_families(spec: StrategySpec) -> set[str]:
    return {FEATURE_SEMANTICS[c.feature].family for c in spec.signal if c.feature in FEATURE_SEMANTICS}


def candidate_pool(theme: str | None = None, *, catalog: FeatureCatalog | None = None) -> list[Proposal]:
    """Offline candidates: momentum AND reversal at every catalog return horizon (weekly rebalancing
    up to two weeks, monthly beyond), plus the cross-sectional library templates that free data can
    test. ``theme`` keeps the families it mentions (all when it names none)."""
    cat = catalog or default_catalog()
    out: list[Proposal] = []
    for feat, label in _RETURN_SIGNALS:
        if feat not in cat:
            continue
        short = FEATURE_SEMANTICS[feat].horizon is not None and FEATURE_SEMANTICS[feat].horizon <= 10
        for direction, word in (("higher_is_better", "momentum"), ("lower_is_better", "reversal")):
            spec = StrategySpec(
                name=f"{word}_{label}",
                idea=f"{label.replace('_', '-')} {word}",
                kind="cross_sectional",
                rebalance="weekly" if short else "monthly",
                signal=[SignalComponent(feature=feat, direction=direction)],  # type: ignore[arg-type]
                portfolio=PortfolioConstruction(style="long_short", selection="quantile", n_quantiles=5),
                assumptions=["Generated by the research loop's offline generator (a return-horizon variant)."],
            )
            out.append(Proposal(spec, source="heuristic"))
    seen = {hypothesis_id(p.spec) for p in out}
    for key, t in TEMPLATES.items():
        if t.needs_institutional_data:
            continue
        spec = t.spec()
        if spec.kind != "cross_sectional" or hypothesis_id(spec) in seen:
            continue
        out.append(Proposal(spec, source="heuristic:template"))
        seen.add(hypothesis_id(spec))
    fams = _theme_families(theme)
    if fams is not None:
        out = [p for p in out if _spec_families(p.spec) & fams]
    return out


class HeuristicGenerator:
    """Offline generator: the candidate pool; the loop's acquisition rule does the choosing."""

    name = "heuristic"

    def __init__(self, catalog: FeatureCatalog | None = None):
        self.catalog = catalog

    def propose(self, theme: str | None, digest: str, feedback: list[str], n: int) -> list[Proposal]:
        return candidate_pool(theme, catalog=self.catalog)


class HypothesisProposal(BaseModel):
    statement: str = Field(description="The claim in one sentence: which stocks outperform which, over what horizon.")
    mechanism: str = Field(description="Why it could work: who is on the other side and why the effect is not arbitraged away.")
    testable_idea: str = Field(description="The strategy to backtest, in the platform's idea language: signal (catalog "
                                           "feature names), direction, long-only or long-short, quantiles, rebalance.")
    uses_experience: str = Field(description="How the experience log shaped this proposal (or why it is exploration).")
    expected_sharpe: float = Field(description="Your expected net Sharpe ratio.")


class ProposalBatch(BaseModel):
    proposals: list[HypothesisProposal]


_GEN_SYSTEM = """\
You are a quantitative researcher who learns from experience. Each round you propose new, falsifiable \
hypotheses about the cross-section of US stock returns that the platform can backtest with the feature catalog \
below. You are given the experience log (every hypothesis tested so far in this market, with the belief a Bayesian \
engine derived from all results, failures included), lessons per concept, and feedback from the last round \
(what was predicted, what was observed).

How to use experience:
- Do not re-propose a refuted claim or a near-copy of it unless you state a concrete reason the outcome should differ.
- Exploit what worked by probing NEARBY but untested variants (another horizon, another measure of the same concept, \
a filter that should strengthen the mechanism) - and explain the mechanism, not just the pattern.
- Explore: at least one proposal per round should test a concept the log has not covered.
- Remember costs: short-horizon signals turn over a lot (one-way costs are charged on every trade).
- Never tune thresholds or lookbacks to the numbers in the log; a hypothesis is a claim about the world.

The testable_idea must be expressible with these catalog features (use their exact names) as a cross-sectional \
ranking signal (higher_is_better / lower_is_better) with optional filters; say long-only or long-short, quintiles \
or deciles and the rebalance frequency (weekly for signals of two weeks or less).

# Feature catalog
{catalog}
"""


class LLMGenerator:
    """Claude proposes hypotheses from the experience digest; ``translator`` makes them testable."""

    def __init__(self, llm: Any, translator: Any, *, effort: str = "high", catalog: FeatureCatalog | None = None,
                 on_error: Callable[[str], None] | None = None):
        self.llm = llm
        self.translator = translator
        self.effort = effort
        self.catalog = catalog or default_catalog()
        self.on_error = on_error
        self._system = _GEN_SYSTEM.replace("{catalog}", self.catalog.to_prompt())

    @property
    def name(self) -> str:
        return f"llm:{getattr(self.llm, 'name', 'llm')}"

    def user_prompt(self, theme: str | None, digest: str, feedback: list[str], n: int) -> str:
        fb = "\n".join(feedback) if feedback else "(first round: no feedback yet)"
        return (
            (f"<theme>\n{theme}\n</theme>\n" if theme else "<theme>\nopen: any cross-sectional equity anomaly\n</theme>\n")
            + f"<experience_log>\n{digest}\n</experience_log>\n"
            f"<last_round_feedback>\n{fb}\n</last_round_feedback>\n\n"
            f"Propose {n} hypotheses."
        )

    def propose(self, theme: str | None, digest: str, feedback: list[str], n: int) -> list[Proposal]:
        batch = self.llm.structured(purpose="hypothesis_generator", system=self._system,
                                    user=self.user_prompt(theme, digest, feedback, n), output_model=ProposalBatch,
                                    effort=self.effort)
        if not isinstance(batch, ProposalBatch):
            batch = ProposalBatch.model_validate(batch)
        out: list[Proposal] = []
        for p in batch.proposals[:n]:
            try:
                spec = self.translator.translate(p.testable_idea).spec
            except Exception as exc:  # noqa: BLE001 - one untranslatable idea must not end the round
                if self.on_error:
                    self.on_error(f"could not make '{p.statement[:80]}' testable: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            statement, s_issues = sanitize_llm_text(p.statement, max_chars=300)
            mechanism, m_issues = sanitize_llm_text(p.mechanism, max_chars=500, mask_numbers=True)
            if (s_issues or m_issues) and self.on_error:
                self.on_error(f"sanitised the proposal text ({', '.join([*s_issues, *m_issues])})")
            out.append(Proposal(spec, statement=statement or None, mechanism=mechanism, source=self.name))
        return out


# ------------------------------------------------------------------------------------------------
# Loop
# ------------------------------------------------------------------------------------------------


#: Evidence of a hypothesis at least this similar that overlaps the holdout window contaminates it.
CONTAMINATION_SIMILARITY = 0.5


@dataclass
class LoopConfig:
    rounds: int = 3
    per_round: int = 3
    candidates_per_round: int = 6
    holdout_years: float = 2.0
    research_years: float | None = 8.0  # None = as much history as the data allow
    holdout_top: int = 2
    end: date | None = None  # last date of the data (default: today)


@dataclass
class TestRecord:
    round: int
    hypothesis_id: str
    statement: str
    spec_name: str
    before: Assessment
    window: tuple[date | None, date]
    kind: str = "backtest"
    observed_sharpe: float | None = None
    update: BeliefUpdate | None = None
    run_id: str | None = None
    error: str | None = None


@dataclass
class LoopReport:
    market: str
    theme: str | None
    generator: str
    judge: str
    research_window: tuple[date | None, date]
    holdout_window: tuple[date, date]
    holdout_uses_before: int
    tests: list[TestRecord] = field(default_factory=list)
    holdout: list[TestRecord] = field(default_factory=list)
    final: list[Belief] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def rec(t: TestRecord) -> dict[str, Any]:
            return {
                "round": t.round, "hypothesis_id": t.hypothesis_id, "statement": t.statement, "spec_name": t.spec_name,
                "kind": t.kind, "window": [_iso(t.window[0]), _iso(t.window[1])], "run_id": t.run_id,
                "predicted": {"mean": t.before.mean, "sd": t.before.sd, "p_positive": t.before.p_positive,
                              "verdict": t.before.verdict, "judge": t.before.judge},
                "observed_sharpe": t.observed_sharpe, "surprise_z": t.update.surprise_z if t.update else None,
                "belief_after": t.update.after.model_dump(mode="json") if t.update else None, "error": t.error,
            }
        return {
            "market": self.market, "theme": self.theme, "generator": self.generator, "judge": self.judge,
            "research_window": [_iso(d) for d in self.research_window],
            "holdout_window": [d.isoformat() for d in self.holdout_window],
            "holdout_uses_before": self.holdout_uses_before,
            "tests": [rec(t) for t in self.tests], "holdout": [rec(t) for t in self.holdout],
            "final": [b.model_dump(mode="json") for b in self.final], "lessons": self.lessons,
            "skipped": self.skipped, "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        rw, hw = self.research_window, self.holdout_window
        lines = [
            f"# Research loop - market '{self.market}'" + (f", theme: {self.theme}" if self.theme else ""),
            "",
            f"Generator: {self.generator}; judge: {self.judge}. Research window {rw[0] or 'start of data'} to {rw[1]}; holdout {hw[0]} to "
            f"{hw[1]} (looked at {self.holdout_uses_before} time(s) before this run).",
            "",
            "## Tests on the research window (in-sample)",
            "",
            "| Round | Hypothesis | Predicted Sharpe | Verdict before | Observed | Surprise z | Belief after |",
            "|---:|---|---:|---|---:|---:|---|",
        ]
        for t in self.tests:
            lines.append(_row(t))
        if not self.tests:
            lines.append("| - | nothing was tested | | | | | |")
        lines += ["", "## Holdout exam (out-of-sample)", ""]
        if self.holdout:
            lines += ["| Hypothesis | Predicted Sharpe | Observed | Surprise z | Belief after |", "|---|---:|---:|---:|---|"]
            for t in self.holdout:
                lines.append(_row(t, with_round=False))
        else:
            lines.append("No hypothesis had a positive posterior after the research rounds, so none sat the holdout exam.")
        lines += ["", "## Beliefs now", "", "| Hypothesis | Belief (net Sharpe) | P(>0) | Runs | Out-of-sample | Status |",
                  "|---|---:|---:|---:|---|---|"]
        for b in self.final:
            oos = f"{b.oos_sharpe:+.2f} over {b.oos_years:.1f}y" if b.oos_sharpe is not None else "none"
            lines.append(f"| [{b.hypothesis_id}] {_short(b.statement)} | {b.mean:+.2f} ± {b.sd:.2f} | {b.p_positive:.0%} | "
                         f"{b.n_evidence} | {oos} | {b.status} |")
        if self.lessons:
            lines += ["", "## Lessons", ""] + [f"- {x}" for x in self.lessons]
        if self.warnings or self.skipped:
            lines += ["", "## Notes", ""] + [f"- {x}" for x in [*self.warnings, *self.skipped]]
        lines += ["", "Simulated research on historical data. Not investment advice."]
        return "\n".join(lines) + "\n"


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def _short(s: str, n: int = 90) -> str:
    s = s.replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def _row(t: TestRecord, *, with_round: bool = True) -> str:
    b = t.before
    obs = f"{t.observed_sharpe:+.2f}" if t.observed_sharpe is not None else (t.error or "n/a")
    z = f"{t.update.surprise_z:+.1f}" if t.update else ""
    after = (f"{t.update.after.mean:+.2f} ± {t.update.after.sd:.2f} ({t.update.after.status})" if t.update else "")
    head = f"| {t.round} " if with_round else ""
    verdict = f" | {b.verdict.replace('_', ' ')}" if with_round else ""
    return (f"{head}| [{t.hypothesis_id}] {_short(t.statement)} | {b.mean:+.2f} ± {b.sd:.2f}{verdict} | {_short(obs, 60)} | "
            f"{z} | {after} |")


class ResearchLoop:
    """See the module docstring.

    Args:
        runner: a :class:`~aitrading.backtest.protocols.BacktestRunner`.
        ledger: the market's :class:`BeliefLedger` (everything is recorded there).
        judge: assesses candidates (default: :class:`BayesianJudge`).
        generator: :class:`HeuristicGenerator` (default) or :class:`LLMGenerator`.
        config: :class:`LoopConfig`.
        progress: callback for one-line progress messages.
    """

    def __init__(self, runner: Any, ledger: BeliefLedger, *, judge: Judge | None = None, generator: Any = None,
                 config: LoopConfig | None = None, progress: Callable[[str], None] | None = None):
        self.runner = runner
        self.ledger = ledger
        self.judge = judge if judge is not None else BayesianJudge(ledger)
        self.generator = generator if generator is not None else HeuristicGenerator()
        self.config = config or LoopConfig()
        self._say = progress or (lambda msg: None)

    def research_view(self) -> CutoffLedger:
        """The ledger as the research rounds see it: no evidence whose data reach the holdout window."""
        return CutoffLedger(self.ledger, self.windows()[1][0])

    def _bound_judge(self, view: Any) -> Any:
        bind = getattr(self.judge, "bind", None)
        return bind(view) if callable(bind) else self.judge

    def windows(self) -> tuple[tuple[date | None, date], tuple[date, date]]:
        """((research start or None = start of data, research end), (holdout start, end))."""
        c = self.config
        end = c.end or date.today()
        h_start = end - timedelta(days=round(c.holdout_years * 365.25))
        r_end = h_start - timedelta(days=1)
        r_start = (r_end - timedelta(days=round(c.research_years * 365.25))) if c.research_years else None
        return (r_start, r_end), (h_start, end)

    def _test(self, learner: Learner, rnd: int, spec: StrategySpec, statement: str | None, mechanism: str, source: str,
              window: tuple[date | None, date], kind: str, oos: bool | None,
              assessment: Assessment | None = None, text_horizon: date | None = None) -> TestRecord:
        s = spec.model_copy(update={"start": window[0], "end": window[1]})
        pre = learner.before(s, purpose=kind, statement=statement, mechanism=mechanism, source=source,  # type: ignore[arg-type]
                             assessment=assessment, text_horizon=text_horizon)
        rec = TestRecord(round=rnd, hypothesis_id=pre.assessment.hypothesis_id, statement=pre.assessment.statement,
                         spec_name=s.name, before=pre.assessment, window=window, kind=kind)
        try:
            result = self.runner.backtest(s)
        except Exception as exc:  # noqa: BLE001 - a failed backtest is reported, the loop goes on
            rec.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            return rec
        rec.run_id = result.run_id
        upd = learner.after(s, result, pre, kind=kind, out_of_sample=oos, statement=statement,  # type: ignore[arg-type]
                            mechanism=mechanism, source=source, text_horizon=text_horizon)
        if upd is None:
            rec.error = "no usable Sharpe ratio (too few return periods)"
        else:
            rec.update = upd
            rec.observed_sharpe = upd.observed_sharpe
        return rec

    def run(self, theme: str | None = None) -> LoopReport:
        c = self.config
        research, holdout = self.windows()
        uses_before = self.ledger.holdout_uses_overlapping(*holdout)
        # Every read during the research rounds goes through the holdout-blind view: digests, forecasts,
        # beliefs, feedback and the choice of finalists never see data from the holdout years, even when
        # ordinary backtests in this ledger already cover them.
        view = self.research_view()
        judge = self._bound_judge(view)
        learner = Learner(view, judge)
        report = LoopReport(market=self.ledger.market, theme=theme, generator=getattr(self.generator, "name", "?"),
                            judge=getattr(self.judge, "name", "?"), research_window=research, holdout_window=holdout,
                            holdout_uses_before=uses_before)
        if uses_before:
            report.warnings.append(
                f"The holdout window overlaps windows already used {uses_before} time(s): it is no longer unseen data, so "
                "its results are recorded as in-sample. Use a different --end or --holdout-years, or wait for new data "
                "(paper trading).")
        feedback: list[str] = []
        tested: dict[str, tuple[StrategySpec, str | None, str, str]] = {}
        for rnd in range(1, c.rounds + 1):
            model = BayesianJudge(view).model()
            digest = experience_digest(model)
            self._say(f"round {rnd}/{c.rounds}: generating hypotheses ...")
            try:
                proposals = self.generator.propose(theme, digest, feedback, c.candidates_per_round)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"round {rnd}: the generator failed ({type(exc).__name__}: {str(exc)[:200]})")
                break
            cands: list[tuple[float, Assessment, Proposal]] = []
            seen: set[str] = set()
            for p in proposals:
                spec = p.spec.model_copy(update={"start": research[0], "end": research[1]})
                hid = hypothesis_id(spec)
                if hid in seen or hid in tested:
                    continue
                seen.add(hid)
                if self._tested_on(hid, research):
                    report.skipped.append(f"round {rnd}: [{hid}] already tested on this research window (re-testing the "
                                          "same data teaches nothing)")
                    continue
                a = judge.assess(spec, statement=p.statement, mechanism=p.mechanism)
                cands.append((a.acquisition, a, p))
            if not cands:
                report.warnings.append(f"round {rnd}: no untested candidate left" + (f" for theme '{theme}'" if theme else ""))
                break
            feedback = []
            chosen = self._select(cands, c.per_round)
            for a, p in chosen:
                self._say(f"  testing [{a.hypothesis_id}] {a.statement[:80]} (predicted {a.mean:+.2f} ± {a.sd:.2f})")
                rec = self._test(learner, rnd, p.spec, p.statement, p.mechanism, p.source, research, "backtest", None,
                                 assessment=a, text_horizon=research[1])
                report.tests.append(rec)
                tested[rec.hypothesis_id] = (p.spec, p.statement, p.mechanism, p.source)
                feedback.append(self._feedback(rec))
            picked = {a.hypothesis_id for a, _ in chosen}
            for _, a, _p in cands:
                if a.hypothesis_id not in picked and a.verdict == "contradicts_experience":
                    feedback.append(f"Not tested (contradicts experience): [{a.hypothesis_id}] {a.statement} - predicted "
                                    f"{a.mean:+.2f} ± {a.sd:.2f}.")
        # holdout exam for the most promising hypotheses of this loop, chosen on research-view beliefs;
        # the forecast is the research view's, the update afterwards uses everything in the ledger
        model = BayesianJudge(view).model()
        ranked = sorted(((b.mean, b) for b in model.beliefs(list(tested))), key=lambda t: -t[0])
        finalists = [b for m, b in ranked if m > 0][: c.holdout_top]
        if finalists:
            contaminated = {b.hypothesis_id: self.contamination(tested[b.hypothesis_id][0], holdout) for b in finalists}
            self.ledger.note_holdout_use(*holdout)
            exam = Learner(self.ledger, judge)
            for b in finalists:
                spec, stmt, mech, src = tested[b.hypothesis_id]
                why = contaminated[b.hypothesis_id]
                if why and uses_before == 0:
                    report.warnings.append(f"[{b.hypothesis_id}] holdout result recorded as in-sample: {why}.")
                self._say(f"holdout exam: [{b.hypothesis_id}] {b.statement[:80]}")
                rec = self._test(exam, 0, spec, stmt, mech, src, holdout, "holdout", uses_before == 0 and why is None,
                                 text_horizon=research[1])
                report.holdout.append(rec)
        model = BayesianJudge(self.ledger).model()
        report.final = sorted(model.beliefs(list(tested)), key=lambda b: -b.mean)
        report.lessons = [x.text for x in lessons(model)]
        n_trials = len(self.ledger.tested_ids())
        report.warnings.append(f"{len(tested)} hypothesis(es) tested in this loop, {n_trials} in this market's ledger in "
                               "total: the more ideas are searched, the more an in-sample winner owes to luck.")
        return report

    @staticmethod
    def _select(cands: list[tuple[float, Assessment, Proposal]], k: int) -> list[tuple[Assessment, Proposal]]:
        """Greedy batch upper-confidence-bound selection: each pick discounts the uncertainty bonus of
        candidates similar to it (|similarity| to an already chosen one), so one round does not spend
        its tests on near-copies of one idea (or on an idea and its exact opposite)."""
        shapes = {a.hypothesis_id: hypothesis_shape(p.spec) for _, a, p in cands}
        chosen: list[tuple[Assessment, Proposal]] = []
        left = list(cands)
        while left and len(chosen) < k:
            def score(t: tuple[float, Assessment, Proposal]) -> tuple[float, str]:
                a = t[1]
                overlap = max((abs(similarity(shapes[a.hypothesis_id], shapes[c.hypothesis_id])) for c, _ in chosen),
                              default=0.0)
                return (-(a.mean + a.sd * (1.0 - overlap)), a.hypothesis_id)
            best = min(left, key=score)
            left.remove(best)
            chosen.append((best[1], best[2]))
        return chosen

    def contamination(self, spec: StrategySpec, holdout: tuple[date, date]) -> str | None:
        """Why the holdout window is not unseen data for ``spec``, or None. It is not unseen when the
        ledger already holds evidence that overlaps the window for this hypothesis or a closely related
        one (|similarity| >= ``CONTAMINATION_SIMILARITY``). Typical cause: an ordinary backtest run to
        the latest data. Whoever chose this loop's theme has seen how such ideas did in those years."""
        shape = hypothesis_shape(spec)
        for e in self.ledger.evidence():
            if e.end < holdout[0] or e.start > holdout[1]:
                continue
            h = self.ledger.hypothesis(e.hypothesis_id)
            if h is None:
                continue
            sim = similarity(shape, hypothesis_shape(h.spec))
            if abs(sim) >= CONTAMINATION_SIMILARITY:
                return (f"{e.kind} run {e.evidence_id.split(':', 1)[-1]} of a related hypothesis [{h.hypothesis_id}] "
                        f"(similarity {sim:+.2f}) already covers {max(e.start, holdout[0])} to {min(e.end, holdout[1])}")
        return None

    def _tested_on(self, hid: str, window: tuple[date | None, date]) -> bool:
        for e in self.ledger.evidence_for(hid):
            # a run's returns start / end within a rebalance period of the window it was asked for
            if e.kind == "backtest" and abs((e.end - window[1]).days) <= 35 and (
                    window[0] is None or abs((e.start - window[0]).days) <= 95):
                return True
        return False

    @staticmethod
    def _feedback(rec: TestRecord) -> str:
        b = rec.before
        if rec.update is None:
            return f"[{rec.hypothesis_id}] {rec.statement}: the test failed ({rec.error})."
        u = rec.update
        return (f"[{rec.hypothesis_id}] {rec.statement}: predicted {b.mean:+.2f} ± {b.sd:.2f} ({b.verdict.replace('_', ' ')}), "
                f"observed {u.observed_sharpe:+.2f} (surprise z {u.surprise_z:+.1f}); belief now {u.after.mean:+.2f} ± "
                f"{u.after.sd:.2f}, {u.after.status}.")
