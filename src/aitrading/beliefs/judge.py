"""Judges: an "intuition" about a hypothesis before it is tested, learned from the ledger.

* :class:`BayesianJudge` (default, deterministic) - the belief engine's posterior for the idea, the
  closest tested hypotheses and a written critique the research agent reads before it spends a
  backtest. No language model is involved, so it cannot be fooled by persuasive wording and its
  numbers are exact functions of the evidence.
* :class:`LLMJudge` - a language model (Claude by default) reads the idea, its mechanism and the
  experience digest, gives (1) a *literature prior* from economic reasoning and published evidence,
  formed before looking at the log, and (2) an overall view that uses the log. The literature
  prior enters the belief engine as one weak pseudo-observation (sd >= ``LLM_PRIOR_MIN_SD``), so the
  data always dominates; hypothesis ids it cites must exist in the digest and numbers in its
  reasoning that are not in the digest are flagged. This is the "prompt-only" way to make a model
  learn from backtests: in-context, through the retrieved ledger, with no training.
* :class:`OpenAICompatibleJudge` - a judge model you trained yourself (see
  :mod:`aitrading.beliefs.dataset` and ``scripts/train_judge_grpo.py``), served behind any
  OpenAI-compatible chat endpoint (vLLM, llama.cpp, Ollama ...). It sees exactly the prompt format
  it was trained on and answers with P(Sharpe > 0).

Every judge returns an :class:`~aitrading.beliefs.models.Assessment`. Record it with
:meth:`BeliefLedger.add_prediction` BEFORE running the test it predicts: the calibration report
(:mod:`aitrading.beliefs.calibration`) then measures which judge's intuition is actually any good.
"""

from __future__ import annotations

import json
import math
import os
import re
from statistics import NormalDist
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aitrading.beliefs.bayes import DEFAULT_HURDLE, LLM_PRIOR_MIN_SD, BeliefModel, normal_cdf
from aitrading.beliefs.digest import experience_digest
from aitrading.beliefs.grounding import HYP_ID_RE, check_text
from aitrading.beliefs.ledger import BeliefLedger, LedgerView, as_view
from aitrading.beliefs.models import Assessment, LLMPrior, Neighbor
from aitrading.beliefs.semantics import hypothesis_id, hypothesis_shape, hypothesis_statement, similarity, spec_from_any
from aitrading.llm.base import StructuredLLM

__all__ = [
    "Judge",
    "BayesianJudge",
    "LLMJudge",
    "LLMView",
    "OpenAICompatibleJudge",
    "RELATED_SIMILARITY",
]

#: Hypotheses at least this similar (in absolute value) count as related experience.
RELATED_SIMILARITY = 0.3
_ND = NormalDist()


class Judge(Protocol):
    name: str

    def assess(self, spec: Any, *, statement: str | None = None, mechanism: str = "") -> Assessment:
        ...

    def bind(self, ledger: Any) -> "Judge":
        """The same judge reading another ledger / view."""
        ...


def _fmt_p(p: float) -> str:
    return f"{p:.0%}"


def _verdict(novel: bool, p_positive: float) -> str:
    if novel:
        return "novel"
    if p_positive >= 0.6:
        return "consistent_with_experience"
    if p_positive <= 0.4:
        return "contradicts_experience"
    return "uncertain"


class BayesianJudge:
    """The belief engine as a judge (see the module docstring)."""

    name = "bayes"

    def __init__(self, ledger: BeliefLedger | LedgerView, *, hurdle: float = DEFAULT_HURDLE,
                 include_llm_priors: bool = True, max_neighbors: int = 5):
        self.ledger = ledger
        self.hurdle = hurdle
        self.include_llm_priors = include_llm_priors
        self.max_neighbors = max_neighbors
        self._model: BeliefModel | None = None
        self._sig: tuple | None = None

    def bind(self, ledger: Any) -> "BayesianJudge":
        """The same judge reading ``ledger`` (e.g. the research loop's holdout-blind view)."""
        return BayesianJudge(ledger, hurdle=self.hurdle, include_llm_priors=self.include_llm_priors,
                             max_neighbors=self.max_neighbors)

    def _signature(self) -> tuple:
        view = as_view(self.ledger)
        ev = tuple(sorted((e.evidence_id, e.recorded_at.isoformat()) for e in view.evidence()))
        pri = tuple(sorted((h.hypothesis_id, h.llm_prior.made_at.isoformat()) for h in view.hypotheses()
                           if h.llm_prior is not None)) if self.include_llm_priors else ()
        return ev, pri

    def model(self) -> BeliefModel:
        sig = self._signature()
        if self._model is None or sig != self._sig:
            self._model = BeliefModel(self.ledger, hurdle=self.hurdle, include_llm_priors=self.include_llm_priors)
            self._sig = sig
        return self._model

    def assess(self, spec: Any, *, statement: str | None = None, mechanism: str = "") -> Assessment:
        s = spec_from_any(spec)
        hid = hypothesis_id(s)
        model = self.model()
        shape = hypothesis_shape(s)
        m, sd, gross, cost = model.predict_net([shape], [s.costs_bps])
        mean, sd = float(m[0]), float(sd[0])
        stmt = (statement or "").strip()
        known = model.hypothesis(hid)
        if not stmt:
            stmt = known.statement if known is not None else hypothesis_statement(s)
        own = model.view.evidence_for(hid)
        tested = [x for x in model.tested() if x != hid]
        sims = [(similarity(shape, model.shape(x)), x) for x in tested]
        related = [(sv, x) for sv, x in sims if abs(sv) >= RELATED_SIMILARITY]
        top = sorted([t for t in sims if abs(t[0]) >= 0.05], key=lambda t: -abs(t[0]))[: self.max_neighbors]
        neighbors: list[Neighbor] = []
        if top:
            nb = model.beliefs([x for _, x in top])
            for (sv, _), b in zip(top, nb):
                neighbors.append(Neighbor(hypothesis_id=b.hypothesis_id, statement=b.statement, similarity=sv,
                                          mean=b.mean, sd=b.sd, p_positive=b.p_positive, n_evidence=b.n_evidence))
        p_pos = normal_cdf(mean / sd)
        verdict = _verdict(not own and not related, p_pos)
        a = Assessment(
            hypothesis_id=hid,
            statement=stmt,
            judge=self.name,
            mean=mean,
            sd=sd,
            p_positive=p_pos,
            p_hurdle=normal_cdf((mean - self.hurdle) / sd),
            hurdle=self.hurdle,
            verdict=verdict,  # type: ignore[arg-type]
            already_tested=len(own),
            related_tested=len(related),
            neighbors=neighbors,
            acquisition=mean + sd,
            details={"gross_mean": float(gross[0]), "cost": float(cost[0]), "costs_bps": s.costs_bps,
                     "prior_mean": model.hyper.m0, "prior_sd": model.hyper.s0, "prior_source": model.hyper.source},
        )
        a.critique = self._critique(a, model)
        return a

    @staticmethod
    def _critique(a: Assessment, model: BeliefModel) -> str:
        lines = [f"Prediction: Sharpe {a.mean:+.2f} ± {a.sd:.2f} net of costs (P(>0) = {_fmt_p(a.p_positive)}, "
                 f"P(>{a.hurdle:g}) = {_fmt_p(a.p_hurdle)}); gross {a.details.get('gross_mean', a.mean):+.2f} minus an "
                 f"estimated cost drag of {a.details.get('cost', 0.0):.2f} at {a.details.get('costs_bps', 0):g} bps."]
        if a.already_tested:
            lines.append(f"This exact hypothesis already has {a.already_tested} run(s) in the ledger; re-testing it on the "
                         "same data adds almost nothing - test it on new data (a holdout or paper trading) instead.")
        if a.verdict == "novel":
            lines.append(f"Novel: nothing related has been tested here, so the prediction is the learned base rate for a "
                         f"new idea (Sharpe {model.hyper.m0:+.2f} ± {model.hyper.s0:.2f}).")
        if a.neighbors:
            lines.append("Related experience:")
            for n in a.neighbors:
                rel = "the opposite claim" if n.similarity <= -0.8 else ("opposite direction" if n.similarity < 0 else "similar")
                lines.append(f"  - [{n.hypothesis_id}] {n.statement} (similarity {n.similarity:+.2f}, {rel}): belief "
                             f"{n.mean:+.2f} ± {n.sd:.2f} net, P(>0) = {_fmt_p(n.p_positive)}, {n.n_evidence} run(s).")
        gross, cost = a.details.get("gross_mean"), a.details.get("cost")
        if gross is not None and cost is not None and gross > 0 and a.mean < 0:
            lines.append(f"Costs decide it: the expected gross Sharpe ({gross:+.2f}) is smaller than the cost drag "
                         f"({cost:.2f}). It needs cheaper execution or slower rebalancing, not a better signal.")
        if a.verdict == "contradicts_experience":
            lines.append("Verdict: CONTRADICTS EXPERIENCE - related evidence in this market points the other way. "
                         "Test it only with a specific reason to expect a different outcome.")
        elif a.verdict == "consistent_with_experience":
            lines.append("Verdict: consistent with experience.")
        elif a.verdict == "uncertain":
            lines.append("Verdict: uncertain - experience neither supports nor contradicts it.")
        if a.related_tested >= 3:
            lines.append(f"Trials: {a.related_tested} related hypotheses were already tested on this market's data; an "
                         "in-sample success here is weak evidence until it holds on data the search has not seen.")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# LLM judge (prompt-only learning)
# ------------------------------------------------------------------------------------------------


class LLMView(BaseModel):
    literature_prior_sharpe: float = Field(
        description="Expected annualised Sharpe ratio NET of costs from economic reasoning and published evidence ALONE, "
                    "formed before reading the experience log.")
    literature_prior_sd: float = Field(gt=0, description="Your uncertainty about that number (Sharpe units, e.g. 0.5).")
    mechanism: str = Field(description="Who is on the other side of the trade and why the effect could persist (or not).")
    overall_p_positive: float = Field(ge=0, le=1, description="P(true net Sharpe > 0) after also weighing the experience log.")
    agrees_with_log: bool = Field(description="Does your overall view agree with the log's prediction?")
    reasoning: str = Field(description="Short argument. Quote numbers only from the experience log or the prediction.")
    concerns: list[str] = Field(default_factory=list, description="Ways the test could mislead (costs, data, overfitting ...).")
    suggested_next_tests: list[str] = Field(default_factory=list)
    cited_hypothesis_ids: list[str] = Field(default_factory=list, description="Ids (h-...) from the log your argument uses.")


_LLM_JUDGE_SYSTEM = """\
You are the skeptical research director of a systematic equity fund. A researcher proposes a hypothesis about \
stock returns. Before any backtest is run you judge how likely it is to be true, using two independent sources:

1. Economics and the published literature (what you knew before this fund existed). Give this as \
literature_prior_sharpe / literature_prior_sd, formed BEFORE you read the experience log. Be honest about \
publication bias: published anomalies lose about a third of their returns out of sample and many do not survive \
transaction costs; short-horizon signals in particular are eaten by costs.
2. The fund's experience log: every hypothesis tested in this market, with the belief a Bayesian engine derived \
from all the backtests (failures included). Similar claims share evidence; similarity -1 means the opposite claim. \
Use it for overall_p_positive and say whether you agree with the engine's prediction.

Rules: quote numbers only from the log or the prediction you are given; cite hypotheses by their ids; never invent \
results. If the log shows related failures, say what would have to be different for this idea to succeed. Keep the \
reasoning under 150 words."""


def _numbers_in(text: str) -> list[float]:
    return [float(m.group(0).rstrip("%")) for m in re.finditer(r"[-+]?\d+(?:\.\d+)?%?", text)]


def _combine(mean: float, sd: float, prior_mean: float, prior_sd: float) -> tuple[float, float]:
    """Precision-weighted combination of two independent Gaussian opinions."""
    w1, w2 = 1.0 / sd ** 2, 1.0 / prior_sd ** 2
    return (w1 * mean + w2 * prior_mean) / (w1 + w2), math.sqrt(1.0 / (w1 + w2))


class LLMJudge:
    """Language-model judge on top of the Bayesian one (see the module docstring)."""

    def __init__(self, llm: StructuredLLM, ledger: BeliefLedger | LedgerView, *, effort: str = "medium",
                 hurdle: float = DEFAULT_HURDLE, record_prior: bool = True):
        self.llm = llm
        self.ledger = ledger
        self.effort = effort
        self.hurdle = hurdle
        self._record_prior_requested = record_prior
        self.record_prior = record_prior and hasattr(ledger, "set_llm_prior")  # a writable ledger, not a view
        self.base = BayesianJudge(ledger, hurdle=hurdle, include_llm_priors=False)

    def bind(self, ledger: Any) -> "LLMJudge":
        return LLMJudge(self.llm, ledger, effort=self.effort, hurdle=self.hurdle, record_prior=self._record_prior_requested)

    @property
    def name(self) -> str:
        return f"llm:{getattr(self.llm, 'name', 'llm')}"

    def user_prompt(self, statement: str, mechanism: str, spec_json: str, base: Assessment, digest: str) -> str:
        return (
            f"<hypothesis>\n{statement}\n</hypothesis>\n"
            + (f"<proposed_mechanism>\n{mechanism}\n</proposed_mechanism>\n" if mechanism else "")
            + f"<testable_definition>\n{spec_json}\n</testable_definition>\n\n"
            f"<experience_log>\n{digest}\n</experience_log>\n\n"
            f"<engine_prediction>\n{base.critique}\n</engine_prediction>\n\n"
            "First give your literature prior (ignoring the log), then your overall view."
        )

    def assess(self, spec: Any, *, statement: str | None = None, mechanism: str = "") -> Assessment:
        s = spec_from_any(spec)
        base = self.base.assess(s, statement=statement, mechanism=mechanism)
        model = self.base.model()
        digest = experience_digest(model, target=hypothesis_shape(s), target_id=base.hypothesis_id)
        spec_json = json.dumps(s.model_dump(mode="json", exclude={"assumptions", "idea"}, exclude_defaults=True),
                               separators=(",", ":"))
        view = self.llm.structured(
            purpose="belief_judge", system=_LLM_JUDGE_SYSTEM,
            user=self.user_prompt(base.statement, mechanism, spec_json, base, digest),
            output_model=LLMView, effort=self.effort,
        )
        if not isinstance(view, LLMView):
            view = LLMView.model_validate(view)
        source = digest + "\n" + base.critique
        known_ids = set(HYP_ID_RE.findall(source))
        report = check_text(view.reasoning, known_ids=known_ids, numbers=_numbers_in(source))
        unknown = sorted({c for c in view.cited_hypothesis_ids if c not in known_ids} | set(report.unknown_ids))
        unverified = report.unverified_numbers
        prior_sd = max(float(view.literature_prior_sd), LLM_PRIOR_MIN_SD)
        if self.record_prior:
            self.ledger.add_hypothesis(s, statement=statement, mechanism=mechanism or view.mechanism, save=False)
            self.ledger.set_llm_prior(base.hypothesis_id, LLMPrior(
                mean=float(view.literature_prior_sharpe), sd=float(view.literature_prior_sd),
                model=str(getattr(self.llm, "name", "llm")), made_at=self.ledger.now(), reasoning=view.mechanism[:500]))
            combined = BayesianJudge(self.ledger, hurdle=self.hurdle, include_llm_priors=True).assess(
                s, statement=statement, mechanism=mechanism)
            mean, sd = combined.mean, combined.sd
        else:
            mean, sd = _combine(base.mean, base.sd, float(view.literature_prior_sharpe), prior_sd)
        p_pos = normal_cdf(mean / sd)
        note = [f"LLM view ({self.name}): literature prior {view.literature_prior_sharpe:+.2f} ± "
                f"{view.literature_prior_sd:.2f}; overall P(>0) = {view.overall_p_positive:.0%}"
                f" ({'agrees' if view.agrees_with_log else 'disagrees'} with the engine). {view.reasoning}"]
        if view.concerns:
            note.append("Concerns: " + "; ".join(view.concerns[:4]))
        if unknown:
            note.append(f"WARNING: cites hypothesis ids that are not in the log: {', '.join(unknown)}")
        if unverified:
            note.append(f"UNVERIFIED numbers in the LLM's reasoning (not in the log): {', '.join(unverified)}")
        verdict = _verdict(base.verdict == "novel", p_pos)
        return base.model_copy(update={
            "judge": f"{self.name}+bayes",
            "mean": mean,
            "sd": sd,
            "p_positive": p_pos,
            "p_hurdle": normal_cdf((mean - self.hurdle) / sd),
            "verdict": verdict,
            "acquisition": mean + sd,
            "critique": base.critique + "\n" + "\n".join(note),
            "details": {**base.details, "bayes_only": {"mean": base.mean, "sd": base.sd, "p_positive": base.p_positive},
                        "llm": view.model_dump(mode="json"), "unknown_citations": unknown,
                        "unverified_numbers": unverified},
        })


# ------------------------------------------------------------------------------------------------
# A judge model you trained, served behind an OpenAI-compatible endpoint
# ------------------------------------------------------------------------------------------------


class OpenAICompatibleJudge:
    """Ask a served judge model for P(Sharpe > 0) (see the module docstring).

    Args:
        ledger: the ledger the experience digest is built from.
        base_url: e.g. ``http://localhost:8000/v1`` (vLLM) or ``http://localhost:11434/v1`` (Ollama).
        model: the served model name.
        api_key: bearer token (default ``$AITRADING_JUDGE_API_KEY``; local servers usually need none).
        client: an ``httpx.Client`` (tests pass one with a mock transport).

    The model's probability replaces the engine's; its mean is the engine's sd x z(p), so the
    research loop can rank by mean + sd. An unusable reply falls back to the Bayesian assessment
    with ``details['error']`` set.
    """

    def __init__(self, ledger: BeliefLedger | LedgerView, *, base_url: str, model: str, api_key: str | None = None,
                 timeout_s: float = 60.0, client: Any = None, hurdle: float = DEFAULT_HURDLE):
        self.ledger = ledger
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.api_key = api_key if api_key is not None else os.environ.get("AITRADING_JUDGE_API_KEY", "")
        self.timeout_s = timeout_s
        self._client = client
        self.hurdle = hurdle
        self.base = BayesianJudge(ledger, hurdle=hurdle)

    def bind(self, ledger: Any) -> "OpenAICompatibleJudge":
        return OpenAICompatibleJudge(ledger, base_url=self.base_url, model=self.model_name, api_key=self.api_key,
                                     timeout_s=self.timeout_s, client=self._client, hurdle=self.hurdle)

    @property
    def name(self) -> str:
        return f"open:{self.model_name}"

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def assess(self, spec: Any, *, statement: str | None = None, mechanism: str = "") -> Assessment:
        from aitrading.beliefs.dataset import judge_messages, parse_judgement

        s = spec_from_any(spec)
        base = self.base.assess(s, statement=statement, mechanism=mechanism)
        digest = experience_digest(self.base.model(), target=hypothesis_shape(s), target_id=base.hypothesis_id)
        messages = judge_messages(base.statement, mechanism, digest)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = self._http().post(f"{self.base_url}/chat/completions", headers=headers, json={
                "model": self.model_name, "messages": messages, "temperature": 0.0, "max_tokens": 400})
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - a judge outage must not stop research
            return base.model_copy(update={"details": {**base.details, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}})
        parsed = parse_judgement(text)
        if parsed is None:
            return base.model_copy(update={"details": {**base.details, "error": "unparseable reply", "reply": text[:500]}})
        p = min(max(parsed["p_positive"], 0.01), 0.99)
        mean = base.sd * _ND.inv_cdf(p)
        return base.model_copy(update={
            "judge": self.name,
            "mean": mean,
            "p_positive": p,
            "verdict": _verdict(base.verdict == "novel", p),
            "p_hurdle": normal_cdf((mean - self.hurdle) / base.sd),
            "acquisition": mean + base.sd,
            "critique": base.critique + f"\nTrained judge ({self.name}): P(>0) = {p:.0%}. {parsed.get('reasoning', '')}".rstrip(),
            "details": {**base.details, "bayes_only": {"mean": base.mean, "sd": base.sd, "p_positive": base.p_positive},
                        "reply": text[:1000]},
        })
