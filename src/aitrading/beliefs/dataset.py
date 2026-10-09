"""Training data and reward for a judge model of your own (the "intuition" model).

The ledger is a growing record of (hypothesis, what was known then, what happened next). This module
turns it into supervised / RL examples for an open-weights judge:

* :func:`judge_messages` - THE prompt format: the same one :class:`~aitrading.beliefs.judge.OpenAICompatibleJudge`
  sends in use, so the model is trained on exactly what it will see.
* :func:`build_examples` - one example per piece of evidence. The context is reconstructed point in
  time: only evidence recorded BEFORE the target was recorded, and for out-of-sample targets
  (holdout, paper) only evidence whose data ENDED before the target's data began (``strict_time``), so
  nothing about the target period can leak into the prompt. Each example carries the Bayesian
  judge's probability in that same context: a trained model is only worth serving if it beats it on
  the held-out (later) examples.
* :func:`judge_reward` - a proper scoring rule: the log score of the stated probability for the
  observed sign, relative to a coin flip, so the policy is paid for calibrated probabilities, not
  for always saying "yes". Unparseable output scores -1.

Caveats that no reward function fixes: outcomes are noisy (one year of returns pins a Sharpe ratio
down to about +/-1), outcomes in one period are correlated (a regime lifts every momentum variant at
once), and a pretrained model has read papers about the backtest period. Train on early periods,
evaluate on later ones, prefer out-of-sample targets, and flag examples whose data predate the base
model's knowledge cutoff (``llm_cutoff``).
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from aitrading.beliefs.bayes import BeliefModel, normal_cdf, predictive_sd
from aitrading.beliefs.digest import experience_digest
from aitrading.beliefs.ledger import BeliefLedger
from aitrading.beliefs.semantics import hypothesis_shape

__all__ = [
    "JUDGE_SYSTEM",
    "judge_messages",
    "parse_judgement",
    "judge_reward",
    "build_examples",
    "write_jsonl",
]

JUDGE_SYSTEM = (
    "You judge hypotheses about stock returns for a systematic fund. Read the hypothesis and the fund's experience "
    "log, then answer with one JSON object: {\"p_positive\": <probability 0-1 that the strategy's true Sharpe ratio "
    "net of costs is above 0 on new data>, \"expected_sharpe\": <number>, \"reasoning\": \"<one or two sentences>\"}. "
    "Be calibrated: most new ideas fail, related failures in the log are evidence, re-tests of the same data are not."
)

_FLOOR = 0.02  # probabilities are clipped to [0.02, 0.98] when scored


def judge_messages(statement: str, mechanism: str, digest: str) -> list[dict[str, str]]:
    user = f"<hypothesis>\n{statement.strip()}\n</hypothesis>\n"
    if mechanism and mechanism.strip():
        user += f"<mechanism>\n{mechanism.strip()}\n</mechanism>\n"
    user += f"<experience_log>\n{digest.strip()}\n</experience_log>\nAnswer with the JSON object only."
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


_JSON_RE = re.compile(r"\{.*\}", re.S)
_P_RE = re.compile(r"p_positive\"?\s*[:=]\s*([0-9]*\.?[0-9]+)")


def parse_judgement(text: str) -> dict[str, Any] | None:
    """``{"p_positive": float in [0, 1], "expected_sharpe": float | None, "reasoning": str}`` or None."""
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    obj: dict[str, Any] | None = None
    for m in _JSON_RE.finditer(cleaned):
        try:
            cand = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(cand, dict) and "p_positive" in cand:
            obj = cand
            break
    if obj is None:
        m = _P_RE.search(cleaned)
        if not m:
            return None
        obj = {"p_positive": m.group(1)}
    try:
        p = float(obj["p_positive"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        return None
    es = obj.get("expected_sharpe")
    try:
        es = float(es) if es is not None else None
    except (TypeError, ValueError):
        es = None
    return {"p_positive": p, "expected_sharpe": es if es is None or math.isfinite(es) else None,
            "reasoning": str(obj.get("reasoning", ""))[:1000]}


def judge_reward(completion: str, outcome_positive: bool) -> float:
    """Log score of the stated probability for the observed outcome minus the coin-flip log score:
    log2(p) + 1 if positive, log2(1 - p) + 1 otherwise (p clipped to [0.02, 0.98]); -1 if unparseable.
    0 = no better than 50/50, 1 = certain and right, about -4.6 = certain and wrong."""
    parsed = parse_judgement(completion)
    if parsed is None:
        return -1.0
    p = min(max(parsed["p_positive"], _FLOOR), 1.0 - _FLOOR)
    return (math.log2(p) if outcome_positive else math.log2(1.0 - p)) + 1.0


def build_examples(ledger: BeliefLedger, *, kinds: Iterable[str] = ("backtest", "holdout", "paper"),
                   strict_time: bool = True, test_from: date | None = None, llm_cutoff: date | None = None,
                   min_context: int = 0) -> list[dict[str, Any]]:
    """Point-in-time judge examples from the ledger (see the module docstring).

    Args:
        kinds: evidence kinds used as targets.
        strict_time: for holdout / paper targets, keep only context evidence whose data ended before
            the target's data started.
        test_from: examples whose target data start on or after this date are ``split = "test"``.
        llm_cutoff: the base model's knowledge cutoff; ``after_llm_cutoff`` marks targets that start later
            (only those are free of what the model may have read about the period).
        min_context: skip targets with fewer context runs than this.
    """
    ks = set(kinds)
    out: list[dict[str, Any]] = []
    for target in ledger.evidence(kinds=ks):
        oos_target = target.kind in ("holdout", "paper")

        def keep(e, t=target, oos=oos_target) -> bool:
            if e.evidence_id == t.evidence_id:
                return False
            if strict_time and oos and e.end >= t.start:
                return False
            return True

        view = ledger.view(before=target.recorded_at, evidence_filter=keep)
        if len(view.evidence()) < min_context:
            continue
        hyp = ledger.hypothesis(target.hypothesis_id)
        if hyp is None:
            continue
        model = BeliefModel(view)
        shape = hypothesis_shape(hyp.spec)
        mean, sd, _, _ = model.predict_net([shape], [target.costs_bps])
        p_bayes_true = normal_cdf(float(mean[0]) / float(sd[0]))
        p_bayes_obs = normal_cdf(float(mean[0]) / predictive_sd(float(sd[0]), target))
        digest = experience_digest(model, target=shape, target_id=hyp.hypothesis_id)
        out.append({
            "example_id": f"{target.evidence_id}",
            "messages": judge_messages(hyp.statement, hyp.mechanism, digest),
            "hypothesis_id": hyp.hypothesis_id,
            "target_kind": target.kind,
            "target_start": target.start.isoformat(),
            "target_end": target.end.isoformat(),
            "recorded_at": target.recorded_at.isoformat(),
            "outcome_sharpe": target.sharpe,
            "outcome_se": target.se,
            "outcome_positive": bool(target.sharpe > 0),
            "out_of_sample": target.out_of_sample,
            "context_runs": len(view.evidence()),
            "bayes_p_positive": p_bayes_obs,
            "bayes_p_true_positive": p_bayes_true,
            "split": "test" if test_from is not None and target.start >= test_from else "train",
            "after_llm_cutoff": None if llm_cutoff is None else bool(target.start > llm_cutoff),
            "market": ledger.market,
        })
    return out


def write_jsonl(examples: list[dict[str, Any]], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return p
