"""Training data for an open-weights ACTOR, from the RL episode logs (weight-level RL).

Unit of data: one proposal with its arm's prompt, rewritten deterministically to the single-proposal
task (the format ``PromptPolicyActor(proposals_per_call=1)`` uses).

* ``actor_examples`` - every logged proposal, point in time. A row is dropped, with a reason, when:
  - its stored prompt is missing, truncated or does not match its hash;
  - the ledger's holdout-blind view at the round's start no longer has the logged evidence hash
    (e.g. after ``beliefs forget``);
  - the reward version or the belief model differs from the export's.
* ``sft_rows`` - rejection sampling: valid proposals with positive reward and advantage, and no
  negative terminal credit.
* ``dpo_pairs`` - pairs from the SAME prompt, labelled by the outcome-blind step reward.
* ``kto_rows`` - finalists with a clean terminal credit only (the rare labels that are not the
  critic's opinion).
* ``grpo_prompts`` / ``grpo_reward`` - online RL with the FROZEN round critic as reward model. It is
  computed offline: no runner is called, and a real-market holdout never enters a reward.

Leakage rules:
* Train rows need ``label_horizon`` < ``t_split``.
* Eval rows need a cutoff >= ``t_split`` + embargo.
* At most 3 positives per hypothesis per split.
* ``after_llm_cutoff`` flags rows the base model cannot have read about.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from aitrading.beliefs.bayes import Hyper, bayes_hash
from aitrading.beliefs.ledger import BeliefLedger, CutoffLedger
from aitrading.beliefs.rl.actor import sketch_to_spec
from aitrading.beliefs.rl.critic import Critic, evidence_hash
from aitrading.beliefs.rl.episodes import EpisodeStore
from aitrading.beliefs.rl.models import ActorBatch, ActorProposal
from aitrading.beliefs.rl.reward import PENALISED, RewardSpec, penalty_for, step_reward

__all__ = ["Snapshot", "actor_examples", "sft_rows", "dpo_pairs", "kto_rows", "grpo_prompts", "grpo_reward"]

_TASK_RE = re.compile(r"<task>.*?</task>", re.S)


@dataclass
class Snapshot:
    market: str
    cutoff: date
    before: datetime
    hyper: dict
    reward_version: str
    bayes_hash: str
    research_window: list
    costs_bps: float
    k: int
    per_round: int


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _single_task(user_text: str, explore: bool, done: list[str]) -> str:
    task = (f"<task>Propose 1 hypothesis with explore={'true' if explore else 'false'}. Already proposed this round: "
            f"{'; '.join(d[:100] for d in done) or 'none'}</task>")
    return _TASK_RE.sub(task, user_text, count=1)


def actor_examples(ledger: BeliefLedger, store: EpisodeStore, *, t_split: date, llm_cutoff: date | None = None,
                   embargo_days: int = 182, spec: RewardSpec | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """(rows, dropped) - see the module docstring."""
    spec = spec or RewardSpec()
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    bh = bayes_hash()
    for ep in store.episodes():
        if ep.reward_version != spec.version or ep.bayes_hash != bh:
            dropped.append({"episode": ep.episode_id, "reason": "version_mismatch"})
            continue
        cut = CutoffLedger(ledger, ep.cutoff)
        for rnd in ep.rounds:
            if evidence_hash(cut.view(before=rnd.started_at)) != rnd.view_evidence_hash:
                dropped.append({"episode": ep.episode_id, "round": str(rnd.round), "reason": "ledger_changed"})
                continue
            arms = {a.arm: a for a in rnd.arms}
            done: dict[str, list[str]] = {}
            for p in rnd.proposals:
                arm = arms.get(p.arm)
                if arm is None or not arm.system_sha:
                    continue
                if arm.user_text is None or arm.truncated or _sha(arm.user_text) != arm.user_sha:
                    dropped.append({"episode": ep.episode_id, "round": str(rnd.round), "reason": "prompt_missing_or_changed"})
                    continue
                system = store.system_prompt(arm.system_sha)
                if system is None:
                    dropped.append({"episode": ep.episode_id, "round": str(rnd.round), "reason": "system_prompt_missing"})
                    continue
                prev = done.setdefault(p.arm, [])
                user = _single_task(arm.user_text, p.explore_claimed, prev)
                prev.append(p.statement)
                terminal = p.credit.terminal_credit
                label_horizon = date.fromisoformat(ep.holdout_window[1]) if terminal is not None else rnd.info_horizon
                eval_row = ep.cutoff >= t_split + timedelta(days=embargo_days)
                rows.append({
                    "example_id": f"{ep.episode_id}:r{rnd.round}:{p.idx}",
                    "prompt_key": f"{ep.episode_id}:r{rnd.round}:{p.arm}",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "completion": json.dumps({"plan": "", "proposals": [p.sketch]}, ensure_ascii=False),
                    "hypothesis_id": p.hypothesis_id, "status": p.status,
                    "reward": p.credit.reward, "advantage": p.credit.advantage, "advantage_std": p.credit.advantage_std,
                    "rho_imputed": p.credit.rho_imputed, "terminal_credit": terminal,
                    "clean_terminal": terminal is not None and any(f.clean and f.hypothesis_id == p.hypothesis_id
                                                                   for f in ep.finalists),
                    "label_horizon": label_horizon.isoformat(), "cutoff": ep.cutoff.isoformat(),
                    "split": "eval" if eval_row else ("train" if label_horizon < t_split else "excluded"),
                    "after_llm_cutoff": None if llm_cutoff is None else bool(ep.cutoff > llm_cutoff),
                    "snapshot": {"market": ep.market, "cutoff": ep.cutoff.isoformat(), "before": rnd.started_at.isoformat(),
                                 "hyper": ep.hyper_frozen, "reward_version": ep.reward_version, "bayes_hash": ep.bayes_hash,
                                 "research_window": ep.research_window, "costs_bps": ep.config.get("costs_bps", 10.0),
                                 "style": ep.config.get("style", "long_short"), "k": ep.config.get("k", spec.k_fin),
                                 "per_round": ep.config.get("per_round", 3)},
                })
    return rows, dropped


def sft_rows(ex: list[dict[str, Any]], *, min_adv: float = 0.0, max_per_hypothesis: int = 3) -> list[dict[str, Any]]:
    out, per = [], {}
    for r in ex:
        if r["split"] == "excluded" or r["status"] in PENALISED or r["reward"] <= 0 or r["advantage"] <= min_adv:
            continue
        if r["terminal_credit"] is not None and r["terminal_credit"] < 0:
            continue
        key = (r["split"], r["hypothesis_id"])
        if per.get(key, 0) >= max_per_hypothesis:
            continue
        per[key] = per.get(key, 0) + 1
        out.append(r)
    return out


def dpo_pairs(ex: list[dict[str, Any]], *, margin: float = 0.01, max_pairs_per_prompt: int = 6) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in ex:
        if r["split"] != "excluded":
            by.setdefault(r["prompt_key"], []).append(r)
    out = []
    for key, rs in by.items():
        pairs = []
        for a in rs:
            for b in rs:
                if a is b or a["reward"] - b["reward"] < margin or a["status"] in PENALISED:
                    continue
                comparable = (a["status"] == "tested") == (b["status"] == "tested") or b["status"] in PENALISED
                if comparable:
                    pairs.append((a["reward"] - b["reward"], a, b))
        pairs.sort(key=lambda t: -t[0])
        for _, a, b in pairs[:max_pairs_per_prompt]:
            out.append({"prompt": a["messages"], "chosen": a["completion"], "rejected": b["completion"], "split": a["split"],
                        "prompt_key": key})
    return out


def kto_rows(ex: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"prompt": r["messages"], "completion": r["completion"], "label": r["terminal_credit"] > 0, "split": r["split"]}
            for r in ex if r["clean_terminal"] and r["terminal_credit"] is not None and r["split"] != "excluded"]


def grpo_prompts(ex: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for r in ex:
        if r["split"] == "train" and r["prompt_key"] not in seen:
            seen.add(r["prompt_key"])
            out.append({"messages": r["messages"], "snapshot": r["snapshot"]})
    return out


def grpo_reward(completions: list[str], snapshot: dict[str, Any] | Snapshot, *, ledger: BeliefLedger,
                spec: RewardSpec | None = None) -> list[float]:
    """Score a GROUP of sampled completions (one ActorProposal each) with the frozen snapshot critic:
    -1 unparseable, the validity penalties, otherwise greedy batch-marginal robust KG x prior new
    information - costs. Offline: no backtest is run, nothing recorded after the snapshot counts."""
    spec = spec or RewardSpec()
    snap = snapshot if isinstance(snapshot, dict) else snapshot.__dict__
    cutoff = date.fromisoformat(str(snap["cutoff"]))
    before = datetime.fromisoformat(str(snap["before"]))
    rw = snap["research_window"]
    window = (date.fromisoformat(rw[0]) if rw[0] else None, date.fromisoformat(rw[1]))
    hyper = Hyper(m0=snap["hyper"]["m0"], s0=snap["hyper"]["s0"], source="frozen")
    spec = spec.model_copy(update={"k_fin": int(snap.get("k", spec.k_fin))})
    critic = Critic(ledger, cutoff=cutoff, research_window=window, hyper=hyper, costs_bps=float(snap.get("costs_bps", 10.0)),
                    spec=spec, before=before, style=str(snap.get("style", "long_short")))
    rewards = [0.0] * len(completions)
    specs, idx_of = [], {}
    for i, text in enumerate(completions):
        prop = _parse_one(text)
        if prop is None:
            rewards[i] = -1.0
            continue
        s, reason = sketch_to_spec(prop, name=f"grpo_{i}", costs_bps=critic.costs_bps, window=window, style=critic.style)
        if s is None:
            rewards[i] = penalty_for(reason or "unparseable", spec)
            continue
        v = critic.validate(s, arm_specs=specs)
        if v is not None:
            rewards[i] = penalty_for(v.split(":", 1)[0], spec)
            continue
        idx_of[len(specs)] = i
        specs.append(s)
    if specs:
        b = critic.evaluate(specs)
        for si, k in critic.greedy(b, list(range(len(specs)))):
            s = specs[si]
            rewards[idx_of[si]] = step_reward(k, spec.rho_prior, 1.0, Critic.complexity(s), spec)
    return rewards


def _parse_one(text: str) -> ActorProposal | None:
    from aitrading.llm.openai_compatible import extract_json

    try:
        obj = json.loads(extract_json(text))
    except (ValueError, TypeError):
        return None
    try:
        if isinstance(obj, dict) and "proposals" in obj:
            b = ActorBatch.model_validate(obj)
            return b.proposals[0] if b.proposals else None
        return ActorProposal.model_validate(obj)
    except ValueError:
        return None
