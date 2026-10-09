"""The RL research loop: an LLM policy improved through its prompt, scored by the Bayesian critic,
with out-of-sample evidence entering only as a TD error.

Formalisation:
* State = what is known (the holdout-blind view of the ledger), summarised by the critic.
* Action = a batch of proposals, sampled by the actor LLM conditioned on the policy text.
* The critic selects which proposals are tested: greedy batch knowledge gradient, or none if no test
  is worth its cost.
* Each proposal's reward is the critic's expected improvement of the exam-worthy set (kappa), times
  the new information its run carried, times the trust in the critic for that concept, minus costs.
  This is the critic's conditional expectation of the potential-based shaping reward, so it does
  not depend on the in-sample draw.
* The policy (the text) changes only when a challenger version beats the champion in a paired
  sequential test on the same observations; clean exam results can veto a champion.
* At the end the best diverse survivors sit ONE exam on unseen years. The standardised error of the
  critic's forecast (TD error) is shared among the proposals that built the belief.

Everything the LLM sees is computed from the holdout-blind view and from policy / credit records with
an information horizon before the holdout. A tripwire checks every prompt before it is sent. Every
call goes through ``StructuredLLM`` (Claude or an open model) behind a hard ``BudgetedLLM`` cap.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from aitrading.beliefs.bayes import normal_cdf
from aitrading.beliefs.digest import experience_digest
from aitrading.beliefs.grounding import leakage_tripwire, render_safe, scrub_statement
from aitrading.beliefs.learner import Learner
from aitrading.beliefs.ledger import BeliefLedger, CutoffLedger
from aitrading.beliefs.loop import LoopConfig, LoopReport, ResearchLoop
from aitrading.beliefs.rl.actor import ActorObservation, ActorResult, HeuristicActor, PromptPolicyActor, sketch_to_spec
from aitrading.beliefs.rl.critic import CONCEPT_CODES, Critic, concept_code, critic_hyper, evidence_hash
from aitrading.beliefs.rl.episodes import EpisodeStore
from aitrading.beliefs.rl.models import (
    ArmPrompt,
    EpisodeRecord,
    FinalistRecord,
    ProposalRecord,
    RoundRecord,
)
from aitrading.beliefs.rl.policy import GroundingContext, PolicyStore, directive_credit, render_policy, rl_lock
from aitrading.beliefs.rl.reflect import ReflectionContext, Reflector
from aitrading.beliefs.rl.reward import (
    INVALID,
    PENALISED,
    RewardSpec,
    advantages,
    arm_score,
    imputed_rho,
    pair_outcome,
    penalty_for,
    step_reward,
)
from aitrading.beliefs.semantics import hypothesis_id
from aitrading.llm.base import BudgetedLLM
from aitrading.screen.catalog import default_catalog
from aitrading.strategy.spec import StrategySpec

__all__ = ["RLResearchLoop", "RLConfig", "RLReport", "LeakageError"]


class LeakageError(RuntimeError):
    """A prompt about to be sent contains something only known from data at or after the holdout."""


@dataclass
class RLConfig(LoopConfig):
    candidates_per_round: int = 8
    costs_bps: float = 10.0
    style: str = "long_short"
    explore_fraction: float = 1.0 / 3.0
    reflect: bool = True
    max_llm_calls: int | None = None
    store_prompts: bool = True
    reward: RewardSpec = field(default_factory=RewardSpec)
    llm_cutoff: date | None = None


@dataclass
class RLReport(LoopReport):
    episode: EpisodeRecord | None = None
    policy: dict = field(default_factory=dict)
    baseline_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["rl"] = {"episode": self.episode.model_dump(mode="json") if self.episode else None, "policy": self.policy,
                   "baseline_scores": self.baseline_scores}
        return d

    def to_markdown(self) -> str:
        md = super().to_markdown()
        ep = self.episode
        if ep is None:
            return md
        lines = ["", "## Reinforcement learning", "",
                 f"Episode {ep.episode_id} ({ep.status}); actor {ep.actor_model}"
                 + (f", reflector {ep.reflector_model}" if ep.reflector_model else "")
                 + f"; reward {ep.reward_version}; LLM calls {ep.llm_calls_total}"
                 + (" (budget exhausted)" if ep.budget_exhausted else "") + ".",
                 f"Policy at start: champion {ep.champion_at_start}"
                 + (f", challenger {ep.challenger_at_start}" if ep.challenger_at_start else "")
                 + f". Now: champion {self.policy.get('champion')}"
                 + (f", challenger {self.policy.get('challenger')}" if self.policy.get("challenger") else "") + ".", "",
                 "| Round | Arms | Proposals | Tested | Arm scores | Pair | Heuristic baseline |",
                 "|---:|---|---:|---:|---|---|---:|"]
        for r in ep.rounds:
            arms = ", ".join(f"{a.arm}:{a.policy_version or '-'}" for a in r.arms) or "heuristic"
            scores = "; ".join(f"{k} {v:+.3f}" for k, v in (r.pair or {}).get("scores", {}).items())
            pair = (r.pair or {}).get("outcome", "")
            base = f"{r.baseline_score:+.3f}" if r.baseline_score is not None else ""
            lines.append(f"| {r.round} | {arms} | {len(r.proposals)} | {len(r.selected)} | {scores} | {pair} | {base} |")
        if ep.finalists:
            lines += ["", "| Finalist | Origin | Belief before | Holdout Sharpe | TD z | Clean |", "|---|---|---:|---:|---:|---|"]
            for f in ep.finalists:
                lines.append(f"| {f.hypothesis_id} | {f.origin or 'external'} | {f.mu:+.2f} ± {f.sd:.2f} | "
                             f"{'' if f.holdout_sharpe is None else f'{f.holdout_sharpe:+.2f}'} | "
                             f"{'' if f.z is None else f'{f.z:+.2f}'} | {'yes' if f.clean else (f.censor_reason or 'no')} |")
        if ep.r_oos is not None:
            lines.append(f"\nOut-of-sample utility of the clean finalists (mean of clip(Sharpe - hurdle, -2, 2)): {ep.r_oos:+.2f}.")
        for h in self.policy.get("history", []):
            lines.append(f"- policy: {h}")
        return md + "\n".join(lines) + "\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RLResearchLoop(ResearchLoop):
    """See the module docstring. ``actor``: :class:`PromptPolicyActor` (an LLM) or :class:`HeuristicActor`."""

    def __init__(self, runner: Any, ledger: BeliefLedger, *, actor: Any, reflector: Reflector | None = None,
                 config: RLConfig | None = None, progress: Callable[[str], None] | None = None,
                 policy_store: PolicyStore | None = None, episode_store: EpisodeStore | None = None):
        cfg = config or RLConfig()
        super().__init__(runner, ledger, config=cfg, progress=progress)
        self.cfg: RLConfig = cfg
        self.spec = cfg.reward.model_copy(update={"k_fin": cfg.holdout_top})
        self.actor = actor
        self.reflector = reflector
        self.policy = policy_store or PolicyStore(ledger, spec=self.spec)
        self.episodes = episode_store or EpisodeStore(ledger)
        self.heuristic = HeuristicActor()
        self.catalog = default_catalog()

    # ------------------------------------------------------------------ public
    def run(self, theme: str | None = None) -> RLReport:  # type: ignore[override]
        with rl_lock(self.ledger):
            self.episodes.mark_aborted()
            self.ingest_delayed()
            return self._run(theme)

    def run_walk_forward(self, theme: str | None, ends: list[date]) -> list[RLReport]:
        """One episode per end date; holdouts must be chronological and must not overlap."""
        if list(ends) != sorted(ends):
            raise ValueError("walk-forward ends must be sorted")
        prev_end: date | None = None
        windows = []
        for e in ends:
            cfg = RLConfig(**{**self.cfg.__dict__, "end": e})
            loop = RLResearchLoop(self.runner, self.ledger, actor=self.actor, reflector=self.reflector, config=cfg,
                                  progress=self._say, policy_store=self.policy, episode_store=self.episodes)
            (_, _), (h0, h1) = loop.windows()
            if prev_end is not None and h0 <= prev_end:
                raise ValueError(f"the holdout starting {h0} overlaps the previous one ending {prev_end}")
            prev_end = h1
            windows.append(loop)
        return [lp.run(theme) for lp in windows]

    def ingest_delayed(self) -> int:
        return self.episodes.attach_paper(self.ledger, self.policy, spec=self.spec)

    # ------------------------------------------------------------------ budget
    def _budget(self) -> tuple[Any, Any]:
        R, C = self.cfg.rounds, self.cfg.candidates_per_round
        llm = getattr(self.actor, "llm", None)
        per_call = getattr(self.actor, "proposals_per_call", None) == 1
        actor_calls = (R * C) if per_call else 2 * R
        cap = self.cfg.max_llm_calls if self.cfg.max_llm_calls is not None else actor_calls + R + 1
        actor_budget = BudgetedLLM(llm, cap, {"rl_actor": actor_calls, "rl_reflect": R + 1}) if llm is not None else None
        refl_llm = getattr(self.reflector, "llm", None)
        if refl_llm is None:
            refl_budget = None
        elif refl_llm is llm:
            refl_budget = actor_budget
        else:
            refl_budget = BudgetedLLM(refl_llm, R + 1, {"rl_reflect": R + 1})
        return actor_budget, refl_budget

    # ------------------------------------------------------------------ rendering
    def _audit(self, cutoff: date) -> list[float]:
        return [e.z for e in self.policy.load().events if e.kind == "terminal" and e.info_horizon < cutoff and e.z is not None]

    def _critic_map(self, c: Critic, theme: str | None, cutoff: date) -> tuple[str, set[str]]:
        ids: set[str] = set()
        h = c.hyper
        lines = [f"- Base rate for a new idea: gross Sharpe {h.m0:+.2f} ± {h.s0:.2f} (learned, frozen for this episode). "
                 f"Hurdle: net {self.spec.eta:g}. Costs: {self.cfg.costs_bps:g} bps one-way."]
        fin = c.finalists_now()
        if fin:
            parts = []
            for hid in fin:
                b = c.model.belief(hid)
                ids.add(hid)
                parts.append(f"[{hid}] {render_safe(b.statement, 120)} - net {b.mean:+.2f} ± {b.sd:.2f}")
            lines.append("- Finalists if research stopped now: " + "; ".join(parts))
        else:
            lines.append("- Finalists if research stopped now: none clears the hurdle")
        open_rows, settled = c.region_map(theme)
        if open_rows:
            lines.append("- Where a test is most valuable:")
            lines += [f"  {r.label} | value {r.value:.3f} | {r.n_tested} tested" for r in open_rows]
        if settled:
            lines.append("- Settled regions:")
            lines += [f"  {r.label} | {r.status} | {r.n_tested} tested" for r in settled]
        audit = self._audit(cutoff)
        lines.append(f"- Critic audit: exams that ended before {cutoff}: "
                     + (f"{len(audit)}, mean exam z {sum(audit) / len(audit):+.2f}" if audit else "no exams yet"))
        return "\n".join(lines), ids

    # ------------------------------------------------------------------ the episode
    def _run(self, theme: str | None) -> RLReport:
        cfg, spec, ledger = self.cfg, self.spec, self.ledger
        research, holdout = self.windows()
        h0, h1, r1 = holdout[0], holdout[1], research[1]
        uses_before = ledger.holdout_uses_overlapping(*holdout)
        view = CutoffLedger(ledger, h0)
        started = ledger.now()
        hyper = critic_hyper(ledger, h0, before=started)
        eid = "e-" + hashlib.sha1(f"{ledger.market}|{theme}|{h0}|{h1}|{started.isoformat()}".encode()).hexdigest()[:12]
        ps = self.policy.state(h0)
        actor_budget, refl_budget = self._budget()
        actor_llm_orig = getattr(self.actor, "llm", None)
        refl_llm_orig = getattr(self.reflector, "llm", None)
        if actor_budget is not None:
            self.actor.llm = actor_budget
        if refl_budget is not None:
            self.reflector.llm = refl_budget
        ep = EpisodeRecord(
            episode_id=eid, status="running", market=ledger.market, theme=theme, reward_version=spec.version,
            bayes_hash=_bayes_hash(), actor_model=getattr(self.actor, "name", "heuristic"),
            reflector_model=getattr(self.reflector, "name", None) if self.reflector else None, started_at=started,
            research_window=[research[0].isoformat() if research[0] else None, r1.isoformat()],
            holdout_window=[h0.isoformat(), h1.isoformat()], cutoff=h0, holdout_uses_before=uses_before,
            hyper_frozen={"m0": hyper.m0, "s0": hyper.s0, "source": hyper.source, "n_obs": hyper.n_obs},
            config={"costs_bps": cfg.costs_bps, "style": cfg.style, "k": spec.k_fin, "per_round": cfg.per_round,
                    "rounds": cfg.rounds, "candidates_per_round": cfg.candidates_per_round},
            champion_at_start=ps.champion, challenger_at_start=ps.challenger,
            after_llm_cutoff=(h0 > cfg.llm_cutoff) if cfg.llm_cutoff else None)
        self.episodes.save(ep)
        report = RLReport(market=ledger.market, theme=theme, generator=getattr(self.actor, "name", "?"), judge="bayes:critic",
                          research_window=research, holdout_window=holdout, holdout_uses_before=uses_before, episode=ep)
        if uses_before:
            report.warnings.append(f"The holdout window was used {uses_before} time(s) before: its results are in-sample "
                                   "and give no terminal reward.")
        last_round: list[str] = []
        tested: dict[str, dict[str, Any]] = {}  # hid -> {spec, key, version, monthly, rho_all}
        proposal_of_evidence: dict[str, str] = {}
        try:
            for t in range(1, cfg.rounds + 1):
                rr, last_round, ps = self._round(t, theme, ep, view, hyper, ps, last_round, tested, proposal_of_evidence,
                                                 report, actor_budget, refl_budget)
                ep.rounds.append(rr)
                self.episodes.save(ep)
                self.policy.save()
            self._exam(ep, report, hyper, tested, proposal_of_evidence, holdout, uses_before)
            ep.status = "finished"
        except LeakageError as exc:
            ep.status = "aborted"
            ep.warnings.append(f"aborted before the exam: {exc}")
            report.warnings.append(f"LEAKAGE: {exc}; the episode was aborted and no exam was taken")
        finally:
            if actor_budget is not None:
                self.actor.llm = actor_llm_orig
                ep.llm_calls_total = actor_budget.total + (refl_budget.total if refl_budget not in (None, actor_budget) else 0)
                ep.budget_exhausted = any(n.startswith("budget") for r in ep.rounds for n in r.notes)
            if refl_budget is not None:
                self.reflector.llm = refl_llm_orig
            ep.finished_at = ledger.now()
            self.episodes.save(ep)
            self.episodes.prune()
            self.policy.save()
        from aitrading.beliefs.bayes import BeliefModel
        from aitrading.beliefs.digest import lessons

        model = BeliefModel(ledger)
        hids = [h for h in tested if h in model.tested()]
        report.final = sorted(model.beliefs(hids), key=lambda b: -b.mean) if hids else []
        report.lessons = [x.text for x in lessons(model)]
        st = self.policy.state(None)
        report.policy = {"champion": st.champion, "challenger": st.challenger, "history": st.history}
        report.baseline_scores = [r.baseline_score for r in ep.rounds if r.baseline_score is not None]
        report.warnings.append(f"{len(tested)} hypothesis(es) tested in this episode, {len(ledger.tested_ids())} in this "
                               "market's ledger.")
        return report

    def _round(self, t: int, theme: str | None, ep: EpisodeRecord, view: CutoffLedger, hyper: Any, ps: Any,
               last_round: list[str], tested: dict[str, dict[str, Any]], proposal_of_evidence: dict[str, str],
               report: RLReport, actor_budget: Any, refl_budget: Any) -> tuple[RoundRecord, list[str], Any]:
        cfg, spec, ledger = self.cfg, self.spec, self.ledger
        research, holdout = self.windows()
        h0, r1 = holdout[0], research[1]
        round_start = ledger.now()
        c = Critic(ledger, cutoff=h0, research_window=research, hyper=hyper, costs_bps=cfg.costs_bps, spec=spec,
                   before=round_start, style=cfg.style)
        rr = RoundRecord(round=t, started_at=round_start, info_horizon=r1, view_evidence_hash=evidence_hash(c.view),
                         n_view_evidence=len(c.view.evidence()), hyper={"m0": hyper.m0, "s0": hyper.s0},
                         value_before=c.value())
        used_before = dict(actor_budget.used) if actor_budget is not None else {}
        self._say(f"round {t}/{cfg.rounds}: actor proposing (champion {ps.champion}"
                  + (f", challenger {ps.challenger}" if ps.challenger else "") + ") ...")
        llm_actor = isinstance(self.actor, PromptPolicyActor)
        arms: list[tuple[str, str | None]] = [("champion", ps.champion)] if llm_actor else [("heuristic", None)]
        if llm_actor and ps.challenger:
            arms.append(("challenger", ps.challenger))
        n_arm = max(1, cfg.candidates_per_round // len(arms))
        n_explore = math.ceil(n_arm * cfg.explore_fraction)
        digest = experience_digest(c.model)
        cmap, cmap_ids = self._critic_map(c, theme, h0)
        credit_rows = [p for _, r, p in self.episodes.rows(cutoff=h0)]
        arm_props: dict[str, list[tuple[Any, str]]] = {}
        arm_labels: dict[str, str] = {}
        rendered_ids = set(cmap_ids)
        for role, vid in arms:
            if vid is not None:
                version = self.policy.version(vid)
                lineage = set(self.policy.lineage(vid))
                rows = [p for p in credit_rows if p.policy_version in lineage]
                ptext, pids = render_policy(version, directive_credit(version, rows), c, role=role)
                rendered_ids |= pids
            else:
                ptext = "<policy>none (offline heuristic proposals)</policy>"
            obs = ActorObservation(theme=theme, round=t, cutoff=h0, info_horizon=r1, role=role, policy_version=vid,
                                   policy_text=ptext, critic_map=cmap, digest=digest, last_round=last_round, n=n_arm,
                                   n_explore=n_explore, rendered_ids=rendered_ids)
            res: ActorResult
            if llm_actor:
                prompt = self.actor.system_prompt() + "\n" + self.actor.user_prompt(obs)
                hits = leakage_tripwire(prompt, ledger, h0)
                if hits:
                    raise LeakageError("; ".join(hits[:5]))
                res = self.actor.propose(obs)
                if res.batch is None:
                    rr.notes.append(("budget" if res.error and "budget" in res.error else "actor failed")
                                    + f" ({role}): heuristic fallback")
                    res = self.heuristic.propose(obs, exclude=set(tested))
                    role_label = "heuristic"
                else:
                    role_label = role
            else:
                res = self.heuristic.propose(obs, exclude=set(tested))
                role_label = "heuristic"
            arm_labels[role] = role_label
            if cfg.store_prompts and res.system_sha:
                self.episodes.save_system_prompt(self.actor.system_prompt())
            rr.arms.append(ArmPrompt(arm=role_label, policy_version=vid if role_label != "heuristic" else None,
                                     purpose=res.purpose, system_sha=res.system_sha, user_sha=res.user_sha,
                                     user_text=res.user_text[:12000] if cfg.store_prompts else None))
            arm_props[role] = [(p, role_label) for p in (res.batch.proposals if res.batch else [])[:n_arm]]
        # ---- validate and value
        records: list[ProposalRecord] = []
        specs: list[StrategySpec] = []
        rec_spec_index: dict[int, int] = {}
        arm_indices: dict[str, list[int]] = {}
        for role, props in arm_props.items():
            arm_valid: list[StrategySpec] = []
            for p, label in props:
                idx = len(records)
                name = f"rl_{ep.episode_id[2:8]}_r{t}_{idx}"
                s, reason = sketch_to_spec(p, name=name, costs_bps=cfg.costs_bps, window=research, style=cfg.style,
                                           catalog=self.catalog)
                status = reason
                if s is not None:
                    v = c.validate(s, arm_specs=arm_valid)
                    status = v.split(":", 1)[0] if v else None
                    reason = v
                vid = dict(arms).get(role) if label != "heuristic" else None
                rec = ProposalRecord(idx=idx, arm=label, policy_version=vid, statement=p.statement[:300],
                                     mechanism=p.mechanism[:400], sketch=p.model_dump(mode="json"),
                                     spec=s.model_dump(mode="json") if s is not None else None,
                                     hypothesis_id=hypothesis_id(s) if s is not None else None,
                                     concept=concept_code(s) if s is not None else None,
                                     status=status or "not_selected", reason=reason, follows=list(p.follows)[:12],
                                     explore_claimed=bool(p.explore), expected_sharpe=float(p.expected_sharpe))
                if s is not None:
                    rec.max_sim_tested = c.novelty(s)
                    rec.explore_verified = rec.explore_claimed and rec.max_sim_tested < spec.novel_sim
                if s is not None and status is None:
                    arm_valid.append(s)
                    rec_spec_index[idx] = len(specs)
                    specs.append(s)
                    arm_indices.setdefault(role, []).append(idx)
                records.append(rec)
        batch = c.evaluate(specs) if specs else None
        tau = c.trust()
        rr.tau = tau
        kappa_arm: dict[int, float] = {}
        if batch is not None:
            for role, idxs in arm_indices.items():
                for si, k in c.greedy(batch, [rec_spec_index[i] for i in idxs]):
                    kappa_arm[[i for i in idxs if rec_spec_index[i] == si][0]] = k
            pooled = c.greedy(batch, list(range(len(specs))))
            inv = {v: k for k, v in rec_spec_index.items()}
            selected = [inv[si] for si, k in pooled if k >= spec.c_test][: cfg.per_round]
            for si, k in pooled:
                i = inv[si]
                records[i].credit.kg_plain = c.kg_plain(batch, si)
                if i not in selected:
                    records[i].status = "below_test_cost" if k < spec.c_test else "not_selected"
        else:
            selected = []
        rr.selected = selected
        # ---- test the selected proposals (research window only)
        rho_arm: dict[int, float] = {}
        tested_this_round: list[tuple[int, str, dict[str, float]]] = []
        base_refs = c.research_refs()
        learner = Learner(view, c.judge, belief_judge=c.judge)
        for i in selected:
            rec = records[i]
            s = specs[rec_spec_index[i]]
            keep, mech, issues = scrub_statement(rec.statement, rec.mechanism, info_horizon=r1)
            rec.flagged = issues[:10]
            a = c.judge.assess(s)
            actor_pred = None
            if rec.arm != "heuristic":
                m = max(-3.0, min(3.0, rec.expected_sharpe))
                actor_pred = ledger.add_prediction(hypothesis_id=rec.hypothesis_id or "", judge=f"actor:{getattr(self.actor, 'name', 'llm')}",
                                                   purpose="backtest", mean=m, sd=0.5, p_positive=normal_cdf(m / 0.5),
                                                   note="actor forecast", save=False)
            self._say(f"  testing [{rec.hypothesis_id}] {(keep or rec.statement)[:70]} (value {kappa_arm.get(i, 0.0):.3f})")
            trec = self._test(learner, t, s, keep, mech, f"rl:{rec.arm}:{rec.policy_version or '-'}", research, "backtest",
                              None, assessment=a, text_horizon=r1)
            report.tests.append(trec)
            rec.predicted = {"mean": a.mean, "sd": a.sd, "p_positive": a.p_positive, "verdict": a.verdict}
            if trec.update is None:
                rec.status = "tested"
                rec.reason = trec.error
                continue
            rec.status = "tested"
            rec.observed_sharpe = trec.observed_sharpe
            rec.evidence_id = trec.update.evidence_id
            if actor_pred is not None and rec.evidence_id not in actor_pred.context_evidence:
                ledger.resolve_prediction(actor_pred.prediction_id, rec.evidence_id, save=False)
            ev = ledger.data.evidence.get(rec.evidence_id)
            monthly = dict(ev.monthly) if ev else {}
            same_arm = [(h, m) for j, h, m in tested_this_round if records[j].arm == rec.arm]
            rho_arm[i] = c.new_information(monthly, rec.hypothesis_id or "", base_refs + same_arm)
            rho_all = c.new_information(monthly, rec.hypothesis_id or "", base_refs + [(h, m) for _, h, m in tested_this_round])
            rec.credit.rho_all = rho_all
            tested_this_round.append((i, rec.hypothesis_id or "", monthly))
            key = f"r{t}:{i}"
            proposal_of_evidence[rec.evidence_id] = key
            tested[rec.hypothesis_id or ""] = {"spec": s, "key": key, "version": rec.policy_version, "monthly": monthly,
                                               "rho_all": rho_all, "statement": keep, "mechanism": mech,
                                               "source": f"rl:{rec.arm}:{rec.policy_version or '-'}"}
        ledger.save()
        # ---- rewards
        rho_hat = imputed_rho(list(rho_arm.values()), spec)
        for rec in records:
            cr = rec.credit
            if rec.status in PENALISED:
                cr.penalty = penalty_for(rec.status, spec)
                cr.reward = cr.penalty
                continue
            s = specs[rec_spec_index[rec.idx]]
            cr.kappa = kappa_arm.get(rec.idx, 0.0)
            if rec.idx in rho_arm:
                cr.rho, cr.rho_imputed = rho_arm[rec.idx], False
            else:
                cr.rho, cr.rho_imputed = rho_hat, True
            cr.tau = tau.get(c.bucket(s), 1.0)
            cr.complexity_excess = Critic.complexity(s)
            cr.reward = step_reward(cr.kappa, cr.rho, cr.tau, cr.complexity_excess, spec)
        scores: dict[str, float] = {}
        for role in arm_props:
            mine = [r for r in records if r.idx in {i for i in range(len(records))} and
                    ((r.arm == arm_labels[role]) and (r.policy_version == (dict(arms).get(role) if arm_labels[role] != "heuristic" else None)))]
            rewards = [r.credit.reward for r in mine]
            adv, adv_std = advantages(rewards)
            for r, x, xs in zip(mine, adv, adv_std):
                r.credit.advantage, r.credit.advantage_std = x, xs
            deficit = max(0, n_explore - sum(1 for r in mine if r.explore_verified)) if arm_labels[role] != "heuristic" else 0
            scores[role] = arm_score(rewards, [r.status for r in mine], cfg.per_round, deficit, spec)
        rr.baseline_score = self._baseline(c, theme, n_arm, set(tested))
        pair: dict[str, Any] = {"scores": scores}
        if arm_labels.get("champion") == "champion" and arm_labels.get("challenger") == "challenger":
            outcome = pair_outcome(scores["champion"], scores["challenger"], spec)
            self.policy.record_pair(ps.champion, ps.challenger, scores["champion"], scores["challenger"], outcome=outcome,
                                    info_horizon=r1, episode_id=ep.episode_id, key=f"{ep.episode_id}:r{t}:pair")
            pair["outcome"] = outcome
            ps = self.policy.state(h0)
        rr.pair = pair
        rr.proposals = records[:16]
        c_after = Critic(ledger, cutoff=h0, research_window=research, hyper=hyper, costs_bps=cfg.costs_bps, spec=spec,
                         before=ledger.now(), style=cfg.style)
        rr.value_after = c_after.value()
        feedback = self._feedback_lines(records, c_after, rho_arm, n_explore)
        rr.feedback_out = feedback[:20]
        # ---- reflection (single-arm rounds only)
        two_arms = len(arms) > 1
        if (self.reflector is not None and cfg.reflect and not two_arms and arm_labels.get("champion") == "champion"
                and ps.challenger is None):
            ctx = self._reflection_context(c_after, ps, records, cmap, rendered_ids, r1)
            edit, _ = self.reflector.reflect(ctx)
            if edit is None:
                rr.notes.append("reflection failed or budget spent: no challenger")
            else:
                if edit.diagnosis:
                    rr.notes.append("diagnosis (audit only): " + edit.diagnosis[:300])
                known = rendered_ids & set(c_after.view.tested_ids())
                gctx = GroundingContext(known_ids=known, catalog_features=set(self.catalog.names()), concepts=set(CONCEPT_CODES))
                v, rejected = self.policy.propose_challenger(
                    edit, base=self.policy.version(ps.champion), ctx=gctx, tabu=ps.tabu, info_horizon=r1,
                    episode_id=ep.episode_id, author=getattr(self.reflector, "name", "reflector"), key=f"{ep.episode_id}:r{t}:challenger")
                if rejected:
                    rr.notes.append("edits rejected: " + "; ".join(",".join(r) for _, r in rejected)[:200])
                if v is not None:
                    rr.notes.append(f"challenger {v.version_id}: {'; '.join(v.edit_summary)}")
                ps = self.policy.state(h0)
        if actor_budget is not None:
            rr.llm_calls = {k: v - used_before.get(k, 0) for k, v in actor_budget.used.items() if v - used_before.get(k, 0)}
            if actor_budget.remaining() == 0:
                rr.notes.append("budget spent")
        return rr, feedback, ps

    # ------------------------------------------------------------------ pieces
    def _baseline(self, c: Critic, theme: str | None, n_arm: int, exclude: set[str]) -> float | None:
        """Diagnostic only: the heuristic pool's top proposals scored by the same critic in the same state."""
        props = self.heuristic.proposals(theme, exclude, n_arm * 3)
        specs = []
        for p in props:
            s, _ = sketch_to_spec(p, name="baseline", costs_bps=self.cfg.costs_bps, window=self.windows()[0],
                                  style=self.cfg.style, catalog=self.catalog)
            if s is not None and c.validate(s, arm_specs=specs) is None:
                specs.append(s)
        if not specs:
            return None
        b = c.evaluate(specs)
        picks = c.greedy(b, list(range(len(specs))))[:n_arm]
        rho = self.spec.rho_prior
        rewards = [step_reward(k, rho, 1.0, Critic.complexity(specs[i]), self.spec) for i, k in picks]
        return arm_score(rewards, ["not_selected"] * len(rewards), self.cfg.per_round, 0, self.spec)

    def _feedback_lines(self, records: list[ProposalRecord], c: Critic, rho_arm: dict[int, float], n_explore: int) -> list[str]:
        out: list[str] = []
        for r in records:
            hid = r.hypothesis_id
            stmt = render_safe(c.model.hypothesis(hid).statement if hid and c.model.hypothesis(hid) else r.statement, 120)
            if r.status == "tested" and r.evidence_id:
                b = c.model.belief(hid) if hid in c.model.tested() else None
                pred = r.predicted or {}
                line = (f"[{hid}] {stmt}: value of the test {r.credit.kappa:.3f}; new information {r.credit.rho:.2f}; predicted "
                        f"{pred.get('mean', 0):+.2f} ± {pred.get('sd', 0):.2f}, observed {r.observed_sharpe:+.2f} on the "
                        f"research window")
                if b is not None:
                    line += f"; belief now {b.mean:+.2f} ± {b.sd:.2f}, {b.status}"
                out.append(line + ".")
                if r.credit.rho is not None and r.credit.rho < 0.3:
                    out.append(f"[{hid}]: its returns were {100 * (1 - r.credit.rho):.0f}% explained by earlier runs - a "
                               "disguised copy earns nothing.")
            elif r.status in ("below_test_cost", "not_selected"):
                why = "below the test cost" if r.status == "below_test_cost" else "batch full"
                out.append(f"[{hid}] {stmt}: value of a test {r.credit.kg_plain:.3f} - not tested ({why}).")
            elif r.status in PENALISED:
                out.append(f"proposal {r.idx}: invalid ({r.status}).")
        claimed = [r for r in records if r.explore_claimed]
        out.append(f"explore quota: {sum(1 for r in claimed if r.explore_verified)} of {len(records)} proposals were in "
                   f"untested territory (kernel check); {n_explore} were asked for per arm.")
        return out

    def _reflection_context(self, c: Critic, ps: Any, records: list[ProposalRecord], cmap: str, rendered_ids: set[str],
                            r1: date) -> ReflectionContext:
        version = self.policy.version(ps.champion)
        lineage = set(self.policy.lineage(ps.champion))
        rows = [p for _, r, p in self.episodes.rows(cutoff=self.windows()[1][0]) if p.policy_version in lineage] + \
               [r for r in records if r.policy_version in lineage]
        credit = directive_credit(version, rows)
        ptext, pids = render_policy(version, credit, c, role="champion")
        tbl = ["round | statement | h-id | follows | explore claimed/verified | value | new information | reward | advantage | status"]
        for r in records[:30]:
            tbl.append(f"now | {render_safe(r.statement, 120)} | {r.hypothesis_id or '-'} | {','.join(r.follows) or '-'} | "
                       f"{'y' if r.explore_claimed else 'n'}/{'y' if r.explore_verified else 'n'} | {r.credit.kappa:.3f} | "
                       f"{'' if r.credit.rho is None else f'{r.credit.rho:.2f}'} | {r.credit.reward:+.4f} | "
                       f"{r.credit.advantage:+.4f} | {r.status}")
        ranked = sorted((r for r in records if r.status not in INVALID), key=lambda r: -r.credit.advantage)
        bw = [f"best: {render_safe(r.statement, 120)} | {render_safe(r.mechanism, 200)} | advantage {r.credit.advantage:+.4f}"
              for r in ranked[:3]]
        bw += [f"worst: {render_safe(r.statement, 120)} | {render_safe(r.mechanism, 200)} | advantage {r.credit.advantage:+.4f}"
               for r in ranked[-3:]]
        return ReflectionContext(policy_table=ptext, round_table="\n".join(tbl), best_worst="\n".join(bw), critic_map=cmap,
                                 concepts=CONCEPT_CODES, history=ps.history[-8:], info_horizon=r1,
                                 rendered_ids=rendered_ids | pids | {r.hypothesis_id for r in records if r.hypothesis_id})

    def _exam(self, ep: EpisodeRecord, report: RLReport, hyper: Any, tested: dict[str, dict[str, Any]],
              proposal_of_evidence: dict[str, str], holdout: tuple[date, date], uses_before: int) -> None:
        cfg, spec, ledger = self.cfg, self.spec, self.ledger
        research = self.windows()[0]
        c = Critic(ledger, cutoff=holdout[0], research_window=research, hyper=hyper, costs_bps=cfg.costs_bps, spec=spec,
                   before=ledger.now(), style=cfg.style)
        finalists = c.finalists([(h, d["monthly"]) for h, d in tested.items()], {h: d["rho_all"] for h, d in tested.items()})
        if not finalists:
            report.warnings.append("No finalist: nothing sat the exam and the holdout was not spent.")
            return
        ledger.note_holdout_use(*holdout)
        exam = Learner(ledger, c.judge, belief_judge=c.judge)
        z_by: dict[str, float] = {}
        utilities: list[float] = []
        for hid in finalists:
            d = tested[hid]
            mu, sd = c._net(hid)
            why = None
            if uses_before:
                why = "holdout used before"
            else:
                why = self.contamination(d["spec"], holdout) or c.return_contamination(hid, holdout, ledger)
            fr = FinalistRecord(hypothesis_id=hid, origin=d["key"], origin_version=d["version"], mu=mu, sd=sd,
                                rho_all=d["rho_all"], clean=why is None, censor_reason=why)
            s = d["spec"]
            a = c.judge.assess(s)
            self._say(f"holdout exam: [{hid}]")
            rec = self._test(exam, 0, s, d["statement"], d["mechanism"], d["source"], holdout, "holdout", why is None,
                             assessment=a, text_horizon=research[1])
            report.holdout.append(rec)
            if rec.update is not None:
                ev = ledger.data.evidence.get(rec.update.evidence_id)
                fr.evidence_id = rec.update.evidence_id
                fr.holdout_sharpe = ev.sharpe if ev else None
                if ev is not None:
                    z, _, _ = c.exam_z(hid, ev)
                    fr.z = z
                    shares, external = c.influence_shares(hid, proposal_of_evidence)
                    fr.shares, fr.external_share = shares, external
                    if fr.clean:
                        z_by[hid] = z
                        utilities.append(max(-2.0, min(2.0, ev.sharpe - spec.eta)))
                        self.policy.record_outcome("terminal", d["version"], hid, z, info_horizon=holdout[1],
                                                   episode_id=ep.episode_id, key=f"{ep.episode_id}:{hid}:{ev.evidence_id}")
            elif why is None:
                fr.censor_reason = "the exam run failed"
            if why and not uses_before:
                report.warnings.append(f"[{hid}] exam not clean: {why}.")
            ep.finalists.append(fr)
        # terminal credit back to the episode's proposals
        for f in ep.finalists:
            if f.hypothesis_id not in z_by:
                continue
            for key, share in f.shares.items():
                rnd, idx = key[1:].split(":")
                for r in ep.rounds:
                    if r.round == int(rnd):
                        for p in r.proposals:
                            if p.idx == int(idx):
                                p.credit.terminal_credit = (p.credit.terminal_credit or 0.0) + share * z_by[f.hypothesis_id]
        ep.r_oos = sum(utilities) / len(utilities) if utilities else None
        ledger.save()


def _bayes_hash() -> str:
    from aitrading.beliefs.bayes import bayes_hash

    return bayes_hash()
