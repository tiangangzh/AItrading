"""The critic of the RL research loop: the frozen Bayesian belief model, read through the
holdout-blind view, turned into values of states and of tests.

* ``value()`` - V = sum over the best ``k_fin`` mutually non-conflicting tested hypotheses of
  (net mean - hurdle)^+. Two hypotheses conflict when the kernel calls them the same bet
  (|similarity| >= 0.6) or their research-window returns move together (|corr| >= 0.6).
* ``kg(...)`` - the knowledge gradient of testing a proposal: E[V after its research-window run] - V
  now, given the tests already picked this round (kriging-believer fantasies). The run's noise is
  correlated, through the signed kernel, with existing runs on the same months and with the other
  planned tests, so copies and mirrors are worth about nothing. Computed exactly in the Gaussian
  model by Gauss-Hermite quadrature.
* Robust KG - the minimum over {the proposal, it without filters, its main component alone}, so
  salting an idea with filters or diluting it with extra components never raises its value.
* ``new_information`` - 1 - adjusted R^2 of a run's monthly returns on the most correlated earlier
  runs: invariant to shifting / rescaling, so it never depends on the observed Sharpe ratio.
* ``trust`` - per concept, 0.5 when the critic's own forecasts there were significantly too
  optimistic, else 1 (it can only lower rewards).
* ``finalists`` / ``exam_z`` / ``influence_shares`` - who sits the exam, the standardised TD error of
  the exam result, and how it is shared among the tests behind the belief.

No language model is involved: the critic reads specs and numbers, never words, and ignores LLM priors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from aitrading.beliefs.bayes import BeliefModel, Hyper, PlannedTest, predictive_sd
from aitrading.beliefs.digest import horizon_bucket
from aitrading.beliefs.judge import BayesianJudge
from aitrading.beliefs.ledger import BeliefLedger, CutoffLedger
from aitrading.beliefs.models import Evidence
from aitrading.beliefs.rl.reward import RewardSpec
from aitrading.beliefs.semantics import (
    FAMILY_LABELS,
    FEATURE_SEMANTICS,
    HypothesisShape,
    hypothesis_id,
    hypothesis_shape,
    primary_concept,
    similarity,
)
from aitrading.strategy.spec import PortfolioConstruction, SignalComponent, StrategySpec

__all__ = ["Critic", "CriticBatch", "RegionRow", "month_keys", "concept_code", "CONCEPT_CODES"]

#: Every (family, sign) concept code the catalog can express, e.g. 'value+' or 'past_return-'.
CONCEPT_CODES = sorted({f"{s.family}{'+' if sgn > 0 else '-'}" for s in FEATURE_SEMANTICS.values() for sgn in (1, -1)})
_REBAL_SESSIONS = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63, "annual": 252}


def month_keys(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + (m == 12), m % 12 + 1)
    return out


def concept_code(spec: StrategySpec) -> str:
    fam, sign, _ = primary_concept(spec)
    return f"{fam}{'+' if sign >= 0 else '-'}"


def _bucket(spec: StrategySpec) -> str | None:
    _, _, hor = primary_concept(spec)
    b = horizon_bucket(hor)
    return None if not b else ("short" if hor <= 10 else "medium" if hor <= 70 else "long")


@dataclass
class RegionRow:
    concept: str
    horizon_bucket: str | None
    label: str
    value: float
    n_tested: int
    status: Literal["open", "high_value", "supported", "refuted"]


@dataclass
class CriticBatch:
    """The joint posterior over every latent a round's selection needs."""

    specs: list[StrategySpec]  # one per proposal (None-free; invalid proposals are not in the batch)
    variants: list[list[int]]  # per proposal: latent indices of its robustness variants (first = itself)
    variant_complexity: list[int]
    latent_shapes: list[HypothesisShape]
    latent_ids: list[str]
    planned_index: dict[int, int]  # latent index -> index of its planned observation in the joint
    mean_gross: np.ndarray
    cost: np.ndarray  # cost drag per latent at the loop's cost level
    cov: np.ndarray
    n_latent: int
    arm_latents: list[int]  # latent indices of the tested arms (A_t)


def _r2_adj(y: np.ndarray, X: np.ndarray) -> float:
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0 or n - p - 1 <= 0:
        return 0.0
    r2 = 1.0 - float(np.sum(resid ** 2)) / tss
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


class Critic:
    """See the module docstring. One instance per round (``before`` = the round start)."""

    def __init__(self, ledger: BeliefLedger, *, cutoff: date, research_window: tuple[date | None, date], hyper: Hyper,
                 costs_bps: float, spec: RewardSpec, before: datetime | None = None, style: str = "long_short"):
        self.ledger = ledger
        self.cutoff = cutoff
        self.research_window = research_window
        self.costs_bps = float(costs_bps)
        self.spec = spec
        self.style = style
        self.hyper = hyper
        self.cut = CutoffLedger(ledger, cutoff)
        self.view = self.cut.view(before=before)
        self.model = BeliefModel(self.view, hurdle=spec.eta, include_llm_priors=False, hyper=hyper)
        self.judge = BayesianJudge(self.view, hurdle=spec.eta, include_llm_priors=False, hyper=hyper, name="bayes:critic")
        r0 = research_window[0] or self._first_month()
        self.research_months = month_keys(r0, research_window[1])
        self.research_years = max(len(self.research_months) / 12.0, 0.25)
        self.nodes, w = hermegauss(spec.gh_nodes)
        self.weights = w / w.sum()
        self._tested = self.model.tested()
        self._runs = self._research_runs()
        self._arms = self._select_arms()
        self._twin_cache: dict[tuple[str, str], bool] = {}

    # ------------------------------------------------------------------ helpers
    def _first_month(self) -> date:
        ev = self.view.evidence()
        return min((e.start for e in ev), default=date(self.research_window[1].year - 10, 1, 1))

    def _research_runs(self) -> dict[str, dict[str, float]]:
        """Monthly returns of each tested hypothesis's latest backtest run inside the research window."""
        out: dict[str, dict[str, float]] = {}
        months = set(self.research_months)
        for e in self.view.evidence():
            if e.kind != "backtest":
                continue
            m = {k: v for k, v in e.monthly.items() if k in months}
            if len(m) >= 12:
                out[e.hypothesis_id] = m
        return out

    def _net(self, hid: str) -> tuple[float, float]:
        mean, sd, _, _ = self.model.predict_net([self.model.shape(hid)], [self.costs_bps])
        return float(mean[0]), float(sd[0])

    def _select_arms(self) -> list[str]:
        if not self._tested:
            return []
        mean, sd, _, _ = self.model.predict_net([self.model.shape(h) for h in self._tested], [self.costs_bps] * len(self._tested))
        order = np.argsort(-(mean + 2 * sd))
        return [self._tested[i] for i in order[: self.spec.arms_max]]

    def _corr(self, a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
        common = sorted(set(a) & set(b))
        if len(common) < 24:
            return 0.0, len(common)
        x = np.array([a[k] for k in common])
        y = np.array([b[k] for k in common])
        if x.std() == 0 or y.std() == 0:
            return 0.0, len(common)
        return float(np.corrcoef(x, y)[0, 1]), len(common)

    def twins(self, a: str, b: str) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key not in self._twin_cache:
            ra, rb = self._runs.get(a), self._runs.get(b)
            self._twin_cache[key] = bool(ra and rb and abs(self._corr(ra, rb)[0]) >= self.spec.twin_corr)
        return self._twin_cache[key]

    # ------------------------------------------------------------------ value
    def _value(self, mu: np.ndarray, conflict: np.ndarray) -> float:
        order = np.argsort(-mu, kind="stable")
        taken: list[int] = []
        total = 0.0
        for i in order:
            if mu[i] <= self.spec.eta:
                break
            if any(conflict[i, j] for j in taken):
                continue
            taken.append(int(i))
            total += float(mu[i] - self.spec.eta)
            if len(taken) >= self.spec.k_fin:
                break
        return total

    def _conflicts(self, shapes: list[HypothesisShape], ids: list[str]) -> np.ndarray:
        n = len(shapes)
        c = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                hit = abs(similarity(shapes[i], shapes[j])) >= self.spec.conflict_sim or (
                    ids[i] in self._runs and ids[j] in self._runs and self.twins(ids[i], ids[j]))
                c[i, j] = c[j, i] = hit
        return c

    def value(self) -> float:
        if not self._arms:
            return 0.0
        shapes = [self.model.shape(h) for h in self._arms]
        mean, _, _, _ = self.model.predict_net(shapes, [self.costs_bps] * len(shapes))
        return self._value(mean, self._conflicts(shapes, self._arms))

    def finalists_now(self) -> list[str]:
        return self.finalists([(h, self._runs.get(h, {})) for h in self._tested], {h: 1.0 for h in self._tested})

    # ------------------------------------------------------------------ validation
    def research_tested(self, hid: str) -> bool:
        w = self.research_window
        for e in self.view.evidence_for(hid):
            if e.kind == "backtest" and abs((e.end - w[1]).days) <= 35 and (w[0] is None or abs((e.start - w[0]).days) <= 95):
                return True
        return False

    def validate(self, spec: StrategySpec, *, arm_specs: list[StrategySpec]) -> str | None:
        """None when ``spec`` may be considered, else the reason (see ``reward.PENALISED``)."""
        if spec.kind != "cross_sectional" or not spec.signal:
            return "unparseable"
        if any(c.feature not in FEATURE_SEMANTICS for c in spec.signal):
            return "unknown_feature"
        if len(spec.signal) > 3:
            return "too_many_components"
        if spec.portfolio.style != self.style:
            return "style_not_allowed"
        hid = hypothesis_id(spec)
        if any(hypothesis_id(s) == hid for s in arm_specs):
            return "duplicate_in_arm"
        if self.research_tested(hid):
            return "retest_same_window"
        shape = hypothesis_shape(spec)
        for s in arm_specs:
            if abs(similarity(shape, hypothesis_shape(s))) >= self.spec.near_dup_sim:
                return f"near_duplicate:{hypothesis_id(s)}"
        need = 0.5 * len(self.research_months)
        for h in self._tested:
            if abs(similarity(shape, self.model.shape(h))) >= self.spec.near_dup_sim and len(self._runs.get(h, {})) >= need:
                return f"near_duplicate:{h}"
        return None

    def novelty(self, spec: StrategySpec) -> float:
        shape = hypothesis_shape(spec)
        return max((abs(similarity(shape, self.model.shape(h))) for h in self._tested), default=0.0)

    def bucket(self, spec: StrategySpec) -> str:
        return "novel" if self.novelty(spec) < self.spec.novel_sim else concept_code(spec)

    # ------------------------------------------------------------------ robustness variants
    @staticmethod
    def variants(spec: StrategySpec) -> list[StrategySpec]:
        out = [spec]
        if spec.filters:
            out.append(spec.model_copy(update={"filters": []}))
        if len(spec.signal) > 1:
            main = max(spec.signal, key=lambda c: c.weight)
            out.append(spec.model_copy(update={"signal": [main.model_copy(update={"weight": 1.0})], "filters": []}))
        seen: set[str] = set()
        uniq = []
        for v in out:
            h = hypothesis_id(v)
            if h not in seen:
                seen.add(h)
                uniq.append(v)
        return uniq

    @staticmethod
    def complexity(spec: StrategySpec) -> int:
        return max(0, len(spec.signal) + len(spec.filters) - 2)

    # ------------------------------------------------------------------ the joint over a batch
    def evaluate(self, specs: list[StrategySpec]) -> CriticBatch:
        latent_shapes: list[HypothesisShape] = []
        latent_ids: list[str] = []
        index: dict[str, int] = {}

        def latent(shape: HypothesisShape, hid: str) -> int:
            if hid not in index:
                index[hid] = len(latent_shapes)
                latent_shapes.append(shape)
                latent_ids.append(hid)
            return index[hid]

        arm_latents = [latent(self.model.shape(h), h) for h in self._arms]
        variants: list[list[int]] = []
        complexity: list[int] = []
        for s in specs:
            idx = [latent(hypothesis_shape(v), hypothesis_id(v)) for v in self.variants(s)]
            variants.append(idx)
            complexity.append(self.complexity(s))
        planned_latents = sorted({i for v in variants for i in v})
        planned = [PlannedTest(latent_shapes[i], self.research_months, self.research_years) for i in planned_latents]
        joint = self.model.joint_planned(latent_shapes, planned)
        cost = np.array([self.model.cost_estimate(s, self.costs_bps)[0] for s in latent_shapes])
        n_lat = len(latent_shapes)
        return CriticBatch(
            specs=list(specs), variants=variants, variant_complexity=complexity, latent_shapes=latent_shapes,
            latent_ids=latent_ids, planned_index={li: n_lat + k for k, li in enumerate(planned_latents)},
            mean_gross=joint.mean.copy(), cost=cost, cov=joint.cov.copy(), n_latent=n_lat, arm_latents=arm_latents)

    def _kg(self, b: CriticBatch, cov: np.ndarray, latent_set: list[int], target: int) -> float:
        """KG of testing latent ``target`` given ``cov`` (already conditioned on fantasies)."""
        p = b.planned_index[target]
        var_y = cov[p, p]
        if var_y <= 1e-12:
            return 0.0
        base = sorted(set(latent_set))
        after = sorted(set(latent_set) | {target})
        mu = b.mean_gross[: b.n_latent] - b.cost
        ids = [b.latent_ids[i] for i in after]
        shapes = [b.latent_shapes[i] for i in after]
        conflict = self._conflicts(shapes, ids)
        pos = {li: k for k, li in enumerate(after)}
        sub = [pos[i] for i in base]
        v0 = self._value(mu[base], conflict[np.ix_(sub, sub)]) if base else 0.0
        sig = np.array([cov[li, p] for li in after]) / math.sqrt(var_y)
        mu_after = mu[after]
        ev = sum(w * self._value(mu_after + sig * z, conflict) for z, w in zip(self.nodes, self.weights))
        return max(0.0, float(ev - v0))

    @staticmethod
    def _fantasise(b: CriticBatch, cov: np.ndarray, target: int) -> np.ndarray:
        """Condition on the planned run of ``target`` reporting its expected value (means unchanged)."""
        p = b.planned_index[target]
        v = cov[p, p]
        if v <= 1e-12:
            return cov
        col = cov[:, p].copy()
        return cov - np.outer(col, col) / v

    def kg_plain(self, b: CriticBatch, i: int) -> float:
        return self._kg(b, b.cov, b.arm_latents, b.variants[i][0])

    def greedy(self, b: CriticBatch, indices: list[int]) -> list[tuple[int, float]]:
        """Greedy batch-marginal robust KG over the proposals ``indices``: [(proposal index, kappa)] in
        pick order. Each pick is fantasised before the next, so near-copies of a pick are worth ~0."""
        cov = b.cov.copy()
        chosen_latents: list[int] = []
        left = list(indices)
        out: list[tuple[int, float]] = []
        while left:
            best, best_k = None, -1.0
            for i in left:
                base = b.arm_latents + chosen_latents
                k = min(self._kg(b, cov, base, li) for li in b.variants[i])
                if k > best_k + 1e-15:
                    best, best_k = i, k
            assert best is not None
            out.append((best, best_k))
            left.remove(best)
            ident = b.variants[best][0]
            chosen_latents.append(ident)
            cov = self._fantasise(b, cov, ident)
        return out

    # ------------------------------------------------------------------ after a test
    def new_information(self, monthly: dict[str, float], hid: str, refs: list[tuple[str, dict[str, float]]]) -> float:
        """1 - adjusted R^2 of ``monthly`` on the (at most 3) most correlated runs in ``refs`` of OTHER
        hypotheses with >= 24 common months (dropping the weakest until >= 24 months are shared)."""
        cands = []
        for rid, m in refs:
            if rid == hid:
                continue
            c, n = self._corr(monthly, m)
            if n >= 24 and c != 0.0:
                cands.append((abs(c), rid, m))
        cands.sort(key=lambda t: -t[0])
        use = [m for _, _, m in cands[:3]]
        while use:
            common = sorted(set(monthly).intersection(*[set(m) for m in use]))
            if len(common) >= 24:
                y = np.array([monthly[k] for k in common])
                X = np.column_stack([[m[k] for k in common] for m in use])
                return float(min(1.0, max(0.0, 1.0 - _r2_adj(y, X))))
            use = use[:-1]
        return 1.0

    def research_refs(self) -> list[tuple[str, dict[str, float]]]:
        return list(self._runs.items())

    def trust(self) -> dict[str, float]:
        """tau per bucket: 0.5 where the critic's resolved forecasts were significantly too optimistic."""
        zs: dict[str, list[float]] = {}
        ev = {e.evidence_id: e for e in self.view.evidence()}
        for p in self.ledger.predictions():
            if p.judge != "bayes:critic" or p.purpose != "backtest" or p.target_evidence_id not in ev:
                continue
            e = ev[p.target_evidence_id]
            h = self.view.hypothesis(p.hypothesis_id)
            bucket = "novel" if p.note == "novel" else (h.concept if h is not None else "novel")
            zs.setdefault(bucket, []).append((e.sharpe - p.mean) / predictive_sd(p.sd, e))
        out = {}
        for b, z in zs.items():
            low = len(z) >= self.spec.trust_min_n and (sum(z) / len(z)) * math.sqrt(len(z)) < self.spec.trust_z
            out[b] = self.spec.tau_low if low else 1.0
        return out

    # ------------------------------------------------------------------ exam
    def finalists(self, tested: list[tuple[str, dict[str, float]]], rho_all: dict[str, float]) -> list[str]:
        cands = []
        for hid, _ in tested:
            if hid not in self.model.tested() or rho_all.get(hid, 0.0) < self.spec.rho_min_finalist:
                continue
            mu, _ = self._net(hid)
            if mu > 0:
                cands.append((mu, hid))
        cands.sort(key=lambda t: (-t[0], t[1]))
        out: list[str] = []
        for _, hid in cands:
            if any(abs(similarity(self.model.shape(hid), self.model.shape(o))) >= self.spec.conflict_sim or self.twins(hid, o)
                   for o in out):
                continue
            out.append(hid)
            if len(out) >= self.spec.k_fin:
                break
        return out

    def exam_z(self, hid: str, ev: Evidence) -> tuple[float, float, float]:
        mu, sd = self._net(hid)
        sdp = predictive_sd(sd, ev)
        z = max(-self.spec.z_clip, min(self.spec.z_clip, (ev.sharpe - mu) / sdp))
        return z, mu, sdp

    def influence_shares(self, hid: str, proposal_of_evidence: dict[str, str]) -> tuple[dict[str, float], float]:
        w, ids, resid = self.model.smoother_row(self.model.shape(hid))
        if not len(w):
            return {}, 1.0
        c = np.maximum(w * resid, 0.0)
        total = float(c.sum())
        if total <= 0:
            return {}, 1.0
        shares: dict[str, float] = {}
        external = 0.0
        for ci, eid in zip(c, ids):
            key = proposal_of_evidence.get(eid or "")
            if key is None:
                external += float(ci) / total
            else:
                shares[key] = shares.get(key, 0.0) + float(ci) / total
        return shares, external

    def return_contamination(self, hid: str, holdout: tuple[date, date], full: BeliefLedger) -> str | None:
        """Why the holdout is not unseen for ``hid`` by RETURNS: evidence overlapping the holdout of a
        hypothesis whose research-window run moves with this one (|corr| >= 0.6)."""
        mine = self._runs.get(hid)
        if not mine:
            return None
        for e in full.evidence():
            if e.end < holdout[0] or e.start > holdout[1] or e.hypothesis_id == hid:
                continue
            other = self._runs.get(e.hypothesis_id)
            if other and abs(self._corr(mine, other)[0]) >= self.spec.twin_corr:
                return f"{e.kind} run of [{e.hypothesis_id}], whose returns move with this one, already covers the holdout"
        return None

    # ------------------------------------------------------------------ the critic map shown to the actor
    def probes(self, theme: str | None) -> list[StrategySpec]:
        from aitrading.beliefs.loop import candidate_pool

        out = [p.spec for p in candidate_pool(theme)]
        seen_fam: set[str] = set()
        for feat, sem in sorted(FEATURE_SEMANTICS.items()):
            for d in ("higher_is_better", "lower_is_better"):
                key = f"{sem.family}{d}"
                if key in seen_fam:
                    continue
                seen_fam.add(key)
                short = sem.horizon is not None and sem.horizon <= 10
                out.append(StrategySpec(name=f"probe_{feat}_{d[:2]}", idea="probe", kind="cross_sectional",
                                        rebalance="weekly" if short else "monthly",
                                        signal=[SignalComponent(feature=feat, direction=d)],  # type: ignore[arg-type]
                                        portfolio=PortfolioConstruction(style=self.style)))  # type: ignore[arg-type]
        uniq, seen = [], set()
        for s in out:
            s = s.model_copy(update={"costs_bps": self.costs_bps, "portfolio": s.portfolio.model_copy(update={"style": self.style})})
            h = hypothesis_id(s)
            if h not in seen and not self.research_tested(h):
                seen.add(h)
                uniq.append(s)
        return uniq[:40]

    def region_map(self, theme: str | None, max_rows: int = 5) -> tuple[list[RegionRow], list[RegionRow]]:
        probes = self.probes(theme)
        rows: dict[tuple[str, str | None], RegionRow] = {}
        if probes:
            b = self.evaluate(probes)
            for i, s in enumerate(probes):
                key = (concept_code(s), _bucket(s))
                v = self.kg_plain(b, i)
                if key not in rows or v > rows[key].value:
                    rows[key] = RegionRow(key[0], key[1], _region_label(key), v, 0, "open")
        settled: dict[tuple[str, str | None], list[str]] = {}
        for bel in self.model.beliefs():
            h = self.view.hypothesis(bel.hypothesis_id)
            if h is None:
                continue
            key = (h.concept, _bucket(StrategySpec.model_validate(h.spec)))
            settled.setdefault(key, []).append(bel.status)
            if key in rows:
                rows[key].n_tested += 1
        open_rows = sorted(rows.values(), key=lambda r: -r.value)[:max_rows]
        for r in open_rows:
            r.status = "high_value" if r.value > self.spec.c_test else "open"
        done = []
        for key, sts in settled.items():
            if all(s == "refuted" for s in sts):
                done.append(RegionRow(key[0], key[1], _region_label(key), 0.0, len(sts), "refuted"))
            elif any(s == "supported" for s in sts):
                done.append(RegionRow(key[0], key[1], _region_label(key), 0.0, len(sts), "supported"))
        done.sort(key=lambda r: (-r.n_tested, r.label))
        return open_rows, done[:max_rows]


def _region_label(key: tuple[str, str | None]) -> str:
    concept, bucket = key
    fam, sign = concept[:-1], concept[-1:]
    label = FAMILY_LABELS.get(fam, (fam, f"inverse {fam}"))[0 if sign == "+" else 1]
    return f"{label}{f', {bucket} horizon' if bucket else ''} [{concept}]"


def critic_hyper(ledger: BeliefLedger, cutoff: date, before: datetime | None = None) -> Hyper:
    """The episode's frozen hyperparameters: empirical Bayes on the holdout-blind view, no LLM priors."""
    return BeliefModel(CutoffLedger(ledger, cutoff).view(before=before), include_llm_priors=False).hyper


def evidence_hash(view: Any) -> str:
    import hashlib

    ids = sorted(e.evidence_id for e in view.evidence())
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16]
