"""Belief engine: what the ledger's evidence says about every hypothesis, tested or not.

Model
-----
A hypothesis h has an unknown true annualised Sharpe ratio BEFORE trading costs, g_h, and a
trading-cost drag c_h (annual cost / annual volatility); what a trader earns is the net Sharpe
g_h - c_h. The two are modelled separately because they relate differently across hypotheses:
"3-day momentum" and "3-day reversal" are opposite bets (g_reversal ~ -g_momentum) that trade
exactly as much (same c), so if momentum loses 1.1 net of costs it does NOT follow that reversal
earns 1.1 - both can lose once costs are paid.

Gross Sharpe ratios are one Gaussian process over hypothesis space:

    g ~ GP(m0, s0^2 * k(h, h'))          k = semantics.similarity (PSD, in [-1, 1])
    y_i = net Sharpe_i + c_i = g_{h(i)} + e_i      one observation per piece of evidence
    e ~ N(0, Sigma)

* ``k`` makes related claims share evidence: a success of 3-day momentum raises the prior on 5-day
  momentum and lowers it on 3-day reversal (similarity -1: the opposite claim).
* ``c_i`` is measured for backtests (the engine reports turnover; cost drag = 2 x turnover x
  rebalances a year x bps / volatility). A hypothesis's own runs give its cost (scaled to the cost
  level asked about); without them, or for evidence that does not report turnover (paper trading),
  c is estimated from the runs of similar hypotheses (weights similarity^2, signs ignored), shrunk
  towards a weak prior by rebalance frequency; its uncertainty is added to the prediction.
* ``Sigma`` is the measurement noise. Its diagonal is the Sharpe ratio's sampling variance (Lo 2002)
  + ``REGIME_SD``^2 / years (a strategy's true Sharpe drifts from one period to the next, so even a
  long backtest does not pin it down exactly) + ``IMPL_SD``^2 (runs of one hypothesis differ in
  implementation: quintiles vs deciles ...). Its off-diagonal terms are what stop repeated testing
  of the same data from looking like new evidence: two runs whose monthly returns overlap for
  n_common months and correlate rho there have correlated errors, rho x n_common / sqrt(n_i x n_j).
  Re-running a strategy on the same years (or a near-identical variant) therefore adds almost
  nothing, while the same claim on new data (a holdout, paper trading) is a genuinely independent
  observation.
* ``m0`` / ``s0`` - how good a brand-new idea is expected to be, and how far ideas differ - are
  learned from the ledger by empirical Bayes (maximum a posteriori over a grid, with a weak
  hyperprior centred on m0 = 0, s0 = 0.5). If most ideas tried in this market failed, s0 shrinks
  and every new in-sample success is discounted more: the engine learns how skeptical to be.

Multiple testing: because every run is in the ledger (failures included) the posterior conditions
on all of them, and a lucky winner among many similar variants is pulled back towards its
neighbours. A lucky winner among UNRELATED ideas is only shrunk towards m0; the learned s0 and the
holdout / paper evidence are the guard there. The research loop also reports the trial count.

Beliefs and predictions are reported NET of costs, at the cost level of the spec being assessed
(or of a hypothesis's first test). The posterior is computed exactly (Cholesky); a few hundred
pieces of evidence take milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aitrading.beliefs.ledger import BeliefLedger, LedgerView, as_view
from aitrading.beliefs.models import Belief, Evidence, Hypothesis
from aitrading.beliefs.semantics import HypothesisShape, hypothesis_shape, similarity, spec_from_any

__all__ = [
    "BeliefModel",
    "Hyper",
    "PlannedTest",
    "JointGaussian",
    "bayes_hash",
    "REGIME_SD",
    "IMPL_SD",
    "DEFAULT_PRIOR_MEAN",
    "DEFAULT_PRIOR_SD",
    "DEFAULT_HURDLE",
    "LLM_PRIOR_MIN_SD",
    "COST_PRIOR",
    "normal_cdf",
    "predictive_sd",
]

REGIME_SD = 0.35  # year-to-year drift of a strategy's true Sharpe ratio
IMPL_SD = 0.10  # differences between implementations of one hypothesis
DEFAULT_PRIOR_MEAN = 0.0  # a new idea, before any experience: no free lunch
DEFAULT_PRIOR_SD = 0.5  # published anomalies are typically 0.3-0.8 gross; most new ideas are ~0 net
HYPER_MEAN_SD = 0.2  # hyperprior on m0: N(0, 0.2^2)
HYPER_LOGSD_SD = 0.5  # hyperprior on log s0: N(log 0.5, 0.5^2)
DEFAULT_HURDLE = 0.3  # a Sharpe worth trading after costs (economically meaningful)
LLM_PRIOR_MIN_SD = 0.5  # an LLM's literature prior is never treated as more precise than ~4 years of data
MIN_COMMON_MONTHS = 6
#: Weak prior on the cost drag (Sharpe units at 10 bps one-way) by rebalance frequency: (mean, sd).
COST_PRIOR = {"daily": (2.0, 1.5), "weekly": (0.8, 0.6), "monthly": (0.15, 0.15), "quarterly": (0.05, 0.05),
              "annual": (0.02, 0.02)}
COST_PRIOR_WEIGHT = 0.5  # pseudo-weight of that prior against similarity^2-weighted measured costs
COST_SD_FLOOR = 0.03
_S0_GRID = np.geomspace(0.08, 1.5, 30)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predictive_sd(sd: float, ev: Evidence) -> float:
    """Spread of a new observation like ``ev`` around a belief with standard deviation ``sd``."""
    return math.sqrt(sd * sd + ev.se * ev.se + REGIME_SD ** 2 / max(ev.years, 0.25) + IMPL_SD ** 2)


@dataclass
class Hyper:
    m0: float = DEFAULT_PRIOR_MEAN
    s0: float = DEFAULT_PRIOR_SD
    source: str = "default"
    n_obs: int = 0


@dataclass
class _Obs:
    hid: str
    y: float  # gross Sharpe observation
    var: float  # sampling + regime variance (correlated part)
    extra: float  # independent variance (implementation, cost estimate, prior width)
    months: dict[str, float] = field(default_factory=dict)
    evidence_id: str | None = None  # None for an LLM-prior pseudo-observation


@dataclass
class PlannedTest:
    """A backtest not run yet: the hypothesis, the calendar months its returns will cover, its length."""

    shape: HypothesisShape
    months: list[str]
    years: float


@dataclass
class JointGaussian:
    """Posterior joint distribution of ``n_latent`` gross Sharpe ratios followed by planned observations."""

    labels: list[str]
    mean: np.ndarray
    cov: np.ndarray
    n_latent: int


def _psd_correlation(r: np.ndarray) -> np.ndarray:
    """Nearest-ish correlation matrix: clip negative eigenvalues, restore a unit diagonal."""
    r = 0.5 * (r + r.T)
    w, v = np.linalg.eigh(r)
    if w.min() >= -1e-10:
        return r
    w = np.clip(w, 1e-8, None)
    out = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(out), 1e-12, None))
    out = out / d[:, None] / d[None, :]
    np.fill_diagonal(out, 1.0)
    return out


def _chol(a: np.ndarray) -> np.ndarray:
    jitter = 1e-9 * max(1.0, float(np.mean(np.diag(a))))
    for _ in range(8):
        try:
            return np.linalg.cholesky(a + jitter * np.eye(len(a)))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("covariance matrix is not positive definite")


def _solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, z)


def _bps(x: float | None) -> float:
    return 10.0 if x is None or not math.isfinite(x) else max(float(x), 0.0)


class BeliefModel:
    """Posterior over hypotheses given a ledger (or a point-in-time view of one).

    Args:
        source: a :class:`BeliefLedger` or :class:`LedgerView`.
        hurdle: Sharpe ratio reported as ``p_hurdle`` = P(net Sharpe > hurdle).
        include_llm_priors: also condition on the literature priors language models gave
            (each a pseudo-observation of the net Sharpe with sd >= ``LLM_PRIOR_MIN_SD``).
        hyper: fixed hyperparameters (default: empirical Bayes from the evidence).
    """

    def __init__(self, source: BeliefLedger | LedgerView, *, hurdle: float = DEFAULT_HURDLE,
                 include_llm_priors: bool = False, hyper: Hyper | None = None):
        self.view = as_view(source)
        self.market = self.view.market
        self.hurdle = float(hurdle)
        self.include_llm_priors = include_llm_priors
        self._hyps: dict[str, Hypothesis] = {h.hypothesis_id: h for h in self.view.hypotheses()}
        self._shapes: dict[str, HypothesisShape] = {}
        self._cost_obs = [(self.shape(e.hypothesis_id), float(e.cost_sharpe), _bps(e.costs_bps))
                          for e in self.view.evidence()
                          if e.hypothesis_id in self._hyps and e.cost_sharpe is not None and math.isfinite(e.cost_sharpe)]
        self._obs = self._observations()
        self._fixed_hyper = hyper
        self.hyper = Hyper()
        self._fit()

    # ------------------------------------------------------------------ setup
    def shape(self, hid: str) -> HypothesisShape:
        if hid not in self._shapes:
            self._shapes[hid] = hypothesis_shape(self._hyps[hid].spec)
        return self._shapes[hid]

    def cost_estimate(self, shape: HypothesisShape, costs_bps: float | None) -> tuple[float, float]:
        """(mean, sd) of the cost drag in Sharpe units for a hypothesis traded at ``costs_bps``."""
        bps = _bps(costs_bps)
        prior_mean, prior_sd = COST_PRIOR.get(shape.rebalance, (0.15, 0.15))
        prior_mean, prior_sd = prior_mean * bps / 10.0, prior_sd * bps / 10.0
        w_sum, wc_sum, wc2_sum = 0.0, 0.0, 0.0
        own: list[float] = []
        for s, c, b in self._cost_obs:
            scaled = c * bps / b if b > 0 else c
            sim = similarity(shape, s)
            if sim >= 0.999:  # the hypothesis's own runs measured its cost: use them
                own.append(scaled)
                continue
            w = sim ** 2
            if w < 1e-3:
                continue
            w_sum += w
            wc_sum += w * scaled
            wc2_sum += w * scaled * scaled
        if own:
            return float(np.mean(own)), max(float(np.std(own)), COST_SD_FLOOR)
        if w_sum <= 0:
            return prior_mean, max(prior_sd, COST_SD_FLOOR)
        total = w_sum + COST_PRIOR_WEIGHT
        mean = (wc_sum + COST_PRIOR_WEIGHT * prior_mean) / total
        spread = max(wc2_sum / w_sum - (wc_sum / w_sum) ** 2, 0.0)
        sd = math.sqrt(spread + (prior_sd * COST_PRIOR_WEIGHT / total) ** 2)
        return mean, max(sd, COST_SD_FLOOR)

    def _observations(self) -> list[_Obs]:
        obs: list[_Obs] = []
        for e in self.view.evidence():
            if e.hypothesis_id not in self._hyps or not math.isfinite(e.sharpe) or not math.isfinite(e.se):
                continue
            var = e.se ** 2 + REGIME_SD ** 2 / max(e.years, 0.25)
            extra = IMPL_SD ** 2
            if e.cost_sharpe is not None and math.isfinite(e.cost_sharpe):
                cost = float(e.cost_sharpe)
            else:
                cost, csd = self.cost_estimate(self.shape(e.hypothesis_id), e.costs_bps)
                extra += csd * csd
            obs.append(_Obs(e.hypothesis_id, float(e.sharpe) + cost, var, extra, dict(e.monthly), e.evidence_id))
        if self.include_llm_priors:
            for h in self._hyps.values():
                if h.llm_prior is not None:
                    sd = max(float(h.llm_prior.sd), LLM_PRIOR_MIN_SD)
                    cost, csd = self.cost_estimate(self.shape(h.hypothesis_id), h.spec.get("costs_bps"))
                    obs.append(_Obs(h.hypothesis_id, float(h.llm_prior.mean) + cost, 0.0, sd * sd + csd * csd, {}))
        return obs

    def _noise_cov(self) -> np.ndarray:
        n = len(self._obs)
        sd = np.sqrt(np.array([o.var for o in self._obs]))
        corr = np.eye(n)
        with_months = [i for i, o in enumerate(self._obs) if len(o.months) >= 2]
        if len(with_months) >= 2:
            frame = pd.DataFrame({i: pd.Series(self._obs[i].months, dtype=float) for i in with_months}).sort_index()
            mask = frame.notna().to_numpy(dtype=float)
            common = mask.T @ mask
            counts = mask.sum(axis=0)
            rho = frame.corr(min_periods=MIN_COMMON_MONTHS).to_numpy(dtype=float)
            rho = np.where(np.isfinite(rho), rho, 0.0)
            overlap = common / np.sqrt(np.outer(counts, counts))
            sub = np.clip(rho * overlap, -1.0, 1.0)
            np.fill_diagonal(sub, 1.0)
            idx = np.array(with_months)
            corr[np.ix_(idx, idx)] = sub
        corr = _psd_correlation(corr)
        cov = corr * np.outer(sd, sd)
        cov[np.diag_indices(n)] += np.array([o.extra for o in self._obs])
        return cov

    def _kernel(self, a: list[str], b: list[str]) -> np.ndarray:
        sa = [self.shape(h) for h in a]
        sb = [self.shape(h) for h in b]
        return np.array([[similarity(x, y) for y in sb] for x in sa], dtype=float).reshape(len(sa), len(sb))

    def _fit(self) -> None:
        n = len(self._obs)
        self._obs_hids = [o.hid for o in self._obs]
        self._y = np.array([o.y for o in self._obs], dtype=float)
        if n == 0:
            self.hyper = self._fixed_hyper or Hyper()
            self._L = None
            return
        self._Kh = self._kernel(self._obs_hids, self._obs_hids)
        self._Sigma = self._noise_cov()
        self.hyper = self._fixed_hyper if self._fixed_hyper is not None else self._empirical_bayes()
        h = self.hyper
        K = h.s0 ** 2 * self._Kh + self._Sigma
        self._L = _chol(K)
        self._alpha = _solve(self._L, self._y - h.m0)

    def _empirical_bayes(self) -> Hyper:
        best: tuple[float, float, float] | None = None
        ones = np.ones(len(self._y))
        for s0 in _S0_GRID:
            K = s0 ** 2 * self._Kh + self._Sigma
            try:
                L = _chol(K)
            except np.linalg.LinAlgError:
                continue
            a = float(ones @ _solve(L, ones))
            b = float(ones @ _solve(L, self._y))
            m0 = b / (a + 1.0 / HYPER_MEAN_SD ** 2)
            r = self._y - m0
            quad = float(r @ _solve(L, r))
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
            lp = (-0.5 * quad - 0.5 * logdet
                  - 0.5 * (m0 / HYPER_MEAN_SD) ** 2
                  - 0.5 * ((math.log(s0) - math.log(DEFAULT_PRIOR_SD)) / HYPER_LOGSD_SD) ** 2)
            if best is None or lp > best[0]:
                best = (lp, m0, float(s0))
        if best is None:
            return Hyper(n_obs=len(self._y))
        return Hyper(m0=best[1], s0=best[2], source="empirical_bayes", n_obs=len(self._y))

    # ------------------------------------------------------------------ predictions
    def predict_gross(self, shapes: list[HypothesisShape]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior (mean, sd) of the GROSS Sharpe ratio for arbitrary hypothesis shapes."""
        h = self.hyper
        m = len(shapes)
        if self._L is None or m == 0:
            return np.full(m, h.m0), np.full(m, h.s0)
        obs_shapes = [self.shape(x) for x in self._obs_hids]
        kt = h.s0 ** 2 * np.array([[similarity(s, o) for o in obs_shapes] for s in shapes], dtype=float)
        mean = h.m0 + kt @ self._alpha
        v = _solve(self._L, kt.T)
        var = h.s0 ** 2 - np.sum(kt * v.T, axis=1)
        return mean, np.sqrt(np.clip(var, 1e-8, None))

    # ------------------------------------------------------------------ planned tests (the RL critic)
    @staticmethod
    def planned_variance(years: float) -> tuple[float, float]:
        """(total, correlated part) of the noise variance of a planned run of ``years``. Lo's variance is
        taken at Sharpe 0, so it does not depend on any outcome."""
        y = max(float(years), 0.25)
        corr = 1.0 / y + REGIME_SD ** 2 / y
        return corr + IMPL_SD ** 2, corr

    def joint_planned(self, latent: list[HypothesisShape], planned: list[PlannedTest]) -> JointGaussian:
        """Joint posterior of the gross Sharpe ratios of ``latent`` and the gross Sharpe a run of each
        ``planned`` test would report, given every observation so far.

        Planned noise is correlated, through the SIGNED kernel, with existing runs whose months it
        shares (similarity x n_common / sqrt(n_i n_c) x sd_i sd_c) and with the other planned tests
        (similarity x correlated variance). So a re-test of the same years, a near-copy or a mirror of
        a planned test carries little new information. Trading-cost uncertainty is ignored here."""
        h = self.hyper
        s2 = h.s0 ** 2
        L, P = len(latent), len(planned)
        shapes = list(latent) + [p.shape for p in planned]
        K = s2 * np.array([[similarity(a, b) for b in shapes] for a in shapes], dtype=float).reshape(L + P, L + P)
        pv = [self.planned_variance(p.years) for p in planned]
        for i in range(P):
            for j in range(P):
                if i == j:
                    K[L + i, L + j] += pv[i][0]
                else:
                    common = len(set(planned[i].months) & set(planned[j].months))
                    ov = common / math.sqrt(max(len(planned[i].months), 1) * max(len(planned[j].months), 1))
                    K[L + i, L + j] += similarity(planned[i].shape, planned[j].shape) * ov * math.sqrt(pv[i][1] * pv[j][1])
        mean = np.full(L + P, h.m0)
        labels = [f"g{i}" for i in range(L)] + [f"y{i}" for i in range(P)]
        if self._L is None:
            return JointGaussian(labels, mean, _psd_cov(K), L)
        obs_shapes = [self.shape(x) for x in self._obs_hids]
        Kzo = s2 * np.array([[similarity(z, o) for o in obs_shapes] for z in shapes], dtype=float).reshape(L + P, len(obs_shapes))
        for i, p in enumerate(planned):
            pm = set(p.months)
            for j, o in enumerate(self._obs):
                if not o.months or o.var <= 0:
                    continue
                common = len(pm & set(o.months))
                if not common:
                    continue
                ov = common / math.sqrt(max(len(p.months), 1) * len(o.months))
                rho = max(-1.0, min(1.0, similarity(p.shape, obs_shapes[j]) * ov))
                Kzo[L + i, j] += rho * math.sqrt(o.var * pv[i][1])
        mean = h.m0 + Kzo @ self._alpha
        V = _solve(self._L, Kzo.T)
        cov = K - Kzo @ V
        return JointGaussian(labels, mean, _psd_cov(cov), L)

    def smoother_row(self, shape: HypothesisShape) -> tuple[np.ndarray, list[str | None], np.ndarray]:
        """``(w, evidence_ids, resid)`` with posterior gross mean = m0 + w @ resid: how much each observation
        moves the belief about ``shape`` (used to share a finalist's exam result among the tests behind it)."""
        if self._L is None:
            return np.zeros(0), [], np.zeros(0)
        h = self.hyper
        obs_shapes = [self.shape(x) for x in self._obs_hids]
        k = h.s0 ** 2 * np.array([similarity(shape, o) for o in obs_shapes], dtype=float)
        w = _solve(self._L, k)
        return w, [o.evidence_id for o in self._obs], self._y - h.m0

    def predict_net(self, shapes: list[HypothesisShape], costs_bps: list[float | None]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(net mean, net sd, gross mean, cost) for each shape traded at the given cost level."""
        g, gsd = self.predict_gross(shapes)
        costs = [self.cost_estimate(s, b) for s, b in zip(shapes, costs_bps)]
        c = np.array([x[0] for x in costs], dtype=float).reshape(len(shapes))
        csd = np.array([x[1] for x in costs], dtype=float).reshape(len(shapes))
        return g - c, np.sqrt(gsd ** 2 + csd ** 2), g, c

    def predict_spec(self, spec) -> tuple[float, float]:
        """Net (mean, sd) for a spec at its own ``costs_bps``."""
        s = spec_from_any(spec)
        mean, sd, _, _ = self.predict_net([hypothesis_shape(s)], [s.costs_bps])
        return float(mean[0]), float(sd[0])

    def predict_ids(self, hids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Net (mean, sd) for hypotheses at the cost level of their first test."""
        mean, sd, _, _ = self.predict_net([self.shape(x) for x in hids], [self._hyps[x].spec.get("costs_bps") for x in hids])
        return mean, sd

    # ------------------------------------------------------------------ summaries
    def tested(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.view.evidence():
            if e.hypothesis_id in self._hyps:
                seen.setdefault(e.hypothesis_id, None)
        return list(seen)

    def belief(self, hid: str) -> Belief:
        return self.beliefs([hid])[0]

    def beliefs(self, hids: list[str] | None = None) -> list[Belief]:
        hids = list(hids) if hids is not None else self.tested()
        if not hids:
            return []
        mean, sd, gross, cost = self.predict_net([self.shape(x) for x in hids],
                                                 [self._hyps[x].spec.get("costs_bps") for x in hids])
        return [self._belief(h, float(a), float(b), float(g), float(c)) for h, a, b, g, c in zip(hids, mean, sd, gross, cost)]

    def _belief(self, hid: str, mean: float, sd: float, gross: float, cost: float) -> Belief:
        ev = self.view.evidence_for(hid)
        own = [e for e in ev if math.isfinite(e.sharpe)]
        oos = [e for e in own if e.out_of_sample]
        oos_sharpe = None
        if oos:
            w = np.array([1.0 / max(e.se ** 2 + REGIME_SD ** 2 / max(e.years, 0.25), 1e-9) for e in oos])
            oos_sharpe = float(np.sum(w * np.array([e.sharpe for e in oos])) / np.sum(w))
        p_pos = normal_cdf(mean / sd) if sd > 0 else float(mean > 0)
        if not own:
            status = "untested"
        elif p_pos >= 0.9:
            status = "supported"
        elif p_pos <= 0.1:
            status = "refuted"
        else:
            status = "uncertain"
        hyp = self._hyps.get(hid)
        return Belief(
            hypothesis_id=hid,
            statement=hyp.statement if hyp else hid,
            mean=mean,
            sd=sd,
            p_positive=p_pos,
            p_hurdle=normal_cdf((mean - self.hurdle) / sd) if sd > 0 else float(mean > self.hurdle),
            hurdle=self.hurdle,
            gross_mean=gross,
            cost=cost,
            n_evidence=len(own),
            years=_union_years(own),
            oos_years=_union_years(oos),
            oos_sharpe=oos_sharpe,
            last_sharpe=own[-1].sharpe if own else None,
            status=status,  # type: ignore[arg-type]
        )

    def hypothesis(self, hid: str) -> Hypothesis | None:
        return self._hyps.get(hid)

    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hyps.values())


def _psd_cov(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    if w.min() >= 0:
        return c
    return (v * np.clip(w, 0.0, None)) @ v.T


def bayes_hash() -> str:
    """Fingerprint of everything that defines the belief model (noise, priors, cost prior, kernel), so
    logged rewards can be matched to the model that computed them."""
    import hashlib
    import json

    from aitrading.beliefs import semantics

    blob = json.dumps({
        "regime": REGIME_SD, "impl": IMPL_SD, "m0": DEFAULT_PRIOR_MEAN, "s0": DEFAULT_PRIOR_SD,
        "hyper": [HYPER_MEAN_SD, HYPER_LOGSD_SD], "cost_prior": COST_PRIOR, "cost_w": COST_PRIOR_WEIGHT,
        "bandwidth": semantics.HORIZON_BANDWIDTH, "family_share": semantics.FAMILY_SHARE,
        "default_share": semantics.DEFAULT_FAMILY_SHARE, "kind": {f"{a}|{b}": v for (a, b), v in semantics._KIND_SIM.items()},
    }, sort_keys=True)
    return "b-" + hashlib.sha1(blob.encode()).hexdigest()[:12]


def _union_years(ev: list[Evidence]) -> float:
    """Years covered by the union of the evidence windows (overlapping runs counted once)."""
    spans = sorted((e.start, e.end) for e in ev if e.end >= e.start)
    total = 0
    cur_s = cur_e = None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += (cur_e - cur_s).days
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += (cur_e - cur_s).days
    return total / 365.25
