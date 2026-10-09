"""Rewards, advantages and the champion / challenger test: pure functions, pinned by ``RewardSpec.version``.

What a proposal is paid (the actor's step reward):
    r = trust x kappa x new_information - c_test - c_complex x complexity_excess

* ``kappa`` - the critic's batch-marginal, robust knowledge gradient: how much testing the
  proposal is EXPECTED to raise the value of the best diverse set of strategies above the hurdle,
  given the proposals picked before it. It is computed before the test, from the critic alone.
* ``new_information`` (rho) - the share of the run's monthly returns not explained by runs already
  tested (1 - adjusted R^2). It is invariant to shifting and rescaling the returns, so it never
  depends on the observed Sharpe ratio. Untested proposals get an imputed value.
* ``trust`` - 1, or 0.5 in a concept where the critic's forecasts have been significantly too
  optimistic. It can only lower rewards.

Never in any reward: the observed in-sample Sharpe, surprise, the realised change in value, the
actor's own forecast, its wording, LLM priors. Out-of-sample results enter only as the standardised
TD error at the holdout exam (and in paper trading), via the policy store's veto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "RewardSpec",
    "step_reward",
    "penalty_for",
    "advantages",
    "imputed_rho",
    "arm_score",
    "pair_outcome",
    "Sprt",
    "PENALISED",
    "INVALID",
]

INVALID = ("unparseable", "unknown_feature", "filter_not_in_menu", "too_many_components", "style_not_allowed")
PENALISED = INVALID + ("duplicate_in_arm", "retest_same_window", "near_duplicate")


class RewardSpec(BaseModel):
    """Every constant of the reward (version 'rv1'); pinned on every logged record."""

    model_config = ConfigDict(frozen=True)

    version: str = "rv1"
    eta: float = 0.3  # net Sharpe hurdle a finalist must clear to have value
    k_fin: int = 2  # finalists that sit the exam
    conflict_sim: float = 0.6  # |kernel similarity| at which two hypotheses are the same bet
    twin_corr: float = 0.6  # |monthly return correlation| at which two runs are the same bet
    arms_max: int = 30  # tested hypotheses the value function considers
    gh_nodes: int = 32  # Gauss-Hermite nodes of the knowledge gradient
    c_test: float = 0.005  # cost of one backtest (a test must be worth at least this)
    c_invalid: float = 0.01
    c_dup: float = 0.005
    c_complex: float = 0.003
    c_quota: float = 0.005
    near_dup_sim: float = 0.95
    novel_sim: float = 0.3
    rho_min_finalist: float = 0.3
    rho_prior: float = 0.85
    rho_prior_w: float = 2.0
    tau_low: float = 0.5
    trust_min_n: int = 5
    trust_z: float = -2.0
    z_clip: float = 3.0
    tie_margin: float = 0.002
    sprt_p1: float = 0.75
    sprt_alpha: float = 0.10
    sprt_beta: float = 0.20
    sprt_max_decisive: int = 12
    veto_min_n: int = 4
    veto_z: float = -2.0


def step_reward(kappa: float, rho: float, tau: float, complexity_excess: int, spec: RewardSpec) -> float:
    return tau * kappa * rho - spec.c_test - spec.c_complex * max(0, int(complexity_excess))


def penalty_for(status: str, spec: RewardSpec) -> float:
    if status in INVALID:
        return -spec.c_invalid
    if status in ("duplicate_in_arm", "retest_same_window", "near_duplicate"):
        return -spec.c_dup
    return 0.0


def advantages(rewards: list[float]) -> tuple[list[float], list[float]]:
    """GRPO-style group baseline: (r - mean, (r - mean) / (sd + 0.01)) within one arm and round."""
    if not rewards:
        return [], []
    m = sum(rewards) / len(rewards)
    sd = math.sqrt(sum((r - m) ** 2 for r in rewards) / len(rewards))
    a = [r - m for r in rewards]
    return a, [x / (sd + 0.01) for x in a]


def imputed_rho(tested_rhos: list[float], spec: RewardSpec) -> float:
    return (sum(tested_rhos) + spec.rho_prior_w * spec.rho_prior) / (len(tested_rhos) + spec.rho_prior_w)


def arm_score(rewards: list[float], statuses: list[str], per_round: int, quota_deficit: int, spec: RewardSpec) -> float:
    """Sum of the best ``per_round`` rewards among valid proposals, plus every penalty, minus the quota charge."""
    valid = sorted((r for r, s in zip(rewards, statuses) if s not in PENALISED), reverse=True)[:per_round]
    pen = sum(r for r, s in zip(rewards, statuses) if s in PENALISED)
    return sum(valid) + pen - spec.c_quota * max(0, quota_deficit)


def pair_outcome(s_champion: float, s_challenger: float, spec: RewardSpec) -> Literal["win", "loss", "tie"]:
    d = s_challenger - s_champion
    if d >= spec.tie_margin:
        return "win"
    if d <= -spec.tie_margin:
        return "loss"
    return "tie"


@dataclass
class Sprt:
    """Wald sequential probability ratio test of the challenger's paired win rate: H0 p = 0.5 vs
    H1 p = ``sprt_p1``. Ties are not decisive. Error control holds under sequential looks."""

    llr: float = 0.0
    wins: int = 0
    losses: int = 0
    decisive: int = 0

    def update(self, outcome: str, spec: RewardSpec) -> Literal["promote", "reject", "continue"]:
        if outcome == "win":
            self.wins += 1
            self.decisive += 1
            self.llr += math.log(spec.sprt_p1 / 0.5)
        elif outcome == "loss":
            self.losses += 1
            self.decisive += 1
            self.llr += math.log((1.0 - spec.sprt_p1) / 0.5)
        upper = math.log((1.0 - spec.sprt_beta) / spec.sprt_alpha)
        lower = math.log(spec.sprt_beta / (1.0 - spec.sprt_alpha))
        if self.llr >= upper:
            return "promote"
        if self.llr <= lower or self.decisive >= spec.sprt_max_decisive:
            return "reject"
        return "continue"
