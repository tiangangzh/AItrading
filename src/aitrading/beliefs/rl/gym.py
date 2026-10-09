"""A ground-truth gym: simulated markets whose true Sharpe ratios are known, with a geometry that is
deliberately NOT the critic's.

The critic values ideas through the catalog kernel, and an actor could learn to exploit the kernel's
blind spots instead of finding real edges. Here the truth is defined differently. The true gross
Sharpe of a hypothesis is m0 + s0 x <w, e(h)>, where e is a hidden embedding:
* ``filter_blind`` - filters do nothing at all;
* ``twins`` - chosen cross-family feature pairs are the same underlying factor.

``w`` is seeded per coordinate. Runs return monthly returns made of:
* the true net drift;
* shared factor shocks per hidden coordinate, so related or twin strategies co-move;
* idiosyncratic noise.

``decay_after`` zeroes every alpha after a date, so in-sample winners fail the exam: the
claw-back test. Everything is deterministic per (seed, hypothesis, month). It is used to test the
loop, and to audit a trained actor, never as evidence about real markets.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import numpy as np

from aitrading.backtest.models import BacktestResult, PerformanceStats
from aitrading.beliefs.semantics import FEATURE_SEMANTICS, hypothesis_id
from aitrading.strategy.spec import StrategySpec

__all__ = ["HiddenWorld", "HiddenWorldRunner", "TWIN_PAIRS"]

#: Cross-family pairs the 'twins' world treats as ONE factor (the critic's kernel says they are unrelated).
TWIN_PAIRS = [("book_to_market", "return_6m_pct"), ("gross_profitability_pct", "volatility_60d_pct"),
              ("asset_growth_yoy_pct", "rel_volume_20d")]
_TURNOVER = {"daily": 60.0, "weekly": 120.0, "monthly": 40.0, "quarterly": 25.0, "annual": 15.0}
_REB = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}


def _seeded(seed: int, key: str) -> np.random.Generator:
    h = int(hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def _month_ends(start: date, end: date) -> list[date]:
    out, y, m = [], start.year, start.month
    while True:
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        d = nxt - timedelta(days=1)
        if d > end:
            break
        if d >= start:
            out.append(d)
        y, m = nxt.year, nxt.month
    return out


class HiddenWorld:
    def __init__(self, seed: int, *, kernel: Literal["default", "filter_blind", "twins"] = "twins", m0: float = 0.0,
                 s0: float = 0.5, decay_after: date | None = None):
        self.seed, self.kernel, self.m0, self.s0, self.decay_after = seed, kernel, m0, s0, decay_after
        self._twin = {}
        for a, b in TWIN_PAIRS:
            self._twin[a] = self._twin[b] = f"twin:{a}+{b}"

    def coordinate(self, feature: str) -> str:
        if self.kernel == "twins" and feature in self._twin:
            return self._twin[feature]
        return f"feat:{feature}"

    def embedding(self, spec: StrategySpec) -> dict[str, float]:
        vec: dict[str, float] = {}
        total = sum(c.weight for c in spec.signal) or 1.0
        for c in spec.signal:
            sem = FEATURE_SEMANTICS.get(c.feature)
            sign = (sem.sign if sem else 1.0) * (1.0 if c.direction == "higher_is_better" else -1.0)
            k = self.coordinate(c.feature)
            vec[k] = vec.get(k, 0.0) + sign * c.weight / total
        if self.kernel != "filter_blind":
            for f in spec.filters:
                vec[f"filter:{f.describe()}"] = vec.get(f"filter:{f.describe()}", 0.0) + 0.3
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def weight(self, coord: str) -> float:
        return float(_seeded(self.seed, f"w|{coord}").standard_normal())

    def true_gross(self, spec: StrategySpec) -> float:
        e = self.embedding(spec)
        return self.m0 + self.s0 * sum(v * self.weight(k) for k, v in e.items())

    def cost_sharpe(self, spec: StrategySpec, vol_pct: float = 15.0) -> float:
        return 2 * _TURNOVER.get(spec.rebalance, 40.0) * _REB.get(spec.rebalance, 12) * spec.costs_bps / 1e4 / vol_pct

    def true_net(self, spec: StrategySpec) -> float:
        return self.true_gross(spec) - self.cost_sharpe(spec)


class HiddenWorldRunner:
    """``backtest(spec)`` -> a BacktestResult with monthly returns from the hidden world."""

    provider_name = "hidden-world"

    def __init__(self, world: HiddenWorld, *, vol: float = 15.0, idio: float = 0.5):
        self.world, self.vol, self.idio = world, vol, idio
        self.calls: list[StrategySpec] = []

    def _shock(self, coord: str, month: str) -> float:
        return float(_seeded(self.world.seed, f"f|{coord}|{month}").standard_normal())

    def backtest(self, spec: StrategySpec, *, label: str | None = None) -> BacktestResult:
        self.calls.append(spec)
        start = spec.start or date(2012, 1, 1)
        end = spec.end or date(2026, 9, 30)
        dates = _month_ends(start, end)
        e = self.world.embedding(spec)
        hid = hypothesis_id(spec)
        mvol = self.vol / 100.0 / math.sqrt(12)
        gross, cost = self.world.true_gross(spec), self.world.cost_sharpe(spec, self.vol)
        rets = []
        for d in dates:
            key = f"{d.year:04d}-{d.month:02d}"
            alive = self.world.decay_after is None or d < self.world.decay_after
            drift = ((gross if alive else 0.0) - cost) * mvol / math.sqrt(12)
            common = sum(v * self._shock(k, key) for k, v in e.items())
            idio = float(_seeded(self.world.seed, f"i|{hid}|{key}").standard_normal())
            shock = math.sqrt(1 - self.idio ** 2) * common + self.idio * idio
            rets.append(drift + mvol * shock)
        r = np.array(rets)
        sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if len(r) > 2 and r.std() > 0 else 0.0
        st = PerformanceStats(
            label="strategy", start=dates[0], end=dates[-1], n_periods=len(dates), periods_per_year=12.0,
            total_return_pct=float((np.prod(1 + r) - 1) * 100), cagr_pct=0.0, volatility_pct=self.vol, sharpe=sharpe,
            sortino=None, max_drawdown_pct=-10.0, max_drawdown_duration_periods=1, calmar=None, hit_rate_pct=50.0,
            best_period_pct=float(r.max() * 100), worst_period_pct=float(r.min() * 100), skew=None, excess_kurtosis=None,
            mean_return_t_stat=None, avg_turnover_pct=_TURNOVER.get(spec.rebalance, 40.0))
        rid = hashlib.sha1(f"{self.world.seed}|{hid}|{start}|{end}|{label}".encode()).hexdigest()[:10]
        return BacktestResult(run_id=f"{spec.name}-{rid}", idea=spec.idea, spec=spec.model_dump(mode="json"),
                              provider=self.provider_name, llm="none", start=dates[0], end=dates[-1], rebalance=spec.rebalance,
                              returns={"strategy": list(map(float, r))}, dates=dates, stats={"strategy": st},
                              started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
