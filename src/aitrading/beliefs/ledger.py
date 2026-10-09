"""The hypothesis ledger: every hypothesis tested in a market, every result (failures included) and
every forecast a judge made before seeing a result.

Why every result: Bayesian updating handles "I tried 20 variants and kept the best" correctly only
if the 19 failures are in the data too. The ledger records each backtest automatically, so the
belief engine conditions on all of them and the research loop can count its own trials.

Layout: ``<root>/<market slug>/ledger.json`` (one JSON document, written atomically), where
``root`` is ``BeliefLedger(root=...)`` > ``$AITRADING_HOME/beliefs`` > ``~/.aitrading/beliefs`` and
the market is the data provider's name (``free``, ``synthetic``, ``bloomberg`` ...): evidence from one
market never moves beliefs about another (a simulated market says nothing about the real one).

Evidence identity: ``backtest:<run_id>`` / ``holdout:<run_id>`` (the run id hashes the spec, the data
and the window, so re-running the same experiment replaces its record instead of counting it twice)
and ``paper:<strategy slug>:<start date>`` (the paper track record grows in place).
"""

from __future__ import annotations

import calendar
import math
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aitrading.backtest.models import BacktestResult
from aitrading.beliefs.models import (
    SCHEMA_VERSION,
    Evidence,
    EvidenceKind,
    HoldoutUse,
    Hypothesis,
    LedgerData,
    LLMPrior,
    Prediction,
)
from aitrading.beliefs.semantics import (
    hypothesis_id,
    hypothesis_key,
    hypothesis_statement,
    primary_concept,
    spec_from_any,
)
from aitrading.strategy.spec import StrategySpec
from aitrading.trading.store import ENV_HOME, StoreError, read_json, slugify_strategy_name, write_json_atomic

__all__ = [
    "BeliefLedger",
    "CutoffLedger",
    "LedgerView",
    "as_view",
    "default_beliefs_root",
    "sharpe_se",
    "evidence_from_backtest",
    "evidence_from_paper",
    "MIN_PERIODS",
]

#: A run needs at least this many return periods (and a Sharpe ratio) to count as evidence.
MIN_PERIODS = 20
_CAVEAT_KEYS = ("SURVIVORSHIP", "NOT POINT-IN-TIME", "LOOK-AHEAD", "END-DATE LABELS")


def default_beliefs_root() -> Path:
    home = os.environ.get(ENV_HOME, "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".aitrading"
    return root / "beliefs"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sharpe_se(sharpe: float, n_periods: int, periods_per_year: float) -> float:
    """Standard error of an annualised Sharpe ratio estimated from ``n_periods`` iid returns (Lo 2002):
    sqrt(ppy x (1 + s^2 / 2) / n) with s the per-period Sharpe. About 1 / sqrt(years) for small Sharpes."""
    if n_periods <= 1 or periods_per_year <= 0:
        return float("inf")
    s = sharpe / math.sqrt(periods_per_year)
    return math.sqrt(periods_per_year * (1.0 + 0.5 * s * s) / n_periods)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def evidence_from_backtest(spec: StrategySpec | dict, result: BacktestResult, *, kind: EvidenceKind = "backtest",
                           out_of_sample: bool = False, recorded_at: datetime | None = None) -> Evidence | None:
    """Evidence from a backtest result, or None when the run has no usable Sharpe ratio."""
    st = result.stats.get("strategy")
    if st is None:
        return None
    sharpe = _finite(st.sharpe)
    if sharpe is None or st.n_periods < MIN_PERIODS or st.periods_per_year <= 0:
        return None
    monthly: dict[str, float] = {}
    for d, r in zip(result.dates, result.returns.get("strategy") or []):
        v = _finite(r)
        if v is not None:
            monthly[f"{d.year:04d}-{d.month:02d}"] = v
    spec_d = spec.model_dump(mode="json") if isinstance(spec, StrategySpec) else dict(spec)
    caveats = [w[:200] for w in result.warnings if any(k in w.upper() for k in _CAVEAT_KEYS)][:3]
    return Evidence(
        cost_sharpe=_cost_sharpe(result, st, _finite(spec_d.get("costs_bps"))),
        evidence_id=f"{kind}:{result.run_id}",
        hypothesis_id=hypothesis_id(spec_d),
        kind=kind,
        run_id=result.run_id,
        start=st.start,
        end=st.end,
        years=st.n_periods / st.periods_per_year,
        sharpe=sharpe,
        se=sharpe_se(sharpe, st.n_periods, st.periods_per_year),
        n_periods=st.n_periods,
        periods_per_year=st.periods_per_year,
        mean_t=_finite(st.mean_return_t_stat),
        alpha_t=_finite(result.regression.alpha_t_stat) if result.regression is not None else None,
        costs_bps=_finite(spec_d.get("costs_bps")),
        monthly=monthly,
        out_of_sample=out_of_sample,
        recorded_at=recorded_at or _utcnow(),
        caveats=caveats,
    )


def _cost_sharpe(result: BacktestResult, st: Any, costs_bps: float | None) -> float | None:
    """Annual trading-cost drag / annual volatility (both in %), from the run's turnover; None if unknown."""
    from aitrading.strategy.interpret import annual_cost_drag_pct, rebalances_per_year

    turnover, vol, reb = _finite(st.avg_turnover_pct), _finite(st.volatility_pct), rebalances_per_year(result)
    if turnover is None or vol is None or vol <= 0 or reb is None or costs_bps is None:
        return None
    return annual_cost_drag_pct(turnover, reb, costs_bps) / vol


def _as_date(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def evidence_from_paper(spec: StrategySpec | dict, strategy_slug: str, summary: dict[str, Any], *,
                        min_months: int = 3, recorded_at: datetime | None = None) -> Evidence | None:
    """Forward (out-of-sample) evidence from a paper account's ``summary()``; None before
    ``min_months`` complete calendar months of track record.

    Monthly returns are compounded from the NAV marks (the last mark of each month; the first month
    starts from the initial capital). The Sharpe ratio is on the paper returns as they are: paper
    cash earns 0%, so for books that hold cash it is not in excess of the risk-free rate (a caveat
    says so)."""
    started = summary.get("started")
    history = summary.get("nav_history") or []
    capital = _finite(summary.get("initial_capital"))
    if not started or not history or capital is None or capital <= 0:
        return None
    month_end: dict[str, float] = {}
    last_day: date | None = None
    for d, nav in history:
        v = _finite(nav)
        if v is None or v <= 0:
            continue
        dd = _as_date(d)
        month_end[f"{dd.year:04d}-{dd.month:02d}"] = v
        last_day = dd
    keys = sorted(month_end)
    if last_day is None or not keys:
        return None
    # the current month counts only once it has ended (a partial month is not a monthly return)
    today_key = f"{last_day.year:04d}-{last_day.month:02d}"
    complete = [k for k in keys if k != today_key]
    if len(complete) < min_months:
        return None
    monthly: dict[str, float] = {}
    prev = capital
    for k in complete:
        monthly[k] = month_end[k] / prev - 1.0
        prev = month_end[k]
    r = np.array(list(monthly.values()), dtype=float)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if not (sd > 0):
        return None
    sharpe = float(r.mean() / sd * math.sqrt(12.0))
    spec_d = spec.model_dump(mode="json") if isinstance(spec, StrategySpec) else dict(spec)
    s = spec_from_any(spec_d)
    caveats = []
    if not (s.kind == "cross_sectional" and s.portfolio.style == "long_short"):
        caveats.append("paper cash earns 0%: this Sharpe ratio is on raw returns, not in excess of the risk-free rate")
    first = _as_date(started)
    y, m = (int(p) for p in complete[-1].split("-"))
    end = date(y, m, calendar.monthrange(y, m)[1])
    return Evidence(
        evidence_id=f"paper:{strategy_slug}:{first.isoformat()}",
        hypothesis_id=hypothesis_id(spec_d),
        kind="paper",
        run_id=None,
        start=first,
        end=end,
        years=len(r) / 12.0,
        sharpe=sharpe,
        se=sharpe_se(sharpe, len(r), 12.0),
        n_periods=len(r),
        periods_per_year=12.0,
        costs_bps=_finite(spec_d.get("costs_bps")),
        monthly=monthly,
        out_of_sample=True,
        recorded_at=recorded_at or _utcnow(),
        caveats=caveats,
    )


def _market_slug(market: str) -> str:
    try:
        return slugify_strategy_name(market or "default")
    except ValueError:
        return "default"


class BeliefLedger:
    """Read / write the ledger of one market (see the module docstring)."""

    def __init__(self, market: str, root: Path | str | None = None, *, clock: Any = None):
        self.market = str(market or "default")
        self.root = Path(root) if root is not None else default_beliefs_root()
        self._clock = clock or _utcnow
        self._data: LedgerData | None = None

    def __repr__(self) -> str:
        return f"BeliefLedger(market={self.market!r}, path={str(self.path)!r})"

    @property
    def path(self) -> Path:
        return self.root / _market_slug(self.market) / "ledger.json"

    def now(self) -> datetime:
        return self._clock()

    # ------------------------------------------------------------------ I/O
    @property
    def data(self) -> LedgerData:
        if self._data is None:
            self._data = self._read()
        return self._data

    def _read(self) -> LedgerData:
        try:
            raw = read_json(self.path)
        except FileNotFoundError:
            return LedgerData(market=self.market)
        if not isinstance(raw, dict):
            raise StoreError(f"{self.path} is not a ledger (expected a JSON object)")
        ver = raw.get("schema_version", SCHEMA_VERSION)
        if isinstance(ver, int) and ver > SCHEMA_VERSION:
            raise StoreError(f"{self.path} was written by a newer version of aitrading (schema {ver}); upgrade to read it")
        try:
            return LedgerData.model_validate(raw)
        except ValueError as e:
            raise StoreError(f"{self.path} is not a valid ledger ({str(e)[:300]}); fix or delete the file") from e

    def save(self) -> None:
        write_json_atomic(self.path, self.data.model_dump(mode="json"))

    def reload(self) -> None:
        self._data = None

    # ------------------------------------------------------------------ queries
    def hypotheses(self) -> list[Hypothesis]:
        return list(self.data.hypotheses.values())

    def hypothesis(self, hid: str) -> Hypothesis | None:
        h = self.data.hypotheses.get(hid)
        if h is None and hid:  # allow unambiguous prefixes ("h-9b38")
            hits = [k for k in self.data.hypotheses if k.startswith(hid)]
            if len(hits) == 1:
                h = self.data.hypotheses[hits[0]]
        return h

    def evidence(self, kinds: Iterable[str] | None = None) -> list[Evidence]:
        ev = sorted(self.data.evidence.values(), key=lambda e: (e.recorded_at, e.evidence_id))
        if kinds is not None:
            ks = set(kinds)
            ev = [e for e in ev if e.kind in ks]
        return ev

    def evidence_for(self, hid: str) -> list[Evidence]:
        return [e for e in self.evidence() if e.hypothesis_id == hid]

    def predictions(self) -> list[Prediction]:
        return list(self.data.predictions)

    def tested_ids(self) -> set[str]:
        return {e.hypothesis_id for e in self.data.evidence.values()}

    # ------------------------------------------------------------------ writes
    def add_hypothesis(self, spec: StrategySpec | dict, *, statement: str | None = None, mechanism: str = "",
                       source: str = "user", save: bool = True) -> Hypothesis:
        """Register the claim behind ``spec`` (idempotent; a later call can add a mechanism / name)."""
        s = spec_from_any(spec)
        hid = hypothesis_id(s)
        h = self.data.hypotheses.get(hid)
        if h is None:
            fam, sign, hor = primary_concept(s)
            h = Hypothesis(
                hypothesis_id=hid,
                statement=(statement or "").strip() or hypothesis_statement(s),
                mechanism=mechanism.strip(),
                concept=f"{fam}{'+' if sign >= 0 else '-'}",
                horizon_days=hor,
                spec=s.model_dump(mode="json"),
                key=hypothesis_key(s),
                source=source,
                names=[s.name] if s.name else [],
                created_at=self.now(),
            )
            self.data.hypotheses[hid] = h
        else:
            if mechanism.strip() and not h.mechanism:
                h.mechanism = mechanism.strip()
            if s.name and s.name not in h.names:
                h.names.append(s.name)
        if save:
            self.save()
        return h

    def set_llm_prior(self, hid: str, prior: LLMPrior, *, save: bool = True) -> None:
        h = self.data.hypotheses.get(hid)
        if h is None:
            raise KeyError(hid)
        h.llm_prior = prior
        if save:
            self.save()

    def add_evidence(self, ev: Evidence, *, save: bool = True) -> Evidence:
        """Insert or replace evidence by id (re-running an experiment replaces its record)."""
        if ev.hypothesis_id not in self.data.hypotheses:
            raise KeyError(f"unknown hypothesis {ev.hypothesis_id}: add_hypothesis() first")
        self.data.evidence[ev.evidence_id] = ev
        if save:
            self.save()
        return ev

    def record_backtest(self, spec: StrategySpec | dict, result: BacktestResult, *, kind: EvidenceKind = "backtest",
                        out_of_sample: bool | None = None, statement: str | None = None, mechanism: str = "",
                        source: str = "user", save: bool = True) -> Evidence | None:
        """Register the hypothesis and record the run (None when the run has no usable Sharpe ratio).

        ``out_of_sample`` defaults to True for holdout runs and for backtests whose data start after
        the hypothesis was first registered."""
        h = self.add_hypothesis(spec, statement=statement, mechanism=mechanism, source=source, save=False)
        ev = evidence_from_backtest(spec, result, kind=kind, out_of_sample=False, recorded_at=self.now())
        if ev is None:
            if save:
                self.save()
            return None
        oos = out_of_sample if out_of_sample is not None else (kind == "holdout" or ev.start > h.created_at.date())
        ev = ev.model_copy(update={"out_of_sample": bool(oos)})
        return self.add_evidence(ev, save=save)

    def record_paper(self, spec: StrategySpec | dict, strategy_slug: str, summary: dict[str, Any], *,
                     save: bool = True) -> Evidence | None:
        ev = evidence_from_paper(spec, strategy_slug, summary, recorded_at=self.now())
        if ev is None:
            return None
        self.add_hypothesis(spec, source="paper", save=False)
        return self.add_evidence(ev, save=save)

    def add_prediction(self, *, hypothesis_id: str, judge: str, purpose: EvidenceKind, mean: float, sd: float,
                       p_positive: float, note: str = "", save: bool = True) -> Prediction:
        pred = Prediction(
            prediction_id="p-" + uuid.uuid4().hex[:12],
            hypothesis_id=hypothesis_id,
            judge=judge,
            made_at=self.now(),
            purpose=purpose,
            mean=float(mean),
            sd=float(sd),
            p_positive=float(p_positive),
            context_evidence=sorted(self.data.evidence),
            note=note,
        )
        self.data.predictions.append(pred)
        if save:
            self.save()
        return pred

    def resolve_prediction(self, prediction_id: str, evidence_id: str, *, save: bool = True) -> None:
        for p in self.data.predictions:
            if p.prediction_id == prediction_id:
                if evidence_id in p.context_evidence:
                    raise ValueError(f"{evidence_id} was visible when {prediction_id} was made; it cannot resolve it")
                p.target_evidence_id = evidence_id
                break
        else:
            raise KeyError(prediction_id)
        if save:
            self.save()

    def note_holdout_use(self, start: date, end: date, *, save: bool = True) -> int:
        """Record one more look at the holdout window [start, end]; returns how often it was used BEFORE."""
        key = f"{start.isoformat()}..{end.isoformat()}"
        use = self.data.holdouts.get(key) or HoldoutUse(start=start, end=end)
        before = use.uses
        use.uses += 1
        use.last_used = self.now()
        self.data.holdouts[key] = use
        if save:
            self.save()
        return before

    def holdout_uses_overlapping(self, start: date, end: date) -> int:
        return sum(u.uses for u in self.data.holdouts.values() if u.start <= end and start <= u.end)

    def forget(self, hid: str, *, save: bool = True) -> int:
        """Remove a hypothesis with its evidence and predictions; returns the number of evidence records removed."""
        h = self.hypothesis(hid)
        if h is None:
            raise KeyError(hid)
        gone = [k for k, e in self.data.evidence.items() if e.hypothesis_id == h.hypothesis_id]
        for k in gone:
            del self.data.evidence[k]
        self.data.predictions = [p for p in self.data.predictions if p.hypothesis_id != h.hypothesis_id]
        del self.data.hypotheses[h.hypothesis_id]
        if save:
            self.save()
        return len(gone)

    # ------------------------------------------------------------------ point-in-time views
    def view(self, *, before: datetime | None = None, evidence_filter: Any = None) -> "LedgerView":
        """Hypotheses and evidence as they were before ``before`` (recorded_at strictly earlier),
        optionally filtered further by ``evidence_filter(evidence) -> bool``."""
        ev = [e for e in self.evidence() if before is None or e.recorded_at < before]
        if evidence_filter is not None:
            ev = [e for e in ev if evidence_filter(e)]
        ids = {e.hypothesis_id for e in ev}
        hyps = [h for h in self.hypotheses() if h.hypothesis_id in ids or before is None or h.created_at < before]
        return LedgerView(market=self.market, hypotheses=hyps, evidence=ev)


class LedgerView:
    """An immutable slice of a ledger (what the belief model is fitted on)."""

    def __init__(self, *, market: str, hypotheses: list[Hypothesis], evidence: list[Evidence]):
        self.market = market
        self._hyps = {h.hypothesis_id: h for h in hypotheses}
        self._ev = list(evidence)

    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hyps.values())

    def hypothesis(self, hid: str) -> Hypothesis | None:
        return self._hyps.get(hid)

    def evidence(self, kinds: Iterable[str] | None = None) -> list[Evidence]:
        if kinds is None:
            return list(self._ev)
        ks = set(kinds)
        return [e for e in self._ev if e.kind in ks]

    def evidence_for(self, hid: str) -> list[Evidence]:
        return [e for e in self._ev if e.hypothesis_id == hid]

    def tested_ids(self) -> set[str]:
        return {e.hypothesis_id for e in self._ev}


class CutoffLedger:
    """A ledger whose reads only see evidence whose DATA ENDED before ``cutoff`` (writes go through).

    The research loop works through one of these, with ``cutoff`` = the holdout start. Every belief,
    digest, forecast and finalist choice made during the research rounds is then blind to the
    holdout years. This holds even when ordinary backtests in the same ledger (which run to the
    latest data) have already covered them.
    """

    def __init__(self, base: BeliefLedger, cutoff: date):
        self.base = base
        self.cutoff = cutoff

    def __repr__(self) -> str:
        return f"CutoffLedger({self.base!r}, cutoff={self.cutoff.isoformat()})"

    def _visible(self, e: Evidence) -> bool:
        return e.end < self.cutoff

    def view(self, *, before: datetime | None = None, evidence_filter: Any = None) -> "LedgerView":
        if evidence_filter is None:
            return self.base.view(before=before, evidence_filter=self._visible)
        return self.base.view(before=before, evidence_filter=lambda e: self._visible(e) and evidence_filter(e))

    def evidence(self, kinds: Iterable[str] | None = None) -> list[Evidence]:
        return [e for e in self.base.evidence(kinds) if self._visible(e)]

    def evidence_for(self, hid: str) -> list[Evidence]:
        return [e for e in self.base.evidence_for(hid) if self._visible(e)]

    def tested_ids(self) -> set[str]:
        return {e.hypothesis_id for e in self.evidence()}

    def __getattr__(self, name: str) -> Any:  # writes, hypotheses, predictions, market, path ...
        return getattr(self.base, name)


def as_view(source: Any) -> LedgerView:
    """A :class:`LedgerView` of a ledger, a :class:`CutoffLedger` or a view (returned as is)."""
    return source if isinstance(source, LedgerView) else source.view()

