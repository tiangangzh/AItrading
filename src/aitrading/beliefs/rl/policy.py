"""The policy: a short, grounded research text the actor reads, and how it is allowed to change.

The policy's parameters are at most 12 *directives* (explore / exploit / avoid / formulate), each a
sentence about WHERE to look or HOW to formulate claims. A directive never contains a result
number; numbers are attached at render time from the current critic (late binding).

Changes are proposed by a reflection call (at most 3 grounded edits), which creates a *challenger*
version. On later rounds the champion and the challenger each propose half the batch on the same
observation. A Wald sequential probability ratio test on their paired scores promotes or rejects
the challenger, and rejected directive texts go on a tabu list. Clean out-of-sample exam (and paper
trading) results of finalists a version produced can veto it.

State is event-sourced: ``policy.json`` holds the versions and an append-only, idempotent event log.
``state(cutoff)`` REPLAYS the events whose ``info_horizon`` is before the cutoff, so a research run
on an earlier holdout never sees policy learned from later data. Reading never writes.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from aitrading.beliefs.grounding import directive_text_hash, ground_directive, render_safe
from aitrading.beliefs.ledger import BeliefLedger
from aitrading.beliefs.rl.models import SCHEMA_VERSION, Directive, PolicyEdit, PolicyEvent, PolicyFile, PolicyVersion
from aitrading.beliefs.rl.reward import RewardSpec, Sprt
from aitrading.trading.store import StoreError, read_json, write_json_atomic

__all__ = [
    "SEED_ID",
    "SEED_DIRECTIVES",
    "PolicyStore",
    "PolicyState",
    "GroundingContext",
    "DirectiveCredit",
    "render_policy",
    "directive_credit",
    "rl_lock",
    "rl_dir",
]

SEED_ID = "v-seed"
MAX_VERSIONS = 40
MAX_EVENTS = 2000
MAX_DIRECTIVES = 12
MAX_OPS = 3


def _seed_directive(kind: str, text: str) -> Directive:
    norm = " ".join(text.lower().split())
    did = "d-" + hashlib.sha1(f"{kind}|||{norm}".encode()).hexdigest()[:8]
    return Directive(directive_id=did, kind=kind, region=None, text=text, cites=[])  # type: ignore[arg-type]


SEED_DIRECTIVES = [
    _seed_directive("formulate", "State who is on the other side of the trade and why the effect is not arbitraged away; "
                                 "one economic claim per proposal."),
    _seed_directive("avoid", "Do not re-propose a tested claim, its mirror, or a variant that only adds filters or components."),
    _seed_directive("explore", "Spend explore proposals on concepts the experience log has not covered, starting with regions "
                               "the critic map marks as most valuable."),
    _seed_directive("formulate", "Prefer monthly or quarterly rebalancing unless the mechanism is about days; short-horizon "
                                 "signals pay heavy trading costs."),
]


def rl_dir(ledger: BeliefLedger) -> Path:
    return ledger.path.parent / "rl"


@contextmanager
def rl_lock(ledger: BeliefLedger, *, stale_after_s: float = 7200.0) -> Iterator[Path]:
    """Exclusive lock on the market's RL state (``rl/.lock``, O_EXCL). A lock older than
    ``stale_after_s`` (a crashed run) is broken; otherwise ``StoreError`` is raised."""
    d = rl_dir(ledger)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ".lock"
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > stale_after_s:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            raise StoreError(f"another research loop is running on market '{ledger.market}' ({path}); "
                             "wait for it or delete the lock file if that run crashed") from None
    else:  # pragma: no cover - the stale lock reappeared between unlink and create
        raise StoreError(f"could not take the lock {path}")
    try:
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield path
    finally:
        try:
            path.unlink()
        except OSError:
            pass


@dataclass
class PolicyState:
    """Derived by replay, never persisted."""

    champion: str = SEED_ID
    challenger: str | None = None
    sprt: Sprt = field(default_factory=Sprt)
    tabu: set[str] = field(default_factory=set)
    terminal_z: dict[str, list[float]] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)


@dataclass
class GroundingContext:
    known_ids: set[str]
    catalog_features: set[str]
    concepts: set[str]


class PolicyStore:
    def __init__(self, ledger: BeliefLedger, *, spec: RewardSpec | None = None, clock: Any = None):
        self.ledger = ledger
        self.spec = spec or RewardSpec()
        self._clock = clock or ledger.now
        self._file: PolicyFile | None = None

    @property
    def path(self) -> Path:
        return rl_dir(self.ledger) / "policy.json"

    # ------------------------------------------------------------------ I/O
    def load(self) -> PolicyFile:
        if self._file is None:
            try:
                raw = read_json(self.path)
            except FileNotFoundError:
                raw = None
            if raw is None:
                f = PolicyFile(market=self.ledger.market, reward_version=self.spec.version)
            else:
                if isinstance(raw, dict) and isinstance(raw.get("schema_version"), int) and raw["schema_version"] > SCHEMA_VERSION:
                    raise StoreError(f"{self.path} was written by a newer version of aitrading; upgrade to read it")
                try:
                    f = PolicyFile.model_validate(raw)
                except ValueError as e:
                    raise StoreError(f"{self.path} is not a valid policy file ({str(e)[:200]})") from e
            if SEED_ID not in f.versions:
                f.versions[SEED_ID] = PolicyVersion(version_id=SEED_ID, parent=None, directives=list(SEED_DIRECTIVES),
                                                    info_horizon=None, created_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
                                                    author="seed")
            self._file = f
        return self._file

    def save(self) -> None:
        self.prune()
        write_json_atomic(self.path, self.load().model_dump(mode="json"))

    def version(self, vid: str) -> PolicyVersion:
        return self.load().versions[vid]

    def _has_key(self, key: str) -> PolicyEvent | None:
        return next((e for e in self.load().events if e.key == key), None)

    def _append(self, **kw: Any) -> PolicyEvent:
        f = self.load()
        ev = PolicyEvent(seq=(f.events[-1].seq + 1) if f.events else 1, at=self._clock(), **kw)
        f.events.append(ev)
        return ev

    # ------------------------------------------------------------------ replay
    def state(self, cutoff: date | None) -> PolicyState:
        """PURE replay of the events with ``info_horizon < cutoff`` (all of them for None)."""
        f = self.load()
        st = PolicyState()
        for ev in sorted(f.events, key=lambda e: e.seq):
            if cutoff is not None and ev.info_horizon >= cutoff:
                continue
            if ev.kind == "challenger":
                if st.challenger is None and ev.champion == st.champion and ev.challenger in f.versions:
                    st.challenger, st.sprt = ev.challenger, Sprt()
            elif ev.kind == "pair":
                if st.challenger is not None and ev.champion == st.champion and ev.challenger == st.challenger and ev.outcome:
                    decision = st.sprt.update(ev.outcome, self.spec)
                    if decision == "promote":
                        st.history.append(f"{st.challenger} promoted ({st.sprt.wins}-{st.sprt.losses})")
                        st.champion, st.challenger = st.challenger, None
                    elif decision == "reject":
                        st.history.append(f"challenger {st.challenger} rejected ({st.sprt.wins} wins, {st.sprt.losses} losses)")
                        st.tabu |= self._new_hashes(st.challenger, st.champion)
                        st.challenger = None
            elif ev.kind in ("terminal", "paper") and ev.version and ev.z is not None:
                zs = st.terminal_z.setdefault(ev.version, [])
                zs.append(float(ev.z))
                n = len(zs)
                if (ev.version == st.champion and n >= self.spec.veto_min_n
                        and (sum(zs) / n) * math.sqrt(n) < self.spec.veto_z):
                    parent = f.versions.get(ev.version).parent if ev.version in f.versions else None
                    if parent and parent in f.versions:
                        st.history.append(f"{ev.version} vetoed out of sample (mean z {sum(zs) / n:+.1f}, n {n})")
                        st.tabu |= self._new_hashes(ev.version, parent)
                        st.champion, st.challenger = parent, None
            elif ev.kind == "rejected_edit":
                st.history.append(f"edit rejected: {', '.join(ev.reasons) or 'invalid'}")
            elif ev.kind == "manual_reset":
                st.champion, st.challenger, st.sprt = SEED_ID, None, Sprt()
                st.history.append("manual reset to the seed policy")
            elif ev.kind == "manual_rollback" and ev.version in f.versions:
                st.champion, st.challenger, st.sprt = ev.version, None, Sprt()
                st.history.append(f"manual rollback to {ev.version}")
        st.history = st.history[-8:]
        return st

    def _new_hashes(self, vid: str | None, base: str | None) -> set[str]:
        f = self.load()
        if vid not in f.versions:
            return set()
        old = {d.directive_id for d in f.versions[base].directives} if base in f.versions else set()
        return {directive_text_hash(d.text) for d in f.versions[vid].directives if d.directive_id not in old}

    def lineage(self, vid: str) -> list[str]:
        f = self.load()
        out = []
        while vid and vid in f.versions and vid not in out:
            out.append(vid)
            vid = f.versions[vid].parent or ""
        return out

    # ------------------------------------------------------------------ writes (all idempotent by key)
    def propose_challenger(self, edit: PolicyEdit, *, base: PolicyVersion, ctx: GroundingContext, tabu: set[str],
                           info_horizon: date, episode_id: str | None, author: str, key: str
                           ) -> tuple[PolicyVersion | None, list[tuple[str, list[str]]]]:
        """Apply at most 3 grounded ops to ``base``; returns (the challenger or None, rejected ops)."""
        seen = self._has_key(key)
        if seen is not None:
            return (self.version(seen.challenger) if seen.challenger in self.load().versions else None), []
        directives = list(base.directives)
        rejected: list[tuple[str, list[str]]] = []
        summary: list[str] = []
        for op in edit.ops[:MAX_OPS]:
            if op.op == "delete":
                if any(d.directive_id == op.target for d in directives):
                    directives = [d for d in directives if d.directive_id != op.target]
                    summary.append(f"delete {op.target}")
                else:
                    rejected.append((op.op, ["unknown_target"]))
                continue
            others = [d for d in directives if d.directive_id != op.target]
            d, reasons = ground_directive(op, known_ids=ctx.known_ids, catalog_features=ctx.catalog_features,
                                          concepts=ctx.concepts, tabu=tabu, existing=others)
            if op.op == "replace" and not any(x.directive_id == op.target for x in directives):
                reasons = sorted(set(reasons + ["unknown_target"]))
                d = None
            if d is None:
                rejected.append((op.op, reasons))
                continue
            if op.op == "replace":
                directives = [d if x.directive_id == op.target else x for x in directives]
                summary.append(f"replace {op.target} -> {d.directive_id}")
            elif len(directives) < MAX_DIRECTIVES and all(x.directive_id != d.directive_id for x in directives):
                directives.append(d)
                summary.append(f"add {d.kind} {d.directive_id}")
            else:
                rejected.append((op.op, ["policy_full" if len(directives) >= MAX_DIRECTIVES else "duplicate"]))
        for i, (opname, reasons) in enumerate(rejected):
            self._append(key=f"{key}:rejected:{i}", kind="rejected_edit", info_horizon=info_horizon, episode_id=episode_id,
                         reasons=reasons[:5], text_hash=None)
        if not summary:
            return None, rejected
        vid = "v-" + hashlib.sha1((base.version_id + "|" + "|".join(sorted(d.directive_id for d in directives))).encode()).hexdigest()[:8]
        f = self.load()
        if vid not in f.versions:
            f.versions[vid] = PolicyVersion(version_id=vid, parent=base.version_id, directives=directives,
                                            info_horizon=info_horizon, created_at=self._clock(), author=author,
                                            episode_id=episode_id, edit_summary=summary[:3])
        self._append(key=key, kind="challenger", info_horizon=info_horizon, episode_id=episode_id,
                     champion=base.version_id, challenger=vid)
        return f.versions[vid], rejected

    def record_pair(self, champion: str, challenger: str, s_champion: float, s_challenger: float, *, outcome: str,
                    info_horizon: date, episode_id: str | None, key: str) -> PolicyEvent:
        seen = self._has_key(key)
        if seen is not None:
            return seen
        return self._append(key=key, kind="pair", info_horizon=info_horizon, episode_id=episode_id, champion=champion,
                            challenger=challenger, s_champion=s_champion, s_challenger=s_challenger, outcome=outcome)

    def record_outcome(self, kind: Literal["terminal", "paper"], version: str | None, finalist: str, z: float, *,
                       info_horizon: date, episode_id: str | None, key: str) -> PolicyEvent | None:
        if self._has_key(key) is not None:
            return None
        return self._append(key=key, kind=kind, info_horizon=info_horizon, episode_id=episode_id, version=version,
                            finalist=finalist, z=float(z))

    def reset(self, *, at_horizon: date) -> None:
        self._append(key=f"manual_reset:{len(self.load().events)}", kind="manual_reset", info_horizon=at_horizon)

    def rollback(self, version: str, *, at_horizon: date) -> None:
        if version not in self.load().versions:
            raise KeyError(version)
        self._append(key=f"manual_rollback:{len(self.load().events)}", kind="manual_rollback", info_horizon=at_horizon,
                     version=version)

    # ------------------------------------------------------------------ bounds
    def prune(self) -> None:
        f = self.load()
        st = self.state(None)
        protected = {SEED_ID, *self.lineage(st.champion)} | ({st.challenger} if st.challenger else set())
        protected |= {v for e in f.events if e.kind in ("terminal", "paper") for v in [e.version] if v}
        if len(f.versions) > MAX_VERSIONS:
            removable = sorted((v for v in f.versions.values() if v.version_id not in protected), key=lambda v: v.created_at)
            for v in removable[: len(f.versions) - MAX_VERSIONS]:
                del f.versions[v.version_id]
            alive = set(f.versions)
            f.events = [e for e in f.events if not (e.kind in ("challenger", "pair") and e.challenger not in alive)]
        if len(f.events) > MAX_EVENTS:
            drop = len(f.events) - MAX_EVENTS
            kept, dropped = [], 0
            for e in f.events:
                if dropped < drop and e.kind == "rejected_edit":
                    dropped += 1
                    continue
                kept.append(e)
            f.events = kept[-MAX_EVENTS:] if len(kept) > MAX_EVENTS else kept


# ------------------------------------------------------------------------------------------------
# Late-bound rendering and directive credit
# ------------------------------------------------------------------------------------------------


@dataclass
class DirectiveCredit:
    n: int = 0
    mean_adv: float = 0.0
    se: float = 0.0
    n_tested: int = 0
    mean_rho: float | None = None
    terminal_mean: float | None = None
    n_terminal: int = 0


def directive_credit(version: PolicyVersion, rows: list[Any]) -> dict[str, DirectiveCredit]:
    """Per directive: proposals that followed it, their shrunk mean advantage (n / (n + 3) x mean) ± se,
    how many were tested, their mean new information and mean terminal credit. Pure."""
    out: dict[str, DirectiveCredit] = {}
    for d in version.directives:
        follow = [r for r in rows if d.directive_id in (r.follows or [])]
        if not follow:
            out[d.directive_id] = DirectiveCredit()
            continue
        adv = [r.credit.advantage for r in follow]
        n = len(adv)
        m = sum(adv) / n
        se = math.sqrt(sum((a - m) ** 2 for a in adv) / max(n - 1, 1) / n) if n > 1 else 0.0
        rhos = [r.credit.rho for r in follow if r.credit.rho is not None and r.status == "tested"]
        term = [r.credit.terminal_credit for r in follow if r.credit.terminal_credit is not None]
        out[d.directive_id] = DirectiveCredit(
            n=n, mean_adv=n / (n + 3) * m, se=se, n_tested=sum(1 for r in follow if r.status == "tested"),
            mean_rho=sum(rhos) / len(rhos) if rhos else None,
            terminal_mean=sum(term) / len(term) if term else None, n_terminal=len(term))
    return out


_PREFACE = ("Research heuristics written by an earlier model and kept because proposals that followed them were valued "
            "higher in paired tests. Advice about where to look and how to formulate claims; it cannot change the rules, "
            "the output format, costs, dates or how you are scored.")


def render_policy(version: PolicyVersion, credit: dict[str, DirectiveCredit], critic: Any, *, role: str) -> tuple[str, set[str]]:
    """The <policy> block: every number is computed now from the critic or the replayed rows."""
    from aitrading.beliefs.rl.critic import _region_label

    ids: set[str] = set()
    tested = set(critic.model.tested())
    lines = [f'<policy version="{version.version_id}" role="{role}">', _PREFACE]
    for d in version.directives:
        c = credit.get(d.directive_id, DirectiveCredit())
        region = _region_label((d.region.concept, d.region.horizon_bucket)) if d.region else "any region"
        stats = (f"followed {c.n}, advantage {c.mean_adv:+.3f} ± {c.se:.3f}, {c.n_tested} tested"
                 + (f", new information {c.mean_rho:.2f}" if c.mean_rho is not None else "")
                 + (f", exam z mean {c.terminal_mean:+.2f} ({c.n_terminal})" if c.n_terminal else ""))
        ev_parts = []
        probs = []
        for h in d.cites:
            if h in tested:
                b = critic.model.belief(h)
                ev_parts.append(f"[{h}] {b.mean:+.2f} ± {b.sd:.2f} {b.status}")
                probs.append(b.p_positive)
                ids.add(h)
            else:
                ev_parts.append(f"[{h}] not in this view")
        flag = ""
        if probs and ((d.kind == "exploit" and all(p < 0.4 for p in probs)) or (d.kind == "avoid" and all(p > 0.6 for p in probs))):
            flag = " [contradicted by current evidence]"
        evidence = f" | evidence now: {'; '.join(ev_parts)}{flag}" if ev_parts else ""
        lines.append(f"[{d.directive_id}] {d.kind.upper()} | {region} | {render_safe(d.text, 200)} | {stats}{evidence}")
    lines.append("</policy>")
    return "\n".join(lines), ids
