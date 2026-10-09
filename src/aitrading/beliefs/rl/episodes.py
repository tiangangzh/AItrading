"""Episode logs of the RL research loop: one bounded JSON file per episode under ``<ledger dir>/rl/``.

Each round records:
* the prompts' hashes (plus a bounded copy of the user text, for point-in-time export);
* the hash of the view's evidence ids;
* every proposal with its credit breakdown;
* the selection, the pair score, the feedback lines and the LLM calls.

Every finalist has a record too. ``rows(cutoff)`` returns only rounds whose information horizon is
before the cutoff, so credit shown to a later run never includes later information.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

from aitrading.beliefs.ledger import BeliefLedger
from aitrading.beliefs.rl.models import SCHEMA_VERSION, EpisodeRecord, IndexRow, ProposalRecord, RoundRecord
from aitrading.beliefs.rl.policy import PolicyStore, rl_dir
from aitrading.trading.store import StoreError, read_json, write_json_atomic

__all__ = ["EpisodeStore", "MAX_EPISODE_BYTES", "MAX_EPISODES"]

MAX_EPISODE_BYTES = 128 * 1024
MAX_EPISODES = 100
MAX_INDEX_ROWS = 5000
MAX_PROMPTS = 20


class EpisodeStore:
    def __init__(self, ledger: BeliefLedger):
        self.ledger = ledger

    @property
    def dir(self) -> Path:
        return rl_dir(self.ledger) / "episodes"

    @property
    def index_path(self) -> Path:
        return rl_dir(self.ledger) / "index.json"

    def _path(self, episode_id: str) -> Path:
        return self.dir / f"{episode_id}.json"

    # ------------------------------------------------------------------ I/O
    def save(self, ep: EpisodeRecord) -> None:
        data = ep.model_dump(mode="json")
        blob = _dumps(data)
        if len(blob) > MAX_EPISODE_BYTES:  # drop stored prompt texts, oldest first, until the file fits
            for rnd in data["rounds"]:
                for arm in rnd["arms"]:
                    if arm.get("user_text"):
                        arm["user_text"], arm["truncated"] = None, True
                        blob = _dumps(data)
                        if len(blob) <= MAX_EPISODE_BYTES:
                            break
                if len(blob) <= MAX_EPISODE_BYTES:
                    break
        write_json_atomic(self._path(ep.episode_id), data)
        self._index(ep)

    def load(self, episode_id: str) -> EpisodeRecord:
        raw = read_json(self._path(episode_id))
        if isinstance(raw, dict) and isinstance(raw.get("schema_version"), int) and raw["schema_version"] > SCHEMA_VERSION:
            raise StoreError(f"episode {episode_id} was written by a newer version of aitrading")
        return EpisodeRecord.model_validate(raw)

    def episodes(self) -> list[EpisodeRecord]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("e-*.json")):
            try:
                out.append(self.load(p.stem))
            except (StoreError, ValueError, OSError):
                continue
        return sorted(out, key=lambda e: e.started_at)

    def index(self) -> list[IndexRow]:
        try:
            raw = read_json(self.index_path)
        except FileNotFoundError:
            return []
        return [IndexRow.model_validate(r) for r in (raw.get("rows", []) if isinstance(raw, dict) else [])]

    def _index(self, ep: EpisodeRecord) -> None:
        rows = [r for r in self.index() if r.episode_id != ep.episode_id]
        rows.append(IndexRow(episode_id=ep.episode_id, market=ep.market, cutoff=ep.cutoff, holdout_window=ep.holdout_window,
                             status=ep.status, n_tested=sum(1 for r in ep.rounds for p in r.proposals if p.status == "tested"),
                             n_finalists=len(ep.finalists), r_oos=ep.r_oos, finished_at=ep.finished_at))
        rows = rows[-MAX_INDEX_ROWS:]
        write_json_atomic(self.index_path, {"schema_version": SCHEMA_VERSION, "rows": [r.model_dump(mode="json") for r in rows]})

    def save_system_prompt(self, text: str) -> str:
        sha = hashlib.sha256(text.encode()).hexdigest()
        d = rl_dir(self.ledger) / "prompts"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{sha}.txt"
        if not p.exists():
            p.write_text(text[: 64 * 1024], encoding="utf-8")
            files = sorted(d.glob("*.txt"), key=lambda q: q.stat().st_mtime)
            for old in files[:-MAX_PROMPTS]:
                old.unlink(missing_ok=True)
        return sha

    def system_prompt(self, sha: str) -> str | None:
        p = rl_dir(self.ledger) / "prompts" / f"{sha}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ------------------------------------------------------------------ queries
    def rows(self, *, cutoff: date | None) -> list[tuple[EpisodeRecord, RoundRecord, ProposalRecord]]:
        out = []
        for ep in self.episodes():
            for rnd in ep.rounds:
                if cutoff is not None and rnd.info_horizon >= cutoff:
                    continue
                for p in rnd.proposals:
                    out.append((ep, rnd, p))
        return out

    def mark_aborted(self) -> int:
        n = 0
        for ep in self.episodes():
            if ep.status == "running":
                ep.status = "aborted"
                ep.warnings = (ep.warnings + ["aborted: the run stopped before finishing"])[-20:]
                self.save(ep)
                n += 1
        return n

    def attach_paper(self, ledger: BeliefLedger, store: PolicyStore, *, min_months: int = 6, spec: Any = None) -> int:
        """Delayed reward: for each finalist with >= ``min_months`` of paper evidence that starts after
        its episode finished, record one 'paper' outcome (idempotent by key). Returns the number recorded."""
        from aitrading.beliefs.bayes import Hyper
        from aitrading.beliefs.rl.critic import Critic
        from aitrading.beliefs.rl.reward import RewardSpec

        spec = spec or RewardSpec()
        n = 0
        for ep in self.episodes():
            if ep.status != "finished" or ep.finished_at is None:
                continue
            changed = False
            for f in ep.finalists:
                papers = [e for e in ledger.evidence_for(f.hypothesis_id)
                          if e.kind == "paper" and e.n_periods >= min_months and e.start > ep.finished_at.date()]
                if not papers:
                    continue
                pe = papers[-1]
                critic = Critic(ledger, cutoff=pe.start, research_window=(None, pe.start), hyper=Hyper(**ep.hyper_frozen),
                                costs_bps=float(f.paper.get("costs_bps", 10.0)) if f.paper else 10.0, spec=spec,
                                before=pe.recorded_at)
                if f.hypothesis_id not in critic.model.tested():
                    continue
                z, mu, sdp = critic.exam_z(f.hypothesis_id, pe)
                ev = store.record_outcome("paper", f.origin_version, f.hypothesis_id, z, info_horizon=pe.end,
                                          episode_id=ep.episode_id, key=f"{ep.episode_id}:{f.hypothesis_id}:{pe.evidence_id}")
                if ev is not None:
                    f.paper = {"evidence_id": pe.evidence_id, "z": z, "mu": mu, "sd_pred": sdp, "months": pe.n_periods}
                    changed = True
                    n += 1
            if changed:
                self.save(ep)
        if n:
            store.save()
        return n

    def prune(self, max_episodes: int = MAX_EPISODES) -> None:
        eps = self.episodes()
        if len(eps) <= max_episodes:
            return
        for ep in eps[: len(eps) - max_episodes]:
            self._path(ep.episode_id).unlink(missing_ok=True)


def _dumps(obj: Any) -> bytes:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
