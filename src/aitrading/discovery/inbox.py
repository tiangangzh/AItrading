"""The trader's idea inbox: a small JSON store of ``IdeaCandidate`` objects.

* Location: ``IdeaInbox(path)`` > ``$AITRADING_HOME/ideas.json`` > ``~/.aitrading/ideas.json``.
* Format: ``{"schema_version": 1, "updated_at": ..., "ideas": [...], "pruned": [...], "quarantined": [...]}``.
  ``pruned`` keeps a small tombstone (id, url, title, status) for every pruned idea so a rejected
  paper that is discovered again is not proposed again. Entries that fail validation (e.g. edited
  by hand) are kept under ``quarantined`` instead of being dropped.
* Writes are atomic (temp file in the same directory + ``fsync`` + ``os.replace``), so a crash
  never leaves a half-written inbox; on Windows ``os.replace`` is retried briefly when another
  process has the file open.
* Concurrency: every read-modify-write runs under an exclusive lock file (``ideas.json.lock``,
  created with ``O_CREAT | O_EXCL``, works on Windows, macOS and Linux). While a process holds the
  lock, a heartbeat thread refreshes the lock file's mtime, so a long ``locked()`` block is never
  mistaken for a crashed one. A lock whose mtime is older than ``stale_lock_s`` (left behind by a
  crashed process) is broken - under a short-lived breaker file (``ideas.json.lock.break``) and only
  after re-checking that it is still the same stale lock, so two waiters can never both break it (and
  one delete the other's fresh lock). A process only ever releases a lock file carrying its own token.
  Reads need no lock because writes are atomic.
* Encoding: the file is written as UTF-8 and read as UTF-8 with or without a BOM (Windows editors
  add one when the file is edited by hand).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, get_args

from pydantic import BaseModel, ValidationError

from aitrading.discovery.models import IdeaCandidate, IdeaStatus, ReplicationReport
from aitrading.discovery.rank import DUPLICATE_TITLE_SIMILARITY, assess_novelty, canonical_url, rank_candidates, score_candidate, titles_match

__all__ = [
    "ENV_HOME",
    "SCHEMA_VERSION",
    "IdeaInbox",
    "InboxError",
    "InboxLockTimeout",
    "default_inbox_path",
]

log = logging.getLogger(__name__)

ENV_HOME = "AITRADING_HOME"
SCHEMA_VERSION = 1
FILE_NAME = "ideas.json"
DECISION_STATUSES = frozenset({"accepted", "rejected", "deferred"})
VALID_STATUSES: frozenset[str] = frozenset(get_args(IdeaStatus))
MAX_TOMBSTONES = 5000


class InboxError(RuntimeError):
    """The inbox file cannot be used (e.g. written by a newer version)."""


class InboxLockTimeout(InboxError, TimeoutError):
    """Another process held the inbox lock for longer than the timeout."""


def default_inbox_path() -> Path:
    home = os.environ.get(ENV_HOME, "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".aitrading"
    return root / FILE_NAME


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _retry_os(fn: Callable[[], Any], attempts: int = 20, delay_s: float = 0.05) -> Any:
    """Retry an OS call that fails transiently on Windows (file opened by another process)."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay_s)
    return None  # pragma: no cover


class _LockFile:
    """Exclusive inter-process lock using an ``O_EXCL`` lock file (portable, stdlib only).

    * The holder refreshes the lock file's mtime from a heartbeat thread (every ``stale_after_s / 4``,
      at most 30 s), so only a lock whose holder died goes stale.
    * Breaking a stale lock is serialised by a breaker file created with ``O_EXCL``; inside it the
      lock is re-read and removed only if it is still the very lock (same content, still stale) that
      was judged stale. A waiter that judged an old lock stale can therefore never delete a fresh lock
      another waiter created after breaking the old one.
    """

    BREAKER_STALE_S = 10.0  # a breaker file older than this was left by a process that crashed mid-break

    def __init__(self, path: Path, *, timeout_s: float, stale_after_s: float, poll_s: float = 0.02):
        self.path = path
        self.breaker_path = path.with_name(path.name + ".break")
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self.poll_s = poll_s
        self._token = ""
        self._stop_heartbeat: threading.Event | None = None
        self._heartbeat: threading.Thread | None = None

    @property
    def heartbeat_interval_s(self) -> float:
        return max(0.01, min(self.stale_after_s / 4.0, 30.0))

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._break_if_stale():
                    continue
            except PermissionError:  # Windows: the lock file is being deleted by its owner right now
                pass
            else:
                self._token = f"pid={os.getpid()} token={uuid.uuid4().hex} acquired={time.time():.3f}"
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(self._token + "\n")
                self._start_heartbeat()
                return
            if time.monotonic() >= deadline:
                raise InboxLockTimeout(
                    f"could not lock {self.path} within {self.timeout_s:.1f}s (another aitrading process is using the inbox; "
                    f"if none is running, delete the lock file)"
                )
            time.sleep(self.poll_s)

    def _read(self) -> tuple[str, float] | None:
        """(content, mtime) of the lock file, or None when it does not exist."""
        try:
            mtime = os.path.getmtime(self.path)
            content = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError:  # unreadable (Windows sharing violation): treat as held and fresh
            return ("<unreadable>", time.time())
        return content, mtime

    def _break_if_stale(self) -> bool:
        """Remove the lock if its holder is gone. Returns True when the caller should retry at once."""
        seen = self._read()
        if seen is None:
            return True  # released between our attempts: retry immediately
        if time.time() - seen[1] <= self.stale_after_s:
            return False
        try:
            gfd = os.open(str(self.breaker_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self._remove_stale_breaker()
            return False  # another process is breaking it right now
        except OSError:
            return False
        os.close(gfd)
        try:
            now = self._read()  # re-check inside the breaker: is it still the same stale lock?
            if now is None:
                return True
            if now[0] != seen[0] or time.time() - now[1] <= self.stale_after_s:
                return False  # a live process re-took (or refreshed) the lock meanwhile
            log.warning("breaking stale inbox lock %s (age %.0fs, %s)", self.path, time.time() - now[1], now[0] or "no owner info")
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            except PermissionError:
                return False
            return True
        finally:
            try:
                os.unlink(self.breaker_path)
            except OSError:  # pragma: no cover - left for _remove_stale_breaker
                pass

    def _remove_stale_breaker(self) -> None:
        try:
            if time.time() - os.path.getmtime(self.breaker_path) > self.BREAKER_STALE_S:
                os.unlink(self.breaker_path)
                log.warning("removed stale inbox lock breaker %s", self.breaker_path)
        except OSError:
            pass

    # ---------------------------------------------------------------- heartbeat
    def _start_heartbeat(self) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._beat, args=(stop, self._token, self.heartbeat_interval_s), name="aitrading-inbox-lock-heartbeat", daemon=True
        )
        self._stop_heartbeat, self._heartbeat = stop, thread
        thread.start()

    def _beat(self, stop: threading.Event, token: str, interval: float) -> None:
        while not stop.wait(interval):
            try:
                if self.path.read_text(encoding="utf-8").strip() != token:
                    return  # no longer ours
                os.utime(self.path, None)
            except FileNotFoundError:
                return
            except OSError:
                continue

    def _stop_beating(self) -> None:
        if self._stop_heartbeat is not None:
            self._stop_heartbeat.set()
        if self._heartbeat is not None and self._heartbeat is not threading.current_thread():
            self._heartbeat.join(timeout=5.0)
        self._stop_heartbeat = self._heartbeat = None

    def release(self) -> None:
        """Remove the lock file - only if it is still ours (it may have been broken as stale and re-taken)."""
        self._stop_beating()
        try:
            owner = self.path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return
        except OSError:
            owner = self._token  # unreadable (Windows sharing violation): assume it is ours
        if owner != self._token:
            log.warning("inbox lock %s was taken over by another process; leaving it in place", self.path)
            return
        try:
            _retry_os(lambda: os.unlink(self.path), attempts=10)
        except FileNotFoundError:
            pass
        except OSError as e:  # pragma: no cover - leave it to the stale-lock breaker
            log.warning("could not remove inbox lock %s: %s", self.path, e)


class IdeaInbox:
    """Persistent, process-safe store of idea candidates (see module docstring)."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        templates: Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        lock_timeout_s: float = 30.0,
        stale_lock_s: float = 120.0,
    ):
        self.path = Path(path).expanduser() if path is not None else default_inbox_path()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._templates = None if templates is None else dict(templates)
        self._clock = clock or _utcnow
        self._lock = _LockFile(self.lock_path, timeout_s=lock_timeout_s, stale_after_s=stale_lock_s)
        self._rlock = threading.RLock()
        self._depth = 0

    # ------------------------------------------------------------------ locking
    @contextmanager
    def locked(self) -> Iterator["IdeaInbox"]:
        """Hold the inbox lock (re-entrant) - wrap several calls to make them one atomic update."""
        with self._rlock:
            if self._depth == 0:
                self._lock.acquire()
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._lock.release()

    # ------------------------------------------------------------------ storage
    def _templates_dict(self) -> dict[str, Any]:
        if self._templates is None:
            from aitrading.discovery.extract import load_library_templates  # noqa: PLC0415

            self._templates = load_library_templates()
        return self._templates

    def _load(self, *, for_write: bool = False) -> dict[str, Any]:
        state: dict[str, Any] = {"ideas": [], "pruned": [], "quarantined": []}
        try:
            raw = _retry_os(lambda: self.path.read_text(encoding="utf-8-sig"))  # accepts a BOM from hand edits
        except FileNotFoundError:
            return state
        if not raw.strip():
            return state
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            if for_write:
                backup = self.path.with_name(f"{self.path.name}.corrupt-{self._clock():%Y%m%dT%H%M%S}")
                _retry_os(lambda: os.replace(self.path, backup))
                log.warning("inbox %s is not valid JSON (%s); moved it to %s and started a new inbox", self.path, e, backup)
            else:
                log.warning("inbox %s is not valid JSON (%s); treating it as empty", self.path, e)
            return state
        if isinstance(data, list):  # bare list (hand-made file)
            data = {"schema_version": SCHEMA_VERSION, "ideas": data}
        if not isinstance(data, dict):
            raise InboxError(f"{self.path}: unexpected top-level JSON type {type(data).__name__}")
        version = data.get("schema_version", 1)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise InboxError(f"{self.path} has schema_version {version!r}; this version of aitrading reads <= {SCHEMA_VERSION}")
        for item in data.get("ideas") or []:
            try:
                state["ideas"].append(IdeaCandidate.model_validate(item))
            except ValidationError as e:
                log.warning("quarantining an invalid inbox entry: %s", e.errors()[:1])
                state["quarantined"].append(item)
        state["pruned"] = [t for t in (data.get("pruned") or []) if isinstance(t, dict)]
        state["quarantined"].extend(q for q in (data.get("quarantined") or []))
        return state

    def _save(self, state: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": self._clock().isoformat(),
            "ideas": [c.model_dump(mode="json") for c in state["ideas"]],
            "pruned": list(state.get("pruned") or [])[-MAX_TOMBSTONES:],
            "quarantined": list(state.get("quarantined") or []),
        }
        data = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            _retry_os(lambda: os.replace(tmp, self.path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @contextmanager
    def _mutate(self) -> Iterator[dict[str, Any]]:
        """Lock, load, yield the state for in-place changes, then save it (unless an exception occurred)."""
        with self.locked():
            state = self._load(for_write=True)
            yield state
            self._save(state)

    # ------------------------------------------------------------------ queries
    def all(self) -> list[IdeaCandidate]:
        """Every idea in storage order."""
        return list(self._load()["ideas"])

    def __len__(self) -> int:
        return len(self._load()["ideas"])

    def __contains__(self, idea_id: object) -> bool:
        return isinstance(idea_id, str) and any(c.idea_id == idea_id for c in self._load()["ideas"])

    @staticmethod
    def _find(ideas: list[IdeaCandidate], idea_id: str) -> int:
        """Index of ``idea_id`` (exact, or a unique prefix of >= 4 characters); raises KeyError."""
        for i, c in enumerate(ideas):
            if c.idea_id == idea_id:
                return i
        if idea_id and len(idea_id) >= 4:
            hits = [i for i, c in enumerate(ideas) if c.idea_id.startswith(idea_id)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise KeyError(f"idea id prefix '{idea_id}' is ambiguous ({len(hits)} matches)")
        raise KeyError(f"no idea with id '{idea_id}' in {_display(ideas)}")

    def get(self, idea_id: str) -> IdeaCandidate | None:
        """The idea with this id (or unique id prefix), or None."""
        ideas = self._load()["ideas"]
        try:
            return ideas[self._find(ideas, idea_id)]
        except KeyError:
            return None

    # ------------------------------------------------------------------ mutations
    def add(self, candidates: Iterable[IdeaCandidate]) -> list[IdeaCandidate]:
        """Add new candidates; returns those actually added (with novelty and score set).

        Duplicates of existing inbox items, of earlier items in the same batch and of pruned ideas are
        skipped. A candidate whose duplicate / novelty / score check raises is logged and skipped; the
        rest of the batch is still stored.
        """
        incoming = list(candidates)
        if not incoming:
            return []
        templates = self._templates_dict()
        added: list[IdeaCandidate] = []
        with self._mutate() as state:
            ideas: list[IdeaCandidate] = state["ideas"]
            for cand in incoming:
                try:  # one malformed candidate (e.g. an unparsable URL from a search result) must not sink the batch
                    if self._is_pruned(cand, state["pruned"]):
                        continue
                    novelty = assess_novelty(cand, ideas, templates)
                    if novelty == "duplicate":
                        continue
                    c = cand.model_copy(update={"novelty": novelty})
                    c = c.model_copy(update={"score": score_candidate(c, today=self._clock().date())})
                except Exception as e:  # noqa: BLE001 - logged and skipped; the rest of the batch is stored
                    log.warning("skipping idea %s (%s): could not check it against the inbox: %s: %s",
                                getattr(cand, "idea_id", "?"), getattr(getattr(cand, "source", None), "url", "?"), type(e).__name__, e)
                    continue
                ideas.append(c)
                added.append(c)
        return added

    def set_status(self, idea_id: str, status: IdeaStatus, note: str | None = None) -> IdeaCandidate:
        """Change an idea's status; accepted / rejected / deferred also stamp ``decided_at``."""
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status '{status}' (valid: {', '.join(sorted(VALID_STATUSES))})")
        with self._mutate() as state:
            ideas = state["ideas"]
            i = self._find(ideas, idea_id)
            now = self._clock()
            update: dict[str, Any] = {"status": status}
            if status in DECISION_STATUSES:
                update["decided_at"] = now
            if note:
                update["notes"] = [*ideas[i].notes, f"[{now:%Y-%m-%d %H:%M} {status}] {note}"]
            ideas[i] = ideas[i].model_copy(update=update)
            return ideas[i]

    def attach_spec(self, idea_id: str, spec: Mapping[str, Any] | BaseModel) -> IdeaCandidate:
        """Store the translated ``StrategySpec`` (as JSON) on the idea."""
        spec_json = spec.model_dump(mode="json") if isinstance(spec, BaseModel) else json.loads(json.dumps(dict(spec), default=str))
        with self._mutate() as state:
            ideas = state["ideas"]
            i = self._find(ideas, idea_id)
            ideas[i] = ideas[i].model_copy(update={"strategy_spec": spec_json})
            return ideas[i]

    def attach_result(
        self,
        idea_id: str,
        run_id: str,
        replication: ReplicationReport | Mapping[str, Any] | None = None,
        status: IdeaStatus = "tested",
    ) -> IdeaCandidate:
        """Record a backtest run (and optional replication report) and set the status (default 'tested')."""
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown status '{status}'")
        if replication is not None and not isinstance(replication, ReplicationReport):
            replication = ReplicationReport.model_validate(replication)
        with self._mutate() as state:
            ideas = state["ideas"]
            i = self._find(ideas, idea_id)
            update: dict[str, Any] = {"backtest_run_id": run_id, "status": status}
            if replication is not None:
                update["replication"] = replication
            ideas[i] = ideas[i].model_copy(update=update)
            return ideas[i]

    def prune(self, older_than_days: float, statuses: Iterable[str] = ("rejected",)) -> int:
        """Remove ideas in ``statuses`` decided (or, if undecided, discovered) more than N days ago.

        Pruned ideas leave a tombstone so they are not re-added when discovered again. Returns the
        number of ideas removed.
        """
        wanted = set(statuses)
        unknown = wanted - VALID_STATUSES
        if unknown:
            raise ValueError(f"unknown status(es): {sorted(unknown)}")
        now = self._clock()
        cutoff = _aware(now) - timedelta(days=float(older_than_days))
        with self._mutate() as state:
            keep, removed = [], []
            for c in state["ideas"]:
                stamp = _aware(c.decided_at or c.discovered_at)
                (removed if c.status in wanted and stamp < cutoff else keep).append(c)
            state["ideas"] = keep
            state["pruned"].extend(
                {"idea_id": c.idea_id, "url": c.source.url, "title": c.source.title, "status": c.status, "pruned_at": now.isoformat()}
                for c in removed
            )
        return len(removed)

    def remove(self, idea_id: str) -> IdeaCandidate:
        """Delete one idea outright (no tombstone - it can be discovered again)."""
        with self._mutate() as state:
            ideas = state["ideas"]
            return ideas.pop(self._find(ideas, idea_id))

    @staticmethod
    def _is_pruned(c: IdeaCandidate, tombstones: list[dict[str, Any]]) -> bool:
        url = canonical_url(c.source.url)
        for t in tombstones:
            if t.get("idea_id") == c.idea_id:
                return True
            if url and url == canonical_url(str(t.get("url") or "")):
                return True
            if titles_match(c.source.title, str(t.get("title") or ""), DUPLICATE_TITLE_SIMILARITY):
                return True
        return False

    # Defined last so the method name does not shadow the ``list`` builtin in annotations above.
    def list(self, status: IdeaStatus | Iterable[str] | None = None, limit: int | None = None) -> list[IdeaCandidate]:
        """Ideas (optionally only those with ``status`` - one status or several), best score first."""
        ideas = self._load()["ideas"]
        if status is not None:
            wanted = {status} if isinstance(status, str) else set(status)
            ideas = [c for c in ideas if c.status in wanted]
        ranked = rank_candidates(ideas)
        return ranked[:limit] if limit is not None else ranked


def _display(ideas: list[IdeaCandidate]) -> str:
    return f"the inbox ({len(ideas)} ideas)"
