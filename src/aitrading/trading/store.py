"""On-disk store of saved strategies (spec + latest backtest + metadata) for the idea lab.

Layout
------
``<root>/<slug>/`` per strategy, where ``root`` is ``StrategyStore(root)`` >
``$AITRADING_HOME/strategies`` > ``~/.aitrading/strategies`` and ``slug`` is a Windows-safe
version of the strategy name (see :func:`slugify_strategy_name`)::

    spec.json       StrategySpec as JSON (``StrategySpec.model_validate_json`` reads it directly)
    backtest.json   BacktestResult as JSON (optional: the latest backtest of exactly this spec)
    meta.json       {"schema_version", "name", "slug", "created", "updated", "idea", "template",
                     "notes", "spec_name", "kind", "rebalance", "has_backtest", "backtest"}
    ledger.json     paper-trading ledger, owned by :class:`aitrading.trading.paper.PaperAccount`

``meta.json`` carries the directory's ``schema_version``; a directory written by a newer version
of aitrading is refused instead of being misread. ``spec.json`` / ``backtest.json`` are the plain
model JSON so other tools can read them without this module.

Names
-----
Any display name is accepted ("Momentum 12-1", "Value/Quality: v2", "CON", "动量策略"); the
directory name is a lowercase ASCII slug of it without characters Windows forbids
(``<>:"/\\|?*``), without trailing dots/spaces and never a reserved device name (CON, PRN, AUX,
NUL, COM0-9, LPT0-9). Names that differ only in case or punctuation map to the same strategy
(Windows and macOS file systems are case-insensitive, so this keeps behaviour identical on every
OS). Names whose letters cannot be transliterated to ASCII, and names longer than 64 characters,
get a short hash suffix so distinct names stay distinct. ``meta.json`` keeps the original name.

Writes
------
Every file is written atomically (temp file in the same directory + ``fsync`` + ``os.replace``,
retried briefly on Windows when another process has the file open), so a crash never leaves a
half-written file. ``save`` writes in the order: remove the old backtest, spec, backtest, meta, so
an interrupted overwrite can never pair a new spec with an old backtest (or vice versa).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from aitrading.backtest.models import BacktestResult
from aitrading.strategy.spec import StrategySpec

__all__ = [
    "ENV_HOME",
    "SCHEMA_VERSION",
    "SPEC_FILE",
    "BACKTEST_FILE",
    "META_FILE",
    "LEDGER_FILE",
    "StoreError",
    "StrategyNotFoundError",
    "StrategyExistsError",
    "StrategyStore",
    "default_store_root",
    "slugify_strategy_name",
    "write_json_atomic",
    "read_json",
]

log = logging.getLogger(__name__)

ENV_HOME = "AITRADING_HOME"
SCHEMA_VERSION = 1
SPEC_FILE = "spec.json"
BACKTEST_FILE = "backtest.json"
META_FILE = "meta.json"
LEDGER_FILE = "ledger.json"
MAX_SLUG_LEN = 64

_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}
)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------


class StoreError(RuntimeError):
    """The strategy store cannot be used (unreadable file, newer schema, invalid name...)."""


class StrategyNotFoundError(StoreError, KeyError):
    """No saved strategy with this name (also a ``KeyError``)."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return Exception.__str__(self)


class StrategyExistsError(StoreError, FileExistsError):
    """A strategy with this name (or slug) already exists and ``overwrite`` was not set."""


# ------------------------------------------------------------------------------------------------
# Paths and names
# ------------------------------------------------------------------------------------------------


def default_store_root() -> Path:
    """``$AITRADING_HOME/strategies`` when the variable is set, else ``~/.aitrading/strategies``."""
    home = os.environ.get(ENV_HOME, "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".aitrading"
    return root / "strategies"


def _name_hash(name: str) -> str:
    return hashlib.sha1(name.strip().casefold().encode("utf-8")).hexdigest()[:8]


def slugify_strategy_name(name: str) -> str:
    """Windows-, macOS- and Linux-safe directory name for a strategy (idempotent on its output).

    Lowercase ASCII letters, digits, ``-`` and ``_`` only; starts with a letter or digit; at most
    64 characters; never a Windows reserved device name. Accented letters are transliterated
    (``"Qualité"`` -> ``"qualite"``); letters without an ASCII form and over-long names get an
    8-character hash suffix so that distinct names do not collide. Raises ``ValueError`` for an
    empty name.
    """
    if not isinstance(name, str):
        raise TypeError(f"strategy name must be a string, got {type(name).__name__}")
    raw = name.strip()
    if not raw:
        raise ValueError("strategy name must not be empty")
    norm = unicodedata.normalize("NFKD", raw)
    lossy = any(ord(ch) > 127 and unicodedata.category(ch)[0] in "LN" for ch in norm)
    s = norm.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    if not s:
        s = f"strategy-{_name_hash(raw)}"
    elif lossy or len(s) > MAX_SLUG_LEN:
        s = s[: MAX_SLUG_LEN - 9].strip("-_") + "-" + _name_hash(raw)
    if s in _WINDOWS_RESERVED:
        s += "_"
    if not _SLUG_RE.match(s):  # pragma: no cover - defensive; the rules above guarantee it
        raise ValueError(f"could not derive a safe directory name from {name!r}")
    return s


# ------------------------------------------------------------------------------------------------
# Atomic JSON I/O (shared with the paper-trading ledger)
# ------------------------------------------------------------------------------------------------


def _retry_os(fn: Callable[[], Any], attempts: int = 20, delay_s: float = 0.05) -> Any:
    """Retry an OS call that fails transiently on Windows (file held open by another process)."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay_s)
    return None  # pragma: no cover


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write ``obj`` as UTF-8 JSON to ``path`` atomically (temp file + fsync + ``os.replace``).

    Readers see either the old file or the complete new one; on failure the temp file is removed
    and the old file is left untouched. Non-finite floats are refused (they are not valid JSON).
    """
    path = Path(path)
    data = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _retry_os(lambda: os.replace(tmp, path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    """Parse a JSON file (``FileNotFoundError`` if missing, ``StoreError`` if not valid JSON)."""
    path = Path(path)
    raw = _retry_os(lambda: path.read_text(encoding="utf-8"))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise StoreError(f"{path} is not valid JSON ({e}); fix or delete the file") from e


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _rmtree(path: Path) -> None:
    """``shutil.rmtree`` that also removes read-only files (Windows)."""

    def _on_error(func, p, _exc):  # noqa: ANN001
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            raise

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:  # pragma: no cover - exercised on Python 3.10/3.11
        shutil.rmtree(path, onerror=_on_error)


# ------------------------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------------------------


def _template_key(spec: StrategySpec) -> str | None:
    """The library template a spec came from (template specs use ``name == key``)."""
    try:
        from aitrading.strategy.library import TEMPLATES  # noqa: PLC0415 - optional, avoids import cycles
    except Exception:  # pragma: no cover - library unavailable
        return None
    return spec.name if spec.name in TEMPLATES else None


def _backtest_headline(bt: BacktestResult) -> dict[str, Any]:
    st = bt.stats.get("strategy")
    return {
        "run_id": bt.run_id,
        "start": bt.start.isoformat(),
        "end": bt.end.isoformat(),
        "provider": bt.provider,
        "total_return_pct": st.total_return_pct if st else None,
        "cagr_pct": st.cagr_pct if st else None,
        "sharpe": st.sharpe if st else None,
        "max_drawdown_pct": st.max_drawdown_pct if st else None,
        "verdict": bt.interpretation.verdict if bt.interpretation else None,
    }


class StrategyStore:
    """Saved strategies on disk (see the module docstring for the layout and guarantees)."""

    def __init__(self, root: Path | str | None = None, *, clock: Callable[[], datetime] | None = None):
        self.root = Path(root).expanduser() if root is not None else default_store_root()
        self._clock = clock or _utcnow

    def __repr__(self) -> str:
        return f"StrategyStore(root={str(self.root)!r})"

    # ------------------------------------------------------------------ paths
    def path(self, name: str) -> Path:
        """Directory of strategy ``name`` (it need not exist yet)."""
        return self.root / slugify_strategy_name(name)

    def exists(self, name: str) -> bool:
        return (self.path(name) / SPEC_FILE).is_file()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and bool(name.strip()) and self.exists(name)

    def _require(self, name: str) -> Path:
        d = self.path(name)
        if not (d / SPEC_FILE).is_file():
            raise StrategyNotFoundError(f"no saved strategy named {name!r} (looked in {d})")
        return d

    # ------------------------------------------------------------------ write
    def save(
        self,
        name: str,
        spec: StrategySpec,
        backtest: BacktestResult | None = None,
        *,
        overwrite: bool = False,
        idea: str | None = None,
        template: str | None = None,
        notes: str | None = None,
    ) -> Path:
        """Save ``spec`` (and optionally its ``backtest``) under ``name``; returns the directory.

        ``idea`` defaults to ``spec.idea``; ``template`` to the library template the spec was
        built from (``spec.name`` if it is a template key). With ``overwrite=True`` an existing
        strategy keeps its ``created`` time (and notes/template unless new ones are given) and
        any paper-trading ledger; its old ``backtest.json`` is removed even when no new backtest
        is given, because it no longer describes the saved spec. Raises
        :class:`StrategyExistsError` if the strategy exists and ``overwrite`` is false.
        """
        if not isinstance(spec, StrategySpec):
            raise TypeError(f"spec must be a StrategySpec, got {type(spec).__name__}")
        if backtest is not None and not isinstance(backtest, BacktestResult):
            raise TypeError(f"backtest must be a BacktestResult or None, got {type(backtest).__name__}")
        d = self.path(name)
        old_meta: dict[str, Any] = {}
        if (d / SPEC_FILE).is_file():
            if not overwrite:
                existing = self._read_meta_raw(d).get("name") or d.name
                raise StrategyExistsError(
                    f"a strategy named {existing!r} already exists in {d}; pass overwrite=True to replace it"
                )
            old_meta = self._read_meta_raw(d)
        now = self._clock().isoformat(timespec="seconds")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "name": name.strip(),
            "slug": d.name,
            "created": old_meta.get("created") or now,
            "updated": now,
            "idea": idea if idea is not None else spec.idea,
            "template": template if template is not None else (old_meta.get("template") or _template_key(spec)),
            "notes": notes if notes is not None else (old_meta.get("notes") or ""),
            "spec_name": spec.name,
            "kind": spec.kind,
            "rebalance": spec.rebalance,
            "has_backtest": backtest is not None,
            "backtest": _backtest_headline(backtest) if backtest is not None else None,
        }
        d.mkdir(parents=True, exist_ok=True)
        bt_path = d / BACKTEST_FILE
        if bt_path.exists():
            _retry_os(lambda: os.unlink(bt_path))
        write_json_atomic(d / SPEC_FILE, spec.model_dump(mode="json"))
        if backtest is not None:
            write_json_atomic(bt_path, backtest.model_dump(mode="json"))
        write_json_atomic(d / META_FILE, meta)
        return d

    def set_notes(self, name: str, notes: str) -> dict[str, Any]:
        """Replace the trader's notes of a saved strategy; returns the updated meta."""
        d = self._require(name)
        meta = self.load_meta(name)
        meta["notes"] = notes
        meta["updated"] = self._clock().isoformat(timespec="seconds")
        meta["schema_version"] = SCHEMA_VERSION
        write_json_atomic(d / META_FILE, meta)
        return meta

    def delete(self, name: str) -> None:
        """Delete the strategy directory (spec, backtest, meta, paper ledger and its archives)."""
        d = self.path(name)
        if not d.is_dir():
            raise StrategyNotFoundError(f"no saved strategy named {name!r} (looked in {d})")
        _rmtree(d)

    # ------------------------------------------------------------------ read
    def _read_meta_raw(self, d: Path) -> dict[str, Any]:
        try:
            data = read_json(d / META_FILE)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise StoreError(f"{d / META_FILE}: expected a JSON object, got {type(data).__name__}")
        version = data.get("schema_version", 1)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            raise StoreError(
                f"{d / META_FILE} has schema_version {version!r}; this version of aitrading reads <= {SCHEMA_VERSION}"
            )
        return data

    def load_spec(self, name: str) -> StrategySpec:
        d = self._require(name)
        self._read_meta_raw(d)  # refuses directories written by a newer schema
        try:
            return StrategySpec.model_validate(read_json(d / SPEC_FILE))
        except ValidationError as e:
            raise StoreError(f"{d / SPEC_FILE} is not a valid StrategySpec: {e}") from e

    def load_backtest(self, name: str) -> BacktestResult | None:
        """The saved backtest, or ``None`` if there is none or it cannot be read (logged)."""
        d = self._require(name)
        try:
            return BacktestResult.model_validate(read_json(d / BACKTEST_FILE))
        except FileNotFoundError:
            return None
        except (ValidationError, StoreError) as e:
            log.warning("ignoring unreadable backtest %s: %s", d / BACKTEST_FILE, e)
            return None

    def load_meta(self, name: str) -> dict[str, Any]:
        """``meta.json`` (synthesised from the spec when missing, e.g. a hand-made directory)."""
        d = self._require(name)
        meta = self._read_meta_raw(d)
        if not meta:
            spec = StrategySpec.model_validate(read_json(d / SPEC_FILE))
            mtime = datetime.fromtimestamp((d / SPEC_FILE).stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            meta = {
                "schema_version": SCHEMA_VERSION,
                "name": d.name,
                "slug": d.name,
                "created": mtime,
                "updated": mtime,
                "idea": spec.idea,
                "template": _template_key(spec),
                "notes": "",
                "spec_name": spec.name,
                "kind": spec.kind,
                "rebalance": spec.rebalance,
                "has_backtest": (d / BACKTEST_FILE).is_file(),
                "backtest": None,
            }
        return meta

    def load(self, name: str) -> tuple[StrategySpec, BacktestResult | None, dict[str, Any]]:
        """``(spec, backtest or None, meta)``; raises :class:`StrategyNotFoundError` if unknown."""
        spec = self.load_spec(name)
        backtest = self.load_backtest(name)
        meta = self.load_meta(name)
        meta["has_backtest"] = backtest is not None
        return spec, backtest, meta

    def list(self) -> list[dict[str, Any]]:
        """One summary dict per saved strategy, sorted by slug.

        Keys: ``name, slug, path, idea, template, notes, kind, rebalance, created, updated,
        has_backtest, backtest`` (headline: run_id, start, end, provider, total_return_pct,
        cagr_pct, sharpe, max_drawdown_pct, verdict - or None) and ``paper_trading`` (a ledger
        exists). Unreadable directories are skipped with a logged warning.
        """
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or d.name.startswith(".") or not (d / SPEC_FILE).is_file():
                continue
            try:
                meta = self._read_meta_raw(d) or self.load_meta(d.name)
            except (StoreError, ValidationError, OSError) as e:
                log.warning("skipping unreadable strategy directory %s: %s", d, e)
                continue
            out.append(
                {
                    "name": meta.get("name") or d.name,
                    "slug": d.name,
                    "path": str(d),
                    "idea": meta.get("idea", ""),
                    "template": meta.get("template"),
                    "notes": meta.get("notes", ""),
                    "kind": meta.get("kind"),
                    "rebalance": meta.get("rebalance"),
                    "created": meta.get("created"),
                    "updated": meta.get("updated"),
                    "has_backtest": (d / BACKTEST_FILE).is_file(),
                    "backtest": meta.get("backtest"),
                    "paper_trading": (d / LEDGER_FILE).is_file(),
                }
            )
        return out
