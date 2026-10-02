"""Tiny, dependency-free disk cache for the free-data provider (JSON, text and DataFrames).

Design goals: runs unchanged on Windows, macOS and Linux; no pyarrow; safe under threads; a
corrupt or half-written entry is treated as a miss, never as an error.

* Location: ``DiskCache(root)`` > ``$AITRADING_CACHE_DIR`` > ``~/.aitrading/cache`` (``Path.home()``).
* Disable: ``DiskCache(enabled=False)`` or ``AITRADING_CACHE_DISABLE=1`` (every get misses, puts
  are no-ops).
* Keys are arbitrary strings (URLs, ticker lists ...). File names are a sanitised slug of the key
  plus a SHA-1 digest, so they never contain characters Windows forbids (``<>:"/\\|?*``), never
  collide with reserved device names (``CON``, ``NUL``, ``COM1`` ...), never end in a dot or space
  and stay far below the 260-character ``MAX_PATH`` limit.
* Expiry: ``put_*(..., ttl_s=...)`` stores an expiry (``None`` = never expires); ``get_*(...,
  ttl_s=...)`` can additionally demand an entry be no older than ``ttl_s`` seconds.
* Writes are atomic (temp file + ``os.replace``), so concurrent readers never see partial files.
* DataFrames are stored with :mod:`pickle` (exact dtypes / indexes round-trip). The cache directory
  is private to the user; never point it at a directory other people can write to.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ENV_CACHE_DIR = "AITRADING_CACHE_DIR"
ENV_CACHE_DISABLE = "AITRADING_CACHE_DISABLE"

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_SLUG = 60

# Handy TTLs (seconds)
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0


def default_cache_dir() -> Path:
    env = os.environ.get(ENV_CACHE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".aitrading" / "cache"


def _env_disabled() -> bool:
    return os.environ.get(ENV_CACHE_DISABLE, "").strip().lower() in {"1", "true", "yes", "on"}


def safe_filename(key: str, ext: str = "") -> str:
    """Deterministic, Windows-safe file name for an arbitrary cache key.

    ``<slug>-<sha1[:20]><ext>``; the slug keeps the name human-readable, the digest keeps it unique
    (two keys that differ only by case or punctuation still map to different files, which matters on
    case-insensitive file systems).
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    slug = _SLUG_RE.sub("_", key).strip("_-")[:_MAX_SLUG] or "key"
    return f"{slug}-{digest}{ext}"


class DiskCache:
    """Key -> value store on disk with optional expiry. Thread-safe for concurrent get/put."""

    def __init__(self, root: Path | str | None = None, *, enabled: bool | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root).expanduser() if root is not None else default_cache_dir()
        self.enabled = (not _env_disabled()) if enabled is None else bool(enabled)
        self._clock = clock

    # ------------------------------------------------------------------ paths
    def path_for(self, key: str, ext: str) -> Path:
        # Fan out into 256 sub-directories (first byte of the digest) so no directory gets huge.
        bucket = hashlib.sha1(key.encode("utf-8")).hexdigest()[:2]
        return self.root / bucket / safe_filename(key, ext)

    def _write_atomic(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _fresh(self, stored_at: float, expires_at: float | None, ttl_s: float | None) -> bool:
        now = self._clock()
        if expires_at is not None and now >= expires_at:
            return False
        if ttl_s is not None and now - stored_at > ttl_s:
            return False
        return True

    def _expiry(self, ttl_s: float | None) -> float | None:
        return None if ttl_s is None else self._clock() + float(ttl_s)

    # ------------------------------------------------------------------ JSON
    def get_json(self, key: str, ttl_s: float | None = None) -> Any | None:
        """Cached JSON value, or ``None`` on miss / expiry / corruption / disabled cache."""
        if not self.enabled:
            return None
        path = self.path_for(key, ".json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                env = json.load(fh)
            if not isinstance(env, dict) or "data" not in env:
                return None
            if not self._fresh(float(env.get("stored_at", 0)), env.get("expires_at"), ttl_s):
                return None
            return env["data"]
        except (OSError, ValueError, TypeError):
            return None

    def put_json(self, key: str, obj: Any, ttl_s: float | None = None) -> None:
        if not self.enabled:
            return
        env = {"key": key, "stored_at": self._clock(), "expires_at": self._expiry(ttl_s), "data": obj}
        try:
            payload = json.dumps(env, default=str, allow_nan=True).encode("utf-8")
            self._write_atomic(self.path_for(key, ".json"), payload)
        except (OSError, TypeError, ValueError):
            pass  # a cache that cannot write is just a slower cache

    # ------------------------------------------------------------------ text
    def get_text(self, key: str, ttl_s: float | None = None) -> str | None:
        val = self.get_json(key, ttl_s)
        return val if isinstance(val, str) else None

    def put_text(self, key: str, text: str, ttl_s: float | None = None) -> None:
        self.put_json(key, text, ttl_s)

    # ------------------------------------------------------------------ frames
    def get_frame(self, key: str, ttl_s: float | None = None) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        path = self.path_for(key, ".pkl")
        try:
            with open(path, "rb") as fh:
                env = pickle.load(fh)
            if not isinstance(env, dict) or not isinstance(env.get("frame"), pd.DataFrame):
                return None
            if not self._fresh(float(env.get("stored_at", 0)), env.get("expires_at"), ttl_s):
                return None
            return env["frame"]
        except Exception:  # noqa: BLE001 - unpickling can raise almost anything on a corrupt file
            return None

    def put_frame(self, key: str, frame: pd.DataFrame, ttl_s: float | None = None) -> None:
        if not self.enabled:
            return
        env = {"key": key, "stored_at": self._clock(), "expires_at": self._expiry(ttl_s), "frame": frame}
        try:
            self._write_atomic(self.path_for(key, ".pkl"), pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
        except (OSError, pickle.PicklingError, TypeError, AttributeError):
            pass

    # ------------------------------------------------------------------ maintenance
    def delete(self, key: str) -> None:
        for ext in (".json", ".pkl"):
            try:
                self.path_for(key, ext).unlink()
            except OSError:
                pass

    def clear(self) -> int:
        """Delete every cache file under ``root``; returns the number of files removed."""
        n = 0
        if not self.root.exists():
            return 0
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix in {".json", ".pkl"}:
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DiskCache(root={str(self.root)!r}, enabled={self.enabled})"
