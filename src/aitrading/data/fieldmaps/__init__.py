"""Per-vendor field maps: the data files that hold every vendor field code the adapters use.

ADR-001 rule: vendor field codes and expressions are *configuration*, never constants in adapter
logic. Each vendor ships a default map next to this module (``<vendor>.json``: ``bloomberg.json``,
``lseg.json``, ...) transcribed from ``docs/VENDOR_REFERENCE.md``. Every entry carries the vendor
expression / mnemonic, its units and the scale to canonical units, a ``status`` and ``notes``.

Status vocabulary (VENDOR_REFERENCE "How to read the status column")
--------------------------------------------------------------------
* ``confirmed``    - seen in vendor-authored code or docs in the form shown. May be compiled; must
  still pass the field-admission log (gate G5) before production use as a filter predicate.
* ``corrected``    - verified only in the form shown; use exactly that form.
* ``unverifiable`` - configurable only; preflighted before use; never a push-down threshold.

Push-down compilers may push only entries whose status is in :data:`PUSHABLE_STATUSES`.

Overrides
---------
``load_fieldmap(vendor)`` returns the default map merged with, in order:

1. the JSON file named by the environment variable ``AITRADING_FIELDMAP_<VENDOR>`` (vendor name
   upper-cased, non-alphanumerics replaced by ``_``; e.g. ``AITRADING_FIELDMAP_BLOOMBERG``), then
2. an explicit ``override`` (a mapping, or a path to a JSON file) passed by the caller.

Merging is a deep merge: mappings merge key by key, any other value (lists included) replaces the
default, and ``null`` deletes the key. So an override such as
``{"features": {"fcf_yield_pct": {"status": "confirmed", "units_verified": true, "admitted": true}}}``
changes just those attributes of one entry (typically after the item passed admission, gate G5:
push-down compilers push only features flagged ``"admitted": true`` or passed in their ``admitted=``
argument). The merged map is validated (every ``status`` must be one of :data:`STATUSES`, every
``admitted`` flag a JSON boolean); errors raise :class:`FieldMapError`.

The loader is deliberately schema-agnostic beyond ``vendor`` and ``status``: each adapter documents
the sections it reads. Keys starting with ``_`` are metadata (``_sources`` lists the files merged)
and are ignored by validation and by :func:`digest`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = [
    "STATUSES",
    "PUSHABLE_STATUSES",
    "ENV_PREFIX",
    "FieldMapError",
    "env_var",
    "default_path",
    "available_vendors",
    "merge",
    "validate",
    "load_fieldmap",
    "iter_statuses",
    "is_pushable_status",
    "digest",
]

STATUSES: tuple[str, ...] = ("confirmed", "corrected", "unverifiable")
PUSHABLE_STATUSES: frozenset[str] = frozenset({"confirmed", "corrected"})
ENV_PREFIX = "AITRADING_FIELDMAP_"

_DIR = Path(__file__).resolve().parent


class FieldMapError(ValueError):
    """A field map (default or override) is missing, unreadable or invalid."""


def _vendor_key(vendor: str) -> str:
    v = str(vendor).strip().lower()
    if not v or not re.fullmatch(r"[a-z0-9_\-]+", v):
        raise FieldMapError(f"invalid vendor name {vendor!r}")
    return v


def env_var(vendor: str) -> str:
    """Name of the override environment variable, e.g. ``AITRADING_FIELDMAP_BLOOMBERG``."""
    return ENV_PREFIX + re.sub(r"[^A-Z0-9]", "_", _vendor_key(vendor).upper())


def default_path(vendor: str) -> Path:
    """Path of the default map shipped with the package (it may not exist)."""
    return _DIR / f"{_vendor_key(vendor)}.json"


def available_vendors() -> list[str]:
    """Vendors that ship a default map."""
    return sorted(p.stem for p in _DIR.glob("*.json"))


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FieldMapError(f"{what} not found: {path}") from e
    except OSError as e:
        raise FieldMapError(f"{what} unreadable: {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise FieldMapError(f"{what} is not valid JSON: {path}: {e}") from e
    if not isinstance(data, dict):
        raise FieldMapError(f"{what} must be a JSON object at the top level: {path}")
    return data


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge ``override`` over ``base`` (mappings merge, other values replace, None deletes)."""
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, val in override.items():
        if val is None:
            out.pop(key, None)
        elif isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def iter_statuses(node: Any, path: str = "") -> Iterator[tuple[str, str, Any]]:
    """Yield ``(path, key, value)`` for every ``status`` / ``*_status`` key in the map."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            if str(k).startswith("_"):
                continue
            here = f"{path}.{k}" if path else str(k)
            if (k == "status" or str(k).endswith("_status")) and not isinstance(v, (Mapping, list)):
                yield here, str(k), v
            else:
                yield from iter_statuses(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_statuses(v, f"{path}[{i}]")


def _iter_key(node: Any, key: str, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(path, value)`` for every ``key`` in the map (metadata keys starting with ``_`` skipped)."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            if str(k).startswith("_"):
                continue
            here = f"{path}.{k}" if path else str(k)
            if k == key:
                yield here, v
            else:
                yield from _iter_key(v, key, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_key(v, key, f"{path}[{i}]")


def validate(fieldmap: Mapping[str, Any], *, vendor: str | None = None, source: str = "") -> None:
    """Raise :class:`FieldMapError` listing every problem (bad ``vendor``, unknown status, non-boolean ``admitted``)."""
    where = f" ({source})" if source else ""
    if not isinstance(fieldmap, Mapping):
        raise FieldMapError(f"field map must be a mapping{where}")
    errors: list[str] = []
    if vendor is not None and "vendor" in fieldmap and str(fieldmap["vendor"]).lower() != _vendor_key(vendor):
        errors.append(f"map is for vendor {fieldmap['vendor']!r}, expected {vendor!r}")
    for path, _key, value in iter_statuses(fieldmap):
        if value not in STATUSES:
            errors.append(f"{path}: status {value!r} is not one of {', '.join(STATUSES)}")
    for path, value in _iter_key(fieldmap, "admitted"):
        if not isinstance(value, bool):  # a string "true" would otherwise silently leave the item unadmitted
            errors.append(f"{path}: admitted must be true or false, got {value!r}")
    if errors:
        raise FieldMapError(f"invalid field map{where}: " + "; ".join(errors))


def load_fieldmap(
    vendor: str,
    override: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Default map for ``vendor`` merged with the env-var file and ``override`` (see module doc).

    Args:
        vendor: e.g. ``"bloomberg"``, ``"lseg"``.
        override: a mapping, or a path to a JSON file, merged last.
        env: environment to read ``AITRADING_FIELDMAP_<VENDOR>`` from (default ``os.environ``).

    Returns a fresh dict (safe to mutate) with ``_sources`` listing what was merged.
    """
    key = _vendor_key(vendor)
    path = default_path(key)
    fm = _read_json(path, f"default field map for {key!r}")
    validate(fm, vendor=key, source=str(path))
    sources = [str(path)]

    environ = os.environ if env is None else env
    var = env_var(key)
    env_file = environ.get(var)
    if env_file:
        p = Path(env_file).expanduser()
        data = _read_json(p, f"field map override from ${var}")
        fm = merge(fm, data)
        validate(fm, vendor=key, source=f"after ${var}={p}")
        sources.append(str(p))

    if override is not None:
        if isinstance(override, Mapping):
            data, label = dict(override), "override:<mapping>"
        else:
            p = Path(override).expanduser()
            data, label = _read_json(p, "field map override"), str(p)
        fm = merge(fm, data)
        validate(fm, vendor=key, source=f"after {label}")
        sources.append(label)

    fm["_sources"] = sources
    return fm


def is_pushable_status(status: Any) -> bool:
    """True for ``confirmed`` / ``corrected`` (the only statuses a push-down compiler may push)."""
    return status in PUSHABLE_STATUSES


def digest(fieldmap: Mapping[str, Any]) -> str:
    """SHA-256 of the map's canonical JSON (metadata keys starting with ``_`` excluded), for audit records."""
    clean = {k: v for k, v in fieldmap.items() if not str(k).startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
