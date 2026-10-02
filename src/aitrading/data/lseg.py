"""LSEG adapter over the LSEG Data Library for Python (``lseg-data`` 2.1.1, ``import lseg.data as ld``).

Implements ``MarketDataProvider`` and ``ScreenPushdown`` (docs/ARCHITECTURE.md Option C, phase 3).
The SDK is imported lazily; without it every data call raises ``ProviderUnavailable`` with install and
entitlement guidance. No vendor code is hard-coded in this module: every field code, unit conversion,
SCREEN fragment and history parameter comes from the field map data file
``src/aitrading/data/fieldmaps/lseg.json`` (see :func:`load_fieldmap`).

Sessions (VENDOR_REFERENCE 2.1)
-------------------------------
The session is opened lazily on the first data call with ``ld.open_session(name=..., app_key=...)``:

* **Platform** (the default, ``session_name='platform.ldp'``): unattended server runs with an LSEG Data
  Platform service account (v1 username/password or v2 ``client_id``/``client_secret``) configured in
  ``lseg-data.config.json``. The library looks for it in ``$LD_LIB_CONFIG_PATH``, then the working
  directory, then ``~`` (not next to the script). Content is licence class L3.
* **Desktop** (opt-in only: ``session_name='desktop.workspace'``, the *library's* default): talks to LSEG
  Workspace running on this machine. The app key is read from ``$LSEG_APP_KEY`` (``app_key_env``) or the
  config file. Workspace is licensed for individual use (licence class L4): it **must not run on a
  server**, and lifting Workspace ``edp-token``s is prohibited (ADR section 7). The adapter never opens
  it unless it is named, says so in ``warnings``, tags its documents L4 and refuses to widen its boundary.

Requests are paced (``request_pause_s``) and counted against ``daily_request_cap`` (Workspace allows
10k requests/day; Platform limits are UNVERIFIED), which raises ``ProviderError`` when exhausted.
Session/authentication/quota errors stop at once (no retries); throttling errors back off
exponentially; a failed ``get_history`` batch is retried one RIC at a time only until a circuit breaker
sees the same error several times in a row.

The HTTP request timeout is raised to 300 s (``ld.get_config().set_param('http.request-timeout', 300)``).

Data
----
* ``get_universe`` / ``pushdown_screen``: ``ld.get_data(universe='SCREEN(...)', ...)``; the SCREEN string is
  built by :mod:`aitrading.screen.compile_lseg` from the field map and recorded verbatim
  (``PushdownResult.query``, ``last_pushdown``). RICs map to canonical tickers by stripping the exchange
  suffix (``AAPL.O`` -> ``AAPL``, ``IBM.N`` -> ``IBM``; a trailing lower-case share class becomes
  ``-X``: ``BRKb.N`` -> ``BRK-B``; a delisting suffix ``^..`` is dropped; index RICs such as ``.SPX`` stay
  as they are). The RIC is kept as ``vendor_id``. Two RICs with the same root keep the first; the
  other uses its full RIC as ticker (warned). Tickers this session has not seen are used as RICs only
  when they are RIC-shaped (an index RIC such as ``.SPX``, or ``<root>.<suffix>`` with an exchange
  suffix from the field map's ``symbology.ric_suffixes``); anything else - including dotted
  share-class tickers such as ``BRK.B``, whose RIC is ``BRKb.N`` - is dropped and flagged unless
  ``rics={ticker: RIC}`` (the maintained symbology table, ADR graft 6) supplies it.
* Snapshots (fundamentals, estimates, short interest, options): ``ld.get_data(rics, codes,
  parameters={'Curn': 'USD', 'SDate': as_of})``, chunked to ~8,000 data points per request. LSEG
  drops bad or unentitled fields silently, so the returned column count is asserted on every response;
  on a mismatch the chunk is re-requested one field at a time and the dropped fields become
  ``LEG NOT EVALUATED`` (NaN plus a warning); later chunks of the same request skip them. When an
  instrument comes back on several rows the first row is kept whole (values are never combined across
  rows) and a warning names it.
* Units: the field map's ``to_canonical`` factor converts to canonical units (``Scale=6`` USD millions
  x 1e6 -> USD absolute; implied-vol points x 0.01 -> fraction). Missing values never become numbers
  (no ``fillna``); non-numeric and infinite values are NaN.
* ``unverifiable`` field-map entries are preflighted on the field map's ``test_ric`` (``IBM.N``) once a
  day per provider, **with the same request parameters as the real request**: the field must come back
  as one non-empty column. A failure suspends the leg (NaN plus a ``LEG NOT EVALUATED`` warning).
  Unverifiable request parameters (the global ``SDate`` and the history ``adjustments``) are preflighted
  the same way (with vs without); a failing one is dropped, warned and logged, and a historical
  ``as_of`` that lost its ``SDate`` anchor gets NaN snapshots rather than current values. Entries
  flagged ``units_verified: false`` are not served at all until an override marks them admitted.
  ``field_log`` records what was used, for the audit trail.
* Push-down (gate G5): only features admitted through the field map (``"admitted": true``) or the
  ``admitted=`` constructor argument are pushed as SCREEN thresholds; the default admits nothing.
* Prices: ``ld.get_history(rics, [TRDPRC_1, ACVOL_UNS, ...], interval='daily', start, end,
  adjustments=[...])``. Technical features are computed locally from this history (the reference's
  recommendation). Whether ``TRDPRC_1`` is fully corporate-action adjusted is UNVERIFIED: a warning
  says so and :meth:`LSEGProvider.crosscheck_adjustment` compares the local 52-week high with
  ``TR.Price52WeekHigh``.
* Fundamentals are thin by design: only fields the reference lists. ``revenue_last_q`` vs
  ``revenue_last_q_prior_year`` is a *single-quarter* YoY (FQ0 vs FQ-4), not LTM; TTM revenue,
  EBITDA, debt and cash are not mapped (NaN). Users can add entries through an override file.
* Documents: news via ``ld.news.get_headlines`` / ``get_story`` (licence class L3, Reuters copyright;
  L4 through a desktop session), paced like every other request; filings via
  ``lseg.data.content.filings`` only when enabled in the field map (its search parameters are not in
  the reference); transcripts (StreetEvents XML over SFTP) are out of scope: none are returned and a
  warning says so. Each document's metadata carries the session and its licence class.

Licensing boundary (ADR section 7)
----------------------------------
The default boundary is ``deny_all_text('lseg')`` with ``allow_numeric_features=False``: LSEG Platform
content (I/B/E/S, StarMine, Reuters, StreetEvents) is licence class L3 - only ranks and booleans may
reach Claude until gate **G3** (written per-dataset AI-use confirmation) - and a Workspace desktop
session is L4 (nothing leaves the workstation). With a ``platform.*`` session, pass
``boundary=DataBoundary(provider='lseg', ..., note='G3: <reference>, <YYYY-MM-DD>, reviewed by <lawyer>')``
to widen it. The note must *start* with ``G3:`` and give the confirmation reference, its date (not in
the future) and the reviewing lawyer, with no ``<placeholder>`` and no "pending"/"not received"
wording; anything else - including this module's own ``BOUNDARY_NOTE`` - is rejected (a policy row
without a citation evaluates as deny, ADR section 7). A desktop session's boundary cannot be widened
at all (L4; G3 covers only L3 loosening).

Field map overrides: ``$AITRADING_FIELDMAP_LSEG`` names a JSON file deep-merged over the default
(a ``null`` value deletes a key), then the ``fieldmap=`` constructor argument (a mapping or a path) is
merged on top. ``status`` must be one of confirmed / corrected / unverifiable.
"""

from __future__ import annotations

import copy
import html
import importlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary, deny_all_text
from aitrading.data.base import Capability, PricePanel, ProviderError, ProviderUnavailable, PushdownResult
from aitrading.screen.catalog import default_catalog
from aitrading.screen.compile_lseg import (
    SCREEN_MODES,
    VALID_STATUSES,
    CompiledQuery,
    compile_screen,
    compile_universe,
    is_point_in_time,
    preflight_codes,
)

__all__ = [
    "LSEGProvider",
    "FieldCheck",
    "load_fieldmap",
    "validate_fieldmap",
    "ric_to_ticker",
    "FIELDMAP_ENV",
    "DEFAULT_FIELDMAP_PATH",
    "BOUNDARY_NOTE",
    "DEFAULT_SESSION",
]

FIELDMAP_ENV = "AITRADING_FIELDMAP_LSEG"
DEFAULT_FIELDMAP_PATH = Path(__file__).with_name("fieldmaps") / "lseg.json"
CONFIG_FILENAME = "lseg-data.config.json"
DEFAULT_SESSION = "platform.ldp"  # VENDOR_REFERENCE 2.1: the unattended/server session
DESKTOP_LICENCE_CLASS = "L4"  # ADR section 7: LSEG Workspace is desktop-licensed
# Exchange suffixes accepted on a bare RIC when the field map lists none. 'A' (NYSE American) and 'U' are left out:
# they collide with dotted share-class / unit tickers ('BRK.A', 'XYZ.U'), which must never be sent as RICs.
DEFAULT_RIC_SUFFIXES = ("N", "O", "OQ", "P", "K", "PK", "Z")

SDK_MISSING = (
    "The LSEG Data Library for Python is not installed. Install it with:  pip install lseg-data==2.1.1   "
    '(or: pip install -e ".[lseg]"). It also needs an LSEG entitlement: either LSEG Workspace running on '
    "this machine (desktop session, individual licence, never on a server; app key in $LSEG_APP_KEY) or an "
    "LSEG Data Platform service account configured in lseg-data.config.json ($LD_LIB_CONFIG_PATH, then the "
    "working directory, then ~) used with session_name='platform.ldp' (the default)."
)
SESSION_HELP = (
    "Platform session (default, session_name='platform.ldp'): provide lseg-data.config.json "
    "($LD_LIB_CONFIG_PATH, then the working directory, then ~) with the service-account credentials. "
    "Desktop session (session_name='desktop.workspace', interactive use only, never on a server): start LSEG "
    "Workspace on this machine and set the app key (App Key Generator) in $LSEG_APP_KEY."
)
DESKTOP_WARNING = (
    "LSEG desktop session ({name}): licence class L4 (LSEG Workspace, individual use). It must not run on a server "
    "or unattended (ADR section 7); nothing derived from it may reach an external model and its boundary cannot be "
    "widened. Use session_name='platform.ldp' for scheduled runs."
)
BOUNDARY_NOTE = (
    "LSEG content is licence class L3 (Platform: I/B/E/S, StarMine, Reuters, StreetEvents) or L4 (Workspace "
    "desktop session): no LSEG text or values may reach an external LLM - only ranks and booleans for L3 - "
    "until gate G3 (written per-dataset AI-use confirmation). Widen only for a platform session, with an explicit "
    "boundary=DataBoundary(provider='lseg', ..., note='G3: <confirmation reference>, <YYYY-MM-DD>, reviewed by "
    "<lawyer>')."
)

# Canonical columns each raw field-map section may map (vendor_id is always the RIC).
RAW_SECTIONS: dict[str, list[str]] = {
    "universe": [c for c in F.UNIVERSE_COLUMNS if c != F.VENDOR_ID],
    "fundamentals": list(F.FUNDAMENTAL_COLUMNS),
    "estimates": list(F.ESTIMATE_COLUMNS),
    "short_interest": list(F.SHORT_INTEREST_COLUMNS),
    "options": list(F.OPTIONS_COLUMNS),
}
_DATE_COLUMNS = {F.PERIOD_END, F.REPORT_DATE, F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE, F.SI_SETTLEMENT_DATE}
_LABEL_COLUMNS = {F.NAME, F.GICS_SECTOR, F.GICS_INDUSTRY, F.EXCHANGE, F.COUNTRY, F.CURRENCY, F.SECURITY_TYPE, F.VENDOR_ID}

# A widened boundary's note is a policy-table citation (ADR section 7): "G3: <reference>, <date>, reviewed by <lawyer>".
_CITATION_RE = re.compile(r"^\s*G3\s*:\s*(?P<body>\S.*)$", re.S)
_CITATION_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CITATION_REVIEWER_RE = re.compile(r"\breview(?:ed\s+by|er\s*:)\s*[A-Za-z]", re.I)
_CITATION_NOT_YET_RE = re.compile(
    r"^(?:pending|awaiting|requested|request(?:ing)?\b|draft|tbd|todo|to\s+be\b|not\b|no\b|none\b|n/?a\b|\?|"
    r"\(?\s*placeholder)", re.I)
_CITATION_NOT_RECEIVED_RE = re.compile(
    r"\b(?:pending|awaiting|tbd|todo|not\s+(?:yet\s+)?(?:received|signed|confirmed|obtained|granted)|"
    r"to\s+be\s+(?:confirmed|received|signed|obtained))\b|\?\?", re.I)

# Error classes that must not be retried request by request (they burn the vendor's request quota).
_QUOTA_ERROR_RE = re.compile(r"quota|daily (?:request )?limit|requests? per day|limit of \d+ exceeded", re.I)
_THROTTLE_ERROR_RE = re.compile(r"\b429\b|too many requests|rate[ -]?limit|throttl", re.I)
_SESSION_ERROR_RE = re.compile(
    r"session (?:is )?(?:closed|expired|not open(?:ed)?|invalid)|no (?:default |open )?session|session expired|"
    r"unauthori[sz]ed|\b401\b|authenticat|invalid[_ ]grant|credential|access token|login", re.I)
# (a 403 / "forbidden" is usually one unentitled dataset, not the session: it is bounded by the circuit breaker)


# =============================================================================================
# Field map
# =============================================================================================


def _read_json(path: str | Path, what: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ValueError(f"{what}: field map file {p} does not exist") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{what}: field map file {p} is not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{what}: field map file {p} must contain a JSON object")
    return data


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge: dicts merge, other values replace, ``None`` deletes the key."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(dict(v) if isinstance(v, Mapping) else v)
    return out


def _walk_statuses(node: Any, path: str, errs: list[str]) -> None:
    if isinstance(node, Mapping):
        for key in ("status", "fallback_status"):
            if key in node and node[key] not in VALID_STATUSES:
                errs.append(f"{path}.{key} = {node[key]!r} (must be one of {sorted(VALID_STATUSES)})")
        for k, v in node.items():
            _walk_statuses(v, f"{path}.{k}" if path else str(k), errs)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_statuses(v, f"{path}[{i}]", errs)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def validate_fieldmap(fm: Mapping[str, Any]) -> list[str]:
    """Human-readable problems with a (merged) LSEG field map; empty list means usable."""
    errs: list[str] = []
    if fm.get("vendor") != "lseg":
        errs.append(f"vendor must be 'lseg' (got {fm.get('vendor')!r})")
    _walk_statuses({k: v for k, v in fm.items() if not str(k).startswith("_") and k != "status_meaning"}, "", errs)
    catalog = default_catalog()
    for feat, e in (fm.get("features") or {}).items():
        if str(feat).startswith("_"):
            continue
        if feat not in catalog:
            errs.append(f"features.{feat}: not a catalog feature")
        if not isinstance(e, Mapping):
            errs.append(f"features.{feat}: must be an object")
            continue
        if "status" not in e:
            errs.append(f"features.{feat}: missing status")
        if e.get("expression") is not None and not isinstance(e.get("expression"), str):
            errs.append(f"features.{feat}.expression must be a string or null")
        if "screen" in e and e["screen"] not in SCREEN_MODES:
            errs.append(f"features.{feat}.screen = {e['screen']!r} (must be one of {sorted(SCREEN_MODES)})")
        if "threshold_scale" in e and (not _is_number(e["threshold_scale"]) or float(e["threshold_scale"]) == 0):
            errs.append(f"features.{feat}.threshold_scale must be a non-zero number")
        if "labels" in e and not (isinstance(e["labels"], list) and all(isinstance(x, str) for x in e["labels"])):
            errs.append(f"features.{feat}.labels must be a list of strings")
        if "admitted" in e and not isinstance(e["admitted"], bool):
            errs.append(f"features.{feat}.admitted must be true or false (gate G5 admission flag)")
    for section, entries in (fm.get("raw") or {}).items():
        allowed = RAW_SECTIONS.get(section)
        if allowed is None:
            errs.append(f"raw.{section}: unknown section (expected one of {sorted(RAW_SECTIONS)})")
            continue
        for key, e in (entries or {}).items():
            if str(key).startswith("_"):
                continue
            if key not in allowed:
                errs.append(f"raw.{section}.{key}: not a canonical column of this dataset")
            if not isinstance(e, Mapping):
                errs.append(f"raw.{section}.{key}: must be an object")
                continue
            if "status" not in e:
                errs.append(f"raw.{section}.{key}: missing status")
            if "to_canonical" in e and not _is_number(e["to_canonical"]):
                errs.append(f"raw.{section}.{key}.to_canonical must be a number")
            if not (e.get("code") or e.get("codes")):
                errs.append(f"raw.{section}.{key}: needs 'code' (or 'codes' + 'ric_template')")
    for name, e in (fm.get("parameters") or {}).items():
        if str(name).startswith("_"):
            continue
        if isinstance(e, Mapping) and e.get("status") == "unverifiable" and e.get("value") is not None \
                and not isinstance(e.get("probe"), str):
            errs.append(f"parameters.{name}: an unverifiable parameter needs a 'probe' field code (it is preflighted)")
    suffixes = (fm.get("symbology") or {}).get("ric_suffixes")
    if suffixes is not None and not (isinstance(suffixes, list) and all(isinstance(s, str) and s for s in suffixes)):
        errs.append("symbology.ric_suffixes must be a list of non-empty strings")
    sess = (fm.get("sessions") or {}).get("default")
    if sess is not None and not (isinstance(sess, str) and sess.strip()):
        errs.append("sessions.default must be a session name such as 'platform.ldp'")
    hist = fm.get("history") or {}
    for key, e in (hist.get("fields") or {}).items():
        if key not in F.PRICE_FIELDS:
            errs.append(f"history.fields.{key}: not one of {F.PRICE_FIELDS}")
        elif not isinstance(e, Mapping) or not e.get("code") or "status" not in e:
            errs.append(f"history.fields.{key}: needs 'code' and 'status'")
    if not isinstance((hist.get("fields") or {}).get(F.CLOSE), Mapping):
        errs.append("history.fields.close is required")
    scr = fm.get("screen") or {}
    if not isinstance(scr.get("universe"), Mapping) or not scr["universe"].get("expression"):
        errs.append("screen.universe.expression is required")
    return errs


def load_fieldmap(override: Mapping[str, Any] | str | Path | None = None, *,
                  env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Default LSEG field map, deep-merged with ``$AITRADING_FIELDMAP_LSEG`` and then ``override``.

    Raises ValueError (listing every problem) when the merged map is invalid.
    """
    env = os.environ if env is None else env
    fm = _read_json(DEFAULT_FIELDMAP_PATH, "default")
    sources = [str(DEFAULT_FIELDMAP_PATH)]
    env_path = (env.get(FIELDMAP_ENV) or "").strip()
    if env_path:
        fm = _deep_merge(fm, _read_json(env_path, f"${FIELDMAP_ENV}"))
        sources.append(env_path)
    if override is not None:
        if isinstance(override, (str, Path)):
            fm = _deep_merge(fm, _read_json(override, "fieldmap="))
            sources.append(str(override))
        elif isinstance(override, Mapping):
            fm = _deep_merge(fm, override)
            sources.append("<fieldmap= mapping>")
        else:
            raise TypeError("fieldmap must be a mapping, a path to a JSON file, or None")
    errs = validate_fieldmap(fm)
    if errs:
        raise ValueError(f"invalid LSEG field map ({' <- '.join(sources)}): " + "; ".join(errs))
    fm["_sources"] = sources
    return fm


# =============================================================================================
# Pure helpers
# =============================================================================================


def ric_to_ticker(ric: str) -> str:
    """Canonical ticker for a RIC: 'AAPL.O' -> 'AAPL', 'BRKb.N' -> 'BRK-B', 'XYZ.N^K20' -> 'XYZ', '.SPX' -> '.SPX'."""
    s = str(ric).strip()
    core = s.split("^", 1)[0]
    root, dot, _suffix = core.rpartition(".")
    if not dot:
        root = core
    if not root:
        return s  # index / chain RICs ('.SPX') have no root: keep the RIC
    m = re.fullmatch(r"([A-Z0-9]+)([a-z]{1,2})", root)
    if m:
        root = f"{m.group(1)}-{m.group(2).upper()}"
    return root.upper()


def _ric_root(ric: str) -> str:
    core = str(ric).split("^", 1)[0]
    root = core.rpartition(".")[0]
    return root or core


def _looks_like_ric(s: str, suffixes: Iterable[str] = DEFAULT_RIC_SUFFIXES) -> bool:
    """An index RIC ('.SPX') or ``<root>.<known exchange suffix>`` (optional '^' delisting tail).

    Dotted share-class tickers such as 'BRK.B' are *not* RICs (the RIC is 'BRKb.N').
    """
    core = str(s).strip().split("^", 1)[0]
    if len(core) > 1 and core.startswith(".") and "." not in core[1:]:
        return True
    root, dot, suffix = core.rpartition(".")
    return bool(dot and root and not root.startswith(".") and suffix in set(suffixes))


def _error_kind(e: BaseException) -> str | None:
    """'quota' / 'throttle' / 'session' for errors that are systemic rather than per request, else None."""
    msg = f"{type(e).__name__}: {e}"
    if _QUOTA_ERROR_RE.search(msg):
        return "quota"
    if _THROTTLE_ERROR_RE.search(msg):
        return "throttle"
    if _SESSION_ERROR_RE.search(msg):
        return "session"
    return None


def _error_signature(msg: str, ric: str) -> str:
    """Error text with the RIC and numbers masked, to spot the same failure repeating across RICs."""
    return re.sub(r"\d+", "#", str(msg).replace(ric, "<RIC>")).strip().lower()


def _citation_problem(note: str, today: date) -> str | None:
    """None when ``note`` is a structured G3 citation (reference, date <= today, reviewer), else the problem."""
    m = _CITATION_RE.match(note or "")
    if m is None:
        return "the note must start with 'G3:' followed by the written AI-use confirmation it relies on"
    body = m.group("body").strip()
    if "<" in body or ">" in body:
        return "the citation still contains a <placeholder>"
    if _CITATION_NOT_YET_RE.match(body) or _CITATION_NOT_RECEIVED_RE.search(body):
        return "the citation says the confirmation has not been received"
    dates = []
    for y, mo, d in _CITATION_DATE_RE.findall(body):
        try:
            dates.append(date(int(y), int(mo), int(d)))
        except ValueError:
            continue
    if not any(d <= today for d in dates):
        return "the citation must give the confirmation date as YYYY-MM-DD (not in the future)"
    if not _CITATION_REVIEWER_RE.search(body):
        return "the citation must name the reviewing lawyer ('reviewed by <name>')"
    return None


def _params_key(params: Mapping[str, Any] | None) -> str:
    return json.dumps(sorted((str(k), str(v)) for k, v in (params or {}).items()))


def _render(template: str, ctx: Mapping[str, Any]) -> str:
    try:
        return str(template).format(**ctx)
    except (KeyError, IndexError, ValueError) as e:
        raise ProviderError(f"field map template {template!r} uses an unknown placeholder ({e})") from e


def _missing(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "<na>", "nat", "null"):
        return True
    return False


def _py(v: Any) -> Any:
    """numpy scalar -> native Python value (readable reprs in notes / audit records)."""
    return v.item() if isinstance(v, np.generic) else v


def _to_float(v: Any) -> float:
    if _missing(v) or isinstance(v, bool):
        return math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _to_ts(v: Any) -> pd.Timestamp:
    if _missing(v):
        return pd.NaT
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return pd.NaT
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _convert(values: pd.Series, entry: Mapping[str, Any]) -> pd.Series:
    """Vendor values -> canonical values (numbers x to_canonical, labels via value_map, dates naive)."""
    kind = entry.get("kind", "number")
    idx = values.index
    if kind == "date":
        out = pd.Series([_to_ts(v) for v in values], index=idx, dtype="object")
        return pd.to_datetime(out, errors="coerce").astype("datetime64[ns]")
    if kind == "label":
        vmap = {str(k).strip().upper(): v for k, v in (entry.get("value_map") or {}).items()}

        def lab(v: Any) -> Any:
            if _missing(v):
                return np.nan
            t = str(v).strip()
            return vmap.get(t.upper(), t) if vmap else t

        return pd.Series([lab(v) for v in values], index=idx, dtype="object")
    scale = float(entry.get("to_canonical", 1.0))
    return pd.Series([_to_float(v) * scale for v in values], index=idx, dtype="float64")


def _assemble(columns: list[str], tickers: list[str], data: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Canonical frame: every column present, numbers float64, dates datetime64[ns], labels object."""
    idx = pd.Index(list(tickers), name="ticker")
    out: dict[str, pd.Series] = {}
    for c in columns:
        s = data.get(c)
        s = s.reindex(idx) if s is not None else pd.Series(np.nan, index=idx, dtype="object")
        if c in _DATE_COLUMNS:
            out[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
        elif c in _LABEL_COLUMNS:
            o = s.astype(object)
            out[c] = o.where(o.notna(), np.nan)
        else:
            out[c] = pd.Series([_to_float(v) for v in s], index=idx, dtype="float64")
    return pd.DataFrame(out, index=idx)[columns]


def _unique(items: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(t).strip() for t in items if t is not None and str(t).strip()))


def _norm_name(s: Any) -> str:
    return re.sub(r"[\s_]", "", str(s)).lower()


def _col(df: pd.DataFrame, *names: str) -> str | None:
    wanted = {_norm_name(n) for n in names}
    for c in df.columns:
        if _norm_name(c) in wanted:
            return c
    return None


def _html_to_text(raw: Any) -> str:
    s = "" if raw is None else str(raw)
    if "<" in s and ">" in s:
        s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
        s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", s)
        s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def _daily_index(index: pd.Index) -> pd.DatetimeIndex:
    ts = pd.to_datetime(pd.Index(index), errors="coerce")
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(ts).normalize()


def _split_history(raw: Any, rics: list[str], codes: list[str]) -> dict[str, pd.DataFrame]:
    """get_history output -> {RIC: frame(date x requested code)}.

    Handles the three layouts the library returns: MultiIndex (RIC, field) columns for several RICs,
    flat field columns for a single RIC, flat RIC columns for a single field.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return {}
    df = raw.copy()
    df.index = _daily_index(df.index)
    df = df[df.index.notna()]
    per: dict[str, pd.DataFrame] = {}
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        lv0 = {str(x) for x in cols.get_level_values(0)}
        level = 0 if any(r in lv0 for r in rics) else 1
        present = {str(x) for x in cols.get_level_values(level)}
        for r in rics:
            if r in present:
                sub = df.xs(r, axis=1, level=level)
                sub.columns = [str(c) for c in sub.columns]
                per[r] = sub
    elif len(rics) == 1:
        sub = df.copy()
        sub.columns = [str(c) for c in sub.columns]
        per[rics[0]] = sub
    elif len(codes) == 1:
        for r in rics:
            if r in [str(c) for c in cols]:
                per[r] = df[[c for c in cols if str(c) == r]].set_axis([codes[0]], axis=1)
    else:
        raise ProviderError(f"unexpected get_history column layout: {[str(c) for c in list(cols)[:10]]}")
    out: dict[str, pd.DataFrame] = {}
    for r, sub in per.items():
        lookup = {_norm_name(c): c for c in sub.columns}
        frame = pd.DataFrame(index=sub.index)
        for code in codes:
            c = lookup.get(_norm_name(code))
            frame[code] = pd.to_numeric(sub[c], errors="coerce").astype("float64") if c is not None else np.nan
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        out[r] = frame
    return out


def _find_config(env: Mapping[str, str] | None = None) -> Path | None:
    env = os.environ if env is None else env
    candidates = []
    if env.get("LD_LIB_CONFIG_PATH"):
        candidates.append(Path(env["LD_LIB_CONFIG_PATH"]).expanduser() / CONFIG_FILENAME)
    candidates += [Path.cwd() / CONFIG_FILENAME, Path.home() / CONFIG_FILENAME]
    return next((p for p in candidates if p.is_file()), None)


class _StopRequests(ProviderError):
    """A systemic request failure (daily cap, quota, session / authentication, persistent throttling):
    never retried request by request and never swallowed as a failed preflight or a per-RIC failure."""


# =============================================================================================
# Field self-check result
# =============================================================================================


@dataclass(frozen=True)
class FieldCheck:
    """One field-admission probe (VENDOR_REFERENCE section 5) on a known security."""

    field: str  # '<section>.<canonical key>', 'feature.<catalog feature>' or 'history.<price field>'
    status_in_map: str  # confirmed / corrected / unverifiable
    returned_value: Any  # raw vendor value (before unit conversion); None when nothing came back
    ok: bool  # one non-empty column / series came back
    note: str  # code, units, canonical conversion, or the failure
    code: str = ""  # the vendor code / expression requested


# =============================================================================================
# Provider
# =============================================================================================


class LSEGProvider:
    """``MarketDataProvider`` + ``ScreenPushdown`` over the LSEG Data Library for Python."""

    name = "lseg"

    def __init__(
        self,
        *,
        boundary: DataBoundary | None = None,
        fieldmap: Mapping[str, Any] | str | Path | None = None,
        session_name: str | None = None,
        app_key_env: str = "LSEG_APP_KEY",
        benchmark: str = ".SPX",
        ld_module: Any | None = None,
        rics: Mapping[str, str] | None = None,
        today: Callable[[], date] | None = None,
        admitted: Iterable[str] | None = None,
        daily_request_cap: int | None = 10_000,
    ) -> None:
        """
        Args:
            boundary: licensing boundary; default denies all LSEG text and values (see module docs).
                A widened boundary needs a ``platform.*`` session, ``provider='lseg'`` and a structured
                G3 citation in its ``note`` (``'G3: <reference>, <YYYY-MM-DD>, reviewed by <lawyer>'``).
            fieldmap: overrides merged over the default field map (+ ``$AITRADING_FIELDMAP_LSEG``): a
                mapping or a path to a JSON file.
            session_name: ``None`` = the field map's ``sessions.default`` (``'platform.ldp'``, the
                unattended session). A desktop session (``'desktop.workspace'``, licence class L4, never on
                a server) is opened only when named here.
            app_key_env: environment variable holding the app key (never logged).
            benchmark: RIC for ``get_benchmark_history`` (default ``.SPX``; falls back to the field
                map's ``benchmark.fallback_ric`` when unentitled).
            ld_module: inject an ``lseg.data``-compatible module (tests / custom sessions).
            rics: explicit ticker -> RIC table (symbology, ADR graft 6).
            today: clock for the point-in-time, preflight, citation and request-cap rules (tests).
            admitted: gate G5 admission set from the run's ``field_validation_log`` (catalog feature
                names). Only admitted features - these plus any ``"admitted": true`` in the field map -
                are pushed as SCREEN thresholds; the default admits nothing.
            daily_request_cap: vendor requests this provider may make per day before it raises
                ``ProviderError`` (Workspace allows 10k/day; Platform limits are UNVERIFIED). None = no cap.
        """
        self.fieldmap: dict[str, Any] = load_fieldmap(fieldmap)
        default_session = str((self.fieldmap.get("sessions") or {}).get("default") or DEFAULT_SESSION)
        self.session_name: str = str(session_name).strip() if session_name else default_session
        self.app_key_env = app_key_env
        self.benchmark = benchmark
        self._today = today or date.today
        if isinstance(admitted, str):
            raise TypeError("admitted must be a collection of feature names, not a string")
        self.admitted: frozenset[str] | None = frozenset(str(a) for a in admitted) if admitted is not None else None
        self.boundary = self._resolve_boundary(boundary)
        self.capabilities: set[Capability] = {
            Capability.PRICES, Capability.FUNDAMENTALS, Capability.ESTIMATES, Capability.SHORT_INTEREST,
            Capability.OPTIONS, Capability.SCREEN_PUSHDOWN,
        }
        docs = self.fieldmap.get("documents") or {}
        if (docs.get("news") or {}).get("enabled"):
            self.capabilities.add(Capability.NEWS)
        if (docs.get("filings") or {}).get("enabled"):
            self.capabilities.add(Capability.FILINGS)
        self.test_ric: str = str(self.fieldmap.get("test_ric") or "IBM.N")
        self.warnings: list[str] = []
        self.field_log: dict[str, dict[str, Any]] = {}
        self.last_pushdown: CompiledQuery | None = None
        self.last_universe_query: str | None = None
        self.request_pause_s = 0.25  # between requests (Workspace allows ~5 req/s)
        self.max_points = 8000  # data points per get_data chunk (reference get_tr)
        self.history_batch_size = 50
        self.daily_request_cap = daily_request_cap
        self.request_count = 0  # vendor requests made today (reset at the first request of a new day)
        self.throttle_retries = 3  # a throttled request is retried after 1 s, 2 s, 4 s (x backoff_base_s)
        self.backoff_base_s = 1.0
        self.circuit_breaker_threshold = 3  # identical single-RIC history failures in a row that stop the fallback
        self._request_day: date | None = None
        self._ld = ld_module
        self._session: Any = None
        self._sleep = time.sleep
        self._ric_of: dict[str, str] = {}
        self._ticker_of: dict[str, str] = {}
        self._names: dict[str, str] = {}
        self._preflight_memo: dict[tuple[str, ...], bool] = {}
        self._param_memo: dict[tuple[str, ...], tuple[bool, dict[str, Any]]] = {}
        self._preflight_day: date | None = None
        for t, r in (rics or {}).items():
            self._ric_of[str(t).strip()] = str(r).strip()
            self._ticker_of[str(r).strip()] = str(t).strip()

    # ------------------------------------------------------------------ housekeeping
    @property
    def is_desktop(self) -> bool:
        """True unless the session is a ``platform.*`` session (desktop / unknown kinds are L4, deny by default)."""
        return self.session_name.split(".", 1)[0].strip().lower() != "platform"

    @property
    def licence_class(self) -> str:
        """Licence class of this session's content: L4 for a desktop (Workspace) session, else L3 (ADR section 7)."""
        return DESKTOP_LICENCE_CLASS if self.is_desktop else "L3"

    def _doc_licence_class(self, cfg: Mapping[str, Any]) -> str:
        return DESKTOP_LICENCE_CLASS if self.is_desktop else str(cfg.get("licence_class", "L3"))

    def _resolve_boundary(self, boundary: DataBoundary | None) -> DataBoundary:
        if boundary is None:
            return deny_all_text("lseg", note=BOUNDARY_NOTE).model_copy(update={"allow_numeric_features": False})
        if boundary.provider != "lseg":
            raise ValueError(f"boundary.provider must be 'lseg' (got {boundary.provider!r})")
        widened = boundary.allow_numeric_features or bool(boundary.allowed_document_kinds)
        if not widened:
            return boundary
        if self.is_desktop:
            raise ValueError(
                f"the LSEG session '{self.session_name}' is a desktop (Workspace) session, licence class L4: nothing "
                "derived from it may reach an external model, so its boundary cannot be widened (gate G3 covers only "
                "L3 loosening; ADR section 7). Use session_name='platform.ldp' with a G3 citation.")
        problem = _citation_problem(boundary.note or "", self._today())
        if problem:
            raise ValueError(
                f"a widened LSEG boundary must cite gate G3 (written per-dataset AI-use confirmation): {problem}. "
                "Expected note='G3: <confirmation reference>, <YYYY-MM-DD>, reviewed by <lawyer>'; a policy row "
                "without a citation evaluates as deny (ADR section 7)")
        return boundary

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _pause(self) -> None:
        if self.request_pause_s > 0:
            self._sleep(self.request_pause_s)

    def _count_request(self, what: str) -> None:
        today = self._today()
        if self._request_day != today:
            self._request_day, self.request_count = today, 0
        cap = self.daily_request_cap
        if cap is not None and self.request_count >= int(cap):
            raise _StopRequests(
                f"LSEG daily request cap reached ({self.request_count} of {int(cap)} requests today); {what} not sent. "
                "Raise daily_request_cap only within the entitlement's limits (Workspace: 10k requests/day; "
                "Platform limits are UNVERIFIED).")
        self.request_count += 1

    def _call(self, what: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """One counted vendor request.

        Quota and session/authentication errors raise ``ProviderError`` at once (retrying them only burns
        the request quota); throttling is retried with exponential backoff (``backoff_base_s`` x 1, 2, 4)
        up to ``throttle_retries`` times. Any other exception propagates unchanged (a per-request failure).
        """
        attempt = 0
        while True:
            self._count_request(what)
            try:
                return fn(**kwargs)
            except ProviderError:
                raise
            except Exception as e:  # noqa: BLE001 - classified below
                kind = _error_kind(e)
                if kind is None:
                    raise
                if kind == "throttle" and attempt < max(0, int(self.throttle_retries)):
                    delay = float(self.backoff_base_s) * (2 ** attempt)
                    attempt += 1
                    self._warn(f"LSEG {what}: throttled by the vendor; backing off exponentially before retrying.")
                    if delay > 0:
                        self._sleep(delay)
                    continue
                how = f"still throttled after {attempt} retries" if kind == "throttle" else f"{kind} error, not retried"
                raise _StopRequests(f"LSEG {what} stopped ({how}): {e}") from e

    def _ldm(self) -> Any:
        if self._ld is None:
            try:
                self._ld = importlib.import_module("lseg.data")
            except ImportError as e:
                raise ProviderUnavailable(SDK_MISSING) from e
        return self._ld

    def _open(self) -> Any:
        """The ``lseg.data`` module with a session open (opened once, lazily)."""
        ld = self._ldm()
        if self._session is None:
            kwargs: dict[str, Any] = {"name": self.session_name}  # always explicit: never the library's desktop default
            key = os.environ.get(self.app_key_env) if self.app_key_env else None
            if key:
                kwargs["app_key"] = key
            try:
                ld.get_config().set_param("http.request-timeout", 300)
            except Exception:  # noqa: BLE001 - optional tuning; the library default (20 s) still works
                pass
            if self.is_desktop:
                self._warn(DESKTOP_WARNING.format(name=self.session_name))
            try:
                session = ld.open_session(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise ProviderUnavailable(f"Could not open an LSEG session ({self.session_name}): {e}. "
                                          f"{SESSION_HELP}") from e
            self._session = session if session is not None else True
        return ld

    def close(self) -> None:
        """Close the LSEG session if this provider opened one."""
        if self._session is not None and self._ld is not None:
            try:
                self._ld.close_session()
            except Exception:  # noqa: BLE001
                pass
        self._session = None

    def diagnostics(self) -> list[str]:
        """Setup problems that would blank out data (empty list = ready). Makes no network call."""
        problems: list[str] = []
        try:
            self._ldm()
        except ProviderUnavailable as e:
            problems.append(str(e))
        cfg = _find_config()
        if not self.is_desktop:
            if cfg is None:
                problems.append(f"Platform session '{self.session_name}' needs {CONFIG_FILENAME} with the service-account "
                                "credentials in $LD_LIB_CONFIG_PATH, the working directory or ~ (none found).")
        else:
            problems.append(DESKTOP_WARNING.format(name=self.session_name))
            if cfg is None and not os.environ.get(self.app_key_env or ""):
                problems.append(f"Desktop session: set ${self.app_key_env} to a Workspace app key (or provide "
                                f"{CONFIG_FILENAME}); LSEG Workspace must be running on this machine (individual "
                                "licence, never on a server).")
        return problems

    # ------------------------------------------------------------------ symbology
    def _register(self, rics: list[str]) -> dict[str, str]:
        """RIC -> canonical ticker for vendor-returned RICs (remembered for later calls)."""
        out: dict[str, str] = {}
        for ric in rics:
            if ric in self._ticker_of:
                out[ric] = self._ticker_of[ric]
                continue
            t = ric_to_ticker(ric)
            owner = self._ric_of.get(t)
            if owner is not None and owner != ric:
                self._warn(f"RIC {ric} maps to ticker {t}, which is already {owner}; {ric} keeps its RIC as ticker.")
                t = ric
            self._ric_of[t] = ric
            self._ticker_of[ric] = t
            out[ric] = t
        return out

    def ric_for(self, ticker: str) -> str | None:
        """The RIC this provider uses for ``ticker`` (None when unresolved).

        Known tickers (SCREEN results, ``rics=``) resolve through the symbology table; otherwise the
        string itself is used only when it is RIC-shaped (``.SPX``, ``IBM.N``, ``XYZ.N^K20``). A dotted
        share-class ticker such as ``BRK.B`` is not a RIC (that RIC is ``BRKb.N``) and stays unresolved.
        """
        t = str(ticker).strip()
        if t in self._ric_of:
            return self._ric_of[t]
        suffixes = (self.fieldmap.get("symbology") or {}).get("ric_suffixes") or DEFAULT_RIC_SUFFIXES
        return t if _looks_like_ric(t, suffixes) else None

    def _resolve(self, tickers: list[str], what: str) -> dict[str, str]:
        out: dict[str, str] = {}
        missing: list[str] = []
        for t in tickers:
            ric = self.ric_for(t)
            if ric:
                out[t] = ric
            else:
                missing.append(t)
        if missing:
            more = ", ..." if len(missing) > 20 else ""
            self._warn(f"LSEG {what}: no RIC for {len(missing)} ticker(s) ({', '.join(missing[:20])}{more}); dropped and "
                       "flagged (values NaN). Load them through get_universe / pushdown_screen first or pass "
                       "rics={ticker: RIC} (the maintained symbology table, ADR graft 6).")
        return out

    # ------------------------------------------------------------------ requests
    def _ctx(self, as_of: date) -> dict[str, str]:
        return {"as_of": as_of.isoformat(), "as_of_1m": (as_of - timedelta(days=30)).isoformat()}

    def _params(self, as_of: date) -> dict[str, Any]:
        """Request parameters exactly as configured in the field map (before parameter preflight)."""
        ctx = self._ctx(as_of)
        out: dict[str, Any] = {}
        for k, e in (self.fieldmap.get("parameters") or {}).items():
            if str(k).startswith("_"):
                continue
            v = e.get("value") if isinstance(e, Mapping) else e
            if v is None:
                continue
            out[k] = _render(v, ctx) if isinstance(v, str) else v
        return out

    def _request_params(self, as_of: date) -> dict[str, Any]:
        """Parameters for real snapshot requests: configured ones minus unverifiable ones that failed preflight.

        Each ``unverifiable`` parameter (the global ``SDate``) is preflighted once a day per value on
        ``test_ric``: its ``probe`` field is requested with and without it (both values go to
        ``field_log['parameters.<name>']``). It is dropped (warned, logged) when the request with it
        returns nothing while the request without it works.
        """
        params = self._params(as_of)
        for name, entry in (self.fieldmap.get("parameters") or {}).items():
            if name not in params or not isinstance(entry, Mapping) or entry.get("status") != "unverifiable":
                continue
            if not self._parameter_ok(str(name), entry, params, as_of):
                params.pop(name, None)
        return params

    def _anchored(self, as_of: date, params: Mapping[str, Any]) -> bool:
        """True when snapshot values requested with ``params`` are as of ``as_of``.

        That needs a parameter flagged ``anchors_as_of`` in the field map (the global ``SDate``), unless
        ``as_of`` is the latest completed session (then the latest values are as-of values).
        """
        anchors = [k for k, e in (self.fieldmap.get("parameters") or {}).items()
                   if isinstance(e, Mapping) and e.get("anchors_as_of")]
        return any(k in params for k in anchors) or is_point_in_time(as_of, self._today())

    def _probe(self, ld: Any, ric: str, code: str, params: Mapping[str, Any] | None) -> tuple[bool, Any, str]:
        """(passed, first non-empty value, why not) for ``get_data(ric, [code], params)``; systemic errors raise."""
        try:
            df = self._get_data_raw(ld, ric, [code], params)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 - any other failure is a failed probe
            return False, None, f"request failed ({e})"
        if not isinstance(df, pd.DataFrame) or df.shape[1] != 2:
            return False, None, "no column returned"
        vals = [v for v in df.iloc[:, 1] if not _missing(v)]
        if not vals:
            return False, None, "column returned but empty"
        return True, _py(vals[0]), ""

    def _parameter_ok(self, name: str, entry: Mapping[str, Any], params: Mapping[str, Any], as_of: date) -> bool:
        self._reset_preflight_if_new_day()
        value = params[name]
        base = {k: v for k, v in params.items() if k != name}
        probe = str(entry.get("probe") or "")
        key = (self.test_ric, f"param:{name}", str(value), _params_key(base), probe)
        if key in self._param_memo:
            keep, log = self._param_memo[key]
            self.field_log[f"parameters.{name}"] = log
            return keep
        log: dict[str, Any] = {"code": probe, "status": "unverifiable", "value": value, "preflight": None,
                               "with": None, "without": None, "used": False, "reason": ""}
        self.field_log[f"parameters.{name}"] = log
        if not probe:  # validate_fieldmap requires a probe; an override could still delete it
            keep, log["reason"] = False, "no 'probe' field configured: an unverifiable parameter is not sent unpreflighted"
        else:
            ld = self._open()
            ok_with, v_with, why_with = self._probe(ld, self.test_ric, probe, {**base, name: value})
            self._pause()
            ok_without, v_without, why_without = self._probe(ld, self.test_ric, probe, base or None)
            log.update({"preflight": ok_with, "with": v_with, "without": v_without})
            if ok_with:
                keep = True
                if ok_without and v_with == v_without and not is_point_in_time(as_of, self._today()):
                    self._warn(f"LSEG parameter {name}={value}: {probe} on {self.test_ric} returned the same value with and "
                               f"without it ({v_with!r}) for a historical as_of; check that {name} is honoured before "
                               "admitting snapshot values (G5).")
            elif ok_without:
                keep = False
                log["reason"] = f"{probe} on {self.test_ric} returned nothing with {name}={value} ({why_with}) but works without it"
            else:
                keep = True  # inconclusive: the probe fails either way, so the parameter is not what is broken
                log["reason"] = (f"inconclusive: {probe} on {self.test_ric} failed with ({why_with}) and without "
                                 f"({why_without}) the parameter")
                self._warn(f"LSEG parameter preflight for {name} inconclusive ({log['reason']}); {name} kept.")
        log["used"] = keep
        if not keep:
            tail = (" Snapshots for a historical as_of are NOT EVALUATED without this as_of anchor."
                    if entry.get("anchors_as_of") else "")
            self._warn(f"LSEG parameter {name}={value} dropped: {log['reason']} (unverifiable parameter, preflight "
                       f"failed).{tail}")
        self._param_memo[key] = (keep, log)
        return keep

    def _get_data_raw(self, ld: Any, universe: Any, codes: list[str], params: Mapping[str, Any] | None) -> pd.DataFrame | None:
        kwargs: dict[str, Any] = {"universe": universe, "fields": list(codes)}
        if params:
            kwargs["parameters"] = dict(params)
        header = getattr(getattr(ld, "HeaderType", None), "NAME", None)
        if header is not None:
            kwargs["header_type"] = header
        return self._call("get_data", ld.get_data, **kwargs)

    def _get_data(self, ld: Any, universe: Any, codes: list[str], params: Mapping[str, Any] | None) -> pd.DataFrame | None:
        try:
            return self._get_data_raw(ld, universe, codes, params)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"LSEG get_data failed for {len(codes)} field(s) ({', '.join(codes[:5])}): {e}") from e

    def _get_tr(self, universe: str | list[str], fields: Mapping[str, str], params: Mapping[str, Any] | None) -> pd.DataFrame:
        """Snapshot ``fields`` ({key: code}) for ``universe`` -> frame indexed by RIC, one column per key.

        Fields LSEG dropped in one chunk are not requested again for later chunks (NaN). An instrument
        that comes back on several rows keeps its first row whole: values are never combined across
        rows (a warning names the instruments).
        """
        keys, codes = list(fields), list(fields.values())
        empty = pd.DataFrame({k: pd.Series(dtype="object") for k in keys}, index=pd.Index([], name="RIC"))
        if not codes:
            return empty
        ld = self._open()
        if isinstance(universe, str):
            chunks: list[Any] = [universe]
        else:
            step = max(1, self.max_points // len(codes))
            chunks = [list(universe[i:i + step]) for i in range(0, len(universe), step)]
        parts: list[pd.DataFrame] = []
        dropped: set[str] = set()
        for i, chunk in enumerate(chunks):
            live = [(k, c) for k, c in zip(keys, codes) if c not in dropped]
            if not live:
                break
            if i:
                self._pause()
            lkeys, lcodes = [k for k, _ in live], [c for _, c in live]
            df = self._get_data(ld, chunk, lcodes, params)
            if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
                continue
            if df.shape[1] == len(lcodes) + 1:
                part = df.copy()
                part.columns = ["RIC", *lkeys]
            else:
                part, newly = self._field_by_field(ld, chunk, lkeys, lcodes, params, [str(c) for c in df.columns])
                dropped |= newly
            parts.append(part)
        if not parts:
            return empty
        out = pd.concat(parts, ignore_index=True)
        for k in keys:
            if k not in out.columns:
                out[k] = np.nan
        out = out[["RIC", *keys]]
        ric = pd.Series([("" if _missing(v) else str(v).strip()) for v in out["RIC"]], index=out.index)
        out = out.assign(RIC=ric)
        out = out[out["RIC"] != ""]
        dup = out["RIC"].duplicated(keep="first")
        if dup.any():
            self._warn_duplicate_rows(sorted(set(out.loc[dup, "RIC"])))
            out = out[~dup]
        return out.set_index("RIC")

    def _warn_duplicate_rows(self, rics: list[str]) -> None:
        more = ", ..." if len(rics) > 20 else ""
        self._warn(f"LSEG returned several rows for {len(rics)} instrument(s) ({', '.join(rics[:20])}{more}); the first "
                   "row of each is kept whole (values are never combined across rows).")

    def _field_by_field(self, ld: Any, chunk: Any, keys: list[str], codes: list[str], params: Mapping[str, Any] | None,
                        returned: list[str]) -> tuple[pd.DataFrame, set[str]]:
        """Re-request a chunk one field at a time -> (frame with a RIC column, codes LSEG dropped)."""
        self._warn(f"LSEG returned {max(0, len(returned) - 1)} of {len(codes)} requested field column(s) ({returned}); "
                   "re-requested one field at a time (LSEG drops bad or unentitled fields silently).")
        series: dict[str, pd.Series] = {}
        dropped: list[str] = []
        for key, code in zip(keys, codes):
            self._pause()
            d1 = self._get_data(ld, chunk, [code], params)
            if d1 is None or not isinstance(d1, pd.DataFrame) or d1.shape[1] != 2:
                dropped.append(code)
                continue
            s = pd.Series(list(d1.iloc[:, 1]), index=[str(v).strip() for v in d1.iloc[:, 0]], dtype="object")
            if s.index.duplicated().any():
                self._warn_duplicate_rows(sorted(set(s.index[s.index.duplicated()])))
            series[key] = s[~s.index.duplicated(keep="first")]
        if dropped:
            self._warn(f"LEG NOT EVALUATED: LSEG returned no column for {', '.join(dropped)} (unentitled or invalid "
                       "field); those values are NaN and the field is not requested again in this call.")
        index = pd.Index(sorted(set().union(*[set(s.index) for s in series.values()]))) if series else pd.Index([])
        frame = pd.DataFrame({k: (series[k].reindex(index) if k in series else pd.Series(np.nan, index=index, dtype="object"))
                              for k in keys}, index=index)
        frame.insert(0, "RIC", list(index))
        return frame.reset_index(drop=True), set(dropped)

    # ------------------------------------------------------------------ preflight / gating
    def _reset_preflight_if_new_day(self) -> None:
        today = self._today()
        if self._preflight_day != today:
            self._preflight_memo = {}
            self._param_memo = {}
            self._preflight_day = today

    def _preflight(self, codes: list[str], ric: str | None = None,
                   params: Mapping[str, Any] | None = None) -> dict[str, bool]:
        """A code passes when ``get_data(ric, [code], params)`` returns exactly one non-empty column.

        ``params`` must be the parameters of the real request the code is preflighted for (a field can
        work bare and fail under ``{'Curn': 'USD', 'SDate': as_of}``); the memo is per day, RIC, code and
        parameters. Quota / session errors propagate instead of failing (and memoising) every field.
        """
        self._reset_preflight_if_new_day()
        ric = ric or self.test_ric
        pkey = _params_key(params)
        todo = [c for c in dict.fromkeys(codes) if (ric, c, pkey) not in self._preflight_memo]
        if todo:
            ld = self._open()
            for i, c in enumerate(todo):
                if i:
                    self._pause()
                ok, _value, _why = self._probe(ld, ric, c, params or None)
                self._preflight_memo[(ric, c, pkey)] = bool(ok)
        return {c: self._preflight_memo[(ric, c, pkey)] for c in codes}

    def _history_kwargs(self, universe: list[str], codes: list[str], start: date, end: date,
                        adjustments: list[Any] | None) -> dict[str, Any]:
        hist = self.fieldmap.get("history") or {}
        kwargs: dict[str, Any] = {"universe": list(universe), "fields": list(codes), "interval": hist.get("interval", "daily"),
                                  "start": start.isoformat(), "end": end.isoformat()}
        if adjustments:
            kwargs["adjustments"] = list(adjustments)
        return kwargs

    def _configured_adjustments(self) -> tuple[list[Any] | None, Mapping[str, Any]]:
        adj = (self.fieldmap.get("history") or {}).get("adjustments")
        entry = adj if isinstance(adj, Mapping) else {}
        value = adj.get("value") if isinstance(adj, Mapping) else adj
        return (list(value) if value else None), entry

    def _history_probe(self, ld: Any, code: str, adjustments: list[Any] | None) -> tuple[bool, str]:
        today = self._today()
        try:
            raw = self._call("get_history", ld.get_history,
                             **self._history_kwargs([self.test_ric], [code], today - timedelta(days=14), today, adjustments))
            got = _split_history(raw, [self.test_ric], [code]).get(self.test_ric)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 - a failed probe
            return False, f"request failed ({e})"
        if got is None or not got[code].notna().any():
            return False, "no daily values"
        return True, ""

    def _history_adjustments(self) -> list[Any] | None:
        """The ``adjustments`` argument real history requests use.

        Unverifiable adjustment values are preflighted once a day on ``test_ric``: the close history is
        requested with and without them; when only the request without them works they are dropped
        (library default adjustments), warned and logged in ``field_log['history.adjustments']``.
        """
        value, entry = self._configured_adjustments()
        if not value or entry.get("status") != "unverifiable":
            return value
        self._reset_preflight_if_new_day()
        close = str(((((self.fieldmap.get("history") or {}).get("fields")) or {}).get(F.CLOSE) or {}).get("code") or "")
        key = (self.test_ric, "history-adjustments", json.dumps(value, default=str), close)
        if key not in self._param_memo:
            log: dict[str, Any] = {"code": close, "status": "unverifiable", "value": value, "preflight": None,
                                   "used": True, "reason": ""}
            self.field_log["history.adjustments"] = log
            keep = True
            if close:
                ld = self._open()
                ok_with, why_with = self._history_probe(ld, close, value)
                self._pause()
                ok_without, why_without = self._history_probe(ld, close, None)
                log.update({"preflight": ok_with, "without": ok_without})
                if not ok_with and ok_without:
                    keep = False
                    log["reason"] = (f"{close} history on {self.test_ric} failed with adjustments={value} ({why_with}) "
                                     "but works without them")
                    self._warn(f"LSEG history: adjustments {value} dropped ({log['reason']}; unverifiable parameter, "
                               "preflight failed); the library default adjustments apply.")
                elif not ok_with:
                    log["reason"] = f"inconclusive: {close} history failed with ({why_with}) and without ({why_without})"
            log["used"] = keep
            self._param_memo[key] = (keep, log)
        keep, log = self._param_memo[key]
        self.field_log["history.adjustments"] = log
        return value if keep else None

    def _preflight_history(self, codes: list[str], adjustments: list[Any] | None) -> dict[str, bool]:
        """History fields pass when ``get_history(test_ric, [code], adjustments=...)`` returns daily values."""
        self._reset_preflight_if_new_day()
        ric = self.test_ric
        akey = json.dumps(adjustments, default=str)
        todo = [c for c in dict.fromkeys(codes) if (ric, f"history:{c}", akey) not in self._preflight_memo]
        if todo:
            ld = self._open()
            for i, c in enumerate(todo):
                if i:
                    self._pause()
                ok, _why = self._history_probe(ld, c, adjustments)
                self._preflight_memo[(ric, f"history:{c}", akey)] = bool(ok)
        return {c: self._preflight_memo[(ric, f"history:{c}", akey)] for c in codes}

    @staticmethod
    def _self_anchored(entry: Mapping[str, Any]) -> bool:
        """The field code itself carries the as_of date (e.g. a ``{as_of}`` placeholder in its SDate)."""
        return "{as_of" in str(entry.get("code", ""))

    def _usable(self, section: str, as_of: date, params: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
        """Field-map entries of a raw section that may be requested now: {key: (rendered code, entry)}.

        Unverifiable entries are preflighted with ``params`` (the real request's parameters). For a
        historical ``as_of`` without an as_of anchor in ``params`` only self-anchored entries (and, for
        the universe, labels) are served: anything else would be today's value (look-ahead).
        """
        ctx = self._ctx(as_of)
        anchored = self._anchored(as_of, params)
        usable: dict[str, tuple[str, Mapping[str, Any]]] = {}
        skipped: dict[str, str] = {}
        need: dict[str, str] = {}
        for key, e in ((self.fieldmap.get("raw") or {}).get(section) or {}).items():
            if str(key).startswith("_") or not isinstance(e, Mapping) or not e.get("code"):
                continue
            code = _render(e["code"], ctx)
            log = {"code": code, "status": e.get("status"), "preflight": None, "used": False, "reason": ""}
            self.field_log[f"{section}.{key}"] = log
            if e.get("units_verified") is False:
                skipped[key] = f"{code}: units UNVERIFIED in the field map (not served until admitted)"
                log["reason"] = skipped[key]
                continue
            if not anchored and not self._self_anchored(e) and not (section == "universe" and e.get("kind") == "label"):
                skipped[key] = f"{code}: no as_of anchor for the historical as_of {as_of} (it would be today's value)"
                log["reason"] = skipped[key]
                continue
            if e.get("status") == "unverifiable":
                need[key] = code
            usable[key] = (code, e)
        if need:
            pf = self._preflight(list(need.values()), params=params)
            for key, code in need.items():
                self.field_log[f"{section}.{key}"]["preflight"] = pf[code]
                if not pf[code]:
                    skipped[key] = f"{code}: unverifiable and its preflight on {self.test_ric} returned no data"
                    self.field_log[f"{section}.{key}"]["reason"] = skipped[key]
                    usable.pop(key, None)
        for key in usable:
            self.field_log[f"{section}.{key}"]["used"] = True
        if skipped:
            self._warn(f"LSEG {section}: LEG NOT EVALUATED for " + "; ".join(f"{k} ({why})" for k, why in skipped.items())
                       + " -> NaN.")
        return usable

    # ------------------------------------------------------------------ snapshots
    def _snapshot(self, section: str, tickers: list[str], as_of: date, columns: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
        tickers = _unique(tickers)
        if not tickers:
            return _assemble(columns, [], {}), {}
        ric_of = self._resolve(tickers, section)
        params = self._request_params(as_of) if ric_of else {}  # no preflight requests without a resolved RIC
        usable = self._usable(section, as_of, params) if ric_of else {}
        data: dict[str, pd.Series] = {}
        if usable:
            frame = self._get_tr(sorted(set(ric_of.values())), {k: c for k, (c, _) in usable.items()}, params)
            for key, (_code, entry) in usable.items():
                conv = _convert(frame[key] if key in frame.columns else pd.Series(np.nan, index=frame.index, dtype="object"), entry)
                data[key] = pd.Series([conv.get(ric_of[t], np.nan) if t in ric_of else np.nan for t in tickers],
                                      index=tickers, dtype=conv.dtype if len(conv) else "object")
        return _assemble(columns, tickers, data), ric_of

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """Fundamentals snapshot as of ``as_of`` (SDate). A report date after as_of blanks the row.

        The guard needs the report date of the same (FQ0) period as the values; a row with values but no
        report date (leg NOT EVALUATED or no data) cannot be checked, and a warning says so.
        """
        df, _ = self._snapshot("fundamentals", tickers, as_of, F.FUNDAMENTAL_COLUMNS)
        values = df.drop(columns=[F.REPORT_DATE, F.PERIOD_END]).notna().any(axis=1)
        unchecked = [str(t) for t in df.index[values & df[F.REPORT_DATE].isna()]]
        if unchecked:
            more = ", ..." if len(unchecked) > 20 else ""
            self._warn(f"LSEG fundamentals: no report date for {len(unchecked)} ticker(s) ({', '.join(unchecked[:20])}"
                       f"{more}): the look-ahead guard (report date after as_of blanks the row) could not run for them; "
                       "their values rely on the SDate anchor alone.")
        late = df[F.REPORT_DATE].notna() & (df[F.REPORT_DATE] > pd.Timestamp(as_of))
        if late.any():
            names = [str(t) for t in df.index[late]]
            self._warn(f"LSEG fundamentals: report date after as_of {as_of} for {', '.join(names[:20])}; rows blanked "
                       "(look-ahead guard).")
            df.loc[late, :] = np.nan
            df[F.PERIOD_END] = df[F.PERIOD_END].astype("datetime64[ns]")
            df[F.REPORT_DATE] = df[F.REPORT_DATE].astype("datetime64[ns]")
        return df

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        df, _ = self._snapshot("estimates", tickers, as_of, F.ESTIMATE_COLUMNS)
        late = df[F.LAST_EARNINGS_DATE].notna() & (df[F.LAST_EARNINGS_DATE] > pd.Timestamp(as_of))
        if late.any():
            df.loc[late, F.LAST_EARNINGS_DATE] = pd.NaT
            self._warn(f"LSEG estimates: last earnings date after as_of {as_of} ignored (look-ahead guard).")
        return df

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        df, _ = self._snapshot("short_interest", tickers, as_of, F.SHORT_INTEREST_COLUMNS)
        late = df[F.SI_SETTLEMENT_DATE].notna() & (df[F.SI_SETTLEMENT_DATE] > pd.Timestamp(as_of))
        if late.any():
            df.loc[late, :] = np.nan
            df[F.SI_SETTLEMENT_DATE] = df[F.SI_SETTLEMENT_DATE].astype("datetime64[ns]")
            self._warn(f"LSEG short interest: settlement after as_of {as_of}; rows blanked (look-ahead guard).")
        return df

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """30d ATM implied vol from ``<root>ATMIV.U`` (call/put mean, vol points -> fraction); fetch survivors only."""
        tickers = _unique(tickers)
        data: dict[str, pd.Series] = {}
        entry = ((self.fieldmap.get("raw") or {}).get("options") or {}).get(F.IV_30D_ATM)
        if tickers and isinstance(entry, Mapping) and entry.get("codes") and entry.get("ric_template"):
            ric_of = self._resolve(tickers, "options")
            ctx = self._ctx(as_of)
            codes = [_render(c, ctx) for c in entry["codes"]]
            ok = bool(ric_of)  # no preflight requests without a resolved RIC
            params = self._request_params(as_of) if ok else {}
            if ok and entry.get("units_verified") is False:
                ok = False
                self._warn(f"LSEG options: LEG NOT EVALUATED for {F.IV_30D_ATM} (units UNVERIFIED in the field map).")
            elif ok and not self._anchored(as_of, params) and not any("{as_of" in str(c) for c in entry["codes"]):
                ok = False
                self._warn(f"LSEG options: LEG NOT EVALUATED for {F.IV_30D_ATM} (no as_of anchor for the historical "
                           f"as_of {as_of}: it would be today's value).")
            elif ok and entry.get("status") == "unverifiable":
                probe = str(entry["ric_template"]).format(root=_ric_root(self.test_ric), ticker=ric_to_ticker(self.test_ric),
                                                          ric=self.test_ric)
                pf = self._preflight(codes, ric=probe, params=params)
                ok = all(pf.values())
                if not ok:
                    self._warn(f"LSEG options: LEG NOT EVALUATED for {F.IV_30D_ATM} (preflight on {probe} failed).")
            if ok:
                iv_ric = {t: str(entry["ric_template"]).format(root=_ric_root(r), ticker=ric_to_ticker(r), ric=r)
                          for t, r in ric_of.items()}
                frame = self._get_tr(sorted(set(iv_ric.values())), {f"c{i}": c for i, c in enumerate(codes)}, params)
                num = pd.DataFrame({c: [_to_float(v) for v in frame[c]] for c in frame.columns}, index=frame.index)
                comb = num.mean(axis=1, skipna=True) if entry.get("combine", "mean") == "mean" else num.iloc[:, 0]
                comb = comb * float(entry.get("to_canonical", 1.0))
                vals = [comb.get(iv_ric[t], np.nan) if t in iv_ric else np.nan for t in tickers]
                data[F.IV_30D_ATM] = pd.Series(vals, index=tickers, dtype="float64")
                miss = [t for t, v in zip(tickers, vals) if t in iv_ric and not math.isfinite(v)]
                if miss:
                    self._warn(f"LSEG options: no 30d ATM implied vol for {', '.join(miss[:20])} (the <root>ATMIV.U pattern "
                               "is verified for WMT only; unresolved RICs are NaN).")
                self.field_log[f"options.{F.IV_30D_ATM}"] = {"code": ", ".join(codes), "status": entry.get("status"),
                                                             "preflight": None, "used": True, "reason": ""}
        if tickers:
            self._warn("LSEG options: put/call volume and open interest (chain-based, unverified) and the 1y implied-vol "
                       "range are not provided (NaN).")
        return _assemble(F.OPTIONS_COLUMNS, tickers, data)

    # ------------------------------------------------------------------ universe + push-down
    def get_universe(self, spec: Any, as_of: date) -> pd.DataFrame:
        """US primary listings from one SCREEN (listing predicates only), indexed by canonical ticker."""
        from aitrading.screen.spec import UniverseSpec  # noqa: PLC0415

        u = spec if spec is not None else UniverseSpec()
        cq = compile_universe(u, self.fieldmap, as_of)
        self.last_universe_query = cq.expression
        if not is_point_in_time(as_of, self._today()):
            self._warn(f"as_of {as_of} is historical: the LSEG SCREEN universe lists instruments active today "
                       "(survivorship bias); snapshot values are anchored with SDate.")
        params = self._request_params(as_of)
        usable = self._usable("universe", as_of, params)
        if not usable:
            raise ProviderError("the LSEG field map has no usable universe fields")
        frame = self._get_tr(cq.expression, {k: c for k, (c, _) in usable.items()}, params)
        rics = [str(r) for r in frame.index]
        if not rics:
            self._warn(f"LSEG universe SCREEN returned 0 instruments ({cq.expression}); check entitlements and field codes.")
        tick = self._register(rics)
        tickers = [tick[r] for r in rics]
        data: dict[str, pd.Series] = {}
        for key, (_code, entry) in usable.items():
            conv = _convert(frame[key], entry)
            data[key] = pd.Series(list(conv), index=tickers, dtype=conv.dtype)
        data[F.VENDOR_ID] = pd.Series(rics, index=tickers, dtype="object")
        if F.CURRENCY not in usable and F.COUNTRY in data:
            data[F.CURRENCY] = pd.Series(["USD" if c == "US" else np.nan for c in data[F.COUNTRY]], index=tickers, dtype="object")
        if F.NAME in data:
            for t, n in data[F.NAME].items():
                if isinstance(n, str):
                    self._names[t] = n
        df = _assemble(F.UNIVERSE_COLUMNS, tickers, data)
        return self._apply_universe_spec(df, u)

    @staticmethod
    def _apply_universe_spec(df: pd.DataFrame, spec: Any) -> pd.DataFrame:
        if spec is None or df.empty:
            return df
        keep = pd.Series(True, index=df.index)
        country = getattr(spec, "country", None)
        if country:
            c = df[F.COUNTRY]
            keep &= c.isna() | (c.astype(str).str.upper() == str(country).upper())
        types = getattr(spec, "security_types", None)
        if types:
            t = df[F.SECURITY_TYPE]
            keep &= t.isna() | t.isin(list(types))
        excl = {str(x).strip().lower() for x in (getattr(spec, "exclude_sectors", None) or [])}
        if excl:
            keep &= ~df[F.GICS_SECTOR].astype(str).str.strip().str.lower().isin(excl)
        return df.loc[keep]

    def pushdown_screen(self, spec: Any, as_of: date) -> PushdownResult:
        """Run the compiled SCREEN: ``ld.get_data(universe=<SCREEN expr>, fields=['TR.CommonName'])``.

        Only admitted (gate G5: ``admitted=`` plus ``"admitted": true`` in the field map), confirmed/corrected
        predicates are pushed, and time-varying ones only when ``as_of`` is the latest completed session
        (see ``aitrading.screen.compile_lseg``); the rest are returned as residual conditions for the
        local engine. The query is recorded verbatim; ``last_pushdown.to_audit()`` records the split, the
        admission set and the as_of lag. Preflights use the SCREEN request's own form (no parameters).
        """
        today = self._today()
        codes = preflight_codes(spec, self.fieldmap, as_of, today=today, admitted=self.admitted)
        pf = self._preflight(codes) if codes else {}
        cq = compile_screen(spec, self.fieldmap, as_of, preflight=pf, today=today, admitted=self.admitted)
        self.last_pushdown = cq
        if not cq.point_in_time:
            self._warn(f"as_of {as_of} is historical (older than the latest completed session): SCREEN evaluates the "
                       "latest values with no date anchor, so only static listing predicates were pushed (every "
                       "time-varying condition is residual); the listing reflects today's active instruments "
                       "(survivorship bias).")
        not_admitted = [r.condition for r in cq.residual if r.reason.startswith("not admitted (G5)")]
        if not_admitted:
            self._warn(f"LSEG push-down: {len(not_admitted)} condition(s) not admitted under gate G5 stay local "
                       f"({', '.join(not_admitted[:10])}); admit them with verify_fields() evidence in the "
                       "field_validation_log, then pass admitted=[...].")
        name_entry = ((self.fieldmap.get("raw") or {}).get("universe") or {}).get(F.NAME) or {}
        name_code = str(name_entry.get("code") or "")
        if not name_code:
            raise ProviderError("the LSEG field map has no raw.universe.name code for the SCREEN result")
        ld = self._open()
        df = self._get_data(ld, cq.expression, [name_code], None)
        rics: list[str] = []
        names: list[Any] = []
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
            for i in range(len(df)):
                r = df.iloc[i, 0]
                if not _missing(r):
                    rics.append(str(r).strip())
                    names.append(df.iloc[i, 1] if df.shape[1] >= 2 else None)
        tick = self._register(list(dict.fromkeys(rics)))
        for r, n in zip(rics, names):
            if isinstance(n, str) and n.strip():
                self._names[tick[r]] = n.strip()
        if not rics:
            self._warn(f"LSEG SCREEN returned 0 instruments: {cq.expression} (check entitlements; a NaN field inside SCREEN "
                       "silently drops every security).")
        return PushdownResult(
            tickers=sorted(dict.fromkeys(tick[r] for r in rics)),
            query=cq.expression,
            pushed_conditions=cq.pushed_conditions,
            residual_conditions=cq.residual_conditions,
        )

    # ------------------------------------------------------------------ prices
    def _history_fields(self, adjustments: list[Any] | None) -> dict[str, str]:
        """Usable history fields {canonical price field: code}; unverifiable ones are preflighted with the
        same ``adjustments`` the real request sends."""
        fields = ((self.fieldmap.get("history") or {}).get("fields") or {})
        usable: dict[str, str] = {}
        need: dict[str, str] = {}
        for key, e in fields.items():
            if key not in F.PRICE_FIELDS or not isinstance(e, Mapping) or not e.get("code"):
                continue
            usable[key] = str(e["code"])
            if e.get("status") == "unverifiable":
                need[key] = str(e["code"])
        if need:
            pf = self._preflight_history(list(need.values()), adjustments)
            failed = [f"{k} ({c})" for k, c in need.items() if not pf[c]]
            for k, c in need.items():
                self.field_log[f"history.{k}"] = {"code": c, "status": "unverifiable", "preflight": pf[c], "used": pf[c],
                                                  "reason": "" if pf[c] else "preflight failed"}
                if not pf[c]:
                    usable.pop(k, None)
            if failed:
                self._warn(f"LSEG history: LEG NOT EVALUATED for {', '.join(failed)} (unverifiable field, preflight on "
                           f"{self.test_ric} returned no data) -> NaN.")
        if F.CLOSE not in usable:
            raise ProviderError("the LSEG field map has no usable history close field")
        return usable

    def _history(self, rics: list[str], fields: Mapping[str, str], start: date, end: date,
                 adjustments: list[Any] | None = None) -> dict[str, pd.DataFrame]:
        """{canonical field: frame(date x RIC)} over [start, end].

        Requests go in batches of ``history_batch_size``. A failed batch is retried one RIC at a time
        (so one bad RIC does not sink the batch) until a circuit breaker sees ``circuit_breaker_threshold``
        single-RIC failures in a row with the same error: then no further per-RIC retries are made in
        this call (each later batch is still tried once). Quota / session errors stop at once.
        """
        ld = self._open()
        if adjustments is None:
            adjustments = self._history_adjustments()
        codes = list(fields.values())
        per: dict[str, pd.DataFrame] = {}
        failed: dict[str, str] = {}
        tripped: str | None = None
        skipped: list[str] = []
        batches = [rics[i:i + self.history_batch_size] for i in range(0, len(rics), max(1, self.history_batch_size))]

        def fetch(universe: list[str]) -> dict[str, pd.DataFrame]:
            raw = self._call("get_history", ld.get_history, **self._history_kwargs(universe, codes, start, end, adjustments))
            return _split_history(raw, universe, codes)

        for i, batch in enumerate(batches):
            if i:
                self._pause()
            try:
                per.update(fetch(batch))
                continue
            except _StopRequests:
                raise
            except Exception as e:  # noqa: BLE001 - retried one RIC at a time below
                batch_error = str(e)
            if len(batch) == 1:
                failed[batch[0]] = batch_error
                continue
            if tripped is not None:
                skipped += batch
                continue
            last_sig, streak = None, 0
            for j, r in enumerate(batch):
                self._pause()
                try:
                    per.update(fetch([r]))
                    last_sig, streak = None, 0
                except _StopRequests:
                    raise
                except Exception as e1:  # noqa: BLE001
                    failed[r] = str(e1)
                    sig = _error_signature(str(e1), r)
                    streak = streak + 1 if sig == last_sig else 1
                    last_sig = sig
                    if streak >= max(1, int(self.circuit_breaker_threshold)):
                        tripped = str(e1)
                        skipped += batch[j + 1:]
                        break
        if failed:
            self._warn(f"LSEG get_history failed for {', '.join(list(failed)[:20])} ({next(iter(failed.values()))}).")
        if tripped is not None:
            self._warn(f"LSEG get_history: circuit breaker tripped after {int(self.circuit_breaker_threshold)} identical "
                       f"single-RIC failures ({tripped}); {len(skipped)} RIC(s) were not retried one by one "
                       "(their prices are NaN).")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        out: dict[str, pd.DataFrame] = {}
        for key, code in fields.items():
            cols = {r: per[r][code] for r in rics if r in per and code in per[r].columns}
            frame = pd.DataFrame(cols) if cols else pd.DataFrame(index=pd.DatetimeIndex([]))
            frame = frame[(frame.index >= lo) & (frame.index <= hi)]
            out[key] = frame
        return out

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        req = _unique(tickers)
        if not req:
            empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), dtype="float64")
            return PricePanel(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
        ric_of = self._resolve(req, "prices")
        if not ric_of:
            raise ProviderError(f"No RIC for any of {len(req)} ticker(s); cannot request LSEG history.")
        adjustments = self._history_adjustments()
        fields = self._history_fields(adjustments)
        frames = self._history(sorted(set(ric_of.values())), fields, start, end, adjustments)
        index = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])), name="date")
        out: dict[str, pd.DataFrame] = {}
        for f in F.PRICE_FIELDS:
            fr = frames.get(f)
            cols = {}
            for t in req:
                r = ric_of.get(t)
                if fr is not None and r is not None and r in fr.columns:
                    cols[t] = fr[r].reindex(index).astype("float64")
                else:
                    cols[t] = pd.Series(np.nan, index=index, dtype="float64")
            out[f] = pd.DataFrame(cols, index=index)[req]
        close = out[F.CLOSE]
        failed = [t for t in req if close[t].isna().all()]
        if len(failed) == len(req):
            raise ProviderError(f"No LSEG price history for any of {len(req)} ticker(s) ({', '.join(req[:10])}) between "
                                f"{start} and {end}.")
        if failed:
            self._warn(f"No LSEG price history for {len(failed)} ticker(s) ({', '.join(failed[:20])}); their prices are NaN.")
        configured, entry = self._configured_adjustments()
        if entry.get("status") == "unverifiable":
            used = f"adjustments={adjustments}" if adjustments else f"library default adjustments ({configured} dropped)"
            self._warn(f"LSEG history: corporate-action adjustment of the price series is UNVERIFIED ({used}); "
                       "cross-check with crosscheck_adjustment() around splits.")
        return PricePanel(out[F.OPEN], out[F.HIGH], out[F.LOW], out[F.CLOSE], out[F.VOLUME])

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        sym = symbol or self.benchmark
        close = (((self.fieldmap.get("history") or {}).get("fields")) or {}).get(F.CLOSE) or {}
        close_code = str(close.get("code") or "")
        if not close_code:
            raise ProviderError("the LSEG field map has no history close field")
        candidates = [sym]
        bm = self.fieldmap.get("benchmark") or {}
        if symbol is None and bm.get("fallback_ric") and bm["fallback_ric"] != sym:
            candidates.append(str(bm["fallback_ric"]))
        for i, ric in enumerate(candidates):
            try:
                s = self._history([ric], {F.CLOSE: close_code}, start, end)[F.CLOSE]
            except _StopRequests:
                raise
            except ProviderError as e:
                self._warn(f"LSEG benchmark {ric}: {e}")
                continue
            if ric in s.columns and s[ric].notna().any():
                if i:
                    self._warn(f"LSEG benchmark {sym} history unavailable (unentitled?); using fallback {ric} "
                               f"({bm.get('fallback_status', 'unverifiable')} in the field map).")
                out = s[ric].astype("float64").rename(ric)
                out.index.name = "date"
                return out
        raise ProviderError(f"No LSEG benchmark history for {', '.join(candidates)} between {start} and {end}.")

    def crosscheck_adjustment(self, tickers: list[str], as_of: date, tolerance: float = 0.02) -> pd.DataFrame:
        """Compare the local 52-week high (history) with the vendor's ``TR.Price52WeekHigh`` (VENDOR_REFERENCE 2.2).

        A ratio far from 1 around a split means the history is not corporate-action adjusted.
        """
        entry = (self.fieldmap.get("features") or {}).get("high_52w") or {}
        code = entry.get("expression")
        if not code:
            raise ProviderError("the LSEG field map has no features.high_52w expression for the cross-check")
        tickers = _unique(tickers)
        ric_of = self._resolve(tickers, "adjustment cross-check")
        params = self._request_params(as_of) if ric_of else {}
        if ric_of and self._anchored(as_of, params):
            vend = self._get_tr(sorted(set(ric_of.values())), {"v": str(code)}, params)
        else:
            vend = pd.DataFrame({"v": pd.Series(dtype="object")})
            if ric_of:
                self._warn(f"LSEG adjustment cross-check: no as_of anchor for the historical as_of {as_of}, so the vendor "
                           f"52-week high ({code}) would be today's value; NOT EVALUATED (NaN).")
        panel = self.get_price_history(list(ric_of), as_of - timedelta(days=365), as_of)
        local_hi = panel.high.max() if panel.high.notna().any().any() else panel.close.max()
        rows = {}
        for t, r in ric_of.items():
            v = _to_float(vend["v"].get(r)) if "v" in vend.columns else math.nan
            lh = _to_float(local_hi.get(t))
            ratio = lh / v if math.isfinite(v) and v > 0 and math.isfinite(lh) else math.nan
            rows[t] = {"vendor_high_52w": v, "local_high_52w": lh, "ratio": ratio,
                       "flagged": bool(math.isfinite(ratio) and abs(ratio - 1.0) > tolerance)}
        df = pd.DataFrame.from_dict(rows, orient="index", columns=["vendor_high_52w", "local_high_52w", "ratio", "flagged"])
        df.index.name = "ticker"
        return df

    # ------------------------------------------------------------------ documents
    def get_documents(self, ticker: str, kinds: set[DocumentKind], start: date, end: date, limit: int = 10) -> list[Document]:
        """NEWS via ld.news; FILING only when enabled in the field map; no TRANSCRIPT / RESEARCH. Newest first.

        Each document's metadata carries the session and its licence class: L3 through a platform
        session, L4 through a desktop (Workspace) session (ADR section 7).
        """
        kinds = set(kinds)
        docs_cfg = self.fieldmap.get("documents") or {}
        if DocumentKind.TRANSCRIPT in kinds:
            self._warn("LSEG transcripts (StreetEvents) are XML over SFTP under a separate licence (L3): out of scope for "
                       "this adapter, so no transcripts are returned.")
        if DocumentKind.RESEARCH in kinds:
            self._warn("LSEG offers no broker research through the Data Library; none returned.")
        wanted = kinds & {DocumentKind.NEWS, DocumentKind.FILING}
        if not wanted:
            return []
        ric = self._resolve([str(ticker).strip()], "documents").get(str(ticker).strip())
        if ric is None:
            return []
        out: list[Document] = []
        if DocumentKind.NEWS in wanted:
            if (docs_cfg.get("news") or {}).get("enabled"):
                out += self._news(str(ticker).strip(), ric, start, end, limit)
            else:
                self._warn("LSEG news is disabled in the field map.")
        if DocumentKind.FILING in wanted:
            if (docs_cfg.get("filings") or {}).get("enabled"):
                out += self._filings(str(ticker).strip(), ric, start, end, limit)
            else:
                self._warn("LSEG filings search is disabled in the field map (its search parameters are not in the "
                           "vendor reference); use SEC EDGAR (public, L0) for filings.")
        out.sort(key=lambda d: d.published_at, reverse=True)
        return out[: max(0, int(limit))]

    def _news(self, ticker: str, ric: str, start: date, end: date, limit: int) -> list[Document]:
        cfg = (self.fieldmap.get("documents") or {}).get("news") or {}
        ld = self._open()
        news = getattr(ld, "news", None)
        if news is None:
            self._warn("This lseg.data build has no news module; no news returned.")
            return []
        query = _render(cfg.get("query_template", "R:{ric}"), {"ric": ric, "ticker": ticker})
        t0, t1 = datetime.combine(start, dtime.min), datetime.combine(end, dtime.max)
        try:
            hl = self._call("news.get_headlines", news.get_headlines, query=query, start=t0, end=t1, count=int(limit))
        except _StopRequests:
            raise
        except Exception as e:  # noqa: BLE001
            self._warn(f"{ticker}: LSEG headlines unavailable ({e}).")
            return []
        if not isinstance(hl, pd.DataFrame) or hl.empty:
            return []
        df = hl.reset_index() if not isinstance(hl.index, pd.RangeIndex) else hl
        c_time = _col(df, "versionCreated", "date", "index")
        c_head = _col(df, "headline", "text", "title")
        c_id = _col(df, "storyId", "story_id")
        c_src = _col(df, "sourceCode", "source")
        if c_id is None or c_head is None:
            self._warn(f"{ticker}: unexpected LSEG headline columns {list(df.columns)}; no news returned.")
            return []
        docs: list[Document] = []
        missing_text = 0
        for _, row in df.iterrows():
            sid = row[c_id]
            if _missing(sid):
                continue
            ts = pd.Timestamp(row[c_time]) if c_time is not None and not _missing(row[c_time]) else pd.NaT
            if ts is pd.NaT:
                continue
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            if not (pd.Timestamp(t0) <= ts <= pd.Timestamp(t1)):
                continue
            title = str(row[c_head]).strip()
            meta = {"vendor": "lseg", "channel": "ld.news", "session": self.session_name,
                    "licence_class": self._doc_licence_class(cfg), "story_id": str(sid), "ric": ric, "query": query}
            self._pause()  # one request per story body: paced like every other request
            try:
                text = _html_to_text(self._call("news.get_story", news.get_story, story_id=str(sid)))
            except _StopRequests:
                raise
            except Exception:  # noqa: BLE001
                text = ""
            if not text:
                missing_text += 1
                text = title
                meta["text_unavailable"] = "true"
            src = str(row[c_src]).strip() if c_src is not None and not _missing(row[c_src]) else ""
            docs.append(Document(doc_id=f"lseg:news:{sid}", ticker=ticker, kind=DocumentKind.NEWS, title=title,
                                 published_at=ts.to_pydatetime(), source=f"LSEG News ({src})" if src else "LSEG News",
                                 text=text, metadata=meta))
        if missing_text:
            self._warn(f"{ticker}: {missing_text} LSEG story body(ies) unavailable; the headline stands in as text.")
        return docs

    def _filings(self, ticker: str, ric: str, start: date, end: date, limit: int) -> list[Document]:
        cfg = (self.fieldmap.get("documents") or {}).get("filings") or {}
        ld = self._open()
        try:
            mod = getattr(getattr(ld, "content", None), "filings", None) or importlib.import_module("lseg.data.content.filings")
        except ImportError as e:
            raise ProviderUnavailable(f"lseg.data.content.filings is unavailable: {e}. {SDK_MISSING}") from e
        feed_name = str(cfg.get("feed", "EDGAR"))
        feed = getattr(getattr(mod, "Feed", None), feed_name, feed_name)
        ctx = {"ric": ric, "ticker": ticker, "start": start.isoformat(), "end": end.isoformat(), "limit": int(limit)}
        params = {k: (_render(v, ctx) if isinstance(v, str) else v) for k, v in (cfg.get("search_parameters") or {}).items()}
        try:
            resp = self._call("filings.search", lambda: mod.search.Definition(feed=feed, **params).get_data())
            df = getattr(getattr(resp, "data", None), "df", None)
        except _StopRequests:
            raise
        except Exception as e:  # noqa: BLE001
            self._warn(f"{ticker}: LSEG filings search failed ({e}).")
            return []
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        c_title = _col(df, "DocumentTitle", "title", "documentTitle", "FormType")
        c_date = _col(df, "FilingDate", "filingDate", "FilingDateTime", "date", "FinancialFilingDate")
        c_id = _col(df, "Filename", "filename", "DocId", "docId", "Dcn", "dcn")
        docs: list[Document] = []
        for _, row in df.iterrows():
            ts = _to_ts(row[c_date]) if c_date is not None else pd.NaT
            if ts is pd.NaT or not (pd.Timestamp(start) <= ts <= pd.Timestamp(end)):
                continue
            fid = str(row[c_id]).strip() if c_id is not None and not _missing(row[c_id]) else f"{ric}:{ts.date()}"
            title = str(row[c_title]).strip() if c_title is not None and not _missing(row[c_title]) else "Filing"
            text = (f"{title}\nFiled {ts.date().isoformat()} (feed {feed_name}). Full text: "
                    f"filings.retrieval.Definition(filename={fid!r}) (not downloaded by this adapter).")
            docs.append(Document(doc_id=f"lseg:filing:{fid}", ticker=ticker, kind=DocumentKind.FILING, title=title,
                                 published_at=ts.to_pydatetime(), source=f"LSEG Filings ({feed_name})", text=text,
                                 metadata={"vendor": "lseg", "channel": "lseg.data.content.filings",
                                           "session": self.session_name, "licence_class": self._doc_licence_class(cfg),
                                           "filename": fid, "ric": ric, "text": "metadata only"}))
        return docs

    # ------------------------------------------------------------------ field self-check (gate G5 evidence)
    def verify_fields(self, sample_ticker: str | None = None, as_of: date | None = None) -> list[FieldCheck]:
        """Request every mapped field alone for one known security (VENDOR_REFERENCE section 5).

        ``sample_ticker`` defaults to the field map's ``test_ric`` (``IBM.N``). A field passes when one
        non-empty column (or history series) comes back. Requests use the real request parameters, after
        the unverifiable parameters (``SDate``, history ``adjustments``) have been preflighted on
        ``test_ric``; those preflights are reported first (``parameters.<name>``, ``history.adjustments``,
        with the values seen with and without them). Record the passing rows (item, test ticker, date,
        value, units) in ``field_validation_log`` before admitting any item as a predicate.
        """
        as_of = as_of or self._today()
        sample = str(sample_ticker or self.test_ric).strip()
        ric = self.ric_for(sample)
        if ric is None:
            raise ProviderError(f"cannot resolve {sample!r} to a RIC: pass a RIC such as {self.test_ric!r}")
        ld = self._open()
        ctx, params = self._ctx(as_of), self._request_params(as_of)  # unverifiable parameters preflighted first
        adjustments = self._history_adjustments()
        checks: list[FieldCheck] = []
        for name, e in (self.fieldmap.get("parameters") or {}).items():
            log = self.field_log.get(f"parameters.{name}")
            if not isinstance(e, Mapping) or e.get("status") != "unverifiable" or not log:
                continue
            checks.append(FieldCheck(
                f"parameters.{name}", "unverifiable", log.get("with"), log.get("preflight") is True,
                f"{log.get('code')} on {self.test_ric} with {name}={log.get('value')!r}: {log.get('with')!r}; without it: "
                f"{log.get('without')!r}" + (f" ({log['reason']})" if log.get("reason") else "")
                + ("" if log.get("used") else f" - {name} DROPPED from requests"), str(log.get("code") or "")))
        adj_log = self.field_log.get("history.adjustments")
        if adj_log and self._configured_adjustments()[1].get("status") == "unverifiable":
            checks.append(FieldCheck(
                "history.adjustments", "unverifiable", adj_log.get("value"), adj_log.get("preflight") is True,
                f"{adj_log.get('code')} history on {self.test_ric} with adjustments={adj_log.get('value')}: "
                f"{'values' if adj_log.get('preflight') else 'no values'}; without them: "
                f"{'values' if adj_log.get('without') else 'no values'}"
                + (f" ({adj_log['reason']})" if adj_log.get("reason") else "")
                + ("" if adj_log.get("used") else " - adjustments DROPPED from requests"), str(adj_log.get("code") or "")))
        first = [True]

        def pause() -> None:
            if not first[0]:
                self._pause()
            first[0] = False

        def snap(name: str, code: str, status: str, universe: str, unit_note: str, entry: Mapping[str, Any]) -> None:
            pause()
            try:
                df = self._get_data_raw(ld, universe, [code], params)
            except _StopRequests:
                raise
            except Exception as e:  # noqa: BLE001
                checks.append(FieldCheck(name, status, None, False, f"{code}: request failed ({e})", code))
                return
            if not isinstance(df, pd.DataFrame) or df.shape[1] != 2 or df.empty:
                cols = list(df.columns) if isinstance(df, pd.DataFrame) else None
                checks.append(FieldCheck(name, status, None, False, f"{code}: no column returned (dropped: unentitled or "
                                         f"invalid field; columns {cols})", code))
                return
            raw = _py(df.iloc[0, 1])
            if _missing(raw):
                checks.append(FieldCheck(name, status, None, False, f"{code}: column returned but empty for {universe}", code))
                return
            conv = _py(_convert(pd.Series([raw], dtype="object"), entry).iloc[0])
            if isinstance(conv, pd.Timestamp):
                conv = conv.date().isoformat()
            units = "" if entry.get("units_verified", True) else " - units UNVERIFIED: record units/sign before admission"
            checks.append(FieldCheck(name, status, raw, True, f"{code} on {universe} as of {as_of}: {raw!r} [{unit_note}] -> "
                                     f"canonical {conv!r}{units}", code))

        for section, entries in (self.fieldmap.get("raw") or {}).items():
            for key, e in (entries or {}).items():
                if str(key).startswith("_") or not isinstance(e, Mapping):
                    continue
                status = str(e.get("status", ""))
                if e.get("code"):
                    snap(f"{section}.{key}", _render(e["code"], ctx), status, ric, str(e.get("unit", e.get("kind", ""))), e)
                elif e.get("codes") and e.get("ric_template"):
                    sub = str(e["ric_template"]).format(root=_ric_root(ric), ticker=ric_to_ticker(ric), ric=ric)
                    for c in e["codes"]:
                        snap(f"{section}.{key}", _render(c, ctx), status, sub, str(e.get("unit", "")), e)
        for feat, e in (self.fieldmap.get("features") or {}).items():
            if isinstance(e, Mapping) and e.get("expression"):
                scale = float(e.get("threshold_scale", 1) or 1)  # catalog value = vendor value / threshold_scale
                snap(f"feature.{feat}", str(e["expression"]), str(e.get("status", "")), ric,
                     f"{e.get('unit', '')}; catalog units = vendor / {scale:g}",
                     {"to_canonical": 1.0 / scale, "units_verified": e.get("units_verified", True),
                      "kind": "label" if e.get("unit") == "label" else "number"})
        for key, e in (((self.fieldmap.get("history") or {}).get("fields")) or {}).items():
            if not isinstance(e, Mapping) or not e.get("code"):
                continue
            code, status = str(e["code"]), str(e.get("status", ""))
            pause()
            try:
                raw = self._call("get_history", ld.get_history,
                                 **self._history_kwargs([ric], [code], as_of - timedelta(days=14), as_of, adjustments))
                got = _split_history(raw, [ric], [code]).get(ric)
                series = got[code].dropna() if got is not None else pd.Series(dtype="float64")
            except _StopRequests:
                raise
            except Exception as ex:  # noqa: BLE001
                checks.append(FieldCheck(f"history.{key}", status, None, False, f"{code}: get_history failed ({ex})", code))
                continue
            if series.empty:
                checks.append(FieldCheck(f"history.{key}", status, None, False, f"{code}: no daily values for {ric} in the "
                                         f"14 days to {as_of}", code))
            else:
                last = float(series.iloc[-1])
                checks.append(FieldCheck(f"history.{key}", status, last, True, f"{code} on {ric}: last daily value {last!r} "
                                         f"({series.index[-1].date()})", code))
        return checks
