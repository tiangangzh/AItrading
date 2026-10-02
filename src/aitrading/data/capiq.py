"""S&P Capital IQ adapter: the GDS REST API for numbers, the Kensho LLM-ready API for transcripts.

Implements ``MarketDataProvider`` (docs/ARCHITECTURE.md Option C, phase 3; docs/VENDOR_REFERENCE.md
section 3). It deliberately does **not** implement ``ScreenPushdown``: the GDS API has no screen function
(GDS is per identifier x mnemonic), Kensho offers only categorical ticker groups, and the only S&P
push-down is SQL over Xpressfeed / Snowflake, which is out of scope for this adapter. The pipeline
therefore screens a CapIQ universe locally.

Universe
--------
An explicit ``tickers=[...]`` list (typically the survivors of a Bloomberg / LSEG screen), or index
constituents via ``index='^SPX'`` (GDSHV with the field map's ``universe.constituents`` entry and
``StartRank`` / ``EndRank``, paged; ^SPX coverage is untested). With neither, ``get_universe`` raises
``ProviderError`` explaining the options. Canonical tickers map to CapIQ identifiers with the field
map's template (``'IBM'`` -> ``'IBM:'``, ``'BRK-B'`` -> ``'BRK.B:'``); ``identifiers={ticker: 'IBM:NYSE'
| 'IQ<id>'}`` is the maintained symbology table (ADR graft 6) that pins listings. Constituents that
cannot be mapped to a ticker are dropped and flagged. Every security is labelled ``common_stock``
(no verified GDS security-type item) and a warning says ADRs / REITs are not excluded.

Transport (plain HTTPS through ``httpx``; no vendor SDK)
-------------------------------------------------------
* Auth: ``POST {base}/authenticate/api/v1/token`` with form-urlencoded ``username`` / ``password``
  (read from ``$CAPIQ_USERNAME`` / ``$CAPIQ_PASSWORD`` at call time, never logged) returns
  ``access_token``; data calls send ``Authorization: Bearer <token>``. ``/tokenRefresh`` and the token
  lifetime are UNVERIFIED, so a data call answered with HTTP 401 re-authenticates once and retries.
  Missing credentials or a rejected login raise ``ProviderUnavailable`` with entitlement guidance.
* Data: ``POST {base}/v3/clientservice.json`` with ``{"inputRequests": [{function, identifier, mnemonic,
  properties}, ...]}`` (functions GDSP / GDSPV for point values, GDSHE for history, GDSHV for
  constituents; properties such as ``periodType``, ``asOfDate``, ``currencyId``, ``startDate`` /
  ``endDate`` in MM/DD/YYYY). Each response is checked: one ``GDSSDKResponse`` element per request,
  same mnemonic, in order; a lone ``ErrMsg`` element is a whole-call error.
* Batching and budget: requests are de-duplicated, cached for the provider's lifetime and sent in
  batches of ``api.max_requests_per_call``. Every identifier x mnemonic request counts toward the
  observed ~10,000 requests/day limit: ``requests_today`` counts what this provider sent, a warning
  fires at ``warn_fraction`` of the limit, a call that would exceed it raises ``ProviderError`` before
  anything is sent, and a vendor "request limit exceeded" message stops further calls that day.
  ``usage()`` reads the account's real metrics (``/v3/usageservice.json``). ``query_log`` keeps every
  request body verbatim for the audit trail.

Field codes are configuration
-----------------------------
Every mnemonic, property, unit and scale comes from ``src/aitrading/data/fieldmaps/capiq.json``,
deep-merged with the JSON file named by ``$AITRADING_FIELDMAP_CAPIQ`` and then the ``fieldmap=``
argument (see :mod:`aitrading.data.fieldmaps`; ``null`` deletes a key). Sections read here:

* ``raw.<dataset>.<canonical column>``: ``mnemonic`` + ``properties`` (placeholders ``{as_of}``,
  ``{as_of_3m}``, ``{as_of_1y}``), ``kind`` (number / label / date), ``to_canonical`` (canonical = vendor
  x to_canonical; CapIQ amounts default to millions, so x1e6 -> USD absolute, percent x0.01 ->
  fraction, capex x-1e6 -> positive cash spent), ``plausible`` (canonical range; values outside are
  blanked - this catches a wrong 1e6 scale or a flipped sign), ``units_verified``, ``status``,
  ``notes``. Other entry forms: ``derive`` (``multiply`` / ``divide`` / ``divide_by_one_plus`` /
  ``reciprocal`` over ``inputs`` that are mnemonics or other columns), ``from_column`` + ``value_map``,
  ``assume`` (a labelled constant), ``available: false`` (no S&P source: NaN).
* ``history.fields.<open|high|low|close|volume>`` for GDSHE price history (``close`` is required).
* ``features.<catalog feature>`` with ``to_catalog``: vendor values in catalog units for
  reconciliation against locally computed features (:meth:`CapIQProvider.get_vendor_features`); they
  may raise reconciliation flags but never replace the local computation.
* ``benchmark``, ``universe.constituents``, ``identifiers``, ``api``, ``documents``.

Rules applied: ``unverifiable`` mnemonics are preflighted alone on ``test_identifier`` (cached per day)
before any bulk use, and a failure suspends the leg (NaN plus a ``LEG NOT EVALUATED`` warning).
``units_verified: false`` entries are served converted with a warning (``strict_units=True``
withholds them). Missing values are NaN, never 0: an ``ErrMsg``, an empty row, a placeholder such as
``Data Unavailable`` / ``NM`` or any non-numeric value in a numeric column. For an ``as_of`` older
than ``snapshot_staleness_days``, numeric items without an as-of property are blanked (no look-ahead)
and a warning notes that ``asOfDate`` point-in-time behaviour is UNVERIFIED for GDS.

Not available from S&P (VENDOR_REFERENCE 3.3)
---------------------------------------------
Implied volatility and put/call (confirmed) and short interest (none verified: "do not ship") return
NaN frames with a warning and cost no requests; the provider does not advertise those capabilities.

Transcripts (Kensho LLM-ready API, licence class L1 pending gate G2)
--------------------------------------------------------------------
Only when a Kensho client object is injected (``kensho=``) **and** the boundary permits TRANSCRIPT
text; otherwise ``get_documents`` returns ``[]`` with a warning and Kensho is never called. The client is
duck-typed; two routes are supported, in this order:

1. Tool-style (VENDOR_REFERENCE 3.3, corrected): ``kensho.get_latest_earnings_from_identifiers(
   identifiers=[ticker])`` -> a record with ``key_dev_id``, ``name`` and ``datetime`` (bare, or keyed by
   the identifier / under ``results``), then ``kensho.get_transcript_from_key_dev_id(key_dev_id=...)`` ->
   ``'Speaker: text'`` paragraphs (a string, an object with ``.raw``, or a list of components with
   ``person_name`` / ``text``). The method names come from the field map.
2. Object-style (ADR step 7; the ``kfinance`` ``Client``): ``kensho.ticker(sym).company.latest_earnings``
   (``.name``, ``.key_dev_id``, ``.datetime``) then ``.transcript.raw``. :func:`kensho_client_from_env`
   builds such a client (``kfinance`` is imported lazily; ``ProviderUnavailable`` when missing).

The call date must fall inside ``[start, end]`` (the drawdown window); one document per ``key_dev_id``,
split into speaker paragraphs, each hashed with SHA-256 and tagged L1 / G2 in the metadata.

Licensing boundary (ADR section 7)
----------------------------------
Default: ``deny_all_text('capiq', note=BOUNDARY_NOTE)`` with ``allow_numeric_features=False``. CIQ GDS
values are licence class L3 (enterprise data, AI use unconfirmed: only ranks and booleans may reach
an external model until gate **G3**); Kensho transcripts are L1 pending gate **G2**. Widen it only with
an explicit ``boundary=DataBoundary(provider='capiq', ...)`` whose ``note`` names the gate: ``G2`` to
permit TRANSCRIPT text, ``G3`` to permit numeric features or other document kinds. A widened boundary
whose note does not cite its gate raises ``ValueError`` (a policy row without a citation is a deny).
Broker research (L6) can never be permitted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import httpx
import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind, TranscriptSegment
from aitrading.core.policy import DataBoundary, deny_all_text
from aitrading.data.base import Capability, PricePanel, ProviderError, ProviderUnavailable
from aitrading.data.fieldmaps import FieldMapError, digest, env_var
from aitrading.data.fieldmaps import load_fieldmap as _load_vendor_fieldmap
from aitrading.screen.catalog import default_catalog

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.spec import UniverseSpec

__all__ = [
    "CapIQProvider",
    "FieldCheck",
    "load_fieldmap",
    "validate_fieldmap",
    "kensho_client_from_env",
    "parse_transcript",
    "is_identifier",
    "identifier_to_ticker",
    "ticker_to_identifier",
    "VENDOR",
    "FIELDMAP_ENV",
    "DEFAULT_BASE_URL",
    "BOUNDARY_NOTE",
    "CREDENTIALS_MISSING",
    "KENSHO_MISSING",
]

VENDOR = "capiq"
FIELDMAP_ENV = env_var(VENDOR)  # AITRADING_FIELDMAP_CAPIQ
DEFAULT_BASE_URL = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest"  # VENDOR_REFERENCE 3.1

BOUNDARY_NOTE = (
    "S&P Capital IQ GDS values are licence class L3 (enterprise data, AI use unconfirmed): only ranks and booleans "
    "may reach an external model until gate G3 (written S&P AI-use confirmation for the dataset). Kensho LLM-ready "
    "API transcripts and line items are licence class L1 pending gate G2 (written S&P confirmation that Kensho data "
    "and transcripts may be processed by Anthropic for inference, with log retention and internal memo "
    "distribution). Until then no S&P text or values reach Claude. Widen only with an explicit "
    "boundary=DataBoundary(provider='capiq', allowed_document_kinds={DocumentKind.TRANSCRIPT}, "
    "note='G2: <S&P confirmation reference>') and/or allow_numeric_features=True with a note citing G3."
)
CREDENTIALS_MISSING = (
    "S&P Capital IQ GDS API credentials are not set: export {user_env} and {pw_env} (the API username and password "
    "from the S&P API welcome letter). The account needs a Capital IQ GDS API entitlement; the observed limit is about "
    "10,000 requests a day, so use it for survivors and cross-checks, not universe-wide screens."
)
KENSHO_MISSING = (
    "The Kensho LLM-ready API client is not installed: pip install kensho-kfinance==8.1.0 (Python >= 3.10). It needs a "
    "Kensho entitlement (client_id plus a private key; TranscriptsPermission for transcripts). Kensho data is licence "
    "class L1 pending gate G2: inject the client with CapIQProvider(kensho=...) and widen the boundary only after G2."
)

DATASET_COLUMNS: dict[str, list[str]] = {
    "universe": [c for c in F.UNIVERSE_COLUMNS if c != F.VENDOR_ID],
    "fundamentals": list(F.FUNDAMENTAL_COLUMNS),
    "estimates": list(F.ESTIMATE_COLUMNS),
    "short_interest": list(F.SHORT_INTEREST_COLUMNS),
    "options": list(F.OPTIONS_COLUMNS),
}
DATE_COLUMNS = frozenset({F.PERIOD_END, F.REPORT_DATE, F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE, F.SI_SETTLEMENT_DATE})
LABEL_COLUMNS = frozenset({F.NAME, F.GICS_SECTOR, F.GICS_INDUSTRY, F.EXCHANGE, F.COUNTRY, F.CURRENCY, F.SECURITY_TYPE, F.VENDOR_ID})
FUTURE_DATE_COLUMNS = frozenset({F.NEXT_EARNINGS_DATE})  # may legitimately lie after as_of
GDS_FUNCTIONS = frozenset({"GDSP", "GDSPV", "GDSHE", "GDSHV", "GDST", "GDSG"})
POINT_FUNCTIONS = frozenset({"GDSP", "GDSPV"})
HISTORY_FUNCTIONS = frozenset({"GDSHE"})
KINDS = ("number", "label", "date")
DERIVE_ARITY = {"multiply": 2, "divide": 2, "divide_by_one_plus": 2, "reciprocal": 1}
PROBE_DAYS = 10  # history window used to preflight / verify a history mnemonic

_GATE_G2 = re.compile(r"\bG2\b")
_GATE_G3 = re.compile(r"\bG3\b")
_LIMIT_RE = re.compile(r"limit.*exceed|exceed.*limit", re.IGNORECASE)
_TICKER_EXCH_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9.\-/&]*):([A-Za-z0-9 .\-]*)$")
_ID_PATTERNS = (
    re.compile(r"^IQT?\d+$", re.IGNORECASE),  # IQ<companyId>, IQT<tradingItemId>
    re.compile(r"^I_[A-Z0-9]{12}$", re.IGNORECASE),  # I_<ISIN>
    re.compile(r"^CSP_[A-Z0-9]{9}$", re.IGNORECASE),  # CSP_<CUSIP>
    re.compile(r"^GV\d+$", re.IGNORECASE),  # GV<gvkey>
    re.compile(r"^\^\S+$"),  # ^<index>
)
_PLAIN_TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,14}$")
_PLACEHOLDER_RE = re.compile(r"^\{(\w+)\}$")
_SPEAKER_RE = re.compile(r"^\s*([A-Z][^:\n]{0,79}?):\s+(\S.*)$")
_QA_HEADER_RE = re.compile(r"^\s*(question[- ]and[- ]answer(?: session)?|questions? and answers?|q\s*&\s*a)\s*:?\s*$", re.IGNORECASE)
_QA_START_RE = re.compile(
    r"\b(?:first|next) question\b|\bnow (?:begin|open|start|take|conduct)\b.{0,60}\bquestion|"
    r"\bopen (?:up )?the (?:call|line|lines|floor) (?:up )?(?:for|to) questions\b",
    re.IGNORECASE,
)


# =============================================================================================
# Small helpers
# =============================================================================================


def _as_date(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def _dedupe(items: Iterable[Any]) -> list[str]:
    out: dict[str, None] = {}
    for x in items or []:
        s = str(x).strip()
        if s:
            out.setdefault(s, None)
    return list(out)


def _value(entry: Any, default: Any = None) -> Any:
    """``entry['value']`` for ``{"value": ..., "status": ...}`` map entries, else the entry itself."""
    if isinstance(entry, Mapping):
        v = entry.get("value")
        return default if v is None else v
    return default if entry is None else entry


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _to_float(v: Any) -> float:
    """Finite float or NaN (never 0 for something unparseable)."""
    if v is None or isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else math.nan
    s = str(v).strip().replace(",", "")
    if not s:
        return math.nan
    try:
        f = float(s)
    except ValueError:
        return math.nan
    return f if math.isfinite(f) else math.nan


def _errmsg(el: Any) -> str:
    if not isinstance(el, Mapping):
        return "malformed response element"
    return str(el.get("ErrMsg") or "").strip()


def _rows(el: Any) -> list[list[Any]]:
    """``Rows`` of a GDSSDKResponse element as lists (``[{"Row": [...]}, ...]``)."""
    if not isinstance(el, Mapping):
        return []
    out: list[list[Any]] = []
    for r in el.get("Rows") or []:
        v = r.get("Row") if isinstance(r, Mapping) else r
        if isinstance(v, (list, tuple)):
            out.append(list(v))
        elif v is not None:
            out.append([v])
    return out


def _render(v: Any, ctx: Mapping[str, Any]) -> Any:
    """Fill ``{placeholder}`` templates; a bare ``"{name}"`` keeps the context value's type (e.g. int ranks)."""
    if not isinstance(v, str):
        return v
    m = _PLACEHOLDER_RE.match(v)
    if m and m.group(1) in ctx:
        return ctx[m.group(1)]
    try:
        return v.format(**ctx)
    except (KeyError, IndexError, ValueError) as e:
        raise FieldMapError(f"property template {v!r} uses an unknown placeholder ({e}); known: {sorted(ctx)}") from e


def _naive_utc(x: Any) -> datetime | None:
    if x is None:
        return None
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(timezone.utc).tz_localize(None)
    return ts.to_pydatetime()


def _attr(obj: Any, *names: str) -> Any:
    """First non-None of ``obj[name]`` / ``obj.name`` for the given names."""
    for n in names:
        if isinstance(obj, Mapping):
            v = obj.get(n)
        else:
            try:
                v = getattr(obj, n, None)
            except Exception:  # noqa: BLE001 - lazy SDK properties may raise
                v = None
        if v is not None:
            return v
    return None


def _call_kw(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with keyword arguments, falling back to positional ones for a different signature."""
    try:
        return fn(**kwargs)
    except TypeError as e:
        if "argument" not in str(e):
            raise
        return fn(*kwargs.values())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================================
# Identifiers
# =============================================================================================


def is_identifier(s: str) -> bool:
    """True for CapIQ identifier forms (VENDOR_REFERENCE 3.1): ``TICKER:EXCH``, ``TICKER:``, ``IQ<id>``,
    ``IQT<id>``, ``I_<ISIN>``, ``CSP_<CUSIP>``, ``GV<gvkey>``, ``^<index>``."""
    t = str(s).strip()
    return bool(_TICKER_EXCH_RE.match(t)) or any(p.match(t) for p in _ID_PATTERNS)


def identifier_to_ticker(identifier: str, class_separator: str = ".") -> str | None:
    """Canonical ticker for a CapIQ identifier: ``'IBM:NYSE'`` -> ``'IBM'``, ``'BRK.B:'`` -> ``'BRK-B'``,
    a plain ``'AAPL'`` -> ``'AAPL'``; ``None`` for ids that carry no ticker (``IQ<id>``, ISIN, CUSIP, ^index)."""
    t = str(identifier).strip()
    m = _TICKER_EXCH_RE.match(t)
    if m:
        root = m.group(1)
    elif any(p.match(t) for p in _ID_PATTERNS):
        return None
    elif _PLAIN_TICKER_RE.match(t):
        root = t
    else:
        return None
    if class_separator and class_separator != "-":
        root = root.replace(class_separator, "-")
    return root.upper()


def ticker_to_identifier(ticker: str, template: str = "{ticker}:", class_separator: str = ".") -> str:
    """CapIQ identifier for a canonical ticker (identifiers pass through): ``'BRK-B'`` -> ``'BRK.B:'``."""
    t = str(ticker).strip()
    if is_identifier(t):
        return t
    root = t.upper().replace("-", class_separator) if class_separator else t.upper()
    return template.format(ticker=root)


def _exchange_part(identifier: str) -> str | None:
    m = _TICKER_EXCH_RE.match(str(identifier).strip())
    if m and m.group(2).strip():
        return m.group(2).strip()
    return None


# =============================================================================================
# Transcripts
# =============================================================================================


def _role(speaker: str) -> str:
    return "Operator" if speaker.strip().lower() == "operator" else ""


def parse_transcript(raw: str) -> list[TranscriptSegment]:
    """Split ``'Speaker: text'`` paragraphs (the Kensho ``transcript.raw`` form) into segments.

    Lines without a ``Speaker:`` prefix continue the current paragraph. The section switches to ``qa`` at a
    "Question-and-Answer" header line or at the first Operator paragraph that opens questions ("first
    question", "now open the line for questions", ...): a heuristic, recorded in the document metadata.
    Roles are only known for the Operator (other roles are left empty rather than guessed).
    """
    segs: list[TranscriptSegment] = []
    section = "prepared_remarks"
    cur: dict[str, Any] | None = None

    def flush() -> None:
        if cur is not None:
            text = " ".join(cur["lines"]).strip()
            if text:
                segs.append(TranscriptSegment(speaker=cur["speaker"], role=_role(cur["speaker"]), section=cur["section"], text=text))

    for line in str(raw or "").splitlines():
        if not line.strip():
            continue
        if _QA_HEADER_RE.match(line):
            flush()
            cur, section = None, "qa"
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            flush()
            speaker, text = m.group(1).strip(), m.group(2).strip()
            if speaker.lower() == "operator" and _QA_START_RE.search(text):
                section = "qa"
            cur = {"speaker": speaker, "lines": [text], "section": section}
        elif cur is None:
            cur = {"speaker": "", "lines": [line.strip()], "section": section}
        else:
            cur["lines"].append(line.strip())
    flush()
    return segs


def _segments_from_payload(payload: Any, depth: int = 0) -> list[TranscriptSegment]:
    """Segments from whatever a Kensho transcript call returned (str, ``.raw``, components list)."""
    if payload is None or depth > 3:
        return []
    if isinstance(payload, str):
        return parse_transcript(payload)
    if isinstance(payload, (list, tuple)):
        segs: list[TranscriptSegment] = []
        section = "prepared_remarks"
        for c in payload:
            if isinstance(c, str):
                segs.extend(parse_transcript(c))
                continue
            text = _attr(c, "text", "component_text", "componentText")
            if not isinstance(text, str) or not text.strip():
                continue
            speaker = str(_attr(c, "person_name", "speaker", "name", "personName") or "").strip()
            ctype = str(_attr(c, "component_type", "type", "componentType") or "").lower()
            if "question" in ctype or "answer" in ctype:
                section = "qa"
            segs.append(TranscriptSegment(speaker=speaker, role=_role(speaker), section=section, text=text.strip()))
        return segs
    inner = _attr(payload, "raw", "transcript", "text", "components")
    return _segments_from_payload(inner, depth + 1) if inner is not None and inner is not payload else []


def _find_earnings(obj: Any, ident: str, depth: int = 0) -> Any:
    """The latest-earnings record (anything with a ``key_dev_id``) inside a Kensho tool result."""
    if obj is None or depth > 5:
        return None
    if isinstance(obj, (list, tuple)):
        for x in obj:
            r = _find_earnings(x, ident, depth + 1)
            if r is not None:
                return r
        return None
    if _attr(obj, "key_dev_id", "keyDevId") is not None:
        return obj
    m: Any = obj if isinstance(obj, Mapping) else (obj.model_dump() if hasattr(obj, "model_dump") else None)
    if not isinstance(m, Mapping):
        return None
    for k, v in m.items():
        if str(k).strip().upper() == ident.strip().upper():
            return _find_earnings(v, ident, depth + 1)
    for k in ("results", "result", "data", "latest_earnings", "earnings"):
        if k in m:
            r = _find_earnings(m[k], ident, depth + 1)
            if r is not None:
                return r
    if len(m) == 1:
        return _find_earnings(next(iter(m.values())), ident, depth + 1)
    return None


def kensho_client_from_env(
    client_id_env: str = "KENSHO_CLIENT_ID",
    private_key_env: str = "KENSHO_PRIVATE_KEY",
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Build a ``kfinance`` ``Client(client_id=..., private_key=...)`` for ``CapIQProvider(kensho=...)``.

    ``kfinance`` (``pip install kensho-kfinance``) is imported lazily; ``ProviderUnavailable`` when it is
    missing or the credentials are not set. Transcripts still need a boundary that cites gate G2.
    """
    try:
        mod = importlib.import_module("kfinance.client.kfinance")
    except ImportError as e:
        raise ProviderUnavailable(KENSHO_MISSING) from e
    environ = os.environ if env is None else env
    cid, key = environ.get(client_id_env), environ.get(private_key_env)
    if not cid or not key:
        raise ProviderUnavailable(f"Kensho credentials are not set: export {client_id_env} and {private_key_env} "
                                  f"(client id and private key from Kensho). {KENSHO_MISSING}")
    try:
        return mod.Client(client_id=cid, private_key=key)
    except Exception as e:  # noqa: BLE001
        raise ProviderUnavailable(f"could not create the Kensho client ({type(e).__name__}: {e})") from e


# =============================================================================================
# Field map
# =============================================================================================


def _entry_errors(path: str, e: Any, scale_key: str, columns: Mapping[str, Any] | None, *, functions: frozenset[str]) -> list[str]:
    if not isinstance(e, Mapping):
        return [f"{path}: must be an object"]
    errs: list[str] = []
    if "status" not in e:
        errs.append(f"{path}: missing status")
    forms = [k for k in ("mnemonic", "derive", "assume", "from_column", "compute") if e.get(k) is not None]
    if not forms and e.get("available") is not False:
        errs.append(f"{path}: needs one of mnemonic / derive / assume / from_column / compute, or available=false")
    if len(forms) > 1:
        errs.append(f"{path}: only one of {forms} may be set")
    if e.get("kind", "number") not in KINDS:
        errs.append(f"{path}.kind = {e.get('kind')!r} (must be one of {list(KINDS)})")
    if e.get("mnemonic") is not None and not isinstance(e.get("mnemonic"), str):
        errs.append(f"{path}.mnemonic must be a string")
    if "function" in e and e["function"] not in functions:
        errs.append(f"{path}.function = {e['function']!r} (must be one of {sorted(functions)})")
    if "properties" in e and not isinstance(e["properties"], Mapping):
        errs.append(f"{path}.properties must be an object")
    for key in ("to_canonical", "to_catalog"):
        if key in e and (not _is_number(e[key]) or float(e[key]) == 0):
            errs.append(f"{path}.{key} must be a non-zero number")
    if "plausible" in e:
        p = e["plausible"]
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_number(x) for x in p) and p[0] <= p[1]):
            errs.append(f"{path}.plausible must be [low, high] numbers with low <= high")
    if "value_map" in e and not isinstance(e["value_map"], Mapping):
        errs.append(f"{path}.value_map must be an object")
    if e.get("derive") is not None:
        op = e["derive"]
        if op not in DERIVE_ARITY:
            errs.append(f"{path}.derive = {op!r} (must be one of {sorted(DERIVE_ARITY)})")
        inputs = e.get("inputs")
        if not isinstance(inputs, list) or (op in DERIVE_ARITY and len(inputs) != DERIVE_ARITY[op]):
            errs.append(f"{path}.inputs must be a list of {DERIVE_ARITY.get(op, '?')} input(s)")
        else:
            for i, inp in enumerate(inputs):
                ipath = f"{path}.inputs[{i}]"
                if isinstance(inp, Mapping) and inp.get("column") is not None:
                    col = inp["column"]
                    target = (columns or {}).get(col)
                    if columns is None or col not in columns:
                        errs.append(f"{ipath}: column {col!r} is not a column of this section")
                    elif isinstance(target, Mapping) and target.get("derive") is not None:
                        errs.append(f"{ipath}: column {col!r} is itself derived (no chains)")
                elif isinstance(inp, Mapping) and inp.get("mnemonic"):
                    errs += _entry_errors(ipath, inp, "to_canonical", None, functions=POINT_FUNCTIONS)
                else:
                    errs.append(f"{ipath}: needs 'column' or 'mnemonic'")
    if e.get("from_column") is not None and (columns is None or e["from_column"] not in columns):
        errs.append(f"{path}.from_column {e['from_column']!r} is not a column of this section")
    return errs


def validate_fieldmap(fm: Mapping[str, Any]) -> list[str]:
    """Human-readable problems with a (merged) S&P Capital IQ field map; empty list means usable."""
    errs: list[str] = []
    if fm.get("vendor") != VENDOR:
        errs.append(f"vendor must be {VENDOR!r} (got {fm.get('vendor')!r})")
    if not isinstance(fm.get("test_identifier"), str) or not fm.get("test_identifier"):
        errs.append("test_identifier (a known security, e.g. 'IBM:NYSE') is required")
    api = fm.get("api") or {}
    for key in ("data_path", "token_path"):
        if not isinstance(_value(api.get(key)), str):
            errs.append(f"api.{key}.value is required")
    for key in ("max_requests_per_call", "daily_request_limit"):
        v = _value(api.get(key))
        if v is not None and (not _is_number(v) or float(v) < 1):
            errs.append(f"api.{key}.value must be a number >= 1")
    raw = fm.get("raw") or {}
    if not isinstance(raw, Mapping):
        errs.append("raw must be an object")
        raw = {}
    for ds, entries in raw.items():
        if ds not in DATASET_COLUMNS:
            errs.append(f"raw.{ds}: unknown dataset (expected one of {sorted(DATASET_COLUMNS)})")
            continue
        if not isinstance(entries, Mapping):
            errs.append(f"raw.{ds} must be an object")
            continue
        for col, e in entries.items():
            if str(col).startswith("_"):
                continue
            if col not in DATASET_COLUMNS[ds]:
                errs.append(f"raw.{ds}.{col}: not a canonical column of {ds}")
            errs += _entry_errors(f"raw.{ds}.{col}", e, "to_canonical", entries, functions=POINT_FUNCTIONS)
    hist = fm.get("history") or {}
    if hist.get("function", "GDSHE") not in HISTORY_FUNCTIONS:
        errs.append(f"history.function must be one of {sorted(HISTORY_FUNCTIONS)}")
    hf = hist.get("fields") or {}
    for key, e in hf.items():
        if key not in F.PRICE_FIELDS:
            errs.append(f"history.fields.{key}: not one of {F.PRICE_FIELDS}")
        errs += _entry_errors(f"history.fields.{key}", e, "to_canonical", None, functions=HISTORY_FUNCTIONS)
    if not (isinstance(hf.get(F.CLOSE), Mapping) and hf[F.CLOSE].get("mnemonic")):
        errs.append("history.fields.close.mnemonic is required")
    catalog = default_catalog()
    feats = fm.get("features") or {}
    for name, e in feats.items():
        if str(name).startswith("_"):
            continue
        if name not in catalog:
            errs.append(f"features.{name}: not a catalog feature")
        errs += _entry_errors(f"features.{name}", e, "to_catalog", feats, functions=POINT_FUNCTIONS)
    cons = (fm.get("universe") or {}).get("constituents")
    if cons is not None:
        if not isinstance(cons, Mapping) or not cons.get("mnemonic") or cons.get("function") not in GDS_FUNCTIONS:
            errs.append("universe.constituents needs mnemonic and a GDS function")
    bm = fm.get("benchmark")
    if bm is not None and (not isinstance(bm, Mapping) or not bm.get("identifier") or not bm.get("mnemonic")):
        errs.append("benchmark needs identifier and mnemonic")
    return errs


def load_fieldmap(override: Mapping[str, Any] | str | os.PathLike[str] | None = None, *,
                  env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Default S&P map deep-merged with ``$AITRADING_FIELDMAP_CAPIQ`` and then ``override``.

    Raises :class:`aitrading.data.fieldmaps.FieldMapError` (a ``ValueError``) listing every problem.
    """
    fm = _load_vendor_fieldmap(VENDOR, override, env=env)
    errs = validate_fieldmap(fm)
    if errs:
        raise FieldMapError(f"invalid S&P Capital IQ field map ({' <- '.join(fm.get('_sources', []))}): " + "; ".join(errs))
    return fm


# =============================================================================================
# Field self-check record
# =============================================================================================


@dataclass
class FieldCheck:
    """One mapped item requested alone on a known security (field-admission procedure, gate G5)."""

    field: str  # 'raw.<dataset>.<column>[.inputs[i]]', 'history.<field>', 'features.<feature>', 'benchmark', ...
    status_in_map: str  # confirmed / corrected / unverifiable
    returned_value: Any  # vendor value as returned (before unit conversion); None when nothing came back
    ok: bool
    note: str = ""
    mnemonic: str = ""
    function: str = ""
    identifier: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    canonical_value: Any = None  # returned_value x to_canonical (raw/history) or x to_catalog (features)
    as_of: date | None = None

    def to_log_row(self, reviewer: str = "") -> dict[str, Any]:
        """Row for ``field_validation_log`` (VENDOR_REFERENCE section 5, step 4)."""
        return {
            "vendor": VENDOR, "item": self.mnemonic, "field": self.field, "function": self.function,
            "params": dict(self.properties), "status": self.status_in_map, "test_ticker": self.identifier,
            "as_of": self.as_of.isoformat() if self.as_of else None, "value": self.returned_value,
            "canonical_value": self.canonical_value, "ok": self.ok, "note": self.note, "reviewer": reviewer,
            "date": date.today().isoformat(),
        }


@dataclass(eq=False)
class _Leg:
    """One mnemonic + rendered properties, with its unit conversion (a column of requests)."""

    label: str
    function: str
    mnemonic: str
    properties: dict[str, Any]
    probe_properties: dict[str, Any]
    kind: str = "number"
    scale: float = 1.0
    plausible: tuple[float, float] | None = None
    units_verified: bool = True
    unit: str = ""
    status: str = "unverifiable"
    as_of_capable: bool = False

    def request(self, identifier: str, *, probe: bool = False) -> dict[str, Any]:
        props = self.probe_properties if probe else self.properties
        return {"function": self.function, "identifier": identifier, "mnemonic": self.mnemonic, "properties": dict(props)}

    @property
    def key(self) -> str:
        return json.dumps([self.function, self.mnemonic, self.properties], sort_keys=True, default=str)

    @property
    def probe_key(self) -> str:
        return json.dumps([self.function, self.mnemonic, self.probe_properties], sort_keys=True, default=str)


# =============================================================================================
# Provider
# =============================================================================================


class CapIQProvider:
    """``MarketDataProvider`` over the S&P Capital IQ GDS REST API (+ Kensho transcripts). No push-down."""

    name = VENDOR

    def __init__(
        self,
        *,
        boundary: DataBoundary | None = None,
        fieldmap: Mapping[str, Any] | str | os.PathLike[str] | None = None,
        username_env: str = "CAPIQ_USERNAME",
        password_env: str = "CAPIQ_PASSWORD",
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        tickers: Sequence[str] | None = None,
        index: str | None = None,
        kensho: Any | None = None,
        identifiers: Mapping[str, str] | None = None,
        preflight: bool = True,
        strict_units: bool = False,
        snapshot_staleness_days: int = 5,
        timeout: float = 60.0,
        today: Any | None = None,
    ) -> None:
        """
        Args:
            boundary: licensing boundary; default denies all S&P text and numeric values (see module docs).
                A widened boundary must have ``provider='capiq'`` and cite gate G2 (transcripts) / G3 (values).
            fieldmap: override merged over the default map and ``$AITRADING_FIELDMAP_CAPIQ`` (mapping or JSON path).
            username_env / password_env: environment variables holding the GDS API credentials.
            base_url: GDS REST root; ``None`` -> the field map's ``api.base_url`` (``DEFAULT_BASE_URL``,
                VENDOR_REFERENCE 3.1).
            transport: ``httpx`` transport (``httpx.MockTransport`` in tests); default: a normal HTTPS transport.
            tickers: explicit universe (canonical tickers or CapIQ identifiers).
            index: index identifier for constituents (``'^SPX'``; a bare ``'SPX'`` gets the ``^``).
            kensho: injected Kensho client (see module docs); transcripts also need a G2 boundary.
            identifiers: symbology table ``{ticker: CapIQ identifier}`` (``'IBM:NYSE'``, ``'IQ<id>'``).
            preflight: preflight ``unverifiable`` mnemonics on ``test_identifier`` before bulk use.
            strict_units: withhold ``units_verified: false`` entries (NaN) instead of serving them converted.
            snapshot_staleness_days: an ``as_of`` older than this is historical (current-only items blanked).
            timeout: HTTP timeout in seconds.
            today: clock returning a ``date`` (tests).
        """
        self.fieldmap: dict[str, Any] = load_fieldmap(fieldmap)
        api = self.fieldmap.get("api") or {}
        self.base_url = str(base_url or _value(api.get("base_url")) or DEFAULT_BASE_URL).rstrip("/")
        self._data_url = self.base_url + str(_value(api.get("data_path")))
        self._token_url = self.base_url + str(_value(api.get("token_path")))
        usage = api.get("usage_path") if isinstance(api.get("usage_path"), Mapping) else {}
        self._usage_url = self.base_url + str(usage["value"]) if usage.get("value") else None
        self._usage_mnemonic = usage.get("mnemonic")
        self.username_env = username_env
        self.password_env = password_env
        self.timeout = float(timeout)
        self.max_requests_per_call = max(1, int(_value(api.get("max_requests_per_call"), 100)))
        limit_cfg = api.get("daily_request_limit")
        self.daily_request_limit = max(1, int(_value(limit_cfg, 10000)))
        self.warn_fraction = float(limit_cfg.get("warn_fraction", 0.8)) if isinstance(limit_cfg, Mapping) else 0.8
        self.date_format = str(_value(api.get("date_format"), "%m/%d/%Y"))
        row_cfg = api.get("history_row") if isinstance(api.get("history_row"), Mapping) else {}
        self._value_index = int(row_cfg.get("value_index", 0))
        self._date_index = int(row_cfg.get("date_index", 1))
        self._missing_values = {str(x).strip().lower() for x in (_value(api.get("missing_values"), []) or [])}
        self._entitlement_values = {str(x).strip().lower() for x in (_value(api.get("entitlement_values"), []) or [])}
        self.test_identifier: str = str(self.fieldmap["test_identifier"])
        ident_cfg = self.fieldmap.get("identifiers") or {}
        self._template = str(_value(ident_cfg.get("ticker_template"), "{ticker}:"))
        self._class_sep = str(_value(ident_cfg.get("share_class_separator"), "."))
        self.preflight = bool(preflight)
        self.strict_units = bool(strict_units)
        self.snapshot_staleness_days = int(snapshot_staleness_days)
        self._today = today or date.today
        self.kensho = kensho
        self.index: str | None = self._index_identifier(index) if index else None
        self.warnings: list[str] = []
        self.query_log: list[str] = []  # every request body sent (GDS JSON, Kensho calls), verbatim
        self.http_calls = 0  # data calls (POST clientservice), excluding auth
        self.auth_calls = 0
        self.requests_today = 0  # identifier x mnemonic requests sent by this provider today
        self._counter_day = self._today()
        self._limit_warned_day: date | None = None
        self._exhausted_day: date | None = None
        self._transport = transport
        self._http: httpx.Client | None = None
        self._token: str | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._preflight_cache: dict[tuple[date, str], tuple[bool, str]] = {}
        self._ids: dict[str, str] = {}
        self._tickers_by_id: dict[str, str] = {}
        self._historical_noted: set[date] = set()
        for t, ident in (identifiers or {}).items():
            self._register(str(t).strip(), str(ident).strip(), force=True)
        self._tickers: list[str] | None = None
        if tickers:
            out: list[str] = []
            for raw in _dedupe(tickers):
                t = raw
                if is_identifier(raw):
                    t = identifier_to_ticker(raw, self._class_sep) or raw
                    self._register(t, raw)
                if t not in out:
                    out.append(t)
            self._tickers = out
        self.boundary = self._resolve_boundary(boundary)
        self.capabilities: set[Capability] = self._capabilities()

    # ------------------------------------------------------------------ housekeeping
    @property
    def tickers(self) -> list[str]:
        """The explicit universe (empty when the universe comes from ``index``)."""
        return list(self._tickers or [])

    @property
    def fieldmap_digest(self) -> str:
        """SHA-256 of the field map in force (for the audit record)."""
        return digest(self.fieldmap)

    @property
    def requests_remaining(self) -> int:
        """Requests left today under the observed daily limit, as counted by this provider."""
        self._roll_day()
        return max(0, self.daily_request_limit - self.requests_today)

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _roll_day(self) -> None:
        d = self._today()
        if d != self._counter_day:
            self._counter_day, self.requests_today = d, 0

    def _is_historical(self, as_of: date) -> bool:
        return as_of < self._today() - timedelta(days=self.snapshot_staleness_days)

    def _note_historical(self, as_of: date) -> None:
        if self._is_historical(as_of) and as_of not in self._historical_noted:
            self._historical_noted.add(as_of)
            self._warn(f"capiq: as_of {as_of} is historical: values are anchored with the asOfDate property, whose "
                       "point-in-time behaviour on GDS is UNVERIFIED (VENDOR_REFERENCE 3.3 lists asOfDate only for "
                       "estimate revisions) - check with verify_fields(as_of=...). Items without an as-of property are "
                       "left blank rather than leaking current values.")

    @staticmethod
    def _index_identifier(index: str) -> str:
        s = str(index).strip()
        return s if (s.startswith("^") or is_identifier(s)) else "^" + s

    def _register(self, ticker: str, identifier: str, *, force: bool = False) -> None:
        if force or ticker not in self._ids:
            self._ids[ticker] = identifier
        self._tickers_by_id.setdefault(identifier, ticker)
        self._tickers_by_id.setdefault(identifier.upper(), ticker)

    def _identifier(self, ticker: str) -> str:
        t = str(ticker).strip()
        if t in self._ids:
            return self._ids[t]
        return ticker_to_identifier(t, self._template, self._class_sep)

    def _resolve_boundary(self, boundary: DataBoundary | None) -> DataBoundary:
        if boundary is None:
            return deny_all_text(VENDOR, note=BOUNDARY_NOTE).model_copy(update={"allow_numeric_features": False})
        if not isinstance(boundary, DataBoundary):
            raise TypeError("boundary must be an aitrading.core.policy.DataBoundary")
        if boundary.provider != VENDOR:
            raise ValueError(f"boundary.provider must be {VENDOR!r} (got {boundary.provider!r})")
        note = boundary.note or ""
        kinds = set(boundary.allowed_document_kinds)
        if (boundary.allow_numeric_features or kinds) and note.strip() == BOUNDARY_NOTE:
            raise ValueError("BOUNDARY_NOTE describes the default deny state and is not a gate citation: a widened "
                             "boundary's note must cite the written confirmation (e.g. 'G2: S&P letter 2026-xx-xx')")
        if DocumentKind.RESEARCH in kinds:
            raise ValueError("broker research is licence class L6 and never reaches an external model (and S&P offers "
                             "none via API): remove DocumentKind.RESEARCH from the boundary")
        if DocumentKind.TRANSCRIPT in kinds and not _GATE_G2.search(note):
            raise ValueError("a boundary that permits Kensho TRANSCRIPT text must cite gate G2 (written S&P confirmation "
                             "that Kensho data and transcripts may be processed by Anthropic) in its note, e.g. "
                             "note='G2: S&P letter 2026-xx-xx, reviewed by <lawyer>'; a policy row without a citation "
                             "evaluates as deny")
        if (boundary.allow_numeric_features or (kinds - {DocumentKind.TRANSCRIPT})) and not _GATE_G3.search(note):
            raise ValueError("a boundary that permits S&P numeric values or other document text must cite gate G3 "
                             "(written per-dataset AI-use confirmation; CIQ GDS is licence class L3) in its note; a "
                             "policy row without a citation evaluates as deny")
        return boundary

    def _fetchable(self, e: Any, *, ignore_units: bool = False) -> bool:
        """True when an entry would be requested (it has an S&P item that is not withheld)."""
        if not isinstance(e, Mapping) or e.get("available") is False or e.get("compute") is not None:
            return False
        if e.get("mnemonic"):
            return ignore_units or not (self.strict_units and e.get("units_verified") is False)
        if e.get("derive") is not None:
            return all(self._fetchable(i, ignore_units=ignore_units)
                       for i in (e.get("inputs") or []) if isinstance(i, Mapping) and i.get("mnemonic"))
        return False

    def _capabilities(self) -> set[Capability]:
        caps: set[Capability] = set()
        hist = ((self.fieldmap.get("history") or {}).get("fields") or {})
        if self._fetchable(hist.get(F.CLOSE)):
            caps.add(Capability.PRICES)
        raw = self.fieldmap.get("raw") or {}
        for ds, cap in (("fundamentals", Capability.FUNDAMENTALS), ("estimates", Capability.ESTIMATES),
                        ("short_interest", Capability.SHORT_INTEREST), ("options", Capability.OPTIONS)):
            if any(self._fetchable(e) for e in (raw.get(ds) or {}).values()):
                caps.add(cap)
        if self.kensho is not None and DocumentKind.TRANSCRIPT in self.boundary.allowed_document_kinds:
            caps.add(Capability.TRANSCRIPTS)
        return caps  # never SCREEN_PUSHDOWN: the GDS API has no screen function

    def _credentials_message(self) -> str:
        return CREDENTIALS_MISSING.format(user_env=f"${self.username_env}", pw_env=f"${self.password_env}")

    def _no_universe_message(self) -> str:
        cons = (self.fieldmap.get("universe") or {}).get("constituents") or {}
        how = f"{cons.get('function', 'GDSHV')} {cons.get('mnemonic', '?')} with StartRank/EndRank" if cons else "not configured"
        return ("S&P Capital IQ cannot screen the universe: the GDS API has no screen function (VENDOR_REFERENCE 3.2). "
                "Pass CapIQProvider(tickers=[...]) - e.g. the survivors of a Bloomberg or LSEG screen, with "
                "identifiers={ticker: 'IBM:NYSE' | 'IQ<id>'} to pin listings - or CapIQProvider(index='^SPX') for index "
                f"constituents ({how}; ^SPX coverage untested). A universe-wide S&P screen needs SQL over "
                "Xpressfeed/Snowflake, which is out of scope for this adapter.")

    def diagnostics(self) -> list[str]:
        """Setup problems that would blank out data (empty list = ready). Makes no network call."""
        problems: list[str] = []
        if not os.environ.get(self.username_env) or not os.environ.get(self.password_env):
            problems.append(self._credentials_message())
        if not self._tickers and not self.index:
            problems.append(self._no_universe_message())
        if not self.base_url.lower().startswith("https://"):
            problems.append(f"base_url {self.base_url!r} is not HTTPS: credentials and tokens would travel in clear text")
        permits_tr = DocumentKind.TRANSCRIPT in self.boundary.allowed_document_kinds
        if self.kensho is not None and not permits_tr:
            problems.append("a Kensho client was injected but the boundary does not permit TRANSCRIPT text (licence class "
                            "L1 pending gate G2), so transcripts are not fetched. After G2 pass boundary=DataBoundary("
                            "provider='capiq', allowed_document_kinds={DocumentKind.TRANSCRIPT}, note='G2: <reference>').")
        elif permits_tr and self.kensho is None:
            problems.append("the boundary permits transcripts but no Kensho client was injected: pass "
                            "kensho=kensho_client_from_env() (or another client exposing the documented methods).")
        self._roll_day()
        if self.requests_today >= self.warn_fraction * self.daily_request_limit:
            problems.append(f"{self.requests_today} of the observed {self.daily_request_limit} GDS requests/day already "
                            "used by this provider today")
        return problems

    def close(self) -> None:
        """Close the HTTP client (the token is dropped)."""
        if self._http is not None:
            self._http.close()
        self._http, self._token = None, None

    def __enter__(self) -> "CapIQProvider":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ HTTP / auth
    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(transport=self._transport, timeout=self.timeout)
        return self._http

    def _authenticate(self) -> str:
        user, pw = os.environ.get(self.username_env), os.environ.get(self.password_env)
        if not user or not pw:
            raise ProviderUnavailable(self._credentials_message())
        self.auth_calls += 1
        try:
            resp = self._client().post(self._token_url, data={"username": user, "password": pw},
                                       headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"cannot reach the S&P Capital IQ token endpoint {self._token_url}: "
                                      f"{type(e).__name__}: {e}") from e
        if resp.status_code in (400, 401, 403):
            raise ProviderUnavailable(f"S&P Capital IQ authentication failed (HTTP {resp.status_code}): check "
                                      f"${self.username_env} / ${self.password_env} and that the account is entitled to the "
                                      "GDS API.")
        if resp.status_code >= 400:
            raise ProviderError(f"S&P Capital IQ token endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            token = resp.json().get("access_token")
        except (ValueError, AttributeError) as e:
            raise ProviderError("S&P Capital IQ token endpoint returned a non-JSON body") from e
        if not token:
            raise ProviderError("S&P Capital IQ token endpoint returned no access_token")
        self._token = str(token)
        return self._token

    def _post(self, url: str, body: str) -> Any:
        """POST a JSON body with the bearer token; re-authenticate once on HTTP 401."""
        client = self._client()
        resp: httpx.Response | None = None
        for attempt in (0, 1):
            token = self._token or self._authenticate()
            try:
                resp = client.post(url, content=body.encode("utf-8"),
                                   headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                            "Accept": "application/json"})
            except httpx.HTTPError as e:
                raise ProviderUnavailable(f"cannot reach S&P Capital IQ at {url}: {type(e).__name__}: {e}") from e
            if resp.status_code == 401 and attempt == 0:
                self._token = None  # expired or revoked: /tokenRefresh is UNVERIFIED, so log in again
                continue
            break
        assert resp is not None
        if resp.status_code in (401, 403):
            raise ProviderUnavailable(f"S&P Capital IQ refused the request (HTTP {resp.status_code}) after "
                                      "re-authenticating: the account may not be entitled to this data.")
        if resp.status_code >= 400:
            raise ProviderError(f"S&P Capital IQ returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError(f"S&P Capital IQ returned a non-JSON body: {resp.text[:200]}") from e

    def usage(self) -> Any:
        """The account's usage metrics (the field map's ``api.usage_path`` entry; [M] community-documented).

        Not counted toward ``requests_today`` (it reads the counters rather than data)."""
        if not self._usage_url or not self._usage_mnemonic:
            raise ProviderError("the field map has no api.usage_path entry")
        body = json.dumps({"inputRequests": [{"mnemonic": self._usage_mnemonic}]}, separators=(",", ":"))
        self.query_log.append(body)
        return self._post(self._usage_url, body)

    # ------------------------------------------------------------------ GDS execution
    def _limit_reached(self, msg: str) -> ProviderError:
        self._exhausted_day = self._today()
        return ProviderError(f"S&P Capital IQ daily request limit reached ({msg}). Nothing more is requested today; "
                             "restrict GDS to survivors / cross-checks and read the account's limits with usage().")

    def _check_budget(self, n: int) -> None:
        self._roll_day()
        if self._exhausted_day == self._counter_day:
            raise ProviderError("S&P Capital IQ reported its daily request limit earlier today: no further GDS requests "
                                "are sent until tomorrow")
        if self.requests_today + n > self.daily_request_limit:
            raise ProviderError(
                f"S&P Capital IQ request budget: this call needs {n} GDS request(s) but only "
                f"{max(0, self.daily_request_limit - self.requests_today)} of the observed daily limit "
                f"({self.daily_request_limit}) remain for this provider today ({self.requests_today} used). GDS costs one "
                "request per identifier x mnemonic (VENDOR_REFERENCE 3.2): use fewer tickers (survivors only), null out "
                "field-map entries, or set api.daily_request_limit after reading the real limit with usage()."
            )

    def _count(self, n: int) -> None:
        self._roll_day()
        self.requests_today += n
        if (self.requests_today >= self.warn_fraction * self.daily_request_limit
                and self._limit_warned_day != self._counter_day):
            self._limit_warned_day = self._counter_day
            pct = 100.0 * self.requests_today / self.daily_request_limit
            self._warn(f"capiq: {self.requests_today} GDS requests sent today by this provider ({pct:.0f}% of the observed "
                       f"daily limit of {self.daily_request_limit}); other sessions on the same account count too - check "
                       "usage() and restrict requests to survivors.")

    def _call(self, chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        body = json.dumps({"inputRequests": chunk}, separators=(",", ":"))
        self.query_log.append(body)
        self._count(len(chunk))
        self.http_calls += 1
        data = self._post(self._data_url, body)
        resp = data.get("GDSSDKResponse") if isinstance(data, Mapping) else None
        if not isinstance(resp, list):
            raise ProviderError("S&P Capital IQ response has no GDSSDKResponse list")
        if len(resp) == 1 and isinstance(resp[0], Mapping) and set(resp[0]) == {"ErrMsg"}:
            msg = str(resp[0]["ErrMsg"])
            if _LIMIT_RE.search(msg):
                raise self._limit_reached(msg)
            raise ProviderError(f"S&P Capital IQ rejected the request: {msg}")
        if len(resp) != len(chunk):
            raise ProviderError(f"S&P Capital IQ returned {len(resp)} result(s) for {len(chunk)} request(s)")
        sent_ids = [str(r["identifier"]).strip().upper() for r in chunk]
        for i, (req, el) in enumerate(zip(chunk, resp)):
            if not isinstance(el, Mapping):
                raise ProviderError("S&P Capital IQ returned a malformed GDSSDKResponse element")
            mn, ide = el.get("Mnemonic"), el.get("Identifier")
            echoed = str(ide).strip().upper() if ide is not None else sent_ids[i]
            # an echoed identifier may be normalised by the vendor; one that belongs to ANOTHER request is a shuffle
            if (mn is not None and str(mn).strip().upper() != str(req["mnemonic"]).upper()) or (
                    echoed != sent_ids[i] and echoed in sent_ids):
                raise ProviderError(f"S&P Capital IQ response out of order: expected {req['mnemonic']} for "
                                    f"{req['identifier']}, got {mn} for {ide}")
            err = _errmsg(el)
            if err and _LIMIT_RE.search(err):
                raise self._limit_reached(err)
        return resp

    def _execute(self, reqs: list[dict[str, Any]], *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Send ``reqs`` (de-duplicated, cached, batched, budget-checked); results aligned with ``reqs``."""
        keys = [json.dumps(r, sort_keys=True, default=str) for r in reqs]
        results: dict[str, dict[str, Any]] = {}
        todo: dict[str, dict[str, Any]] = {}
        for k, r in zip(keys, reqs):
            if use_cache and k in self._cache:
                results[k] = self._cache[k]
            elif k not in todo:
                todo[k] = r
        if todo:
            self._check_budget(len(todo))
            items = list(todo.items())
            for i in range(0, len(items), self.max_requests_per_call):
                chunk = items[i:i + self.max_requests_per_call]
                for (k, _), el in zip(chunk, self._call([r for _, r in chunk])):
                    self._cache[k] = dict(el)
                    results[k] = self._cache[k]
        return [results[k] for k in keys]

    def clear_cache(self) -> None:
        """Forget cached GDS responses and preflight results (the request counter is kept)."""
        self._cache.clear()
        self._preflight_cache.clear()

    # ------------------------------------------------------------------ legs, gating, conversion
    def _ctx(self, as_of: date, start: date | None = None, end: date | None = None) -> dict[str, Any]:
        fmt = lambda d: d.strftime(self.date_format)  # noqa: E731
        ctx: dict[str, Any] = {"as_of": fmt(as_of), "as_of_3m": fmt(as_of - timedelta(days=91)),
                               "as_of_1y": fmt(as_of - timedelta(days=365))}
        if start is not None:
            ctx["start"] = fmt(start)
        if end is not None:
            ctx["end"] = fmt(end)
        return ctx

    def _make_leg(self, label: str, entry: Mapping[str, Any], ctx: Mapping[str, Any], scale_key: str,
                  *, function: str = "GDSP", probe_ctx: Mapping[str, Any] | None = None) -> _Leg:
        tmpl = dict(entry.get("properties") or {})
        props = {str(k): _render(v, ctx) for k, v in tmpl.items()}
        probe = {str(k): _render(v, probe_ctx) for k, v in tmpl.items()} if probe_ctx is not None else dict(props)
        plaus = entry.get("plausible")
        scale = entry.get(scale_key)
        return _Leg(
            label=label,
            function=str(entry.get("function") or function),
            mnemonic=str(entry["mnemonic"]),
            properties=props,
            probe_properties=probe,
            kind=str(entry.get("kind") or "number"),
            scale=float(scale) if scale is not None else 1.0,
            plausible=(float(plaus[0]), float(plaus[1])) if plaus else None,
            units_verified=entry.get("units_verified") is not False,
            unit=str(entry.get("unit") or ""),
            status=str(entry.get("status") or "unverifiable"),
            as_of_capable=any(isinstance(v, str) and "{as_of" in v for v in tmpl.values()),
        )

    def _is_missing_text(self, v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        s = str(v).strip()
        return not s or s.lower() in self._missing_values

    def _parse_date(self, v: Any) -> pd.Timestamp:
        if self._is_missing_text(v):
            return pd.NaT
        s = str(v).strip()
        try:
            return pd.Timestamp(datetime.strptime(s, self.date_format))
        except ValueError:
            pass
        try:
            ts = pd.Timestamp(s)
        except (TypeError, ValueError):
            return pd.NaT
        if ts is pd.NaT:
            return pd.NaT
        if ts.tzinfo is not None:
            ts = ts.tz_convert(timezone.utc).tz_localize(None)
        return ts.normalize()

    def _convert(self, raw: Any, leg: _Leg, *, check_plausible: bool = True) -> tuple[Any, str | None]:
        """Vendor value -> canonical value (or NaN / NaT) and an issue label (None when clean or plainly missing)."""
        missing: Any = pd.NaT if leg.kind == "date" else math.nan
        if self._is_missing_text(raw):
            return missing, None
        if isinstance(raw, str) and raw.strip().lower() in self._entitlement_values:
            return missing, f"not entitled ({raw.strip()})"
        if leg.kind == "label":
            return str(raw).strip(), None
        if leg.kind == "date":
            d = self._parse_date(raw)
            return (d, None) if d is not pd.NaT else (pd.NaT, f"unparseable date {str(raw)[:20]!r}")
        f = _to_float(raw)
        if math.isnan(f):
            return math.nan, f"non-numeric {str(raw)[:20]!r}"
        c = f * leg.scale
        if check_plausible and leg.plausible and not (leg.plausible[0] <= c <= leg.plausible[1]):
            return math.nan, f"implausible {c:.4g} outside [{leg.plausible[0]:g}, {leg.plausible[1]:g}] (scale or sign?)"
        return c, None

    def _probe_result(self, leg: _Leg, el: Mapping[str, Any]) -> tuple[bool, str, Any]:
        """(ok, note, vendor value) for one probe element (preflight / verify_fields)."""
        err = _errmsg(el)
        if err:
            return False, f"ErrMsg: {err}", None
        rows = _rows(el)
        if leg.function in HISTORY_FUNCTIONS:
            vals = [r[self._value_index] for r in rows if len(r) > self._value_index]
        elif leg.function == "GDSHV":
            vals = [r[0] for r in rows if r]
        else:
            vals = [rows[0][0]] if rows and rows[0] else []
        vals = [v for v in vals if not self._is_missing_text(v)]
        if not vals:
            return False, "empty result (wrong mnemonic, missing entitlement or no data)", None
        value = vals[-1] if leg.function in HISTORY_FUNCTIONS else vals[0]
        if isinstance(value, str) and value.strip().lower() in self._entitlement_values:
            return False, f"not entitled ({value.strip()})", value
        if leg.kind == "number" and all(math.isnan(_to_float(v)) for v in vals):
            return False, f"non-numeric value {str(value)[:30]!r}", value
        return True, f"returned {str(value)[:30]!r}", value

    def _gate(self, legs: Iterable[_Leg], as_of: date) -> dict[str, str]:
        """``leg.key -> reason`` for every leg that must NOT be used (preflights unverifiable legs, cached per day)."""
        reasons: dict[str, str] = {}
        hist = self._is_historical(as_of)
        probe: dict[str, _Leg] = {}
        for leg in legs:
            if leg.key in reasons:
                continue
            if self.strict_units and not leg.units_verified:
                reasons[leg.key] = f"units UNVERIFIED ({leg.unit or '?'}) and strict_units=True"
            elif hist and leg.function in POINT_FUNCTIONS and leg.kind != "label" and not leg.as_of_capable:
                reasons[leg.key] = f"current-only item (no as-of property) for historical as_of {as_of}"
            elif leg.status == "unverifiable" and self.preflight:
                probe.setdefault(leg.key, leg)
        if probe:
            results = self._preflight(list(probe.values()))
            for key, leg in probe.items():
                ok, note = results[leg.probe_key]
                if not ok:
                    reasons[key] = f"preflight on {self.test_identifier} failed ({note})"
        return reasons

    def _preflight(self, legs: list[_Leg]) -> dict[str, tuple[bool, str]]:
        day = self._today()
        out: dict[str, tuple[bool, str]] = {}
        pending: dict[str, _Leg] = {}
        for leg in legs:
            cached = self._preflight_cache.get((day, leg.probe_key))
            if cached is not None:
                out[leg.probe_key] = cached
            else:
                pending.setdefault(leg.probe_key, leg)
        if pending:
            els = self._execute([leg.request(self.test_identifier, probe=True) for leg in pending.values()])
            for (pk, leg), el in zip(pending.items(), els):
                ok, note, _ = self._probe_result(leg, el)
                self._preflight_cache[(day, pk)] = (ok, note)
                out[pk] = (ok, note)
        return out

    def _leg_series(self, leg: _Leg, els: Mapping[str, Mapping[str, Any]], tickers: list[str]) -> pd.Series:
        """Canonical values of one point leg for ``tickers`` (warns about errors / invalid values)."""
        vals: list[Any] = []
        issues: Counter[str] = Counter()
        n_err = 0
        for t in tickers:
            el = els.get(t)
            err = _errmsg(el) if el is not None else "no response"
            if err:
                n_err += 1
                issues[f"ErrMsg {err[:60]!r}"] += 1
                vals.append(pd.NaT if leg.kind == "date" else math.nan)
                continue
            rows = _rows(el)
            v, issue = self._convert(rows[0][0] if rows and rows[0] else None, leg)
            if issue:
                issues[issue] += 1
            vals.append(v)
        if tickers and n_err == len(tickers):
            self._warn(f"LEG NOT EVALUATED: capiq {leg.label} ({leg.mnemonic}) failed for every requested name "
                       f"({next(iter(issues))}); NaN.")
        elif issues:
            bad = sum(issues.values())
            self._warn(f"capiq {leg.label} ({leg.mnemonic}): {bad} of {len(tickers)} value(s) set to NaN - "
                       + "; ".join(f"{k} x{n}" for k, n in issues.most_common(3)))
        return _typed_series(vals, tickers, leg.kind)

    def _fetch(
        self,
        prefix: str,
        entries: Mapping[str, Any],
        cols: Sequence[str],
        tickers: list[str],
        ids: Mapping[str, str],
        as_of: date,
        *,
        scale_key: str,
        fallbacks: Mapping[str, pd.Series] | None = None,
    ) -> tuple[dict[str, pd.Series], list[str]]:
        """Values for ``cols`` of one map section: ``({column: Series}, [columns with no S&P source])``."""
        ctx = self._ctx(as_of)
        direct: dict[str, _Leg] = {}
        derived: dict[str, tuple[Mapping[str, Any], list[tuple[str, Any]]]] = {}
        post: dict[str, Mapping[str, Any]] = {}
        unavailable: list[str] = []
        for col in cols:
            e = entries.get(col)
            path = f"{prefix}.{col}"
            if not isinstance(e, Mapping) or e.get("available") is False or e.get("compute") is not None:
                unavailable.append(col)
            elif e.get("derive") is not None:
                ins: list[tuple[str, Any]] = []
                for i, inp in enumerate(e.get("inputs") or []):
                    if inp.get("column") is not None:
                        ins.append(("column", str(inp["column"])))
                    else:
                        ins.append(("leg", self._make_leg(f"{path}.inputs[{i}]", inp, ctx, "to_canonical")))
                derived[col] = (e, ins)
            elif e.get("mnemonic"):
                direct[col] = self._make_leg(path, e, ctx, scale_key)
            elif e.get("assume") is not None or e.get("from_column") is not None:
                post[col] = e
            else:
                unavailable.append(col)
        legs = list(direct.values()) + [x for _, ins in derived.values() for k, x in ins if k == "leg"]
        reasons = self._gate(legs, as_of)
        uniq: dict[str, _Leg] = {}
        for leg in legs:
            if leg.key not in reasons:
                uniq.setdefault(leg.key, leg)
        reqs: list[dict[str, Any]] = []
        owners: list[tuple[str, str]] = []
        for key, leg in uniq.items():
            for t in tickers:
                reqs.append(leg.request(ids[t]))
                owners.append((key, t))
        els = self._execute(reqs) if reqs else []
        by_key: dict[str, dict[str, Mapping[str, Any]]] = {}
        for (key, t), el in zip(owners, els):
            by_key.setdefault(key, {})[t] = el

        def series(leg: _Leg) -> pd.Series:
            if leg.key in reasons:
                self._warn(f"LEG NOT EVALUATED: capiq {leg.label} ({leg.mnemonic}): {reasons[leg.key]}; NaN.")
                return _typed_series([None] * len(tickers), tickers, leg.kind)
            return self._leg_series(leg, by_key.get(leg.key, {}), tickers)

        out: dict[str, pd.Series] = {}
        unverified: list[_Leg] = []
        for col, leg in direct.items():
            s = series(leg)
            fb = (fallbacks or {}).get(col)
            if fb is not None:
                s = s.where(s.notna(), fb.reindex(s.index))
            vmap = (entries.get(col) or {}).get("value_map")
            if leg.kind == "label" and isinstance(vmap, Mapping):
                s = _map_labels(s, vmap, entries[col].get("unmapped", "pass"))
            out[col] = s
            if leg.key not in reasons and not leg.units_verified:
                unverified.append(leg)
        for col, (e, ins) in derived.items():
            parts: list[pd.Series] = []
            for k, x in ins:
                if k == "column":
                    parts.append(pd.to_numeric(out.get(x, pd.Series(math.nan, index=pd.Index(tickers), dtype="float64")),
                                               errors="coerce").astype("float64"))
                else:
                    parts.append(series(x).astype("float64"))
                    if x.key not in reasons and not x.units_verified:
                        unverified.append(x)
            out[col] = self._derive(f"{prefix}.{col}", e, parts, scale_key)
        for col, e in post.items():
            if e.get("assume") is not None:
                out[col] = _typed_series([e["assume"]] * len(tickers), tickers, "label")
                self._warn(f"capiq {prefix}.{col}: no verified S&P item; every name is labelled {e['assume']!r} "
                           f"({e.get('notes', '')})".rstrip())
            else:
                src = out.get(str(e["from_column"]))
                if src is None:
                    src = _typed_series([None] * len(tickers), tickers, "label")
                out[col] = _map_labels(src, e.get("value_map") or {}, e.get("unmapped", "nan"))
        if unverified:
            desc = ", ".join(f"{leg.label} ({leg.mnemonic} x{leg.scale:g}: {leg.unit or '?'})"
                             for leg in {id(x): x for x in unverified}.values())
            self._warn(f"capiq {prefix}: units UNVERIFIED for {desc}. Values are served converted to canonical units "
                       "(implausible ones blanked); admit them with verify_fields() and the field_validation_log (gate G5) "
                       "and set units_verified=true in an override, or pass strict_units=True to withhold them.")
        return out, unavailable

    def _derive(self, path: str, e: Mapping[str, Any], parts: list[pd.Series], scale_key: str) -> pd.Series:
        op = str(e["derive"])
        a = parts[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            if op == "multiply":
                r = a * parts[1]
            elif op == "divide":
                b = parts[1]
                r = a / b.where(b != 0)
            elif op == "divide_by_one_plus":
                b = 1.0 + parts[1]
                r = a / b.where(b > 0)
            else:  # reciprocal
                r = 1.0 / a.where(a != 0)
        scale = e.get(scale_key)
        r = (r * float(scale) if scale is not None else r).astype("float64")
        r = r.where(np.isfinite(r))
        plaus = e.get("plausible")
        if plaus:
            bad = r.notna() & ~r.between(float(plaus[0]), float(plaus[1]))
            if bad.any():
                self._warn(f"capiq {path}: {int(bad.sum())} implausible derived value(s) set to NaN")
                r = r.mask(bad)
        return r

    # ------------------------------------------------------------------ universe
    def _members(self, as_of: date) -> list[str]:
        if self._tickers:
            return list(self._tickers)
        if self.index:
            return self._constituents(as_of)
        raise ProviderError(self._no_universe_message())

    def _constituents(self, as_of: date) -> list[str]:
        e = (self.fieldmap.get("universe") or {}).get("constituents")
        if not isinstance(e, Mapping) or not e.get("mnemonic"):
            raise ProviderError("the S&P field map has no universe.constituents entry: pass tickers=[...]")
        page = max(1, int(e.get("page_size", 600)))
        pages = max(1, int(e.get("max_pages", 10)))
        vi = int(e.get("value_index", 0))
        found: list[str] = []
        truncated = True
        for p in range(pages):
            ctx = {**self._ctx(as_of), "start_rank": p * page + 1, "end_rank": (p + 1) * page}
            leg = self._make_leg("universe.constituents", e, ctx, "to_canonical", function="GDSHV")
            el = self._execute([leg.request(self.index or "")])[0]
            err = _errmsg(el)
            if err:
                if p == 0:
                    raise ProviderError(f"index constituents ({leg.function} {leg.mnemonic}) failed for {self.index}: {err}. "
                                        "Index coverage is untested (VENDOR_REFERENCE 3.3): check the entitlement or pass "
                                        "tickers=[...].")
                truncated = False
                break
            vals = [str(r[vi]).strip() for r in _rows(el) if len(r) > vi and not self._is_missing_text(r[vi])]
            found += vals
            if len(vals) < page:
                truncated = False
                break
        if truncated:
            self._warn(f"capiq: the constituent list of {self.index} may be truncated at {pages * page} names; raise "
                       "universe.constituents.max_pages in an override")
        tickers: list[str] = []
        unresolved: list[str] = []
        for v in _dedupe(found):
            t = self._tickers_by_id.get(v) or self._tickers_by_id.get(v.upper())
            if t is None:
                t = identifier_to_ticker(v, self._class_sep)
                if t is None:
                    unresolved.append(v)
                    continue
                if _TICKER_EXCH_RE.match(v):
                    self._register(t, v)
            if t not in tickers:
                tickers.append(t)
        if unresolved:
            self._warn(f"capiq: {len(unresolved)} constituent(s) of {self.index} have no ticker and were dropped and "
                       f"flagged ({', '.join(unresolved[:10])}): supply them through identifiers={{ticker: '<CapIQ id>'}} "
                       "(the maintained symbology table, ADR graft 6).")
        if not tickers:
            self._warn(f"capiq: index {self.index} returned no usable constituents")
        if self._is_historical(as_of):
            self._warn(f"capiq: constituents of {self.index} are today's members, not those of {as_of} "
                       "(survivorship bias in a historical run)")
        return tickers

    def get_universe(self, spec: "UniverseSpec | None", as_of: date) -> pd.DataFrame:
        """Explicit tickers or index constituents, indexed by canonical ticker, columns ``fields.UNIVERSE_COLUMNS``.

        Raises ``ProviderError`` when neither ``tickers`` nor ``index`` is configured (S&P GDS cannot screen).
        Known country mismatches and excluded sectors are dropped; the local engine re-applies every filter.
        """
        a = _as_date(as_of)
        tickers = self._members(a)
        self._note_historical(a)
        if self._tickers and self._is_historical(a):
            self._warn(f"capiq: the explicit ticker list is used as given for historical as_of {a} (no point-in-time "
                       "membership check)")
        ids = {t: self._identifier(t) for t in tickers}
        entries = (self.fieldmap.get("raw") or {}).get("universe") or {}
        # exchange part of a 'TICKER:EXCH' identifier fills a missing exchange (the value_map applies to both)
        fallback = _typed_series([_exchange_part(ids[t]) for t in tickers], tickers, "label")
        data, _ = self._fetch("raw.universe", entries, DATASET_COLUMNS["universe"], tickers, ids, a,
                              scale_key="to_canonical", fallbacks={F.EXCHANGE: fallback})
        data[F.VENDOR_ID] = _typed_series([ids[t] for t in tickers], tickers, "label")
        df = self._frame(F.UNIVERSE_COLUMNS, tickers, data, a)
        return self._apply_universe_spec(df, spec)

    @staticmethod
    def _apply_universe_spec(df: pd.DataFrame, spec: Any) -> pd.DataFrame:
        if spec is None or df.empty:
            return df
        keep = pd.Series(True, index=df.index)
        country = getattr(spec, "country", None)
        if country:
            c = df[F.COUNTRY]
            keep &= c.isna() | (c.astype(object).map(lambda x: str(x).strip().upper()) == str(country).strip().upper())
        excl = {str(x).strip().lower() for x in (getattr(spec, "exclude_sectors", None) or [])}
        if excl:
            keep &= ~df[F.GICS_SECTOR].map(lambda x: str(x).strip().lower() if isinstance(x, str) else "").isin(excl)
        return df.loc[keep]

    # ------------------------------------------------------------------ snapshots
    def _frame(self, columns: Sequence[str], tickers: list[str], data: Mapping[str, pd.Series], as_of: date) -> pd.DataFrame:
        idx = pd.Index(list(tickers), name=F.TICKER, dtype=object)
        cols: dict[str, pd.Series] = {}
        for c in columns:
            kind = "date" if c in DATE_COLUMNS else ("label" if c in LABEL_COLUMNS else "number")
            s = data.get(c)
            if s is None:
                s = _typed_series([None] * len(tickers), tickers, kind)
            else:
                s = _coerce(pd.Series(list(s), index=list(s.index), dtype=object).reindex(list(tickers)), kind)
            if kind == "date" and c not in FUTURE_DATE_COLUMNS:
                late = s.notna() & (s > pd.Timestamp(as_of))
                if late.any():
                    self._warn(f"capiq: {int(late.sum())} {c} value(s) after as_of {as_of} blanked (look-ahead guard)")
                    s = s.mask(late)
            cols[c] = pd.Series(s.to_numpy(), index=idx, dtype=s.dtype)
        return pd.DataFrame(cols, index=idx)

    def _dataset(self, dataset: str, tickers: Iterable[str], as_of: date) -> pd.DataFrame:
        a = _as_date(as_of)
        tick = _dedupe(tickers)
        cols = DATASET_COLUMNS[dataset]
        entries = (self.fieldmap.get("raw") or {}).get(dataset) or {}
        if not any(self._fetchable(entries.get(c)) for c in cols):
            if any(self._fetchable(entries.get(c), ignore_units=True) for c in cols):
                self._warn(f"LEG NOT EVALUATED: capiq {dataset}: every mapped item has UNVERIFIED units and "
                           "strict_units=True - NaN frame.")
            else:
                first = next((e for e in entries.values() if isinstance(e, Mapping) and e.get("notes")), {})
                self._warn(f"capiq: {dataset} not available from S&P - NaN for {', '.join(cols)}. "
                           f"{first.get('notes', '')}".rstrip())
            return self._frame(cols, tick, {}, a)
        if not tick:
            return self._frame(cols, tick, {}, a)
        self._note_historical(a)
        ids = {t: self._identifier(t) for t in tick}
        data, unavailable = self._fetch(f"raw.{dataset}", entries, cols, tick, ids, a, scale_key="to_canonical")
        if unavailable:
            self._warn(f"capiq {dataset}: no S&P mapping (NaN) for {', '.join(unavailable)}")
        return self._frame(cols, tick, data, a)

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.FUNDAMENTAL_COLUMNS`` in USD absolute (CapIQ millions x1e6), anchored with asOfDate."""
        return self._dataset("fundamentals", tickers, as_of)

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.ESTIMATE_COLUMNS`` (consensus estimates; 3-month-ago values via asOfDate = as_of - 91 days)."""
        return self._dataset("estimates", tickers, as_of)

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.SHORT_INTEREST_COLUMNS``: no verified S&P field ('do not ship') -> NaN frame + warning."""
        return self._dataset("short_interest", tickers, as_of)

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.OPTIONS_COLUMNS``: implied vol and put/call are not available from S&P -> NaN frame + warning."""
        return self._dataset("options", tickers, as_of)

    def get_vendor_features(self, tickers: list[str], as_of: date, features: Iterable[str] | None = None) -> pd.DataFrame:
        """Vendor-computed values of catalog features in catalog units (``features`` section), for reconciliation.

        These may raise reconciliation flags against the locally computed features; they never replace them.
        Features without an S&P item (computed locally, or not available from S&P) are NaN with a warning.
        """
        a = _as_date(as_of)
        tick = _dedupe(tickers)
        entries = self.fieldmap.get("features") or {}
        catalog = default_catalog()
        names = list(features) if features is not None else [n for n, e in entries.items() if self._fetchable(e)]
        unknown = [n for n in names if n not in catalog]
        if unknown:
            raise ValueError(f"not catalog features: {unknown}")
        ids = {t: self._identifier(t) for t in tick}
        data, unavailable = (self._fetch("features", entries, names, tick, ids, a, scale_key="to_catalog")
                             if tick else ({}, []))
        if unavailable:
            self._warn(f"capiq features: no S&P item (computed locally or not available from S&P) for "
                       f"{', '.join(unavailable)}; NaN")
        idx = pd.Index(tick, name=F.TICKER, dtype=object)
        out = pd.DataFrame(index=idx)
        for n in names:
            kind = "label" if catalog[n].dtype == "category" else "number"
            s = data.get(n)
            s = _typed_series([None] * len(tick), tick, kind) if s is None else _coerce(s.reindex(tick), kind)
            out[n] = pd.Series(s.to_numpy(), index=idx, dtype=s.dtype)
        return out

    # ------------------------------------------------------------------ prices
    def _history_legs(self, as_of: date, start: date) -> dict[str, _Leg]:
        hist = self.fieldmap.get("history") or {}
        ctx = self._ctx(as_of, start=start, end=as_of)
        probe_ctx = self._ctx(as_of, start=as_of - timedelta(days=PROBE_DAYS), end=as_of)
        legs: dict[str, _Leg] = {}
        for f in F.PRICE_FIELDS:
            entry = (hist.get("fields") or {}).get(f)
            if not isinstance(entry, Mapping) or not entry.get("mnemonic") or entry.get("available") is False:
                continue
            merged = {**entry, "properties": {**(hist.get("properties") or {}), **(entry.get("properties") or {})}}
            legs[f] = self._make_leg(f"history.{f}", merged, ctx, "to_canonical",
                                     function=str(hist.get("function") or "GDSHE"), probe_ctx=probe_ctx)
        return legs

    def _history_series(self, leg: _Leg, el: Mapping[str, Any] | None, start: date, end: date,
                        issues: Counter[str]) -> pd.Series:
        err = _errmsg(el) if el is not None else "no response"
        if err:
            issues[f"ErrMsg {err[:60]!r}"] += 1
            return pd.Series(dtype="float64")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        pts: dict[pd.Timestamp, float] = {}
        need = max(self._value_index, self._date_index)
        for row in _rows(el):
            if len(row) <= need:
                continue
            d = self._parse_date(row[self._date_index])
            if d is pd.NaT:
                issues["unparseable date"] += 1
                continue
            if d < lo or d > hi:  # outside the window (never after as_of)
                continue
            v, issue = self._convert(row[self._value_index], leg, check_plausible=False)
            if issue:
                issues[issue] += 1
            pts[d] = v
        s = pd.Series(pts, dtype="float64").sort_index()
        if leg.plausible and s.notna().any():
            med = float(s.median())
            if not (leg.plausible[0] <= med <= leg.plausible[1]):
                issues[f"implausible median {med:.4g} outside [{leg.plausible[0]:g}, {leg.plausible[1]:g}] (scale?)"] += 1
                s = s * math.nan
        return s

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        """Daily OHLCV via GDSHE (``startDate`` / ``endDate``); adjusted close and volume x to_canonical.

        Unverifiable OHLC mnemonics are preflighted (NaN frames + warning when they fail). Raises
        ``ProviderError`` when no close comes back for any ticker.
        """
        s, e = _as_date(start), _as_date(end)
        if s > e:
            raise ValueError(f"start {s} is after end {e}")
        tick = _dedupe(tickers)
        legs = self._history_legs(e, s)
        if F.CLOSE not in legs:
            raise ProviderError("the S&P field map has no usable history.fields.close entry")
        if not tick:
            empty = pd.DataFrame(index=pd.DatetimeIndex([]), columns=[], dtype="float64")
            return PricePanel(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
        reasons = self._gate(legs.values(), e)
        if legs[F.CLOSE].key in reasons:
            raise ProviderError(f"capiq close history unavailable: {reasons[legs[F.CLOSE].key]}")
        for f, leg in legs.items():
            if leg.key in reasons:
                self._warn(f"LEG NOT EVALUATED: capiq {leg.label} ({leg.mnemonic}): {reasons[leg.key]}; NaN {f} prices.")
        usable = {f: leg for f, leg in legs.items() if leg.key not in reasons}
        reqs: list[dict[str, Any]] = []
        owners: list[tuple[str, str]] = []
        for f, leg in usable.items():
            for t in tick:
                reqs.append(leg.request(self._identifier(t)))
                owners.append((f, t))
        els = self._execute(reqs)
        series: dict[str, dict[str, pd.Series]] = {f: {} for f in F.PRICE_FIELDS}
        issues: dict[str, Counter[str]] = {f: Counter() for f in usable}
        for (f, t), el in zip(owners, els):
            series[f][t] = self._history_series(usable[f], el, s, e, issues[f])
        for f, c in issues.items():
            if c:
                self._warn(f"capiq history.{f} ({usable[f].mnemonic}): " + "; ".join(f"{k} x{n}" for k, n in c.most_common(3)))
        dates = sorted({d for f in series for ser in series[f].values() for d in ser.index})
        idx = pd.DatetimeIndex(dates, name="date")
        frames: dict[str, pd.DataFrame] = {}
        for f in F.PRICE_FIELDS:
            frames[f] = pd.DataFrame({t: series[f].get(t, pd.Series(dtype="float64")).reindex(idx) for t in tick},
                                     index=idx, columns=tick, dtype="float64")
        close = frames[F.CLOSE]
        if close.notna().sum().sum() == 0:
            raise ProviderError(f"S&P Capital IQ returned no {legs[F.CLOSE].mnemonic} history for any of {len(tick)} "
                                f"ticker(s) between {s} and {e}")
        missing = [t for t in tick if not close[t].notna().any()]
        if missing:
            self._warn(f"capiq: no price history for {len(missing)} ticker(s) ({', '.join(missing[:20])}); NaN")
        return PricePanel(frames[F.OPEN], frames[F.HIGH], frames[F.LOW], close, frames[F.VOLUME])

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        """Close of the benchmark: the field map's ``benchmark`` (``^SPX``, unverifiable) with its fallback, or ``symbol``."""
        s, e = _as_date(start), _as_date(end)
        hist = self.fieldmap.get("history") or {}
        close_entry = (hist.get("fields") or {}).get(F.CLOSE) or {}
        if symbol:
            cands = [(self._identifier(symbol), str(close_entry.get("mnemonic")), str(close_entry.get("status", "")))]
        else:
            bm = self.fieldmap.get("benchmark") or {}
            cands = [(str(bm["identifier"]), str(bm["mnemonic"]), str(bm.get("status", "")))]
            if bm.get("fallback_identifier") and bm.get("fallback_mnemonic"):
                cands.append((str(bm["fallback_identifier"]), str(bm["fallback_mnemonic"]), str(bm.get("fallback_status", ""))))
        ctx = self._ctx(e, start=s, end=e)
        failures: list[str] = []
        for i, (ident, mnem, status) in enumerate(cands):
            entry = {"mnemonic": mnem, "properties": hist.get("properties") or {}, "status": status, "to_canonical": 1}
            leg = self._make_leg("benchmark", entry, ctx, "to_canonical", function=str(hist.get("function") or "GDSHE"))
            el = self._execute([leg.request(ident)])[0]
            ser = self._history_series(leg, el, s, e, Counter()).dropna()
            if len(ser):
                if i > 0:
                    self._warn(f"capiq benchmark: {cands[0][0]} {cands[0][1]} returned no data ({failures[0]}); using the "
                               f"fallback {ident} {mnem} ({status})")
                ser.name = symbol or ident
                ser.index.name = "date"
                return ser
            failures.append(_errmsg(el) or "no data")
        raise ProviderError("S&P Capital IQ benchmark history unavailable: "
                            + "; ".join(f"{c[0]} {c[1]}: {f}" for c, f in zip(cands, failures)))

    # ------------------------------------------------------------------ documents (Kensho)
    def get_documents(
        self,
        ticker: str,
        kinds: set[DocumentKind] | None,
        start: date,
        end: date,
        limit: int = 10,
    ) -> list[Document]:
        """Earnings-call transcripts from the Kensho LLM-ready API, newest first; nothing else.

        Returns ``[]`` (with a warning, without calling Kensho) unless a Kensho client was injected AND the
        boundary permits TRANSCRIPT text (gate G2). News / filings / research are not served by this adapter.
        """
        wanted = set(DocumentKind) if kinds is None else {DocumentKind(k) for k in kinds}
        docs_cfg = self.fieldmap.get("documents") or {}
        for k in sorted(wanted - {DocumentKind.TRANSCRIPT}, key=lambda x: x.value):
            note = (docs_cfg.get(k.value) or {}).get("notes", "no verified S&P API")
            self._warn(f"capiq: {k.value} documents are not served by this adapter: {note}")
        if DocumentKind.TRANSCRIPT not in wanted:
            return []
        if DocumentKind.TRANSCRIPT not in self.boundary.allowed_document_kinds:
            self._warn("capiq: Kensho transcripts are licence class L1 pending gate G2: the boundary does not permit "
                       "TRANSCRIPT text, so none are fetched. " + BOUNDARY_NOTE)
            return []
        if self.kensho is None:
            self._warn("capiq: no Kensho client injected (CapIQProvider(kensho=...)): no transcripts are returned")
            return []
        s, e = _as_date(start), _as_date(end)
        doc = self._kensho_transcript(str(ticker).strip(), s, e)
        docs = [doc] if doc is not None else []
        docs.sort(key=lambda d: d.published_at, reverse=True)
        return docs[: max(0, int(limit))]

    def _kensho_transcript(self, ticker: str, start: date, end: date) -> Document | None:
        cfg = (self.fieldmap.get("documents") or {}).get("transcripts") or {}
        k = self.kensho
        f_latest = getattr(k, str(cfg.get("latest_earnings_method") or ""), None) if cfg.get("latest_earnings_method") else None
        f_tr = getattr(k, str(cfg.get("transcript_method") or ""), None) if cfg.get("transcript_method") else None
        try:
            if callable(f_latest) and callable(f_tr):
                self.query_log.append(f"kensho:{cfg['latest_earnings_method']}(identifiers={[ticker]!r})")
                rec = _find_earnings(_call_kw(f_latest, identifiers=[ticker]), ticker)
                if rec is None:
                    self._warn(f"capiq: Kensho returned no latest earnings call for {ticker}")
                    return None
                key = _attr(rec, "key_dev_id", "keyDevId")
                name, when = _attr(rec, "name", "event_name", "title"), _naive_utc(_attr(rec, "datetime", "date", "event_datetime"))
                if not self._in_window(ticker, key, when, start, end):
                    return None
                self.query_log.append(f"kensho:{cfg['transcript_method']}(key_dev_id={key!r})")
                segs = _segments_from_payload(_call_kw(f_tr, key_dev_id=key))
            elif callable(getattr(k, "ticker", None)):
                self.query_log.append(f"kensho:ticker({ticker!r}).company.latest_earnings")
                earnings = k.ticker(ticker).company.latest_earnings
                if earnings is None:
                    self._warn(f"capiq: Kensho returned no latest earnings call for {ticker}")
                    return None
                key = _attr(earnings, "key_dev_id", "keyDevId")
                name, when = _attr(earnings, "name", "title"), _naive_utc(_attr(earnings, "datetime", "date"))
                if not self._in_window(ticker, key, when, start, end):
                    return None
                self.query_log.append(f"kensho:latest_earnings.transcript.raw (key_dev_id={key!r})")
                segs = _segments_from_payload(_attr(earnings, "transcript"))
            else:
                self._warn("capiq: the injected Kensho client exposes neither the documented tool methods "
                           f"({cfg.get('latest_earnings_method')} / {cfg.get('transcript_method')}) nor ticker(); "
                           "no transcripts")
                return None
        except ProviderError:
            raise
        except Exception as ex:  # noqa: BLE001 - permission errors etc. degrade to "no transcript"
            self._warn(f"capiq: Kensho transcript request for {ticker} failed ({type(ex).__name__}: {ex}); none returned "
                       "(TranscriptsPermission needed)")
            return None
        if not segs:
            self._warn(f"capiq: Kensho transcript {key} for {ticker} is empty")
            return None
        return self._transcript_document(ticker, key, name, when, segs, cfg)

    def _in_window(self, ticker: str, key: Any, when: datetime | None, start: date, end: date) -> bool:
        if key is None:
            self._warn(f"capiq: Kensho latest earnings for {ticker} has no key_dev_id; skipped")
            return False
        if when is None:
            self._warn(f"capiq: Kensho call {key} for {ticker} has no date; skipped (the call date must fall inside "
                       "the requested window)")
            return False
        if not (start <= when.date() <= end):
            self._warn(f"capiq: latest Kensho earnings call for {ticker} ({when.date()}, key_dev_id {key}) is outside "
                       f"[{start}, {end}]; not returned")
            return False
        return True

    def _transcript_document(self, ticker: str, key: Any, name: Any, when: datetime, segs: list[TranscriptSegment],
                             cfg: Mapping[str, Any]) -> Document:
        paragraphs = [(f"{s.speaker}: {s.text}" if s.speaker else s.text) for s in segs]
        text = "\n\n".join(paragraphs)
        return Document(
            doc_id=f"kensho-transcript-{key}",
            ticker=ticker,
            kind=DocumentKind.TRANSCRIPT,
            title=str(name) if name else f"{ticker} earnings call",
            published_at=when,
            source="S&P Global Kensho LLM-ready API",
            url=None,
            text=text,
            segments=segs,
            metadata={
                "vendor": VENDOR,
                "channel": str(cfg.get("channel") or "kensho"),
                "key_dev_id": str(key),
                "licence_class": str(cfg.get("licence_class") or "L1"),
                "gate": str(cfg.get("gate") or "G2"),
                "boundary_note": (self.boundary.note or "")[:500],
                "text_sha256": _sha256(text),
                "paragraph_sha256": ",".join(_sha256(p) for p in paragraphs),
                "section_detection": "heuristic (Q&A header or Operator opening questions)",
            },
        )

    # ------------------------------------------------------------------ field self-check (G5)
    def _check_legs(self, as_of: date) -> list[tuple[_Leg, str | None]]:
        """Every mapped mnemonic as a leg, with a fixed identifier where the item is not per-security."""
        ctx = self._ctx(as_of)
        out: list[tuple[_Leg, str | None]] = []

        def add(path: str, e: Any, scale_key: str) -> None:
            if not isinstance(e, Mapping) or e.get("available") is False:
                return
            if e.get("mnemonic"):
                out.append((self._make_leg(path, e, ctx, scale_key), None))
            for i, inp in enumerate(e.get("inputs") or []):
                if isinstance(inp, Mapping) and inp.get("mnemonic"):
                    out.append((self._make_leg(f"{path}.inputs[{i}]", inp, ctx, "to_canonical"), None))

        for ds, entries in (self.fieldmap.get("raw") or {}).items():
            for col, e in (entries or {}).items():
                add(f"raw.{ds}.{col}", e, "to_canonical")
        probe_start = as_of - timedelta(days=PROBE_DAYS)
        for leg in self._history_legs(as_of, probe_start).values():
            out.append((leg, None))
        for name, e in (self.fieldmap.get("features") or {}).items():
            add(f"features.{name}", e, "to_catalog")
        hist = self.fieldmap.get("history") or {}
        bm = self.fieldmap.get("benchmark") or {}
        hctx = self._ctx(as_of, start=probe_start, end=as_of)
        for label, ik, mk, sk in (("benchmark", "identifier", "mnemonic", "status"),
                                  ("benchmark.fallback", "fallback_identifier", "fallback_mnemonic", "fallback_status")):
            if bm.get(ik) and bm.get(mk):
                entry = {"mnemonic": bm[mk], "properties": hist.get("properties") or {}, "status": bm.get(sk, ""), "to_canonical": 1}
                out.append((self._make_leg(label, entry, hctx, "to_canonical", function=str(hist.get("function") or "GDSHE")),
                            str(bm[ik])))
        cons = (self.fieldmap.get("universe") or {}).get("constituents")
        if self.index and isinstance(cons, Mapping) and cons.get("mnemonic"):
            cctx = {**ctx, "start_rank": 1, "end_rank": 5}
            out.append((self._make_leg("universe.constituents", {**cons, "kind": "label"}, cctx, "to_canonical",
                                       function="GDSHV"), self.index))
        return out

    def verify_fields(self, sample_ticker: str | None = None, as_of: date | None = None) -> list[FieldCheck]:
        """Request every mapped item alone on a known security (field-admission procedure, gate G5).

        Covers raw columns (and derived inputs), history fields (last ``PROBE_DAYS`` days), feature items, the
        benchmark and, when ``index`` is set, the constituents call. Fresh requests (no cache); they count
        toward the daily budget. A check fails on an ``ErrMsg``, an empty result, a non-numeric value for a
        numeric item or a canonical value outside ``plausible``. Record passing rows with
        ``FieldCheck.to_log_row()`` before admitting an item (and setting ``units_verified``).
        """
        a = _as_date(as_of) if as_of is not None else self._today()
        ident = self._identifier(sample_ticker) if sample_ticker else self.test_identifier
        legs = self._check_legs(a)
        reqs = [leg.request(fixed or ident) for leg, fixed in legs]
        els = self._execute(reqs, use_cache=False)
        return [self._field_check(leg, el, fixed or ident, a) for (leg, fixed), el in zip(legs, els)]

    def _field_check(self, leg: _Leg, el: Mapping[str, Any], ident: str, as_of: date) -> FieldCheck:
        guidance = {
            "confirmed": "confirmed",
            "corrected": "corrected: use exactly this form",
            "unverifiable": "UNVERIFIABLE: configurable only, preflighted before every run, never a threshold until admitted (G5)",
        }.get(leg.status, leg.status)
        notes = [guidance]
        base = {"mnemonic": leg.mnemonic, "function": leg.function, "identifier": ident,
                "properties": dict(leg.properties), "as_of": as_of}
        ok, note, value = self._probe_result(leg, el)
        if leg.function in HISTORY_FUNCTIONS:
            rows = _rows(el)
            if rows:
                notes.append(f"{len(rows)} row(s), first row {rows[0][:3]!r}")
        elif leg.function == "GDSHV":
            rows = _rows(el)
            notes.append(f"{len(rows)} member row(s), first {[r[0] for r in rows[:3] if r]!r}")
        if not ok:
            return FieldCheck(leg.label, leg.status, value, False, "; ".join(notes + [note]), **base)
        canonical: Any = value
        if leg.kind == "number":
            c, issue = self._convert(value, leg)
            canonical = None if (isinstance(c, float) and math.isnan(c)) else c
            notes.append(f"unit {leg.unit or '?'}; x{leg.scale:g} -> {c:.6g}" if not issue else f"unit {leg.unit or '?'}; {issue}")
            if issue:
                ok = False
        elif leg.kind == "date":
            canonical = self._parse_date(value)
        if not leg.units_verified:
            notes.append("units UNVERIFIED: compare with a filing or a second source, then set units_verified=true in an "
                         "override")
        return FieldCheck(leg.label, leg.status, value, ok, "; ".join(notes), canonical_value=canonical, **base)


# =============================================================================================
# Frame helpers
# =============================================================================================


def _coerce(s: pd.Series, kind: str) -> pd.Series:
    if kind == "number":
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if kind == "date":
        return pd.to_datetime(s, errors="coerce")
    return s.astype(object).where(s.notna(), np.nan)


def _typed_series(values: Sequence[Any], tickers: Sequence[str], kind: str) -> pd.Series:
    idx = pd.Index(list(tickers), dtype=object)
    if kind == "number":
        return pd.Series([_to_float(v) for v in values], index=idx, dtype="float64")
    if kind == "date":
        return pd.to_datetime(pd.Series(list(values), index=idx, dtype=object), errors="coerce")
    vals = [v if (isinstance(v, str) and v.strip()) else np.nan for v in values]
    return pd.Series(vals, index=idx, dtype=object)


def _map_labels(s: pd.Series, value_map: Mapping[str, Any], unmapped: str) -> pd.Series:
    """Map labels case-insensitively; unmapped labels pass through (``'pass'``) or become NaN (``'nan'``)."""
    lookup = {str(k).strip().lower(): v for k, v in value_map.items()}

    def one(x: Any) -> Any:
        if not isinstance(x, str) or not x.strip():
            return np.nan
        hit = lookup.get(x.strip().lower())
        if hit is not None:
            return hit
        return x if unmapped == "pass" else np.nan

    return pd.Series([one(x) for x in s], index=s.index, dtype=object)
