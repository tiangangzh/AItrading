"""Bloomberg adapter (zone T): BQL inside BQuant, or the Desktop / Server API (``blpapi``).

Where this runs (ADR-001, Phase 1)
----------------------------------
Bloomberg values are computed and kept where they are licensed to live, i.e. **zone T** (the
entitled user's BQuant / Terminal workstation). Two backends:

* ``backend='bql'`` (default) - the ``bql`` package that exists only inside BQuant (BQNT<GO> Desktop
  or BQuant Enterprise): ``bq = bql.Service()``; requests are BQL *strings* executed with
  ``bq.execute(query)`` and read back with ``bql.combined_df(response)`` (VENDOR_REFERENCE 1.1 shows the
  ``bql.Request`` object form; executing the equivalent string is what lets every query be logged
  verbatim - ``verify_fields()`` exercises exactly this path, so run it once in BQuant to confirm).
  Universe (``filter(equitiesuniv(['ACTIVE','PRIMARY']), cntry_of_risk()=='US')``), price history
  (``ca_adj='full'``), fundamentals, estimates and options come from BQL; the screen push-down is
  compiled by :mod:`aitrading.screen.compile_bql` into ONE BQL request. Items that exist only as
  BDP mnemonics (short interest, earnings dates, BEst EPS) are read through the Desktop API on the
  same workstation when ``blpapi`` is available; otherwise those legs are NOT EVALUATED (NaN plus a
  warning), never zero.
* ``backend='blpapi'`` - the Desktop API (``//blp/refdata`` on ``localhost:8194`` by default):
  ``ReferenceDataRequest`` (BDP) and ``HistoricalDataRequest`` (BDH). It cannot screen: the universe
  is an explicit ``tickers=[...]`` list or a saved EQS screen (``universe_expr`` -> ``BeqsRequest``).
  ``//blp/bqlsvc`` is prohibited (ADR section 7) and never used.

Field codes are configuration
-----------------------------
Every item, mnemonic, unit and scale comes from ``aitrading/data/fieldmaps/bloomberg.json``
(override with ``$AITRADING_FIELDMAP_BLOOMBERG`` or the ``fieldmap=`` argument; see
:mod:`aitrading.data.fieldmaps`). ``raw.<dataset>.<column>`` maps canonical columns
(:mod:`aitrading.core.fields`) per channel (``bql`` / ``bdp`` / ``bdh`` / ``derived``) with
``to_canonical`` (canonical = vendor x to_canonical), ``status`` and ``notes``. Rules applied here:

* ``unverifiable`` entries are preflighted alone on ``test_security`` (VENDOR_REFERENCE section 5)
  before they enter a bulk request; a failure suspends that leg (``LEG NOT EVALUATED`` warning).
* ``units_verified: false`` entries are not served (NaN + warning) until admitted.
* BDP returns current values only, so for an ``as_of`` older than ``snapshot_staleness_days`` BDP
  columns are left blank rather than leaking today's values; BQL items carry ``dates='<as_of>'``.
* Look-ahead guards: a short-interest settlement after ``as_of`` blanks the row; any other
  historical date after ``as_of`` is dropped.
* Field exceptions that mean "invalid / unauthorised field" abort the whole leg (ADR risk table);
  per-security exceptions blank that cell. A column that comes back missing for every requested
  security is reported as ``LEG NOT EVALUATED``. Missing values are NaN, never 0.
* :meth:`BloombergProvider.verify_fields` runs every mapped item alone on a known security and
  returns one :class:`FieldCheck` per item: the field-admission procedure (gate G5).

Licensing boundary
------------------
The default boundary is ``deny_all_text(provider='bloomberg', note=BOUNDARY_NOTE)`` with
``allow_numeric_features=False``: Terminal / Desktop API / BQuant Desktop data is licence class L4,
so nothing derived from it - text, values, even tickers or ranks - may reach an external model until
gate G1 (written Bloomberg approval) clears, and even then only ids, ranks and booleans. EDF Textual
News (L5) and broker research (L6) never do. ``get_documents`` therefore returns ``[]`` with a
warning; if a wider boundary is passed it raises ``NotImplementedError`` because no Bloomberg text
API is verified (transcripts: no BQL/BDP item; use SEC EDGAR for filings, Kensho after G2).

Tickers are returned in canonical form (``'AAPL'`` from ``'AAPL US Equity'``, ``'BRK-B'`` from
``'BRK/B US Equity'``); the Bloomberg id is kept in the ``vendor_id`` column and in an internal map
used for later requests. ``query_log`` records every vendor request verbatim (BQL strings and
blpapi request payloads) for the audit trail; ``warnings`` collects every degradation.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from aitrading.core import fields as F
from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary, deny_all_text
from aitrading.data.base import Capability, PricePanel, ProviderError, ProviderUnavailable, PushdownResult
from aitrading.data.fieldmaps import PUSHABLE_STATUSES, FieldMapError, digest, env_var, load_fieldmap
from aitrading.screen.compile_bql import (
    CompileError,
    CompiledQuery,
    bql_quote,
    compile_screen,
    expand_refs,
    render_template,
)

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.spec import ScreenSpec, UniverseSpec

__all__ = [
    "BloombergProvider",
    "FieldCheck",
    "BOUNDARY_NOTE",
    "BQL_MISSING",
    "BLPAPI_MISSING",
    "split_vendor_id",
    "to_canonical_ticker",
    "to_vendor_id",
]

VENDOR = "bloomberg"

BOUNDARY_NOTE = (
    "Bloomberg Terminal / Desktop API / BQuant Desktop data is licence class L4 (ADR-001 section 7): it is computed "
    "and kept in the vendor environment (zone T). Nothing derived from it - text, values, tickers or ranks - may be "
    "sent to an external model until gate G1 (written Bloomberg approval for survivor ids, ranks and booleans to leave "
    "zone T and be stored firm-side) clears, and even then only ids, ranks and booleans. EDF Textual News is L5 and "
    "broker research L6: never. Widen this boundary only with the written approval recorded in the policy table."
)
BQL_MISSING = (
    "The 'bql' package is not available. It ships only inside Bloomberg BQuant (BQNT<GO> on an entitled Terminal, or "
    "BQuant Enterprise) and cannot be installed from PyPI: run BloombergProvider(backend='bql') in a BQuant notebook, "
    "or use BloombergProvider(backend='blpapi') against the Desktop API on a Terminal workstation."
)
BLPAPI_MISSING = (
    "The Bloomberg API SDK 'blpapi' is not installed. Install it from Bloomberg's own index (it is not on pypi.org): "
    "python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi "
    "(3.26.9.1 needs Python >= 3.10). It also needs a running, logged-in Bloomberg Terminal (Desktop API on "
    "localhost:8194) or a B-PIPE / Server API entitlement, and the user must be entitled to every field requested."
)

DATASET_COLUMNS: dict[str, list[str]] = {
    "universe": [c for c in F.UNIVERSE_COLUMNS if c != F.VENDOR_ID],
    "prices": list(F.PRICE_FIELDS),
    "fundamentals": list(F.FUNDAMENTAL_COLUMNS),
    "estimates": list(F.ESTIMATE_COLUMNS),
    "short_interest": list(F.SHORT_INTEREST_COLUMNS),
    "options": list(F.OPTIONS_COLUMNS),
}
DATE_COLUMNS = frozenset({F.PERIOD_END, F.REPORT_DATE, F.LAST_EARNINGS_DATE, F.NEXT_EARNINGS_DATE, F.SI_SETTLEMENT_DATE})
LABEL_COLUMNS = frozenset({F.NAME, F.GICS_SECTOR, F.GICS_INDUSTRY, F.EXCHANGE, F.COUNTRY, F.CURRENCY, F.SECURITY_TYPE, F.VENDOR_ID})
FORWARD_DATE_COLUMNS = frozenset({F.NEXT_EARNINGS_DATE})  # expected to be after as_of: no look-ahead guard
_YELLOW_KEYS = ("Equity", "Index", "Comdty", "Curncy", "Corp", "Govt", "Mtge", "Muni", "Pfd", "M-Mkt")
_YELLOW_LOWER = {k.lower(): k for k in _YELLOW_KEYS}
_FATAL_CATEGORIES = {"BAD_FLD", "NO_AUTH"}
_FATAL_SUBCATEGORIES = {"INVALID_FIELD", "NOT_AUTHORIZED", "NO_AUTH", "FIELD_NOT_AUTHORIZED"}
_FATAL_TEXT = re.compile(r"invalid|not valid|unknown field|not authori|entitle", re.IGNORECASE)


# =============================================================================================
# Pure helpers
# =============================================================================================


def split_vendor_id(vendor_id: str) -> tuple[str, str | None, str | None]:
    """``'AAPL US Equity'`` -> ``('AAPL', 'US', 'Equity')``; ``'SPX Index'`` -> ``('SPX', None, 'Index')``."""
    parts = str(vendor_id).strip().split()
    yellow = None
    if parts and parts[-1].lower() in _YELLOW_LOWER:
        yellow = _YELLOW_LOWER[parts.pop().lower()]
    root = parts[0] if parts else ""
    exch = parts[1] if len(parts) > 1 else None
    return root, exch, yellow


def to_canonical_ticker(vendor_id: str, home_codes: Iterable[str] = ("US",)) -> str:
    """Canonical ticker for a Bloomberg id.

    ``'AAPL US Equity'`` -> ``'AAPL'``; ``'BRK/B US Equity'`` -> ``'BRK-B'``; a non-home exchange keeps
    its code (``'VOD LN Equity'`` -> ``'VOD.LN'``); non-equity ids (``'SPX Index'``) are returned unchanged.
    """
    root, exch, yellow = split_vendor_id(vendor_id)
    if yellow not in (None, "Equity") or not root:
        return str(vendor_id).strip()
    root = root.upper().replace("/", "-")
    homes = {str(c).upper() for c in home_codes}
    if exch is None or exch.upper() in homes:
        return root
    return f"{root}.{exch.upper()}"


def to_vendor_id(ticker: str, default_exchange: str = "US") -> str:
    """Bloomberg id for a canonical ticker (``'AAPL'`` -> ``'AAPL US Equity'``, ``'BRK-B'`` -> ``'BRK/B US Equity'``).

    A string that already ends in a yellow key is returned unchanged.
    """
    t = str(ticker).strip()
    if not t:
        raise ValueError("empty ticker")
    last = t.split()[-1].lower()
    if last in _YELLOW_LOWER and len(t.split()) > 1:
        return t
    u = t.upper()
    m = re.fullmatch(r"([A-Z0-9]+)[-./]([A-Z])", u)
    if m:
        return f"{m.group(1)}/{m.group(2)} {default_exchange} Equity"
    m = re.fullmatch(r"([A-Z0-9/]+)\.([A-Z]{2})", u)
    if m:
        return f"{m.group(1)} {m.group(2)} Equity"
    if " " in u:
        return f"{t} Equity"
    return f"{u} {default_exchange} Equity"


def _as_date(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def _chunks(items: Sequence[str], n: int) -> Iterator[list[str]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def _dedupe(items: Iterable[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        s = str(it).strip()
        if s:
            seen.setdefault(s, None)
    return list(seen)


def _norm_col(c: Any) -> str:
    return str(c).strip().lstrip("#").strip().lower()


def _normalise_bql_frame(raw: Any) -> pd.DataFrame:
    """``bql.combined_df`` output -> flat frame with ``id``, optional ``date`` and lower-case item columns.

    BQL indexes the result by ``ID``; headers may come back upper-case (``MAXLINE``) or with the
    ``#`` of a let variable, so every header is normalised before any lookup.
    """
    df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    idx_names = [_norm_col(n) for n in df.index.names if n is not None]
    if idx_names or "id" not in {_norm_col(c) for c in df.columns}:
        df = df.reset_index()
    cols = [_norm_col(c) for c in df.columns]
    if "id" not in cols and "index" in cols:
        cols[cols.index("index")] = "id"
    df.columns = cols
    return df.loc[:, ~pd.Index(cols).duplicated()]


def _security_data(node: Any) -> list[Mapping[str, Any]]:
    """Every ``securityData`` element in a ``Message.toPy()`` dict (list for BDP, single for BDH)."""
    if isinstance(node, Mapping):
        if "securityData" in node:
            sd = node["securityData"]
            items = sd if isinstance(sd, list) else [sd]
            return [x for x in items if isinstance(x, Mapping)]
        out: list[Mapping[str, Any]] = []
        for v in node.values():
            out += _security_data(v)
        return out
    if isinstance(node, list):
        out = []
        for v in node:
            out += _security_data(v)
        return out
    return []


def _err_text(info: Any) -> str:
    if isinstance(info, Mapping):
        parts = [str(info.get(k)) for k in ("category", "subcategory", "message") if info.get(k)]
        return " / ".join(parts) if parts else json.dumps(dict(info), default=str)
    return str(info)


def _is_fatal(info: Any) -> bool:
    """Field exception meaning the field itself is invalid or not entitled (abort the leg)."""
    if not isinstance(info, Mapping):
        return bool(_FATAL_TEXT.search(str(info)))
    if str(info.get("category", "")).upper() in _FATAL_CATEGORIES:
        return True
    if str(info.get("subcategory", "")).upper() in _FATAL_SUBCATEGORIES:
        return True
    return bool(_FATAL_TEXT.search(str(info.get("message", ""))))


def _missing(v: Any) -> bool:
    if v is None or v is pd.NaT:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _to_datetime(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _column_kind(col: str, entry: Mapping[str, Any] | None = None) -> str:
    if entry and entry.get("kind") in ("number", "label", "date"):
        return str(entry["kind"])
    if col in DATE_COLUMNS:
        return "date"
    if col in LABEL_COLUMNS:
        return "label"
    return "number"


def _typed(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in DATE_COLUMNS:
            df[c] = _to_datetime(df[c])
        elif c in LABEL_COLUMNS:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df[cols]


class _RequestRejected(ProviderError):
    """A blpapi request could not be built (unknown request element)."""


# =============================================================================================
# Field self-check record
# =============================================================================================


@dataclass
class FieldCheck:
    """Result of requesting one mapped item alone on a known security (field admission, gate G5)."""

    field: str  # "features.rsi_14", "raw.short_interest.short_interest_shares", "universe.country", ...
    status_in_map: str
    returned_value: Any
    ok: bool
    note: str = ""
    channel: str = ""  # bql | bdp | bdh | derived
    expression: str = ""  # exact item / mnemonic requested (dates rendered)
    test_security: str = ""
    as_of: date | None = None

    def to_log_row(self, reviewer: str = "") -> dict[str, Any]:
        """Row for ``field_validation_log`` (VENDOR_REFERENCE section 5, step 4)."""
        return {
            "vendor": VENDOR, "item": self.expression, "field": self.field, "channel": self.channel,
            "status": self.status_in_map, "test_ticker": self.test_security,
            "as_of": self.as_of.isoformat() if self.as_of else None, "value": self.returned_value,
            "ok": self.ok, "note": self.note, "reviewer": reviewer, "date": date.today().isoformat(),
        }


# =============================================================================================
# Provider
# =============================================================================================


class BloombergProvider:
    """``MarketDataProvider`` + ``ScreenPushdown`` over Bloomberg BQL (BQuant) or the Desktop API."""

    name = VENDOR

    def __init__(
        self,
        *,
        backend: str = "bql",
        boundary: DataBoundary | None = None,
        fieldmap: Mapping[str, Any] | str | None = None,
        universe_expr: str | None = None,
        benchmark: str = "SPX Index",
        session: Any | None = None,
        refdata_session: Any | None = None,
        tickers: Sequence[str] | None = None,
        bql_module: Any | None = None,
        blpapi_module: Any | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout_ms: int | None = None,
        preflight: bool = True,
        test_security: str | None = None,
        admitted: Iterable[str] | None = None,
        batch_size: int = 500,
        history_batch_size: int = 100,
        snapshot_staleness_days: int = 5,
        today: Callable[[], date] | None = None,
    ) -> None:
        """
        Args:
            backend: ``'bql'`` (inside BQuant) or ``'blpapi'`` (Desktop / Server API).
            boundary: licensing boundary; default denies all text AND numeric features (L4, gate G1).
            fieldmap: override (mapping or JSON path) merged over the default map and
                ``$AITRADING_FIELDMAP_BLOOMBERG``.
            universe_expr: ``bql``: BQL universe replacing ``equitiesuniv(['ACTIVE','PRIMARY'])``, placeholders
                allowed (``"members('RAY Index', dates='{as_of}')"`` for point-in-time backtests).
                ``blpapi``: name of a saved EQS screen (``'GLOBAL:<name>'`` for a global one) run with BeqsRequest.
            benchmark: Bloomberg security for ``get_benchmark_history`` / ``{benchmark}`` placeholders.
            session: the primary session: a ``bql.Service()`` (bql) or a started ``blpapi.Session`` (blpapi).
            refdata_session: ``blpapi.Session`` for BDP-only fields under the bql backend (default: opened lazily).
            tickers: explicit universe for the blpapi backend.
            bql_module / blpapi_module: inject SDK modules (tests); otherwise imported lazily.
            host / port / timeout_ms: Desktop API connection (defaults from the map: localhost, 8194, 30000).
            preflight: preflight ``unverifiable`` items alone on ``test_security`` before bulk use.
            test_security: known security for preflights / ``verify_fields`` (default ``IBM US Equity``).
            admitted: optional G5 admission set; when given only these features are pushed down.
            batch_size / history_batch_size: securities per BDP / BQL-snapshot and per history request.
            snapshot_staleness_days: BDP (current-only) columns are blank for an as_of older than this.
            today: clock (tests).
        """
        if backend not in ("bql", "blpapi"):
            raise ValueError("backend must be 'bql' or 'blpapi'")
        self.backend = backend
        self.fieldmap: dict[str, Any] = load_fieldmap(VENDOR, fieldmap)
        reqs = self.fieldmap.get("requests") or {}
        self.host = host or str(self._req_value(reqs, "host", "localhost"))
        self.port = int(port or self._req_value(reqs, "port", 8194))
        self.timeout_ms = int(timeout_ms or self._req_value(reqs, "timeout_ms", 30000))
        self.refdata_service = str(self._req_value(reqs, "refdata_service", "//blp/refdata"))
        if "bqlsvc" in self.refdata_service.lower():
            raise ValueError("//blp/bqlsvc is prohibited (ADR-001 section 7); use //blp/refdata or BQuant")
        self.universe_expr = universe_expr
        self.benchmark = benchmark
        self.test_security = test_security or str(self.fieldmap.get("test_security") or "IBM US Equity")
        self.preflight = bool(preflight)
        self.admitted = set(admitted) if admitted is not None else None
        self.batch_size = max(1, int(batch_size))
        self.history_batch_size = max(1, int(history_batch_size))
        self.snapshot_staleness_days = int(snapshot_staleness_days)
        self._today = today or date.today
        self._bql_mod = bql_module
        self._blp_mod = blpapi_module
        self._bq_service = session if backend == "bql" else None
        self._refdata = session if backend == "blpapi" else refdata_session
        self._owns_refdata = False
        self._opened: set[int] = set()
        self._bdh_adjust = True
        self._tickers = _dedupe(tickers) if tickers else None
        self._ids: dict[str, str] = {}
        self._preflight_cache: dict[tuple[str, str], tuple[bool, str]] = {}
        self.warnings: list[str] = []
        self.query_log: list[str] = []
        self.last_pushdown: CompiledQuery | None = None
        self.last_pushdown_frame: pd.DataFrame | None = None  # vendor values of the last push-down (zone T only)
        country_map = ((((self.fieldmap.get("raw") or {}).get("universe") or {}).get("country") or {}).get("derived") or {}).get("value_map") or {}
        self._home_codes = tuple(k for k, v in country_map.items() if v == "US") or ("US",)
        self.boundary = self._init_boundary(boundary)
        self.capabilities: set[Capability] = self._capabilities()

    # ------------------------------------------------------------------ housekeeping
    @staticmethod
    def _req_value(reqs: Mapping[str, Any], key: str, default: Any) -> Any:
        v = reqs.get(key)
        if isinstance(v, Mapping):
            return v.get("value", default) if v.get("value") is not None else default
        return default if v is None else v

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    @property
    def fieldmap_digest(self) -> str:
        """SHA-256 of the field map in force (for the audit record)."""
        return digest(self.fieldmap)

    def _init_boundary(self, boundary: DataBoundary | None) -> DataBoundary:
        if boundary is None:
            return deny_all_text(provider=VENDOR, note=BOUNDARY_NOTE).model_copy(update={"allow_numeric_features": False})
        if not isinstance(boundary, DataBoundary):
            raise TypeError("boundary must be an aitrading.core.policy.DataBoundary")
        if boundary.provider != VENDOR:
            raise ValueError(f"boundary is for provider {boundary.provider!r}, expected {VENDOR!r}")
        if boundary.allow_numeric_features or boundary.allowed_document_kinds:
            self._warn(
                "bloomberg: a boundary wider than the ADR-001 default was passed. Bloomberg Terminal / Desktop API / "
                "BQuant data is licence class L4: it may leave zone T only after gate G1 (written Bloomberg approval), "
                "and then only as ids, ranks and booleans. Record the approval in the policy table."
            )
        return boundary

    def _channels(self, dataset: str) -> tuple[str, ...]:
        if dataset == "prices":
            return ("bql",) if self.backend == "bql" else ("bdh",)
        return ("bql", "bdp", "derived") if self.backend == "bql" else ("bdp", "derived")

    def _choose(self, spec: Any, dataset: str) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(spec, Mapping):
            return None
        for ch in self._channels(dataset):
            e = spec.get(ch)
            if not isinstance(e, Mapping):
                continue
            if ch == "derived" or e.get("expression") or e.get("code") or e.get("assume") is not None:
                return ch, dict(e)
        return None

    def _capabilities(self) -> set[Capability]:
        raw = self.fieldmap.get("raw") or {}
        caps: set[Capability] = set()
        by_dataset = {"prices": Capability.PRICES, "fundamentals": Capability.FUNDAMENTALS,
                      "estimates": Capability.ESTIMATES, "short_interest": Capability.SHORT_INTEREST,
                      "options": Capability.OPTIONS}
        for ds, cap in by_dataset.items():
            for col in DATASET_COLUMNS[ds]:
                choice = self._choose((raw.get(ds) or {}).get(col), ds)
                if choice and choice[1].get("units_verified") is not False:
                    caps.add(cap)
                    break
        if self.backend == "bql":
            caps.add(Capability.SCREEN_PUSHDOWN)
        return caps

    def _is_stale(self, as_of: date) -> bool:
        return as_of < self._today() - timedelta(days=self.snapshot_staleness_days)

    def _vendor_id(self, ticker: str) -> str:
        t = str(ticker).strip()
        return self._ids.get(t) or to_vendor_id(t)

    def _canonical(self, vendor_id: str) -> str:
        t = to_canonical_ticker(vendor_id, self._home_codes)
        self._ids.setdefault(t, str(vendor_id))
        return t

    # ------------------------------------------------------------------ SDK access (lazy)
    def _bql(self) -> Any:
        if self._bql_mod is None:
            try:
                self._bql_mod = importlib.import_module("bql")
            except ImportError as e:
                raise ProviderUnavailable(BQL_MISSING) from e
        return self._bql_mod

    def _bq(self) -> Any:
        if self._bq_service is None:
            mod = self._bql()
            try:
                self._bq_service = mod.Service()
            except Exception as e:  # noqa: BLE001
                raise ProviderUnavailable(f"bql.Service() failed ({type(e).__name__}: {e}): is this a BQuant session with an entitled login?") from e
        return self._bq_service

    def _blpapi(self) -> Any:
        if self._blp_mod is None:
            try:
                self._blp_mod = importlib.import_module("blpapi")
            except ImportError as e:
                raise ProviderUnavailable(BLPAPI_MISSING) from e
        return self._blp_mod

    def _refdata_session(self) -> Any:
        if self._refdata is None:
            blp = self._blpapi()
            opts = blp.SessionOptions()
            opts.setServerHost(self.host)
            opts.setServerPort(self.port)
            s = blp.Session(opts)
            try:
                started = s.start()
            except Exception as e:  # noqa: BLE001
                raise ProviderUnavailable(f"could not start a Bloomberg API session on {self.host}:{self.port}: {e}") from e
            if not started:
                raise ProviderUnavailable(
                    f"could not start a Bloomberg API session on {self.host}:{self.port}: is the Terminal running and "
                    "logged in (Desktop API), or is the B-PIPE / Server API host reachable?"
                )
            self._refdata, self._owns_refdata = s, True
        if id(self._refdata) not in self._opened:
            if not self._refdata.openService(self.refdata_service):
                raise ProviderUnavailable(f"could not open {self.refdata_service} (entitlement or connectivity problem)")
            self._opened.add(id(self._refdata))
        return self._refdata

    def close(self) -> None:
        """Stop a Desktop API session this provider opened itself."""
        if self._owns_refdata and self._refdata is not None:
            try:
                self._refdata.stop()
            finally:
                self._refdata, self._owns_refdata = None, False

    def diagnostics(self) -> list[str]:
        """Setup problems that would blank out data (empty list = ready). Does not open sessions."""
        problems: list[str] = []
        try:
            self._bql() if self.backend == "bql" else self._blpapi()
        except ProviderUnavailable as e:
            problems.append(str(e))
        if self.backend == "bql":
            raw = self.fieldmap.get("raw") or {}
            bdp_cols = [col for ds, cols in raw.items() if isinstance(cols, Mapping) for col, spec in cols.items()
                        if ds in DATASET_COLUMNS and (self._choose(spec, ds) or ("", {}))[0] == "bdp"]
            if bdp_cols:
                try:
                    self._blpapi()
                except ProviderUnavailable as e:
                    problems.append(f"BDP-only columns {', '.join(sorted(set(bdp_cols)))} will be NOT EVALUATED: {e}")
        elif not self._tickers and not self.universe_expr:
            problems.append("the blpapi backend cannot screen: pass tickers=[...] or universe_expr='<saved EQS screen>' "
                            "for get_universe (or run backend='bql' inside BQuant)")
        return problems

    # ------------------------------------------------------------------ BQL plumbing
    def _bql_run(self, query: str, expected: Sequence[str]) -> pd.DataFrame:
        mod = self._bql()
        bq = self._bq()
        self.query_log.append(query)
        try:
            resp = bq.execute(query)
            raw = mod.combined_df(resp)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"BQL request failed ({type(e).__name__}: {e}); query: {query}") from e
        df = _normalise_bql_frame(raw)
        missing = [v for v in expected if v not in df.columns]
        if "id" not in df.columns:
            missing.insert(0, "ID")
        if missing:
            raise ProviderError(f"BQL response lacks column(s) {missing} (got {list(df.columns)}); query: {query}")
        return df

    @staticmethod
    def _bql_query(items: Mapping[str, str], for_clause: str) -> str:
        lets = " ".join(f"#{k}={v};" for k, v in items.items())
        return f"let({lets}) get({', '.join('#' + k for k in items)}) for({for_clause})"

    @staticmethod
    def _bql_list(ids: Sequence[str]) -> str:
        return "[" + ",".join(bql_quote(i) for i in ids) + "]"

    def _bql_snapshot(self, items: Mapping[str, str], for_clause: str) -> pd.DataFrame:
        df = self._bql_run(self._bql_query(items, for_clause), list(items))
        if "date" in df.columns:
            df = df.sort_values("date", kind="stable")
        df["id"] = df["id"].astype(str)
        return df.groupby("id", sort=False)[list(items)].last()

    def _bql_history(self, items: Mapping[str, str], ids: Sequence[str]) -> dict[str, pd.DataFrame]:
        out: dict[str, list[pd.DataFrame]] = {k: [] for k in items}
        for chunk in _chunks(list(ids), self.history_batch_size):
            df = self._bql_run(self._bql_query(items, self._bql_list(chunk)), list(items))
            if "date" not in df.columns:
                if df[list(items)].map(_missing).all().all():
                    continue  # nothing came back for this chunk (no series at all)
                raise ProviderError("BQL history response has no DATE column")
            df["date"] = _to_datetime(df["date"])
            df["id"] = df["id"].astype(str)
            df = df.dropna(subset=["date"]).drop_duplicates(["date", "id"], keep="last")
            for k in items:
                out[k].append(df.pivot(index="date", columns="id", values=k))
        return {k: (pd.concat(v, axis=1) if v else pd.DataFrame()) for k, v in out.items()}

    # ------------------------------------------------------------------ blpapi plumbing
    def _blp_request(self, request_type: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        blp = self._blpapi()
        s = self._refdata_session()
        req = s.getService(self.refdata_service).createRequest(request_type)
        try:
            req.fromPy(dict(payload))
        except Exception as e:  # noqa: BLE001
            raise _RequestRejected(f"{request_type} rejected the request elements ({type(e).__name__}: {e})") from e
        self.query_log.append(f"{request_type} {json.dumps(dict(payload), sort_keys=True, default=str)}")
        s.sendRequest(req)
        out: list[dict[str, Any]] = []
        while True:
            ev = s.nextEvent(self.timeout_ms)
            et = ev.eventType()
            if et == blp.Event.TIMEOUT:
                raise ProviderError(f"{request_type} timed out after {self.timeout_ms} ms")
            for msg in ev:
                d = msg.toPy()
                if isinstance(d, Mapping):
                    if d.get("responseError"):
                        raise ProviderError(f"{request_type} failed: {_err_text(d['responseError'])}")
                    out.append(dict(d))
            if et == blp.Event.RESPONSE:
                return out

    @staticmethod
    def _bdp_key(code: str, overrides: Mapping[str, Any] | None) -> str:
        if not overrides:
            return code
        return code + "|" + json.dumps(dict(overrides), sort_keys=True, default=str)

    def _bdp(
        self, ids: Sequence[str], fields_: Sequence[tuple[str, Mapping[str, Any] | None]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[str, str, bool]]]]:
        """BDP: ``values[security][key]`` and ``errors[key] = [(security, message, fatal)]``.

        Fields are grouped by their override set (one ReferenceDataRequest per group and chunk),
        with the payload shape of the reference ``bdp()``: securities, fields, overrides.
        """
        groups: dict[tuple[tuple[str, Any], ...], list[str]] = {}
        for code, ov in fields_:
            key = tuple(sorted((ov or {}).items()))
            if code not in groups.setdefault(key, []):
                groups[key].append(code)
        values: dict[str, dict[str, Any]] = {}
        errors: dict[str, list[tuple[str, str, bool]]] = {}
        for ov, codes in groups.items():
            for chunk in _chunks(list(ids), self.batch_size):
                payload = {"securities": chunk, "fields": list(codes),
                           "overrides": [{"fieldId": k, "value": v} for k, v in ov]}
                for msg in self._blp_request("ReferenceDataRequest", payload):
                    for sd in _security_data(msg):
                        sec = str(sd.get("security", ""))
                        if sd.get("securityError"):
                            self._warn(f"bloomberg: {sec}: security error {_err_text(sd['securityError'])}")
                            continue
                        fd = sd.get("fieldData") or {}
                        row = values.setdefault(sec, {})
                        for code in codes:
                            if code in fd:
                                row[self._bdp_key(code, dict(ov))] = fd[code]
                        for fe in sd.get("fieldExceptions") or []:
                            info = fe.get("errorInfo") or {}
                            key = self._bdp_key(str(fe.get("fieldId")), dict(ov))
                            errors.setdefault(key, []).append((sec, _err_text(info), _is_fatal(info)))
        return values, errors

    def _history_elements(self) -> tuple[dict[str, Any], dict[str, Any]]:
        reqs = self.fieldmap.get("requests") or {}
        base = dict((reqs.get("history") or {}).get("elements") or {"periodicitySelection": "DAILY"})
        adj_cfg = reqs.get("history_adjustment") or {}
        adj = dict(adj_cfg.get("elements") or {}) if adj_cfg.get("enabled", True) else {}
        return base, adj

    def _bdh(self, ids: Sequence[str], codes: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        """BDH: ``{code: DataFrame(date x security)}``."""
        base, adj = self._history_elements()
        series: dict[str, dict[str, pd.Series]] = {c: {} for c in codes}
        for chunk in _chunks(list(ids), self.history_batch_size):
            payload: dict[str, Any] = {"securities": chunk, "fields": list(codes), **base}
            if adj and self._bdh_adjust:
                payload.update(adj)
            payload["startDate"] = start.strftime("%Y%m%d")
            payload["endDate"] = end.strftime("%Y%m%d")
            try:
                msgs = self._blp_request("HistoricalDataRequest", payload)
            except _RequestRejected as e:
                if not (adj and self._bdh_adjust):
                    raise ProviderError(str(e)) from e
                self._bdh_adjust = False
                self._warn(f"bloomberg: HistoricalDataRequest rejected the history_adjustment elements ({e}); retried "
                           "without them, so BDH prices are NOT dividend-adjusted")
                for k in adj:
                    payload.pop(k, None)
                msgs = self._blp_request("HistoricalDataRequest", payload)
            for msg in msgs:
                for sd in _security_data(msg):
                    sec = str(sd.get("security", ""))
                    if sd.get("securityError"):
                        self._warn(f"bloomberg: {sec}: security error {_err_text(sd['securityError'])}")
                        continue
                    for fe in sd.get("fieldExceptions") or []:
                        self._warn(f"bloomberg: {sec}: BDH field exception on {fe.get('fieldId')}: {_err_text(fe.get('errorInfo'))}")
                    rows = sd.get("fieldData") or []
                    if isinstance(rows, Mapping):
                        rows = [rows]
                    if not rows:
                        continue
                    frame = pd.DataFrame(list(rows))
                    if "date" not in frame.columns:
                        continue
                    idx = pd.to_datetime(frame["date"], errors="coerce")
                    for c in codes:
                        if c in frame.columns:
                            ser = pd.Series(frame[c].to_numpy(), index=idx)
                            prev = series[c].get(sec)
                            series[c][sec] = ser if prev is None else pd.concat([prev, ser])
        out: dict[str, pd.DataFrame] = {}
        for c, per in series.items():
            cleaned = {k: v[~v.index.duplicated(keep="last")].sort_index() for k, v in per.items()}
            out[c] = pd.DataFrame(cleaned) if cleaned else pd.DataFrame()
        return out

    def _beqs(self, screen: str, as_of: date) -> list[str]:
        cfg = (self.fieldmap.get("requests") or {}).get("beqs") or {}
        screen_type, name = str(cfg.get("screen_type") or "PRIVATE"), screen.strip()
        m = re.fullmatch(r"(?i)(PRIVATE|GLOBAL):(.+)", name)
        if m:
            screen_type, name = m.group(1).upper(), m.group(2).strip()
        payload: dict[str, Any] = {"screenName": name, "screenType": screen_type}
        if cfg.get("group"):
            payload["Group"] = cfg["group"]
        if as_of < self._today():
            payload["asOfDate"] = as_of.strftime(str(cfg.get("as_of_date_format") or "%Y%m%d"))
        ids: list[str] = []
        for msg in self._blp_request("BeqsRequest", payload):
            for sd in _security_data(msg):
                sec = str(sd.get("security", "")).strip()
                if sec and sec not in ids:
                    ids.append(sec)
        if not ids:
            self._warn(f"bloomberg: saved EQS screen {screen!r} returned no securities")
        return ids

    # ------------------------------------------------------------------ field-map driven fetch
    def _render(self, expr: str, as_of: date, start: date | None = None, end: date | None = None) -> str:
        expanded, _ = expand_refs(str(expr), self.fieldmap)
        return render_template(expanded, as_of, start=start, end=end, benchmark=self.benchmark)

    def _plan(self, dataset: str, columns: Sequence[str], as_of: date) -> dict[str, tuple[str, dict[str, Any]]]:
        raw = (self.fieldmap.get("raw") or {}).get(dataset) or {}
        plan: dict[str, tuple[str, dict[str, Any]]] = {}
        unmapped: list[str] = []
        unverified: list[str] = []
        stale: list[str] = []
        for col in columns:
            choice = self._choose(raw.get(col), dataset)
            if choice is None:
                unmapped.append(col)
                continue
            ch, e = choice
            if e.get("units_verified") is False:
                unverified.append(f"{col} ({ch} {e.get('expression') or e.get('code')})")
                continue
            if ch == "bdp" and self._is_stale(as_of):
                stale.append(col)
                continue
            plan[col] = (ch, e)
        if unmapped:
            self._warn(f"bloomberg: no {self.backend}-backend field-map entry for {dataset} column(s) {', '.join(unmapped)}: "
                       f"left NaN (add an admitted item via ${env_var(VENDOR)} to fill them)")
        if unverified:
            self._warn(f"bloomberg: {dataset} column(s) {', '.join(unverified)} have unverified units: not served until "
                       "admitted (check with verify_fields(), then set units_verified=true in a field-map override)")
        if stale:
            self._warn(f"bloomberg: BDP returns current values only, so {dataset} column(s) {', '.join(stale)} are blank "
                       f"for as_of {as_of} (more than {self.snapshot_staleness_days} days before today {self._today()})")
        return plan

    def _preflight_ok(self, channel: str, col: str, entry: Mapping[str, Any], as_of: date) -> bool:
        """Run an ``unverifiable`` item alone on ``test_security`` once per session; False suspends the leg."""
        if not self.preflight or entry.get("status") != "unverifiable" or channel == "derived":
            return True
        item = entry.get("expression") or entry.get("code")
        if not item:
            return True
        key = (channel, f"{item}|{json.dumps(entry.get('overrides') or {}, sort_keys=True)}")
        if key not in self._preflight_cache:
            chk = self._check_entry(f"preflight.{col}", channel, entry, self.test_security, as_of, column=col)
            self._preflight_cache[key] = (chk.ok, chk.note)
        ok, note = self._preflight_cache[key]
        if not ok:
            self._warn(f"LEG NOT EVALUATED: bloomberg {col} ({channel} {item}) failed its preflight on "
                       f"{self.test_security}: {note}")
        return ok

    def _convert(self, raw: pd.Series, col: str, entry: Mapping[str, Any]) -> pd.Series:
        kind = _column_kind(col, entry)
        if kind == "date":
            return _to_datetime(raw)
        if kind == "label":
            vm = entry.get("value_map") or {}

            def lab(v: Any) -> Any:
                if _missing(v):
                    return np.nan
                s = str(v).strip()
                if s in vm:
                    return vm[s]
                if col == F.SECURITY_TYPE and vm:
                    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
                return s

            return raw.map(lab).astype(object)
        x = pd.to_numeric(raw, errors="coerce").astype("float64") * float(entry.get("to_canonical", 1) or 1)
        x = x.where(np.isfinite(x))
        plaus = entry.get("plausible")
        if isinstance(plaus, (list, tuple)) and len(plaus) == 2 and x.notna().any():
            med = float(x.median())
            if not (float(plaus[0]) <= med <= float(plaus[1])):
                self._warn(f"bloomberg: {col} looks mis-scaled (median {med:.4g} outside the plausible range "
                           f"[{float(plaus[0]):g}, {float(plaus[1]):g}] in canonical units): check unit / to_canonical "
                           "in the field map with verify_fields()")
        return x

    def _derive(self, vendor_id: str, entry: Mapping[str, Any]) -> Any:
        src = entry.get("from")
        if src == "vendor_id_exchange_code":
            _, exch, _ = split_vendor_id(vendor_id)
            return (entry.get("value_map") or {}).get(exch or "", np.nan)
        raise FieldMapError(f"unknown derivation {src!r} in the Bloomberg field map")

    def _fetch(
        self,
        plan: Mapping[str, tuple[str, dict[str, Any]]],
        ids: Sequence[str] | None,
        as_of: date,
        *,
        for_clause: str | None = None,
    ) -> tuple[dict[str, pd.Series], list[str]]:
        """Fetch planned columns -> ``({column: Series indexed by vendor id}, ids)``.

        With ``for_clause`` (BQL universe) the ids come from the BQL response.
        """
        out: dict[str, pd.Series] = {}
        by_ch: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for col, (ch, e) in plan.items():
            by_ch.setdefault(ch, []).append((col, e))

        # ---- BQL
        items: dict[str, str] = {}
        entries: dict[str, dict[str, Any]] = {}
        assumed: list[tuple[str, dict[str, Any]]] = []
        for col, e in by_ch.get("bql", []):
            if not e.get("expression"):
                assumed.append((col, e))
                continue
            if self._preflight_ok("bql", col, e, as_of):
                items[col] = self._render(e["expression"], as_of)
                entries[col] = e
        if for_clause is not None and not items:
            probe = (self.fieldmap.get("universe") or {}).get("probe") or {}
            if not probe.get("expression"):
                raise ProviderError("no BQL item to request for the universe (field map has no universe.probe)")
            items["probe"] = self._render(probe["expression"], as_of)
        if items:
            if for_clause is not None:
                df = self._bql_snapshot(items, for_clause)
                ids = list(df.index)
            else:
                parts = [self._bql_snapshot(items, self._bql_list(chunk)) for chunk in _chunks(list(ids or []), self.batch_size)]
                df = pd.concat(parts) if parts else pd.DataFrame(columns=list(items))
            for col, e in entries.items():
                out[col] = self._convert(df[col], col, e)
        ids = list(ids or [])
        for col, e in assumed:
            self._warn(f"bloomberg: {col} has no verified BQL item; every security is labelled {e.get('assume')!r} "
                       f"(field-map assumption: {e.get('notes', '')})")
            out[col] = pd.Series([e.get("assume")] * len(ids), index=ids, dtype=object)

        # ---- BDP (primary on blpapi, secondary channel on bql)
        bdp = by_ch.get("bdp", [])
        if bdp and ids:
            try:
                usable = [(col, e) for col, e in bdp if self._preflight_ok("bdp", col, e, as_of)]
                values, errors = self._bdp(ids, [(e["code"], e.get("overrides")) for _, e in usable]) if usable else ({}, {})
            except ProviderError as err:
                if self.backend == "blpapi":
                    raise
                self._warn(f"LEG NOT EVALUATED: bloomberg {', '.join(c for c, _ in bdp)} need the Desktop API (BDP) "
                           f"alongside BQL and it is unavailable: {err}")
            else:
                for col, e in usable:
                    key = self._bdp_key(e["code"], e.get("overrides"))
                    errs = errors.get(key, [])
                    if any(f for _, _, f in errs):
                        msg = next(m for _, m, f in errs if f)
                        self._warn(f"LEG NOT EVALUATED: bloomberg {col} (BDP {e['code']}) field exception: {msg}")
                        continue
                    if errs:
                        self._warn(f"bloomberg: {col} (BDP {e['code']}) unavailable for {len(errs)} security(ies), "
                                   f"e.g. {errs[0][0]}: {errs[0][1]}")
                    out[col] = self._convert(pd.Series({i: values.get(i, {}).get(key) for i in ids}, dtype=object), col, e)

        # ---- derived from the Bloomberg id
        for col, e in by_ch.get("derived", []):
            ser = pd.Series({i: self._derive(i, e) for i in ids}, dtype=object)
            out[col] = self._convert(ser, col, {**e, "value_map": None})

        if ids:
            for col, ser in out.items():
                if ser.reindex(ids).isna().all():
                    self._warn(f"LEG NOT EVALUATED: bloomberg {col} came back missing for every requested security")
        return out, ids

    def _frame(self, dataset: str, tickers: Sequence[str], ids: Sequence[str], values: Mapping[str, pd.Series],
               as_of: date) -> pd.DataFrame:
        cols = DATASET_COLUMNS[dataset]
        df = pd.DataFrame(index=pd.Index(list(tickers), name="ticker"), columns=cols, dtype=object)
        for col, ser in values.items():
            if col in df.columns:
                df[col] = ser.reindex(list(ids)).to_numpy()
        df = _typed(df, cols)
        if dataset == "short_interest" and F.SI_SETTLEMENT_DATE in df.columns:
            late = df[F.SI_SETTLEMENT_DATE] > pd.Timestamp(as_of)
            if late.any():
                df.loc[late, cols] = np.nan
                df = _typed(df, cols)
                self._warn(f"bloomberg: {int(late.sum())} short-interest row(s) settle after as_of {as_of}: blanked (look-ahead guard)")
        for c in cols:
            if c in DATE_COLUMNS and c not in FORWARD_DATE_COLUMNS:
                late = df[c] > pd.Timestamp(as_of)
                if late.any():
                    df.loc[late, c] = pd.NaT
                    self._warn(f"bloomberg: {int(late.sum())} {c} value(s) after as_of {as_of} dropped (look-ahead guard)")
        return df

    def _snapshot(self, dataset: str, tickers: Sequence[str], as_of: Any) -> pd.DataFrame:
        a = _as_date(as_of)
        tick = _dedupe(tickers)
        if not tick:
            return self._frame(dataset, [], [], {}, a)
        if self.backend == "bql":
            self._bq()  # the primary SDK must be present (raises ProviderUnavailable)
        else:
            self._refdata_session()
        ids = [self._vendor_id(t) for t in tick]
        values, _ = self._fetch(self._plan(dataset, DATASET_COLUMNS[dataset], a), ids, a)
        return self._frame(dataset, tick, ids, values, a)

    # ------------------------------------------------------------------ MarketDataProvider
    def get_universe(self, spec: "UniverseSpec | None", as_of: date) -> pd.DataFrame:
        """Eligible securities indexed by canonical ticker, ``fields.UNIVERSE_COLUMNS``.

        ``bql``: ``filter(<equitiesuniv(['ACTIVE','PRIMARY']) | universe_expr>, cntry_of_risk()=='<country>')``
        (country defaults to 'US' when ``spec`` is None). ``blpapi``: ``tickers`` or a saved EQS screen.
        Then the cheap reference filters ``spec.country`` (listing country derived from the Bloomberg
        exchange code) and ``spec.security_types`` are applied locally; missing values never pass.
        """
        a = _as_date(as_of)
        country = (getattr(spec, "country", None) if spec is not None else "US") or None
        types = [str(t).strip().lower() for t in (getattr(spec, "security_types", None) or [])] if spec is not None else []
        cols = DATASET_COLUMNS["universe"]
        if self.backend == "bql":
            self._bq()
            for_clause = self._universe_for_clause(country, a)
            values, ids = self._fetch(self._plan("universe", cols, a), None, a, for_clause=for_clause)
        else:
            self._refdata_session()
            if self._tickers:
                ids = [self._vendor_id(t) for t in self._tickers]
            elif self.universe_expr:
                ids = self._beqs(self.universe_expr, a)
            else:
                raise ProviderError("the Desktop API cannot screen the universe: pass tickers=[...] or "
                                    "universe_expr='<saved EQS screen>' (BeqsRequest), or use backend='bql' inside BQuant")
            values, ids = self._fetch(self._plan("universe", cols, a), ids, a)
        tick: list[str] = []
        keep_ids: list[str] = []
        for i in ids:
            t = self._canonical(i)
            if t in tick:
                self._warn(f"bloomberg: duplicate canonical ticker {t} ({i}); kept {self._ids.get(t)}")
                continue
            tick.append(t)
            keep_ids.append(i)
        df = self._frame("universe", tick, keep_ids, values, a)
        df[F.VENDOR_ID] = pd.Series(keep_ids, index=df.index, dtype=object)
        if country:
            df = df[df[F.COUNTRY].fillna("").astype(str).str.upper() == str(country).upper()]
        if types:
            df = df[df[F.SECURITY_TYPE].fillna("").astype(str).str.lower().isin(types)]
        return df[F.UNIVERSE_COLUMNS]

    def _universe_for_clause(self, country: str | None, as_of: date) -> str:
        uni = self.fieldmap.get("universe") or {}
        base_expr = self.universe_expr or (uni.get("base") or {}).get("expression")
        if not base_expr:
            raise ProviderError("field map has no universe.base expression")
        base = render_template(str(base_expr), as_of, benchmark=self.benchmark)
        if self.universe_expr is None and self._is_stale(as_of):
            self._warn("bloomberg: equitiesuniv(['ACTIVE','PRIMARY']) is today's universe (equitiesuniv(dates=) is "
                       f"UNVERIFIED): as_of {as_of} is survivorship-biased; pass universe_expr=\"members('<index>', "
                       "dates='{as_of}')\" for a point-in-time universe")
        c = uni.get("country") or {}
        if country:
            if not re.fullmatch(r"[A-Za-z]{2}", str(country).strip()):
                raise ValueError(f"UniverseSpec.country must be an ISO alpha-2 code, got {country!r}")
            if c.get("expression") and c.get("status") in PUSHABLE_STATUSES:
                item = render_template(str(c["expression"]), as_of, benchmark=self.benchmark)
                return f"filter({base}, {item}=={bql_quote(str(country).strip().upper())})"
        if self.universe_expr:
            return base
        raise ProviderError("equitiesuniv(...) must be wrapped in filter(): pass a UniverseSpec.country (with an admissible "
                            "universe.country item) or a universe_expr")

    def get_price_history(self, tickers: list[str], start: date, end: date) -> PricePanel:
        """Adjusted daily OHLCV on trading sessions in [start, end] (BQL ``ca_adj='full'`` / BDH)."""
        s, e = _as_date(start), _as_date(end)
        tick = _dedupe(tickers)
        if not tick:
            empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=pd.Index([], name="ticker"), dtype="float64")
            return PricePanel(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
        ids = [self._vendor_id(t) for t in tick]
        frames = self._history(ids, s, e, list(F.PRICE_FIELDS))
        return self._panel(frames, ids, tick, s, e)

    def _history(self, ids: Sequence[str], start: date, end: date, cols: Sequence[str]) -> dict[str, pd.DataFrame]:
        if self.backend == "bql":
            self._bq()
        else:
            self._refdata_session()
        plan = self._plan("prices", cols, end)
        if F.CLOSE not in plan:
            raise ProviderError("the Bloomberg field map has no usable close-price item for this backend")
        usable = {c: (ch, en) for c, (ch, en) in plan.items() if self._preflight_ok(ch, c, en, end)}
        out: dict[str, pd.DataFrame] = {}
        if self.backend == "bql":
            items = {c: self._render(en["expression"], end, start=start, end=end) for c, (_, en) in usable.items()}
            got = self._bql_history(items, ids)
        else:
            codes = {c: str(en["code"]) for c, (_, en) in usable.items()}
            by_code = self._bdh(ids, list(dict.fromkeys(codes.values())), start, end)
            got = {c: by_code.get(code, pd.DataFrame()) for c, code in codes.items()}
        for c, frame in got.items():
            factor = float(usable[c][1].get("to_canonical", 1) or 1)
            out[c] = frame.apply(pd.to_numeric, errors="coerce").astype("float64") * factor if not frame.empty else frame
        return out

    def _panel(self, frames: Mapping[str, pd.DataFrame], ids: Sequence[str], tick: Sequence[str], s: date, e: date) -> PricePanel:
        idx = pd.DatetimeIndex([])
        for f in frames.values():
            if not f.empty:
                idx = idx.union(pd.DatetimeIndex(f.index))
        idx = idx[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))].sort_values()
        full = {c: frames[c].reindex(index=idx, columns=list(ids)) if c in frames and not frames[c].empty
                else pd.DataFrame(np.nan, index=idx, columns=list(ids)) for c in F.PRICE_FIELDS}
        vol, close = full[F.VOLUME], full[F.CLOSE]
        if vol.notna().any().any():
            sessions = idx[vol.notna().any(axis=1).to_numpy()]  # fill='prev' prices would otherwise create holiday rows
        else:
            sessions = idx[close.notna().any(axis=1).to_numpy()]
            if len(sessions):
                self._warn("bloomberg: no volume returned; sessions inferred from closes (filled holidays may remain)")
        renamed = {}
        for c, f in full.items():
            g = f.reindex(sessions)
            g.columns = pd.Index(list(tick), name="ticker")
            g.index.name = "date"
            renamed[c] = g.astype("float64")
        missing = [t for t in tick if renamed[F.CLOSE][t].isna().all()]
        if missing and len(missing) == len(tick):
            raise ProviderError(f"Bloomberg returned no prices for any of {list(tick)[:10]} in [{s}, {e}]")
        if missing:
            self._warn(f"bloomberg: no prices for {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''} in [{s}, {e}]")
        return PricePanel(renamed[F.OPEN], renamed[F.HIGH], renamed[F.LOW], renamed[F.CLOSE], renamed[F.VOLUME])

    def get_benchmark_history(self, start: date, end: date, symbol: str | None = None) -> pd.Series:
        """Close of the benchmark (default ``SPX Index``, a price index) on its trading sessions."""
        s, e = _as_date(start), _as_date(end)
        sym = symbol or self.benchmark
        vid = self._vendor_id(sym)
        frames = self._history([vid], s, e, [F.CLOSE, F.VOLUME])
        panel = self._panel(frames, [vid], [sym], s, e)
        ser = panel.close[sym].dropna()
        ser.name = sym
        return ser

    def get_fundamentals(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.FUNDAMENTAL_COLUMNS``; BQL items anchored with ``dates='<as_of>'`` (point-in-time)."""
        return self._snapshot("fundamentals", tickers, as_of)

    def get_estimates(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.ESTIMATE_COLUMNS`` (BQL BEst items; BEST_EPS / earnings dates via BDP)."""
        return self._snapshot("estimates", tickers, as_of)

    def get_short_interest(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.SHORT_INTEREST_COLUMNS`` via BDP (SHORT_INT / EQY_FLOAT / SHORT_INT_DT, all unverifiable)."""
        df = self._snapshot("short_interest", tickers, as_of)
        si, flt = df[F.SHORT_INTEREST_SHARES], df[F.FLOAT_SHARES]
        ratio = (si / flt).where((flt > 0) & si.notna())
        if ratio.notna().sum() >= 3 and float(ratio.median()) > 1.0:
            spec = ((self.fieldmap.get("raw") or {}).get("short_interest") or {}).get(F.FLOAT_SHARES) or {}
            ch, e = self._choose(spec, "short_interest") or ("?", {})
            self._warn(f"bloomberg: median short interest exceeds float: the float item ({ch} "
                       f"{e.get('code') or e.get('expression')}) is probably in millions; set "
                       f"raw.short_interest.{F.FLOAT_SHARES}.{ch}.to_canonical=1e6 in a field-map override")
        return df

    def get_options_summary(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """``fields.OPTIONS_COLUMNS`` (30d ATM implied vol; put/call has no verified item)."""
        return self._snapshot("options", tickers, as_of)

    def get_documents(
        self,
        ticker: str,
        kinds: set[DocumentKind] | None,
        start: date,
        end: date,
        limit: int = 10,
    ) -> list[Document]:
        """Bloomberg text is not fetched (licensing boundary, gate G1).

        Returns ``[]`` with a warning while the boundary denies the requested kinds (the default).
        If a wider boundary permits a kind, raises ``NotImplementedError``: no Bloomberg text API is
        verified (VENDOR_REFERENCE 1.3: transcripts have no BQL/BDP item, EDF Textual News is
        black-box only, broker research is Terminal-only; filings come from SEC EDGAR).
        """
        wanted = set(DocumentKind) if kinds is None else {DocumentKind(k) for k in kinds}
        allowed = wanted & set(self.boundary.allowed_document_kinds)
        if not allowed:
            self._warn(
                "bloomberg: document text is not fetched: Bloomberg Terminal/Desktop/BQuant content is licence class "
                "L4 and stays in zone T until gate G1 (EDF Textual News L5 and broker research L6 never leave). Use SEC "
                "EDGAR (L0) for filings and Kensho transcripts after gate G2."
            )
            return []
        docs = self.fieldmap.get("documents") or {}
        reasons = [f"{k.value}: {(docs.get(k.value) or {}).get('notes', 'no verified Bloomberg API')}"
                   for k in sorted(allowed, key=lambda k: k.value)]
        raise NotImplementedError(
            "the boundary passed to BloombergProvider permits document text, but no verified Bloomberg API exists for "
            "it, so nothing is fetched: " + " | ".join(reasons)
        )

    # ------------------------------------------------------------------ ScreenPushdown
    def compile_pushdown(self, spec: "ScreenSpec", as_of: date) -> CompiledQuery:
        """Compile (without executing) the BQL push-down, for the analyst's approval of the split."""
        return compile_screen(spec, self.fieldmap, as_of, universe_expr=self.universe_expr,
                              admitted=self.admitted, benchmark=self.benchmark)

    def pushdown_screen(self, spec: "ScreenSpec", as_of: date) -> PushdownResult:
        """Run ONE BQL request narrowing the universe; residual predicates are left to the local engine.

        The query (recorded verbatim in ``PushdownResult.query`` and ``query_log``) never ranks or
        truncates; ranking / top-N happen locally after every residual predicate (incl. short interest).
        """
        if self.backend != "bql":
            raise ProviderError("screen push-down needs backend='bql' (BQuant); the Desktop API cannot screen a universe")
        a = _as_date(as_of)
        try:
            cq = self.compile_pushdown(spec, a)
        except (CompileError, FieldMapError) as e:
            raise ProviderError(f"BQL push-down compile failed: {e}") from e
        for w in cq.warnings:
            self._warn(f"bloomberg push-down: {w}")
        df = self._bql_run(cq.query, list(cq.lets))
        tickers: list[str] = []
        for vid in pd.unique(df["id"].astype(str)):
            t = self._canonical(vid)
            if t not in tickers:
                tickers.append(t)
        self.last_pushdown, self.last_pushdown_frame = cq, df
        if not tickers:
            self._warn("bloomberg push-down returned 0 securities: a NaN item inside filter() silently drops every "
                       "security; check the pushed items with verify_fields() before trusting an empty screen")
        return PushdownResult(
            tickers=tickers,
            query=cq.query,
            pushed_conditions=[c.describe() for c in cq.pushed] + [f"universe: {u}" for u in cq.universe_conditions],
            residual_conditions=cq.residual_descriptions + [f"universe: {u}" for u in cq.universe_residual],
        )

    # ------------------------------------------------------------------ field self-check (G5)
    def verify_fields(self, sample_ticker: str | None = None, as_of: date | None = None) -> list[FieldCheck]:
        """Request every mapped item alone on a known security (field-admission procedure, gate G5).

        Covers the universe items, helpers and features (BQL backend) and every ``raw`` entry reachable
        on this backend (BQL, BDP / BDH, derived). A check fails on an error, a field exception, a
        missing column, an all-NaN result or an implausible scale. Record passing values with
        ``FieldCheck.to_log_row()`` before admitting an item.
        """
        a = _as_date(as_of) if as_of is not None else self._today()
        sec = self._vendor_id(sample_ticker) if sample_ticker else self.test_security
        if self.backend == "bql":
            self._bq()
        else:
            self._refdata_session()
        checks: list[FieldCheck] = []
        if self.backend == "bql":
            uni = self.fieldmap.get("universe") or {}
            for key in ("country", "probe"):
                e = uni.get(key)
                if isinstance(e, Mapping) and e.get("expression"):
                    checks.append(self._check_entry(f"universe.{key}", "bql", e, sec, a))
            for section in ("helpers", "features"):
                for name, e in (self.fieldmap.get(section) or {}).items():
                    if isinstance(e, Mapping) and e.get("expression"):
                        checks.append(self._check_entry(f"{section}.{name}", "bql", e, sec, a))
        for dataset, cols in (self.fieldmap.get("raw") or {}).items():
            if dataset not in DATASET_COLUMNS or not isinstance(cols, Mapping):
                continue
            for col, spec in cols.items():
                if not isinstance(spec, Mapping):
                    continue
                for ch in self._channels(dataset):
                    e = spec.get(ch)
                    if not isinstance(e, Mapping) or (ch != "derived" and not (e.get("expression") or e.get("code"))):
                        continue
                    checks.append(self._check_entry(f"raw.{dataset}.{col}", ch, e, sec, a, column=col))
        return checks

    def _check_entry(self, name: str, channel: str, entry: Mapping[str, Any], security: str, as_of: date,
                     *, column: str | None = None) -> FieldCheck:
        status = str(entry.get("status"))
        item = str(entry.get("expression") or entry.get("code") or entry.get("from") or "")
        guidance = {
            "confirmed": "confirmed",
            "corrected": "corrected: use exactly this form",
            "unverifiable": "UNVERIFIABLE: configurable only, preflight before every run, never a threshold until admitted (G5)",
        }.get(status, status)
        notes = [guidance]
        if entry.get("verify"):
            notes.append("VERIFY " + "; ".join(str(v) for v in entry["verify"]))
        rendered = item
        value: Any = None
        try:
            if channel == "bql":
                rendered = self._render(item, as_of, start=as_of - timedelta(days=10), end=as_of)
                df = self._bql_run(f"let(#v={rendered};) get(#v) for([{bql_quote(security)}])", ["v"])
                if "date" in df.columns:
                    df = df.sort_values("date", kind="stable")
                vals = df["v"][[not _missing(v) for v in df["v"]]]
                value = _py(vals.iloc[-1]) if len(vals) else None
            elif channel == "bdp":
                code, ov = str(entry["code"]), entry.get("overrides")
                rendered = code + (f" {json.dumps(dict(ov), sort_keys=True)}" if ov else "")
                values, errors = self._bdp([security], [(code, ov)])
                errs = errors.get(self._bdp_key(code, ov))
                if errs:
                    return FieldCheck(name, status, None, False, "; ".join(notes + [f"field exception: {errs[0][1]}"]),
                                      channel, rendered, security, as_of)
                value = _py(values.get(security, {}).get(self._bdp_key(code, ov)))
            elif channel == "bdh":
                frames = self._bdh([security], [item], as_of - timedelta(days=10), as_of)
                frame = frames.get(item, pd.DataFrame())
                vals = frame[security].dropna() if security in frame.columns else pd.Series(dtype=float)
                value = _py(vals.iloc[-1]) if len(vals) else None
            elif channel == "derived":
                value = _py(self._derive(security, entry))
            else:
                raise FieldMapError(f"unknown channel {channel!r}")
        except (ProviderError, FieldMapError, KeyError) as e:
            return FieldCheck(name, status, None, False, "; ".join(notes + [f"request failed: {e}"]), channel, rendered,
                              security, as_of)
        if _missing(value):
            return FieldCheck(name, status, None, False,
                              "; ".join(notes + ["all-NaN result (wrong item, missing entitlement or no data)"]),
                              channel, rendered, security, as_of)
        ok = True
        if column is not None and _column_kind(column, entry) == "number":
            try:
                canon = float(value) * float(entry.get("to_canonical", 1) or 1)
            except (TypeError, ValueError):
                return FieldCheck(name, status, value, False, "; ".join(notes + ["non-numeric value for a numeric column"]),
                                  channel, rendered, security, as_of)
            notes.append(f"vendor unit {entry.get('unit', '?')}; x{entry.get('to_canonical', 1)} -> canonical {canon:.6g}")
            plaus = entry.get("plausible")
            if isinstance(plaus, (list, tuple)) and len(plaus) == 2 and not (float(plaus[0]) <= canon <= float(plaus[1])):
                ok = False
                notes.append(f"implausible scale: canonical {canon:.6g} outside [{float(plaus[0]):g}, {float(plaus[1]):g}]")
        elif name.startswith("features.") and entry.get("threshold_scale"):
            try:
                notes.append(f"vendor unit {entry.get('unit', '?')}; catalog value = vendor / {entry['threshold_scale']} "
                             f"= {float(value) / float(entry['threshold_scale']):.6g} {entry.get('catalog_unit', '')}".rstrip())
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if entry.get("units_verified") is False:
            notes.append("units UNVERIFIED: confirm the unit before setting units_verified=true")
        return FieldCheck(name, status, value, ok, "; ".join(notes), channel, rendered, security, as_of)
