"""Deterministic compiler: ``ScreenSpec`` -> LSEG ``SCREEN(...)`` push-down expression.

The compiler never invents a vendor code: every expression comes from the LSEG field map data file
(``src/aitrading/data/fieldmaps/lseg.json``, overridable with ``$AITRADING_FIELDMAP_LSEG``; see
:func:`aitrading.data.lseg.load_fieldmap`). The LLM never writes this string.

Grammar (docs/VENDOR_REFERENCE.md section 2.2)
---------------------------------------------
``SCREEN(U(IN(<universe>)), cond, cond, ..., CURN=USD)``. Commas mean AND. Only the forms that the
reference shows are emitted:

* ``U(IN(Equity(active,public,primary))/*UNV:Public*/)`` universe, then the listing predicates
  ``IN(TR.ExchangeCountryCode,"US")``, ``IN(TR.InstrumentTypeCode,"ORD")``,
  ``NOT_IN(TR.ExchangeMarketIdCode,"OTCM")``;
* numeric comparisons with no spaces, ``<code><op><number>`` with ``op`` in ``> >= < <=``, e.g.
  ``TR.CompanyMarketCap(Scale=6)>=2000``. ``between`` compiles to a ``>=`` and a ``<=`` predicate
  (the official form), never to ``BETWEEN(...)``;
* single-value ``IN(<code>,"<label>")`` / ``NOT_IN(<code>,"<label>")`` for category features. A
  multi-value ``not_in`` becomes one ``NOT_IN`` per value (still an AND); a multi-value ``in`` stays
  local because the multi-value arity is not shown in the reference;
* ``CURN=USD`` last.

Thresholds are converted from catalog units to vendor units with the field map's
``threshold_scale`` using exact decimal arithmetic (``market_cap_usd_bn >= 2`` ->
``TR.CompanyMarketCap(Scale=6)>=2000``, USD millions). A negative scale flips the operator (for a
vendor field that reports a drop as a positive number). Numbers are printed in plain positional
notation, never ``2B`` or ``2e+09``.

What is pushed
--------------
A condition is pushed only when its feature's field-map entry has

1. an ``expression`` (features computed locally have ``"expression": null``);
2. ``status`` ``confirmed`` or ``corrected`` (``unverifiable`` is never pushed);
3. units not flagged ``"units_verified": false`` (unverified units enter only as z-scores, ADR);
4. ``screen`` ``official`` (verified inside an official SCREEN), or ``preflight`` *and* a passing
   preflight for that code today (pass ``preflight={code: bool}``; see :func:`preflight_codes`).
   ``screen: none`` marks a meaning-changing mapping that is never pushed.

Everything else - ``other_feature`` comparisons, numeric ``==`` / ``!=``, ``any_of`` OR groups,
unknown features - is a residual condition evaluated by the local engine. The local engine re-checks
every condition on the pushed survivors, so a push-down can only narrow the universe; a missing value
never passes either side (a NaN field inside SCREEN drops the instrument).

Point in time: SCREEN evaluates today's values. When ``as_of`` is more than
``max_as_of_lag_days`` before ``today`` only the static listing predicates are pushed and every
time-varying condition is residual (``CompiledQuery.point_in_time`` is False); the listing itself
still reflects today's active instruments (survivorship), which the provider warns about.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, ScreenSpec, UniverseSpec

__all__ = [
    "PUSHABLE_STATUSES",
    "VALID_STATUSES",
    "MAX_AS_OF_LAG_DAYS",
    "PushedPredicate",
    "ResidualCondition",
    "CompiledQuery",
    "compile_screen",
    "compile_universe",
    "preflight_codes",
    "screen_expression",
    "format_number",
    "is_point_in_time",
]

VALID_STATUSES = frozenset({"confirmed", "corrected", "unverifiable"})
PUSHABLE_STATUSES = frozenset({"confirmed", "corrected"})
SCREEN_OFFICIAL, SCREEN_PREFLIGHT, SCREEN_NONE = "official", "preflight", "none"
SCREEN_MODES = frozenset({SCREEN_OFFICIAL, SCREEN_PREFLIGHT, SCREEN_NONE})
MAX_AS_OF_LAG_DAYS = 5

_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
_LIQUIDITY_FEATURE = "avg_dollar_volume_20d_usd_mn"
_PRICE_FEATURE = "price"
_SECTOR_FEATURE = "gics_sector"


# =============================================================================================
# Results
# =============================================================================================


@dataclass(frozen=True)
class PushedPredicate:
    """One spec / universe condition evaluated inside SCREEN."""

    condition: str  # Condition.describe() or the local engine's universe label
    expressions: tuple[str, ...]  # the SCREEN arguments it compiled to
    feature: str  # catalog feature ('' for universe-definition predicates)
    status: str  # field-map status of the expression (confirmed / corrected)


@dataclass(frozen=True)
class ResidualCondition:
    """A condition the local engine evaluates, with the reason it was not pushed."""

    condition: str
    reason: str


@dataclass(frozen=True)
class CompiledQuery:
    expression: str  # the exact string sent as ``universe=`` (recorded verbatim for the audit trail)
    as_of: date
    predicates: tuple[str, ...]  # SCREEN arguments between U(...) and CURN=USD, in order
    pushed: tuple[PushedPredicate, ...]
    residual: tuple[ResidualCondition, ...]
    point_in_time: bool  # False: as_of is historical, only static listing predicates were pushed
    fieldmap_version: str
    fieldmap_sha256: str
    preflight: tuple[tuple[str, bool], ...] = field(default_factory=tuple)  # code -> passed, as used

    @property
    def pushed_conditions(self) -> list[str]:
        return list(dict.fromkeys(p.condition for p in self.pushed))

    @property
    def residual_conditions(self) -> list[str]:
        return list(dict.fromkeys(r.condition for r in self.residual))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.expression.encode("utf-8")).hexdigest()

    def to_audit(self) -> dict[str, Any]:
        """JSON-serialisable record for the run's audit trail (ADR section 7)."""
        return {
            "vendor": "lseg",
            "query": self.expression,
            "query_sha256": self.sha256,
            "as_of": self.as_of.isoformat(),
            "point_in_time": self.point_in_time,
            "fieldmap_version": self.fieldmap_version,
            "fieldmap_sha256": self.fieldmap_sha256,
            "pushed": [{"condition": p.condition, "expressions": list(p.expressions), "feature": p.feature,
                        "status": p.status} for p in self.pushed],
            "residual": [{"condition": r.condition, "reason": r.reason} for r in self.residual],
            "preflight": dict(self.preflight),
        }


# =============================================================================================
# Helpers
# =============================================================================================


def format_number(x: float | int | Decimal) -> str:
    """Plain positional decimal: 2000.0 -> '2000', -20 -> '-20', 0.5 -> '0.5', 1e10 -> '10000000000'."""
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    if not d.is_finite():
        raise ValueError(f"cannot format non-finite number {x!r}")
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _decimal(x: Any) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, bool):
        raise TypeError("boolean is not a number")
    if isinstance(x, int):
        return Decimal(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError("non-finite")
        return Decimal(repr(x))
    return Decimal(str(x))


def is_point_in_time(as_of: date, today: date | None = None, max_as_of_lag_days: int = MAX_AS_OF_LAG_DAYS) -> bool:
    """True when SCREEN's current values can stand in for ``as_of`` (as_of within the allowed lag)."""
    today = today or date.today()
    return as_of >= today - timedelta(days=int(max_as_of_lag_days))


def _fm(fieldmap: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if fieldmap is None:
        from aitrading.data.lseg import load_fieldmap  # noqa: PLC0415 - avoid an import cycle

        return load_fieldmap()
    return fieldmap


def _fieldmap_meta(fm: Mapping[str, Any]) -> tuple[str, str]:
    public = {k: v for k, v in fm.items() if not str(k).startswith("_")}
    digest = hashlib.sha256(json.dumps(public, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return str(fm.get("version", "")), digest


def _status_ok(entry: Mapping[str, Any] | None) -> bool:
    return isinstance(entry, Mapping) and entry.get("status") in PUSHABLE_STATUSES


def _quote(label: str) -> str | None:
    s = str(label).strip()
    if not s or '"' in s or "\n" in s:
        return None
    return f'"{s}"'


def _feature_entry(fm: Mapping[str, Any], feature: str) -> Mapping[str, Any] | None:
    entry = (fm.get("features") or {}).get(feature)
    return entry if isinstance(entry, Mapping) else None


def _unpushable_reason(entry: Mapping[str, Any] | None, preflight: Mapping[str, bool]) -> str | None:
    """None when the feature expression may be pushed, else the reason it stays local."""
    if entry is None:
        return "no LSEG expression in the field map"
    expr = entry.get("expression")
    if not expr:
        how = {"local": "computed locally", "survivors": "fetched for survivors only"}.get(
            str(entry.get("compute", "local")), f"computed {entry.get('compute')}")
        note = str(entry.get("notes", "")).strip()
        return f"{how} (field map has no SCREEN expression)" + (f": {note}" if note else "")
    status = entry.get("status")
    if status not in PUSHABLE_STATUSES:
        return f"field map status '{status}' for {expr}: never pushed (preflight + admission first)"
    if entry.get("units_verified") is False:
        return f"units/sign of {expr} UNVERIFIED in the field map: z-score only until admitted (G5)"
    mode = entry.get("screen", SCREEN_PREFLIGHT)
    if mode == SCREEN_NONE:
        note = str(entry.get("notes", "")).strip()
        return f"{expr} is not pushed (field map screen=none)" + (f": {note}" if note else "")
    if mode == SCREEN_PREFLIGHT and not preflight.get(str(expr), False):
        return f"{expr} is not verified inside an official SCREEN and has no passing preflight today"
    return None


def _scaled_threshold(value: float, entry: Mapping[str, Any]) -> Decimal:
    return _decimal(value) * _decimal(entry.get("threshold_scale", 1))


class _Builder:
    def __init__(self) -> None:
        self.predicates: list[str] = []
        self.pushed: list[PushedPredicate] = []
        self.residual: list[ResidualCondition] = []

    def push(self, label: str, exprs: list[str], feature: str, status: str) -> None:
        for e in exprs:
            if e not in self.predicates:
                self.predicates.append(e)
        self.pushed.append(PushedPredicate(label, tuple(exprs), feature, status))

    def keep_local(self, label: str, reason: str) -> None:
        self.residual.append(ResidualCondition(label, reason))


# =============================================================================================
# Condition compilation
# =============================================================================================


def _compile_numeric(cond: Condition, expr: str, entry: Mapping[str, Any]) -> tuple[list[str], str | None]:
    if cond.other_feature is not None:
        return [], "comparison against another feature is evaluated locally"
    try:
        scale = _decimal(entry.get("threshold_scale", 1))
    except (InvalidOperation, TypeError, ValueError):
        return [], f"invalid threshold_scale for {expr} in the field map"
    if scale == 0:
        return [], f"threshold_scale 0 for {expr} in the field map"
    flip = scale < 0

    def pred(op: str, value: float | None) -> str:
        if value is None or not math.isfinite(float(value)):
            raise ValueError("missing threshold")
        o = _FLIP[op] if flip else op
        return f"{expr}{o}{format_number(_scaled_threshold(value, entry))}"

    try:
        if cond.op == "between":
            if cond.value is None or cond.value_high is None or cond.value > cond.value_high:
                return [], "malformed 'between' (local engine reports it)"
            return [pred(">=", cond.value), pred("<=", cond.value_high)], None
        if cond.op in _FLIP:
            return [pred(cond.op, cond.value)], None
    except ValueError:
        return [], "missing or non-finite threshold (local engine reports it)"
    if cond.op in ("==", "!="):
        return [], f"numeric '{cond.op}' is evaluated locally (float tolerance semantics differ)"
    return [], f"operator '{cond.op}' does not apply to a numeric feature"


def _compile_category(cond: Condition, expr: str, entry: Mapping[str, Any]) -> tuple[list[str], str | None]:
    if cond.other_feature is not None:
        return [], "category features cannot be compared to another feature"
    if cond.values:
        labels = list(cond.values)
    elif cond.op in ("==", "!=") and cond.value is not None:
        labels = [f"{cond.value:g}"]
    else:
        return [], "category condition without values (local engine reports it)"
    known = entry.get("labels")
    if known:
        # SCREEN label matching is exact while the local engine is case-insensitive: push only labels
        # that resolve (case-insensitively) to a vendor label listed in the field map.
        canon = {str(k).strip().casefold(): str(k) for k in known}
        unknown = [v for v in labels if str(v).strip().casefold() not in canon]
        if unknown:
            return [], f"label(s) {unknown} not among the field map's vendor labels for {expr}"
        labels = [canon[str(v).strip().casefold()] for v in labels]
    quoted = [_quote(v) for v in labels]
    if any(q is None for q in quoted):
        return [], "category label is empty or contains a double quote"
    if cond.op in ("in", "=="):
        if len(quoted) != 1:
            return [], "multi-value IN is evaluated locally (multi-value arity not shown in the reference)"
        return [f"IN({expr},{quoted[0]})"], None
    if cond.op in ("not_in", "!="):
        return [f"NOT_IN({expr},{q})" for q in dict.fromkeys(quoted)], None
    return [], f"operator '{cond.op}' does not apply to a category feature"


def _compile_condition(cond: Condition, fm: Mapping[str, Any], catalog: FeatureCatalog,
                       preflight: Mapping[str, bool]) -> tuple[list[str], str, str | None]:
    """(SCREEN predicates, status, None) or ([], '', reason)."""
    if cond.feature not in catalog:
        return [], "", f"unknown feature '{cond.feature}'"
    entry = _feature_entry(fm, cond.feature)
    reason = _unpushable_reason(entry, preflight)
    if reason:
        return [], "", reason
    assert entry is not None
    expr = str(entry["expression"])
    if catalog[cond.feature].dtype == "category" or cond.op in ("in", "not_in"):
        if catalog[cond.feature].dtype != "category":
            return [], "", f"'{cond.op}' only applies to category features (local engine reports it)"
        preds, why = _compile_category(cond, expr, entry)
    else:
        preds, why = _compile_numeric(cond, expr, entry)
    if why:
        return [], "", why
    return preds, str(entry["status"]), None


# =============================================================================================
# Universe
# =============================================================================================


def _listing(universe: UniverseSpec, fm: Mapping[str, Any], b: _Builder) -> None:
    """Static listing predicates: country, security type, configured listing filters."""
    scr = fm.get("screen") or {}
    country = universe.country
    label = f"country == {country}"
    if country:
        entry = scr.get("country")
        if _status_ok(entry) and entry.get("template") and _quote(country) is not None:
            b.push(label, [str(entry["template"]).format(value=str(country).strip().upper())], "", str(entry["status"]))
        else:
            b.keep_local(label, "country predicate is not a confirmed/corrected field-map template")
    types = list(universe.security_types or [])
    if types:
        label = f"security_type in [{', '.join(types)}]"
        entry = scr.get("security_type")
        codes = (entry or {}).get("values") or {}
        if not _status_ok(entry) or not entry.get("template"):
            b.keep_local(label, "security-type predicate is not a confirmed/corrected field-map template")
        elif len(types) != 1:
            b.keep_local(label, "several security types: multi-value IN is evaluated locally")
        elif types[0] not in codes or _quote(codes[types[0]]) is None:
            b.keep_local(label, f"no verified LSEG instrument-type code for '{types[0]}'")
        else:
            b.push(label, [str(entry["template"]).format(value=codes[types[0]])], "", str(entry["status"]))
    for flt in scr.get("listing_filters") or []:
        if isinstance(flt, Mapping) and _status_ok(flt) and flt.get("expression"):
            b.push(f"universe: {flt.get('label') or flt['expression']}", [str(flt["expression"])], "", str(flt["status"]))


def _universe_time_varying(universe: UniverseSpec, fm: Mapping[str, Any], catalog: FeatureCatalog,
                           preflight: Mapping[str, bool], b: _Builder, point_in_time: bool) -> None:
    """Price floor, liquidity floor and sector exclusions (labels match the local engine's funnel)."""
    if universe.min_price is not None:
        cond = Condition(feature=_PRICE_FEATURE, op=">=", value=universe.min_price)
        _push_or_keep(f"{_PRICE_FEATURE} >= {universe.min_price:g}", cond, fm, catalog, preflight, b, point_in_time)
    if universe.min_avg_dollar_volume_usd_mn is not None:
        cond = Condition(feature=_LIQUIDITY_FEATURE, op=">=", value=universe.min_avg_dollar_volume_usd_mn)
        _push_or_keep(f"{_LIQUIDITY_FEATURE} >= {universe.min_avg_dollar_volume_usd_mn:g}", cond, fm, catalog,
                      preflight, b, point_in_time)
    if universe.exclude_sectors:
        cond = Condition(feature=_SECTOR_FEATURE, op="not_in", values=list(universe.exclude_sectors))
        _push_or_keep(f"{_SECTOR_FEATURE} not in [{', '.join(universe.exclude_sectors)}]", cond, fm, catalog,
                      preflight, b, point_in_time)


def _push_or_keep(label: str, cond: Condition, fm: Mapping[str, Any], catalog: FeatureCatalog,
                  preflight: Mapping[str, bool], b: _Builder, point_in_time: bool) -> None:
    if not point_in_time:
        b.keep_local(label, "as_of is historical: SCREEN evaluates current values only (no point-in-time)")
        return
    preds, status, why = _compile_condition(cond, fm, catalog, preflight)
    if why:
        b.keep_local(label, why)
    else:
        b.push(label, preds, cond.feature, status)


def screen_expression(predicates: list[str] | tuple[str, ...], fieldmap: Mapping[str, Any] | None = None) -> str:
    """``SCREEN(<universe>, <predicates...>, CURN=USD)`` joined with ', ' exactly as the reference does."""
    fm = _fm(fieldmap)
    scr = fm.get("screen") or {}
    uni, cur = scr.get("universe") or {}, scr.get("currency") or {}
    if not _status_ok(uni) or not uni.get("expression"):
        raise ValueError("field map screen.universe must be a confirmed/corrected expression")
    parts = [str(uni["expression"]), *predicates]
    if _status_ok(cur) and cur.get("expression"):
        parts.append(str(cur["expression"]))
    return f"SCREEN({', '.join(parts)})"


def compile_universe(universe: UniverseSpec | None, fieldmap: Mapping[str, Any] | None = None,
                     as_of: date | None = None) -> CompiledQuery:
    """SCREEN for the provider's universe: static listing predicates only (no as_of dependence).

    ``point_in_time`` is always False: the listing is today's active instruments whatever ``as_of`` is.
    """
    fm = _fm(fieldmap)
    b = _Builder()
    _listing(universe or UniverseSpec(), fm, b)
    version, digest = _fieldmap_meta(fm)
    return CompiledQuery(screen_expression(b.predicates, fm), as_of or date.min, tuple(b.predicates),
                         tuple(b.pushed), tuple(b.residual), False, version, digest)


# =============================================================================================
# Public entry points
# =============================================================================================


def preflight_codes(spec: ScreenSpec, fieldmap: Mapping[str, Any] | None = None, as_of: date | None = None, *,
                    today: date | None = None, max_as_of_lag_days: int = MAX_AS_OF_LAG_DAYS,
                    catalog: FeatureCatalog | None = None) -> list[str]:
    """SCREEN expressions that need a passing preflight today before they can be pushed for ``spec``."""
    fm = _fm(fieldmap)
    catalog = catalog or default_catalog()
    if as_of is not None and not is_point_in_time(as_of, today, max_as_of_lag_days):
        return []
    feats: list[str] = [c.feature for c in spec.conditions if c.other_feature is None]
    u = spec.universe
    if u.min_price is not None:
        feats.append(_PRICE_FEATURE)
    if u.min_avg_dollar_volume_usd_mn is not None:
        feats.append(_LIQUIDITY_FEATURE)
    if u.exclude_sectors:
        feats.append(_SECTOR_FEATURE)
    out: list[str] = []
    for f in dict.fromkeys(feats):
        entry = _feature_entry(fm, f)
        if (f in catalog and entry is not None and entry.get("expression") and _status_ok(entry)
                and entry.get("units_verified") is not False
                and entry.get("screen", SCREEN_PREFLIGHT) == SCREEN_PREFLIGHT):
            out.append(str(entry["expression"]))
    return list(dict.fromkeys(out))


def compile_screen(spec: ScreenSpec, fieldmap: Mapping[str, Any] | None, as_of: date, *,
                   preflight: Mapping[str, bool] | None = None, today: date | None = None,
                   max_as_of_lag_days: int = MAX_AS_OF_LAG_DAYS, catalog: FeatureCatalog | None = None) -> CompiledQuery:
    """Compile ``spec`` to one LSEG ``SCREEN(...)`` expression plus the pushed / residual split.

    Args:
        spec: the screen.
        fieldmap: the merged LSEG field map (``aitrading.data.lseg.load_fieldmap()``); None loads it.
        as_of: absolute as-of date of the run (recorded; decides point-in-time push-down).
        preflight: ``{expression: passed}`` from today's preflight on the test RIC; expressions marked
            ``screen: preflight`` are pushed only when they passed. None = nothing passed.
        today: clock for the point-in-time rule (default ``date.today()``).
        max_as_of_lag_days: an ``as_of`` older than this pushes static listing predicates only.
    """
    fm = _fm(fieldmap)
    catalog = catalog or default_catalog()
    pf = {str(k): bool(v) for k, v in (preflight or {}).items()}
    pit = is_point_in_time(as_of, today, max_as_of_lag_days)
    b = _Builder()
    _listing(spec.universe, fm, b)
    _universe_time_varying(spec.universe, fm, catalog, pf, b, pit)
    for cond in spec.conditions:
        _push_or_keep(cond.describe(), cond, fm, catalog, pf, b, pit)
    for group in spec.any_of:
        label = "any of: " + " | ".join(c.describe() for c in group)
        b.keep_local(label, "any_of OR groups are evaluated locally (parenthesised OR form not shown in the reference)")
    version, digest = _fieldmap_meta(fm)
    used = tuple(sorted((k, v) for k, v in pf.items()))
    return CompiledQuery(screen_expression(b.predicates, fm), as_of, tuple(b.predicates), tuple(b.pushed),
                         tuple(b.residual), pit, version, digest, used)
