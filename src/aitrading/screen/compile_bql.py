"""Compile a ScreenSpec into ONE Bloomberg BQL request string (push-down, ADR-001 Phase 1).

The compiler is deterministic and data-driven: every BQL item comes from the Bloomberg field map
(``aitrading/data/fieldmaps/bloomberg.json``, see :mod:`aitrading.data.fieldmaps`); nothing here
names a vendor item. The LLM never writes BQL: it emits a typed ScreenSpec and this module turns
the admissible part of it into a query that follows the grammar in ``docs/VENDOR_REFERENCE.md``
section 1.2:

* Shape: ``let(#<feature>=<expr>; ...) get(#<feature>, ...) for(<universe>)``. Each pushed feature is
  one ``let`` variable (``let`` variables can be used inside ``filter()``). Helper references
  (``#px_2y`` ...) inside a field-map expression are expanded inline, so every ``let`` is built only
  from vendor primitives.
* Universe: ``equitiesuniv(['ACTIVE','PRIMARY'])`` (or the caller's ``universe_expr``, e.g.
  ``members('RAY Index', dates='{as_of}')`` for backtests), always wrapped in ``filter()``.
* Filter order: nested ``filter()``. The inner filter holds the cheap static predicates (country,
  size, sector, fundamentals); the outer filter holds the time-series studies, so they only run on
  what survives. Predicates are combined with infix ``and``.
* Dates: every date is an absolute ISO date derived from ``as_of`` (never ``'0D'``).
* Literals: plain numbers (``2000000000``, never ``'2B'``).
* Units: thresholds are converted from catalog units to the vendor unit with the entry's
  ``threshold_scale`` (vendor value = catalog value x threshold_scale): ``market_cap_usd_bn`` 2 ->
  ``2000000000`` against ``cur_mkt_cap(currency='USD')``; ``drawdown_from_52w_high_pct`` -45 ->
  ``-0.45`` against the fractional ``px_last()/hi52 - 1``. Decimal arithmetic keeps literals exact.
* No ranking: the query never contains ``groupzscore`` / ``grouprank`` / ``groupsort`` or a top-N.
  Every pushed predicate, including short interest when its map entry is confirmed/corrected, sits
  inside ``filter()``; the spec's residual predicates (short interest by default) then run locally
  on ALL survivors and only after that are names ranked and truncated (ADR graft 2: never filter
  after ``grouprank``).

What is pushed
--------------
A condition is pushed only when every feature it uses has a field-map entry with a non-null
``expression`` whose status (and the status of every helper it references) is ``confirmed`` or
``corrected``, ``pushdown`` is not false, ``units_verified`` is not false (items with unverified
units enter only as z-scores, ADR graft 1), the entry's ``catalog_unit`` matches the catalog, and
the operator is expressible in the reference grammar:

* numeric ``>``, ``>=``, ``<``, ``<=``, ``between`` (two inclusive bounds), including
  ``feature <op> multiplier x other_feature`` when both sides are pushable;
* a category ``==`` / single-label ``in`` whose label is in the entry's closed ``allowed_values``
  list (emitted in the map's spelling, so no spec text is ever interpolated into the query).

Everything else is *residual* (with a recorded reason) and is evaluated by the local engine:
numeric ``==`` / ``!=`` (tolerance semantics), ``not_in`` / ``!=`` / multi-label ``in`` (the
reference grammar shows no ``or`` / ``!=`` in string form), boolean features, every ``any_of``
group (pushing part of an OR group would drop names), and unmapped / unverifiable features.
The optional ``admitted`` set enforces gate G5 strictly: only features listed there are pushed.

A missing value never passes: in BQL a NaN comparison inside ``filter()`` is false (the "silent
drop"), and locally the engine treats missing as failing. Because the local engine re-evaluates
every condition on the survivors, a push-down can only narrow the universe, never admit a name.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Collection, Mapping

import pandas as pd

from aitrading.data.fieldmaps import PUSHABLE_STATUSES, FieldMapError, digest
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, ScreenSpec

__all__ = [
    "CompileError",
    "CompiledQuery",
    "compile_screen",
    "render_template",
    "expand_refs",
    "feature_pushability",
    "bql_number",
    "bql_quote",
    "DEFAULT_UNIVERSE",
]

DEFAULT_UNIVERSE = "equitiesuniv(['ACTIVE','PRIMARY'])"
NUMERIC_PUSH_OPS = (">", ">=", "<", "<=", "between")
_FLIP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
_STATIC_SOURCES = ("reference", "fundamental")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REF = re.compile(r"#([A-Za-z_][A-Za-z0-9_]*)")
_QUOTED = re.compile(r"('[^']*')")


class CompileError(ValueError):
    """The spec / field map cannot be compiled into a valid BQL request."""


@dataclass(frozen=True)
class CompiledQuery:
    """One BQL request plus the pushed / residual split the analyst approves (ADR step 2)."""

    query: str
    pushed: list[Condition]
    residual: list[Condition]
    universe_conditions: list[str]  # UniverseSpec filters pushed vendor-side (human-readable)
    universe_residual: list[str] = field(default_factory=list)  # UniverseSpec filters left to the local engine
    residual_groups: list[list[Condition]] = field(default_factory=list)  # any_of groups (always local)
    reasons: dict[str, str] = field(default_factory=dict)  # describe() / universe label -> why it is residual
    lets: dict[str, str] = field(default_factory=dict)  # let variable -> fully expanded BQL expression
    warnings: list[str] = field(default_factory=list)  # VERIFY caveats of pushed items, survivorship notes
    as_of: date | None = None
    fieldmap_digest: str = ""

    @property
    def sha256(self) -> str:
        """Hash of the query string, for the audit record."""
        return hashlib.sha256(self.query.encode("utf-8")).hexdigest()

    @property
    def pushed_descriptions(self) -> list[str]:
        return [c.describe() for c in self.pushed]

    @property
    def residual_descriptions(self) -> list[str]:
        out = [c.describe() for c in self.residual]
        out += ["any of: " + " | ".join(c.describe() for c in g) for g in self.residual_groups]
        return out


# ------------------------------------------------------------------------------------------------
# Small pure helpers (shared with the Bloomberg provider)
# ------------------------------------------------------------------------------------------------


def _as_date(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, pd.Timestamp):
        return x.date()
    return pd.Timestamp(x).date()


def _iso(d: date | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def render_template(
    template: str,
    as_of: date | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    benchmark: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> str:
    """Fill the field-map placeholders of a BQL template with absolute dates.

    ``{as_of}`` -> ISO as-of date; ``{d<N>}`` -> as_of minus N calendar days; ``{as_of_1y}`` -> as_of
    minus one calendar year; ``{start}`` / ``{end}`` -> history window; ``{benchmark}`` -> benchmark
    security; ``extra`` adds more names. An unknown placeholder (or one whose value is not supplied)
    raises :class:`FieldMapError`, so a template can never silently fall back to a relative date.
    """
    a = pd.Timestamp(_as_date(as_of)) if as_of is not None else None

    def value(name: str) -> str:
        if extra and name in extra:
            return str(extra[name])
        if name == "as_of" and a is not None:
            return _iso(a)
        if name == "as_of_1y" and a is not None:
            return _iso(a - pd.DateOffset(years=1))
        m = re.fullmatch(r"d(\d+)", name)
        if m and a is not None:
            return _iso(a - pd.Timedelta(days=int(m.group(1))))
        if name == "start" and start is not None:
            return _iso(start)
        if name == "end" and end is not None:
            return _iso(end)
        if name == "benchmark" and benchmark:
            return str(benchmark)
        raise FieldMapError(f"template placeholder {{{name}}} cannot be filled in {template!r}")

    return _PLACEHOLDER.sub(lambda m: value(m.group(1)), template)


def _needs_parens(expr: str) -> bool:
    """True if ``expr`` has an operator at nesting depth 0 (outside quotes)."""
    depth = 0
    in_q = False
    s = expr.strip()
    for i, ch in enumerate(s):
        if ch == "'":
            in_q = not in_q
        elif in_q:
            continue
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch in "+-*/<>=!":
            return True
        elif depth == 0 and ch == " " and (s[i:].startswith(" and ") or s[i:].startswith(" or ")):
            return True
    return False


def expand_refs(
    expr: str,
    fieldmap: Mapping[str, Any],
    *,
    _stack: tuple[str, ...] = (),
    _seen: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Expand ``#name`` references (field-map helpers, then features) inline.

    Returns ``(expanded, refs)`` where ``refs`` maps every referenced name (transitively) to its
    status. Composite sub-expressions are parenthesised. Unknown names, references to entries
    without an expression and cycles raise :class:`FieldMapError`.
    """
    seen: dict[str, str] = {} if _seen is None else _seen
    helpers = fieldmap.get("helpers") or {}
    features = fieldmap.get("features") or {}

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in _stack:
            raise FieldMapError(f"circular reference in field map: {' -> '.join((*_stack, name))}")
        entry = helpers.get(name) if name in helpers else features.get(name)
        if not isinstance(entry, Mapping):
            raise FieldMapError(f"expression {expr!r} references unknown #{name}")
        inner = entry.get("expression")
        if not inner:
            raise FieldMapError(f"#{name} has no expression in the field map")
        seen[name] = str(entry.get("status"))
        expanded, _ = expand_refs(str(inner), fieldmap, _stack=(*_stack, name), _seen=seen)
        return f"({expanded})" if _needs_parens(expanded) else expanded

    parts = _QUOTED.split(expr)
    out = [p if p.startswith("'") and p.endswith("'") and len(p) >= 2 else _REF.sub(sub, p) for p in parts]
    return "".join(out), seen


def bql_number(x: float | int | Decimal) -> str:
    """Plain-number BQL literal: exact decimal, no exponent ('2000000000', '-0.45', '1.5')."""
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    if not d.is_finite():
        raise CompileError(f"non-finite literal {x!r}")
    d = d.normalize()
    if d == 0:
        return "0"
    if d == d.to_integral_value():
        return str(int(d))
    return format(d, "f")


def bql_quote(s: str) -> str:
    """Single-quoted BQL string literal. Rejects characters that could break out of the literal."""
    text = str(s)
    if any(ch in text for ch in "'\\\n\r\t") or "\x00" in text:
        raise CompileError(f"refusing to quote {s!r} into BQL")
    return f"'{text}'"


def _dec(x: float) -> Decimal:
    try:
        return Decimal(repr(float(x)))
    except (InvalidOperation, ValueError, TypeError) as e:  # pragma: no cover - guarded by callers
        raise CompileError(f"invalid number {x!r}") from e


# ------------------------------------------------------------------------------------------------
# Pushability
# ------------------------------------------------------------------------------------------------


def feature_pushability(
    feature: str,
    fieldmap: Mapping[str, Any],
    catalog: FeatureCatalog | None = None,
    admitted: Collection[str] | None = None,
) -> tuple[bool, str]:
    """``(True, "")`` if ``feature`` may be pushed into BQL, else ``(False, reason)``."""
    catalog = catalog or default_catalog()
    if feature not in catalog:
        return False, f"unknown feature '{feature}'"
    entry = (fieldmap.get("features") or {}).get(feature)
    if not isinstance(entry, Mapping):
        return False, "no Bloomberg mapping in the field map"
    status = entry.get("status")
    expr = entry.get("expression")
    if not expr:
        return False, f"no BQL expression in the field map (status {status}); evaluated locally"
    if status not in PUSHABLE_STATUSES:
        return False, f"status '{status}': only confirmed/corrected expressions are pushed (preflight, then admit it)"
    if entry.get("pushdown") is False:
        return False, "pushdown disabled in the field map (admit the item in BQLX, then set pushdown=true)"
    if entry.get("units_verified") is False:
        return False, "units unverified: z-score only until admitted (ADR graft 1)"
    cat_unit = entry.get("catalog_unit")
    if cat_unit is not None and cat_unit != catalog[feature].unit:
        return False, f"field-map catalog_unit '{cat_unit}' != catalog unit '{catalog[feature].unit}'"
    try:
        _, refs = expand_refs(str(expr), fieldmap)
    except FieldMapError as e:
        return False, f"invalid field-map expression: {e}"
    bad = sorted(n for n, st in refs.items() if st not in PUSHABLE_STATUSES)
    if bad:
        return False, f"depends on non-admissible item(s) {', '.join('#' + b for b in bad)}"
    if catalog[feature].dtype == "number":
        ts = entry.get("threshold_scale")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(ts) or ts == 0:
            return False, "missing or invalid threshold_scale in the field map"
    if admitted is not None and feature not in admitted:
        return False, "not in the field-admission log (gate G5)"
    return True, ""


# ------------------------------------------------------------------------------------------------
# Compiler
# ------------------------------------------------------------------------------------------------


@dataclass
class _Ctx:
    fieldmap: Mapping[str, Any]
    catalog: FeatureCatalog
    admitted: Collection[str] | None

    def entry(self, feature: str) -> Mapping[str, Any]:
        return (self.fieldmap.get("features") or {})[feature]

    def stage(self, feature: str) -> str:
        st = self.entry(feature).get("stage")
        if st in ("static", "series"):
            return st
        return "static" if self.catalog[feature].source in _STATIC_SOURCES else "series"


def _compile_condition(cond: Condition, ctx: _Ctx) -> tuple[str | None, str, str, list[str]]:
    """``(fragment | None, residual reason, stage, let names)``."""
    is_category = cond.feature in ctx.catalog and ctx.catalog[cond.feature].dtype == "category"
    errs = [] if is_category else cond.structural_errors()  # the engine reads category '==' from 'values'
    if errs:
        return None, "structurally invalid: " + "; ".join(errs), "", []
    ok, why = feature_pushability(cond.feature, ctx.fieldmap, ctx.catalog, ctx.admitted)
    if not ok:
        return None, why, "", []
    fdef = ctx.catalog[cond.feature]
    entry = ctx.entry(cond.feature)
    var = f"#{cond.feature}"
    stage = ctx.stage(cond.feature)

    if is_category:
        if cond.other_feature is not None:
            return None, "category features cannot be compared to another feature", "", []
        if cond.op not in ("==", "in"):
            return None, f"'{cond.op}' on a category is evaluated locally (no '!=' / 'not' / 'or' in the reference string grammar)", "", []
        if cond.values:
            labels = list(cond.values)
        elif cond.value is not None:
            labels = [f"{cond.value:g}"]
        else:
            labels = []
        if len(labels) != 1:
            return None, "a multi-label 'in' needs infix 'or' (not in the reference string grammar): evaluated locally", "", []
        allowed = entry.get("allowed_values") or []
        canon = {str(a).strip().casefold(): str(a) for a in allowed}
        label = canon.get(str(labels[0]).strip().casefold())
        if label is None:
            reason = ("no closed allowed_values list in the field map" if not allowed
                      else f"label {labels[0]!r} is not in the field map's allowed_values")
            return None, reason + " (spec text is never interpolated into BQL)", "", []
        try:
            return f"{var}=={bql_quote(label)}", "", stage, [cond.feature]
        except CompileError as e:
            return None, f"cannot quote the label: {e}", "", []

    if fdef.dtype != "number":
        return None, f"{fdef.dtype} features are evaluated locally", "", []
    if cond.op not in NUMERIC_PUSH_OPS:
        return None, f"numeric '{cond.op}' is evaluated locally (tolerance semantics)", "", []

    ts = _dec(entry["threshold_scale"])

    def scaled(v: float | None) -> Decimal:
        if v is None or not math.isfinite(float(v)):
            raise CompileError("non-finite threshold")
        return _dec(v) * ts

    try:
        if cond.op == "between":
            lo, hi = scaled(cond.value), scaled(cond.value_high)
            if ts < 0:
                lo, hi = hi, lo
            return f"{var} >= {bql_number(lo)} and {var} <= {bql_number(hi)}", "", stage, [cond.feature]
        op = cond.op if ts > 0 else _FLIP[cond.op]
        if cond.other_feature is not None:
            ok2, why2 = feature_pushability(cond.other_feature, ctx.fieldmap, ctx.catalog, ctx.admitted)
            if not ok2:
                return None, f"other feature '{cond.other_feature}': {why2}", "", []
            if ctx.catalog[cond.other_feature].dtype != "number":
                return None, "other_feature must be numeric", "", []
            ts_b = _dec(ctx.entry(cond.other_feature)["threshold_scale"])
            if ts <= 0 or ts_b <= 0 or not math.isfinite(cond.multiplier):
                return None, "feature-vs-feature comparison needs positive scales and a finite multiplier", "", []
            mult = _dec(cond.multiplier) * ts / ts_b
            rhs = f"#{cond.other_feature}" if mult == 1 else f"{bql_number(mult)}*#{cond.other_feature}"
            stage2 = "series" if "series" in (stage, ctx.stage(cond.other_feature)) else "static"
            return f"{var} {op} {rhs}", "", stage2, [cond.feature, cond.other_feature]
        return f"{var} {op} {bql_number(scaled(cond.value))}", "", stage, [cond.feature]
    except CompileError as e:
        return None, f"cannot render a literal: {e}", "", []


def compile_screen(
    spec: ScreenSpec,
    fieldmap: Mapping[str, Any],
    as_of: date | datetime | str,
    *,
    universe_expr: str | None = None,
    catalog: FeatureCatalog | None = None,
    admitted: Collection[str] | None = None,
    benchmark: str | None = None,
) -> CompiledQuery:
    """Compile ``spec`` into one BQL request anchored to ``as_of`` (see module docstring).

    Args:
        spec: the screen.
        fieldmap: the Bloomberg field map (``load_fieldmap('bloomberg')``).
        as_of: absolute as-of date; every date in the query derives from it.
        universe_expr: replaces ``equitiesuniv(['ACTIVE','PRIMARY'])`` (placeholders allowed, e.g.
            ``members('RAY Index', dates='{as_of}')`` for a point-in-time backtest universe).
        catalog: feature catalog (default catalog).
        admitted: optional G5 admission set; when given, only these features are pushed.
        benchmark: security for ``{benchmark}`` placeholders (default: the map's benchmark).

    Raises :class:`CompileError` when nothing can wrap ``equitiesuniv`` in ``filter()``.
    """
    a = _as_date(as_of)
    catalog = catalog or default_catalog()
    ctx = _Ctx(fieldmap, catalog, admitted)
    bench = benchmark or str((fieldmap.get("benchmark") or {}).get("security") or "SPX Index")
    uni = fieldmap.get("universe") or {}

    pushed: list[Condition] = []
    residual: list[Condition] = []
    reasons: dict[str, str] = {}
    inner: list[str] = []
    outer: list[str] = []
    let_names: list[str] = []
    universe_conditions: list[str] = []
    universe_residual: list[str] = []
    warnings: list[str] = []

    def place(frag: str, stage: str, names: list[str]) -> None:
        (inner if stage == "static" else outer).append(frag)
        for n in names:
            if n not in let_names:
                let_names.append(n)

    # ---- universe ----------------------------------------------------------------------------
    if universe_expr:
        base = render_template(universe_expr, a, benchmark=bench)
    else:
        b = uni.get("base") or {}
        if not b.get("expression") or b.get("status") not in PUSHABLE_STATUSES:
            raise CompileError("field map has no admissible universe.base expression")
        base = render_template(str(b["expression"]), a, benchmark=bench)
    is_equitiesuniv = base.replace(" ", "").startswith("equitiesuniv(")

    u = spec.universe
    if u.country:
        c_entry = uni.get("country") or {}
        country = str(u.country).strip().upper()
        label = f"country == {country}"
        if not re.fullmatch(r"[A-Z]{2}", country):
            universe_residual.append(label)
            reasons[label] = "country is not an ISO alpha-2 code"
        elif not c_entry.get("expression") or c_entry.get("status") not in PUSHABLE_STATUSES:
            universe_residual.append(label)
            reasons[label] = "no admissible country item in the field map"
        else:
            item = render_template(str(c_entry["expression"]), a, benchmark=bench)
            inner.append(f"{item}=={bql_quote(country)}")
            universe_conditions.append(f"{label} (pushed as {item}=={country!r}: country of risk, a proxy for the listing country; re-checked locally)")
    if u.security_types:
        label = f"security_type in [{', '.join(u.security_types)}]"
        universe_residual.append(label)
        reasons[label] = "no verified BQL security-type item (ADR/REIT exclusion not yet possible in BQL)"
    for col, floor, src in (("price", u.min_price, "min_price"),
                            ("avg_dollar_volume_20d_usd_mn", u.min_avg_dollar_volume_usd_mn, "min_avg_dollar_volume_usd_mn")):
        if floor is None:
            continue
        cond = Condition(feature=col, op=">=", value=float(floor), rationale=f"UniverseSpec.{src}")
        frag, why, stage, names = _compile_condition(cond, ctx)
        if frag is None:
            universe_residual.append(cond.describe())
            reasons[cond.describe()] = why
        else:
            place(frag, stage, names)
            universe_conditions.append(f"{cond.describe()} (UniverseSpec.{src})")
    if u.exclude_sectors:
        label = f"gics_sector not in [{', '.join(u.exclude_sectors)}]"
        universe_residual.append(label)
        reasons[label] = "exclusion needs '!=' / 'not' (not in the reference string grammar): evaluated locally"

    # ---- conditions --------------------------------------------------------------------------
    for cond in spec.conditions:
        frag, why, stage, names = _compile_condition(cond, ctx)
        if frag is None:
            residual.append(cond)
            reasons[cond.describe()] = why
        else:
            pushed.append(cond)
            place(frag, stage, names)
    residual_groups = [list(g) for g in spec.any_of]
    for g in residual_groups:
        label = "any of: " + " | ".join(c.describe() for c in g)
        reasons[label] = "OR groups are evaluated locally (infix 'or' is not in the reference string grammar; pushing part of a group would drop names)"

    # ---- assemble ----------------------------------------------------------------------------
    for_clause = base
    if inner:
        for_clause = f"filter({for_clause}, {' and '.join(inner)})"
    if outer:
        for_clause = f"filter({for_clause}, {' and '.join(outer)})"
    if is_equitiesuniv and not (inner or outer):
        raise CompileError("equitiesuniv(...) must be wrapped in filter() but nothing could be pushed "
                           "(set UniverseSpec.country or pass a universe_expr such as members(...))")

    get_names = list(let_names)
    lets: dict[str, str] = {}
    for n in let_names:
        expanded, _ = expand_refs(str(ctx.entry(n)["expression"]), fieldmap)
        lets[n] = render_template(expanded, a, benchmark=bench)
    if not lets:
        probe = uni.get("probe") or {}
        if not probe.get("expression") or probe.get("status") not in PUSHABLE_STATUSES:
            raise CompileError("nothing to get(): no pushed feature and no admissible universe.probe item")
        lets["probe"] = render_template(str(probe["expression"]), a, benchmark=bench)
        get_names = ["probe"]

    let_clause = "let(" + " ".join(f"#{n}={e};" for n, e in lets.items()) + ")"
    query = f"{let_clause} get({', '.join('#' + n for n in get_names)}) for({for_clause})"
    for banned in ("grouprank", "groupzscore", "groupsort"):
        if banned in query:  # pragma: no cover - only reachable through a hostile field map
            raise CompileError(f"compiled query contains '{banned}': ranking must happen after every filter, locally")

    if pushed or universe_conditions:
        warnings.append("pushed items still need a field_validation_log entry (gate G5) before production use: "
                        "run BloombergProvider.verify_fields() on a known security and record the values")
    helpers = fieldmap.get("helpers") or {}
    for n in let_names:
        notes = [str(v) for v in ctx.entry(n).get("verify") or []]
        _, refs = expand_refs(str(ctx.entry(n)["expression"]), fieldmap)
        for r in refs:
            notes += [str(v) for v in (helpers.get(r) or {}).get("verify") or []]
        if notes:
            warnings.append(f"#{n} is pushed with open items to check in BQLX: " + "; ".join(dict.fromkeys(notes)))
    if is_equitiesuniv:
        warnings.append("equitiesuniv(['ACTIVE','PRIMARY']) is today's universe (equitiesuniv(dates=) is UNVERIFIED): "
                        "for a historical as_of pass universe_expr=\"members('<index>', dates='{as_of}')\"")

    return CompiledQuery(
        query=query,
        pushed=pushed,
        residual=residual,
        universe_conditions=universe_conditions,
        universe_residual=universe_residual,
        residual_groups=residual_groups,
        reasons=reasons,
        lets=lets,
        warnings=warnings,
        as_of=a,
        fieldmap_digest=digest(fieldmap),
    )
