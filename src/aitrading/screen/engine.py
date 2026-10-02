"""Local, deterministic evaluation of a ScreenSpec on a feature frame.

The feature frame is a DataFrame indexed by ticker with every catalog feature column plus the raw
reference columns ``name``, ``security_type`` and ``country`` from the universe. This engine is
the source of truth for screen membership: a vendor push-down may only narrow the frame first.

Semantics (see ``aitrading.screen.spec``)
-----------------------------------------
* A missing value never satisfies a condition, including ``!=`` and ``not_in``: missing data
  excludes. Non-finite numbers (+/-inf) count as missing.
* ``between`` is inclusive on both ends; ``>``, ``>=``, ``<``, ``<=`` are exact comparisons.
* Numeric ``==`` / ``!=`` use ``np.isclose`` (default tolerances), so 1.0 / 0.0 bool features and
  float noise compare as expected.
* Category features (catalog dtype ``category``, or a non-numeric column not in the catalog) and
  every ``in`` / ``not_in`` compare labels as stripped, case-insensitive strings. A category
  ``==`` / ``!=`` uses ``values`` when given (equal to any / none of them), else ``value`` as text.
* ``other_feature`` comparisons evaluate ``feature <op> multiplier * other_feature``; a missing
  value on either side excludes the name.
* Universe filters run first (country, security types, minimum price, liquidity floor, excluded
  sectors; a None threshold or an empty list disables a filter), then ``conditions`` in spec order,
  then each ``any_of`` group. Each step is one FunnelStep: ``passed_alone`` counts the whole frame,
  ``remaining`` is cumulative, ``missing_data`` counts names that were still remaining and were
  dropped by this step because an input was missing. For an any_of group that is a name for which
  no alternative passed and at least one alternative had missing data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aitrading.core.models import FunnelStep
from aitrading.screen.catalog import FeatureCatalog, default_catalog
from aitrading.screen.spec import Condition, ScreenSpec, UniverseSpec

PRICE_COLUMN = "price"
LIQUIDITY_COLUMN = "avg_dollar_volume_20d_usd_mn"
SECTOR_COLUMN = "gics_sector"
COUNTRY_COLUMN = "country"
SECURITY_TYPE_COLUMN = "security_type"

_Step = tuple[str, pd.Series, pd.Series]  # (label, passes, excluded for missing data)


class ScreenValidationError(ValueError):
    """The spec cannot be executed on this frame; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid screen: " + "; ".join(self.errors))


@dataclass
class ScreenOutcome:
    survivors: list[str]  # sorted tickers that passed every step
    mask: pd.Series  # bool, indexed like the frame
    funnel: list[FunnelStep]
    universe_size: int  # rows in the frame before any filter


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ScreenValidationError([f"feature frame has no column '{name}'"])
    return frame[name]


def _numeric(col: pd.Series) -> pd.Series:
    x = pd.to_numeric(col, errors="coerce").astype("float64")
    return x.where(np.isfinite(x))


def _labels(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(normalised text, missing mask) for a label column."""
    text = col.astype(object).where(col.notna(), "").astype(str).str.strip().str.casefold()
    return text, text.eq("")


def _norm(values: list[str]) -> list[str]:
    return [str(v).strip().casefold() for v in values]


def _is_category(feature: str, col: pd.Series, catalog: FeatureCatalog) -> bool:
    if feature in catalog:
        return catalog[feature].dtype == "category"
    return not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))


def _evaluate(cond: Condition, frame: pd.DataFrame, catalog: FeatureCatalog) -> tuple[pd.Series, pd.Series]:
    """(passes, missing) boolean Series indexed like ``frame``."""
    col = _column(frame, cond.feature)

    if cond.op in ("in", "not_in") or _is_category(cond.feature, col, catalog):
        if cond.op not in ("in", "not_in", "==", "!="):
            raise ScreenValidationError([f"{cond.describe()}: category feature '{cond.feature}' only supports in/not_in/==/!="])
        if cond.other_feature is not None:
            raise ScreenValidationError([f"{cond.describe()}: cannot compare category features"])
        if not cond.values and (cond.op in ("in", "not_in") or cond.value is None):
            raise ScreenValidationError([f"{cond.feature} {cond.op}: needs a non-empty 'values' list"])
        text, missing = _labels(col)
        labels = _norm(cond.values) if cond.values else _norm([f"{cond.value:g}"])
        hit = text.isin(labels)
        passes = (hit if cond.op in ("in", "==") else ~hit) & ~missing
        return passes, missing

    errs = cond.structural_errors()
    if errs:
        raise ScreenValidationError(errs)
    x = _numeric(col)
    missing = x.isna()
    if cond.op == "between":
        return x.ge(cond.value) & x.le(cond.value_high), missing
    if cond.other_feature is not None:
        rhs = cond.multiplier * _numeric(_column(frame, cond.other_feature))
        missing = missing | rhs.isna()
    else:
        rhs = pd.Series(float(cond.value), index=frame.index)
    xa, ra = x.to_numpy(), rhs.to_numpy()
    with np.errstate(invalid="ignore"):
        if cond.op == ">":
            res = xa > ra
        elif cond.op == ">=":
            res = xa >= ra
        elif cond.op == "<":
            res = xa < ra
        elif cond.op == "<=":
            res = xa <= ra
        elif cond.op == "==":
            res = np.isclose(xa, ra)
        else:  # "!="
            res = ~np.isclose(xa, ra)
    return pd.Series(res, index=frame.index) & ~missing, missing


def evaluate_condition(cond: Condition, frame: pd.DataFrame, *, catalog: FeatureCatalog | None = None) -> pd.Series:
    """Boolean mask (indexed like ``frame``) of the rows satisfying ``cond``; missing data is False."""
    passes, _ = _evaluate(cond, frame, catalog or default_catalog())
    return passes.astype(bool)


def _universe_steps(universe: UniverseSpec, frame: pd.DataFrame) -> list[_Step]:
    steps: list[_Step] = []

    def labels_step(label: str, column: str, wanted: list[str], keep_if_in: bool) -> None:
        text, missing = _labels(_column(frame, column))
        hit = text.isin(_norm(wanted))
        steps.append((label, (hit if keep_if_in else ~hit) & ~missing, missing))

    def floor_step(column: str, floor: float) -> None:
        x = _numeric(_column(frame, column))
        steps.append((f"{column} >= {floor:g}", x.ge(floor), x.isna()))

    if universe.country:
        labels_step(f"{COUNTRY_COLUMN} == {universe.country}", COUNTRY_COLUMN, [universe.country], True)
    if universe.security_types:
        types = ", ".join(universe.security_types)
        labels_step(f"{SECURITY_TYPE_COLUMN} in [{types}]", SECURITY_TYPE_COLUMN, universe.security_types, True)
    if universe.min_price is not None:
        floor_step(PRICE_COLUMN, universe.min_price)
    if universe.min_avg_dollar_volume_usd_mn is not None:
        floor_step(LIQUIDITY_COLUMN, universe.min_avg_dollar_volume_usd_mn)
    if universe.exclude_sectors:
        sectors = ", ".join(universe.exclude_sectors)
        labels_step(f"{SECTOR_COLUMN} not in [{sectors}]", SECTOR_COLUMN, universe.exclude_sectors, False)
    return steps


def _universe_columns(universe: UniverseSpec) -> list[str]:
    needed = [
        (bool(universe.country), COUNTRY_COLUMN),
        (bool(universe.security_types), SECURITY_TYPE_COLUMN),
        (universe.min_price is not None, PRICE_COLUMN),
        (universe.min_avg_dollar_volume_usd_mn is not None, LIQUIDITY_COLUMN),
        (bool(universe.exclude_sectors), SECTOR_COLUMN),
    ]
    return [col for on, col in needed if on]


def _run_steps(steps: list[_Step], index: pd.Index) -> tuple[pd.Series, list[FunnelStep]]:
    remaining = pd.Series(True, index=index)
    funnel: list[FunnelStep] = []
    for label, passes, missing in steps:
        dropped_missing = int((remaining & missing & ~passes).sum())
        remaining = remaining & passes
        funnel.append(
            FunnelStep(label=label, passed_alone=int(passes.sum()), remaining=int(remaining.sum()), missing_data=dropped_missing)
        )
    return remaining.astype(bool), funnel


def apply_universe(universe: UniverseSpec, frame: pd.DataFrame) -> tuple[pd.Series, list[FunnelStep]]:
    """Mask of rows passing the universe filters and one FunnelStep per enabled filter."""
    return _run_steps(_universe_steps(universe, frame), frame.index)


def _frame_errors(spec: ScreenSpec, frame: pd.DataFrame, catalog: FeatureCatalog) -> list[str]:
    errs: list[str] = []
    if frame.index.has_duplicates:
        dupes = sorted({str(t) for t in frame.index[frame.index.duplicated()]})
        errs.append(f"feature frame has duplicate tickers: {', '.join(dupes[:10])}")
    needed = sorted(f for f in spec.features() if f in catalog) + _universe_columns(spec.universe)
    missing = [c for c in dict.fromkeys(needed) if c not in frame.columns]
    if missing:
        errs.append(f"feature frame is missing columns: {', '.join(missing)}")
    return errs


def run_screen(spec: ScreenSpec, frame: pd.DataFrame, catalog: FeatureCatalog | None = None) -> ScreenOutcome:
    """Validate ``spec`` against the catalog and the frame, then evaluate it step by step.

    Raises ScreenValidationError listing every problem when the spec is not executable.
    """
    catalog = catalog or default_catalog()
    errors = spec.validate_against(catalog) + _frame_errors(spec, frame, catalog)
    if errors:
        raise ScreenValidationError(errors)

    steps = _universe_steps(spec.universe, frame)
    for cond in spec.conditions:
        passes, missing = _evaluate(cond, frame, catalog)
        steps.append((cond.describe(), passes, missing))
    for group in spec.any_of:
        results = [_evaluate(cond, frame, catalog) for cond in group]
        passes = pd.concat([p for p, _ in results], axis=1).any(axis=1)
        any_missing = pd.concat([m for _, m in results], axis=1).any(axis=1)
        label = "any of: " + " | ".join(cond.describe() for cond in group)
        steps.append((label, passes, any_missing & ~passes))

    mask, funnel = _run_steps(steps, frame.index)
    survivors = sorted(str(t) for t in frame.index[mask.to_numpy()])
    return ScreenOutcome(survivors=survivors, mask=mask, funnel=funnel, universe_size=len(frame))
