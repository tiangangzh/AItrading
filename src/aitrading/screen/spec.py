"""Typed screen specification.

This is the contract between the natural-language layer (an LLM emits a ScreenSpec via structured
output) and the deterministic engines (local pandas evaluation, or push-down compilers that emit
vendor queries such as BQL or LSEG SCREEN expressions).

Semantics
---------
* ``conditions`` are ANDed.
* ``any_of`` is a list of groups; each group is ORed internally and the groups are ANDed with
  ``conditions``. (Two levels are enough for real screens and keep the JSON schema non-recursive,
  which matters for structured outputs.)
* A condition compares a feature to a constant (``value``), a range (``value``..``value_high``,
  inclusive), a set of category labels (``values``), or another feature scaled by ``multiplier``:
  ``feature <op> multiplier * other_feature``.
* A NaN feature value never satisfies a condition (missing data excludes, it does not include).
* Feature names and units come from the feature catalog (``aitrading.screen.catalog``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.screen.catalog import FeatureCatalog

ComparisonOp = Literal[">", ">=", "<", "<=", "==", "!="]
Op = Literal[">", ">=", "<", "<=", "==", "!=", "between", "in", "not_in"]


class Condition(BaseModel):
    feature: str = Field(description="Feature name from the catalog, e.g. 'price_vs_sma_200_pct'.")
    op: Op
    value: float | None = Field(None, description="Constant for comparison ops; lower bound for 'between'.")
    value_high: float | None = Field(None, description="Upper bound for 'between' (inclusive).")
    values: list[str] | None = Field(None, description="Category labels for 'in' / 'not_in'.")
    other_feature: str | None = Field(
        None, description="Compare against another feature instead of a constant: feature <op> multiplier * other_feature."
    )
    multiplier: float = Field(1.0, description="Scale applied to other_feature.")
    rationale: str = Field("", description="Which part of the observation this condition encodes.")

    def describe(self) -> str:
        if self.op == "between":
            return f"{self.feature} between {_fmt(self.value)} and {_fmt(self.value_high)}"
        if self.op in ("in", "not_in"):
            vals = ", ".join(self.values or [])
            return f"{self.feature} {'in' if self.op == 'in' else 'not in'} [{vals}]"
        if self.other_feature:
            rhs = self.other_feature if self.multiplier == 1.0 else f"{_fmt(self.multiplier)} x {self.other_feature}"
            return f"{self.feature} {self.op} {rhs}"
        return f"{self.feature} {self.op} {_fmt(self.value)}"

    def features(self) -> set[str]:
        return {self.feature} | ({self.other_feature} if self.other_feature else set())

    def structural_errors(self) -> list[str]:
        """Errors that do not depend on the catalog (operand shape for the operator)."""
        errs: list[str] = []
        if self.op == "between":
            if self.value is None or self.value_high is None:
                errs.append(f"{self.describe()}: 'between' needs value and value_high")
            elif self.value > self.value_high:
                errs.append(f"{self.describe()}: value must be <= value_high")
        elif self.op in ("in", "not_in"):
            if not self.values:
                errs.append(f"{self.feature} {self.op}: needs a non-empty 'values' list")
        else:
            has_const = self.value is not None
            has_other = self.other_feature is not None
            if has_const == has_other:
                errs.append(f"{self.feature} {self.op}: set exactly one of 'value' or 'other_feature'")
        return errs


class RankFactor(BaseModel):
    feature: str
    direction: Literal["higher_is_better", "lower_is_better"]
    weight: float = Field(1.0, gt=0, description="Relative weight; weights are normalised to sum to 1.")
    rationale: str = ""


class UniverseSpec(BaseModel):
    country: str = Field("US", description="Primary-listing country (ISO alpha-2).")
    security_types: list[str] = Field(default_factory=lambda: ["common_stock"])
    min_price: float | None = Field(5.0, description="Exclude names below this USD price (penny-stock filter).")
    min_avg_dollar_volume_usd_mn: float | None = Field(5.0, description="Liquidity floor: 20-day average dollar volume, USD mn.")
    exclude_sectors: list[str] = Field(default_factory=list, description="GICS sectors to exclude.")


class ScreenSpec(BaseModel):
    name: str = Field(description="Short slug-like name for the screen.")
    observation: str = Field(description="The investment observation, verbatim.")
    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    conditions: list[Condition] = Field(description="ANDed conditions.")
    any_of: list[list[Condition]] = Field(default_factory=list, description="Groups of ORed conditions; groups are ANDed.")
    ranking: list[RankFactor] = Field(description="Factors used to rank survivors.")
    top_n: int = Field(10, ge=1, le=100)
    assumptions: list[str] = Field(default_factory=list, description="Interpretive choices made translating the observation.")
    unsupported_requests: list[str] = Field(default_factory=list, description="Parts of the observation that no catalog feature can express.")

    def all_conditions(self) -> list[Condition]:
        return [*self.conditions, *(c for group in self.any_of for c in group)]

    def features(self) -> set[str]:
        out: set[str] = set()
        for c in self.all_conditions():
            out |= c.features()
        out |= {f.feature for f in self.ranking}
        return out

    def validate_against(self, catalog: "FeatureCatalog") -> list[str]:
        """Return a list of human-readable errors; empty means the spec is executable."""
        errs: list[str] = []
        for c in self.all_conditions():
            errs.extend(c.structural_errors())
            for name in c.features():
                if name not in catalog:
                    errs.append(f"unknown feature '{name}'" + _suggest(name, catalog))
            if c.feature in catalog:
                fdef = catalog[c.feature]
                if fdef.dtype == "category" and c.op not in ("in", "not_in", "==", "!="):
                    errs.append(f"{c.describe()}: category feature '{c.feature}' only supports in/not_in")
                if fdef.dtype != "category" and c.op in ("in", "not_in"):
                    errs.append(f"{c.describe()}: 'in'/'not_in' only apply to category features")
                if c.other_feature and c.other_feature in catalog:
                    odef = catalog[c.other_feature]
                    if odef.dtype == "category" or fdef.dtype == "category":
                        errs.append(f"{c.describe()}: cannot compare category features numerically")
                    elif odef.unit != fdef.unit:
                        errs.append(f"{c.describe()}: unit mismatch ({fdef.unit} vs {odef.unit})")
        for g in self.any_of:
            if len(g) < 2:
                errs.append("each any_of group needs at least two alternatives (use 'conditions' for a single one)")
        if not self.ranking:
            errs.append("ranking needs at least one factor")
        for f in self.ranking:
            if f.feature not in catalog:
                errs.append(f"unknown ranking feature '{f.feature}'" + _suggest(f.feature, catalog))
            elif catalog[f.feature].dtype == "category":
                errs.append(f"cannot rank on category feature '{f.feature}'")
        return errs


def _fmt(x: float | None) -> str:
    if x is None:
        return "?"
    return f"{x:g}"


def _suggest(name: str, catalog: "FeatureCatalog") -> str:
    import difflib

    close = difflib.get_close_matches(name, list(catalog.names()), n=3, cutoff=0.5)
    return f" (did you mean: {', '.join(close)}?)" if close else ""
