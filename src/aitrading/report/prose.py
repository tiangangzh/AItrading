"""Numbers in free text (Claude's prose) checked against the numbers the engine actually produced.

The interpreter is told to use only numbers from the backtest result, and the numbers it *cites*
(``cited_metrics``) are verified programmatically. Its prose (summary, key findings) is not covered
by that check - an interpretation can cite nothing and still state "Sharpe 1.4" - so the reports
run this second, looser check before showing it:

* :func:`stat_numbers` finds the statistic-like numbers in a text: decimals (``1.4``, ``0.42``) and
  numbers with a unit (``9%``, ``3.2x``, ``25 bps``, ``0.6pp``, ``$2.5bn``). Bare integers (years,
  counts, look-backs such as ``12-1`` or ``200-day``) and numbers glued to letters (``12m``,
  ``Q3``, ``FF5``) are not treated as claims.
* :func:`unverified_numbers` returns those that match none of the known values: a match allows
  rounding to the digits written (``1.4`` matches 1.35 to 1.45), ignores the sign ("fell 28%" for
  -28.1) and accepts the fraction / percent and million / billion scalings.
* :func:`interpretation_unverified_numbers` applies it to a ``BacktestResult``'s interpretation,
  with every number of the result (statistics, regression, quantiles, factor checks, spec, data
  usage, warnings) and a few derived ones (period length, strategy minus benchmark, cost drag) as
  the known values.

A number that is not found is *unverified*, not necessarily wrong (it may be derived from two
results); the reports label it that way.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "StatNumber",
    "stat_numbers",
    "numbers_in",
    "numeric_values",
    "unverified_numbers",
    "interpretation_texts",
    "interpretation_unverified_numbers",
]

# A number not preceded by a word character, a dot, '^' or '/' (so not part of an identifier, a
# decimal, an exponent or a URL), with an optional unit right after it.
_TOKEN = re.compile(
    r"(?<![\w.^/])"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?:\.(?P<frac>\d+))?"
    r"(?!\.\d)"
    r"(?P<unit>\s?%|\s?[A-Za-z×]+)?"
)
_PCT_UNITS = {"%", "percent", "pct"}
_PLAIN_UNITS = {"x", "×", "pp", "bp", "bps"}
_BIG_UNITS = {"bn", "b", "billion", "mn", "mm", "million"}
_SCALES = (1.0, 100.0, 0.01)
_BIG_SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001)


@dataclass(frozen=True)
class StatNumber:
    text: str  # as written, e.g. "1.4", "9%", "3.2x"
    value: float  # absolute value of the number as written (units not applied)
    decimals: int
    big_unit: bool  # million / billion: also compare at x1000 / /1000


def _unit(raw: str | None) -> tuple[str | None, bool]:
    """(normalised unit or None, attached-without-space)."""
    if not raw:
        return None, False
    attached = not raw[0].isspace()
    return raw.strip().lower(), attached


def _scan(text: str) -> Iterable[tuple[re.Match[str], str | None, bool]]:
    for m in _TOKEN.finditer(text or ""):
        unit, attached = _unit(m.group("unit"))
        yield m, unit, attached


def stat_numbers(text: str) -> list[StatNumber]:
    """The statistic-like numbers in ``text`` (see the module docstring), in order."""
    out: list[StatNumber] = []
    for m, unit, attached in _scan(text):
        frac = m.group("frac")
        known_unit = unit in _PCT_UNITS or unit in _PLAIN_UNITS or unit in _BIG_UNITS
        if unit is not None and not known_unit:
            if attached:  # '12m', '3rd', '50d': an identifier or a period, not a claim
                continue
            unit = None  # the next word ("1.4 with ...")
        if frac is None and unit is None:
            continue
        value = float(m.group("int").replace(",", "") + ("." + frac if frac else ""))
        end = m.end() if unit is not None else (m.end("frac") if frac else m.end("int"))
        shown = m.string[m.start():end].strip()
        out.append(StatNumber(text=shown, value=abs(value), decimals=len(frac or ""), big_unit=unit in _BIG_UNITS))
    return out


def numbers_in(text: str) -> list[float]:
    """Every number in ``text`` (integers too), as written - used to collect known values."""
    vals = []
    for m, _unit_, _attached in _scan(text):
        frac = m.group("frac")
        vals.append(float(m.group("int").replace(",", "") + ("." + frac if frac else "")))
    return vals


def numeric_values(obj: Any, *, skip_keys: Iterable[str] = ()) -> list[float]:
    """Every finite number in a nested structure (dicts, lists, models' ``model_dump``) plus the
    numbers written inside its strings."""
    skip = set(skip_keys)
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            v = float(x)
            if math.isfinite(v):
                out.append(v)
        elif isinstance(x, str):
            out.extend(numbers_in(x))
        elif isinstance(x, Mapping):
            for k, v in x.items():
                if k not in skip:
                    walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
        elif hasattr(x, "model_dump"):
            walk(x.model_dump(mode="json"))
        else:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return
            if math.isfinite(v):
                out.append(v)

    walk(obj)
    return out


def _matches(n: StatNumber, known: Sequence[float]) -> bool:
    tol = 0.5 * 10.0 ** (-n.decimals) * (1 + 1e-6) + 1e-12  # the number as rounded to the digits written
    scales = _BIG_SCALES if n.big_unit else _SCALES
    for k in known:
        a = abs(k)
        for s in scales:
            if abs(n.value - a * s) <= tol:
                return True
    return False


def unverified_numbers(texts: Iterable[str | None], known: Iterable[float]) -> list[str]:
    """Statistic-like numbers in ``texts`` that match none of ``known`` (as written, de-duplicated)."""
    known_vals = [abs(float(k)) for k in known if isinstance(k, (int, float)) and not isinstance(k, bool) and math.isfinite(float(k))]
    out: list[str] = []
    for text in texts:
        for n in stat_numbers(text or ""):
            if n.text not in out and not _matches(n, known_vals):
                out.append(n.text)
    return out


# ------------------------------------------------------------------------------------------------
# Backtest interpretations
# ------------------------------------------------------------------------------------------------


def interpretation_texts(interp: Any) -> list[str]:
    """The interpreter's free-text fields whose numbers are checked (summary, key findings)."""
    if interp is None:
        return []
    return [str(interp.summary or "")] + [str(x) for x in (interp.key_findings or [])]


def interpretation_unverified_numbers(result: Any) -> list[str]:
    """Numbers in a backtest interpretation's summary / key findings that are not in the result
    (statistics, regression, quantiles, factor checks, spec, data usage, warnings, period)."""
    interp = getattr(result, "interpretation", None)
    texts = interpretation_texts(interp)
    if not any(stat_numbers(t) for t in texts):
        return []
    known = numeric_values([result.stats, result.regression, result.quantiles, result.factor_checks, result.spec,
                            result.data_usage, result.warnings, result.idea])
    known += _derived_values(result)
    return unverified_numbers(texts, known)


def _derived_values(result: Any) -> list[float]:
    """Numbers an honest interpretation derives from the result: the period in years, the number of
    periods, strategy-minus-benchmark CAGR / Sharpe, and the turnover cost arithmetic."""
    out: list[float] = [float(len(result.dates or []))]
    try:
        out.append((result.end - result.start).days / 365.25)  # "over 8.0 years"
    except (TypeError, AttributeError):
        pass
    st0 = result.stats.get("strategy") if isinstance(getattr(result, "stats", None), dict) else None
    n, ppy = getattr(st0, "n_periods", None), getattr(st0, "periods_per_year", None)
    if isinstance(n, (int, float)) and isinstance(ppy, (int, float)) and ppy > 0:
        out.append(float(n) / float(ppy))  # years of return observations ("only 2.4 years of data")
    st, bench = result.stats.get("strategy"), result.stats.get("benchmark")
    for field in ("cagr_pct", "sharpe", "volatility_pct", "max_drawdown_pct"):
        a, b = getattr(st, field, None), getattr(bench, field, None)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out.append(a - b)
    turn = getattr(st, "avg_turnover_pct", None)
    if isinstance(turn, (int, float)):
        out.append(2 * turn)  # one-way turnover -> share of the book traded
        try:
            from aitrading.strategy.interpret import annual_cost_drag_pct, rebalances_per_year

            n_reb = rebalances_per_year(result)
            costs = (result.spec or {}).get("costs_bps")
            if n_reb and isinstance(costs, (int, float)) and not isinstance(costs, bool):
                out += [float(n_reb), float(annual_cost_drag_pct(turn, n_reb, float(costs)))]
        except Exception:  # noqa: BLE001 - helpers unavailable: fewer derived values, more numbers flagged
            pass
    return [v for v in out if isinstance(v, float) and math.isfinite(v)]
