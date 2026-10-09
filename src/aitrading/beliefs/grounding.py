"""Grounding checks for text a language model writes about the ledger.

Anything an LLM writes that is fed back into later prompts can amplify a hallucination round after
round: a reflection, a lesson, a policy note, a judge's reasoning. So every hypothesis id it cites
must exist, and every number it states must be one the ledger can back up. :func:`check_text`
reports both; callers drop or flag what fails.

Numbers are matched against a set of reference values allowing for rounding to the number of
decimals written (``0.48`` matches 0.4813, ``48%`` matches 0.4813 written as a percentage), with
signs required to agree. Small integers (counts, years written as digits, horizons such as
``12-1``) are not checked: they carry little risk and are everywhere in normal prose.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["GroundingReport", "check_text", "reference_numbers", "HYP_ID_RE"]

HYP_ID_RE = re.compile(r"\bh-[0-9a-f]{12}\b")
_NUM_RE = re.compile(r"(?<![\w.])([-+−]?)(\d+(?:\.\d+)?)(%?)")
_FREE_INTEGERS = 31  # integers up to this (and years 1900-2100) are not checked


@dataclass
class GroundingReport:
    unknown_ids: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unknown_ids and not self.unverified_numbers


def reference_numbers(values: Iterable[float | None]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _decimals(tok: str) -> int:
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _matches(value: float, written: str, refs: list[float], percent: bool) -> bool:
    dec = _decimals(written)
    tol = 0.5 * 10 ** (-dec) + 1e-9
    for r in refs:
        cands = [r * 100.0] if percent else [r, r * 100.0]
        if percent and abs(r) > 1.0:  # a value already in percent units
            cands.append(r)
        for c in cands:
            if abs(c - value) <= tol and (value == 0 or c == 0 or (c > 0) == (value > 0)):
                return True
    return False


def check_text(text: str, *, known_ids: Iterable[str], numbers: Iterable[float]) -> GroundingReport:
    """Unknown hypothesis ids and unverifiable numbers in ``text`` (see the module docstring)."""
    ids = set(known_ids)
    refs = reference_numbers(numbers)
    rep = GroundingReport()
    rep.unknown_ids = sorted({m.group(0) for m in HYP_ID_RE.finditer(text or "")} - ids)
    scrubbed = HYP_ID_RE.sub(" ", text or "")
    bad: list[str] = []
    for m in _NUM_RE.finditer(scrubbed):
        sign, digits, pct = m.groups()
        value = float(digits) * (-1.0 if sign in ("-", "−") else 1.0)
        if "." not in digits and not pct:
            n = int(digits)
            if n <= _FREE_INTEGERS or 1900 <= n <= 2100:
                continue
        if not _matches(value, digits, refs, bool(pct)):
            bad.append(m.group(0).replace("−", "-"))
    rep.unverified_numbers = sorted(set(bad))
    return rep
