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
from typing import Any, Iterable

__all__ = [
    "GroundingReport", "check_text", "reference_numbers", "sanitize_llm_text", "render_safe", "scrub_statement",
    "structured_refs", "leakage_tripwire", "HYP_ID_RE",
]

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


# ------------------------------------------------------------------------------------------------
# Sanitising LLM-written text before it is stored (and therefore shown to later prompts)
# ------------------------------------------------------------------------------------------------

_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‍⁠﻿]")
_RESULT_NUM_RE = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?\s*(?:%|bps?\b|basis points\b)|(?<![\w.])[-+−]?\d+\.\d+")


def sanitize_llm_text(text: str, *, max_chars: int = 500, mask_numbers: bool = False) -> tuple[str, list[str]]:
    """Clean LLM-written text before it is persisted, because it reaches later prompts.

    Returns ``(clean text, issues)``. Cleaning:
    * removes control and zero-width characters, markup-like tags (``<system>``, ``</experience_log>``)
      and links;
    * drops instruction-like sentences (the discovery module's prompt-injection detector);
    * collapses whitespace and truncates to ``max_chars``.

    With ``mask_numbers`` (for mechanism / reasoning prose, which should explain, not report results),
    decimals, percentages and basis points are replaced by ``[n]``. Their values can only be checked
    against the ledger, and an unchecked number repeated in every later prompt amplifies a
    hallucination.
    """
    from aitrading.discovery.extract import _TAG_LIKE, strip_instruction_like
    from aitrading.discovery.textutil import truncate_text

    issues: list[str] = []
    t = _CTRL_RE.sub("", text or "")
    if _TAG_LIKE.search(t):
        issues.append("markup removed")
        t = _TAG_LIKE.sub(" ", t)
    if _URL_RE.search(t):
        issues.append("link removed")
        t = _URL_RE.sub("[link removed]", t)
    t, flagged = strip_instruction_like(t)
    if flagged:
        issues.append(f"{len(flagged)} instruction-like sentence(s) removed")
    if mask_numbers and _RESULT_NUM_RE.search(t):
        issues.append("numbers masked")
        t = _RESULT_NUM_RE.sub("[n]", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        t = truncate_text(t, max_chars)
        issues.append("truncated")
    return t, issues


# ------------------------------------------------------------------------------------------------
# Rendering stored text into prompts, scrubbing statements, structured references, leakage tripwire
# ------------------------------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)")
_SIGNED_NUM_RE = re.compile(r"(?<![\w.])[-+−]\d+(?:\.\d+)?")


def render_safe(text: str, max_chars: int = 300) -> str:
    """Stored text as it may appear inside a prompt block: NFKC-normalised, angle brackets turned into
    ‹ › (so it cannot close or open a block), whitespace and newlines collapsed (so it cannot forge a
    digest row), truncated."""
    import unicodedata

    t = unicodedata.normalize("NFKC", text or "")
    t = _CTRL_RE.sub("", t).replace("<", "‹").replace(">", "›")
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= max_chars else t[: max_chars - 1].rstrip() + "…"


def scrub_statement(statement: str, mechanism: str, *, info_horizon: Any = None) -> tuple[str | None, str, list[str]]:
    """Clean a model-written statement and mechanism before they are stored in the ledger.

    The statement is all or nothing: if anything had to be changed (markup, injection, a hypothesis
    id, a result number, a year at or after ``info_horizon``), ``None`` is returned and the caller
    stores the deterministic ``hypothesis_statement(spec)``. The raw text then lives only in audit
    logs. The mechanism keeps its words, with numbers, ids and late years masked.
    """
    issues: list[str] = []

    def clean(text: str, max_chars: int) -> tuple[str, list[str]]:
        t, found = sanitize_llm_text(text, max_chars=max_chars, mask_numbers=True)
        if HYP_ID_RE.search(t):
            found.append("hypothesis id removed")
            t = HYP_ID_RE.sub("[id]", t)
        if _SIGNED_NUM_RE.search(t):
            found.append("signed number masked")
            t = _SIGNED_NUM_RE.sub("[n]", t)
        if info_horizon is not None:
            def late(m: re.Match) -> str:
                return "[year]" if int(m.group(1)) >= info_horizon.year else m.group(0)
            t2 = _YEAR_RE.sub(late, t)
            if t2 != t:
                found.append("late year masked")
                t = t2
        return t, found

    st, s_issues = clean(statement or "", 300)
    mech, m_issues = clean(mechanism or "", 400)
    issues += [f"statement: {x}" for x in s_issues] + [f"mechanism: {x}" for x in m_issues]
    norm_in = re.sub(r"\s+", " ", (statement or "").strip())
    keep = st if (st and not s_issues and st == norm_in) else None
    return keep, mech, issues


def structured_refs(model: Any, ids: Iterable[str]) -> list[float]:
    """Reference numbers for :func:`check_text`, taken from structured beliefs (never from rendered
    text, whose ids and dates would launder arbitrary digits): every number field of the beliefs of
    ``ids`` plus the base rate."""
    out: list[float] = [model.hyper.m0, model.hyper.s0, getattr(model, "hurdle", 0.3)]
    tested = set(model.tested())
    for b in model.beliefs([i for i in ids if i in tested]):
        out += [b.mean, b.sd, b.p_positive, b.p_hurdle, b.gross_mean, b.cost, b.years, b.oos_years]
        out += [x for x in (b.oos_sharpe, b.last_sharpe) if x is not None]
    return reference_numbers(out)


def leakage_tripwire(prompt: str, ledger: Any, cutoff: Any) -> list[str]:
    """What in ``prompt`` could only be known from data on or after ``cutoff``: ids of hypotheses whose
    ONLY evidence ends on or after it, run / evidence ids of such evidence, and their model-written
    (non-canonical) statements. An empty list means clean."""
    from aitrading.beliefs.semantics import hypothesis_statement

    hits: list[str] = []
    base = getattr(ledger, "base", ledger)
    ev = base.evidence()
    late = [e for e in ev if e.end >= cutoff]
    early_ids = {e.hypothesis_id for e in ev if e.end < cutoff}
    text = prompt or ""
    for e in late:
        if e.evidence_id in text or (e.run_id and e.run_id in text):
            hits.append(f"evidence {e.evidence_id}")
        if e.hypothesis_id not in early_ids:
            if e.hypothesis_id in text:
                hits.append(f"hypothesis {e.hypothesis_id} (only post-cutoff evidence)")
            h = base.hypothesis(e.hypothesis_id)
            if h is not None and len(h.statement) > 20 and h.statement != hypothesis_statement(h.spec) and h.statement in text:
                hits.append(f"statement of {e.hypothesis_id}")
    return sorted(set(hits))


# ------------------------------------------------------------------------------------------------
# Policy directives (the RL loop's in-prompt parameters)
# ------------------------------------------------------------------------------------------------

_HORIZON_EXPR = re.compile(r"(?<![\w.])(\d{1,3})(?:\s*-\s*1)?[\s-]*(?:days?|weeks?|months?|sessions?|years?|month)\b", re.I)
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _text_numbers_ok(text: str) -> list[str]:
    """Reason codes for numbers a directive may not contain: only small integers, horizon expressions
    (an integer <= 252 followed by a day / week / month / session / year word) and '12-1' are allowed."""
    reasons: list[str] = []
    if _YEAR_RE.search(text) or re.search(r"\b\d{4}\b", text):
        reasons.append("year")
    scrubbed = _HORIZON_EXPR.sub(lambda m: " " if int(m.group(1)) <= 252 else m.group(0), text)
    scrubbed = scrubbed.replace("12-1", " ")
    if re.search(r"\d+\.\d+|\d+\s*%|\bbps?\b|basis points|(?<![\w.])[-+−]\d", scrubbed):
        reasons.append("number")
    else:
        for m in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", scrubbed):
            if int(m.group(1)) > 31:
                reasons.append("number")
                break
    return sorted(set(reasons))


def ground_directive(op: Any, *, known_ids: set[str], catalog_features: set[str], concepts: set[str], tabu: set[str],
                     existing: list[Any]) -> tuple[Any | None, list[str]]:
    """Turn an LLM edit op (add / replace) into a ``Directive`` or reject it with reason codes.

    Reasons: markup / injection, unknown_id (every id in the text must be cited, and cited ids must be
    among ``known_ids`` = the ids the renderer showed AND that have evidence), number, year,
    unknown_feature, unknown_concept, no_cite (exploit / avoid), too_long, near_duplicate_text, tabu.
    The directive id is assigned by code from its content."""
    import hashlib

    from aitrading.beliefs.rl.models import Directive

    reasons: list[str] = []
    raw = op.text or ""
    if len(raw) > 200:
        reasons.append("too_long")
    clean, issues = sanitize_llm_text(raw, max_chars=200)
    if any(i.startswith(("markup", "link")) for i in issues):
        reasons.append("markup")
    if any("instruction-like" in i for i in issues):
        reasons.append("injection")
    if not clean.strip():
        reasons.append("empty")
    ids_in_text = set(HYP_ID_RE.findall(raw))
    cites = list(dict.fromkeys(op.cites or []))[:4]
    if not ids_in_text <= set(cites) or not set(cites) <= known_ids:
        reasons.append("unknown_id")
    reasons += _text_numbers_ok(HYP_ID_RE.sub(" ", raw))
    for tok in _SNAKE_RE.findall(raw):
        if tok not in catalog_features:
            reasons.append("unknown_feature")
            break
    kind = op.kind or "formulate"
    region = op.region
    if region is not None and region.concept not in concepts:
        reasons.append("unknown_concept")
    if kind in ("exploit", "avoid") and not cites:
        reasons.append("no_cite")
    words = set(re.findall(r"[a-z]+", clean.lower()))
    for d in existing:
        other = set(re.findall(r"[a-z]+", d.text.lower()))
        if words and other and len(words & other) / len(words | other) >= 0.8:
            reasons.append("near_duplicate_text")
            break
    norm = re.sub(r"\s+", " ", clean.lower()).strip()
    text_hash = hashlib.sha1(norm.encode()).hexdigest()[:12]
    if text_hash in tabu:
        reasons.append("tabu")
    if reasons:
        return None, sorted(set(reasons))
    key = f"{kind}|{region.concept if region else ''}|{region.horizon_bucket if region else ''}|{norm}"
    did = "d-" + hashlib.sha1(key.encode()).hexdigest()[:8]
    return Directive(directive_id=did, kind=kind, region=region, text=clean, cites=cites), []


def directive_text_hash(text: str) -> str:
    import hashlib

    clean, _ = sanitize_llm_text(text or "", max_chars=200)
    return hashlib.sha1(re.sub(r"\s+", " ", clean.lower()).strip().encode()).hexdigest()[:12]
