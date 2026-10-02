"""Programmatic grounding of an explanation: every quote and every cited number is checked.

Quotes (``QuoteEvidence``)
    Both the quote and the source are normalised (HTML entities, Unicode compatibility forms,
    curly quotes / dashes -> ASCII, whitespace collapsed, case-folded); surrounding quote marks and
    leading / trailing ellipses are stripped from the quote. A quote containing an ellipsis
    ("...", "…", "[...]") is split into fragments; fragments shorter than ``MIN_FRAGMENT_CHARS``
    are ignored, at least one must remain, and all remaining fragments must occur in order. The
    cited document is searched first (its ``text`` and, for transcripts, the
    ``Speaker (Role): text`` rendering of its segments, so a quote copied from an excerpt with or
    without the attribution prefix verifies). A quote found only in another provided document is
    a ``mismatch`` naming that document; a quote found nowhere is ``not_found``. For transcript
    quotes with a ``speaker``, a clear mis-attribution (the quote lies in another speaker's turn) is
    a ``mismatch``.

Numbers (``QuantEvidence``)
    The feature must exist in the feature table (else ``not_found``). A value passes if
    ``|a - b| <= abs_tol`` or ``|a - b| <= rel_tol * |b|``, or if ``a`` equals ``b`` rounded to the
    number of decimals ``a`` is written with (12.3 for 12.345). Missing vs missing (None / NaN)
    verifies; anything else is a ``mismatch`` with both values in the detail.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from numbers import Number
from typing import Any

from aitrading.core.models import (
    DislocationThesis,
    Document,
    EvidenceCheck,
    GroundingReport,
    QuantEvidence,
    QuoteEvidence,
    TranscriptSegment,
)
from aitrading.narrative.excerpts import segment_prefix

__all__ = [
    "MIN_FRAGMENT_CHARS",
    "normalize_text",
    "quote_fragments",
    "verify_thesis",
    "verify_quote",
    "verify_quant",
    "values_match",
]

MIN_FRAGMENT_CHARS = 12

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "​": None, "‌": None, "‍": None, "⁠": None, "﻿": None, "­": None,
})
_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\s*){3,}\]|\(\s*(?:\.\s*){3,}\)|(?:\.\s*){2}\.)\s*")
_EDGE_JUNK = " \t\n\"'"


def normalize_text(text: str) -> str:
    """Canonical form for verbatim comparison (see module docstring)."""
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    return _WS.sub(" ", text.translate(_CHAR_MAP)).strip().casefold()


def _strip_edges(q: str) -> str:
    """Strip surrounding quote marks and leading / trailing ellipses (already normalised text)."""
    prev = None
    while q != prev:
        prev = q
        q = q.strip(_EDGE_JUNK)
        m = _ELLIPSIS.match(q)
        if m and m.start() == 0:
            q = q[m.end() :]
        for m in _ELLIPSIS.finditer(q):
            if m.end() == len(q):
                q = q[: m.start()]
                break
    return q


def quote_fragments(quote: str) -> list[str]:
    """Normalised fragments of ``quote`` to locate in order (fragments < MIN_FRAGMENT_CHARS dropped)."""
    q = _strip_edges(normalize_text(quote))
    parts = (_strip_edges(p) for p in _ELLIPSIS.split(q))
    return [p for p in parts if len(p) >= MIN_FRAGMENT_CHARS]


def _find_in_order(fragments: list[str], haystack: str) -> int | None:
    """Start position of the first fragment if all fragments occur in order, else None."""
    pos, first = 0, None
    for frag in fragments:
        i = haystack.find(frag, pos)
        if i < 0:
            return None
        if first is None:
            first = i
        pos = i + len(frag)
    return first


@dataclass
class _DocIndex:
    doc: Document
    spaces: list[str]  # normalised search spaces: Document.text, transcript rendering
    segments: list[tuple[TranscriptSegment, str]]  # (segment, normalised text)
    speakers: list[str]  # normalised speaker names


def _index(doc: Document) -> _DocIndex:
    spaces = [normalize_text(doc.text)]
    segs: list[tuple[TranscriptSegment, str]] = []
    if doc.segments:
        rendering = "\n\n".join(segment_prefix(s) + s.text for s in doc.segments)
        spaces.append(normalize_text(rendering))
        segs = [(s, normalize_text(s.text)) for s in doc.segments]
    speakers = sorted({normalize_text(s.speaker) for s in doc.segments if s.speaker.strip()}, key=len, reverse=True)
    return _DocIndex(doc, spaces, segs, speakers)


_PREFIX = re.compile(r"^(?P<name>[^():\n]{1,80}?)\s*(?:\((?P<role>[^()]{1,60})\))?\s*:\s+")


def _strip_speaker_prefix(quote_norm: str, idx: _DocIndex, speaker: str | None) -> str | None:
    """The quote without a leading ``Speaker (Role):`` attribution, if it names a known speaker."""
    m = _PREFIX.match(quote_norm)
    if not m:
        return None
    name = m.group("name").strip(_EDGE_JUNK)
    known = set(idx.speakers)
    if speaker:
        known.add(normalize_text(speaker))
    roles = {normalize_text(s.role) for s, _ in idx.segments if s.role}
    if name in known or name in roles:
        return quote_norm[m.end() :]
    return None


def _locate(fragments: list[str], idx: _DocIndex) -> bool:
    return any(_find_in_order(fragments, space) is not None for space in idx.spaces)


_STOP = frozenset({"the", "mr", "ms", "mrs", "dr", "and", "of", "chief", "officer", "executive", "financial"})
_ROLE_ALIASES = {
    "ceo": ("chief executive", "ceo"),
    "cfo": ("chief financial", "cfo"),
    "coo": ("chief operating", "coo"),
    "cto": ("chief technology", "cto"),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w'-]+", text))


def _speaker_matches(claimed: str, seg: TranscriptSegment) -> bool:
    """Lenient: the claimed speaker shares the segment speaker's surname / names, or names its role."""
    c = normalize_text(claimed)
    ct = _tokens(c)
    if not ct:
        return True
    name = [t for t in re.findall(r"[\w'-]+", normalize_text(seg.speaker)) if t not in _STOP]
    if name and (name[-1] in ct or (ct - _STOP and (ct - _STOP) <= set(name))):
        return True
    role = normalize_text(seg.role)
    rt = _tokens(role)
    if rt and rt <= ct:
        return True
    for key, aliases in _ROLE_ALIASES.items():
        if (key in rt or any(a in role for a in aliases)) and (key in ct or any(a in c for a in aliases)):
            return True
    return False


def _speaker_check(fragments: list[str], idx: _DocIndex, speaker: str | None) -> str | None:
    """Detail of a clear mis-attribution, or None (no speaker, not a transcript, or consistent)."""
    if not speaker or not speaker.strip() or not idx.segments:
        return None
    holders = [seg for seg, text in idx.segments if _find_in_order(fragments, text) is not None]
    if not holders or any(_speaker_matches(speaker, seg) for seg in holders):
        return None
    who = ", ".join(sorted({f"{s.speaker} ({s.role})" for s in holders}))
    return f"quote is from {who}, not '{speaker}'"


def _short(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def verify_quote(q: QuoteEvidence, indexes: dict[str, _DocIndex], order: list[str]) -> EvidenceCheck:
    """Check one quote against the indexed documents (``order``: doc_ids in caller order)."""

    def check(status: str, detail: str) -> EvidenceCheck:
        return EvidenceCheck(kind="quote", ref=q.doc_id, claim=q.quote, status=status, detail=detail)

    fragments = quote_fragments(q.quote)
    if not fragments:
        return check("not_found", f"quote too short to verify (needs a fragment of >= {MIN_FRAGMENT_CHARS} chars)")
    cited = _resolve_doc_id(q.doc_id, indexes)

    def found_in(idx: _DocIndex) -> list[str] | None:
        """Fragments as located (prefix-stripped if needed), or None."""
        if _locate(fragments, idx):
            return fragments
        stripped = _strip_speaker_prefix(_strip_edges(normalize_text(q.quote)), idx, q.speaker)
        if stripped is not None:
            alt = [p for p in (_strip_edges(x) for x in _ELLIPSIS.split(stripped)) if len(p) >= MIN_FRAGMENT_CHARS]
            if alt and _locate(alt, idx):
                return alt
        return None

    n = len(fragments)
    what = "verbatim" if n == 1 else f"all {n} fragments in order"
    if cited is not None:
        located = found_in(indexes[cited])
        if located is not None:
            mis = _speaker_check(located, indexes[cited], q.speaker)
            if mis:
                return check("mismatch", f"found in {cited} but {mis}")
            return check("verified", f"found {what} in {cited}")
    others = [d for d in order if d != cited and found_in(indexes[d]) is not None]
    if others:
        where = ", ".join(others[:3]) + (f" (+{len(others) - 3} more)" if len(others) > 3 else "")
        head = f"not in cited document {q.doc_id}" if cited is not None else f"cited doc_id '{q.doc_id}' not provided"
        return check("mismatch", f"{head}; found in {where}")
    if cited is None:
        return check("not_found", f"cited doc_id '{q.doc_id}' not provided and quote not found in any document")
    if n > 1:
        idx = indexes[cited]
        k = 0  # number of leading fragments that do occur in order
        while k < n and any(_find_in_order(fragments[: k + 1], s) is not None for s in idx.spaces):
            k += 1
        return check("not_found", f"fragment {k + 1}/{n} ('{_short(fragments[k], 40)}') not found in {cited} or any other document")
    return check("not_found", f"quote not found in {cited} or any other provided document")


def _resolve_doc_id(doc_id: str, indexes: dict[str, _DocIndex]) -> str | None:
    if doc_id in indexes:
        return doc_id
    want = html.unescape(doc_id).strip().strip("\"'").casefold()
    for k in indexes:
        if k.strip().casefold() == want:
            return k
    return None


# --------------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------------


def _as_number(x: Any) -> float | None:
    """float(x) for real numbers and numeric strings; NaN for missing; None if non-numeric."""
    if x is None:
        return math.nan
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, Number):
        try:
            return float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return math.nan  # e.g. pandas.NA
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s.lower() in {"", "nan", "none", "null", "n/a", "na"}:
            return math.nan
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan if str(x) in {"<NA>", "NaT"} else None


def _decimals(a: float) -> int:
    d = Decimal(repr(a)).normalize()
    exp = d.as_tuple().exponent
    return max(0, -exp) if isinstance(exp, int) else 0


def values_match(a: float, b: float, *, rel_tol: float = 0.02, abs_tol: float = 0.05) -> tuple[bool, str]:
    """Whether evidence value ``a`` matches table value ``b`` (both finite or infinite floats), and why."""
    if math.isinf(a) or math.isinf(b):
        return (a == b), ("both infinite" if a == b else "infinite value differs")
    diff = abs(a - b)
    if diff <= abs_tol:
        return True, f"|diff|={diff:.4g} <= abs_tol {abs_tol:g}"
    if diff <= rel_tol * abs(b):
        return True, f"|diff|={diff:.4g} within rel_tol {rel_tol:g}"
    nd = _decimals(a)
    try:
        exp = Decimal(1).scaleb(-nd)
        db = Decimal(repr(b))
        da = Decimal(repr(a))
        if da in (db.quantize(exp, rounding=ROUND_HALF_UP), db.quantize(exp, rounding=ROUND_HALF_EVEN)):
            return True, f"equals table value rounded to {nd} dp"
    except InvalidOperation:  # pragma: no cover - absurd magnitudes
        pass
    return False, f"|diff|={diff:.4g}"


def _fmt(x: float) -> str:
    return "NaN" if math.isnan(x) else f"{x:.6g}"


def verify_quant(e: QuantEvidence, features: dict[str, Any], *, rel_tol: float = 0.02, abs_tol: float = 0.05) -> EvidenceCheck:
    """Check one cited feature value against the candidate's feature table."""
    claim = f"{e.feature}={e.value}"

    def check(status: str, detail: str) -> EvidenceCheck:
        return EvidenceCheck(kind="quant", ref=e.feature, claim=claim, status=status, detail=detail)

    if e.feature not in features:
        near = [k for k in features if k.casefold() == e.feature.strip().casefold()]
        hint = f" (did you mean '{near[0]}'?)" if near else ""
        return check("not_found", f"feature '{e.feature}' not in feature table{hint}")
    raw = features[e.feature]
    b = _as_number(raw)
    a = math.nan if e.value is None else float(e.value)
    if b is None:
        return check("mismatch", f"table value is non-numeric ({raw!r}); evidence {_fmt(a)}")
    if math.isnan(a) and math.isnan(b):
        return check("verified", "both missing (None/NaN)")
    if math.isnan(b):
        return check("mismatch", f"evidence {_fmt(a)} but table value is missing (NaN)")
    if math.isnan(a):
        return check("mismatch", f"evidence is null but table value is {_fmt(b)}")
    ok, why = values_match(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if ok:
        return check("verified", f"table {_fmt(b)}; {why}")
    detail = f"evidence {_fmt(a)} vs table {_fmt(b)} ({why})"
    if b != 0 and (values_match(a, b * 100, rel_tol=rel_tol, abs_tol=abs_tol)[0] or values_match(a, b / 100, rel_tol=rel_tol, abs_tol=abs_tol)[0]):
        detail += "; looks like a x100 unit error"
    return check("mismatch", detail)


# --------------------------------------------------------------------------------------------
# Thesis
# --------------------------------------------------------------------------------------------


def verify_thesis(
    thesis: DislocationThesis,
    documents: list[Document],
    features: dict[str, float | str | None],
    *,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05,
) -> GroundingReport:
    """Check every ``quant_evidence`` item, then every ``narrative_evidence`` quote (in thesis order)."""
    checks = [verify_quant(e, features, rel_tol=rel_tol, abs_tol=abs_tol) for e in thesis.quant_evidence]
    indexes: dict[str, _DocIndex] = {}
    for d in documents:
        if d.doc_id not in indexes:
            indexes[d.doc_id] = _index(d)
    order = list(indexes)
    checks += [verify_quote(q, indexes, order) for q in thesis.narrative_evidence]
    return GroundingReport(ticker=thesis.ticker, checks=checks)
