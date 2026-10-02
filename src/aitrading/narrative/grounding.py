"""Programmatic grounding of an explanation: every quote and every cited number is checked.

Quotes (``QuoteEvidence``)
    Both the quote and the source are normalised (HTML entities, Unicode compatibility forms,
    curly quotes / dashes -> ASCII, whitespace collapsed, case-folded); surrounding quote marks and
    leading / trailing ellipses are stripped from the quote. A match must start and end on word
    boundaries of the source ("profitable" does not verify inside "unprofitable", "1.2%" not inside
    "11.2%"). A quote with an interior ellipsis ("...", "…", "[...]") verifies when it occurs
    verbatim, ellipsis included (the source itself has one); otherwise it is split into fragments,
    and EVERY fragment must have at least ``MIN_FRAGMENT_CHARS`` characters (a shorter one, e.g. an
    invented "... by 900 bps" tail, cannot be checked, so the quote is ``not_found``), the fragments
    must occur in order within ONE sentence of the source, and the words the ellipsis skips must not
    contain a negation ("We do ... expect" for "We do not expect"); stitched or negation-skipping
    quotes are a ``mismatch``. The cited document is searched first (its ``text`` and, for
    transcripts, the ``Speaker (Role): text`` rendering of its segments, so a quote copied from an
    excerpt with or without the attribution prefix verifies). A quote found only in another provided
    document is a ``mismatch`` naming that document; a quote found nowhere is ``not_found``. For
    transcript quotes with a ``speaker``, a clear mis-attribution (the quote lies in another
    speaker's turn) is a ``mismatch``.

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
from aitrading.narrative.excerpts import segment_prefix, sentence_spans

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
# Words that, when skipped by "...", can reverse what the source says (as in discovery.extract).
_GAP_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|none|nothing|cannot|without|fails?|failed|lacks?|lacked|hardly|barely|"
    r"insignificant(?:ly)?|unprofitable|unable|unlikely|\w+n't)\b"
)
_MAX_OCCURRENCES = 50


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


def _split(q: str) -> list[str]:
    """Non-empty fragments of an already normalised, edge-stripped quote, split at ellipses."""
    return [p for p in (_strip_edges(x) for x in _ELLIPSIS.split(q)) if p]


def quote_fragments(quote: str) -> list[str]:
    """Normalised fragments of ``quote`` between ellipses, in order (none dropped: a fragment shorter
    than ``MIN_FRAGMENT_CHARS`` makes the quote unverifiable unless it occurs verbatim)."""
    return _split(_strip_edges(normalize_text(quote)))


def _canon_ellipses(text: str) -> str:
    """``text`` with every ellipsis written " ... " (for verbatim matches of quotes containing one)."""
    return _WS.sub(" ", _ELLIPSIS.sub(" ... ", text)).strip()


def _occurrences(haystack: str, frag: str, start: int = 0):
    """Start offsets (from ``start``) where ``frag`` occurs beginning and ending on word boundaries."""
    left, right = frag[:1].isalnum(), frag[-1:].isalnum()
    pos, found = start, 0
    while found < _MAX_OCCURRENCES:
        i = haystack.find(frag, pos)
        if i < 0:
            return
        j = i + len(frag)
        if (not left or i == 0 or not haystack[i - 1].isalnum()) and (not right or j == len(haystack) or not haystack[j].isalnum()):
            found += 1
            yield i
        pos = i + 1


def _find_in_order(fragments: list[str], haystack: str) -> int | None:
    """Start position of the first fragment if all fragments occur in order (word-bounded), else None.

    Taking the earliest occurrence of each fragment is optimal for existence, so this is greedy."""
    pos, first = 0, None
    for frag in fragments:
        i = next(_occurrences(haystack, frag, pos), None)
        if i is None:
            return None
        if first is None:
            first = i
        pos = i + len(frag)
    return first


def _in_sentence(fragments: list[str], sentence: str) -> str | None:
    """'ok' if the fragments occur in order in ``sentence`` with no negation in the skipped text,
    'negated_gap' if they only occur skipping a negation, else None (backtracking, memoised)."""
    saw_negated = False
    memo: dict[tuple[int, int], bool] = {}

    def place(k: int, pos: int) -> bool:
        nonlocal saw_negated
        if k == len(fragments):
            return True
        if (k, pos) not in memo:
            ok = False
            for i in _occurrences(sentence, fragments[k], pos):
                if k > 0 and _GAP_NEGATION.search(sentence[pos:i]):
                    saw_negated = True
                    continue
                if place(k + 1, i + len(fragments[k])):
                    ok = True
                    break
            memo[(k, pos)] = ok
        return memo[(k, pos)]

    if place(0, 0):
        return "ok"
    return "negated_gap" if saw_negated else None


@dataclass
class _DocIndex:
    doc: Document
    spaces: list[str]  # normalised search spaces: Document.text, transcript rendering
    segments: list[tuple[TranscriptSegment, str]]  # (segment, normalised text)
    speakers: list[str]  # normalised speaker names
    _sentences: list[str] | None = None
    _canon: list[str] | None = None

    def sentences(self) -> list[str]:
        """Normalised sentences of ``Document.text`` and of each transcript segment (built lazily)."""
        if self._sentences is None:
            raws = [self.doc.text or ""] + [s.text for s, _ in self.segments]
            self._sentences = [normalize_text(r[a:b]) for r in raws for a, b in sentence_spans(r)]
        return self._sentences

    def canon_spaces(self) -> list[str]:
        if self._canon is None:
            self._canon = [_canon_ellipses(sp) for sp in self.spaces]
        return self._canon


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


_VERIFIED = ("verbatim", "in_sentence")


def _match(fragments: list[str], idx: _DocIndex) -> str | None:
    """How ``fragments`` occur in ``idx``: 'verbatim' (one fragment, or the quote with its ellipses
    as written), 'in_sentence' (in order within one sentence, no negation skipped), 'negated_gap',
    'cross_sentence' (in order, but stitched across sentences), or None."""
    if len(fragments) == 1:
        return "verbatim" if len(fragments[0]) >= MIN_FRAGMENT_CHARS and _locate(fragments, idx) else None
    joined = " ... ".join(fragments)
    if any(_find_in_order([joined], sp) is not None for sp in idx.canon_spaces()):
        return "verbatim"
    if any(len(f) < MIN_FRAGMENT_CHARS for f in fragments):
        return None
    outcomes = {_in_sentence(fragments, sent) for sent in idx.sentences()}
    if "ok" in outcomes:
        return "in_sentence"
    if "negated_gap" in outcomes:
        return "negated_gap"
    return "cross_sentence" if _locate(fragments, idx) else None


def verify_quote(q: QuoteEvidence, indexes: dict[str, _DocIndex], order: list[str]) -> EvidenceCheck:
    """Check one quote against the indexed documents (``order``: doc_ids in caller order)."""

    def check(status: str, detail: str) -> EvidenceCheck:
        return EvidenceCheck(kind="quote", ref=q.doc_id, claim=q.quote, status=status, detail=detail)

    fragments = quote_fragments(q.quote)
    if not fragments or (len(fragments) == 1 and len(fragments[0]) < MIN_FRAGMENT_CHARS):
        return check("not_found", f"quote too short to verify (needs a fragment of >= {MIN_FRAGMENT_CHARS} chars)")
    cited = _resolve_doc_id(q.doc_id, indexes)

    def found_in(idx: _DocIndex) -> tuple[str, list[str]] | None:
        """(how, fragments as located - prefix-stripped if needed) for the best match, or None."""
        best: tuple[str, list[str]] | None = None
        variants = [fragments]
        stripped = _strip_speaker_prefix(_strip_edges(normalize_text(q.quote)), idx, q.speaker)
        if stripped is not None:
            alt = _split(stripped)
            if alt:
                variants.append(alt)
        for frags in variants:
            how = _match(frags, idx)
            if how in _VERIFIED:
                return how, frags
            if how is not None and best is None:
                best = (how, frags)
        return best

    n = len(fragments)
    hit = found_in(indexes[cited]) if cited is not None else None
    if hit is not None and hit[0] in _VERIFIED:
        mis = _speaker_check(hit[1], indexes[cited], q.speaker)
        if mis:
            return check("mismatch", f"found in {cited} but {mis}")
        what = "verbatim" if hit[0] == "verbatim" else f"all {len(hit[1])} fragments in order within one sentence"
        return check("verified", f"found {what} in {cited}")
    others = [d for d in order if d != cited and (h := found_in(indexes[d])) is not None and h[0] in _VERIFIED]
    if others:
        where = ", ".join(others[:3]) + (f" (+{len(others) - 3} more)" if len(others) > 3 else "")
        head = f"not in cited document {q.doc_id}" if cited is not None else f"cited doc_id '{q.doc_id}' not provided"
        return check("mismatch", f"{head}; found in {where}")
    if cited is None:
        return check("not_found", f"cited doc_id '{q.doc_id}' not provided and quote not found in any document")
    short = [f for f in fragments if len(f) < MIN_FRAGMENT_CHARS]
    if short:
        return check("not_found", f"fragment '{_short(short[0], 40)}' between '...' is too short to verify (each needs >= "
                                  f"{MIN_FRAGMENT_CHARS} chars); quote one contiguous passage instead")
    if hit is not None and hit[0] == "negated_gap":
        return check("mismatch", f"fragments found in {cited}, but the words skipped by '...' contain a negation that "
                                 "changes the meaning")
    if hit is not None and hit[0] == "cross_sentence":
        return check("mismatch", f"fragments occur in {cited} but not within one sentence ('...' may only skip words "
                                 "inside a sentence)")
    if n > 1:
        idx = indexes[cited]
        k = 0  # number of leading fragments that do occur in order
        while k < n - 1 and any(_find_in_order(fragments[: k + 1], s) is not None for s in idx.spaces):
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
