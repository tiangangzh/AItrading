"""Budgeted, verbatim excerpts of narrative documents for the explainer prompt.

The explainer is asked to quote its sources character-for-character, and every quote is later
verified against ``Document.text`` (``aitrading.narrative.grounding``). Excerpts therefore never
alter characters inside the text they keep: they only choose *which* paragraphs / sentences to
keep, cut at paragraph or sentence boundaries, join kept blocks with a blank line and mark each
omission with a line ``[...]``.

Selection
---------
* Transcripts (with segments): candidate units are executive prepared-remarks paragraphs and
  Q&A exchanges (an analyst question plus the answers that follow it); operator and investor-
  relations boilerplate is only kept when the whole call fits. Units are scored by BM25 on the
  focus terms, the CFO's guidance / outlook paragraph is always kept when present, and the kept
  units are rendered in call order as ``Speaker (Role): text``.
* Other documents: paragraphs (or, for paragraphs larger than the budget, sentences) are
  scored the same way, with a small bonus for the lede, and kept in original order.
* A document that fits the budget is returned whole (``truncated=False``).
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from aitrading.core.models import Document, DocumentKind, TranscriptSegment

if TYPE_CHECKING:  # pragma: no cover
    from aitrading.narrative.retrieval import NarrativeBundle

__all__ = [
    "DEFAULT_FOCUS_TERMS",
    "GAP_MARKER",
    "Excerpt",
    "build_excerpt",
    "build_bundle_excerpts",
    "render_documents_for_prompt",
    "neutralise_document_tags",
    "segment_prefix",
    "render_transcript",
    "paragraph_spans",
    "sentence_spans",
]

DEFAULT_FOCUS_TERMS: list[str] = [
    "guidance", "outlook", "expect", "forecast", "consensus", "demand", "orders", "order intake", "bookings",
    "backlog", "book-to-bill", "pipeline", "inventory", "destocking", "channel", "sell-through", "margin",
    "gross margin", "pricing", "price", "price realization", "competition", "competitive", "market share", "share",
    "churn", "retention", "renewal", "customer", "one-time", "one-off", "transitory", "temporary", "timing",
    "headwind", "tailwind", "FX", "currency", "constant currency", "supply", "supply chain", "capacity",
    "buyback", "repurchase", "dividend", "cash flow", "free cash flow", "leverage", "balance sheet", "net cash",
    "visibility", "structural", "exposure", "recovery", "normalize", "slowdown", "weakness", "decline",
    "restructuring", "impairment", "cost actions", "savings", "conservative",
]

# Terms that speak directly to *why* a stock dislocated count double in the relevance score.
_HIGH_SIGNAL_TERMS = frozenset({
    "guidance", "outlook", "consensus", "conservative", "demand", "orders", "order intake", "backlog",
    "book-to-bill", "destocking", "inventory", "one-time", "one-off", "transitory", "temporary", "pricing",
    "price realization", "competition", "competitive", "market share", "churn", "retention", "structural",
    "slowdown", "weakness", "decline", "headwind", "visibility", "exposure", "recovery", "normalize",
})

GAP_MARKER = "[...]"
_SEP = "\n\n"
_LEDE_BONUS = 0.5
_MAX_TOPUP_ATTEMPTS = 50

_BOILERPLATE_ROLES = frozenset({"operator", "ir", "investor relations", "moderator", "host"})
_GUIDANCE_RX = re.compile(
    r"\b(guidance|outlook|forecast|we expect|we now expect|we continue to expect|we anticipate|full[- ]year|"
    r"for the (?:full|fiscal|calendar|coming|next|first|second|third|fourth) (?:fiscal )?(?:year|quarter|half))\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------------------------
# Paragraph / sentence segmentation (spans into the original string; never rewrites text)
# --------------------------------------------------------------------------------------------

_BLANK_LINE_RX = re.compile(r"\n[^\S\n]*\n\s*")
_LINE_BREAK_RX = re.compile(r"(?<=[.!?:\"”’)])[^\S\n]*\n\s*")
_SENT_END_RX = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s)")
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "inc", "corp", "co", "ltd", "llc", "plc", "jr", "sr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "e.u", "no", "nos", "approx", "fig", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "dept", "est", "a.m", "p.m", "l.p", "n.a", "s.a", "mt", "ft",
})
_SENT_START_CHARS = frozenset("\"'“‘([$")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraph_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """``(start, end)`` spans of the paragraphs of ``text[start:end]``, whitespace-trimmed.

    Paragraphs are separated by blank lines; text without any blank line is split at line breaks
    that follow sentence-ending punctuation (so hard-wrapped sentences are never cut).
    """
    end = len(text) if end is None else end
    rx = _BLANK_LINE_RX if _BLANK_LINE_RX.search(text, start, end) else _LINE_BREAK_RX
    spans: list[tuple[int, int]] = []
    pos = start
    for m in rx.finditer(text, start, end):
        s, e = _trim(text, pos, m.start())
        if s < e:
            spans.append((s, e))
        pos = m.end()
    s, e = _trim(text, pos, end)
    if s < e:
        spans.append((s, e))
    return spans


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """``(start, end)`` spans of the sentences of ``text[start:end]``, whitespace-trimmed.

    A boundary is sentence-ending punctuation (plus closing quotes / brackets) followed by
    whitespace and an upper-case letter, digit or opening quote / bracket, unless the period ends
    a known abbreviation ("Inc.", "U.S.") or a single-letter initial.
    """
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    pos = start
    for m in _SENT_END_RX.finditer(text, start, end):
        nxt = m.end()
        while nxt < end and text[nxt].isspace():
            nxt += 1
        if nxt >= end:
            break
        ch = text[nxt]
        if not (ch.isupper() or ch.isdigit() or ch in _SENT_START_CHARS):
            continue
        if m.group(0) == ".":
            w = m.start()
            while w > pos and (text[w - 1].isalpha() or text[w - 1] == "."):
                w -= 1
            word = text[w : m.start()]
            if word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                continue
        s, e = _trim(text, pos, m.end())
        if s < e:
            spans.append((s, e))
        pos = nxt
    s, e = _trim(text, pos, end)
    if s < e:
        spans.append((s, e))
    return spans


# --------------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------------


def segment_prefix(seg: TranscriptSegment) -> str:
    """``"Speaker (Role): "`` - the attribution prefix used for transcript excerpts."""
    speaker, role = (seg.speaker or "").strip(), (seg.role or "").strip()
    if not speaker:
        return f"{role}: " if role else ""
    return f"{speaker} ({role}): " if role else f"{speaker}: "


def render_transcript(segments: Iterable[TranscriptSegment]) -> str:
    """Full rendering of a transcript, one ``Speaker (Role): text`` block per segment."""
    return _SEP.join(segment_prefix(s) + s.text.strip() for s in segments if s.text.strip())


def neutralise_document_tags(text: str) -> str:
    """Escape ``<`` in any ``<document`` / ``</document`` substring so text cannot forge prompt tags."""
    return re.sub(r"<(?=/?document)", "&lt;", text, flags=re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# Excerpts
# --------------------------------------------------------------------------------------------


@dataclass
class Excerpt:
    doc_id: str
    kind: DocumentKind
    title: str
    published_at: datetime
    text: str
    truncated: bool


@dataclass
class _Piece:
    src: int  # transcript segment index, or -1 for Document.text
    start: int
    end: int
    para: int  # paragraph id (unique per document)
    seq: int = -1  # position among all content pieces, in document order


@dataclass
class _Unit:
    pieces: list[_Piece]
    kind: str  # "para" | "pair" | "turn" | "sent"
    must: bool = False
    score: float = 0.0
    order: int = 0
    guidance: bool = False  # the CFO guidance paragraph (or one of its sentences)
    _len: int = field(default=0, repr=False)


@lru_cache(maxsize=64)
def _term_patterns(terms: tuple[str, ...]) -> tuple[tuple[re.Pattern[str], float], ...]:
    """(pattern, weight) per focus term; multi-word terms match across spaces / hyphens."""
    out = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        body = r"[\s-]+".join(re.escape(w) for w in t.split())
        weight = 2.0 if t.casefold() in _HIGH_SIGNAL_TERMS else 1.0
        out.append((re.compile(rf"(?<!\w){body}(?:s|es|d|ed|ing)?(?!\w)", re.IGNORECASE), weight))
    return tuple(out)


def _bm25(
    texts: list[str], patterns: tuple[tuple[re.Pattern[str], float], ...], k1: float = 1.2, b: float = 0.75
) -> list[float]:
    """Okapi BM25 of each text against the (weighted) focus terms, with idf over ``texts``."""
    n = len(texts)
    scores = [0.0] * n
    if n == 0 or not patterns:
        return scores
    lens = [max(1, len(t.split())) for t in texts]
    avg = sum(lens) / n
    for rx, weight in patterns:
        tf = [len(rx.findall(t)) for t in texts]
        df = sum(1 for f in tf if f)
        if not df:
            continue
        idf = weight * math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for i, f in enumerate(tf):
            if f:
                scores[i] += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * lens[i] / avg))
    return scores


class _Source:
    """Text sources of one document (segment texts, or the document text) and their prefixes."""

    def __init__(self, doc: Document, transcript: bool):
        self.doc = doc
        self.segs = doc.segments if transcript else []

    def text(self, src: int) -> str:
        return self.segs[src].text if src >= 0 else self.doc.text

    def prefix(self, src: int) -> str:
        return segment_prefix(self.segs[src]) if src >= 0 else ""

    def piece_text(self, p: _Piece) -> str:
        return self.text(p.src)[p.start : p.end]


def _is_boilerplate(seg: TranscriptSegment) -> bool:
    return (seg.role or "").strip().casefold() in _BOILERPLATE_ROLES or (seg.speaker or "").strip().casefold() == "operator"


def _is_analyst(seg: TranscriptSegment) -> bool:
    return "analyst" in (seg.role or "").casefold()


def _is_cfo(seg: TranscriptSegment) -> bool:
    r = (seg.role or "").casefold()
    return "cfo" in r or "chief financial" in r or "finance" in r


def _transcript_units(source: _Source) -> tuple[list[_Unit], tuple[int, int] | None]:
    """Base units (paragraphs / Q&A exchanges) and the (segment, para id) of the CFO guidance paragraph."""
    units: list[_Unit] = []
    pair: _Unit | None = None
    para_id = 0
    best: tuple[int, int, int] | None = None  # (hits, segment, para id): most guidance hits, latest on ties
    for si, seg in enumerate(source.segs):
        if _is_boilerplate(seg):
            pair = None
            continue
        pieces = []
        for s, e in paragraph_spans(seg.text):
            pieces.append(_Piece(si, s, e, para_id))
            if seg.section == "prepared_remarks" and _is_cfo(seg):
                hits = len(_GUIDANCE_RX.findall(seg.text, s, e))
                if hits and (best is None or hits >= best[0]):
                    best = (hits, si, para_id)
            para_id += 1
        if not pieces:
            continue
        if seg.section == "qa":
            if _is_analyst(seg):
                pair = _Unit(pieces, "pair")
                units.append(pair)
            elif pair is not None:
                pair.pieces.extend(pieces)
            else:
                units.append(_Unit(pieces, "turn"))
        else:
            pair = None
            units.extend(_Unit([p], "para") for p in pieces)
    return units, (best[1], best[2]) if best else None


def _plain_units(source: _Source) -> list[_Unit]:
    return [_Unit([_Piece(-1, s, e, i)], "para") for i, (s, e) in enumerate(paragraph_spans(source.doc.text))]


def _est_len(unit: _Unit, source: _Source) -> int:
    """Upper-bound-ish rendered length: text + attribution prefix + a separator per piece."""
    return sum(p.end - p.start + len(source.prefix(p.src)) + len(_SEP) for p in unit.pieces)


def _refine(units: list[_Unit], source: _Source, max_chars: int) -> list[_Unit]:
    """Split units that can never fit: exchanges into paragraphs, paragraphs into sentences."""
    out: list[_Unit] = []
    for u in units:
        if _est_len(u, source) <= max_chars:
            out.append(u)
            continue
        for p in u.pieces:
            single = _Unit([p], "para")
            if _est_len(single, source) <= max_chars:
                out.append(single)
                continue
            txt = source.text(p.src)
            for s, e in sentence_spans(txt, p.start, p.end):
                out.append(_Unit([_Piece(p.src, s, e, p.para)], "sent"))
    return out


def _assemble(selected: list[_Unit], n_pieces: int, source: _Source) -> str:
    pieces = sorted((p for u in selected for p in u.pieces), key=lambda p: p.seq)
    blocks: list[str] = []
    cur: list = []  # [src, para, start, end, prefix]
    prev: _Piece | None = None

    def flush() -> None:
        if cur:
            src, _, s, e, pre = cur
            blocks.append(pre + source.text(src)[s:e])
            cur.clear()

    for p in pieces:
        if prev is None:
            if p.seq > 0:
                blocks.append(GAP_MARKER)
            cur[:] = [p.src, p.para, p.start, p.end, source.prefix(p.src)]
        elif p.seq == prev.seq + 1 and p.src == prev.src and p.para == prev.para:
            cur[3] = p.end
        elif p.seq == prev.seq + 1:
            flush()
            cur[:] = [p.src, p.para, p.start, p.end, source.prefix(p.src) if p.src != prev.src else ""]
        else:
            flush()
            blocks.append(GAP_MARKER)
            cur[:] = [p.src, p.para, p.start, p.end, source.prefix(p.src)]
        prev = p
    flush()
    if prev is not None and prev.seq < n_pieces - 1:
        blocks.append(GAP_MARKER)
    return _SEP.join(blocks)


def _full_text(doc: Document) -> str:
    if doc.kind == DocumentKind.TRANSCRIPT and doc.segments:
        return render_transcript(doc.segments)
    return doc.text.strip()


def build_excerpt(doc: Document, max_chars: int, focus_terms: list[str] | None = None) -> Excerpt:
    """Verbatim excerpt of ``doc`` of at most ``max_chars`` characters (see module docstring)."""

    def make(text: str, truncated: bool) -> Excerpt:
        return Excerpt(doc.doc_id, doc.kind, doc.title, doc.published_at, text, truncated)

    full = _full_text(doc)
    if len(full) <= max(0, max_chars):
        return make(full, False)
    if max_chars <= 0:
        return make("", True)

    transcript = doc.kind == DocumentKind.TRANSCRIPT and bool(doc.segments)
    source = _Source(doc, transcript)
    guidance_para: tuple[int, int] | None = None
    if transcript:
        units, guidance_para = _transcript_units(source)
    else:
        units = _plain_units(source)
    units = _refine(units, source, max_chars)

    n_pieces = 0
    for i, u in enumerate(units):
        u.order = i
        u._len = _est_len(u, source)
        for p in u.pieces:
            p.seq = n_pieces
            n_pieces += 1
            if guidance_para is not None and (p.src, p.para) == guidance_para:
                u.guidance = True
    if not units:
        return make("", True)

    terms = tuple(DEFAULT_FOCUS_TERMS if focus_terms is None else focus_terms)
    texts = [" ".join(source.piece_text(p) for p in u.pieces) for u in units]
    for u, sc in zip(units, _bm25(texts, _term_patterns(terms))):
        u.score = sc
    if not transcript:
        units[0].score += _LEDE_BONUS
    for u, t in zip(units, texts):
        # A guidance paragraph that had to be split keeps only its guidance sentences as must-haves.
        u.must = u.guidance and (u.kind != "sent" or bool(_GUIDANCE_RX.search(t)))

    overhead = 2 * (len(GAP_MARKER) + len(_SEP))
    ranked = sorted(units, key=lambda u: (not u.must, -u.score, u.order))
    selected: list[_Unit] = []
    used = overhead
    for u in ranked:
        cost = u._len + len(GAP_MARKER) + len(_SEP)
        if used + cost <= max_chars:
            selected.append(u)
            used += cost
    text = _assemble(selected, n_pieces, source)
    while len(text) > max_chars and selected:
        selected.remove(min(selected, key=lambda u: (u.must, u.score, -u.order)))
        text = _assemble(selected, n_pieces, source)
    # The cost above is an upper bound (gap markers, prefixes); top up with exact length checks.
    chosen = {id(u) for u in selected}
    attempts = 0
    for u in ranked:
        if attempts >= _MAX_TOPUP_ATTEMPTS or len(text) >= max_chars:
            break
        if id(u) in chosen or u._len - len(_SEP) > max_chars - len(text) + len(GAP_MARKER) + len(_SEP):
            continue
        attempts += 1
        candidate = _assemble(selected + [u], n_pieces, source)
        if len(candidate) <= max_chars:
            selected.append(u)
            chosen.add(id(u))
            text = candidate
    return make(text, True)


def _allocate(needs: list[int], weights: list[float], total: int) -> list[int]:
    """Weighted water-filling: documents needing less than their share free budget for the others."""
    alloc = [0] * len(needs)
    active = [i for i, n in enumerate(needs) if n > 0]
    remaining = float(max(0, total))
    while active:
        wsum = sum(weights[i] for i in active)
        share = {i: remaining * weights[i] / wsum for i in active}
        capped = [i for i in active if needs[i] <= share[i]]
        if not capped:
            for i in active:
                alloc[i] = int(share[i])
            break
        for i in capped:
            alloc[i] = needs[i]
            remaining -= needs[i]
        active = [i for i in active if i not in capped]
    return alloc


_KIND_WEIGHT = {DocumentKind.TRANSCRIPT: 2.0, DocumentKind.FILING: 1.0, DocumentKind.NEWS: 1.0, DocumentKind.RESEARCH: 1.0}


def build_bundle_excerpts(
    bundle: "NarrativeBundle",
    max_chars_total: int = 40_000,
    per_doc_max: int | None = None,
    *,
    focus_terms: list[str] | None = None,
) -> list[Excerpt]:
    """Excerpts for every document in ``bundle`` (in bundle order) within a total character budget.

    The budget is shared by weighted water-filling (transcripts weigh double); ``per_doc_max``
    (typically the provider boundary's ``max_chars_per_document``) caps each document. Documents
    whose excerpt comes out empty are left out.
    """
    docs = list(bundle.documents)
    if not docs:
        return []
    cap = per_doc_max if per_doc_max is not None else max_chars_total
    needs = [min(len(_full_text(d)), max(0, cap)) for d in docs]
    alloc = _allocate(needs, [_KIND_WEIGHT.get(d.kind, 1.0) for d in docs], max_chars_total)
    out: list[Excerpt] = []
    for d, n in zip(docs, alloc):
        if n <= 0:
            continue
        ex = build_excerpt(d, n, focus_terms)
        if ex.text:
            out.append(ex)
    return out


def _attr(value: str) -> str:
    return html.escape(" ".join(str(value).split()), quote=True)


def render_documents_for_prompt(excerpts: list[Excerpt]) -> str:
    """``<document doc_id=".." kind=".." title=".." published="YYYY-MM-DD">text</document>`` blocks.

    Attribute values are HTML-escaped; inside the text only ``<document`` / ``</document``
    substrings are neutralised (``<`` -> ``&lt;``), so ordinary quotes stay verbatim.
    """
    blocks = []
    for ex in excerpts:
        kind = getattr(ex.kind, "value", ex.kind)
        published = ex.published_at.date().isoformat() if isinstance(ex.published_at, datetime) else str(ex.published_at)
        blocks.append(
            f'<document doc_id="{_attr(ex.doc_id)}" kind="{_attr(kind)}" title="{_attr(ex.title)}" '
            f'published="{published}">\n{neutralise_document_tags(ex.text)}\n</document>'
        )
    return _SEP.join(blocks)
