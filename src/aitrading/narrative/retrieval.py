"""Point-in-time document retrieval for one ticker, with the provider's data boundary enforced.

``gather_documents`` asks the provider for transcripts, news and filings published in
``[as_of - lookback_days, as_of]``, orders them by usefulness to the explainer (latest earnings
calls first, then news newest first, then filings, then research), drops anything outside the
window (defence in depth against look-ahead), de-duplicates by ``doc_id`` and finally applies
``provider.boundary.filter_documents`` so only text the licence permits can reach an LLM. Every
document that is dropped for a reason other than the per-kind count limits is recorded in
``NarrativeBundle.withheld`` for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from aitrading.core.models import Document, DocumentKind
from aitrading.core.policy import DataBoundary
from aitrading.data.base import Capability, MarketDataProvider

__all__ = ["NarrativeBundle", "gather_documents", "DEFAULT_KINDS", "KIND_ORDER"]

DEFAULT_KINDS: frozenset[DocumentKind] = frozenset({DocumentKind.TRANSCRIPT, DocumentKind.NEWS, DocumentKind.FILING})
# Priority order of kinds in a bundle (the boundary's per-ticker document cap keeps the head).
KIND_ORDER: tuple[DocumentKind, ...] = (DocumentKind.TRANSCRIPT, DocumentKind.NEWS, DocumentKind.FILING, DocumentKind.RESEARCH)
_CAPABILITY: dict[DocumentKind, Capability] = {
    DocumentKind.TRANSCRIPT: Capability.TRANSCRIPTS,
    DocumentKind.NEWS: Capability.NEWS,
    DocumentKind.FILING: Capability.FILINGS,
    DocumentKind.RESEARCH: Capability.RESEARCH,
}


@dataclass
class NarrativeBundle:
    """Documents for one ticker that may be shown to the explainer, plus why others were withheld."""

    ticker: str
    documents: list[Document]
    withheld: list[str] = field(default_factory=list)

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def by_kind(self, kind: DocumentKind) -> list[Document]:
        return [d for d in self.documents if d.kind == kind]


def _utc_naive(dt: datetime) -> datetime:
    """Comparable sort key for naive and tz-aware datetimes (aware ones are converted to UTC)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo is not None else dt


def _as_date(d: date) -> date:
    return d.date() if isinstance(d, datetime) else d


def gather_documents(
    provider: MarketDataProvider,
    ticker: str,
    as_of: date,
    *,
    lookback_days: int = 120,
    kinds: set[DocumentKind] | None = None,
    max_transcripts: int = 2,
    max_news: int = 6,
    max_filings: int = 2,
) -> NarrativeBundle:
    """Fetch the narrative documents for ``ticker`` known on ``as_of``.

    Only kinds the provider declares a capability for are requested (a provider without a
    ``capabilities`` attribute is asked for every kind). Research uses the ``max_news`` limit.
    A provider exception for one kind is recorded in ``withheld`` and the other kinds proceed.
    """
    if lookback_days < 0:
        raise ValueError(f"lookback_days must be >= 0, got {lookback_days}")
    as_of = _as_date(as_of)
    start = as_of - timedelta(days=lookback_days)
    wanted = set(DEFAULT_KINDS if kinds is None else {DocumentKind(k) for k in kinds})
    caps = getattr(provider, "capabilities", None)
    limits = {
        DocumentKind.TRANSCRIPT: max_transcripts,
        DocumentKind.NEWS: max_news,
        DocumentKind.FILING: max_filings,
        DocumentKind.RESEARCH: max_news,
    }
    withheld: list[str] = []
    seen: set[str] = set()
    ordered: list[Document] = []
    for kind in KIND_ORDER:
        limit = limits[kind]
        if kind not in wanted or limit <= 0:
            continue
        if caps is not None and _CAPABILITY[kind] not in caps:
            continue
        try:
            # Over-fetch a little so documents dropped below still leave ``limit`` usable ones.
            fetched = provider.get_documents(ticker, {kind}, start, as_of, limit=2 * limit + 2)
        except Exception as exc:  # noqa: BLE001 - one failing feed must not sink the others
            withheld.append(f"{kind.value}: provider error ({type(exc).__name__}: {exc})")
            continue
        keep: list[Document] = []
        for doc in fetched or []:
            if doc.kind != kind:
                continue  # a provider ignoring the kind filter; the document's own kind is fetched separately
            day = _as_date(doc.published_at)
            if day > as_of:
                withheld.append(f"{doc.doc_id}: published {day.isoformat()} after as_of {as_of.isoformat()}")
            elif day < start:
                withheld.append(f"{doc.doc_id}: published {day.isoformat()} before lookback start {start.isoformat()}")
            else:
                keep.append(doc)
        keep.sort(key=lambda d: (_utc_naive(d.published_at), d.doc_id), reverse=True)
        n_kind = 0
        for doc in keep:
            if n_kind >= limit:
                break
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            ordered.append(doc)
            n_kind += 1
    boundary: DataBoundary | None = getattr(provider, "boundary", None)
    if boundary is None:
        boundary = DataBoundary(provider=str(getattr(provider, "name", "unknown")))
    kept, reasons = boundary.filter_documents(ordered)
    withheld.extend(reasons)
    return NarrativeBundle(ticker=ticker, documents=kept, withheld=withheld)
