"""Licensing-aware data boundary: what each provider's data may be sent to an external LLM.

Institutional data licences differ on whether vendor content may be transmitted to a third-party
model. The boundary is declared per provider, enforced in code before any prompt is built, and
recorded in the run's audit trail. Defaults are conservative; widen them only when the firm's
agreement with the vendor permits it (e.g. an enterprise feed / data licence, or a vendor-sanctioned
LLM connector).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from aitrading.core.models import Document, DocumentKind


class DataBoundary(BaseModel):
    provider: str
    allow_numeric_features: bool = Field(True, description="Derived feature values (ratios, indicators) may be sent.")
    allowed_document_kinds: set[DocumentKind] = Field(
        default_factory=lambda: {DocumentKind.TRANSCRIPT, DocumentKind.NEWS, DocumentKind.FILING},
        description="Document kinds whose text may be sent. Broker research is excluded by default.",
    )
    max_chars_per_document: int = Field(12_000, description="Upper bound on excerpt text sent per document.")
    max_documents_per_ticker: int = 4
    note: str = ""

    def permits(self, doc: Document) -> bool:
        return doc.kind in self.allowed_document_kinds

    def filter_documents(self, docs: list[Document]) -> tuple[list[Document], list[str]]:
        """Return (permitted documents, human-readable reasons for anything withheld)."""
        kept: list[Document] = []
        withheld: list[str] = []
        for d in docs:
            if not self.permits(d):
                withheld.append(f"{d.doc_id}: {d.kind.value} text not permitted for provider '{self.provider}'")
            else:
                kept.append(d)
        if len(kept) > self.max_documents_per_ticker:
            for d in kept[self.max_documents_per_ticker :]:
                withheld.append(f"{d.doc_id}: over max_documents_per_ticker={self.max_documents_per_ticker}")
            kept = kept[: self.max_documents_per_ticker]
        return kept, withheld


def deny_all_text(provider: str, note: str = "") -> DataBoundary:
    """Boundary for data that must never leave the vendor environment as text."""
    return DataBoundary(provider=provider, allowed_document_kinds=set(), note=note)
