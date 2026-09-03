"""Plain data models shared by ingestion modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one source unit.

    PDF page numbers are 1-based. Formats without reliable physical pagination
    use ``None`` rather than inventing a page number.
    """

    document: str
    page: int | None
    text: str

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "document": self.document,
            "page": self.page,
            "character_count": len(self.text),
        }
        if include_text:
            result["text"] = self.text
        return result


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A successfully extracted document and all of its source units."""

    source_path: Path
    document: str
    pages: tuple[ExtractedPage, ...]

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "document": self.document,
            "source_path": str(self.source_path),
            "unit_count": len(self.pages),
            "character_count": self.character_count,
            "pages": [page.to_dict(include_text=include_text) for page in self.pages],
        }


@dataclass(frozen=True, slots=True)
class ExtractionFailure:
    """A safe, serializable failure for one source document."""

    document: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "document": self.document,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(slots=True)
class ExtractionBatch:
    """Results from scanning a document folder."""

    documents: list[ExtractedDocument] = field(default_factory=list)
    errors: list[ExtractionFailure] = field(default_factory=list)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "error_count": len(self.errors),
            "documents": [
                document.to_dict(include_text=include_text)
                for document in self.documents
            ],
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrieval-ready slice with exact source provenance.

    ``start_char`` is inclusive and ``end_char`` is exclusive. Together they
    allow a chunk to be checked against the original extracted page text.
    """

    chunk_id: str
    text: str
    document: str
    page: int | None
    chunk_index: int
    start_char: int
    end_char: int

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "character_count": len(self.text),
        }
        if include_text:
            result["text"] = self.text
        return result


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A ranked chunk returned by the traditional retrieval layer."""

    chunk: TextChunk
    rank: int
    score: float
    confidence: float
    matched_terms: tuple[str, ...]
    exact_match: bool

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result = self.chunk.to_dict(include_text=include_text)
        result.update(
            {
                "rank": self.rank,
                "score": round(self.score, 6),
                "confidence": round(self.confidence, 6),
                "matched_terms": list(self.matched_terms),
                "exact_match": self.exact_match,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Display-safe citation derived only from retrieved chunk metadata."""

    document: str
    page: int | None
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Final pipeline result returned to the API and UI."""

    answer: str
    sources: tuple[SourceCitation, ...]
    grounded: bool
    retrieval_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "grounded": self.grounded,
            "retrieval_confidence": round(self.retrieval_confidence, 6),
        }
