"""Sentence-aware, model-independent chunking with source provenance."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Iterable, Sequence

from app.config import get_settings
from app.document_loader import DocumentFolderError, DocumentLoader
from app.models import ExtractedDocument, ExtractedPage, TextChunk


class ChunkingConfigurationError(ValueError):
    """Raised when chunk size or overlap cannot produce safe chunks."""


class DocumentChunker:
    """Split extracted source units while retaining their metadata.

    Sizes are measured in Unicode characters, not model-specific tokens.
    Sentence/paragraph boundaries are preferred, whitespace is the second
    choice, and a hard character boundary is used only when necessary.
    """

    SENTENCE_ENDINGS = frozenset(".!?…。！？")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ChunkingConfigurationError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ChunkingConfigurationError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingConfigurationError(
                "chunk_overlap must be smaller than chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(
        self, documents: Iterable[ExtractedDocument]
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: ExtractedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        source_unit_count = len(document.pages)
        for source_unit_index, page in enumerate(document.pages, start=1):
            chunks.extend(
                self.chunk_page(
                    page,
                    source_unit_index=source_unit_index,
                    source_unit_count=source_unit_count,
                )
            )
        return chunks

    def chunk_page(
        self,
        page: ExtractedPage,
        *,
        source_unit_index: int = 1,
        source_unit_count: int = 1,
    ) -> list[TextChunk]:
        text = page.text
        if not text.strip():
            return []

        sentence_boundaries = self._sentence_boundaries(text)
        whitespace_boundaries = self._whitespace_boundaries(text)
        spans = self._chunk_spans(text, sentence_boundaries, whitespace_boundaries)
        document_key = self._document_key(page.document)
        page_key = self._page_key(
            page.page,
            source_unit_index=source_unit_index,
            source_unit_count=source_unit_count,
        )

        chunks: list[TextChunk] = []
        for chunk_index, (start, end) in enumerate(spans, start=1):
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_key}_{page_key}_c{chunk_index:02d}",
                    text=text[start:end],
                    document=page.document,
                    page=page.page,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
        return chunks

    def _chunk_spans(
        self,
        text: str,
        sentence_boundaries: Sequence[int],
        whitespace_boundaries: Sequence[int],
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = self._skip_whitespace(text, 0, len(text))

        while cursor < len(text):
            raw_end = self._choose_end(
                text,
                cursor,
                sentence_boundaries,
                whitespace_boundaries,
            )
            start = self._skip_whitespace(text, cursor, raw_end)
            end = self._trim_trailing_whitespace(text, start, raw_end)
            if end > start:
                spans.append((start, end))

            if raw_end >= len(text):
                break

            next_cursor = self._choose_next_start(
                text,
                cursor,
                raw_end,
                sentence_boundaries,
                whitespace_boundaries,
            )
            if next_cursor <= cursor:
                next_cursor = cursor + 1
            cursor = next_cursor

        return spans

    def _choose_end(
        self,
        text: str,
        start: int,
        sentence_boundaries: Sequence[int],
        whitespace_boundaries: Sequence[int],
    ) -> int:
        maximum_end = min(start + self.chunk_size, len(text))
        if maximum_end == len(text):
            return maximum_end

        minimum_good_end = start + max(1, self.chunk_size // 2)
        sentence_end = self._latest_boundary(
            sentence_boundaries, minimum_good_end, maximum_end
        )
        if sentence_end is not None:
            return sentence_end

        whitespace_end = self._latest_boundary(
            whitespace_boundaries, minimum_good_end, maximum_end
        )
        return whitespace_end if whitespace_end is not None else maximum_end

    def _choose_next_start(
        self,
        text: str,
        previous_start: int,
        previous_end: int,
        sentence_boundaries: Sequence[int],
        whitespace_boundaries: Sequence[int],
    ) -> int:
        if self.chunk_overlap == 0:
            return self._skip_whitespace(text, previous_end, len(text))

        ideal = max(previous_start + 1, previous_end - self.chunk_overlap)
        sentence_start = self._earliest_boundary(
            sentence_boundaries, ideal, previous_end - 1
        )
        if sentence_start is not None:
            return self._skip_whitespace(text, sentence_start, previous_end)

        whitespace_start = self._earliest_boundary(
            whitespace_boundaries, ideal, previous_end - 1
        )
        if whitespace_start is not None:
            return self._skip_whitespace(text, whitespace_start, previous_end)

        return ideal

    @classmethod
    def _sentence_boundaries(cls, text: str) -> list[int]:
        boundaries: list[int] = []
        for index, character in enumerate(text):
            if character in cls.SENTENCE_ENDINGS or character == "\n":
                boundaries.append(index + 1)
        return boundaries

    @staticmethod
    def _whitespace_boundaries(text: str) -> list[int]:
        return [
            match.end()
            for match in re.finditer(r"\s+", text)
        ]

    @staticmethod
    def _latest_boundary(
        boundaries: Sequence[int], minimum: int, maximum: int
    ) -> int | None:
        index = bisect_right(boundaries, maximum) - 1
        if index >= 0 and boundaries[index] >= minimum:
            return boundaries[index]
        return None

    @staticmethod
    def _earliest_boundary(
        boundaries: Sequence[int], minimum: int, maximum: int
    ) -> int | None:
        index = bisect_left(boundaries, minimum)
        if index < len(boundaries) and boundaries[index] <= maximum:
            return boundaries[index]
        return None

    @staticmethod
    def _skip_whitespace(text: str, start: int, limit: int) -> int:
        while start < limit and text[start].isspace():
            start += 1
        return start

    @staticmethod
    def _trim_trailing_whitespace(text: str, start: int, end: int) -> int:
        while end > start and text[end - 1].isspace():
            end -= 1
        return end

    @staticmethod
    def _document_key(document: str) -> str:
        stem = unicodedata.normalize("NFKC", Path(document).stem)
        normalized = "".join(
            character if character.isalnum() else "_" for character in stem
        )
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "document"

    @staticmethod
    def _page_key(
        page: int | None,
        *,
        source_unit_index: int,
        source_unit_count: int,
    ) -> str:
        if page is not None:
            return f"p{page}"
        if source_unit_count > 1:
            return f"pna_u{source_unit_index:02d}"
        return "pna"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Extract and chunk local competition documents."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=settings.documents_dir,
        help="Folder containing PDF, TXT, and DOCX files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Maximum character overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full chunk text in the JSON report.",
    )
    args = parser.parse_args()

    try:
        extraction = DocumentLoader().load(args.documents)
        chunker = DocumentChunker(args.chunk_size, args.chunk_overlap)
    except (DocumentFolderError, ChunkingConfigurationError) as exc:
        parser.error(str(exc))

    chunks = chunker.chunk_documents(extraction.documents)
    report = {
        "settings": {
            "chunk_size": chunker.chunk_size,
            "chunk_overlap": chunker.chunk_overlap,
        },
        "document_count": len(extraction.documents),
        "chunk_count": len(chunks),
        "chunks": [chunk.to_dict(include_text=args.include_text) for chunk in chunks],
        "extraction_errors": [error.to_dict() for error in extraction.errors],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
