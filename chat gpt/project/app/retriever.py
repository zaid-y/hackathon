"""Local Thai-aware BM25 retrieval and persisted index support."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.chunker import DocumentChunker
from app.config import get_settings
from app.document_loader import DocumentLoader
from app.models import RetrievedChunk, TextChunk


INDEX_SCHEMA_VERSION = 1
_TERM_PATTERN = re.compile(r"[a-z0-9]+|[\u0E00-\u0E7F]+", re.IGNORECASE)
_ENGLISH_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)
_THAI_STOP_WORDS = frozenset(
    {"การ", "กับ", "ของ", "คือ", "จาก", "ด้วย", "ที่", "หรือ", "อยู่", "และ", "ใน", "เป็น", "ให้", "ได้", "มี"}
)
_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("หน่วยกิต",), "จำนวนหน่วยกิตรวม หน่วยกิตรวมตลอดหลักสูตร โครงสร้างหลักสูตร"),
    (("ค่าใช้จ่าย", "ค่าเรียน"), "ค่าธรรมเนียมการศึกษา ค่าเล่าเรียน"),
    (("รับสมัคร", "สมัครเรียน"), "คุณสมบัติผู้สมัคร เกณฑ์การรับสมัคร"),
    (("เรียนกี่ปี", "ระยะเวลาเรียน"), "ระยะเวลาการศึกษา แผนการศึกษา"),
)


class IndexFormatError(RuntimeError):
    """Raised when a persisted retrieval index is missing or incompatible."""


def normalize_text(text: str) -> str:
    """Normalize Unicode, case, and whitespace while preserving Thai vowels."""

    # NFKC decomposes Thai SARA AM (ำ) into two code points, which makes debug
    # terms unnatural. NFC keeps Thai text readable while still canonicalizing.
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    """Tokenize English/numbers and create local n-grams for unsegmented Thai.

    Thai words commonly appear without spaces. Character bigrams and trigrams
    provide robust keyword matching without an external tokenizer or model.
    """

    tokens: list[str] = []
    for segment in _TERM_PATTERN.findall(normalize_text(text)):
        if re.fullmatch(r"[\u0E00-\u0E7F]+", segment):
            if segment not in _THAI_STOP_WORDS:
                tokens.append(segment)
            for width in (2, 3):
                if len(segment) >= width:
                    tokens.extend(
                        segment[index : index + width]
                        for index in range(len(segment) - width + 1)
                    )
        elif segment not in _ENGLISH_STOP_WORDS:
            tokens.append(segment)
    return tokens


def expand_query(text: str) -> str:
    """Add conservative Thai curriculum synonyms for stronger factual recall."""

    normalized = normalize_text(text)
    additions = [
        expansion
        for triggers, expansion in _QUERY_EXPANSIONS
        if any(trigger in normalized for trigger in triggers)
    ]
    return " ".join([text, *additions]) if additions else text


class BM25Retriever:
    """A small, serializable BM25 index over metadata-preserving chunks."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        self.chunks: list[TextChunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._average_document_length = 0.0

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks)

    def build(self, chunks: Iterable[TextChunk]) -> None:
        self.chunks = list(chunks)
        self._term_frequencies = []
        self._document_frequencies = Counter()
        self._document_lengths = []

        for chunk in self.chunks:
            frequencies = Counter(tokenize(chunk.text))
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(sum(frequencies.values()))
            self._document_frequencies.update(frequencies.keys())

        total_length = sum(self._document_lengths)
        self._average_document_length = (
            total_length / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.is_ready or not query.strip():
            return []

        query_terms = tokenize(expand_query(query))
        if not query_terms:
            return []
        unique_query_terms = tuple(dict.fromkeys(query_terms))
        query_counts = Counter(query_terms)
        corpus_size = len(self.chunks)
        normalized_query = normalize_text(query)
        weighted_query_total = sum(
            self._idf(term, corpus_size) * count
            for term, count in query_counts.items()
        ) or 1.0

        candidates: list[tuple[float, float, tuple[str, ...], bool, int]] = []
        for index, (chunk, frequencies, length) in enumerate(
            zip(self.chunks, self._term_frequencies, self._document_lengths)
        ):
            bm25_score = 0.0
            matched_weight = 0.0
            matched_terms: list[str] = []
            for term, query_frequency in query_counts.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                idf = self._idf(term, corpus_size)
                normalization = self.k1 * (
                    1
                    - self.b
                    + self.b
                    * length
                    / max(self._average_document_length, 1.0)
                )
                bm25_score += (
                    idf
                    * (term_frequency * (self.k1 + 1))
                    / (term_frequency + normalization)
                    * min(query_frequency, 3)
                )
                matched_weight += idf * query_frequency
                matched_terms.append(term)

            if bm25_score <= 0:
                continue

            exact_match = (
                len(normalized_query) >= 3
                and normalized_query in normalize_text(chunk.text)
            )
            coverage = min(1.0, matched_weight / weighted_query_total)
            score = bm25_score + (coverage * 2.0) + (2.0 if exact_match else 0.0)
            confidence = min(
                1.0,
                0.60 * coverage
                + 0.30 * (bm25_score / (bm25_score + 3.0))
                + (0.10 if exact_match else 0.0),
            )
            visible_terms = tuple(
                term
                for term in unique_query_terms
                if term in set(matched_terms) and len(term) > 1
            )
            candidates.append(
                (score, confidence, visible_terms[:20], exact_match, index)
            )

        candidates.sort(key=lambda item: (-item[0], item[4]))
        results: list[RetrievedChunk] = []
        for rank, (score, confidence, matched, exact, index) in enumerate(
            candidates[:top_k], start=1
        ):
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[index],
                    rank=rank,
                    score=score,
                    confidence=confidence,
                    matched_terms=matched,
                    exact_match=exact,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        index_path = Path(path).expanduser().resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "algorithm": "bm25-thai-char-ngram",
            "k1": self.k1,
            "b": self.b,
            "corpus_fingerprint": self.corpus_fingerprint(),
            "chunks": [chunk.to_dict(include_text=True) for chunk in self.chunks],
        }
        temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(index_path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        index_path = Path(path).expanduser().resolve()
        if not index_path.exists():
            raise IndexFormatError(f"Index does not exist: {index_path}")
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexFormatError(f"Could not read index {index_path}: {exc}") from exc

        if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise IndexFormatError(
                f"Unsupported index schema: {payload.get('schema_version')!r}"
            )
        try:
            retriever = cls(k1=float(payload["k1"]), b=float(payload["b"]))
            chunks = [
                TextChunk(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    document=item["document"],
                    page=item.get("page"),
                    chunk_index=int(item["chunk_index"]),
                    start_char=int(item["start_char"]),
                    end_char=int(item["end_char"]),
                )
                for item in payload["chunks"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexFormatError(f"Malformed index {index_path}: {exc}") from exc
        retriever.build(chunks)
        return retriever

    def corpus_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _idf(self, term: str, corpus_size: int) -> float:
        frequency = self._document_frequencies.get(term, 0)
        return math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))


def build_index(index_path: Path) -> tuple[BM25Retriever, list[dict[str, str]]]:
    settings = get_settings()
    extraction = DocumentLoader().load(settings.documents_dir)
    chunks = DocumentChunker(
        settings.chunk_size, settings.chunk_overlap
    ).chunk_documents(extraction.documents)
    retriever = BM25Retriever()
    retriever.build(chunks)
    retriever.save(index_path)
    return retriever, [error.to_dict() for error in extraction.errors]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    default_index = settings.index_dir / "bm25_index.json"
    parser = argparse.ArgumentParser(description="Build and search the local BM25 index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Extract, chunk, and save the index.")
    index_parser.add_argument("--output", type=Path, default=default_index)

    search_parser = subparsers.add_parser("search", help="Search an existing index.")
    search_parser.add_argument("query")
    search_parser.add_argument("--index", type=Path, default=default_index)
    search_parser.add_argument("--top-k", type=int, default=settings.top_k)
    search_parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()

    if args.command == "index":
        retriever, errors = build_index(args.output)
        report = {
            "index": str(args.output.resolve()),
            "chunk_count": len(retriever.chunks),
            "fingerprint": retriever.corpus_fingerprint(),
            "extraction_errors": errors,
        }
    else:
        retriever = BM25Retriever.load(args.index)
        results = retriever.search(args.query, top_k=args.top_k)
        report = {
            "query": args.query,
            "result_count": len(results),
            "results": [
                result.to_dict(include_text=args.include_text) for result in results
            ],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
