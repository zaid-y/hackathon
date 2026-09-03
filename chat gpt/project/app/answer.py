"""Grounded retrieval-to-ThaiLLM answer and citation orchestration."""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.chunker import DocumentChunker
from app.config import Settings
from app.document_loader import DocumentLoader
from app.models import AnswerResult, RetrievedChunk, SourceCitation
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retriever import BM25Retriever, IndexFormatError
from app.thailmm import ThaiLLMClient, ThaiLLMProvider


INSUFFICIENT_INFORMATION_MESSAGE = (
    "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่กำหนด"
)
_SOURCE_MARKER_PATTERN = re.compile(
    r"[ \t]*\[\s*SOURCE\s+\d+\s*\]", re.IGNORECASE
)


def hide_source_markers(text: str) -> str:
    """Remove model-facing source labels from the user-facing answer."""

    cleaned = _SOURCE_MARKER_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


class InvalidQuestionError(ValueError):
    """Raised when a question is empty or unreasonably large."""


class AnswerService:
    """Retrieve evidence, apply a confidence gate, call ThaiLLM, cite metadata."""

    def __init__(
        self,
        *,
        retriever: BM25Retriever,
        provider: ThaiLLMProvider,
        top_k: int = 5,
        relevance_threshold: float = 0.35,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0 <= relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between zero and one")
        self.retriever = retriever
        self.provider = provider
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.last_debug: dict[str, Any] | None = None

    def answer(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise InvalidQuestionError("Question cannot be empty")
        if len(normalized_question) > 4000:
            raise InvalidQuestionError("Question cannot exceed 4000 characters")

        bounded_history = tuple(history[-12:])
        prior_user_questions = [
            content.strip()
            for role, content in bounded_history
            if role == "user" and content.strip()
        ][-3:]
        retrieval_query = " ".join([*prior_user_questions, normalized_question])
        retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        relevant = [
            result
            for result in retrieved
            if result.confidence >= self.relevance_threshold
        ]
        best_confidence = retrieved[0].confidence if retrieved else 0.0

        self.last_debug = {
            "query": normalized_question,
            "retrieval_query": retrieval_query,
            "history_turn_count": len(bounded_history),
            "top_k": self.top_k,
            "relevance_threshold": self.relevance_threshold,
            "retrieved_chunks": [
                result.to_dict(include_text=True) for result in retrieved
            ],
            "selected_chunk_ids": [result.chunk.chunk_id for result in relevant],
            "final_prompt": None,
        }

        if not relevant:
            return AnswerResult(
                answer=INSUFFICIENT_INFORMATION_MESSAGE,
                sources=(),
                grounded=False,
                retrieval_confidence=best_confidence,
            )

        user_prompt = build_user_prompt(
            normalized_question,
            relevant,
            history=bounded_history,
        )
        self.last_debug["final_prompt"] = {
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
        }
        response = self.provider.answer(SYSTEM_PROMPT, user_prompt)
        return AnswerResult(
            answer=hide_source_markers(response.text),
            sources=self._citations(relevant),
            grounded=True,
            retrieval_confidence=best_confidence,
        )

    @staticmethod
    def _citations(results: list[RetrievedChunk]) -> tuple[SourceCitation, ...]:
        grouped: dict[tuple[str, int | None], list[str]] = {}
        for result in results:
            key = (result.chunk.document, result.chunk.page)
            grouped.setdefault(key, []).append(result.chunk.chunk_id)
        return tuple(
            SourceCitation(document=document, page=page, chunk_ids=tuple(chunk_ids))
            for (document, page), chunk_ids in grouped.items()
        )


class RAGRuntime:
    """Load the saved index once and coordinate explicit reindex operations."""

    def __init__(
        self,
        settings: Settings,
        provider: ThaiLLMProvider | None = None,
    ) -> None:
        self.settings = settings
        self.index_path = settings.index_dir / "bm25_index.json"
        self.provider = provider or ThaiLLMClient(
            api_key=settings.thailmm_api_key,
            base_url=settings.thailmm_base_url,
            model=settings.thailmm_model,
            timeout_seconds=settings.thailmm_timeout_seconds,
        )
        self._retriever: BM25Retriever | None = None
        self._answer_service: AnswerService | None = None
        self._lock = threading.Lock()
        self.last_index_report: dict[str, Any] | None = None

    def ensure_ready(self) -> AnswerService:
        if self._answer_service is not None:
            return self._answer_service
        with self._lock:
            if self._answer_service is None:
                try:
                    self._retriever = BM25Retriever.load(self.index_path)
                except IndexFormatError:
                    self._rebuild_locked()
                self._answer_service = self._new_answer_service()
        return self._answer_service

    def reindex(self) -> dict[str, Any]:
        with self._lock:
            self._rebuild_locked()
            self._answer_service = self._new_answer_service()
            return dict(self.last_index_report or {})

    def answer(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> AnswerResult:
        return self.ensure_ready().answer(question, history=history)

    @property
    def debug_snapshot(self) -> dict[str, Any] | None:
        service = self._answer_service
        return service.last_debug if service is not None else None

    @property
    def index_ready(self) -> bool:
        return self._retriever is not None and self._retriever.is_ready

    def _rebuild_locked(self) -> None:
        extraction = DocumentLoader().load(self.settings.documents_dir)
        chunks = DocumentChunker(
            self.settings.chunk_size, self.settings.chunk_overlap
        ).chunk_documents(extraction.documents)
        retriever = BM25Retriever()
        retriever.build(chunks)
        retriever.save(self.index_path)
        self._retriever = retriever
        self.last_index_report = {
            "index_path": str(self.index_path),
            "document_count": len(extraction.documents),
            "chunk_count": len(chunks),
            "corpus_fingerprint": retriever.corpus_fingerprint(),
            "extraction_errors": [error.to_dict() for error in extraction.errors],
        }

    def _new_answer_service(self) -> AnswerService:
        return AnswerService(
            retriever=self._retriever or BM25Retriever(),
            provider=self.provider,
            top_k=self.settings.top_k,
            relevance_threshold=self.settings.relevance_threshold,
        )
