"""Grounded retrieval-to-ThaiLLM answer and citation orchestration."""

from __future__ import annotations

import re
import json
from dataclasses import replace
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.chunker import DocumentChunker
from app.config import Settings
from app.document_loader import DocumentLoader
from app.models import AnswerResult, RetrievedChunk, SourceCitation
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.preferences import AnswerOptions
from app.multilingual import normalize_query
from app.answer_language import detect_language, refusal, present, LanguageRenderingError
from app.grounding import (REFUSAL, SELECT_PROMPT, question_facets, evidence_for,
                           validate_selection, render_answer)
from app.retriever import (
    BM25Retriever,
    IndexFormatError,
    is_cross_document_query,
    preferred_documents,
)
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
        options: AnswerOptions | None = None,
    ) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise InvalidQuestionError("Question cannot be empty")
        if len(normalized_question) > 4000:
            raise InvalidQuestionError("Question cannot exceed 4000 characters")

        preferences = options or AnswerOptions()
        language = detect_language(normalized_question)
        top_k = preferences.top_k if preferences.top_k is not None else self.top_k
        threshold = max(self.relevance_threshold, 0.55) if preferences.evidence_mode == "strict" else self.relevance_threshold
        bounded_history = tuple(history[-preferences.history_messages:]) if preferences.history_messages else ()
        prior_user_questions = [
            content.strip()
            for role, content in bounded_history
            if role == "user" and content.strip()
        ][-3:]
        search_question = normalize_query(normalized_question)
        retrieval_query = normalize_query(" ".join([*prior_user_questions, normalized_question]))
        facets = question_facets(search_question)
        candidate_count = max(len(self.retriever.chunks), 40)
        candidates = self.retriever.search(
            retrieval_query,
            top_k=candidate_count,
        )
        preferred = preferred_documents(search_question)
        known_documents = {"AIT.pdf", "IT2565.pdf", "IT_inter2565.pdf", "DSBA.pdf"}
        retrieved = [r for r in candidates if not preferred or r.chunk.document in preferred
                     or (r.chunk.document not in known_documents
                         and preferred_documents(r.chunk.text) == preferred)]
        subject = re.search(r"หลักสูตร\s+([a-z][a-z0-9_-]*)", normalized_question, re.I)
        if subject and not preferred:
            phrase = subject.group(0).casefold()
            retrieved = [r for r in retrieved if phrase in r.chunk.text.casefold()
                         or Path(r.chunk.document).stem.casefold() == subject.group(1).casefold()]
        # Explicit local section evidence can pass relevance independently of an
        # aggregate BM25 score. Never manufacture a confidence score for it.
        all_evidence = [e for r in retrieved for e in evidence_for(r, facets)]
        evidence_ids = {e.result.chunk.chunk_id for e in all_evidence}
        relevant = [
            result
            for result in retrieved
            if result.chunk.chunk_id in evidence_ids
            and (preferences.evidence_mode != "strict" or result.confidence >= threshold)
        ]
        relevant_ids = {r.chunk.chunk_id for r in relevant}
        evidence = [e for e in all_evidence if e.result.chunk.chunk_id in relevant_ids]
        secret = getattr(self.provider, "api_key", "")
        if secret:
            evidence = [e for e in evidence if secret not in e.quote]
            allowed_ids = {e.result.chunk.chunk_id for e in evidence}
            relevant = [r for r in relevant if r.chunk.chunk_id in allowed_ids]
        # Bound context without silently dropping conflicts: refuse if too large.
        oversized = sum(len(e.quote) for e in evidence) > 24000 or len(evidence) > 80
        if oversized: evidence = []; relevant = []
        evidence = [replace(e, id=f"E{i+1}") for i,e in enumerate(evidence)]
        best_confidence = retrieved[0].confidence if retrieved else 0.0

        debug = {
            "query": normalized_question,
            "normalized_query": search_question,
            "retrieval_query": retrieval_query,
            "history_turn_count": len(bounded_history),
            "top_k": top_k,
            "relevance_threshold": threshold,
            "retrieved_chunks": [
                result.to_dict(include_text=True) for result in retrieved
            ],
            "selected_chunk_ids": [result.chunk.chunk_id for result in relevant],
            "final_prompt": None,
            "context_sent": None,
            "thailmm_response": None,
            "validation_errors": ["context_too_large"] if oversized else [],
            "requested_facets": facets,
            "relevance_method": "explicit_local_field_or_section_match; strict mode also requires BM25 threshold",
            "final_validated_answer": refusal(language),
            "answer_language": language,
            "cited_sources": [],
        }
        self.last_debug = debug

        if not relevant:
            if secret: self.last_debug = json.loads(json.dumps(self.last_debug).replace(secret,"[REDACTED]"))
            return AnswerResult(
                answer=refusal(language),
                sources=(),
                grounded=False,
                retrieval_confidence=best_confidence,
            )

        user_prompt = json.dumps({"question": normalized_question,
                                  "evidence": [e.public() for e in evidence]}, ensure_ascii=False)
        system_prompt = SELECT_PROMPT
        debug["final_prompt"] = {
            "system": system_prompt,
            "user": user_prompt,
        }
        debug["context_sent"] = user_prompt
        try:
            response = self.provider.answer(system_prompt, user_prompt)
        except Exception as exc:
            debug["validation_errors"] = ["provider_error:" + type(exc).__name__]
            self.last_debug = json.loads(json.dumps(debug).replace(secret,"[REDACTED]")) if secret else debug
            raise
        # Redact the configured key even if an upstream failure echoes it.
        raw = response.text.replace(secret, "[REDACTED]") if secret else response.text
        debug["thailmm_response"] = raw
        accepted, errors = validate_selection(raw, evidence)
        try:
            answer_text = present(accepted, facets, language)
        except LanguageRenderingError:
            debug.update(validation_errors=[*errors, 'verified_translation_unavailable'],
                         final_validated_answer=None, cited_sources=[])
            self.last_debug = json.loads(json.dumps(debug).replace(secret,"[REDACTED]")) if secret else debug
            raise
        cited = list({e.result.chunk.chunk_id: e.result for e in accepted}.values())
        citations = self._citations(cited)
        debug.update(validation_errors=errors, final_validated_answer=answer_text,
                               cited_sources=[c.to_dict() for c in citations])
        self.last_debug = json.loads(json.dumps(debug).replace(secret,"[REDACTED]")) if secret else debug
        return AnswerResult(
            answer=answer_text,
            sources=citations,
            grounded=bool(accepted),
            retrieval_confidence=best_confidence,
        )

    def _select_evidence(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Keep evidence focused for one curriculum or diverse for comparisons."""

        limit = self.top_k if top_k is None else top_k
        preferred = set(preferred_documents(question))
        if preferred:
            focused = [
                result
                for result in candidates
                if result.chunk.document in preferred
            ]
            if focused:
                if any(term in question for term in ("กี่ปี", "ระยะเวลา", "เรียนกี่", "อายุกี่")):
                    # Overview headings may contain PDF private-use vowel glyphs.
                    # Retain duration evidence alongside the higher-scoring credits.
                    duration = [result for result in focused if re.search(
                        r"ระยะเวล.{0,12}การศ.{0,6}กษา.{0,12}หล.{0,3}กสูตร|หล.{0,3}กสูตรปร.{0,3}ญญาตร.{0,3}\s*\d+\s*ปี",
                        result.chunk.text,
                    )]
                    if duration:
                        selected = duration[:1]
                        selected.extend(result for result in focused
                                        if result.chunk.chunk_id != selected[0].chunk.chunk_id)
                        return selected[:limit]
                return focused[:limit]

        if is_cross_document_query(question):
            selected: list[RetrievedChunk] = []
            seen_documents: set[str] = set()
            for result in candidates:
                document = result.chunk.document
                if document not in seen_documents:
                    selected.append(result)
                    seen_documents.add(document)
                if len(selected) == limit:
                    break
            if len(selected) < limit:
                selected_ids = {item.chunk.chunk_id for item in selected}
                selected.extend(
                    result
                    for result in candidates
                    if result.chunk.chunk_id not in selected_ids
                )
            return selected[:limit]

        return candidates[:limit]

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
        options: AnswerOptions | None = None,
    ) -> AnswerResult:
        return self.ensure_ready().answer(question, history=history, options=options)

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
