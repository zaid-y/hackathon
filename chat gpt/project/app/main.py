"""FastAPI entry point for the ThaiLLM document assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.answer import InvalidQuestionError, RAGRuntime
from app.chunker import ChunkingConfigurationError, DocumentChunker
from app.config import get_settings
from app.document_loader import DocumentFolderError, DocumentLoader
from app.thailmm import (
    ThaiLLMAdapterRequiredError,
    ThaiLLMConfigurationError,
    ThaiLLMMalformedResponseError,
    ThaiLLMRateLimitError,
    ThaiLLMServiceError,
    ThaiLLMTimeoutError,
)


settings = get_settings()
loader = DocumentLoader()
chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
runtime = RAGRuntime(settings)
frontend_dir = settings.project_root / "frontend"

app = FastAPI(
    title="ThaiLLM Document Assistant",
    description="Grounded local-document RAG API using ThaiLLM",
    version="0.7.0",
)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list["ChatMessage"] = Field(default_factory=list, max_length=20)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "7-user-interface"}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/status")
def api_status() -> dict:
    provider_configured = bool(getattr(runtime.provider, "is_configured", True))
    return {
        "status": "ready" if runtime.index_path.exists() else "needs_index",
        "index_exists": runtime.index_path.exists(),
        "thailmm_configured": provider_configured,
        "documents_dir": str(settings.documents_dir),
        "debug": settings.debug,
    }


@app.post("/api/ask")
def ask(request: QuestionRequest) -> dict:
    try:
        history = [(message.role, message.content) for message in request.history]
        return runtime.answer(request.question, history=history).to_dict()
    except InvalidQuestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ThaiLLMConfigurationError, ThaiLLMAdapterRequiredError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ThaiLLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ThaiLLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ThaiLLMServiceError, ThaiLLMMalformedResponseError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/admin/reindex")
def reindex() -> dict:
    try:
        return runtime.reindex()
    except (DocumentFolderError, ChunkingConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/admin/debug")
def debug_snapshot() -> dict:
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Debug mode is disabled")
    return runtime.debug_snapshot or {"message": "No question has been processed yet"}


@app.get("/admin/documents")
def inspect_documents(
    include_text: bool = Query(
        default=False,
        description="Include extracted text. Keep false for a compact report.",
    ),
) -> dict:
    """Run extraction and return metadata plus safe per-file errors."""

    try:
        return loader.load(settings.documents_dir).to_dict(include_text=include_text)
    except DocumentFolderError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/admin/chunks")
def inspect_chunks(
    include_text: bool = Query(
        default=False,
        description="Include chunk text. Keep false for a compact report.",
    ),
) -> dict:
    """Extract and chunk configured documents for retrieval debugging."""

    try:
        extraction = loader.load(settings.documents_dir)
        chunks = chunker.chunk_documents(extraction.documents)
        return {
            "settings": {
                "chunk_size": chunker.chunk_size,
                "chunk_overlap": chunker.chunk_overlap,
            },
            "document_count": len(extraction.documents),
            "chunk_count": len(chunks),
            "chunks": [
                chunk.to_dict(include_text=include_text) for chunk in chunks
            ],
            "extraction_errors": [
                error.to_dict() for error in extraction.errors
            ],
        }
    except (DocumentFolderError, ChunkingConfigurationError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
