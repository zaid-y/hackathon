"""
FastAPI Server for ThaiLLM RAG Pipeline
Provides REST API and WebSocket for frontend integration
"""
import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import ThaiLLMConfig, RAGConfig, load_config_from_env
from .rag_pipeline import (
    ThaiLLMRAGPipeline,
    PipelineMode,
    RAGResponse,
    create_rag_pipeline,
    Document
)
from .retriever import RetrievalStrategy
from .document_extractor import load_documents_for_rag


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[str] = "enhanced"
    top_k: Optional[int] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0
    retrieval_passed_threshold: bool = True
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    metadata: Dict[str, Any] = {}


class AddDocumentsRequest(BaseModel):
    file_paths: List[str]
    chunk_size: int = 500
    chunk_overlap: int = 50


class AddTextsRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str
    thaillm_api: bool
    documents_count: int
    competition_mode: bool
    debug_mode: bool


class ConfigResponse(BaseModel):
    top_k: int
    similarity_threshold: float
    bm25_threshold: float
    max_context_length: int
    enhance_prompts: bool
    model: str
    competition_mode: bool
    debug_mode: bool


# Global pipeline instance
pipeline: Optional[ThaiLLMRAGPipeline] = None


def get_pipeline() -> ThaiLLMRAGPipeline:
    """Get or create the global pipeline instance"""
    global pipeline
    if pipeline is None:
        thaillm_config, rag_config = load_config_from_env()
        pipeline = create_rag_pipeline(
            thaillm_config=thaillm_config,
            rag_config=rag_config,
            mode=PipelineMode.ENHANCED
        )
    return pipeline


# FastAPI app
app = FastAPI(
    title="ThaiLLM RAG API",
    description="RAG Pipeline for ThaiLLM - Competition Ready",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    thaillm_config, rag_config = load_config_from_env()
    pipeline = create_rag_pipeline(
        thaillm_config=thaillm_config,
        rag_config=rag_config,
        mode=PipelineMode.ENHANCED
    )
    print(f"ThaiLLM RAG Pipeline started - Competition mode: {pipeline.competition_mode}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pipeline
    if pipeline:
        pipeline.close()
        pipeline = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("""
    <html>
        <head><title>ThaiLLM RAG</title></head>
        <body>
            <h1>ThaiLLM RAG API</h1>
            <p>Frontend not found. Please build the frontend first.</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    p = get_pipeline()
    health = p.health_check()
    return HealthResponse(
        status="healthy" if health.get("thaillm_api", False) else "degraded",
        thaillm_api=health.get("thaillm_api", False),
        documents_count=len(p.retriever.documents) if hasattr(p.retriever, 'documents') else 0,
        competition_mode=p.competition_mode,
        debug_mode=p.debug_mode
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    p = get_pipeline()
    return ConfigResponse(
        top_k=p.rag_config.top_k,
        similarity_threshold=p.rag_config.similarity_threshold,
        bm25_threshold=p.rag_config.bm25_threshold,
        max_context_length=p.rag_config.max_context_length,
        enhance_prompts=p.rag_config.enhance_prompts,
        model=p.thaillm_config.model,
        competition_mode=p.competition_mode,
        debug_mode=p.debug_mode
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a RAG query"""
    p = get_pipeline()

    # Override top_k if provided
    original_top_k = p.rag_config.top_k
    if request.top_k:
        p.rag_config.top_k = request.top_k

    try:
        # Set mode if provided
        if request.mode:
            try:
                p.mode = PipelineMode(request.mode)
            except ValueError:
                pass

        response: RAGResponse = p.query(request.query)

        return QueryResponse(
            answer=response.answer,
            query=response.query,
            sources=response.sources,
            confidence=response.confidence,
            retrieval_passed_threshold=response.retrieval_passed_threshold,
            total_time_ms=response.total_time_ms,
            retrieval_time_ms=response.retrieval_result.retrieval_time_ms if response.retrieval_result else 0,
            metadata=response.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        p.rag_config.top_k = original_top_k


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Execute a RAG query with streaming response"""
    from fastapi.responses import StreamingResponse

    p = get_pipeline()

    # Override top_k if provided
    original_top_k = p.rag_config.top_k
    if request.top_k:
        p.rag_config.top_k = request.top_k

    async def generate():
        try:
            if request.mode:
                try:
                    p.mode = PipelineMode(request.mode)
                except ValueError:
                    pass

            # We need to run the streaming query
            # Since query_stream is a generator, we'll wrap it
            for chunk in p.query_stream(request.query):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            p.rag_config.top_k = original_top_k

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/documents")
async def add_documents(request: AddDocumentsRequest):
    """Add documents from file paths"""
    p = get_pipeline()
    try:
        count = p.add_documents_from_files(
            request.file_paths,
            request.chunk_size,
            request.chunk_overlap
        )
        return {"added": count, "message": f"Added {count} document chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/texts")
async def add_texts(request: AddTextsRequest):
    """Add raw texts as documents"""
    p = get_pipeline()
    try:
        p.add_texts(request.texts, request.metadatas)
        return {"added": len(request.texts), "message": f"Added {len(request.texts)} document chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/directory")
async def add_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """Add all documents from a directory"""
    p = get_pipeline()
    try:
        count = p.add_documents_from_directory(
            directory, extensions, recursive, chunk_size, chunk_overlap
        )
        return {"added": count, "message": f"Added {count} document chunks from {directory}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents")
async def clear_documents():
    """Clear all documents from the index"""
    p = get_pipeline()
    p.retriever.clear()
    return {"message": "All documents cleared"}


@app.get("/api/stats")
async def get_stats():
    """Get pipeline statistics"""
    p = get_pipeline()
    stats = p.get_stats()
    return {
        "total_queries": stats.total_queries,
        "successful_queries": stats.successful_queries,
        "failed_queries": stats.failed_queries,
        "avg_retrieval_time_ms": stats.avg_retrieval_time_ms,
        "avg_generation_time_ms": stats.avg_generation_time_ms,
        "avg_total_time_ms": stats.avg_total_time_ms,
        "documents_count": len(p.retriever.documents) if hasattr(p.retriever, 'documents') else 0
    }


# WebSocket for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    p = get_pipeline()

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            query = request.get("query", "")
            mode = request.get("mode", "enhanced")
            top_k = request.get("top_k")

            if top_k:
                p.rag_config.top_k = top_k

            try:
                if mode:
                    p.mode = PipelineMode(mode)
            except ValueError:
                pass

            # Send start signal
            await websocket.send_text(json.dumps({"type": "start", "query": query}))

            # Stream response
            full_answer = ""
            retrieval_result = None

            for chunk in p.query_stream(query):
                full_answer += chunk
                await websocket.send_text(json.dumps({"type": "chunk", "content": chunk}))

            # Get final response data
            response: RAGResponse = p.query(query)

            # Send completion with sources
            await websocket.send_text(json.dumps({
                "type": "complete",
                "answer": full_answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "retrieval_passed_threshold": response.retrieval_passed_threshold,
                "total_time_ms": response.total_time_ms,
                "retrieval_time_ms": response.retrieval_result.retrieval_time_ms if response.retrieval_result else 0
            }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        manager.disconnect(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    import uvicorn
    uvicorn.run("thaillm_rag.api_server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ThaiLLM RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_server()