"""
ThaiLLM RAG Pipeline
A complete RAG system for ThaiLLM with prompt enhancement, retrieval, and generation.
"""

from .config import ThaiLLMConfig, RAGConfig, load_config_from_env
from .thaillm_client import ThaiLLMClient, ThaiLLMError, ChatResponse
from .prompt_enhancer import PromptEnhancer, EnhancedQuery
from .retriever import (
    Document,
    RetrievalResult,
    BaseRetriever,
    KeywordRetriever,
    BM25Retriever,
    VectorRetriever,
    HybridRetriever,
    RetrievalStrategy,
    create_retriever
)
from .rag_pipeline import (
    ThaiLLMRAGPipeline,
    RAGResponse,
    PipelineStats,
    PipelineMode,
    create_rag_pipeline
)
from .document_extractor import (
    ExtractedDocument,
    ChunkingConfig,
    ThaiTokenizer,
    ThaiTextChunker,
    DocumentExtractor,
    PDFExtractor,
    DocxExtractor,
    TextExtractor,
    HTMLExtractor,
    WebExtractor,
    create_extractor,
    extract_documents,
    extract_multiple_sources,
    load_documents_for_rag,
    HAS_PDFPLUMBER,
    HAS_PYMUPDF,
    HAS_DOCX,
    HAS_BS4,
    HAS_PYTHAINLP,
    HAS_DEEPCUT,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "ThaiLLMConfig",
    "RAGConfig",
    "load_config_from_env",
    # Client
    "ThaiLLMClient",
    "ThaiLLMError",
    "ChatResponse",
    # Prompt Enhancement
    "PromptEnhancer",
    "EnhancedQuery",
    # Retrieval
    "Document",
    "RetrievalResult",
    "BaseRetriever",
    "KeywordRetriever",
    "BM25Retriever",
    "VectorRetriever",
    "HybridRetriever",
    "RetrievalStrategy",
    "create_retriever",
    # Pipeline
    "ThaiLLMRAGPipeline",
    "RAGResponse",
    "PipelineStats",
    "PipelineMode",
    "create_rag_pipeline",
    # Document Extraction
    "ExtractedDocument",
    "ChunkingConfig",
    "ThaiTokenizer",
    "ThaiTextChunker",
    "DocumentExtractor",
    "PDFExtractor",
    "DocxExtractor",
    "TextExtractor",
    "HTMLExtractor",
    "WebExtractor",
    "create_extractor",
    "extract_documents",
    "extract_multiple_sources",
    "load_documents_for_rag",
    # Feature flags
    "HAS_PDFPLUMBER",
    "HAS_PYMUPDF",
    "HAS_DOCX",
    "HAS_BS4",
    "HAS_PYTHAINLP",
    "HAS_DEEPCUT",
]