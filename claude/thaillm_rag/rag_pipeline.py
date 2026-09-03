"""
Main RAG Pipeline for ThaiLLM
Orchestrates: Prompt Enhancement → Retrieval → Generation
"""
import os
import time
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass, field
from enum import Enum

from .config import ThaiLLMConfig, RAGConfig, load_config_from_env
from .thaillm_client import ThaiLLMClient, ChatResponse, ThaiLLMError
from .prompt_enhancer import PromptEnhancer, EnhancedQuery
from .retriever import BaseRetriever, RetrievalResult, Document, RetrievalStrategy, create_retriever
from .document_extractor import load_documents_for_rag, ThaiTextChunker, ChunkingConfig


class PipelineMode(Enum):
    """RAG pipeline operation modes"""
    SIMPLE = "simple"           # Direct query → retrieve → generate
    ENHANCED = "enhanced"       # Enhance query → retrieve → generate
    MULTI_QUERY = "multi_query" # Enhance → multiple retrievals → merge → generate


@dataclass
class RAGResponse:
    """Complete RAG pipeline response"""
    answer: str
    query: str
    enhanced_query: Optional[EnhancedQuery] = None
    retrieval_result: Optional[RetrievalResult] = None
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Pipeline execution statistics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0


class ThaiLLMRAGPipeline:
    """
    Complete RAG pipeline for ThaiLLM:
    1. (Optional) Enhance user query
    2. Retrieve relevant documents
    3. Generate answer using ThaiLLM
    """

    def __init__(
        self,
        thaillm_config: Optional[ThaiLLMConfig] = None,
        rag_config: Optional[RAGConfig] = None,
        retriever: Optional[BaseRetriever] = None,
        mode: PipelineMode = PipelineMode.ENHANCED
    ):
        # Load configs
        self.thaillm_config = thaillm_config or ThaiLLMConfig()
        self.rag_config = rag_config or RAGConfig()
        self.mode = mode

        # Initialize components
        self.client = ThaiLLMClient(self.thaillm_config)
        self.enhancer = PromptEnhancer(self.client, self.rag_config)
        self.retriever = retriever or create_retriever(RetrievalStrategy.BM25)

        # Statistics
        self.stats = PipelineStats()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever"""
        self.retriever.add_documents(documents)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Add raw texts as documents"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(content=text, metadata=metadata))
        self.add_documents(documents)

    def add_documents_from_files(
        self,
        file_paths: List[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """
        Load documents from files/URLs, chunk them, and add to the retriever.

        Args:
            file_paths: List of file paths or URLs
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap tokens between chunks

        Returns:
            Number of chunks added
        """
        docs_data = load_documents_for_rag(file_paths, chunk_size, chunk_overlap)
        self.add_texts([d["content"] for d in docs_data], [d["metadata"] for d in docs_data])
        return len(docs_data)

    def add_documents_from_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to directory
            extensions: File extensions to include (default: all supported)
            recursive: Whether to search subdirectories
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap tokens between chunks

        Returns:
            Number of chunks added
        """
        import glob

        if extensions is None:
            extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"]

        pattern = "**/*" if recursive else "*"
        file_paths = []
        for ext in extensions:
            file_paths.extend(glob.glob(os.path.join(directory, pattern + ext), recursive=recursive))

        if not file_paths:
            return 0

        return self.add_documents_from_files(file_paths, chunk_size, chunk_overlap)

    def query(self, user_query: str) -> RAGResponse:
        """
        Execute full RAG pipeline for a user query.

        Args:
            user_query: User's question

        Returns:
            RAGResponse with answer and metadata
        """
        start_total = time.perf_counter()

        try:
            # Step 1: Enhance query (if enabled)
            enhanced_query = None
            retrieval_query = user_query

            if self.mode in (PipelineMode.ENHANCED, PipelineMode.MULTI_QUERY):
                enhanced_query = self.enhancer.enhance(user_query)
                retrieval_query = enhanced_query.expanded  # Use expanded for retrieval

            # Step 2: Retrieve documents
            retrieval_result = self._retrieve(retrieval_query, enhanced_query)

            # Step 3: Generate answer
            answer = self._generate(user_query, retrieval_result)

            total_time = (time.perf_counter() - start_total) * 1000

            # Update stats
            self._update_stats(total_time, retrieval_result.retrieval_time_ms, 0, success=True)

            return RAGResponse(
                answer=answer,
                query=user_query,
                enhanced_query=enhanced_query,
                retrieval_result=retrieval_result,
                total_time_ms=total_time,
                metadata={
                    "mode": self.mode.value,
                    "num_docs_retrieved": len(retrieval_result.documents) if retrieval_result else 0,
                }
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_total) * 1000
            self._update_stats(total_time, 0, 0, success=False)
            raise

    def _retrieve(
        self,
        query: str,
        enhanced_query: Optional[EnhancedQuery] = None
    ) -> RetrievalResult:
        """Retrieve documents using single or multi-query strategy"""
        if self.mode == PipelineMode.MULTI_QUERY and enhanced_query:
            # Multi-query retrieval: retrieve for each variant, merge results
            all_docs = []
            seen_content = set()

            for variant in enhanced_query.all_variants:
                result = self.retriever.retrieve(variant, top_k=self.rag_config.top_k)
                for doc in result.documents:
                    # Deduplicate by content hash
                    content_hash = hash(doc.content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)

            # Sort by score and take top-k
            all_docs.sort(key=lambda d: d.score, reverse=True)
            top_docs = all_docs[:self.rag_config.top_k]

            return RetrievalResult(
                documents=top_docs,
                query=query,
                total_candidates=len(all_docs),
                retrieval_time_ms=sum(r.retrieval_time_ms for r in [])  # simplified
            )
        else:
            # Single query retrieval
            return self.retriever.retrieve(query, top_k=self.rag_config.top_k)

    def _generate(self, user_query: str, retrieval_result: RetrievalResult) -> str:
        """Generate answer using ThaiLLM with retrieved context"""
        start_gen = time.perf_counter()

        # Build context from retrieved documents
        context_parts = []
        total_length = 0

        for doc in retrieval_result.documents:
            doc_text = doc.content
            # Add metadata if available
            if doc.metadata:
                source = doc.metadata.get("source", "")
                if source:
                    doc_text = f"[แหล่งที่มา: {source}]\n{doc_text}"

            if total_length + len(doc_text) > self.rag_config.max_context_length:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        context = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีบริบทที่เกี่ยวข้อง"

        # Build prompt
        prompt = self.rag_config.rag_prompt_template.format(
            context=context,
            question=user_query
        )

        # Call ThaiLLM
        try:
            response = self.client.chat_structured(
                messages=[
                    {"role": "system", "content": self.rag_config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.thaillm_config.temperature,
                max_tokens=self.thaillm_config.max_tokens,
                top_p=self.thaillm_config.top_p
            )

            gen_time = (time.perf_counter() - start_gen) * 1000

            # Note: successful_queries is incremented in _update_stats, so we just calculate the average
            # We use successful_queries + 1 since _update_stats hasn't been called yet
            n = self.stats.successful_queries + 1
            self.stats.avg_generation_time_ms = (
                (self.stats.avg_generation_time_ms * (n - 1) + gen_time) / n
            )

            return response.content.strip()

        except ThaiLLMError as e:
            raise RuntimeError(f"ThaiLLM generation failed: {e}")

    def query_stream(self, user_query: str) -> Generator[str, None, RAGResponse]:
        """
        Execute RAG pipeline with streaming response.

        Yields:
            Text chunks as they're generated

        Returns:
            Final RAGResponse
        """
        start_total = time.perf_counter()

        # Step 1: Enhance query
        enhanced_query = None
        retrieval_query = user_query

        if self.mode in (PipelineMode.ENHANCED, PipelineMode.MULTI_QUERY):
            enhanced_query = self.enhancer.enhance(user_query)
            retrieval_query = enhanced_query.expanded

        # Step 2: Retrieve
        retrieval_result = self._retrieve(retrieval_query, enhanced_query)

        # Step 3: Build context and stream generation
        context_parts = []
        total_length = 0

        for doc in retrieval_result.documents:
            doc_text = doc.content
            if doc.metadata:
                source = doc.metadata.get("source", "")
                if source:
                    doc_text = f"[แหล่งที่มา: {source}]\n{doc_text}"

            if total_length + len(doc_text) > self.rag_config.max_context_length:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        context = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีบริบทที่เกี่ยวข้อง"

        prompt = self.rag_config.rag_prompt_template.format(
            context=context,
            question=user_query
        )

        # Stream from ThaiLLM
        full_answer = ""
        try:
            for chunk in self.client.chat_stream(
                messages=[
                    {"role": "system", "content": self.rag_config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.thaillm_config.temperature,
                max_tokens=self.thaillm_config.max_tokens,
                top_p=self.thaillm_config.top_p
            ):
                full_answer += chunk
                yield chunk

            total_time = (time.perf_counter() - start_total) * 1000
            self._update_stats(total_time, retrieval_result.retrieval_time_ms, 0, success=True)

            # Return final response via generator return value (Python 3.3+)
            return RAGResponse(
                answer=full_answer,
                query=user_query,
                enhanced_query=enhanced_query,
                retrieval_result=retrieval_result,
                total_time_ms=total_time,
                metadata={"mode": self.mode.value}
            )

        except ThaiLLMError as e:
            total_time = (time.perf_counter() - start_total) * 1000
            self._update_stats(total_time, 0, 0, success=False)
            raise RuntimeError(f"ThaiLLM streaming failed: {e}")

    def _update_stats(
        self,
        total_time: float,
        retrieval_time: float,
        generation_time: float,
        success: bool
    ) -> None:
        """Update pipeline statistics"""
        self.stats.total_queries += 1

        if success:
            self.stats.successful_queries += 1
            n = self.stats.successful_queries
            self.stats.avg_retrieval_time_ms = (
                (self.stats.avg_retrieval_time_ms * (n - 1) + retrieval_time) / n
            )
            self.stats.avg_total_time_ms = (
                (self.stats.avg_total_time_ms * (n - 1) + total_time) / n
            )
        else:
            self.stats.failed_queries += 1

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics"""
        return self.stats

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        return {
            "thaillm_api": self.client.health_check(),
            "retriever": len(self.retriever.documents) > 0 if hasattr(self.retriever, 'documents') else True,
        }

    def close(self):
        """Clean up resources"""
        self.client.close()


def create_rag_pipeline(
    thaillm_config: Optional[ThaiLLMConfig] = None,
    rag_config: Optional[RAGConfig] = None,
    retriever: Optional[BaseRetriever] = None,
    mode: PipelineMode = PipelineMode.ENHANCED
) -> ThaiLLMRAGPipeline:
    """Factory function to create RAG pipeline"""
    return ThaiLLMRAGPipeline(thaillm_config, rag_config, retriever, mode)