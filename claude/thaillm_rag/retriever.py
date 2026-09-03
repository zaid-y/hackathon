"""
Document Retrieval Module for ThaiLLM RAG
Supports multiple retrieval strategies: BM25 (default), TF-IDF keyword, vector, hybrid
"""
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Try to import Thai word segmentation libraries
try:
    from pythainlp.tokenize import word_tokenize
    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False

try:
    import deepcut
    HAS_DEEPCUT = True
except ImportError:
    HAS_DEEPCUT = False


@dataclass
class Document:
    """Document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of retrieval operation"""
    documents: List[Document]
    query: str
    total_candidates: int
    retrieval_time_ms: float


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR = "vector"
    KEYWORD = "keyword"        # TF-IDF (existing)
    BM25 = "bm25"              # BM25 (recommended for keyword retrieval)
    HYBRID = "hybrid"


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve relevant documents for query"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index"""
        pass


class KeywordRetriever(BaseRetriever):
    """
    Simple keyword-based retriever using TF-IDF-like scoring.
    Good baseline without requiring vector embeddings.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self.documents: List[Document] = []
        self.tokenizer = tokenizer or self._default_tokenizer
        self._term_freq: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: freq}
        self._doc_freq: Dict[str, int] = {}  # term -> num docs containing term
        self._doc_lengths: Dict[int, int] = {}

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer for Thai + English with proper Thai word segmentation"""
        # Try pythainlp first (best for Thai)
        if HAS_PYTHAINLP:
            try:
                tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
                return [t.lower() for t in tokens if len(t) > 1]
            except Exception:
                pass

        # Try deepcut as fallback
        if HAS_DEEPCUT:
            try:
                tokens = deepcut.tokenize(text)
                return [t.lower() for t in tokens if len(t) > 1]
            except Exception:
                pass

        # Fallback: simple character/word splitting
        tokens = re.findall(r'[฀-๿]+|[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and rebuild index"""
        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            doc.id = str(doc_id)
            self.documents.append(doc)

            # Index terms
            tokens = self.tokenizer(doc.content)
            self._doc_lengths[doc_id] = len(tokens)

            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            for term, freq in term_counts.items():
                if term not in self._term_freq:
                    self._term_freq[term] = {}
                self._term_freq[term][doc_id] = freq
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve using TF-IDF scoring"""
        import time
        start = time.perf_counter()

        query_tokens = self._default_tokenizer(query)
        if not query_tokens:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000)

        N = len(self.documents)
        if N == 0:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000)

        # Score documents
        scores: Dict[int, float] = {}

        for token in query_tokens:
            if token not in self._term_freq:
                continue

            idf = 1.0
            if self._doc_freq[token] > 0:
                import math
                idf = math.log(N / self._doc_freq[token])

            for doc_id, tf in self._term_freq[token].items():
                # TF-IDF with length normalization
                tf_normalized = tf / max(self._doc_lengths.get(doc_id, 1), 1)
                scores[doc_id] = scores.get(doc_id, 0) + tf_normalized * idf

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id < len(self.documents):
                doc = self.documents[doc_id]
                doc.score = score
                results.append(doc)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(results, query, N, elapsed)

    def clear(self) -> None:
        """Clear index"""
        self.documents = []
        self._term_freq = {}
        self._doc_freq = {}
        self._doc_lengths = {}


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever - probabilistic ranking function.
    Better than TF-IDF for keyword retrieval as it handles term saturation and document length normalization.
    No external dependencies - pure Python implementation.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.documents: List[Document] = []
        self.tokenizer = tokenizer or self._default_tokenizer
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Document length normalization parameter

        # BM25 index structures
        self._term_freq: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: freq}
        self._doc_freq: Dict[str, int] = {}              # term -> num docs containing term
        self._doc_lengths: Dict[int, int] = {}           # doc_id -> num tokens
        self._avg_doc_length: float = 0.0

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer for Thai + English"""
        if HAS_PYTHAINLP:
            try:
                tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
                return [t.lower() for t in tokens if len(t) > 1]
            except Exception:
                pass

        if HAS_DEEPCUT:
            try:
                tokens = deepcut.tokenize(text)
                return [t.lower() for t in tokens if len(t) > 1]
            except Exception:
                pass

        # Fallback
        tokens = re.findall(r'[฀-๿]+|[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and rebuild BM25 index"""
        start_id = len(self.documents)

        for i, doc in enumerate(documents):
            doc_id = start_id + i
            doc.id = str(doc_id)
            self.documents.append(doc)

            # Index terms
            tokens = self.tokenizer(doc.content)
            self._doc_lengths[doc_id] = len(tokens)

            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            for term, freq in term_counts.items():
                if term not in self._term_freq:
                    self._term_freq[term] = {}
                self._term_freq[term][doc_id] = freq
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        # Update average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def _bm25_score(self, query_tokens: List[str], doc_id: int) -> float:
        """Calculate BM25 score for a document"""
        import math

        score = 0.0
        doc_len = self._doc_lengths.get(doc_id, 1)

        for token in query_tokens:
            if token not in self._term_freq:
                continue

            tf = self._term_freq[token].get(doc_id, 0)
            if tf == 0:
                continue

            # IDF component
            df = self._doc_freq[token]
            N = len(self.documents)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # TF component with saturation
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length))

            score += idf * tf_component

        return score

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve using BM25 scoring"""
        import time
        start = time.perf_counter()

        query_tokens = self._default_tokenizer(query)
        if not query_tokens:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000)

        N = len(self.documents)
        if N == 0:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000)

        # Score all documents
        scores: Dict[int, float] = {}
        for doc_id in range(N):
            scores[doc_id] = self._bm25_score(query_tokens, doc_id)

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            if doc_id < len(self.documents):
                doc = self.documents[doc_id]
                doc.score = score
                results.append(doc)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(results, query, N, elapsed)

    def clear(self) -> None:
        """Clear index"""
        self.documents = []
        self._term_freq = {}
        self._doc_freq = {}
        self._doc_lengths = {}
        self._avg_doc_length = 0.0


class VectorRetriever(BaseRetriever):
    """
    Vector-based retriever using sentence embeddings.
    Requires sentence-transformers or compatible embedding model.
    """

    def __init__(
        self,
        embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        similarity_fn: Optional[Callable[[List[float], List[float]], float]] = None
    ):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.embedder = embedder
        self.similarity_fn = similarity_fn or self._cosine_similarity

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity"""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and compute embeddings"""
        if not self.embedder:
            raise ValueError("Embedder function required for VectorRetriever")

        texts = [doc.content for doc in documents]
        new_embeddings = self.embedder(texts)

        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            doc.id = str(start_id + i)
            self.documents.append(doc)
            self.embeddings.append(new_embeddings[i])

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve using vector similarity"""
        import time
        start = time.perf_counter()

        if not self.embedder or not self.documents:
            return RetrievalResult([], query, len(self.documents), (time.perf_counter() - start) * 1000)

        query_embedding = self.embedder([query])[0]

        # Compute similarities
        similarities = [
            (i, self.similarity_fn(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:top_k]:
            doc = self.documents[idx]
            doc.score = score
            results.append(doc)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(results, query, len(self.documents), elapsed)

    def clear(self) -> None:
        """Clear index"""
        self.documents = []
        self.embeddings = []


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining keyword and vector scores.
    """

    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        vector_retriever: Optional[VectorRetriever] = None,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight

    def add_documents(self, documents: List[Document]) -> None:
        """Add to both retrievers"""
        self.keyword_retriever.add_documents(documents)
        if self.vector_retriever:
            self.vector_retriever.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve using combined scores"""
        import time
        start = time.perf_counter()

        # Get results from both retrievers
        kw_result = self.keyword_retriever.retrieve(query, top_k * 2)  # Get more for merging

        if self.vector_retriever:
            vec_result = self.vector_retriever.retrieve(query, top_k * 2)
        else:
            vec_result = RetrievalResult([], query, 0, 0)

        # Combine scores
        combined_scores: Dict[str, float] = {}
        combined_docs: Dict[str, Document] = {}

        # Keyword scores
        for doc in kw_result.documents:
            doc_id = doc.id or str(id(doc))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + doc.score * self.keyword_weight
            combined_docs[doc_id] = doc

        # Vector scores
        for doc in vec_result.documents:
            doc_id = doc.id or str(id(doc))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + doc.score * self.vector_weight
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc

        # Sort and return top-k
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_ids[:top_k]:
            doc = combined_docs[doc_id]
            doc.score = score
            results.append(doc)

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(results, query, len(self.keyword_retriever.documents), elapsed)

    def clear(self) -> None:
        """Clear both retrievers"""
        self.keyword_retriever.clear()
        if self.vector_retriever:
            self.vector_retriever.clear()


def create_retriever(
    strategy: RetrievalStrategy = RetrievalStrategy.BM25,
    embedder: Optional[Callable] = None,
    **kwargs
) -> BaseRetriever:
    """Factory function to create retriever by strategy"""
    if strategy == RetrievalStrategy.BM25:
        return BM25Retriever(**kwargs)
    elif strategy == RetrievalStrategy.KEYWORD:
        return KeywordRetriever()
    elif strategy == RetrievalStrategy.VECTOR:
        if not embedder:
            raise ValueError("Embedder required for vector retrieval")
        return VectorRetriever(embedder=embedder)
    elif strategy == RetrievalStrategy.HYBRID:
        # Default to BM25 for keyword component
        kw = BM25Retriever(**kwargs)
        vec = VectorRetriever(embedder=embedder) if embedder else None
        return HybridRetriever(kw, vec)
    else:
        return BM25Retriever(**kwargs)