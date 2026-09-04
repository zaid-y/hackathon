"""
Document Retrieval Module for ThaiLLM RAG
Supports multiple retrieval strategies: BM25 (default), TF-IDF keyword, vector, hybrid
Optimized for Thai language with hybrid scoring: BM25 + exact phrase + keyword + metadata + number/date matching
"""
import re
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

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
    scores: Dict[str, float] = field(default_factory=dict)  # doc_id -> component scores breakdown


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR = "vector"
    KEYWORD = "keyword"        # TF-IDF (existing)
    BM25 = "bm25"              # BM25 (recommended for keyword retrieval)
    HYBRID = "hybrid"


class ThaiQueryProcessor:
    """
    Thai-aware query processor for extracting search signals.
    Handles Thai/English mixed queries, numbers, dates, names, exact phrases.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self.tokenizer = tokenizer or self._default_tokenizer

        # Thai question words to optionally down-weight
        self.question_words = {
            'คือ', 'อะไร', 'ที่', 'ซึ่ง', 'ได้', 'จะ', 'มี', 'เป็น', 'อย่างไร',
            'ทำไม', 'เมื่อไหร่', 'ที่ไหน', 'ใคร', 'กี่', 'เท่าไหร่', 'เท่าไร',
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'the', 'a', 'an', 'is', 'are', 'can', 'do', 'does'
        }

        # Thai stopwords for keyword extraction
        self.stopwords = self.question_words.union({
            'และ', 'หรือ', 'แต่', 'เพราะ', 'ถ้า', 'ว่า', 'ใน', 'ของ', 'จาก', 'ไป', 'มา',
            'กับ', 'เพื่อ', 'เกี่ยวกับ', '關於', '关于', 'vs', 'vs.'
        })

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
        # Fallback: simple character/word splitting
        tokens = re.findall(r'[฀-๿]+|[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/tokens"""
        return self.tokenizer(text)

    def extract_signals(self, query: str) -> Dict[str, Any]:
        """
        Extract useful search signals from a user query.
        Returns structured signals for hybrid retrieval.
        """
        tokens = self.tokenize(query)

        # Extract exact phrases (quoted terms)
        exact_phrases = re.findall(r'"([^"]+)"|"([^"]+)"|' + "'([^']+)'", query)
        exact_phrases = [p for group in exact_phrases for p in group if p]

        # Extract numbers (including Thai numbers)
        numbers = re.findall(r'\d+(?:[.,]\d+)?|[๐-๙]+', query)

        # Extract dates (various formats)
        dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s*(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s*\d{4}|\d{1,2}:\d{2}', query)

        # Extract potential names (capitalized words in English, or Thai names patterns)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

        # Extract English terms inside Thai text
        english_terms = re.findall(r'\b[a-zA-Z]{2,}\b', query)

        # Extract keywords (non-stopwords)
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        # Extract Thai number words
        thai_numbers = re.findall(r'(?:ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน)+', query)

        return {
            "original_query": query,
            "tokens": tokens,
            "keywords": keywords,
            "exact_phrases": exact_phrases,
            "numbers": numbers,
            "dates": dates,
            "names": names,
            "english_terms": english_terms,
            "thai_numbers": thai_numbers,
            "question_type": self._detect_question_type(query)
        }

    def _detect_question_type(self, query: str) -> str:
        """Detect question type for potential routing"""
        query_lower = query.lower()
        if any(w in query_lower for w in ['กี่', 'เท่าไหร่', 'เท่าไร', 'how many', 'how much', 'จำนวน', 'เปอร์เซ็นต์', '%', 'percent']):
            return "number"
        if any(w in query_lower for w in ['เมื่อไหร่', 'วันที่', 'เวลา', 'when', 'date', 'time']):
            return "date"
        if any(w in query_lower for w in ['ใคร', 'who', 'ชื่อ']):
            return "name"
        if any(w in query_lower for w in ['ที่ไหน', 'where', 'สถานที่', 'location']):
            return "location"
        if any(w in query_lower for w in ['ทำไม', 'why', 'เหตุผล', 'reason']):
            return "reason"
        if any(w in query_lower for w in ['อย่างไร', 'how', 'วิธี', 'method', 'ขั้นตอน', 'steps']):
            return "procedure"
        if any(w in query_lower for w in ['คืออะไร', 'what is', 'what are', 'คือ', 'หมายถึง', 'definition']):
            return "definition"
        if any(w in query_lower for w in ['ต่างกัน', 'แตกต่าง', 'เปรียบเทียบ', 'compare', 'difference', 'vs']):
            return "comparison"
        if any(w in query_lower for w in ['รายการ', 'ลิสต์', 'list', 'มีอะไรบ้าง', 'what are the']):
            return "list"
        return "general"


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


class EnhancedBM25Retriever(BM25Retriever):
    """
    Enhanced BM25 retriever with hybrid scoring for Thai language.
    Combines:
    - BM25 score (base)
    - Exact phrase matching bonus
    - Keyword overlap bonus
    - Metadata/title matching bonus
    - Number/date matching bonus
    - English term matching bonus
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        # Hybrid scoring weights
        bm25_weight: float = 1.0,
        phrase_weight: float = 2.0,
        keyword_weight: float = 0.5,
        metadata_weight: float = 1.0,
        number_date_weight: float = 3.0,
        english_weight: float = 1.5
    ):
        super().__init__(tokenizer, k1, b)
        self.query_processor = ThaiQueryProcessor(tokenizer)
        # Hybrid weights
        self.bm25_weight = bm25_weight
        self.phrase_weight = phrase_weight
        self.keyword_weight = keyword_weight
        self.metadata_weight = metadata_weight
        self.number_date_weight = number_date_weight
        self.english_weight = english_weight

    def _calculate_phrase_bonus(self, query_signals: Dict[str, Any], doc: Document) -> float:
        """Calculate bonus for exact phrase matches"""
        bonus = 0.0
        content_lower = doc.content.lower()

        for phrase in query_signals.get("exact_phrases", []):
            if phrase.lower() in content_lower:
                # Bonus scales with phrase length
                bonus += self.phrase_weight * len(phrase.split())

        return bonus

    def _calculate_keyword_bonus(self, query_signals: Dict[str, Any], doc: Document) -> float:
        """Calculate bonus for keyword overlap (beyond BM25)"""
        keywords = query_signals.get("keywords", [])
        if not keywords:
            return 0.0

        content_tokens = set(self.tokenizer(doc.content.lower()))
        keyword_set = set(keywords)
        overlap = len(content_tokens & keyword_set)

        # Bonus per unique keyword match
        return self.keyword_weight * overlap

    def _calculate_metadata_bonus(self, query_signals: Dict[str, Any], doc: Document) -> float:
        """Calculate bonus for metadata matches (source name, title, etc.)"""
        bonus = 0.0
        metadata_text = ""

        # Collect searchable metadata fields
        for key in ["source_name", "title", "source", "file_name", "filename", "source_path"]:
            if key in doc.metadata and doc.metadata[key]:
                metadata_text += " " + str(doc.metadata[key]).lower()

        if not metadata_text:
            return 0.0

        keywords = query_signals.get("keywords", [])
        metadata_tokens = set(self.tokenizer(metadata_text))
        keyword_set = set(keywords)
        overlap = len(metadata_tokens & keyword_set)

        return self.metadata_weight * overlap

    def _calculate_number_date_bonus(self, query_signals: Dict[str, Any], doc: Document) -> float:
        """Calculate bonus for exact number/date matches (critical for accuracy)"""
        bonus = 0.0
        content = doc.content

        # Check numbers
        query_numbers = query_signals.get("numbers", [])
        query_thai_numbers = query_signals.get("thai_numbers", [])
        all_query_numbers = query_numbers + query_thai_numbers

        for num in all_query_numbers:
            if num and num in content:
                bonus += self.number_date_weight

        # Check dates
        query_dates = query_signals.get("dates", [])
        for date in query_dates:
            if date and date in content:
                bonus += self.number_date_weight

        return bonus

    def _calculate_english_bonus(self, query_signals: Dict[str, Any], doc: Document) -> float:
        """Calculate bonus for English term matches inside Thai documents"""
        english_terms = query_signals.get("english_terms", [])
        if not english_terms:
            return 0.0

        content_lower = doc.content.lower()
        bonus = 0.0

        for term in english_terms:
            if term.lower() in content_lower:
                bonus += self.english_weight

        return bonus

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve using enhanced hybrid BM25 scoring"""
        import time
        start = time.perf_counter()

        query_signals = self.query_processor.extract_signals(query)
        query_tokens = query_signals["tokens"]

        if not query_tokens:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000, scores={})

        N = len(self.documents)
        if N == 0:
            return RetrievalResult([], query, 0, (time.perf_counter() - start) * 1000, scores={})

        # Score all documents with hybrid approach
        scored_docs = []
        for doc_id in range(N):
            doc = self.documents[doc_id]

            # Base BM25 score
            bm25_score = self._bm25_score(query_tokens, doc_id)

            # Hybrid bonuses
            phrase_bonus = self._calculate_phrase_bonus(query_signals, doc)
            keyword_bonus = self._calculate_keyword_bonus(query_signals, doc)
            metadata_bonus = self._calculate_metadata_bonus(query_signals, doc)
            number_date_bonus = self._calculate_number_date_bonus(query_signals, doc)
            english_bonus = self._calculate_english_bonus(query_signals, doc)

            # Combined score
            total_score = (
                bm25_score * self.bm25_weight +
                phrase_bonus +
                keyword_bonus +
                metadata_bonus +
                number_date_bonus +
                english_bonus
            )

            # Store score breakdown for debugging
            score_breakdown = {
                "bm25": bm25_score,
                "phrase": phrase_bonus,
                "keyword": keyword_bonus,
                "metadata": metadata_bonus,
                "number_date": number_date_bonus,
                "english": english_bonus,
                "total": total_score
            }

            scored_docs.append((doc_id, total_score, score_breakdown))

        # Sort by total score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        score_map = {}
        for doc_id, total_score, breakdown in scored_docs[:top_k]:
            if doc_id < len(self.documents):
                doc = self.documents[doc_id]
                doc.score = total_score
                results.append(doc)
                score_map[doc.id or str(doc_id)] = breakdown

        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(results, query, N, elapsed, scores=score_map)


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
        # Default to Enhanced BM25 for hybrid component
        kw = EnhancedBM25Retriever(**kwargs)
        vec = VectorRetriever(embedder=embedder) if embedder else None
        return HybridRetriever(kw, vec)
    else:
        return EnhancedBM25Retriever(**kwargs)