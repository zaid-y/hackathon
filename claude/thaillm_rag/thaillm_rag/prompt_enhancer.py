"""
Prompt Enhancement Module for ThaiLLM RAG
Enhances user queries for better retrieval using ThaiLLM itself
"""
import re
import json
from typing import List, Optional
from dataclasses import dataclass

from .config import ThaiLLMConfig, RAGConfig
from .thaillm_client import ThaiLLMClient


@dataclass
class EnhancedQuery:
    """Container for enhanced query variants"""
    original: str
    expanded: str
    decomposed: List[str]
    keyword_focused: str
    all_variants: List[str]


class PromptEnhancer:
    """
    Enhances user queries to improve retrieval quality.
    Uses ThaiLLM to generate multiple query variants.
    """

    def __init__(
        self,
        thaillm_client: ThaiLLMClient,
        config: Optional[RAGConfig] = None
    ):
        self.client = thaillm_client
        self.config = config or RAGConfig()

    def enhance(self, user_query: str) -> EnhancedQuery:
        """
        Enhance a user query into multiple variants for better retrieval.

        Args:
            user_query: Original user question

        Returns:
            EnhancedQuery with multiple variants
        """
        if not self.config.enhance_prompts:
            return EnhancedQuery(
                original=user_query,
                expanded=user_query,
                decomposed=[user_query],
                keyword_focused=self._extract_keywords(user_query),
                all_variants=[user_query]
            )

        # Use LLM to enhance the query
        enhanced_text = self._llm_enhance(user_query)

        # Parse the enhanced response
        expanded, decomposed, keyword_focused = self._parse_enhancement(enhanced_text, user_query)

        # Combine all variants (deduplicated)
        all_variants = self._deduplicate([
            user_query,
            expanded,
            keyword_focused,
            *decomposed
        ])

        return EnhancedQuery(
            original=user_query,
            expanded=expanded,
            decomposed=decomposed,
            keyword_focused=keyword_focused,
            all_variants=all_variants
        )

    def _llm_enhance(self, user_query: str) -> str:
        """Call ThaiLLM to enhance the query"""
        prompt = self.config.enhancement_prompt_template.format(user_query=user_query)

        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": "คุณคือนักวิเคราะห์คำถามมืออาชีพ"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            return response.strip()
        except Exception as e:
            print(f"[PromptEnhancer] Enhancement failed: {e}, using fallback")
            return self._fallback_enhance(user_query)

    def _fallback_enhance(self, user_query: str) -> str:
        """Simple rule-based enhancement when LLM fails"""
        keywords = self._extract_keywords(user_query)
        expanded = f"{user_query} กรุณาอธิบายรายละเอียดและตัวอย่าง"
        decomposed = self._decompose_query(user_query)

        return f"""1. {expanded}
2. {' | '.join(decomposed)}
3. {keywords}"""

    def _parse_enhancement(self, enhanced_text: str, original: str) -> tuple[str, List[str], str]:
        """Parse LLM enhancement output into structured variants"""
        lines = [line.strip() for line in enhanced_text.split('\n') if line.strip()]

        expanded = original
        decomposed = [original]
        keyword_focused = self._extract_keywords(original)

        for line in lines:
            # Remove numbering
            clean = re.sub(r'^\d+[\.\)]\s*', '', line)

            if any(kw in clean.lower() for kw in ['ขยาย', 'expanded', 'ละเอียด', 'รายละเอียด']):
                expanded = clean
            elif any(kw in clean.lower() for kw in ['แยก', 'decomposed', 'ส่วนย่อย', 'ย่อย']):
                # Split by common separators
                parts = re.split(r'[|,،؛]', clean)
                decomposed = [p.strip() for p in parts if p.strip()]
            elif any(kw in clean.lower() for kw in ['คำสำคัญ', 'keyword', 'สำคัญ']):
                keyword_focused = clean

        return expanded, decomposed, keyword_focused

    def _extract_keywords(self, query: str) -> str:
        """Extract key terms from query (simple Thai/English)"""
        # Remove question words
        stopwords = {
            'คือ', 'อะไร', 'ที่', 'ซึ่ง', 'ได้', 'จะ', 'มี', 'เป็น', 'อย่างไร',
            'ทำไม', 'เมื่อไหร่', 'ที่ไหน', 'ใคร', 'what', 'how', 'why', 'when', 'where', 'who',
            'the', 'a', 'an', 'is', 'are', 'can', 'do', 'does', '?', '?'
        }

        # Tokenize (simple split for Thai+English)
        tokens = re.findall(r'[฀-๿]+|[a-zA-Z]+', query.lower())
        keywords = [t for t in tokens if t not in stopwords and len(t) > 1]

        return ' '.join(keywords[:10])  # Top 10 keywords

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions"""
        # Simple decomposition by question marks and conjunctions
        parts = re.split(r'\?|\s+(?:และ|และว่า|หรือ|หรือว่า|แต่|แต่ว่า|เพราะ|เพราะว่า)\s+', query)
        parts = [p.strip() + '?' for p in parts if p.strip() and not p.strip().endswith('?')]

        if not parts:
            parts = [query]

        return parts[:5]  # Max 5 sub-questions

    def _deduplicate(self, variants: List[str]) -> List[str]:
        """Remove duplicate variants (case-insensitive)"""
        seen = set()
        unique = []
        for v in variants:
            key = v.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(v)
        return unique


# Convenience function
def create_prompt_enhancer(
    thaillm_config: ThaiLLMConfig,
    rag_config: Optional[RAGConfig] = None
) -> PromptEnhancer:
    """Factory function to create PromptEnhancer with ThaiLLM client"""
    client = ThaiLLMClient(thaillm_config)
    return PromptEnhancer(client, rag_config)