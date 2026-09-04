"""
Configuration for ThaiLLM RAG Pipeline
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThaiLLMConfig:
    """ThaiLLM API configuration"""
    # OpenAI-compatible endpoint: /v1/chat/completions
    api_url: str = os.environ.get("THAILLM_API_URL", "https://api.thaillm.or.th/v1/chat/completions")
    api_key: str = os.environ.get("THAILLM_API_KEY", "")
    # Available models: OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2, Typhoon-S-ThaiLLM-8B-Instruct, Pathumma-ThaiLLM-qwen3-8b-think-3.0.0, THaLLE-0.2-ThaiLLM-8B-fa, qwen3.5-9b, qwen3.6-35b-a3b
    model: str = os.environ.get("THAILLM_MODEL", "OpenThaiGPT-ThaiLLM-8B-Instruct-v7.2")
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    bm25_threshold: float = 1.0  # Minimum BM25 score for relevance
    max_context_length: int = 4000  # chars

    # Prompt enhancement settings
    enhance_prompts: bool = True
    enhancement_model: str = "thaillm-7b-instruct"  # can use smaller/faster model

    # Generation settings
    system_prompt: str = """คุณคือผู้ช่วย AI ภาษาไทยที่ช่วยตอบคำถามจากเอกสารเท่านั้น

กฎสำคัญที่ต้องปฏิบัติตลอดเวลา:
1. ตอบคำถามโดยใช้ข้อมูลจากบริบท (context) ที่ให้มาเท่านั้น
2. หากไม่พบคำตอบในบริctx ให้ตอบ: "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารที่กำหนด"
3. ห้ามใช้ความรู้ภายนอก ห้ามเดา ห้ามสร้างข้อมูลขึ้นมา
4. ระบุตัวเลข วันที่ ชื่อเฉพาะ ร้อยละ ตามเอกสารอย่างเคร่งครัด
5. อ้างอิงแหล่งที่มาโดยระบุชื่อเอกสาร และหมายเลขหน้า (ถ้ามี)
6. ตอบเป็นภาษาไทย อย่างเป็นธรรมชาติ และกระชับ
7. ไม่ต้องขอโทษหรืออธิบายเพิ่มเติม - ตอบตรงไปตรงมา"""

    # Prompt templates
    enhancement_prompt_template: str = """คุณคือนักวิเคราะห์คำถามมืออาชีพ งานของคุณคือปรับปรุงคำถามของผู้ใช้ให้ชัดเจน เฉพาะเจาะจง และเหมาะสมสำหรับการค้นหาข้อมูล

คำถามต้นฉบับ: {user_query}

กรุณาสร้างคำถามที่ปรับปรุงแล้ว (1-3 รูปแบบ) ที่จะช่วยให้ระบบค้นหาข้อมูลได้แม่นยำขึ้น:
1. คำถามที่ขยายความหมาย (expanded)
2. คำถามที่แยกส่วนย่อย (decomposed)
3. คำถามด้วยคำสำคัญ (keyword-focused)

ตอบเฉพาะคำถามที่ปรับปรุงแล้ว ไม่ต้องอธิบาย"""

    rag_prompt_template: str = """บริบทจากเอกสาร:
{context}

คำถาม: {question}

คำตอบ:"""


# Default configurations
DEFAULT_THILL_CONFIG = ThaiLLMConfig()
DEFAULT_RAG_CONFIG = RAGConfig()


def load_config_from_env() -> tuple[ThaiLLMConfig, RAGConfig]:
    """Load configuration from environment variables"""
    thaillm_config = ThaiLLMConfig(
        api_url=os.environ.get("THAILLM_API_URL", DEFAULT_THILL_CONFIG.api_url),
        api_key=os.environ.get("THAILLM_API_KEY", DEFAULT_THILL_CONFIG.api_key),
        model=os.environ.get("THAILLM_MODEL", DEFAULT_THILL_CONFIG.model),
        timeout=int(os.environ.get("THAILLM_TIMEOUT", DEFAULT_THILL_CONFIG.timeout)),
        max_tokens=int(os.environ.get("THAILLM_MAX_TOKENS", DEFAULT_THILL_CONFIG.max_tokens)),
        temperature=float(os.environ.get("THAILLM_TEMPERATURE", DEFAULT_THILL_CONFIG.temperature)),
        top_p=float(os.environ.get("THAILLM_TOP_P", DEFAULT_THILL_CONFIG.top_p)),
    )

    rag_config = RAGConfig(
        top_k=int(os.environ.get("RAG_TOP_K", DEFAULT_RAG_CONFIG.top_k)),
        similarity_threshold=float(os.environ.get("RAG_SIMILARITY_THRESHOLD", DEFAULT_RAG_CONFIG.similarity_threshold)),
        bm25_threshold=float(os.environ.get("RAG_BM25_THRESHOLD", DEFAULT_RAG_CONFIG.bm25_threshold)),
        max_context_length=int(os.environ.get("RAG_MAX_CONTEXT_LENGTH", DEFAULT_RAG_CONFIG.max_context_length)),
        enhance_prompts=os.environ.get("RAG_ENHANCE_PROMPTS", "true").lower() == "true",
        enhancement_model=os.environ.get("RAG_ENHANCEMENT_MODEL", DEFAULT_RAG_CONFIG.enhancement_model),
        system_prompt=os.environ.get("RAG_SYSTEM_PROMPT", DEFAULT_RAG_CONFIG.system_prompt),
    )

    return thaillm_config, rag_config