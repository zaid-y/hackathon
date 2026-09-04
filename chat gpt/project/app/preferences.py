"""Validated, request-local controls; document grounding is never optional."""

from typing import Literal
import re
from pydantic import BaseModel, ConfigDict, Field


class AnswerOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int | None = Field(default=None, ge=3, le=8, strict=True)
    evidence_mode: Literal["balanced", "strict"] = "balanced"
    history_messages: Literal[0, 6, 12, 20] = 12
    answer_style: Literal["concise", "balanced", "detailed"] = "concise"
    answer_language: Literal["auto", "th", "en", "zh"] = "auto"

    def resolved_language(self, question: str) -> str:
        from app.answer_language import detect_language
        return detect_language(question)

    def language_instruction(self, question: str) -> str:
        language = self.resolved_language(question)
        names = {"th": "ภาษาไทย (Thai)", "en": "ภาษาอังกฤษ (English)",
                 "zh": "ภาษาจีน (Chinese)", "ja": "ภาษาญี่ปุ่น (Japanese)",
                 "ko": "ภาษาเกาหลี (Korean)"}
        target = names.get(language, "ภาษาเดียวกับคำถามต้นฉบับ ไม่ใช่ภาษาไทย / the original question's language, not Thai")
        if language == "th":
            return "\nREQUIRED OUTPUT: Answer once in Thai only. Do not duplicate the Thai answer. Use only CONTEXT evidence."
        return (f"\nต้องตอบสองภาษาเสมอ: ส่วนแรกภาษาไทย หัวข้อ 'ภาษาไทย' "
                f"ส่วนที่สอง{target} ใช้ชื่อภาษานั้นเป็นหัวข้อ ห้ามตอบเพียงภาษาเดียว "
                "ทั้งสองส่วนต้องมีข้อเท็จจริง ตัวเลข และเงื่อนไขตรงกัน อ้างอิงเฉพาะ CONTEXT\n"
                f"REQUIRED BILINGUAL OUTPUT: First a complete Thai answer under 'ภาษาไทย', "
                f"then the equivalent complete answer in {target}, under that language's name. "
                "Separate the sections with a blank line. Use plain-text headings, no Markdown markup. "
                "Both sections must preserve the same facts, numbers, qualifications and uncertainty. "
                "Do not translate the question instead of answering it. Do not add outside facts. "
                "If evidence is insufficient, keep the prescribed Thai refusal in the Thai section "
                "and provide only its translation in the other section.")

    def refusal_message(self, question: str, thai_message: str) -> str:
        translations = {
            "en": ("English", "There is not enough relevant information in the provided documents."),
            "zh": ("中文", "提供的文档中没有足够的相关信息。"),
            "ja": ("日本語", "提供された文書には十分な関連情報がありません。"),
            "ko": ("한국어", "제공된 문서에 관련 정보가 충분하지 않습니다."),
        }
        translated = translations.get(self.resolved_language(question))
        if translated is None:
            return thai_message
        heading, message = translated
        return f"ภาษาไทย:\n{thai_message}\n\n{heading}:\n{message}"

    def prompt_guidance(self) -> str:
        styles = {
            "concise": "Answer briefly and directly, preserving all necessary facts.",
            "balanced": "Give a balanced explanation with useful supporting details.",
            "detailed": "Explain in detail with clear sections or lists, but only where evidence supports them.",
        }
        languages = {
            "auto": "Write in Thai first, then in the language of the user's current question. If it is Thai, answer once.",
            "th": "Write the answer in Thai.",
            "en": "Write the answer in English after an equivalent Thai section.",
            "zh": "Write the answer in Chinese after an equivalent Thai section.",
        }
        return (styles[self.answer_style] + " " + languages['auto']
                + " These presentation preferences never override the document-only system rules."
                + " If evidence is insufficient, use the prescribed refusal message.")
