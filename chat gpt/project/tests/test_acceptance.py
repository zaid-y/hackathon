from __future__ import annotations

from app.answer import INSUFFICIENT_INFORMATION_MESSAGE, AnswerService
from app.models import TextChunk
from app.retriever import BM25Retriever
from app.thailmm import ThaiLLMResponse


class EvidenceTestThaiLLM:
    """Deterministic Phase 8 double; never contacts or imitates another LLM."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer(self, system_prompt: str, user_prompt: str) -> ThaiLLMResponse:
        self.calls.append(user_prompt)
        if "500,000" in user_prompt and "50,000 บาท" in user_prompt:
            return ThaiLLMResponse(
                text="ไม่ใช่ รางวัลชนะเลิศคือ 50,000 บาท [SOURCE 1]"
            )
        if "หัวหน้าทีมชื่ออะไร" in user_prompt:
            return ThaiLLMResponse(
                text="หัวหน้าทีมชื่อสมชาย และประกาศผลวันที่ 28 กุมภาพันธ์ 2569 [SOURCE 1] [SOURCE 2]"
            )
        return ThaiLLMResponse(text="คำตอบที่ยึดตามหลักฐาน [SOURCE 1]")


def _chunk(chunk_id: str, text: str, document: str, page: int) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        text=text,
        document=document,
        page=page,
        chunk_index=1,
        start_char=0,
        end_char=len(text),
    )


def _acceptance_service(
    *, threshold: float = 0.28
) -> tuple[AnswerService, EvidenceTestThaiLLM]:
    retriever = BM25Retriever()
    retriever.build(
        [
            _chunk(
                "rules_p1_c01",
                "ผู้สมัครต้องมีอายุอย่างน้อย 18 ปี ทีมต้องมีสมาชิก 3 ถึง 5 คน",
                "rules.pdf",
                1,
            ),
            _chunk(
                "rules_p2_c01",
                "หัวหน้าทีมชื่อสมชาย และผู้สมัครทุกคนต้องยืนยันตัวตน",
                "rules.pdf",
                2,
            ),
            _chunk(
                "prizes_p3_c01",
                "รางวัลชนะเลิศคือ 50,000 บาท",
                "prizes.pdf",
                3,
            ),
            _chunk(
                "schedule_p4_c01",
                "กำหนดส่งผลงานวันที่ 15 กุมภาพันธ์ 2569 เวลา 23:59 น. และประกาศผลวันที่ 28 กุมภาพันธ์ 2569",
                "schedule.pdf",
                4,
            ),
        ]
    )
    provider = EvidenceTestThaiLLM()
    return (
        AnswerService(
            retriever=retriever,
            provider=provider,
            top_k=4,
            relevance_threshold=threshold,
        ),
        provider,
    )


def test_a_question_clearly_answered_in_document() -> None:
    service, provider = _acceptance_service()

    result = service.answer("ผู้สมัครต้องมีอายุอย่างน้อยเท่าไร")

    assert result.grounded is True
    assert provider.calls
    assert result.sources[0].document == "rules.pdf"
    assert result.sources[0].page == 1


def test_b_question_requires_multiple_chunks() -> None:
    service, provider = _acceptance_service(threshold=0.24)

    result = service.answer("อายุขั้นต่ำและรางวัลชนะเลิศคือเท่าไร")

    source_documents = {source.document for source in result.sources}
    assert {"rules.pdf", "prizes.pdf"}.issubset(source_documents)
    assert "[SOURCE 2]" in provider.calls[0]


def test_c_question_not_present_is_refused_before_thailmm() -> None:
    service, provider = _acceptance_service(threshold=0.4)

    result = service.answer("สนามบินจัดงานตั้งอยู่จังหวัดอะไร")

    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.grounded is False
    assert result.sources == ()
    assert provider.calls == []


def test_d_similar_but_incorrect_information_uses_document_value() -> None:
    service, _ = _acceptance_service()

    result = service.answer("รางวัลชนะเลิศคือ 500,000 บาทใช่หรือไม่")

    assert "ไม่ใช่" in result.answer
    assert "50,000 บาท" in result.answer
    assert result.sources[0].document == "prizes.pdf"


def test_e_thai_language_question() -> None:
    service, provider = _acceptance_service()

    result = service.answer("กำหนดส่งผลงานวันไหน")

    assert result.grounded is True
    assert result.sources[0].document == "schedule.pdf"
    assert "15 กุมภาพันธ์ 2569" in provider.calls[0]


def test_f_exact_number_date_and_name_lookup() -> None:
    service, _ = _acceptance_service(threshold=0.24)

    result = service.answer("หัวหน้าทีมชื่ออะไร และประกาศผลวันที่เท่าไร")

    assert "สมชาย" in result.answer
    assert "28 กุมภาพันธ์ 2569" in result.answer
    assert {source.page for source in result.sources}.issuperset({2, 4})
