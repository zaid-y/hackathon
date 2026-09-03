from __future__ import annotations

from app.answer import (
    INSUFFICIENT_INFORMATION_MESSAGE,
    AnswerService,
    InvalidQuestionError,
    hide_source_markers,
)
from app.models import TextChunk
from app.retriever import BM25Retriever
from app.thailmm import ThaiLLMResponse


class FakeThaiLLM:
    def __init__(self, answer: str = "คำตอบจากบริบท [SOURCE 1]") -> None:
        self.response = answer
        self.calls: list[tuple[str, str]] = []

    def answer(self, system_prompt: str, user_prompt: str) -> ThaiLLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return ThaiLLMResponse(text=self.response)


def _service(*, threshold: float = 0.35) -> tuple[AnswerService, FakeThaiLLM]:
    chunks = [
        TextChunk(
            chunk_id="rules_p12_c01",
            text="ผู้สมัครต้องมีอายุอย่างน้อย 18 ปี สมัครภายในวันที่ 15 มกราคม 2569",
            document="rules.pdf",
            page=12,
            chunk_index=1,
            start_char=0,
            end_char=70,
        ),
        TextChunk(
            chunk_id="prizes_p3_c01",
            text="รางวัลชนะเลิศ 50,000 บาท ประกาศผลวันที่ 28 กุมภาพันธ์ 2569",
            document="prizes.pdf",
            page=3,
            chunk_index=1,
            start_char=0,
            end_char=62,
        ),
    ]
    retriever = BM25Retriever()
    retriever.build(chunks)
    provider = FakeThaiLLM()
    return (
        AnswerService(
            retriever=retriever,
            provider=provider,
            top_k=3,
            relevance_threshold=threshold,
        ),
        provider,
    )


def test_retrieval_calls_thailmm_and_returns_metadata_citation() -> None:
    service, provider = _service()

    result = service.answer("ผู้สมัครต้องมีอายุเท่าไร")

    assert result.grounded is True
    assert result.answer == "คำตอบจากบริบท"
    assert provider.calls
    assert result.sources[0].document == "rules.pdf"
    assert result.sources[0].page == 12
    assert result.sources[0].chunk_ids == ("rules_p12_c01",)


def test_all_source_markers_are_hidden_from_visible_answer() -> None:
    text = "ข้อแรก [SOURCE 1]\nข้อสอง [source 2] [ SOURCE 15 ]"

    assert hide_source_markers(text) == "ข้อแรก\nข้อสอง"


def test_low_confidence_refuses_without_calling_thailmm() -> None:
    service, provider = _service(threshold=0.6)

    result = service.answer("สนามบินตั้งอยู่จังหวัดอะไร")

    assert result.answer == INSUFFICIENT_INFORMATION_MESSAGE
    assert result.grounded is False
    assert result.sources == ()
    assert provider.calls == []


def test_prompt_and_retrieval_are_available_for_debugging() -> None:
    service, _ = _service()

    service.answer("รางวัลชนะเลิศเท่าไร")

    assert service.last_debug is not None
    assert service.last_debug["query"] == "รางวัลชนะเลิศเท่าไร"
    assert service.last_debug["retrieved_chunks"]
    assert service.last_debug["final_prompt"] is not None
    assert "THAILLM_API_KEY" not in str(service.last_debug)


def test_follow_up_uses_recent_history_for_retrieval_and_prompt() -> None:
    service, provider = _service()

    result = service.answer(
        "แล้วประกาศผลวันไหน",
        history=(("user", "รางวัลชนะเลิศเท่าไร"), ("assistant", "50,000 บาท")),
    )

    assert result.grounded is True
    assert service.last_debug is not None
    assert "รางวัลชนะเลิศเท่าไร" in service.last_debug["retrieval_query"]
    assert "CONVERSATION HISTORY" in provider.calls[0][1]


def test_empty_question_is_rejected() -> None:
    service, provider = _service()

    try:
        service.answer("   ")
    except InvalidQuestionError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected InvalidQuestionError")
    assert provider.calls == []
