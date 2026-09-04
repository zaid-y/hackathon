from __future__ import annotations
import json

from app.answer import (
    INSUFFICIENT_INFORMATION_MESSAGE,
    AnswerService,
    InvalidQuestionError,
    hide_source_markers,
)
from app.models import TextChunk
from app.preferences import AnswerOptions
from app.retriever import BM25Retriever
from app.thailmm import ThaiLLMResponse


class FakeThaiLLM:
    def __init__(self, answer: str | None = None) -> None:
        self.response = answer
        self.calls: list[tuple[str, str]] = []

    def answer(self, system_prompt: str, user_prompt: str) -> ThaiLLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return ThaiLLMResponse(text=self.response if self.response is not None else json.dumps(
            {"evidence_ids": [e["id"] for e in json.loads(user_prompt)["evidence"]]}))


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


def test_options_are_request_local_and_preserve_grounding() -> None:
    service, provider = _service()
    history = [("user", "previous-secret-context")]
    service.answer("ผู้สมัครต้องมีอายุเท่าไร", history=history,
                   options=AnswerOptions(top_k=8, history_messages=0,
                       answer_style="detailed", answer_language="en", evidence_mode="strict"))
    assert service.last_debug["history_turn_count"] == 0
    assert "previous-secret-context" not in service.last_debug["retrieval_query"]
    assert service.last_debug["top_k"] == 8
    assert service.last_debug["relevance_threshold"] == 0.55
    assert provider.calls
    from app.grounding import SELECT_PROMPT
    assert provider.calls[-1][0] == SELECT_PROMPT
    assert "evidence" in provider.calls[-1][1]
    assert "previous-secret-context" not in provider.calls[-1][1]
    service.answer("ผู้สมัครต้องมีอายุเท่าไร")
    assert service.last_debug["top_k"] == 3
    assert service.last_debug["relevance_threshold"] == 0.35
    assert "PRESENTATION PREFERENCES" not in provider.calls[-1][1]


def test_context_limit_bounds_server_history() -> None:
    service, _ = _service()
    service.answer("ผู้สมัครต้องมีอายุเท่าไร", history=[("user", str(i)) for i in range(20)],
                   options=AnswerOptions(history_messages=6))
    assert service.last_debug["history_turn_count"] == 6


def test_strict_evidence_never_lowers_server_threshold() -> None:
    service, _ = _service(threshold=0.9)
    service.answer("ผู้สมัครต้องมีอายุเท่าไร", options=AnswerOptions(evidence_mode="strict"))
    assert service.last_debug["relevance_threshold"] == 0.9


def test_retrieval_calls_thailmm_and_returns_metadata_citation() -> None:
    service, provider = _service()

    result = service.answer("ผู้สมัครต้องมีอายุเท่าไร")

    assert result.grounded is True
    assert "ผู้สมัครต้องมีอายุอย่างน้อย 18 ปี" in result.answer
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
    assert "ประกาศผลวันที่" in provider.calls[0][1]


def test_program_specific_question_filters_similar_curriculum_documents() -> None:
    chunks = [
        TextChunk(
            chunk_id="regular",
            text="หลักสูตรเทคโนโลยีสารสนเทศ จำนวนหน่วยกิตรวมตลอดหลักสูตร 129 หน่วยกิต",
            document="IT2565.pdf",
            page=1,
            chunk_index=1,
            start_char=0,
            end_char=43,
        ),
        TextChunk(
            chunk_id="international",
            text="หลักสูตรเทคโนโลยีสารสนเทศทางธุรกิจ หลักสูตรนานาชาติ จำนวนหน่วยกิตรวมตลอดหลักสูตร 126 หน่วยกิต",
            document="IT_inter2565.pdf",
            page=1,
            chunk_index=1,
            start_char=0,
            end_char=68,
        ),
    ]
    retriever = BM25Retriever()
    retriever.build(chunks)
    provider = FakeThaiLLM()
    service = AnswerService(retriever=retriever, provider=provider, top_k=2)

    result = service.answer("หลักสูตรนานาชาติมีกี่หน่วยกิต")

    assert [source.document for source in result.sources] == ["IT_inter2565.pdf"]


def test_empty_question_is_rejected() -> None:
    service, provider = _service()

    try:
        service.answer("   ")
    except InvalidQuestionError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected InvalidQuestionError")
    assert provider.calls == []
