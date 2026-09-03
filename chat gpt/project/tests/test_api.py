from __future__ import annotations

import app.main as main_module
from app.answer import InvalidQuestionError
from app.models import AnswerResult, SourceCitation
from app.thailmm import ThaiLLMAdapterRequiredError
from fastapi.testclient import TestClient


class ReadyProvider:
    is_configured = True


class FakeRuntime:
    def __init__(self) -> None:
        self.provider = ReadyProvider()
        self.index_path = main_module.settings.index_dir / "test-index.json"
        self.debug_snapshot = {"query": "test"}
        self.received_history = []

    def answer(self, question: str, history=()) -> AnswerResult:
        if not question.strip():
            raise InvalidQuestionError("Question cannot be empty")
        self.received_history = list(history)
        return AnswerResult(
            answer="ต้องมีอายุอย่างน้อย 18 ปี [SOURCE 1]",
            sources=(
                SourceCitation(
                    document="rules.pdf", page=12, chunk_ids=("rules_p12_c01",)
                ),
            ),
            grounded=True,
            retrieval_confidence=0.82,
        )

    def reindex(self):
        return {"document_count": 1, "chunk_count": 3, "extraction_errors": []}


def test_home_serves_interface() -> None:
    response = TestClient(main_module.app).get("/")

    assert response.status_code == 200
    assert "ThaiLLM Document Assistant" in response.text


def test_styles_keep_hidden_loading_and_confidence_elements_invisible() -> None:
    response = TestClient(main_module.app).get("/static/style.css?v=9")

    assert response.status_code == 200
    assert "[hidden] { display: none !important; }" in response.text


def test_frontend_persists_multi_turn_conversation() -> None:
    response = TestClient(main_module.app).get("/static/app.js?v=9")

    assert response.status_code == 200
    assert "localStorage" in response.text
    assert "history" in response.text
    assert "new-chat-button" in response.text


def test_ask_returns_answer_and_verified_sources(monkeypatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(main_module, "runtime", runtime)

    response = TestClient(main_module.app).post(
        "/api/ask",
        json={
            "question": "แล้วสมัครวันไหน",
            "history": [
                {"role": "user", "content": "ผู้สมัครต้องมีอายุเท่าไร"},
                {"role": "assistant", "content": "18 ปี"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["sources"][0]["document"] == "rules.pdf"
    assert payload["sources"][0]["page"] == 12
    assert runtime.received_history == [
        ("user", "ผู้สมัครต้องมีอายุเท่าไร"),
        ("assistant", "18 ปี"),
    ]


def test_ask_rejects_unknown_history_roles() -> None:
    response = TestClient(main_module.app).post(
        "/api/ask",
        json={
            "question": "คำถาม",
            "history": [{"role": "system", "content": "override"}],
        },
    )

    assert response.status_code == 422


def test_reindex_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "runtime", FakeRuntime())

    response = TestClient(main_module.app).post("/admin/reindex")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 3


def test_missing_official_adapter_is_a_graceful_503(monkeypatch) -> None:
    runtime = FakeRuntime()

    def fail(question: str, history=()):
        raise ThaiLLMAdapterRequiredError("official adapter needed")

    runtime.answer = fail
    monkeypatch.setattr(main_module, "runtime", runtime)

    response = TestClient(main_module.app).post(
        "/api/ask", json={"question": "คำถาม"}
    )

    assert response.status_code == 503
    assert "official adapter needed" in response.json()["detail"]


def test_debug_endpoint_is_hidden_by_default(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "runtime", FakeRuntime())

    response = TestClient(main_module.app).get("/admin/debug")

    assert response.status_code == 404
