from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.thailmm import (
    ThaiLLMClient,
    ThaiLLMConfigurationError,
    ThaiLLMMalformedResponseError,
    ThaiLLMRateLimitError,
    ThaiLLMResponse,
    ThaiLLMServiceError,
    ThaiLLMTimeoutError,
)


class DocumentedTestAdapter(ThaiLLMClient):
    """Test-only adapter; it does not represent the organizers' API."""

    ADAPTER_IMPLEMENTED = True

    def _build_official_request(self, system_prompt: str, user_prompt: str):
        return {"test_model": self.model, "test_prompt": user_prompt}, {"X-Test": "1"}

    def _parse_official_response(self, response):
        value = response.get("test_answer")
        if not isinstance(value, str):
            raise ThaiLLMMalformedResponseError("Missing test answer")
        return value


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


def _client() -> DocumentedTestAdapter:
    return DocumentedTestAdapter(
        api_key="test-key",
        base_url="https://example.invalid/thailmm",
        model="test-model",
        timeout_seconds=2,
    )


def test_production_adapter_builds_openai_compatible_request() -> None:
    client = ThaiLLMClient(
        api_key="secret-key", base_url="https://example.invalid/v1", model="/model"
    )

    payload, headers = client._build_official_request("system", "user")

    assert payload == {
        "model": "/model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "stream": False,
        "temperature": 0,
    }
    assert headers == {"apikey": "secret-key", "User-Agent": "litellm"}
    assert client.endpoint_url == "https://example.invalid/v1/chat/completions"


def test_complete_endpoint_is_not_duplicated() -> None:
    client = ThaiLLMClient(
        api_key="key",
        base_url="https://example.invalid/v1/chat/completions/",
        model="/model",
    )

    assert client.endpoint_url == "https://example.invalid/v1/chat/completions"


def test_production_adapter_parses_openai_compatible_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def respond(request, timeout):
        captured["url"] = request.full_url
        captured["apikey"] = request.get_header("Apikey")
        captured["user_agent"] = request.get_header("User-agent")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(
            '{"choices":[{"message":{"content":"คำตอบจากไทยแอลแอลเอ็ม"}}]}'.encode(
                "utf-8"
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = ThaiLLMClient(
        api_key="secret-key",
        base_url="https://example.invalid/v1",
        model="/model",
    )

    result = client.answer("system", "user")

    assert result == ThaiLLMResponse(text="คำตอบจากไทยแอลแอลเอ็ม")
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["apikey"] == "secret-key"
    assert captured["user_agent"] == "litellm"
    assert captured["body"]["model"] == "/model"


def test_missing_configuration_is_clear() -> None:
    client = ThaiLLMClient(api_key="", base_url="", model="")

    with pytest.raises(ThaiLLMConfigurationError, match="THAILLM_API_KEY"):
        client.answer("system", "user")


def test_documented_adapter_can_return_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeHTTPResponse('{"test_answer":"คำตอบ"}'.encode("utf-8")),
    )

    response = _client().answer("system", "user")

    assert response == ThaiLLMResponse(text="คำตอบ")


def test_rate_limit_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def rate_limited(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "rate", {}, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", rate_limited)

    with pytest.raises(ThaiLLMRateLimitError):
        _client().answer("system", "user")


def test_forbidden_includes_safe_provider_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(request, timeout):
        body = b'{"detail":"API key test-key is not allowed for this model"}'
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", forbidden)

    with pytest.raises(ThaiLLMServiceError) as captured:
        _client().answer("system", "user")

    message = str(captured.value)
    assert "authorization denied (HTTP 403)" in message
    assert "not allowed for this model" in message
    assert "test-key" not in message
    assert "[redacted]" in message


def test_timeout_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(request, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr("urllib.request.urlopen", timed_out)

    with pytest.raises(ThaiLLMTimeoutError):
        _client().answer("system", "user")


def test_invalid_json_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeHTTPResponse(b"not-json"),
    )

    with pytest.raises(ThaiLLMMalformedResponseError):
        _client().answer("system", "user")


def test_missing_openai_compatible_answer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeHTTPResponse(b'{"choices":[]}'),
    )
    client = ThaiLLMClient(
        api_key="key", base_url="https://example.invalid/v1", model="/model"
    )

    with pytest.raises(ThaiLLMMalformedResponseError, match="message.content"):
        client.answer("system", "user")
