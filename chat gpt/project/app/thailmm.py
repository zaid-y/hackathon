"""Isolated client for ThaiLLM's OpenAI-compatible chat API.

Only ThaiLLM is called. There is deliberately no alternate LLM fallback.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class ThaiLLMError(RuntimeError):
    """Base class for safe ThaiLLM failures."""


class ThaiLLMConfigurationError(ThaiLLMError):
    """Required connection configuration is missing."""


class ThaiLLMAdapterRequiredError(ThaiLLMError):
    """The official API request/response mapping has not been supplied."""


class ThaiLLMTimeoutError(ThaiLLMError):
    """The ThaiLLM request exceeded its timeout."""


class ThaiLLMRateLimitError(ThaiLLMError):
    """ThaiLLM rejected the request due to rate limiting."""


class ThaiLLMServiceError(ThaiLLMError):
    """ThaiLLM returned an HTTP or transport failure."""


class ThaiLLMMalformedResponseError(ThaiLLMError):
    """ThaiLLM returned a response that cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class ThaiLLMResponse:
    text: str


class ThaiLLMProvider(Protocol):
    """Small interface that lets the answer pipeline be tested independently."""

    def answer(self, system_prompt: str, user_prompt: str) -> ThaiLLMResponse: ...


class ThaiLLMClient:
    """HTTP transport for ThaiLLM's OpenAI-compatible chat endpoint."""

    ADAPTER_IMPLEMENTED = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def endpoint_url(self) -> str:
        """Accept either an API base ending in /v1 or the complete endpoint."""

        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def is_configured(self) -> bool:
        return bool(
            self.api_key
            and self.base_url
            and self.model
            and self.ADAPTER_IMPLEMENTED
        )

    def answer(self, system_prompt: str, user_prompt: str) -> ThaiLLMResponse:
        self._validate_configuration()
        payload, headers = self._build_official_request(system_prompt, user_prompt)
        response = self._post_json(payload, headers)
        text = self._parse_official_response(response).strip()
        if not text:
            raise ThaiLLMMalformedResponseError("ThaiLLM returned an empty answer")
        return ThaiLLMResponse(text=text)

    def _validate_configuration(self) -> None:
        missing = [
            name
            for name, value in (
                ("THAILLM_API_KEY", self.api_key),
                ("THAILLM_BASE_URL", self.base_url),
                ("THAILLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise ThaiLLMConfigurationError(
                "Missing ThaiLLM configuration: " + ", ".join(missing)
            )
        if self.timeout_seconds <= 0:
            raise ThaiLLMConfigurationError(
                "THAILLM_TIMEOUT_SECONDS must be greater than zero"
            )

    def _build_official_request(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        return (
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0,
            },
            {"apikey": self.api_key, "User-Agent": "litellm"},
        )

    def _parse_official_response(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ThaiLLMMalformedResponseError(
                "ThaiLLM response is missing choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise ThaiLLMMalformedResponseError(
                "ThaiLLM answer content must be text"
            )
        return content

    def _post_json(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **headers}
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw_response = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ThaiLLMRateLimitError(
                    "ThaiLLM rate limit reached; please retry shortly"
                ) from exc
            detail = self._safe_http_error_detail(exc)
            if exc.code in {401, 403}:
                message = f"ThaiLLM authorization denied (HTTP {exc.code})"
            else:
                message = f"ThaiLLM returned HTTP {exc.code}"
            if detail:
                message = f"{message}: {detail}"
            raise ThaiLLMServiceError(
                message
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ThaiLLMTimeoutError(
                f"ThaiLLM did not respond within {self.timeout_seconds:g} seconds"
            ) from exc
        except urllib.error.URLError as exc:
            raise ThaiLLMServiceError(
                f"Could not connect to ThaiLLM: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ThaiLLMMalformedResponseError(
                "ThaiLLM returned invalid JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise ThaiLLMMalformedResponseError(
                "ThaiLLM response must be a JSON object"
            )
        return parsed

    def _safe_http_error_detail(self, exc: urllib.error.HTTPError) -> str:
        """Extract a short provider message without ever echoing the API key."""

        try:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeError):
            return ""

        candidate: Any = None
        if isinstance(parsed, dict):
            candidate = parsed.get("detail") or parsed.get("message")
            error = parsed.get("error")
            if candidate is None and isinstance(error, str):
                candidate = error
            elif candidate is None and isinstance(error, dict):
                candidate = error.get("message") or error.get("detail")
        if not isinstance(candidate, str):
            return ""

        cleaned = " ".join(candidate.split())
        if self.api_key:
            cleaned = cleaned.replace(self.api_key, "[redacted]")
        return cleaned[:300]
