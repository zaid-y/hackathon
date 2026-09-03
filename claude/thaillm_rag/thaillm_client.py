"""
ThaiLLM API Client
Handles all communication with ThaiLLM API
"""
import os
import json
import time
import requests
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ThaiLLMConfig


@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Structured response from ThaiLLM"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class ThaiLLMError(Exception):
    """Custom exception for ThaiLLM API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ThaiLLMClient:
    """
    Client for ThaiLLM API with retry logic, error handling, and streaming support.
    Compatible with OpenAI-style API format.
    """

    def __init__(self, config: Optional[ThaiLLMConfig] = None):
        self.config = config or ThaiLLMConfig()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers"""
        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            # ThaiLLM uses Bearer token authentication
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload"""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        elif self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        if top_p is not None:
            payload["top_p"] = top_p
        elif self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        # Add any extra parameters
        payload.update(kwargs)

        return payload

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Non-streaming chat completion.

        Args:
            messages: List of {"role": "...", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens in response
            top_p: Nucleus sampling parameter

        Returns:
            Assistant response text
        """
        response = self._make_request(messages, temperature, max_tokens, top_p, stream=False, **kwargs)
        return response.content

    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Chat completion with full response metadata.

        Returns:
            ChatResponse with content, usage, finish_reason, etc.
        """
        return self._make_request(messages, temperature, max_tokens, top_p, stream=False, **kwargs)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Streaming chat completion.

        Yields:
            Text chunks as they arrive
        """
        payload = self._build_payload(messages, temperature, max_tokens, top_p, stream=True, **kwargs)
        headers = self._get_headers()

        try:
            response = self.session.post(
                self.config.api_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # Handle SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: "
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            raise ThaiLLMError("Request timed out")
        except requests.exceptions.HTTPError as e:
            raise ThaiLLMError(f"API error: {e}", status_code=e.response.status_code, response_body=e.response.text)
        except requests.exceptions.RequestException as e:
            raise ThaiLLMError(f"Network error: {e}")

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        stream: bool,
        **kwargs
    ) -> ChatResponse:
        """Make API request and parse response"""
        payload = self._build_payload(messages, temperature, max_tokens, top_p, stream, **kwargs)
        headers = self._get_headers()

        try:
            response = self.session.post(
                self.config.api_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Parse OpenAI-compatible response
            try:
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # Handle cases where content might be in different fields
                if not content and "delta" in choice:
                    content = choice["delta"].get("content", "")

                return ChatResponse(
                    content=content,
                    model=data.get("model", self.config.model),
                    usage=data.get("usage"),
                    finish_reason=choice.get("finish_reason"),
                    raw_response=data
                )
            except (KeyError, IndexError) as e:
                # Try alternative response formats
                content = self._extract_content_fallback(data)
                return ChatResponse(
                    content=content,
                    model=self.config.model,
                    raw_response=data
                )

        except requests.exceptions.Timeout:
            raise ThaiLLMError("Request timed out. Check your connection or the API status.")
        except requests.exceptions.HTTPError as e:
            raise ThaiLLMError(
                f"API returned an error: {e}",
                status_code=e.response.status_code,
                response_body=e.response.text
            )
        except requests.exceptions.RequestException as e:
            raise ThaiLLMError(f"Network error calling the API: {e}")

    def _extract_content_fallback(self, data: Dict[str, Any]) -> str:
        """Try to extract content from various response formats"""
        # Format: {"response": "..."}
        if "response" in data:
            return str(data["response"])

        # Format: {"text": "..."}
        if "text" in data:
            return str(data["text"])

        # Format: {"output": "..."}
        if "output" in data:
            return str(data["output"])

        # Format: {"result": "..."}
        if "result" in data:
            return str(data["result"])

        # Format: {"data": {"content": "..."}}
        if "data" in data and isinstance(data["data"], dict):
            if "content" in data["data"]:
                return str(data["data"]["content"])

        raise ThaiLLMError(f"Unexpected response format: {json.dumps(data, ensure_ascii=False)[:500]}")

    def health_check(self) -> bool:
        """Check if API is reachable"""
        try:
            # Try a minimal request
            self.chat([{"role": "user", "content": "ping"}], max_tokens=5)
            return True
        except Exception:
            return False

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()