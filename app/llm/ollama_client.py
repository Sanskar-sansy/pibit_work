"""
Ollama REST API client with retry logic, token tracking,
latency measurement, and persistent call logging.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or is unreachable."""


class OllamaClient:
    """
    Client for the Ollama local inference server.

    Handles:
    - Model switching
    - Retry with exponential backoff
    - Timeout management
    - Token and latency tracking
    - Cost estimation (placeholder for future pricing)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Aggregated stats for this client instance
        self._total_calls: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Call the Ollama /api/generate endpoint.

        Args:
            model: Ollama model name (e.g. 'mistral', 'llama3').
            prompt: The user prompt string.
            system: Optional system prompt.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Max tokens to generate.
            stream: Whether to stream the response.

        Returns:
            Dict with keys: response, prompt_tokens, completion_tokens,
                            total_tokens, latency_ms, model.

        Raises:
            OllamaError: On connection failure or non-200 response.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        return self._call_with_retry(payload)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Call the Ollama /api/chat endpoint (OpenAI-compatible messages format).

        Args:
            model: Ollama model name.
            messages: List of {'role': ..., 'content': ...} dicts.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Max tokens to generate.

        Returns:
            Same structure as generate().
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        return self._call_chat_with_retry(payload)

    def list_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        try:
            resp = httpx.get(
                f"{self.base_url}/api/tags", timeout=10
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning(f"Could not list Ollama models: {exc}")
            return []

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def stats(self) -> dict[str, Any]:
        """Return aggregated usage statistics."""
        return {
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset aggregated statistics."""
        self._total_calls = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_latency_ms = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute generate call with retry logic."""
        attempt = 0
        last_exc: Exception = RuntimeError("Unknown error")
        wait = self.retry_delay

        while attempt < self.max_retries:
            try:
                return self._execute_generate(payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                attempt += 1
                logger.warning(
                    f"Ollama generate attempt {attempt}/{self.max_retries} failed: {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                wait = min(wait * 2, 30.0)
            except OllamaError:
                raise

        raise OllamaError(
            f"Ollama unavailable after {self.max_retries} attempts. Last error: {last_exc}"
        )

    def _call_chat_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute chat call with retry logic."""
        attempt = 0
        last_exc: Exception = RuntimeError("Unknown error")
        wait = self.retry_delay

        while attempt < self.max_retries:
            try:
                return self._execute_chat(payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                attempt += 1
                logger.warning(
                    f"Ollama chat attempt {attempt}/{self.max_retries} failed: {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                wait = min(wait * 2, 30.0)
            except OllamaError:
                raise

        raise OllamaError(
            f"Ollama unavailable after {self.max_retries} attempts. Last error: {last_exc}"
        )

    def _execute_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make the actual HTTP call to /api/generate and parse response."""
        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise httpx.TimeoutException(str(exc), request=None) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama /api/generate returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        text = data.get("response", "")

        # Ollama returns eval_count (completion tokens) and prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", self._estimate_tokens(payload.get("prompt", "")))
        completion_tokens = data.get("eval_count", self._estimate_tokens(text))

        self._update_stats(prompt_tokens, completion_tokens, elapsed_ms)

        return {
            "response": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": elapsed_ms,
            "model": payload.get("model", "unknown"),
        }

    def _execute_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make the actual HTTP call to /api/chat and parse response."""
        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise httpx.TimeoutException(str(exc), request=None) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama /api/chat returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        message = data.get("message", {})
        text = message.get("content", "")

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", self._estimate_tokens(text))

        self._update_stats(prompt_tokens, completion_tokens, elapsed_ms)

        return {
            "response": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": elapsed_ms,
            "model": payload.get("model", "unknown"),
        }

    def _update_stats(
        self, prompt_tokens: int, completion_tokens: int, latency_ms: float
    ) -> None:
        self._total_calls += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_latency_ms += latency_ms

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)
