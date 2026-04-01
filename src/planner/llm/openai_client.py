"""OpenAI-compatible client for LLM interactions (works with vLLM, OpenAI, etc.)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Client for OpenAI-compatible API endpoints (vLLM, OpenAI, etc.)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        """
        Initialize OpenAI-compatible client.

        Args:
            model: Model name to use. Falls back to OPENAI_MODEL env var.
            base_url: Base URL for the API. Falls back to OPENAI_BASE_URL env var.
            api_key: API key (optional for vLLM). Falls back to OPENAI_API_KEY env var.
        """
        self.model = model or os.getenv("OPENAI_MODEL", "default")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")  # vLLM doesn't need real key

        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.timeout = float(os.getenv("LLM_TIMEOUT", "120.0"))  # 2 minute default timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Send chat messages to OpenAI-compatible API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Response dict with 'message' containing 'content'
        """
        try:
            # Log the request
            if messages:
                last_msg = messages[-1]
                logger.info(
                    f"[LLM REQUEST] Role: {last_msg.get('role')}, Content length: {len(last_msg.get('content', ''))} chars"
                )

            # Build request payload
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048,
            }

            if format_json:
                # Request structured JSON output
                payload["response_format"] = {"type": "json_object"}

            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key and self.api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Make request
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.chat_endpoint,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            # Extract content from OpenAI format
            content = data["choices"][0]["message"]["content"]

            # Log response
            logger.info("=" * 80)
            logger.info(f"[LLM RESPONSE] Model: {self.model}, Response length: {len(content)} chars")
            logger.info("[LLM RESPONSE CONTENT - START]")
            logger.info(content)
            logger.info("[LLM RESPONSE CONTENT - END]")
            logger.info("=" * 80)

            # Return in Ollama-compatible format for backward compatibility
            return {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling LLM API: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"LLM API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error calling LLM API: {e}")
            raise RuntimeError(f"LLM API connection failed: {e}") from e
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input prompt string
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        logger.info(
            f"[LLM GENERATE] Prompt length: {len(prompt)} chars, JSON format: {format_json}, Temperature: {temperature}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, format_json=format_json, temperature=temperature)
        return str(response["message"]["content"])

    def extract_structured_data(
        self,
        prompt: str,
        schema_description: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Extract structured data from prompt using JSON format.

        Args:
            prompt: Input prompt describing what to extract
            schema_description: Description of expected JSON schema
            temperature: Lower temperature for more consistent extraction

        Returns:
            Parsed JSON dict
        """
        full_prompt = f"""{prompt}

{schema_description}

Return ONLY valid JSON matching the schema above. Do not include any explanation or additional text."""

        response_text = self.generate_completion(
            full_prompt, format_json=True, temperature=temperature
        )

        try:
            result: dict[str, Any] = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            logger.error(f"JSON error: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            with httpx.Client(timeout=5.0) as client:
                # Try a minimal health check (some endpoints have /health or /v1/models)
                response = client.get(f"{self.base_url}/v1/models")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM service not available: {e}")
            return False
