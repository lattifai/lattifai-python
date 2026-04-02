"""Gemini LLM client using Google GenAI SDK."""

import asyncio
import logging
from typing import Any, Optional

from lattifai.llm.base import BaseLLMClient, parse_json_response

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """LLM client for Google Gemini models.

    Wraps google-genai SDK with lazy initialization and async execution.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        if self._client is None:
            from google import genai

            if not self._api_key:
                raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass api_key.")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        response = await self._call(prompt, model=model, system=system, temperature=temperature)
        if not response.text:
            raise RuntimeError("Empty response from Gemini API")
        return response.text

    async def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        response = await self._call(prompt, model=model, system=system, temperature=temperature, json_mode=True)
        if not response.text:
            raise RuntimeError("Empty response from Gemini API")
        return parse_json_response(response.text)

    async def generate_content(
        self,
        contents,
        *,
        model: Optional[str] = None,
        config=None,
    ):
        """Low-level Gemini generate_content for multimodal use (transcription).

        Args:
            contents: Gemini Part or list of Parts (audio, video, text).
            model: Model name override.
            config: GenerateContentConfig instance.

        Returns:
            Raw GenerateContentResponse from Gemini SDK.
        """
        client = self._get_client()
        resolved_model = self._resolve_model(model)

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            ),
        )

    async def _call(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ):
        """Internal: build config and call Gemini API."""
        from google.genai.types import GenerateContentConfig

        client = self._get_client()
        resolved_model = self._resolve_model(model)

        config_kwargs = {}
        if system:
            config_kwargs["system_instruction"] = system
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = GenerateContentConfig(**config_kwargs) if config_kwargs else None

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=resolved_model,
                contents=prompt,
                config=config,
            ),
        )

    # -- Convenience: expose underlying client for file uploads etc. --

    @property
    def raw_client(self):
        """Access the underlying genai.Client for file uploads, etc."""
        return self._get_client()
