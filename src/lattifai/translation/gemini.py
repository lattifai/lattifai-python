"""Gemini translator using Google GenAI SDK."""

import asyncio
import logging

from lattifai.config.translation import TranslationConfig
from lattifai.translation.base import BaseTranslator

logger = logging.getLogger(__name__)


class GeminiTranslator(BaseTranslator):
    """Translator using Google Gemini API with JSON structured output."""

    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return f"gemini:{self.config.model_name}"

    def _get_client(self):
        if self._client is None:
            from google import genai

            if not self.config.api_key:
                raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass api_key in config.")
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini API with JSON response mode."""
        from google.genai.types import GenerateContentConfig

        client = self._get_client()
        config = GenerateContentConfig(response_mime_type="application/json")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=config,
            ),
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini API")

        return response.text
