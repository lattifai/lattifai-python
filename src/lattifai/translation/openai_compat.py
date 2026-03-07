"""OpenAI-compatible translator for vLLM, local deployments, and OpenAI API."""

import asyncio
import logging

from lattifai.config.translation import TranslationConfig
from lattifai.translation.base import BaseTranslator

logger = logging.getLogger(__name__)


class OpenAITranslator(BaseTranslator):
    """Translator using OpenAI-compatible API (works with vLLM, Ollama, etc.)."""

    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        base = self.config.api_base_url or "openai"
        return f"openai:{self.config.model_name}@{base}"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider. Install with: pip install openai")

            kwargs = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            elif not self.config.api_base_url:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key in config.")
            else:
                # For local deployments, api_key may not be needed
                kwargs["api_key"] = self.config.api_key or "not-needed"

            if self.config.api_base_url:
                kwargs["base_url"] = self.config.api_base_url

            self._client = OpenAI(**kwargs)
        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        client = self._get_client()

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            ),
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from OpenAI-compatible API")

        return content
