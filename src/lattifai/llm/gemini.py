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

    # google-genai 2.x defaults HttpOptions.timeout to None (infinite). With no
    # timeout, a stalled server response keeps the TCP connection open until
    # Linux's TCP keepalive (default 7200s = 2h) tears it down — observed as a
    # 2h26m Release Tests hang on `gemini-3.1-pro-preview` (CI run 25904188835).
    #
    # Timeout strategy:
    #  - Default (None passed in) -> use the SDK's None semantics for the
    #    cached client. Callers that know their workload size SHOULD pass an
    #    explicit `http_timeout_ms` (e.g. audio transcription scales it by
    #    media duration), in which case a fresh non-cached client is built
    #    just for that call.
    #  - Units are milliseconds in google-genai 2.x (1.x → 2.x breaking change).
    #  - httpx interprets a single `timeout=N` value as the cap on every
    #    phase (connect/read/write/pool); since non-streaming generate_content
    #    sends no data until the full response is ready, this is effectively
    #    a budget for "how long the server can take to produce the response."

    def _get_client(self, http_timeout_ms: Optional[int] = None):
        """Return a google-genai Client.

        When ``http_timeout_ms`` is provided, build a fresh (un-cached) client
        with that HTTP timeout. When ``None``, reuse the cached default
        client (no explicit timeout) so repeated short calls don't pay the
        per-call construction cost.
        """
        from google import genai
        from google.genai.types import HttpOptions

        if not self._api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass api_key.")

        if http_timeout_ms is not None:
            return genai.Client(
                api_key=self._api_key,
                http_options=HttpOptions(timeout=http_timeout_ms),
            )

        if self._client is None:
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
        http_timeout_ms: Optional[int] = None,
    ):
        """Low-level Gemini generate_content for multimodal use (transcription).

        Args:
            contents: Gemini Part or list of Parts (audio, video, text).
            model: Model name override.
            config: GenerateContentConfig instance.
            http_timeout_ms: Optional per-call HTTP timeout in milliseconds.
                When provided, a fresh genai.Client with that timeout is
                built for this call (the default cached client is not used).
                Callers that know their workload (e.g. audio transcription
                scales by duration) should set this to avoid both infinite
                hangs and false timeouts on long syntheses.

        Returns:
            Raw GenerateContentResponse from Gemini SDK.
        """
        client = self._get_client(http_timeout_ms=http_timeout_ms)
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
        """Access the underlying genai.Client (cached, no explicit timeout).

        Prefer ``get_client(http_timeout_ms=...)`` when the caller knows the
        expected workload size — both ``files.upload`` and ``generate_content``
        can hang indefinitely on a stalled server with google-genai 2.x's
        default ``HttpOptions.timeout=None``.
        """
        return self._get_client()

    def get_client(self, http_timeout_ms: Optional[int] = None):
        """Get a genai.Client, optionally with a per-call HTTP timeout.

        When ``http_timeout_ms`` is provided, a fresh non-cached client is
        built so the timeout applies to both file uploads and content
        generation on that client. When ``None``, returns the cached client
        with whatever timeout it was constructed with (default: none).
        """
        return self._get_client(http_timeout_ms=http_timeout_ms)
