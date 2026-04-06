"""OpenAI-compatible LLM client for OpenAI, vLLM, SGLang, Ollama, etc."""

import asyncio
import logging
from typing import Any, Optional

from lattifai.llm.base import BaseLLMClient, parse_json_response

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI-compatible APIs.

    Works with OpenAI, vLLM, SGLang, Ollama, and any provider that
    implements the /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning: bool = True,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self._base_url = base_url
        self._reasoning = reasoning
        self._client = None

    @property
    def provider_name(self) -> str:
        model = self._default_model or ""
        if model.startswith("gemini"):
            return "gemini"
        return "openai"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            elif not self._base_url:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")
            else:
                kwargs["api_key"] = "not-needed"

            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = OpenAI(**kwargs)
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
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from OpenAI-compatible API")
        if not self._reasoning:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning:
                logger.debug("Stripped %d chars of reasoning_content (reasoning=False)", len(reasoning))
        return content

    async def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        # Try json_mode first; fall back to plain text for thinking models
        # (Qwen3/3.5, DeepSeek-R1) that don't support response_format=json_object
        try:
            response = await self._call(prompt, model=model, system=system, temperature=temperature, json_mode=True)
        except Exception:
            logger.debug("json_mode not supported, retrying without response_format constraint")
            response = await self._call(prompt, model=model, system=system, temperature=temperature, json_mode=False)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from OpenAI-compatible API")
        return parse_json_response(content)

    async def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        **kwargs,
    ):
        """Low-level chat completions for multimodal or multi-turn use.

        Args:
            messages: OpenAI-format message list.
            model: Model name override.
            temperature: Sampling temperature.
            json_mode: Request JSON output.
            **kwargs: Extra arguments passed to completions.create().

        Returns:
            Raw ChatCompletion response.
        """
        client = self._get_client()
        resolved_model = self._resolve_model(model)

        # Extract timeout from kwargs — it's a client-level param, not a create() param
        timeout = kwargs.pop("timeout", None)

        create_kwargs = {"model": resolved_model, "messages": messages, **kwargs}
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}
        if timeout is not None:
            create_kwargs["timeout"] = timeout

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(**create_kwargs),
        )
        if not self._reasoning:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning:
                logger.debug("Stripped %d chars of reasoning_content (reasoning=False)", len(reasoning))
                response.choices[0].message.reasoning_content = None
        return response

    async def _call(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ):
        """Internal: build messages and call chat completions."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(messages, model=model, temperature=temperature, json_mode=json_mode)
