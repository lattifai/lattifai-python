"""Base LLM client abstraction for LattifAI."""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Unified interface for LLM providers.

    Provides two core methods: text generation and structured JSON generation.
    All methods are async-first; sync wrappers are provided for non-async callers.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        self._api_key = api_key
        self._default_model = model

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g. 'gemini', 'openai')."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User message / prompt text.
            model: Model name override (uses default if None).
            system: System instruction.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """Generate structured JSON output.

        Args:
            prompt: User message / prompt text.
            model: Model name override (uses default if None).
            system: System instruction.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON (dict or list).
        """

    def _resolve_model(self, model: Optional[str] = None) -> str:
        """Resolve model name from argument or default."""
        resolved = model or self._default_model
        if not resolved:
            raise ValueError(f"No model specified for {self.provider_name} client")
        return resolved

    # -- Sync wrappers --

    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for generate()."""
        return _run_async(self.generate(prompt, **kwargs))

    def generate_json_sync(self, prompt: str, **kwargs) -> Any:
        """Synchronous wrapper for generate_json()."""
        return _run_async(self.generate_json(prompt, **kwargs))


def _run_async(coro):
    """Run an async coroutine from sync context, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside an existing event loop — use run_in_executor with a new loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, handling markdown code blocks and thinking tokens."""
    text = text.strip()
    # Strip thinking tokens from reasoning models (e.g. Qwen3/3.5, DeepSeek-R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        # Strip markdown code fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Fallback: extract first JSON object if surrounded by non-JSON text
    if not text.startswith(("{", "[")):
        json_match = re.search(r"(\{[^{}]*\}|\[.*\])", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
    return json.loads(text)
