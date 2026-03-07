"""Unified LLM client abstraction for LattifAI.

Provides a consistent interface across Gemini, OpenAI, vLLM, and other providers.

Usage:
    from lattifai.llm import create_client

    # Gemini
    client = create_client("gemini", api_key="...", model="gemini-3-flash-preview")
    result = await client.generate_json("Return a JSON array of colors")

    # OpenAI / vLLM
    client = create_client("openai", api_key="...", model="gpt-4o")
    client = create_client("openai", base_url="http://localhost:8000/v1", model="Qwen3-ASR")
"""

import os
from typing import Optional

from lattifai.llm.base import BaseLLMClient, parse_json_response
from lattifai.llm.gemini import GeminiClient
from lattifai.llm.openai_compat import OpenAIClient

__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "OpenAIClient",
    "create_client",
    "parse_json_response",
]


def create_client(
    provider: str = "gemini",
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: Provider name ('gemini' or 'openai').
        api_key: API key. Falls back to GEMINI_API_KEY or OPENAI_API_KEY env var.
        model: Default model name.
        base_url: Base URL for OpenAI-compatible APIs (vLLM, SGLang, etc.).
        **kwargs: Extra provider-specific arguments.

    Returns:
        Configured BaseLLMClient instance.
    """
    provider = provider.lower()

    if provider == "gemini":
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        return GeminiClient(api_key=api_key, model=model, **kwargs)
    elif provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: 'gemini', 'openai'")
