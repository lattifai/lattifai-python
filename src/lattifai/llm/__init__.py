"""Unified LLM client abstraction for LattifAI.

All text/JSON generation goes through OpenAI-compatible protocol.
Gemini models use Google's OpenAI-compatible endpoint; GeminiClient
is reserved for multimodal use (transcription with audio/video).

Usage:
    from lattifai.llm import create_client

    # Gemini (via OpenAI-compatible endpoint)
    client = create_client("gemini", api_key="...", model="gemini-3-flash-preview")
    result = await client.generate_json("Return a JSON array of colors")

    # OpenAI / vLLM / SGLang / Ollama
    client = create_client("openai", api_key="...", model="gpt-4o")
    client = create_client("openai", base_url="http://localhost:8000/v1", model="Qwen3-ASR")

    # For multimodal (audio/video) — use GeminiClient directly
    from lattifai.llm import GeminiClient
    client = GeminiClient(api_key="...", model="gemini-2.5-flash")
    response = await client.generate_content(parts, config=config)
"""

import os
from typing import Optional

from lattifai.llm.base import BaseLLMClient, parse_json_response
from lattifai.llm.gemini import GeminiClient
from lattifai.llm.openai_compat import OpenAIClient

# Gemini OpenAI-compatible endpoint
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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

    Both 'gemini' and 'openai' providers return an OpenAIClient.
    Gemini models are accessed via Google's OpenAI-compatible endpoint.
    'transformers' provider loads models locally via HuggingFace transformers.

    Args:
        provider: Provider name ('gemini', 'openai', or 'transformers').
        api_key: API key. Falls back to GEMINI_API_KEY or OPENAI_API_KEY env var.
        model: Default model name (HuggingFace ID for 'transformers').
        base_url: Base URL override for OpenAI-compatible APIs.
        **kwargs: Extra provider-specific arguments.
            For 'transformers': device, dtype, max_new_tokens.

    Returns:
        Configured LLM client instance.
    """
    provider = provider.lower()

    if provider == "gemini":
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        base_url = base_url or GEMINI_OPENAI_BASE_URL
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
    elif provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
    elif provider in ("transformers", "huggingface", "hf"):
        from lattifai.llm.transformers import TransformersClient

        return TransformersClient(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: 'gemini', 'openai', 'transformers'")
