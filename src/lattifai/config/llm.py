"""LLM provider configuration for LattifAI.

Shared by DiarizationConfig, TranslationConfig, and any future LLM consumer.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from lattifai.llm.base import BaseLLMClient


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Resolves API key and base URL from explicit values, environment variables,
    or ~/.lattifai/config.toml — in that order.
    """

    provider: Literal["gemini", "openai"] = "gemini"
    """LLM provider: 'gemini' or 'openai' (OpenAI-compatible)."""

    model_name: str = "gemini-2.5-flash"
    """Model name."""

    api_key: Optional[str] = None
    """API key. Falls back to GEMINI_API_KEY or OPENAI_API_KEY env var, then config.toml."""

    api_base_url: Optional[str] = None
    """Base URL for OpenAI-compatible endpoint (vLLM, SGLang, Ollama, etc.)."""

    def __post_init__(self) -> None:
        """Resolve API key and base URL from environment / config.toml."""
        if self.api_key is None:
            self.api_key = self._resolve_api_key()

        if self.provider == "openai":
            if self.api_base_url is None:
                self.api_base_url = self._resolve_base_url()
            if self.model_name == "gemini-2.5-flash":
                # User switched to openai but kept the gemini default model — try env override
                env_model = os.environ.get("OPENAI_MODEL")
                if env_model:
                    self.model_name = env_model

    def create_client(self) -> "BaseLLMClient":
        """Create an LLM client from this configuration."""
        from lattifai.llm import create_client

        return create_client(
            self.provider,
            api_key=self.api_key,
            model=self.model_name,
            base_url=self.api_base_url,
        )

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key: env var > .env > config.toml."""
        # Load .env so keys defined there become visible via os.environ
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass

        if self.provider == "gemini":
            env_key = os.environ.get("GEMINI_API_KEY")
            config_key = "gemini_api_key"
        else:
            env_key = os.environ.get("OPENAI_API_KEY")
            config_key = "openai_api_key"

        if env_key:
            return env_key

        try:
            from lattifai.cli.config import get_config_value

            return get_config_value(config_key)
        except (ImportError, OSError):
            return None

    def _resolve_base_url(self) -> Optional[str]:
        """Resolve base URL for OpenAI-compatible providers: env var > config.toml."""
        env_url = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if env_url:
            return env_url

        try:
            from lattifai.cli.config import get_config_value

            return get_config_value("openai_api_base_url")
        except (ImportError, OSError):
            return None
