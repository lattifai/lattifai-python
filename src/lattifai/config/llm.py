"""LLM provider configuration for LattifAI.

Shared by TranslationConfig, DiarizationConfig, etc.
Each caller passes section= to bind LLMConfig to its config.toml section.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from lattifai.llm.base import BaseLLMClient


def resolve_toml_value(section: str, key: str) -> Optional[str]:
    """Read a value from config.toml [section].key.

    Shared helper for all config classes that read from config.toml.
    Returns None if not set or config system is unavailable.
    """
    try:
        from lattifai.cli.config import get_config_value

        return get_config_value(f"{section}.{key}")
    except (ImportError, OSError):
        return None


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Resolution order: explicit value > config.toml [section] > fallback.
    Uses None as sentinel — only None triggers config.toml lookup.
    Explicit values (including "gemini") are never overwritten.
    """

    provider: Optional[Literal["gemini", "openai"]] = None
    """LLM provider. None = resolve from config.toml, fallback to 'gemini'."""

    model_name: Optional[str] = None
    """Model name. None = resolve from config.toml -> fallback -> raise."""

    api_key: Optional[str] = None
    """API key. Falls back to GEMINI_API_KEY or OPENAI_API_KEY env var, then config.toml."""

    api_base_url: Optional[str] = None
    """Base URL for OpenAI-compatible endpoint (vLLM, SGLang, Ollama, etc.)."""

    section: str = ""
    """TOML section name for config.toml resolution (e.g. "translation")."""

    fallback_model: Optional[str] = None
    """Fallback model when config.toml has no value. Set by consumer (e.g. TranslationConfig)."""

    def __post_init__(self) -> None:
        """Resolve defaults from config.toml, then API key and base URL."""
        # Step 1: fill None fields from config.toml [section]
        if self.section:
            if self.model_name is None:
                self.model_name = resolve_toml_value(self.section, "model_name")
            if self.provider is None:
                saved = resolve_toml_value(self.section, "provider")
                if saved and saved in ("gemini", "openai"):
                    self.provider = saved  # type: ignore[assignment]

        # Step 2: apply fallbacks
        if self.model_name is None:
            self.model_name = self.fallback_model

        if self.provider is None:
            self.provider = "gemini"

        # Step 3: validate required fields
        if not self.model_name:
            if self.section:
                raise ValueError(
                    f"No model configured for [{self.section}].\n" f"  lai config set {self.section}.model_name <model>"
                )
            raise ValueError(
                "No model_name provided for LLMConfig.\n"
                "  Pass model_name= explicitly or set section= for config.toml lookup."
            )

        # Step 4: resolve API credentials
        if self.api_key is None:
            self.api_key = self._resolve_api_key()

        if self.provider == "openai" and self.api_base_url is None:
            self.api_base_url = self._resolve_base_url()

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
