"""LLM provider configuration for LattifAI.

Shared by TranslationConfig, DiarizationConfig, etc.
Each caller passes section= to bind LLMConfig to its config.toml section.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lattifai.llm.base import BaseLLMClient


def resolve_toml_value(section: str, key: str) -> Optional[str]:
    """Read a value from config.toml [section].key as a string.

    Shared helper for all config classes that read from config.toml.
    Supports dotted sections for nested tables (e.g., ``"diarization.llm"``).
    Returns None if not set or config system is unavailable.
    """
    try:
        from lattifai.config.toml_mixin import resolve_toml_raw_value

        raw = resolve_toml_raw_value(section, key)
        if raw is None:
            return None
        return str(raw)
    except (ImportError, OSError):
        return None


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Resolution order: explicit value > config.toml [section] > fallback.
    Uses None as sentinel — only None triggers config.toml lookup.

    Provider is inferred from model_name prefix:
    - "gemini*" → gemini (GEMINI_API_KEY + Google endpoint)
    - otherwise → openai (OPENAI_API_KEY + user-provided or default endpoint)
    """

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

    reasoning: bool = True
    """Enable reasoning/thinking for models that support it (e.g. doubao-seed, o1, deepseek-r1).
    When False, reasoning_content is stripped from responses.
    Default True — set to False for faster non-reasoning responses."""

    @staticmethod
    def _infer_provider(model_name: Optional[str]) -> str:
        """Infer LLM provider from model name prefix.

        Returns "gemini" for models starting with "gemini", "openai" otherwise.
        """
        if model_name and model_name.startswith("gemini"):
            return "gemini"
        return "openai"

    @property
    def provider(self) -> str:
        """Inferred provider based on model_name. Read-only."""
        return self._infer_provider(self.model_name)

    def __post_init__(self) -> None:
        """Resolve defaults from config.toml, then API key and base URL."""
        # Step 1: fill model_name and reasoning from config.toml [section]
        if self.section:
            if not self.model_name:
                self.model_name = resolve_toml_value(self.section, "model_name")
            reasoning_val = resolve_toml_value(self.section, "reasoning")
            if reasoning_val is not None:
                self.reasoning = reasoning_val.lower() in ("true", "1", "yes")

        # Step 2: apply fallbacks
        if not self.model_name:
            self.model_name = self.fallback_model

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

        # Step 4: resolve API credentials (provider is now inferred from model_name)
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
            reasoning=self.reasoning,
        )

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key: [section].api_key > env var > global config > .env."""
        # Section-specific api_key has highest priority (e.g. [translation.llm].api_key)
        if self.section:
            section_key = resolve_toml_value(self.section, "api_key")
            if section_key:
                return section_key

        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass

        if self.provider == "gemini":
            env_key = os.environ.get("GEMINI_API_KEY")
            config_key = "GEMINI_API_KEY"
        else:
            env_key = os.environ.get("OPENAI_API_KEY")
            config_key = "OPENAI_API_KEY"

        if env_key:
            return env_key

        try:
            from lattifai.cli.config import get_config_value

            return get_config_value(config_key)
        except (ImportError, OSError):
            return None

    def _resolve_base_url(self) -> Optional[str]:
        """Resolve base URL: [section].api_base_url > env var > global config."""
        # Section-specific has highest priority
        if self.section:
            section_url = resolve_toml_value(self.section, "api_base_url")
            if section_url:
                return section_url

        env_url = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if env_url:
            return env_url

        try:
            from lattifai.cli.config import get_config_value

            return get_config_value("OPENAI_API_BASE_URL")
        except (ImportError, OSError):
            return None
