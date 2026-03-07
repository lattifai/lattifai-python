"""Translation module for LattifAI."""

from lattifai.config.translation import TranslationConfig
from lattifai.llm import create_client

from .base import BaseTranslator

__all__ = [
    "BaseTranslator",
    "create_translator",
]


def create_translator(config: TranslationConfig) -> BaseTranslator:
    """Create a translator instance based on provider in configuration.

    Args:
        config: Translation configuration.

    Returns:
        BaseTranslator instance backed by the configured LLM provider.
    """
    client = create_client(
        provider=config.provider,
        api_key=config.api_key,
        model=config.model_name,
        base_url=config.api_base_url,
    )
    return BaseTranslator(config, client)
