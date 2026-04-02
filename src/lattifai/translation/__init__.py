"""Translation module for LattifAI."""

from lattifai.config.translation import TranslationConfig

from .base import BaseTranslator

__all__ = [
    "BaseTranslator",
    "create_translator",
]


def create_translator(config: TranslationConfig) -> BaseTranslator:
    """Create a translator instance based on LLM configuration.

    Args:
        config: Translation configuration.

    Returns:
        BaseTranslator instance backed by the configured LLM provider.
    """
    return BaseTranslator(config, config.llm.create_client())
