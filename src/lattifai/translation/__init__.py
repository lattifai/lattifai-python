"""Translation module for LattifAI."""

from lattifai.config.translation import TranslationConfig

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
        BaseTranslator instance.
    """
    if config.provider == "gemini":
        from .gemini import GeminiTranslator

        return GeminiTranslator(config)
    elif config.provider == "openai":
        from .openai_compat import OpenAITranslator

        return OpenAITranslator(config)
    else:
        raise ValueError(
            f"Unsupported translation provider: '{config.provider}'. " f"Supported providers: 'gemini', 'openai'."
        )
