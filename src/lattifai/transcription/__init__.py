"""Transcription module for LattifAI."""

from typing import TYPE_CHECKING, Optional

from lattifai.config import TranscriptionConfig

from .gemini import GeminiTranscriber
from .lattifai import LattifAITranscriber

if TYPE_CHECKING:
    from .base import BaseTranscriber

__all__ = [
    "LattifAITranscriber",
    "GeminiTranscriber",
    "create_transcriber",
]


def create_transcriber(
    transcription_config: TranscriptionConfig,
) -> "BaseTranscriber":
    """
    Create a transcriber instance based on model_name in configuration.

    This factory method automatically selects the appropriate transcriber
    implementation based on the model_name specified in TranscriptionConfig.

    Args:
        transcription_config: Transcription configuration. If None, uses default
                            (which defaults to Gemini 2.5 Pro).

    Returns:
        BaseTranscriber: An instance of GeminiTranscriber or LattifAITranscriber

    Raises:
        ValueError: If model_name is not supported or ambiguous.

    Example:
        >>> from lattifai.config import TranscriptionConfig
        >>> from lattifai.transcription import create_transcriber
        >>>
        >>> # Create Gemini transcriber (default)
        >>> transcriber = create_transcriber()
        >>>
        >>> # Create specific transcriber
        >>> config = TranscriptionConfig(model_name="gemini-2.5-pro")
        >>> transcriber = create_transcriber(config)
        >>>
        >>> # Use local model
        >>> config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b")
        >>> transcriber = create_transcriber(config)
    """
    model_name = transcription_config.model_name

    # Gemini models (API-based)
    if "gemini" in model_name:
        assert (
            transcription_config.gemini_api_key is not None
        ), "Gemini API key must be provided in TranscriptionConfig for Gemini models."
        return GeminiTranscriber(transcription_config=transcription_config)

    # LattifAI local models (HuggingFace/NVIDIA models)
    # Pattern: nvidia/*, iic/*, or any HF model path
    elif "/" in model_name:
        return LattifAITranscriber(transcription_config=transcription_config)

    else:
        # No clear indicator, raise error
        raise ValueError(
            f"Cannot determine transcriber for model_name='{transcription_config.model_name}'. "
            f"Supported patterns: \n"
            f"  - Gemini API models: 'gemini-2.5-pro', 'gemini-3-pro-preview'\n"
            f"  - Local HF models: 'nvidia/parakeet-*', 'iic/SenseVoiceSmall', etc.\n"
            f"Please specify a valid model_name."
        )
