"""Transcription module for LattifAI."""

from typing import Optional

from lattifai.config import EventConfig, TranscriptionConfig

from .base import BaseTranscriber

__all__ = [
    "BaseTranscriber",
    "create_transcriber",
]


def create_transcriber(
    transcription_config: TranscriptionConfig,
    event_config: Optional[EventConfig] = None,
) -> "BaseTranscriber":
    """
    Create a transcriber instance based on model_name in configuration.

    Optionally injects an event_detector for VAD-based audio segmentation.
    When ``event_config`` is provided (or defaults are used), a
    ``LattifAIEventDetector`` is created and attached to the transcriber
    so that long audio is automatically split into speech segments.

    Args:
        transcription_config: Transcription configuration.
        event_config: Event detection configuration. When ``enabled=True``,
            a LattifAIEventDetector is injected into the transcriber for
            VAD segmentation. Pass ``None`` to skip event detection.

    Returns:
        BaseTranscriber: A transcriber instance with optional event_detector.
    """
    model_name = transcription_config.model_name

    # vLLM/SGLang-served models (any model with api_base_url set)
    if transcription_config.api_base_url:
        from .vllm import VLLMTranscriber

        transcriber = VLLMTranscriber(transcription_config=transcription_config)

    # Gemini models (API-based)
    elif "gemini" in model_name:
        assert (
            transcription_config.gemini_api_key is not None
        ), "Gemini API key must be provided in TranscriptionConfig for Gemini models."
        from .gemini import GeminiTranscriber

        transcriber = GeminiTranscriber(transcription_config=transcription_config)

    # LattifAI local models (HuggingFace/NVIDIA models)
    # Pattern: nvidia/*, iic/*, or any HF model path
    elif "/" in model_name:
        from .lattifai import LattifAITranscriber

        transcriber = LattifAITranscriber(transcription_config=transcription_config)

    else:
        raise ValueError(
            f"Cannot determine transcriber for model_name='{transcription_config.model_name}'. "
            f"Supported patterns: \n"
            f"  - vLLM/SGLang models: set api_base_url='http://localhost:8000/v1'\n"
            f"  - Gemini API models: 'gemini-2.5-pro', 'gemini-2.5-flash', etc.\n"
            f"  - Local HF models: 'nvidia/parakeet-*', 'iic/SenseVoiceSmall', etc.\n"
            f"Please specify a valid model_name."
        )

    # Inject event_detector for transcribers that need VAD segmentation
    # (e.g. VLLMTranscriber, LattifAITranscriber — not GeminiTranscriber)
    if transcriber.needs_vad:
        if event_config is None:
            event_config = EventConfig(enabled=True)
        else:
            event_config.enabled = True
        from lattifai.event import LattifAIEventDetector

        event_config.client_wrapper = transcription_config.client_wrapper
        event_config.model_path = transcription_config.lattice_model_path
        transcriber.event_detector = LattifAIEventDetector(config=event_config)

    return transcriber
