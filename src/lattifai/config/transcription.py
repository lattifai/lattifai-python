"""Transcription service configuration for LattifAI."""

import os
from dataclasses import dataclass
from typing import Literal, Optional

from ..utils import _select_device

SUPPORTED_TRANSCRIPTION_MODELS = Literal[
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/canary-1b-v2",
    "iic/SenseVoiceSmall",
]


@dataclass
class TranscriptionConfig:
    """
    Transcription service configuration.

    Settings for audio/video transcription using various providers.
    """

    model_name: SUPPORTED_TRANSCRIPTION_MODELS = "gemini-2.5-pro"  # gemini-3-pro-preview
    """Model name for transcription."""

    gemini_api_key: Optional[str] = None
    """Gemini API key. If None, reads from GEMINI_API_KEY environment variable."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device for transcription models."""

    max_retries: int = 0
    """Maximum number of retry attempts for failed transcription requests."""

    force_overwrite: bool = False
    """Force overwrite existing transcription files."""

    verbose: bool = False
    """Enable debug logging for transcription operations."""

    language: Optional[str] = None
    """Target language code for transcription (e.g., 'en', 'zh', 'ja')."""

    # Speaker Diarization
    speaker_diarization: bool = False
    """Apply diarization to assign speaker labels to each segment."""

    num_speakers: Optional[int] = None
    """Number of speakers, when known. If not set, diarization will attempt to infer the number of speakers."""

    min_speakers: Optional[int] = None
    """Minimum number of speakers. Has no effect when `num_speakers` is provided."""

    max_speakers: Optional[int] = None
    """Maximum number of speakers. Has no effect when `num_speakers` is provided."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""

        if self.model_name not in SUPPORTED_TRANSCRIPTION_MODELS.__args__:
            raise ValueError(
                f"Unsupported model_name: '{self.model_name}'. "
                f"Supported models are: {SUPPORTED_TRANSCRIPTION_MODELS.__args__}"
            )

        # Load environment variables from .env file
        from dotenv import find_dotenv, load_dotenv

        # Try to find and load .env file from current directory or parent directories
        load_dotenv(find_dotenv(usecwd=True))

        # Auto-load Gemini API key from environment if not provided
        if self.gemini_api_key is None:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Validate max_retries
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Validate device
        if self.device not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got '{self.device}'")

        if self.device == "auto":
            self.device = _select_device(self.device)
