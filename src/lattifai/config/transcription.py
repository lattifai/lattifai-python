"""Transcription service configuration for LattifAI."""

import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TranscriptionConfig:
    """
    Transcription service configuration.

    Settings for audio/video transcription using various providers.
    """

    model_name: str = "gemini-2.5-pro"
    """Model name for transcription."""

    gemini_api_key: Optional[str] = None
    """Gemini API key. If None, reads from GEMINI_API_KEY environment variable."""

    device: Literal["cpu", "cuda", "mps"] = "cpu"
    """Computation device for transcription models."""

    max_retries: int = 0
    """Maximum number of retry attempts for failed transcription requests."""

    force_overwrite: bool = False
    """Force overwrite existing transcription files."""

    verbose: bool = False
    """Enable debug logging for transcription operations."""

    language: Optional[str] = None
    """Target language code for transcription (e.g., 'en', 'zh', 'ja')."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""

        # Load environment variables from .env file
        from dotenv import find_dotenv, load_dotenv

        # Try to find and load .env file from current directory or parent directories
        load_dotenv(find_dotenv(usecwd=True))

        # Auto-load Gemini API key from environment if not provided
        if self.gemini_api_key is None:
            object.__setattr__(self, "gemini_api_key", os.environ.get("GEMINI_API_KEY"))

        # Validate max_retries
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'")
