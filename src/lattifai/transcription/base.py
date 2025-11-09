"""Base transcriber interface for LattifAI."""

from pathlib import Path
from typing import Protocol, Union


class BaseTranscriber(Protocol):
    """Base interface for transcription services."""

    async def transcribe(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file or URL
            output_dir: Directory for output files

        Returns:
            dict: Transcription results with metadata
        """
        ...

    async def transcribe_url(self, url: str) -> str:
        """
        Transcribe audio from URL.

        Args:
            url: Audio/video URL (e.g., YouTube)

        Returns:
            str: Transcribed text
        """
        ...
