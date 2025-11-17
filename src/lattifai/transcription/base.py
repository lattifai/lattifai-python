"""Base transcriber abstractions for LattifAI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union


class BaseTranscriber(ABC):
    """
    Base class that standardizes how transcribers handle inputs/outputs.

    Subclasses only need to implement the media-specific transcription
    routines, while the base class handles URL vs file routing and saving
    the resulting transcript to disk.
    """

    # Subclasses should override these properties
    name: str = "Transcriber"
    file_suffix: str = ".txt"

    async def transcribe(
        self,
        media_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Transcribe an audio/video source and persist the transcript.

        Args:
            media_path: Path to audio file or URL.
            output_dir: Directory for output files.

        Returns:
            dict: Transcription results with metadata.
        """
        media_path_str = str(media_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if self._is_url(media_path_str):
            transcript = await self.transcribe_url(media_path_str)
            media_source_for_name = media_path_str
        else:
            transcript = await self.transcribe_file(Path(media_path))
            media_source_for_name = str(media_path)

        output_file = self._write_transcript(output_dir_path, media_source_for_name, transcript)
        return self._build_result(transcript, output_file)

    @abstractmethod
    async def transcribe_url(self, url: str) -> str:
        """
        Transcribe audio from a remote URL (e.g., YouTube).
        """

    @abstractmethod
    async def transcribe_file(self, media_file_path: Union[str, Path]) -> str:
        """
        Transcribe audio from a local media file.
        """

    def _build_result(self, transcript: str, output_file: Path) -> Dict[str, Any]:
        """
        Compose the result dictionary. Subclasses can override to add metadata.
        """
        return {"transcript": transcript, "output_file": str(output_file)}

    def _write_transcript(self, output_dir: Path, media_source: str, transcript: str) -> Path:
        """
        Persist transcript text to disk and return the file path.
        """
        file_stem = Path(media_source).stem or "transcript"
        output_file = output_dir / f"{file_stem}_{self.name}{self.file_suffix}"
        output_file.write_text(transcript, encoding="utf-8")
        return output_file

    @staticmethod
    def _is_url(value: str) -> bool:
        """Best-effort detection of web URLs."""
        return value.startswith(("http://", "https://"))
