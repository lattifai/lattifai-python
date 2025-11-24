"""Base transcriber abstractions for LattifAI."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from lattifai.audio2 import AudioData
from lattifai.subtitle import Supervision


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
    supports_url: bool = True
    """Whether this transcriber supports direct URL transcription."""

    async def __call__(self, url_or_data: Union[str, AudioData]) -> str:
        """Main entry point for transcription."""
        return await self.transcribe(url_or_data)

    async def transcribe(self, url_or_data: Union[str, AudioData]) -> str:
        if isinstance(url_or_data, AudioData):
            return await self.transcribe_file(url_or_data)
        elif self._is_url(url_or_data):
            if not self.supports_url:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support URL transcription. "
                    f"Please download the file first and use transcribe_file()."
                )
            return await self.transcribe_url(url_or_data)  # URL
        return await self.transcribe_file(url_or_data)  # file path

    @abstractmethod
    async def transcribe_url(self, url: str) -> str:
        """
        Transcribe audio from a remote URL (e.g., YouTube).
        """

    @abstractmethod
    async def transcribe_file(self, media_file: Union[str, Path, AudioData]) -> Union[str, List[Supervision]]:
        """
        Transcribe audio from a local media file.
        """

    @abstractmethod
    def write(self, transcript: Union[str, List[Supervision]], output_file: Path, encoding: str = "utf-8") -> Path:
        """
        Persist transcript text to disk and return the file path.
        """

    @staticmethod
    def _is_url(value: str) -> bool:
        """Best-effort detection of web URLs."""
        return value.startswith(("http://", "https://"))
