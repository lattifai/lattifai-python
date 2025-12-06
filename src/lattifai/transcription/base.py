"""Base transcriber abstractions for LattifAI."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from lattifai.audio2 import AudioData
from lattifai.caption import Caption
from lattifai.config import TranscriptionConfig
from lattifai.logging import get_logger


class BaseTranscriber(ABC):
    """
    Base class that standardizes how transcribers handle inputs/outputs.

    Subclasses only need to implement the media-specific transcription
    routines, while the base class handles URL vs file routing and saving
    the resulting transcript to disk.
    """

    # Subclasses should override these properties
    file_suffix: str = ".txt"
    supports_url: bool = True
    """Whether this transcriber supports direct URL transcription."""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Initialize base transcriber.

        Args:
            config: Transcription configuration.
        """
        # Initialize config with default if not provided
        if config is None:
            config = TranscriptionConfig()

        self.config = config
        self.logger = get_logger("transcription")

    @property
    def name(self) -> str:
        """Human-readable name of the transcriber."""

    @property
    def file_name(self) -> str:
        """File name identifier for the transcriber."""
        return f"{self.name.replace('/', '_')}{self.file_suffix}"

    async def __call__(self, url_or_data: Union[str, AudioData], language: Optional[str] = None) -> str:
        """Main entry point for transcription."""
        return await self.transcribe(url_or_data, language=language)

    async def transcribe(self, url_or_data: Union[str, AudioData], language: Optional[str] = None) -> str:
        """
        Route transcription based on input type.

        For URL inputs, only works if the transcriber supports direct URL transcription.
        Otherwise, the caller should download the media first and pass AudioData.

        Args:
            url_or_data: URL string or AudioData object to transcribe.
            language: Optional language code for transcription (e.g., 'en', 'zh').
        """
        if isinstance(url_or_data, AudioData):
            return await self.transcribe_file(url_or_data, language=language)
        elif self._is_url(url_or_data):
            if self.supports_url:
                return await self.transcribe_url(url_or_data, language=language)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} does not support direct URL transcription. "
                    f"Please download the media first and pass AudioData instead."
                )
        return await self.transcribe_file(url_or_data, language=language)  # file path

    @abstractmethod
    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio from a remote URL (e.g., YouTube).

        Args:
            url: URL of the audio/video to transcribe.
            language: Optional language code for transcription.
        """

    @abstractmethod
    async def transcribe_file(
        self, media_file: Union[str, Path, AudioData], language: Optional[str] = None
    ) -> Union[str, Caption]:
        """
        Transcribe audio from a local media file.

        Args:
            media_file: Path to media file or AudioData object.
            language: Optional language code for transcription.
        """

    @abstractmethod
    def write(self, transcript: Union[str, Caption], output_file: Path, encoding: str = "utf-8") -> Path:
        """
        Persist transcript text to disk and return the file path.
        """

    @staticmethod
    def _is_url(value: str) -> bool:
        """Best-effort detection of web URLs."""
        return value.startswith(("http://", "https://"))
