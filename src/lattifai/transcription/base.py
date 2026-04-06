"""Base transcriber abstractions for LattifAI."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.data import Caption
from lattifai.logging import get_logger

if TYPE_CHECKING:
    from lattifai_core.event import LEDOutput

logger = logging.getLogger(__name__)


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
    needs_vad: bool = False
    """Whether this transcriber needs VAD segmentation via event_detector."""

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
        self.event_detector = None  # Injected by mixin for VAD segmentation

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the transcriber.

        Returns:
            str: Identifier for the transcriber (e.g., 'gemini', 'parakeet').
        """

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
    def transcribe_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        language: Optional[str] = None,
    ) -> Union[Supervision, List[Supervision]]:
        """
        Transcribe audio from a numpy array and return Supervision.

        Args:
            audio_array: Audio data as numpy array (shape: [samples]).
            language: Optional language code for transcription.

        Returns:
            Supervision object with transcription info.
        """

    @abstractmethod
    def write(self, transcript: Union[str, Caption], output_file: Path, encoding: str = "utf-8") -> Path:
        """
        Persist transcript text to disk and return the file path.
        """

    def _vad_segment(
        self, audio: AudioData, vad_chunk_size: float = 30.0, vad_max_gap: float = 4.0
    ) -> Tuple[List[Tuple[float, float]], Optional["LEDOutput"]]:
        """Run VAD via event_detector to split audio into speech segments.

        Args:
            audio: AudioData to analyze.
            vad_chunk_size: Maximum chunk size in seconds for VAD processing.
            vad_max_gap: Maximum gap in seconds to merge adjacent segments.

        Returns:
            Tuple of (segments, led_output). segments is a list of (start, end) tuples.
            Both are empty/None if event_detector is unavailable or VAD fails.
        """
        if self.event_detector is None:
            return [], None

        led = self.event_detector.detect(audio, fast_mode=True, vad_chunk_size=vad_chunk_size, vad_max_gap=vad_max_gap)
        segments = [(ev.start_time, ev.end_time) for ev in led.audio_events.get_tier_by_name("VAD")]
        logger.info("VAD detected %d speech segments (total %.1fs audio)", len(segments), audio.duration)
        return segments, led

    @staticmethod
    def _slice_audio_by_segments(audio: AudioData, segments: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Slice AudioData into numpy chunks based on VAD segments."""
        return [audio.ndarray[0, int(s * audio.sampling_rate) : int(e * audio.sampling_rate)] for s, e in segments]

    @staticmethod
    def _is_url(value: str) -> bool:
        """Best-effort detection of web URLs."""
        return value.startswith(("http://", "https://"))
