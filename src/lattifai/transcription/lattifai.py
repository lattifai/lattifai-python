"""Transcription module with config-driven architecture."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from lattifai.audio2 import AudioData
from lattifai.caption import Caption, Supervision
from lattifai.config import TranscriptionConfig
from lattifai.transcription.base import BaseTranscriber


class LattifAITranscriber(BaseTranscriber):
    """
    LattifAI local transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    Note: This transcriber only supports local file transcription, not URLs.
    """

    file_suffix = ".ass"
    supports_url = False

    def __init__(self, transcription_config: TranscriptionConfig):
        """
        Initialize LattifAI transcriber.

        Args:
            transcription_config: Transcription configuration.
        """
        super().__init__(config=transcription_config)
        self._transcriber = None

    @property
    def name(self) -> str:
        return self.config.model_name

    def _ensure_transcriber(self):
        """Lazily initialize the core transcriber."""
        if self._transcriber is None:
            from lattifai_core.transcription import LattifAITranscriber as CoreLattifAITranscriber

            self._transcriber = CoreLattifAITranscriber.from_pretrained(model_config=self.config)
        return self._transcriber

    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        """URL transcription not supported for LattifAI local models."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL transcription. "
            "Please download the file first and use transcribe_file()."
        )

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        transcriber = self._ensure_transcriber()
        transcription, audio_events = transcriber.transcribe(media_file, language=language, num_workers=2)
        return Caption.from_transcription_results(transcription=transcription, audio_events=audio_events)

    def transcribe_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        language: Optional[str] = None,
    ) -> Union[Supervision, List[Supervision]]:
        """
        Transcribe audio from a numpy array (or list of arrays) and return Supervision.

        Args:
            audio: Audio data as numpy array (shape: [samples]),
                   or a list of such arrays for batch processing.
            language: Optional language code for transcription.

        Returns:
            Supervision object (or list of Supervision objects) with transcription and alignment info.
        """
        transcriber = self._ensure_transcriber()
        return transcriber.transcribe(
            audio, language=language, return_hypotheses=True, progress_bar=False, timestamps=True
        )[0]

    def write(
        self, transcript: Caption, output_file: Path, encoding: str = "utf-8", cache_audio_events: bool = True
    ) -> Path:
        """
        Persist transcript text to disk and return the file path.
        """
        transcript.write(
            output_file,
            include_speaker_in_text=False,
        )
        if cache_audio_events and transcript.audio_events:
            from tgt import write_to_file

            events_file = output_file.with_suffix(".AED")
            write_to_file(transcript.audio_events, events_file, format="long")

        return output_file
