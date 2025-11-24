"""Transcription module with config-driven architecture."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from lattifai.audio2 import AudioData
from lattifai.config import TranscriptionConfig
from lattifai.subtitle import Supervision
from lattifai.transcription.base import BaseTranscriber
from lattifai.transcription.prompts import get_prompt_loader  # noqa: F401


class LattifAITranscriber(BaseTranscriber):
    """
    LattifAI local transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    Note: This transcriber only supports local file transcription, not URLs.
    """

    # Transcriber metadata
    name = "LattifAI"
    file_suffix = ".ass"
    supports_url = False

    def __init__(
        self,
        transcription_config: TranscriptionConfig,
    ):
        """
        Initialize Gemini transcriber.

        Args:
            transcription_config: Transcription configuration. If None, uses default.
        """
        self.config = transcription_config
        self.logger = logging.getLogger(__name__)
        self._system_prompt: Optional[str] = None

        from lattifai_core.transcription import LattifAITranscriber as CoreLattifAITranscriber

        self._transcriber = CoreLattifAITranscriber.from_pretrained(model_config=self.config)

    async def transcribe_url(self, url: str) -> str:
        """
        URL transcription not supported for LattifAI local models.

        This method exists to satisfy the BaseTranscriber interface but
        will never be called because supports_url = False and the base
        class checks this flag before calling this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL transcription. "
            f"Please download the file first and use transcribe_file()."
        )

    async def transcribe_file(self, media_file: Union[str, Path, AudioData]) -> List[Supervision]:
        supervisions = self._transcriber.transcribe(media_file, num_workers=2)
        return supervisions

    def write(self, transcript: List[Supervision], output_file: Path, encoding: str = "utf-8") -> Path:
        """
        Persist transcript text to disk and return the file path.
        """
        from lattifai.subtitle import SubtitleIO

        SubtitleIO.write(
            transcript,
            output_path=output_file,
            include_speaker_in_text=False,
        )

    def _get_transcription_prompt(self) -> str:
        """Get (and cache) transcription system prompt from prompts module."""
        if self._system_prompt is not None:
            return self._system_prompt

        base_prompt = ""  # TODO

        self._system_prompt = base_prompt
        return self._system_prompt
