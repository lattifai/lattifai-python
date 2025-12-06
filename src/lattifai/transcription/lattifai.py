"""Transcription module with config-driven architecture."""

from pathlib import Path
from typing import Optional, Union

from lattifai.audio2 import AudioData
from lattifai.caption import Caption
from lattifai.config import TranscriptionConfig
from lattifai.transcription.base import BaseTranscriber
from lattifai.transcription.prompts import get_prompt_loader  # noqa: F401


class LattifAITranscriber(BaseTranscriber):
    """
    LattifAI local transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    Note: This transcriber only supports local file transcription, not URLs.
    """

    # Transcriber metadata
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
        super().__init__(
            config=transcription_config,
        )

        self._system_prompt: Optional[str] = None
        self._transcriber = None

    @property
    def name(self) -> str:
        return f"{self.config.model_name}"

    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        """
        URL transcription not supported for LattifAI local models.

        This method exists to satisfy the BaseTranscriber interface but
        will never be called because supports_url = False and the base
        class checks this flag before calling this method.

        Args:
            url: URL to transcribe (not supported)
            language: Optional language code (not used)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL transcription. "
            f"Please download the file first and use transcribe_file()."
        )

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        if self._transcriber is None:
            from lattifai_core.transcription import LattifAITranscriber as CoreLattifAITranscriber

            self._transcriber = CoreLattifAITranscriber.from_pretrained(model_config=self.config)

        transcription, audio_events = self._transcriber.transcribe(media_file, language=language, num_workers=2)
        caption = Caption.from_transcription_results(
            transcription=transcription,
            audio_events=audio_events,
        )

        return caption

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

    def _get_transcription_prompt(self) -> str:
        """Get (and cache) transcription system prompt from prompts module."""
        if self._system_prompt is not None:
            return self._system_prompt

        base_prompt = ""  # TODO

        self._system_prompt = base_prompt
        return self._system_prompt
