"""vLLM/SGLang transcription via OpenAI-compatible /v1/audio/transcriptions API.

Works with any ASR model served by vLLM or SGLang, including:
- Whisper (openai/whisper-large-v3-turbo, etc.)
- Qwen3-ASR (Qwen/Qwen3-ASR-0.6B, Qwen/Qwen3-ASR-1.7B)
- GLM-ASR (GLM-ASR-Nano-2512)
- Fun-ASR (Fun-ASR-Nano-2512)
- VibeVoice, Voxtral, etc.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.data import Caption
from lattifai.transcription.base import BaseTranscriber


class VLLMTranscriber(BaseTranscriber):
    """
    Transcription via vLLM/SGLang OpenAI-compatible API.

    Uses the standardized /v1/audio/transcriptions endpoint which works
    uniformly across all vLLM-supported ASR models.
    """

    file_suffix = ".txt"
    supports_url = False

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._api_base_url = transcription_config.api_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return self.config.model_name

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------
    def _transcribe_audio_file(self, file_path: Path) -> str:
        """Send audio file to /v1/audio/transcriptions and return text."""
        import httpx

        url = f"{self._api_base_url}/audio/transcriptions"

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "audio/wav")}
            data = {"model": self.config.model_name}
            if self.config.language:
                data["language"] = self.config.language
            if self.config.temperature is not None:
                data["temperature"] = str(self.config.temperature)
            if self.config.prompt:
                data["prompt"] = self.config.prompt

            resp = httpx.post(url, files=files, data=data, timeout=300.0)
            resp.raise_for_status()

        result = resp.json()
        return result.get("text", "")

    # ------------------------------------------------------------------
    # Transcription methods
    # ------------------------------------------------------------------
    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        raise NotImplementedError("VLLMTranscriber does not support URL transcription. Download the media first.")

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        """Transcribe a local audio file via the vLLM transcriptions API.

        Args:
            media_file: Path to audio file or AudioData object.
            language: Optional language hint.

        Returns:
            Caption containing the transcription.
        """
        if isinstance(media_file, AudioData):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, media_file.waveform, media_file.sample_rate)
                file_path = Path(f.name)
        else:
            file_path = Path(media_file)

        text = self._transcribe_audio_file(file_path)

        supervision = Supervision(start=0.0, end=0.0, text=text, speaker=None)
        return Caption(supervisions=[supervision])

    def transcribe_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        language: Optional[str] = None,
    ) -> Union[Supervision, List[Supervision]]:
        """Transcribe numpy audio array(s) via vLLM API."""
        if isinstance(audio, list):
            return [self._transcribe_single_numpy(a, language) for a in audio]
        return self._transcribe_single_numpy(audio, language)

    def _transcribe_single_numpy(self, audio: np.ndarray, language: Optional[str] = None) -> Supervision:
        """Transcribe a single numpy array."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            file_path = Path(f.name)

        text = self._transcribe_audio_file(file_path)
        return Supervision(start=0.0, end=0.0, text=text, speaker=None)

    def write(self, transcript: Union[str, Caption], output_file: Path, encoding: str = "utf-8") -> Path:
        """Write transcription to a text file."""
        output_file = Path(output_file)
        if isinstance(transcript, Caption):
            text = "\n".join(s.text for s in transcript.supervisions if s.text)
        else:
            text = transcript
        output_file.write_text(text, encoding=encoding)
        return output_file
