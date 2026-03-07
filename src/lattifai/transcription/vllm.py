"""vLLM/SGLang transcription via OpenAI-compatible API.

Supports three API modes:
- 'transcriptions': /v1/audio/transcriptions (multipart file upload)
- 'chat': /v1/chat/completions (base64 audio_url in messages)
- 'realtime': /v1/realtime WebSocket (for Voxtral Realtime models)

Works with any ASR model served by vLLM or SGLang, including:
- Whisper (openai/whisper-large-v3-turbo, etc.)
- Qwen3-ASR (Qwen/Qwen3-ASR-0.6B, Qwen/Qwen3-ASR-1.7B)
- GLM-ASR (GLM-ASR-Nano-2512)
- Voxtral Realtime (mistralai/Voxtral-Mini-4B-Realtime-2602)
- VibeVoice, etc.
"""

import asyncio
import base64
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.data import Caption
from lattifai.llm import OpenAIClient
from lattifai.transcription.base import BaseTranscriber

_ASR_TAG_RE = re.compile(r"^<\|([a-z]{2,})\|>")
_QWEN_ASR_RE = re.compile(r"^language\s+(\w+)<asr_text>(.*?)(?:</asr_text>)?$", re.DOTALL)


def _parse_asr_output(text: str) -> Tuple[Optional[str], str]:
    """Parse ASR output that may contain language/text tags.

    Supported formats:
    - ``<|en|>Hello world`` — Whisper-style language tag
    - ``language English<asr_text>Hello world</asr_text>`` — Qwen3-ASR style

    Returns:
        (language_code or None, cleaned_text)
    """
    # Qwen3-ASR format: language English<asr_text>...</asr_text>
    m = _QWEN_ASR_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    # Whisper-style: <|en|>...
    m = _ASR_TAG_RE.match(text)
    if m:
        return m.group(1), text[m.end() :].strip()
    return None, text.strip()


class VLLMTranscriber(BaseTranscriber):
    """
    Transcription via vLLM/SGLang OpenAI-compatible API.

    Uses the standardized /v1/audio/transcriptions endpoint which works
    uniformly across all vLLM-supported ASR models.

    When an event_detector is injected (by the mixin), uses VAD to split
    long audio into segments before sending to the API.
    """

    file_suffix = ".txt"
    supports_url = False
    needs_vad = True

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._api_base_url = transcription_config.api_base_url.rstrip("/")
        self._supports_verbose_json = True
        self._llm_client = OpenAIClient(
            api_key="not-needed",
            model=transcription_config.model_name,
            base_url=self._api_base_url,
        )

    @property
    def name(self) -> str:
        return self.config.model_name

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------
    _MIME_TYPES = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
    }

    def _transcribe_audio_file(
        self, file_path: Path, language: Optional[str] = None
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Transcribe an audio file, dispatching to the configured API mode.

        Returns:
            (supervisions, detected_language)
        """
        if self.config.api_mode == "realtime":
            text, lang = self._transcribe_via_realtime(file_path, language=language)
            info = sf.info(str(file_path))
            return [Supervision(text=text, start=0.0, duration=info.duration, language=lang, speaker=None)], lang
        elif self.config.api_mode == "chat":
            text, lang = self._transcribe_via_chat(file_path, language=language)
            info = sf.info(str(file_path))
            return [Supervision(text=text, start=0.0, duration=info.duration, language=lang, speaker=None)], lang
        return self._transcribe_via_transcriptions(file_path, language=language)

    def _transcribe_via_transcriptions(
        self, file_path: Path, language: Optional[str] = None
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Send audio file to /v1/audio/transcriptions.

        Tries verbose_json first for segment-level timestamps; falls back to
        plain JSON when the model doesn't support verbose_json (e.g. Qwen3-ASR).

        Returns:
            (list_of_supervisions, detected_language)
        """
        import httpx

        url = f"{self._api_base_url}/audio/transcriptions"
        mime_type = self._MIME_TYPES.get(file_path.suffix.lower(), "audio/wav")
        lang = language or self.config.language

        data: dict = {"model": self.config.model_name}
        if lang:
            data["language"] = lang.split("-")[0].lower()
        if self.config.temperature is not None:
            data["temperature"] = max(self.config.temperature, 0.01)
        if self.config.prompt:
            data["prompt"] = self.config.prompt

        if self._supports_verbose_json:
            data["response_format"] = "verbose_json"
            data["timestamp_granularities[]"] = "segment"
        else:
            data["response_format"] = "json"

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, mime_type)}
            resp = httpx.post(url, files=files, data=data, timeout=120.0)

        if resp.status_code == 400 and "verbose_json" in resp.text:
            # Model doesn't support verbose_json — remember and retry with json
            import colorful

            from lattifai.utils import safe_print

            safe_print(
                colorful.yellow(f"⚠️  verbose_json not supported by {self.config.model_name}, falling back to json")
            )
            self._supports_verbose_json = False
            data.pop("timestamp_granularities[]", None)
            data["response_format"] = "json"
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, mime_type)}
                resp = httpx.post(url, files=files, data=data, timeout=120.0)

        if resp.status_code != 200:
            import logging

            logging.getLogger(__name__).error("vLLM transcriptions error %d: %s", resp.status_code, resp.text)
        resp.raise_for_status()

        result = resp.json()
        detected_lang = result.get("language") or lang

        # verbose_json returns segments with timestamps
        segments = result.get("segments")
        if segments:
            supervisions = [
                Supervision(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    duration=seg["end"] - seg["start"],
                    language=detected_lang,
                    speaker=None,
                )
                for seg in segments
                if seg.get("text", "").strip()
            ]
        else:
            # No segments (json mode or empty verbose_json) — single supervision
            raw_text = result.get("text", "")
            _, cleaned = _parse_asr_output(raw_text)
            info = sf.info(str(file_path))
            supervisions = [
                Supervision(text=cleaned, start=0.0, duration=info.duration, language=detected_lang, speaker=None)
            ]

        return supervisions, detected_lang

    def _transcribe_via_realtime(self, file_path: Path, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Send audio to /v1/realtime WebSocket and return (text, language).

        Protocol (OpenAI Realtime API):
        1. Connect to ws://<host>/v1/realtime
        2. Receive session.created
        3. Send session.update with model
        4. Send audio chunks as input_audio_buffer.append (PCM16 base64)
        5. Send input_audio_buffer.commit with final=True
        6. Receive transcription.delta / transcription.done
        """
        import librosa

        # Load and convert to PCM16 @ 16kHz
        audio, _ = librosa.load(str(file_path), sr=16000, mono=True)
        pcm16 = (audio * 32767).astype(np.int16)
        audio_bytes = pcm16.tobytes()

        # Build WebSocket URL from api_base_url (http -> ws)
        ws_url = self._api_base_url.replace("https://", "wss://").replace("http://", "ws://")
        # /v1 -> /v1/realtime
        if ws_url.endswith("/v1"):
            ws_url += "/realtime"
        elif not ws_url.endswith("/realtime"):
            ws_url = ws_url.rstrip("/") + "/v1/realtime"

        async def _ws_transcribe():
            import websockets

            async with websockets.connect(ws_url) as ws:
                # 1. Wait for session.created
                resp = json.loads(await ws.recv())
                if resp["type"] != "session.created":
                    raise RuntimeError(f"Expected session.created, got: {resp['type']}")

                # 2. Send session.update with model
                await ws.send(json.dumps({"type": "session.update", "model": self.config.model_name}))

                # 3. Initial commit to signal ready
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                # 4. Send audio in 4KB chunks
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i : i + chunk_size]
                    await ws.send(
                        json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()})
                    )

                # 5. Signal done
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                # 6. Collect transcription
                text_parts = []
                while True:
                    resp = json.loads(await ws.recv())
                    if resp["type"] == "transcription.delta":
                        text_parts.append(resp["delta"])
                    elif resp["type"] == "transcription.done":
                        return resp.get("text", "".join(text_parts))
                    elif resp["type"] == "error":
                        raise RuntimeError(f"Realtime API error: {resp['error']}")

        from lattifai.llm.base import _run_async

        text = _run_async(_ws_transcribe())
        lang = language or self.config.language
        return text, lang

    def _transcribe_via_chat(self, file_path: Path, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Send audio as base64 data URI to /v1/chat/completions and return (text, language)."""
        from lattifai.llm.base import _run_async

        mime_type = self._MIME_TYPES.get(file_path.suffix.lower(), "audio/wav")
        audio_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")

        content = [{"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{audio_b64}"}}]

        if self.config.prompt:
            content.insert(0, {"type": "text", "text": self.config.prompt})

        messages = [{"role": "user", "content": content}]
        temperature = max(self.config.temperature, 0.01) if self.config.temperature is not None else None

        response = _run_async(self._llm_client.chat(messages, temperature=temperature, timeout=300.0))

        raw_text = response.choices[0].message.content
        detected_lang, cleaned = _parse_asr_output(raw_text)
        return cleaned, detected_lang or language or self.config.language

    def _transcribe_numpy_chunk(
        self, audio: np.ndarray, sr: int, language: Optional[str] = None
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Save numpy array to temp wav and transcribe. Returns (supervisions, language)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            return self._transcribe_audio_file(Path(f.name), language=language)

    # ------------------------------------------------------------------
    # Transcription methods
    # ------------------------------------------------------------------
    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        raise NotImplementedError("VLLMTranscriber does not support URL transcription. Download the media first.")

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        """Transcribe a local audio file via the vLLM transcriptions API.

        When AudioData is provided and event_detector is available, uses VAD
        to split audio into speech segments before transcription.
        """
        if isinstance(media_file, AudioData):
            return self._transcribe_audio_data(media_file, language=language)

        # File path — send directly
        file_path = Path(media_file)
        supervisions, _ = self._transcribe_audio_file(file_path, language=language)
        return Caption(supervisions=supervisions)

    def _transcribe_audio_data(self, audio: AudioData, language: Optional[str] = None) -> Caption:
        """Transcribe AudioData, using VAD segmentation if available."""
        segments, led = self._vad_segment(audio, vad_chunk_size=120.0)

        if segments:
            from tqdm import tqdm

            chunks = self._slice_audio_by_segments(audio, segments)
            all_supervisions = []
            pbar = tqdm(
                zip(segments, chunks),
                total=len(segments),
                desc="Transcribing",
                unit="seg",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
            )
            for (start, end), chunk in pbar:
                dur = len(chunk) / audio.sampling_rate
                pbar.set_postfix_str(f"{start:.1f}s-{end:.1f}s ({dur:.1f}s)")
                chunk_sups, _ = self._transcribe_numpy_chunk(chunk, audio.sampling_rate, language=language)
                for sup in chunk_sups:
                    if sup.text and sup.text.strip():
                        # Offset timestamps relative to the VAD segment start
                        sup.start += start
                        all_supervisions.append(sup)
            pbar.close()
            caption = Caption(supervisions=all_supervisions)
        else:
            # No VAD — transcribe full audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio.ndarray.T, audio.sampling_rate)
                supervisions, _ = self._transcribe_audio_file(Path(f.name), language=language)
            caption = Caption(supervisions=supervisions)

        if led is not None:
            caption.event = led
        return caption

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
        """Transcribe a single numpy array. Returns first supervision."""
        supervisions, _ = self._transcribe_numpy_chunk(audio, 16000, language=language)
        sup = supervisions[0] if supervisions else Supervision(text="", speaker=None)
        return sup

    def write(
        self, transcript: Union[str, Caption], output_file: Path, encoding: str = "utf-8", cache_event: bool = False
    ) -> Path:
        """Write transcription to file. Format is auto-detected from file extension."""
        output_file = Path(output_file)
        if isinstance(transcript, Caption):
            transcript.write(output_file, include_speaker_in_text=False)
        else:
            output_file.write_text(transcript, encoding=encoding)
        if cache_event and isinstance(transcript, Caption) and transcript.event:
            events_file = output_file.with_suffix(".LED")
            transcript.event.write(events_file)
        return output_file
