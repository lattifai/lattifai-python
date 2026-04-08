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
import io
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

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

    # Audio tokens per second after all downsampling stages.
    # Used to estimate max audio duration from max_model_len.
    # Sources: vllm/model_executor/models/{qwen2_audio,ultravox,voxtral,glmasr}.py
    _TOKENS_PER_SECOND = {
        "qwen": 25.0,  # Whisper conv(50fps) + 2x pooling in _get_feat_extract_output_lengths
        "ultravox": 6.25,  # Whisper conv(50fps) / stack_factor=8, confirmed in ultravox.py docstring
        "voxtral": 12.5,  # Whisper conv(50fps) / downsample_factor=4
        "glm": 12.0,  # conv2(50fps), merge_factor=4: (50-4)//4+1 ≈ 12
        "gemma": 6.25,  # USM encoder (6.25 tok/s); hard 30s limit per audio clip
    }
    _DEFAULT_TOKENS_PER_SECOND = 25.0

    # Dedicated ASR models that need no prompt — they transcribe by default.
    _ASR_MODEL_KEYWORDS = (
        "whisper",
        "qwen3-asr",
        "fun-asr",
        "glm-asr",
        "sensevoice",
        "voxtral",
        "parakeet",
        "canary",
        "gemma",
    )

    # Default system prompt for general-purpose LLMs doing ASR via chat mode.
    _DEFAULT_ASR_SYSTEM_PROMPT = (
        "You are a speech transcription assistant. "
        "Output only the exact verbatim transcription of the audio. "
        "Do not add any commentary, explanation, or summary."
    )

    def _is_dedicated_asr_model(self) -> bool:
        """Check if the model is a dedicated ASR model (no prompt needed)."""
        model_lower = self.config.model_name.lower()
        return any(k in model_lower for k in self._ASR_MODEL_KEYWORDS)

    def _resolve_system_prompt(self) -> Optional[str]:
        """Resolve system prompt: config value > default for general-purpose LLMs > None for ASR models.

        Returns None to skip system message, empty string is treated as explicit disable.
        """
        sp = self.config.system_prompt
        if sp is not None:
            # Explicit empty string = disable system prompt
            if sp == "":
                return None
            # File path support: if the value is an existing file, read its contents
            p = Path(sp)
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
            return sp
        # No explicit system_prompt — use default for general-purpose LLMs only
        if not self._is_dedicated_asr_model():
            return self._DEFAULT_ASR_SYSTEM_PROMPT
        return None

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._api_base_url = transcription_config.api_base_url.rstrip("/")
        self._supports_verbose_json = True
        self._vad_chunk_size: Optional[float] = transcription_config.vad_chunk_size
        self._llm_client = OpenAIClient(
            api_key="not-needed",
            model=transcription_config.model_name,
            base_url=self._api_base_url,
        )

        # Auto-route: general-purpose LLMs must use chat mode (transcriptions API is ASR-only)
        if not self._is_dedicated_asr_model() and self.config.api_mode == "transcriptions":
            logger.info("Auto-switching api_mode to 'chat' for general-purpose model %s", self.config.model_name)
            self.config.api_mode = "chat"

    @property
    def name(self) -> str:
        return self.config.model_name

    def _get_vad_chunk_size(self) -> float:
        """Return VAD chunk size, auto-estimating from model's max_model_len if needed."""
        if self._vad_chunk_size is not None:
            return self._vad_chunk_size

        # Check shared hard audio encoder limits (whisper 30s, gemma 30s, etc.)
        max_secs = self._get_max_audio_seconds()
        if max_secs is not None:
            self._vad_chunk_size = max_secs
            return max_secs

        # Try to query /v1/models for max_model_len
        try:
            import httpx

            resp = httpx.get(f"{self._api_base_url}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    max_model_len = data[0].get("max_model_len")
                    if max_model_len:
                        tps = self._guess_tokens_per_second()
                        # Reserve 40% for text output tokens
                        audio_budget = max_model_len * 0.6
                        estimated = audio_budget / tps
                        # Clamp to [30, 600] seconds
                        self._vad_chunk_size = max(30.0, min(estimated, 600.0))
                        logger.info(
                            "Auto-estimated vad_chunk_size=%.0fs (max_model_len=%d, tokens/s=%.1f)",
                            self._vad_chunk_size,
                            max_model_len,
                            tps,
                        )
                        return self._vad_chunk_size
        except Exception:
            pass

        # Fallback default
        self._vad_chunk_size = 120.0
        return self._vad_chunk_size

    def _guess_tokens_per_second(self) -> float:
        """Guess audio tokens/second from model name."""
        model_lower = self.config.model_name.lower()
        for key, tps in self._TOKENS_PER_SECOND.items():
            if key in model_lower:
                return tps
        return self._DEFAULT_TOKENS_PER_SECOND

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
        self, file_or_buf: Union[Path, io.BytesIO], language: Optional[str] = None
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Send audio to /v1/audio/transcriptions.

        Accepts a file path or an in-memory BytesIO buffer (for VAD chunks).
        Tries verbose_json first for segment-level timestamps; falls back to
        plain JSON when the model doesn't support verbose_json (e.g. Qwen3-ASR).

        Returns:
            (list_of_supervisions, detected_language)
        """
        import httpx

        url = f"{self._api_base_url}/audio/transcriptions"
        lang = language or self.config.language

        # Resolve file name and mime type
        if isinstance(file_or_buf, Path):
            file_name = file_or_buf.name
            mime_type = self._MIME_TYPES.get(file_or_buf.suffix.lower(), "audio/wav")
        else:
            file_name = "chunk.mp3"
            mime_type = "audio/mpeg"

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

        def _post(buf: Union[io.BytesIO, "io.BufferedReader"]) -> httpx.Response:
            files = {"file": (file_name, buf, mime_type)}
            return httpx.post(url, files=files, data=data, timeout=120.0)

        def _open_and_post() -> httpx.Response:
            if isinstance(file_or_buf, Path):
                with open(file_or_buf, "rb") as f:
                    return _post(f)
            else:
                file_or_buf.seek(0)
                return _post(file_or_buf)

        resp = _open_and_post()

        if resp.status_code == 400 and "verbose_json" in resp.text:
            from lattifai.theme import theme
            from lattifai.utils import safe_print

            safe_print(theme.warn(f"⚠️  verbose_json not supported by {self.config.model_name}, falling back to json"))
            self._supports_verbose_json = False
            data.pop("timestamp_granularities[]", None)
            data["response_format"] = "json"
            resp = _open_and_post()

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
                    language=seg.get("language") or detected_lang,
                    speaker=None,
                )
                for seg in segments
                if seg.get("text", "").strip()
            ]
            # Update detected_lang from first segment if top-level was missing
            if not detected_lang and supervisions:
                detected_lang = supervisions[0].language
        else:
            # No segments (json mode or empty verbose_json) — single supervision
            raw_text = result.get("text", "")
            parsed_lang, cleaned = _parse_asr_output(raw_text)
            detected_lang = parsed_lang or detected_lang
            if isinstance(file_or_buf, Path):
                duration = sf.info(str(file_or_buf)).duration
            else:
                file_or_buf.seek(0)
                duration = sf.info(file_or_buf).duration
            supervisions = [
                Supervision(text=cleaned, start=0.0, duration=duration, language=detected_lang, speaker=None)
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

                # 2. Send session.update with model (and optional instructions)
                session_cfg = {"type": "session.update", "model": self.config.model_name}
                if self.config.prompt:
                    session_cfg["instructions"] = self.config.prompt
                await ws.send(json.dumps(session_cfg))

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

        raw_text = _run_async(_ws_transcribe())
        detected_lang, cleaned = _parse_asr_output(raw_text)
        return cleaned, detected_lang or language or self.config.language

    def _transcribe_via_chat(self, file_path: Path, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Send audio as base64 data URI to /v1/chat/completions and return (text, language)."""
        from lattifai.llm.base import _run_async

        mime_type = self._MIME_TYPES.get(file_path.suffix.lower(), "audio/wav")
        audio_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")

        audio_data_uri = f"data:{mime_type};base64,{audio_b64}"
        if self.config.audio_content_type == "input_audio":
            # mlx-vlm format: {"type": "input_audio", "input_audio": {"data": "<path>", "format": "<ext>"}}
            # mlx-vlm load_audio expects file path or URL, not base64
            audio_fmt = file_path.suffix.lstrip(".").lower() or "wav"
            content = [{"type": "input_audio", "input_audio": {"data": str(file_path.resolve()), "format": audio_fmt}}]
        elif self.config.audio_content_type == "audio":
            # Google Gemma4 native format: {"type": "audio", "audio": "data:<mime>;base64,<b64>"}
            content = [{"type": "audio", "audio": audio_data_uri}]
        else:
            # vLLM format (default): {"type": "audio_url", "audio_url": {"url": "data:<mime>;base64,<b64>"}}
            content = [{"type": "audio_url", "audio_url": {"url": audio_data_uri}}]

        # Build user-level text prompt
        user_prompt = self.config.prompt
        if not user_prompt and not self._is_dedicated_asr_model():
            # General-purpose LLMs need explicit transcription instruction
            lang_name = {
                "zh": "Chinese",
                "en": "English",
                "ja": "Japanese",
                "ko": "Korean",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "ru": "Russian",
            }.get(
                (language or self.config.language or "")[:2],
                language or self.config.language or "the original language",
            )
            user_prompt = f"Transcribe this audio verbatim in {lang_name}."

        if user_prompt:
            text_item = {"type": "text", "text": user_prompt}
            if self.config.chat_audio_first:
                # [audio, text] — Google Gemma4 convention
                content.append(text_item)
            else:
                # [text, audio] — vLLM convention (default)
                content.insert(0, text_item)

        # Build messages with system prompt for general-purpose LLMs
        messages = []
        sys_prompt = self._resolve_system_prompt()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": content})

        # Temperature: default to 0.0 (greedy) for non-ASR models for deterministic transcription
        if self.config.temperature is not None:
            temperature = max(self.config.temperature, 0.01)
        elif not self._is_dedicated_asr_model():
            temperature = 0.01  # greedy decoding for ASR accuracy
        else:
            temperature = None
        max_tokens = self.config.max_tokens or 4096

        if self.config.verbose:
            import logging

            _log = logging.getLogger("lattifai.transcription.vllm")

            def _redact(item):
                t = item.get("type", "")
                return {"type": t, t: "<redacted>"} if t in ("audio_url", "input_audio", "audio") else item

            _redacted = [
                (
                    {"role": m["role"], "content": [_redact(i) for i in m["content"]]}
                    if isinstance(m.get("content"), list)
                    else m
                )
                for m in messages
            ]
            _log.info("messages=%s", _redacted)
            _log.info("temperature=%s, max_tokens=%s", temperature, max_tokens)

        response = _run_async(
            self._llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=300.0)
        )

        raw_text = response.choices[0].message.content
        detected_lang, cleaned = _parse_asr_output(raw_text)
        return cleaned, detected_lang or language or self.config.language

    def _transcribe_numpy_chunk(
        self, audio: np.ndarray, sr: int, language: Optional[str] = None
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Encode numpy array to MP3 and transcribe.

        For transcriptions API: uses in-memory BytesIO (no disk I/O).
        For chat/realtime API: writes temp file (chat requires file path for base64 encoding).
        MP3 reduces size ~10x vs WAV, avoiding vLLM's 25MB limit.
        """
        if self.config.api_mode == "transcriptions":
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="MP3")
            buf.seek(0)
            return self._transcribe_via_transcriptions(buf, language=language)
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
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
        supervisions, detected_lang = self._transcribe_audio_file(file_path, language=language)
        return Caption(supervisions=supervisions, language=detected_lang)

    def _transcribe_audio_data(self, audio: AudioData, language: Optional[str] = None) -> Caption:
        """Transcribe AudioData, using VAD segmentation if available."""
        segments, led = self._vad_segment(audio, vad_chunk_size=self._get_vad_chunk_size())

        if segments:
            chunks = self._slice_audio_by_segments(audio, segments)
            batch_size = getattr(self.config, "batch_size", 1) or 1

            if batch_size > 1:
                all_supervisions, detected_lang = self._transcribe_chunks_batch(
                    segments, chunks, audio.sampling_rate, language, batch_size
                )
            else:
                all_supervisions, detected_lang = self._transcribe_chunks_sequential(
                    segments, chunks, audio.sampling_rate, language
                )
            caption = Caption(supervisions=all_supervisions, language=detected_lang)
        else:
            # No VAD — transcribe full audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio.ndarray.T, audio.sampling_rate)
                supervisions, detected_lang = self._transcribe_audio_file(Path(f.name), language=language)
            caption = Caption(supervisions=supervisions, language=detected_lang)

        if led is not None:
            caption.event = led
        return caption

    def _transcribe_chunks_sequential(
        self,
        segments: list,
        chunks: list,
        sr: int,
        language: Optional[str],
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Transcribe VAD chunks one at a time."""
        from tqdm import tqdm

        all_supervisions = []
        detected_lang = None
        pbar = tqdm(
            zip(segments, chunks),
            total=len(segments),
            desc="Transcribing",
            unit="seg",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        )
        for (start, end), chunk in pbar:
            dur = len(chunk) / sr
            pbar.set_postfix_str(f"{start:.1f}s-{end:.1f}s ({dur:.1f}s)")
            chunk_sups, chunk_lang = self._transcribe_numpy_chunk(chunk, sr, language=language)
            if chunk_lang and not detected_lang:
                detected_lang = chunk_lang
            for sup in chunk_sups:
                if sup.text and sup.text.strip():
                    sup.start += start
                    all_supervisions.append(sup)
        pbar.close()
        return all_supervisions, detected_lang

    def _transcribe_chunks_batch(
        self,
        segments: list,
        chunks: list,
        sr: int,
        language: Optional[str],
        batch_size: int,
    ) -> Tuple[List[Supervision], Optional[str]]:
        """Transcribe VAD chunks concurrently using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        all_results: list = [None] * len(segments)
        detected_lang = None

        pbar = tqdm(
            total=len(segments),
            desc=f"Transcribing (batch={batch_size})",
            unit="seg",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            future_to_idx = {}
            for idx, (seg, chunk) in enumerate(zip(segments, chunks)):
                future = pool.submit(self._transcribe_numpy_chunk, chunk, sr, language)
                future_to_idx[future] = (idx, seg)

            for future in as_completed(future_to_idx):
                idx, (start, end) = future_to_idx[future]
                chunk_sups, chunk_lang = future.result()
                if chunk_lang and not detected_lang:
                    detected_lang = chunk_lang
                # Offset timestamps and filter empty
                valid_sups = []
                for sup in chunk_sups:
                    if sup.text and sup.text.strip():
                        sup.start += start
                        valid_sups.append(sup)
                all_results[idx] = valid_sups
                pbar.update(1)

        pbar.close()

        # Flatten in original segment order
        all_supervisions = []
        for sups in all_results:
            if sups:
                all_supervisions.extend(sups)
        return all_supervisions, detected_lang

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
            from lattifai.caption.config import RenderConfig

            transcript.write(output_file, render=RenderConfig(include_speaker_in_text=False))
        else:
            output_file.write_text(transcript, encoding=encoding)
        if cache_event and isinstance(transcript, Caption) and transcript.event:
            events_file = output_file.with_suffix(".LED")
            transcript.event.write(events_file)
        return output_file
