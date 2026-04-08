"""MLX transcription via mlx-audio and mlx-vlm on Apple Silicon.

Supports two backends:
- 'mlx-audio': Qwen3-ASR, Voxtral (dedicated ASR models)
- 'mlx-vlm': Gemma-4 (multimodal LLMs with audio support)

Models are auto-mapped from original HuggingFace IDs to mlx-community
quantized versions. Users can also specify mlx-community IDs directly.
"""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.data import Caption
from lattifai.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)

# Model name -> (mlx-community base ID, backend)
# All mapped IDs get quantization suffix appended: base + "-" + mlx_quantization
_MLX_MODEL_MAP = {
    "Qwen/Qwen3-ASR-0.6B": ("mlx-community/Qwen3-ASR-0.6B", "mlx-audio"),
    "Qwen/Qwen3-ASR-1.7B": ("mlx-community/Qwen3-ASR-1.7B", "mlx-audio"),
    "mistralai/Voxtral-Mini-4B-2602": ("mlx-community/Voxtral-Mini-4B-2602", "mlx-audio"),
    "google/gemma-4-E2B-it": ("mlx-community/gemma-4-E2B-it", "mlx-vlm"),
    "google/gemma-4-E4B-it": ("mlx-community/gemma-4-E4B-it", "mlx-vlm"),
}

# Keywords in model name that indicate mlx-vlm backend (for mlx-community/* passthrough)
_MLX_VLM_KEYWORDS = ("gemma",)


def _is_mlx_model(model_name: str, device: str) -> bool:
    """Check if a model should be routed to MLXTranscriber.

    Args:
        model_name: Model identifier (e.g. "Qwen/Qwen3-ASR-0.6B" or "mlx-community/...").
        device: Resolved device string ("mps", "cuda", "cpu").

    Returns:
        True if this model should use MLXTranscriber.
    """
    # Explicit mlx-community models always use MLX
    if model_name.startswith("mlx-community/"):
        return True
    # Mapped models only on Apple Silicon (mps)
    if model_name in _MLX_MODEL_MAP and device == "mps":
        return True
    return False


class MLXTranscriber(BaseTranscriber):
    """
    MLX transcription on Apple Silicon via mlx-audio and mlx-vlm.

    Supports in-process inference without a separate model server.
    Auto-maps original HuggingFace model IDs to mlx-community quantized versions.
    """

    file_suffix = ".ass"
    supports_url = False
    needs_vad = True

    # MLX backends (mlx-audio, mlx-vlm) have practical issues with long audio
    # (OOM, silent truncation, degraded accuracy) even when the underlying model
    # theoretically supports longer contexts. Cap all models to 30s chunks;
    # VAD segmentation handles splitting transparently.
    _MLX_MAX_AUDIO_SECONDS = 30.0

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._model = None
        self._processor = None  # mlx-vlm only
        self._backend = self._resolve_backend()
        self._mlx_model_id = self._resolve_mlx_model_id()

    @property
    def name(self) -> str:
        return self.config.model_name

    def _resolve_backend(self) -> Literal["mlx-audio", "mlx-vlm"]:
        """Determine which MLX backend to use based on model name."""
        model_name = self.config.model_name

        # Check mapping table first
        if model_name in _MLX_MODEL_MAP:
            return _MLX_MODEL_MAP[model_name][1]

        # For mlx-community/* passthrough, detect from keywords
        model_lower = model_name.lower()
        if any(kw in model_lower for kw in _MLX_VLM_KEYWORDS):
            return "mlx-vlm"

        return "mlx-audio"

    def _resolve_mlx_model_id(self) -> str:
        """Resolve the actual mlx-community model ID to load.

        For mapped models: appends quantization suffix (e.g. "-8bit").
        For mlx-community/* models: passes through unchanged.
        """
        model_name = self.config.model_name

        # Passthrough: already an mlx-community ID
        if model_name.startswith("mlx-community/"):
            return model_name

        # Auto-map: look up base ID and append quantization suffix
        if model_name in _MLX_MODEL_MAP:
            base_id = _MLX_MODEL_MAP[model_name][0]
            quant = self.config.mlx_quantization
            return f"{base_id}-{quant}"

        # Fallback: use as-is (will likely fail at load time)
        return model_name

    def _get_max_audio_seconds(self) -> float:
        """Override: cap all MLX models to 30s chunks for reliability."""
        return self._MLX_MAX_AUDIO_SECONDS

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------
    def _ensure_model(self):
        """Lazy-load the MLX model. Called before first inference."""
        if self._model is not None:
            return

        if self._backend == "mlx-audio":
            try:
                from mlx_audio.stt import load_model
            except ImportError:
                raise ImportError(
                    "mlx-audio is required for MLX ASR transcription. " "Install with: pip install lattifai[mlx]"
                )
            logger.info("Loading mlx-audio model: %s", self._mlx_model_id)
            self._model = load_model(self._mlx_model_id)
        else:
            try:
                from mlx_vlm import load
            except ImportError:
                raise ImportError(
                    "mlx-vlm is required for MLX multimodal transcription. " "Install with: pip install lattifai[mlx]"
                )
            logger.info("Loading mlx-vlm model: %s", self._mlx_model_id)
            self._model, self._processor = load(self._mlx_model_id)

        logger.info("MLX model loaded: %s (backend=%s)", self._mlx_model_id, self._backend)

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------
    def _transcribe_chunks(
        self,
        chunks: List[np.ndarray],
        segments: List[Tuple[float, float]],
        language: Optional[str] = None,
    ) -> List[Supervision]:
        """Transcribe a list of audio chunks with time offsets."""
        from tqdm import tqdm

        all_supervisions = []
        pbar = tqdm(
            zip(segments, chunks),
            total=len(segments),
            desc="Transcribing (MLX)",
            unit="seg",
        )

        for (start, end), chunk in pbar:
            pbar.set_postfix_str(f"{start:.1f}s-{end:.1f}s")
            sups = self._transcribe_single_chunk(chunk, start, end, language)
            all_supervisions.extend(sups)

        pbar.close()
        return all_supervisions

    def _transcribe_single_chunk(
        self,
        audio: np.ndarray,
        offset: float,
        end: float,
        language: Optional[str] = None,
    ) -> List[Supervision]:
        """Transcribe a single audio chunk. Dispatches to the correct backend."""
        if self._backend == "mlx-audio":
            return self._transcribe_mlx_audio(audio, offset, language)
        return self._transcribe_mlx_vlm(audio, offset, end - offset, language)

    def _transcribe_mlx_audio(
        self,
        audio: np.ndarray,
        offset: float,
        language: Optional[str] = None,
    ) -> List[Supervision]:
        """Transcribe via mlx-audio (Qwen3-ASR, Voxtral).

        mlx-audio returns STTOutput with segment-level timestamps.
        """
        result = self._model.generate(
            audio,
            language=language,
            temperature=self.config.temperature or 0.0,
            verbose=False,
        )

        if not result.segments:
            return [
                Supervision(
                    text=result.text.strip(),
                    start=offset,
                    duration=len(audio) / 16000,
                    language=result.language if isinstance(result.language, str) else language,
                )
            ]

        return [
            Supervision(
                text=seg["text"].strip(),
                start=seg["start"] + offset,
                duration=seg["end"] - seg["start"],
                language=seg.get("language") or (result.language if isinstance(result.language, str) else language),
            )
            for seg in result.segments
            if seg.get("text", "").strip()
        ]

    def _transcribe_mlx_vlm(
        self,
        audio: np.ndarray,
        offset: float,
        duration: float,
        language: Optional[str] = None,
    ) -> List[Supervision]:
        """Transcribe via mlx-vlm (Gemma-4).

        mlx-vlm returns plain text, no segment timestamps.
        """
        from mlx_vlm import generate

        prompt = self._build_asr_prompt(language)
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        audio = audio.astype(np.float32)

        result = generate(
            self._model,
            self._processor,
            prompt=prompt,
            audio=[audio],
            max_tokens=self.config.max_tokens or 512,
            temperature=self.config.temperature or 0.0,
        )

        text = result.text if hasattr(result, "text") else str(result)
        return [
            Supervision(
                text=text.strip(),
                start=offset,
                duration=duration,
                language=language,
            )
        ]

    def _build_asr_prompt(self, language: Optional[str] = None) -> str:
        """Build ASR prompt for mlx-vlm models (Gemma-4)."""
        if self.config.prompt:
            return self.config.prompt

        from lattifai.languages import get_language_name

        lang_name = get_language_name(language) if language else None
        if lang_name:
            return (
                f"Transcribe the following speech segment in {lang_name} into {lang_name} text.\n\n"
                "Follow these specific instructions for formatting the answer:\n"
                "* Only output the transcription, with no newlines.\n"
                "* When transcribing numbers, write the digits, "
                "i.e. write 1.7 and not one point seven, and write 3 instead of three."
            )
        return (
            "Transcribe the following speech segment in its original language.\n\n"
            "Follow these specific instructions for formatting the answer:\n"
            "* Only output the transcription, with no newlines.\n"
            "* When transcribing numbers, write the digits, "
            "i.e. write 1.7 and not one point seven, and write 3 instead of three."
        )

    # ------------------------------------------------------------------
    # Public API (BaseTranscriber abstract methods)
    # ------------------------------------------------------------------
    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL transcription. "
            "Please download the file first and use transcribe_file()."
        )

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        self._ensure_model()

        if isinstance(media_file, AudioData):
            return self._transcribe_audio_data(media_file, language=language)

        from lattifai.audio2 import load_audio

        audio = load_audio(str(media_file), target_sr=16000)
        return self._transcribe_audio_data(audio, language=language)

    def _transcribe_audio_data(self, audio: AudioData, language: Optional[str] = None) -> Caption:
        """Transcribe AudioData with VAD segmentation."""
        max_secs = self._get_max_audio_seconds() or 120.0
        segments, led = self._vad_segment(audio, vad_chunk_size=max_secs)

        if segments:
            segments = self._split_long_segments(segments, max_secs)
            chunks = self._slice_audio_by_segments(audio, segments)
            supervisions = self._transcribe_chunks(chunks, segments, language)
        else:
            audio_np = audio.ndarray[0] if audio.ndarray.ndim == 2 else audio.ndarray
            supervisions = self._transcribe_single_chunk(audio_np, 0.0, audio.duration, language)

        caption = Caption(supervisions=supervisions, language=language)
        if led is not None:
            caption.event = led
        return caption

    def transcribe_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        language: Optional[str] = None,
    ) -> Union[Supervision, List[Supervision]]:
        self._ensure_model()

        if isinstance(audio, list):
            results = []
            for a in audio:
                sups = self._transcribe_single_chunk(a, 0.0, len(a) / 16000, language)
                results.append(sups[0] if sups else Supervision(text="", speaker=None))
            return results

        sups = self._transcribe_single_chunk(audio, 0.0, len(audio) / 16000, language)
        return sups[0] if sups else Supervision(text="", speaker=None)

    def write(
        self, transcript: Union[str, Caption], output_file: Path, encoding: str = "utf-8", cache_event: bool = False
    ) -> Path:
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
