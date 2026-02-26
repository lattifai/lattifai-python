"""Transcription module with config-driven architecture.

Inlines the core transcription logic from lattifai_core, removing the
dependency on lattifai_core.transcription. Event detection reuses
lattifai.event.LattifAIEventDetector.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_ASR_SAMPLE_RATE = 16000  # All supported ASR models expect 16kHz audio

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.data import Caption
from lattifai.transcription.base import BaseTranscriber

_nemo_logging_suppressed = False


def _suppress_nemo_logging():
    """Suppress verbose NeMo / Hydra / Lightning logging. Executed once."""
    global _nemo_logging_suppressed
    if _nemo_logging_suppressed:
        return
    _nemo_logging_suppressed = True

    os.environ["HYDRA_FULL_ERROR"] = "0"
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    for name in [
        "nemo_logger",
        "nemo",
        "nemo.collections",
        "nemo.core",
        "nemo.utils",
        "nemo.collections.asr",
        "numexpr",
        "torch.distributed",
        "pytorch_lightning",
        "apex",
        "megatron",
        "megatron.core",
        "megatron_init",
        "config",
        "export_config_manager",
        "training_telemetry_provider",
        "onelogger",
    ]:
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.CRITICAL)
        lgr.propagate = False

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for pat in [
        ".*Megatron.*",
        ".*megatron.*",
        ".*Apex.*",
        ".*apex.*",
        ".*Redirects.*",
        ".*OneLogger.*",
        ".*error_handling_strategy.*",
        ".*exporters.*",
        ".*telemetry.*",
        ".*training_data.*",
        ".*validation_data.*",
        ".*Dropout.*",
        ".*Attn.*",
    ]:
        warnings.filterwarnings("ignore", message=pat)


class LattifAITranscriber(BaseTranscriber):
    """
    LattifAI local transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    Loads NeMo ASR models or OmniSenseVoice locally — no lattifai_core.transcription dependency.
    """

    file_suffix = ".ass"
    supports_url = False

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._asr_model = None
        self._event_detector = None

    @property
    def name(self) -> str:
        return self.config.model_name

    # ------------------------------------------------------------------
    # Event detector — reuse lattifai.event.LattifAIEventDetector
    # ------------------------------------------------------------------
    @property
    def event_detector(self):
        """Lazy-init event detector from lattifai.event."""
        if self._event_detector is None:
            try:
                from lattifai.config.event import EventConfig
                from lattifai.event import LattifAIEventDetector

                event_config = EventConfig(
                    enabled=True,
                    device=self.config.device,
                    model_path=self.config.lattice_model_path or "",
                    client_wrapper=self.config.client_wrapper,
                )
                self._event_detector = LattifAIEventDetector(event_config)
            except Exception as e:
                error_msg = str(e)
                if any(
                    kw in error_msg
                    for kw in [
                        "numpy.core.multiarray",
                        "_import_array",
                        "NumPy 1.x",
                        "NumPy 2.",
                        "sed_scores_eval",
                        "csebbs",
                    ]
                ):
                    warnings.warn(
                        f"Failed to initialize EventDetector due to NumPy compatibility issue. "
                        f"Audio event detection will be disabled. "
                        f"Original error: {e}",
                        RuntimeWarning,
                    )
                    return None
                raise
        return self._event_detector

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_asr_model(self):
        """Load the ASR model based on config.model_name. Lazy imports torch/nemo/omnisense."""
        import torch

        model_name = self.config.model_name
        device = self.config.device

        if model_name in ["nvidia/parakeet-tdt-0.6b-v3", "nvidia/canary-1b-v2"]:
            _suppress_nemo_logging()
            warnings.filterwarnings("ignore")

            from nemo.collections.asr.models import ASRModel

            # MPS doesn't support float64 (used by torchmetrics internals),
            # so load on CPU first, then move to MPS.
            load_device = "cpu" if str(device) == "mps" else device
            asr_model = ASRModel.from_pretrained(
                model_name=model_name, map_location=torch.device(load_device), strict=True
            )
            if str(device) == "mps":
                asr_model = asr_model.to(torch.float32).to(device)
            return asr_model

        elif model_name == "iic/SenseVoiceSmall":
            from omnisense.models import OmniSenseVoiceSmall

            device_id = -1
            if "cuda" in str(device):
                device_id = str(device).replace("cuda", "").strip(":") or -1

            return OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=False, device_id=device_id, device=str(device))

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def _ensure_model(self):
        """Lazy initialise: suppress logging → check permission → load model."""
        if self._asr_model is None:
            _suppress_nemo_logging()
            self._asr_model = self._load_asr_model()
        return self._asr_model

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------
    def _transcribe_impl(
        self,
        audio: Union[str, List[str], np.ndarray, List[np.ndarray], AudioData],
        language: Optional[str] = None,
        batch_size: int = 4,
        return_hypotheses: bool = True,
        num_workers: int = 0,
        timestamps: Optional[bool] = None,
        progress_bar: bool = True,
    ) -> Tuple[Any, Any]:
        """Run inference and return (hypotheses_or_supervisions, led_output)."""
        import torch

        segments: List[Tuple[float, float]] = []
        led = None

        with torch.inference_mode():
            # If input is AudioData, run event detection for VAD segmentation
            if hasattr(audio, "sampling_rate") and hasattr(audio, "ndarray"):
                assert audio.ndarray.ndim == 2, "AudioData.ndarray must be 2D (channels, samples)"
                assert audio.ndarray.shape[0] == 1, "AudioData.ndarray must have 1 channel for transcription"

                # AED — use lattifai.event wrapper
                detector = self.event_detector
                if detector is not None:
                    try:
                        led = detector.detect(audio, fast_mode=True, vad_chunk_size=30.0, vad_max_gap=4.0)
                    except Exception as e:
                        logger.warning("Event detection failed, proceeding without VAD: %s", e)
                        led = None

                if led is not None:
                    segments = [
                        (event.start_time, event.end_time) for event in led.audio_events.get_tier_by_name("VAD")
                    ]
                    audio = [
                        audio.ndarray[0, int(start * audio.sampling_rate) : int(end * audio.sampling_rate)]
                        for start, end in segments
                    ]

            model_name = self.config.model_name
            asr_model = self._asr_model

            if model_name in ["nvidia/parakeet-tdt-0.6b-v3", "nvidia/canary-1b-v2"]:
                hypotheses = asr_model.transcribe(
                    audio,
                    batch_size=batch_size,
                    return_hypotheses=return_hypotheses,
                    num_workers=num_workers,
                    timestamps=timestamps,
                    **(
                        {"source_lang": language, "target_lang": language}
                        if model_name == "nvidia/canary-1b-v2"
                        else {}
                    ),
                )
                hypotheses = self._to_supervisions(audio, hypotheses, return_hypotheses)

            elif model_name == "iic/SenseVoiceSmall":
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                elif isinstance(audio, list):
                    audio = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in audio]
                hypotheses = asr_model.transcribe(
                    audio,
                    batch_size=batch_size,
                    num_workers=0,
                    language=language,
                    timestamps=timestamps,
                    progressbar=progress_bar,
                )
                hypotheses = self._to_supervisions(audio, hypotheses, return_hypotheses)
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            # Add timestamps offsets from VAD segments
            if return_hypotheses and segments:
                assert len(hypotheses) == len(
                    segments
                ), f"Number of hypotheses {len(hypotheses)} does not match number of segments {len(segments)}"
                for sup, (start, end) in zip(hypotheses, segments):
                    sup.start = start
                    sup.duration = end - start
                    if sup.alignment and "word" in sup.alignment:
                        for word in sup.alignment["word"]:
                            word.with_offset(start)

        return hypotheses, led

    def _to_supervisions(self, audio: Any, hypotheses: Any, return_hypotheses: bool = False) -> Any:
        """Convert model output to Supervision objects."""
        # Guard OmniTranscription import
        is_omnisense = False
        try:
            from omnisense.models import OmniTranscription

            is_omnisense = isinstance(hypotheses[0], OmniTranscription)
        except (ImportError, IndexError):
            pass

        if return_hypotheses:
            if isinstance(audio, (list, tuple)):
                if is_omnisense:
                    return [
                        Supervision(
                            text=hyp.text,
                            duration=_audio.shape[-1] / _ASR_SAMPLE_RATE,
                            alignment={"word": hyp.words} if hyp.words is not None else None,
                            custom={
                                "language": hyp.language,
                                "emotion": hyp.emotion,
                                "event": hyp.event,
                                "textnorm": hyp.textnorm,
                            },
                        )
                        for _audio, hyp in zip(audio, hypotheses)
                    ]
                return [
                    Supervision(text=hyp.text, duration=_audio.shape[-1] / _ASR_SAMPLE_RATE)
                    for _audio, hyp in zip(audio, hypotheses)
                ]

            hyp = hypotheses[0]
            if is_omnisense:
                return Supervision(
                    text=hyp.text,
                    duration=audio.shape[-1] / _ASR_SAMPLE_RATE,
                    alignment={"word": hyp.words} if hyp.words is not None else None,
                    custom={
                        "language": hyp.language,
                        "emotion": hyp.emotion,
                        "event": hyp.event,
                        "textnorm": hyp.textnorm,
                    },
                )
            return Supervision(text=hyp.text, duration=audio.shape[-1] / _ASR_SAMPLE_RATE)

        # text only
        if isinstance(audio, (list, tuple)):
            return [hyp.text for hyp in hypotheses]
        return hypotheses[0].text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        """URL transcription not supported for LattifAI local models."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support URL transcription. "
            "Please download the file first and use transcribe_file()."
        )

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> Caption:
        self._ensure_model()
        transcription, event = self._transcribe_impl(media_file, language=language, num_workers=2)
        return Caption.from_transcription_results(transcription=transcription, event=event)

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
        self._ensure_model()
        result = self._transcribe_impl(
            audio, language=language, return_hypotheses=True, progress_bar=False, timestamps=True
        )
        return result[0]

    def write(self, transcript: Caption, output_file: Path, encoding: str = "utf-8", cache_event: bool = True) -> Path:
        """Persist transcript text to disk and return the file path."""
        transcript.write(output_file, include_speaker_in_text=False)
        if cache_event and transcript.event:
            events_file = output_file.with_suffix(".LED")
            transcript.event.write(events_file)
        return output_file
