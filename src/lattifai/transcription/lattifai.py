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
from tqdm import tqdm

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
    Loads NeMo ASR models, OmniSenseVoice, FunASR, or Qwen3-ASR locally.
    """

    file_suffix = ".ass"
    supports_url = False
    needs_vad = True

    def __init__(self, transcription_config: TranscriptionConfig):
        super().__init__(config=transcription_config)
        self._asr_model = None

    @property
    def name(self) -> str:
        return self.config.model_name

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

        elif model_name in ("FunAudioLLM/Fun-ASR-Nano-2512", "FunAudioLLM/Fun-ASR-MLT-Nano-2512"):
            import sys

            import funasr
            from funasr import AutoModel

            # funasr's fun_asr_nano uses bare `from ctc import CTC` which needs the
            # package's own directory on sys.path.  Without this, funasr silently
            # swallows the ImportError and FunASRNano never gets registered.
            # See: https://github.com/FunAudioLLM/Fun-ASR/issues/84
            nano_dir = os.path.join(os.path.dirname(funasr.__file__), "models", "fun_asr_nano")
            if nano_dir not in sys.path:
                sys.path.insert(0, nano_dir)

            # Explicitly import the model module to trigger @tables.register
            import funasr.models.fun_asr_nano.model  # noqa: F401

            hub = self.config.model_hub  # "huggingface" or "modelscope"
            return AutoModel(model=model_name, trust_remote_code=True, device=str(device), hub=hub, disable_update=True)

        elif model_name in ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"):
            # Use vendored qwen_asr with transformers >=5.5 compat fixes
            from lattifai.vendor.qwen_asr import Qwen3ASRModel

            # Map LattifAI device to qwen_asr device_map.
            # MPS is not supported (MRoPE matmul dimension mismatch) — fall back to CPU.
            device_str = str(device)
            if device_str == "mps":
                logger.warning("Qwen3-ASR does not support MPS; falling back to CPU")
                device_map = "cpu"
            elif device_str == "cuda":
                device_map = "cuda:0"
            else:
                device_map = device_str

            # bfloat16 for CUDA, float32 for CPU
            dtype = torch.bfloat16 if "cuda" in device_map else torch.float32

            # Force eager attention — transformers >=5.5 sdpa returns a shape
            # incompatible with qwen_asr's custom reshape logic.
            model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=256,
                attn_implementation="eager",
            )
            logger.info("Loaded %s on %s", model_name, device_map)
            return model

        elif model_name.startswith("google/gemma-4-"):
            # Gemma-4: multimodal model with USM audio encoder (30s hard limit per clip).
            # Requires transformers>=5.5.0.
            # https://huggingface.co/google/gemma-4-E2B-it#6-audio
            from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

            device_str = str(device)

            # ASR context budget: 30s audio × 25 tok/s = 750 audio tokens + ~512 output.
            # Default 128K context pre-allocates massive KV cache; cap it for ASR.
            _ASR_MAX_CONTEXT = 2048
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = _ASR_MAX_CONTEXT
            if hasattr(config, "text_config") and hasattr(config.text_config, "max_position_embeddings"):
                config.text_config.max_position_embeddings = _ASR_MAX_CONTEXT

            # device_map="auto" requires accelerate + CUDA; on MPS/CPU load explicitly.
            if device_str in ("mps", "cpu") or device_str.startswith("cuda:"):
                if device_str.startswith("cuda"):
                    dtype = torch.bfloat16
                elif device_str == "mps":
                    dtype = torch.float16  # MPS doesn't support bfloat16
                else:
                    dtype = torch.float32
                model = AutoModelForMultimodalLM.from_pretrained(model_name, config=config, dtype=dtype).to(device_str)
            else:
                # CUDA with auto-dispatch
                model = AutoModelForMultimodalLM.from_pretrained(
                    model_name, config=config, dtype="auto", device_map="auto"
                )
            processor = AutoProcessor.from_pretrained(model_name)
            logger.info("Loaded %s on %s", model_name, model.device)
            return (model, processor)

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
            if isinstance(audio, AudioData):
                assert audio.ndarray.ndim == 2, "AudioData.ndarray must be 2D (channels, samples)"
                assert audio.ndarray.shape[0] == 1, "AudioData.ndarray must have 1 channel for transcription"

                max_secs = self._get_max_audio_seconds()
                segments, led = self._vad_segment(audio, vad_chunk_size=max_secs or 30.0)
                # Enforce hard audio encoder limits — split any segments still exceeding the cap
                if max_secs and segments:
                    segments = self._split_long_segments(segments, max_secs)
                if segments:
                    audio = self._slice_audio_by_segments(audio, segments)

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
                    verbose=progress_bar,
                )
                hypotheses = self._to_supervisions(audio, hypotheses, return_hypotheses)

            elif model_name == "iic/SenseVoiceSmall":
                # OmniSenseVoice accepts np.ndarray, str, or Path — unwrap AudioData/Tensor
                if isinstance(audio, AudioData):
                    audio = audio.ndarray
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                elif isinstance(audio, list):
                    audio = [
                        a.ndarray if isinstance(a, AudioData) else a.cpu().numpy() if isinstance(a, torch.Tensor) else a
                        for a in audio
                    ]
                hypotheses = asr_model.transcribe(
                    audio,
                    batch_size=batch_size,
                    num_workers=0,
                    language=language,
                    timestamps=timestamps,
                    progressbar=progress_bar,
                )
                hypotheses = self._to_supervisions(audio, hypotheses, return_hypotheses)

            elif model_name in ("FunAudioLLM/Fun-ASR-Nano-2512", "FunAudioLLM/Fun-ASR-MLT-Nano-2512"):
                # Fun-ASR-Nano: funasr AutoModel, accepts file paths or torch.Tensor
                device = self.config.device
                if isinstance(audio, np.ndarray):
                    audio = [audio]
                inputs = audio if isinstance(audio, list) else [audio]
                hypotheses = []
                iterable = tqdm(inputs, desc="Transcribing", unit="seg") if progress_bar else inputs
                for inp in iterable:
                    if isinstance(inp, np.ndarray):
                        dur = inp.shape[-1] / _ASR_SAMPLE_RATE
                        inp = torch.from_numpy(inp).to(device)
                    elif isinstance(inp, torch.Tensor):
                        dur = inp.shape[-1] / _ASR_SAMPLE_RATE
                        inp = inp.to(device)
                    else:
                        dur = 0.0
                    res = asr_model.generate(
                        input=[inp], cache={}, batch_size=1, language=language or "auto", itn=True, disable_pbar=True
                    )
                    text = res[0]["text"] if res else ""
                    hypotheses.append(Supervision(text=text, duration=dur))

            elif model_name in ("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"):
                # Qwen3-ASR: native batch inference via qwen_asr package
                # qwen_asr.prepare_audio accepts str or tuple(ndarray, sample_rate),
                # NOT bare ndarray. Wrap numpy arrays as (array, 16000) tuples.
                if isinstance(audio, np.ndarray):
                    audio_inputs = [audio]
                elif isinstance(audio, list):
                    audio_inputs = audio
                else:
                    audio_inputs = [audio]

                audio_inputs = [(a, _ASR_SAMPLE_RATE) if isinstance(a, np.ndarray) else a for a in audio_inputs]

                lang_list = [language] * len(audio_inputs) if language else [None] * len(audio_inputs)

                results = asr_model.transcribe(audio=audio_inputs, language=lang_list)

                hypotheses = []
                for i, r in enumerate(results):
                    inp = audio_inputs[i]
                    # inp may be (ndarray, sr) tuple or bare ndarray
                    arr = inp[0] if isinstance(inp, tuple) else inp
                    dur = arr.shape[-1] / _ASR_SAMPLE_RATE if isinstance(arr, np.ndarray) else 0.0
                    hypotheses.append(
                        Supervision(
                            text=r.text,
                            duration=dur,
                            language=getattr(r, "language", language),
                        )
                    )

            elif model_name.startswith("google/gemma-4-"):
                # Gemma-4: multimodal transcription via transformers.
                # USM encoder has a hard 30s limit — audio beyond 30s is silently dropped.
                # VAD segmentation (needs_vad=True) handles chunking automatically.
                model, processor = asr_model  # Unpack (model, processor) tuple

                if isinstance(audio, np.ndarray):
                    audio_inputs = [audio]
                elif isinstance(audio, list):
                    audio_inputs = audio
                else:
                    audio_inputs = [audio]

                hypotheses = []
                if self.config.prompt:
                    prompt_text = self.config.prompt
                else:
                    # Official Gemma-4 ASR prompt template
                    from lattifai.languages import get_language_name

                    lang_name = get_language_name(language) if language else None
                    if lang_name:
                        prompt_text = (
                            f"Transcribe the following speech segment in {lang_name} into {lang_name} text.\n\n"
                            "Follow these specific instructions for formatting the answer:\n"
                            "* Only output the transcription, with no newlines.\n"
                            "* When transcribing numbers, write the digits, "
                            "i.e. write 1.7 and not one point seven, and write 3 instead of three."
                        )
                    else:
                        prompt_text = (
                            "Transcribe the following speech segment in its original language.\n\n"
                            "Follow these specific instructions for formatting the answer:\n"
                            "* Only output the transcription, with no newlines.\n"
                            "* When transcribing numbers, write the digits, "
                            "i.e. write 1.7 and not one point seven, and write 3 instead of three."
                        )

                iterable = tqdm(audio_inputs, desc="Transcribing", unit="seg") if progress_bar else audio_inputs
                for inp in iterable:
                    # Compute duration and prepare audio for the processor.
                    # apply_chat_template accepts file paths, URLs, or raw numpy arrays.
                    if isinstance(inp, np.ndarray):
                        dur = inp.shape[-1] / _ASR_SAMPLE_RATE
                        if inp.ndim == 2:
                            inp = inp.squeeze(0)
                        inp = inp.astype(np.float32)
                    elif isinstance(inp, torch.Tensor):
                        dur = inp.shape[-1] / _ASR_SAMPLE_RATE
                        inp = inp.squeeze(0).cpu().numpy().astype(np.float32)
                    elif isinstance(inp, str):
                        dur = 0.0  # duration computed later from output
                    else:
                        dur = 0.0

                    # Audio before text per official best practice
                    # https://huggingface.co/google/gemma-4-E2B-it#6-audio
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": inp},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    ).to(model.device)
                    input_len = inputs["input_ids"].shape[-1]

                    output_ids = model.generate(**inputs, max_new_tokens=512)
                    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
                    text = processor.parse_response(response)
                    hypotheses.append(Supervision(text=str(text).strip(), duration=dur))

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

        if not return_hypotheses:
            if isinstance(audio, (list, tuple)):
                return [hyp.text for hyp in hypotheses]
            return hypotheses[0].text

        def _make_sup(arr, hyp):
            duration = arr.shape[-1] / _ASR_SAMPLE_RATE
            if is_omnisense:
                return Supervision(
                    text=hyp.text,
                    duration=duration,
                    alignment={"word": hyp.words} if hyp.words is not None else None,
                    custom={
                        "language": hyp.language,
                        "emotion": hyp.emotion,
                        "event": hyp.event,
                        "textnorm": hyp.textnorm,
                    },
                )
            return Supervision(text=hyp.text, duration=duration)

        if isinstance(audio, (list, tuple)):
            return [_make_sup(arr, hyp) for arr, hyp in zip(audio, hypotheses)]
        return _make_sup(audio, hypotheses[0])

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

    def write(self, transcript: Caption, output_file: Path, encoding: str = "utf-8", cache_event: bool = False) -> Path:
        """Persist transcript text to disk and return the file path."""
        from lattifai.caption.config import RenderConfig

        transcript.write(output_file, render=RenderConfig(include_speaker_in_text=False))
        if cache_event and transcript.event:
            events_file = output_file.with_suffix(".LED")
            transcript.event.write(events_file)
        return output_file
