"""Transcription service configuration for LattifAI."""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from ..utils import _select_device

if TYPE_CHECKING:
    from ..client import SyncAPIClient

SUPPORTED_TRANSCRIPTION_MODELS = Literal[
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # "gemini-3-pro-preview",  # Deprecated, auto-switched to gemini-3.1-pro-preview
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/canary-1b-v2",
    "iic/SenseVoiceSmall",
    "FunAudioLLM/Fun-ASR-Nano-2512",
    "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-E4B-it",
    "mistralai/Voxtral-Mini-4B-2602",
    # Any model served via vLLM/SGLang with api_base_url is also supported
    # (Whisper, Qwen3-ASR, GLM-ASR, VibeVoice, Voxtral, etc.)
]


@dataclass
class TranscriptionConfig:
    """
    Transcription service configuration.

    Settings for audio/video transcription using various providers.
    """

    _toml_section = "transcription"

    model_name: Optional[str] = None
    """Model name for transcription. None = resolve from config.toml, fallback to nvidia/parakeet-tdt-0.6b-v3.
    See SUPPORTED_TRANSCRIPTION_MODELS for built-in models.
    Any model name is accepted when api_base_url is set (vLLM/SGLang)."""

    model_hub: Literal["huggingface", "modelscope"] = "huggingface"
    """Which model hub to use when resolving lattice models for transcription."""

    gemini_api_key: Optional[str] = None
    """Gemini API key. If None, reads from GEMINI_API_KEY environment variable."""

    http_timeout_ms: Optional[int] = None
    """HTTP request timeout (milliseconds) for cloud transcription providers
    (Gemini, vLLM endpoints, etc.). When None, transcribers auto-scale the
    timeout by audio duration using a 1-hour-audio → 10-minute-timeout ratio
    (audio_sec / 6 * 1000, floored at 30s, capped at 30min). Set an explicit
    integer to override — e.g. tests pin this to a small value like 100
    so timeout behavior can be exercised without hitting the floor."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device for transcription models."""

    max_retries: int = 2
    """Maximum number of retry attempts for failed transcription requests."""

    force_overwrite: bool = False
    """Force overwrite existing transcription files."""

    verbose: bool = False
    """Enable debug logging for transcription operations."""

    language: Optional[str] = None
    """Target language code for transcription (e.g., 'en', 'zh', 'ja')."""

    system_prompt: Optional[str] = None
    """System prompt for chat mode (vLLM/SGLang only).
    Overrides the default ASR system prompt for general-purpose LLMs.
    If the value is an existing file path, the file contents will be used.
    Otherwise, the value is used directly as the system prompt text.
    Set to empty string "" to disable system prompt entirely."""

    prompt: Optional[str] = None
    """Custom prompt text or path to prompt file for transcription.
    If the value is an existing file path, the file contents will be used.
    Otherwise, the value is used directly as the prompt text."""

    description: Optional[str] = None
    """Media description from platforms like YouTube, Xiaoyuzhou (小宇宙), etc.
    Used to provide context for transcription."""

    thinking: bool = True
    """Enable Gemini's thinking mode (Gemini models only). Set to False to disable thinking."""

    include_thoughts: bool = False
    """Include Gemini's thinking process in the output (Gemini models only). Requires thinking=True."""

    temperature: Optional[float] = None
    """Sampling temperature for generation. Higher values increase randomness."""

    top_k: Optional[float] = None
    """Top-k sampling parameter. Limits token selection to top k candidates."""

    top_p: Optional[float] = None
    """Nucleus sampling parameter. Limits token selection by cumulative probability."""

    lattice_model_path: Optional[str] = None
    """Path to local LattifAI model. Will be auto-set in LattifAI client."""

    api_base_url: Optional[str] = None
    """Base URL for OpenAI-compatible API server (e.g. http://localhost:8000/v1).
    When set, routes to VLLMTranscriber which uses the /v1/audio/transcriptions endpoint.
    Works with any ASR model served via vLLM/SGLang (Whisper, Qwen3-ASR, GLM-ASR, etc.)."""

    api_mode: Literal["transcriptions", "chat", "realtime"] = "transcriptions"
    """API mode for vLLM/SGLang.
    'transcriptions' (default) uses /v1/audio/transcriptions (multipart upload, best for dedicated ASR models).
    'chat' uses /v1/chat/completions with audio_url (base64, required for general-purpose LLMs like Gemma-3n).
    'realtime' uses /v1/realtime WebSocket endpoint (for Voxtral Realtime models)."""

    audio_content_type: Literal["audio_url", "input_audio", "audio"] = "audio_url"
    """Audio content type in chat mode messages.
    'audio_url' (default) sends {"type": "audio_url", "audio_url": {"url": "data:..."}} (vLLM format).
    'input_audio' sends {"type": "input_audio", "input_audio": {"data": "...", "format": "..."}} (mlx-vlm format).
    'audio' sends {"type": "audio", "audio": "data:..."} (Google Gemma4 native format)."""

    chat_audio_first: bool = False
    """Content ordering in chat mode messages.
    False (default): text before audio [text, audio] (vLLM convention).
    True: audio before text [audio, text] (Google Gemma4 convention)."""

    verbose: bool = False
    """Print debug info (messages structure, temperature, max_tokens) for chat mode requests."""

    max_tokens: Optional[int] = None
    """Maximum output tokens for chat/realtime API modes (vLLM/SGLang only).
    If None, defaults to 4096 for chat mode. Increase for long audio transcription."""

    batch_size: int = 1
    """Number of concurrent requests for VAD chunk transcription (vLLM/SGLang only).
    Set >1 to parallelize HTTP requests to the server. Requires server-side concurrency support.
    Note: gemma-3n does NOT support concurrent audio requests in vLLM."""

    vad_chunk_size: Optional[float] = None
    """Maximum audio chunk size in seconds for VAD segmentation (vLLM/SGLang only).
    If None, auto-estimated from the model's max_model_len and tokens_per_second."""

    mlx_quantization: Literal["4bit", "8bit"] = "8bit"
    """Quantization level for MLX models (mlx-community).
    Only applies when auto-mapping original model IDs (e.g. Qwen/Qwen3-ASR-0.6B).
    Ignored when mlx-community model ID is specified directly."""

    client_wrapper: Optional["SyncAPIClient"] = field(default=None, repr=False)
    """Reference to the SyncAPIClient instance. Auto-set during client initialization."""

    # Deprecated model -> replacement mapping
    # https://ai.google.dev/gemini-api/docs/deprecations
    _DEPRECATED_MODELS = {
        "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    }

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""

        # Resolve model: None -> config.toml -> built-in default
        if self.model_name is None:
            from lattifai.config.llm import resolve_toml_value

            saved = resolve_toml_value("transcription", "model_name")
            self.model_name = saved or "nvidia/parakeet-tdt-0.6b-v3"

        # Auto-switch deprecated models
        if self.model_name in self._DEPRECATED_MODELS:
            replacement = self._DEPRECATED_MODELS[self.model_name]
            import logging

            logging.getLogger(__name__).warning(
                f"Model '{self.model_name}' is deprecated, auto-switching to '{replacement}'"
            )
            self.model_name = replacement

        # When api_base_url is set, any model name is valid (forwarded to vLLM/SGLang server)
        # mlx-community/* model IDs are also allowed (for MLX in-process inference)
        if (
            self.model_name not in SUPPORTED_TRANSCRIPTION_MODELS.__args__
            and not self.api_base_url
            and not self.model_name.startswith("mlx-community/")
        ):
            raise ValueError(
                f"Unsupported model_name: '{self.model_name}'. "
                f"Supported models are: {SUPPORTED_TRANSCRIPTION_MODELS.__args__}. "
                f"For vLLM/SGLang-served models, set api_base_url to enable any model. "
                f"For MLX models, use mlx-community/* model IDs."
            )

        # Load environment variables from .env file
        from dotenv import find_dotenv, load_dotenv

        # Try to find and load .env file from current directory or parent directories
        load_dotenv(find_dotenv(usecwd=True))

        # Auto-load Gemini API key: env var > config.toml
        if self.gemini_api_key is None:
            env_val = os.environ.get("GEMINI_API_KEY")
            if not env_val:
                try:
                    from lattifai.cli.config import get_config_value

                    env_val = get_config_value("GEMINI_API_KEY")
                except ImportError:
                    pass
            self.gemini_api_key = env_val

        # Validate max_retries
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        # Validate device
        if self.device not in ("cpu", "cuda", "mps", "auto") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got '{self.device}'")

        if self.device == "auto":
            self.device = _select_device(self.device)
