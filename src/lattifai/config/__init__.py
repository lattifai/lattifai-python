"""Configuration system for LattifAI using nemo_run."""

# Re-export caption config classes from lattifai-captions package
from lattifai.caption.config import (
    ALL_CAPTION_FORMATS,
    CAPTION_FORMATS,
    INPUT_CAPTION_FORMATS,
    OUTPUT_CAPTION_FORMATS,
    ASSConfig,
    CaptionFonts,
    InputCaptionFormat,
    LRCConfig,
    OutputCaptionFormat,
    RenderConfig,
    StandardizationConfig,
)

from .alignment import AlignmentConfig

# CaptionConfig and sub-configs defined in lattifai-python (workflow config)
from .caption import CaptionConfig, CaptionInputConfig, CaptionOutputConfig
from .client import ClientConfig
from .diarization import DiarizationConfig
from .event import EventConfig
from .llm import LLMConfig
from .media import AUDIO_FORMATS, MEDIA_FORMATS, VIDEO_FORMATS, MediaConfig
from .summarization import SummarizationConfig
from .transcription import TranscriptionConfig
from .translation import TranslationConfig

__all__ = [
    "EventConfig",
    "ClientConfig",
    "AlignmentConfig",
    "CaptionConfig",
    "CaptionInputConfig",
    "CaptionOutputConfig",
    "ASSConfig",
    "CaptionFonts",
    "RenderConfig",
    "LRCConfig",
    "StandardizationConfig",
    "InputCaptionFormat",
    "OutputCaptionFormat",
    "INPUT_CAPTION_FORMATS",
    "OUTPUT_CAPTION_FORMATS",
    "ALL_CAPTION_FORMATS",
    "CAPTION_FORMATS",
    "TranscriptionConfig",
    "DiarizationConfig",
    "LLMConfig",
    "MediaConfig",
    "AUDIO_FORMATS",
    "VIDEO_FORMATS",
    "MEDIA_FORMATS",
    "SummarizationConfig",
    "TranslationConfig",
]
