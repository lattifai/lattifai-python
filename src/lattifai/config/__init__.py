"""Configuration system for LattifAI using nemo_run."""

# Re-export caption config classes from lattifai-captions package
from lattifai.caption.config import (
    ALL_CAPTION_FORMATS,
    CAPTION_FORMATS,
    INPUT_CAPTION_FORMATS,
    OUTPUT_CAPTION_FORMATS,
    CaptionConfig,
    CaptionFonts,
    CaptionStyle,
    InputCaptionFormat,
    KaraokeConfig,
    OutputCaptionFormat,
    StandardizationConfig,
)

from .alignment import AlignmentConfig
from .client import ClientConfig
from .diarization import DiarizationConfig
from .event import EventConfig
from .media import AUDIO_FORMATS, MEDIA_FORMATS, VIDEO_FORMATS, MediaConfig
from .transcription import TranscriptionConfig

__all__ = [
    "EventConfig",
    "ClientConfig",
    "AlignmentConfig",
    "CaptionConfig",
    "CaptionFonts",
    "CaptionStyle",
    "KaraokeConfig",
    "StandardizationConfig",
    "InputCaptionFormat",
    "OutputCaptionFormat",
    "INPUT_CAPTION_FORMATS",
    "OUTPUT_CAPTION_FORMATS",
    "ALL_CAPTION_FORMATS",
    "CAPTION_FORMATS",
    "TranscriptionConfig",
    "DiarizationConfig",
    "MediaConfig",
    "AUDIO_FORMATS",
    "VIDEO_FORMATS",
    "MEDIA_FORMATS",
]
