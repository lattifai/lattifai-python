"""Caption processing module for LattifAI.

This module provides comprehensive caption/subtitle processing capabilities:
- Multi-format reading and writing (SRT, VTT, ASS, TTML, etc.)
- Professional NLE integration (Avid, Final Cut Pro, Premiere Pro, DaVinci Resolve)
- Audio workstation support (Pro Tools, Adobe Audition)
- Advanced features: timecode offset, overlap resolution, word-level timing
"""

from ..config.caption import InputCaptionFormat, OutputCaptionFormat
from .caption import Caption
from .formats.gemini import GeminiReader, GeminiSegment, GeminiWriter
from .formats.nle.audition import (
    AuditionCSVConfig,
    AuditionCSVWriter,
    EdiMarkerConfig,
    EdiMarkerWriter,
)

# Professional NLE format writers (re-exported from formats/)
from .formats.nle.avid import AvidDSConfig, AvidDSWriter, FrameRate
from .formats.nle.fcpxml import FCPXMLConfig, FCPXMLStyle, FCPXMLWriter
from .formats.nle.premiere import PremiereXMLConfig, PremiereXMLWriter
from .formats.ttml import TTMLConfig, TTMLFormat, TTMLRegion, TTMLStyle
from .parsers.text_parser import normalize_text
from .standardize import (
    CaptionStandardizer,
    CaptionValidator,
    StandardizationConfig,
    ValidationResult,
    apply_margins_to_captions,
    standardize_captions,
)
from .supervision import Supervision

# Create TTMLWriter alias for backward compatibility
TTMLWriter = TTMLFormat

# Utility functions
from .utils import (
    CollisionMode,
    TimecodeOffset,
    apply_timecode_offset,
    detect_overlaps,
    format_srt_timestamp,
    generate_srt_content,
    resolve_overlaps,
    split_long_lines,
)

__all__ = [
    # Core classes
    "Caption",
    "Supervision",
    # Standardization
    "CaptionStandardizer",
    "CaptionValidator",
    "StandardizationConfig",
    "ValidationResult",
    "standardize_captions",
    "apply_margins_to_captions",
    # Gemini format support
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    # Text utilities
    "normalize_text",
    # Format types
    "InputCaptionFormat",
    "OutputCaptionFormat",
    # Professional format writers
    "AvidDSWriter",
    "AvidDSConfig",
    "FCPXMLWriter",
    "FCPXMLConfig",
    "FCPXMLStyle",
    "PremiereXMLWriter",
    "PremiereXMLConfig",
    "AuditionCSVWriter",
    "AuditionCSVConfig",
    "EdiMarkerWriter",
    "EdiMarkerConfig",
    "TTMLWriter",
    "TTMLConfig",
    "TTMLStyle",
    "TTMLRegion",
    # Utilities
    "CollisionMode",
    "TimecodeOffset",
    "apply_timecode_offset",
    "resolve_overlaps",
    "detect_overlaps",
    "split_long_lines",
    "format_srt_timestamp",
    "generate_srt_content",
]
