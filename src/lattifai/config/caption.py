"""Caption I/O configuration for LattifAI.

DEPRECATED: Import from lattifai.caption.config instead.
This module is kept for backwards compatibility.
"""

import warnings

warnings.warn(
    "Importing from lattifai.config.caption is deprecated. " "Use 'from lattifai.caption.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from lattifai.caption.config for backwards compatibility
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

__all__ = [
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
]
