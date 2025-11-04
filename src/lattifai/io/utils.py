"""
Utility constants and helper functions for subtitle I/O operations
"""

# Supported subtitle formats for reading/writing
SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "md"]

# Input subtitle formats (includes special formats like 'auto' and 'gemini')
INPUT_SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "auto", "gemini"]

# Output subtitle formats (includes special formats like 'TextGrid' and 'json')
OUTPUT_SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "TextGrid", "json"]

# All subtitle formats combined (for file detection)
ALL_SUBTITLE_FORMATS = list(set(SUBTITLE_FORMATS + ["TextGrid", "json", "gemini"]))
