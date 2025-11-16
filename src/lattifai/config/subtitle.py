"""Subtitle I/O configuration for LattifAI."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

# Supported subtitle formats for reading/writing
SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "md"]

# Input subtitle formats (includes special formats like 'auto' and 'gemini')
INPUT_SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "auto", "gemini"]

# Output subtitle formats (includes special formats like 'TextGrid' and 'json')
OUTPUT_SUBTITLE_FORMATS = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "TextGrid", "json"]

# All subtitle formats combined (for file detection)
ALL_SUBTITLE_FORMATS = list(set(SUBTITLE_FORMATS + ["TextGrid", "json", "gemini"]))

# Type aliases for better type hints
InputSubtitleFormat = Literal["auto", "srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "gemini"]
OutputSubtitleFormat = Literal["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "TextGrid", "json"]


@dataclass
class SubtitleConfig:
    """
    Subtitle I/O configuration.

    Controls subtitle file reading, writing, and formatting options.
    """

    input_format: InputSubtitleFormat = "auto"
    """Input subtitle format: 'auto', 'srt', 'vtt', 'ass', 'txt', or 'json'."""

    input_path: Optional[str] = None
    """Path to input subtitle file."""

    output_format: OutputSubtitleFormat = "srt"
    """Output subtitle format: 'srt', 'vtt', 'ass', 'txt', or 'json'."""

    output_path: Optional[str] = None
    """Path to output subtitle file."""

    normalize_text: bool = False
    """Clean HTML entities and normalize whitespace in subtitle text."""

    split_sentence: bool = False
    """Re-segment subtitles intelligently based on punctuation and semantics."""

    word_level: bool = False
    """Include word-level timestamps in alignment results (useful for karaoke, dubbing)."""

    include_speaker_in_text: bool = True
    """Preserve speaker labels in subtitle text content."""

    encoding: str = "utf-8"
    """Character encoding for reading/writing subtitle files (default: utf-8)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Expand and normalize input path if provided, but don't require it to exist yet
        # (it might be set later after downloading subtitles)
        if self.input_path is not None:
            self.input_path = str(Path(self.input_path).expanduser())

        if self.output_path is not None:
            self.output_path = str(Path(self.output_path).expanduser())
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.input_format not in INPUT_SUBTITLE_FORMATS:
            raise ValueError(f"input_format must be one of {INPUT_SUBTITLE_FORMATS}, got '{self.input_format}'")

        if self.output_format not in OUTPUT_SUBTITLE_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_SUBTITLE_FORMATS}, got '{self.output_format}'")

    def check_sanity(self) -> bool:
        """Perform sanity checks on the configuration."""
        assert self.is_input_path_existed(), "Input subtitle path must be provided and exist."

    def is_input_path_existed(self) -> bool:
        """Check if input subtitle path is provided and exists."""
        if self.input_path is None:
            return False

        input_file = Path(self.input_path).expanduser()
        self.input_path = str(input_file)
        return input_file.exists() and input_file.is_file()
