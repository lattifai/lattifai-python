"""Subtitle I/O configuration for LattifAI."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from lhotse.utils import Pathlike

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

    include_speaker_in_text: bool = True
    """Preserve speaker labels in subtitle text content."""

    normalize_text: bool = False
    """Clean HTML entities and normalize whitespace in subtitle text."""

    split_sentence: bool = False
    """Re-segment subtitles intelligently based on punctuation and semantics."""

    word_level: bool = False
    """Include word-level timestamps in alignment results (useful for karaoke, dubbing)."""

    encoding: str = "utf-8"
    """Character encoding for reading/writing subtitle files (default: utf-8)."""

    use_transcription: bool = False
    """Use transcription service (e.g., Gemini) instead of downloading YouTube subtitles."""

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

    def set_input_path(self, path: Pathlike) -> Path:
        """
        Set input subtitle path and validate it.

        Args:
            path: Path to input subtitle file (str or Path)

        Returns:
            Resolved path as Path object

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the path is not a file
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input subtitle file does not exist: '{resolved}'")
        if not resolved.is_file():
            raise ValueError(f"Input subtitle path is not a file: '{resolved}'")
        self.input_path = str(resolved)
        self.check_input_sanity()
        return resolved

    def set_output_path(self, path: Pathlike) -> Path:
        """
        Set output subtitle path and create parent directories if needed.

        Args:
            path: Path to output subtitle file (str or Path)

        Returns:
            Resolved path as Path object
        """
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = str(resolved)
        return resolved

    def check_input_sanity(self) -> None:
        """
        Validate that input_path is properly configured and accessible.

        Raises:
            ValueError: If input_path is not set or is invalid
            FileNotFoundError: If input_path does not exist
        """
        if not self.input_path:
            raise ValueError("input_path is required but not set in SubtitleConfig")

        input_file = Path(self.input_path).expanduser()
        if not input_file.exists():
            raise FileNotFoundError(
                f"Input subtitle file does not exist: '{input_file}'. " "Please check the path and try again."
            )
        if not input_file.is_file():
            raise ValueError(
                f"Input subtitle path is not a file: '{input_file}'. " "Expected a valid subtitle file path."
            )

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
