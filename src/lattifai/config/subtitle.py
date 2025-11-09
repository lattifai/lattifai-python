"""Subtitle I/O configuration for LattifAI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class SubtitleConfig:
    """
    Subtitle I/O configuration.

    Controls subtitle file reading, writing, and formatting options.
    """

    input_format: Literal["auto", "srt", "vtt", "ass", "txt", "json"] = "auto"
    """Input subtitle format. 'auto' attempts automatic detection."""

    input_path: Optional[str] = None
    """Path to input subtitle file (optional)."""

    output_format: Literal["srt", "vtt", "ass", "txt", "json"] = "srt"
    """Output subtitle format."""

    output_path: Optional[str] = None
    """Path to output subtitle file (optional)."""

    normalize_text: bool = False
    """Clean HTML entities and normalize text content."""

    split_sentence: bool = False
    """Enable intelligent sentence re-splitting based on punctuation semantics."""

    word_level: bool = False
    """Include/Output word-level timestamps in alignment results."""

    include_speaker_in_text: bool = True
    """Include speaker labels in subtitle text content."""

    encoding: str = "utf-8"
    """File encoding for reading and writing subtitle files."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_path is not None:
            assert self.is_input_path_existed(), f"Input subtitle path '{self.input_path}' does not exist."

        if self.output_path is not None:
            self.output_path = str(Path(self.output_path).expanduser())
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        valid_input_formats = ["auto", "srt", "vtt", "ass", "txt", "json"]
        if self.input_format not in valid_input_formats:
            raise ValueError(f"input_format must be one of {valid_input_formats}, got '{self.input_format}'")

        valid_output_formats = ["srt", "vtt", "ass", "txt", "json"]
        if self.output_format not in valid_output_formats:
            raise ValueError(f"output_format must be one of {valid_output_formats}, got '{self.output_format}'")

    def check_sanity(self) -> None:
        """Perform sanity checks on the configuration."""
        assert self.is_input_path_existed(), "Input subtitle path must be provided and exist."

    def is_input_path_existed(self) -> bool:
        """Check if input subtitle path is provided and exists."""
        if self.input_path is None:
            return False

        input_file = Path(self.input_path).expanduser()
        self.input_path = str(input_file)
        return input_file.exists() and input_file.is_file()
