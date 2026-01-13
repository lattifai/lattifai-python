"""Caption I/O configuration for LattifAI."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, get_args

from lhotse.utils import Pathlike

# =============================================================================
# Format Type Definitions (Single Source of Truth)
# =============================================================================

# Type alias for input caption formats (all formats with registered readers)
InputCaptionFormat = Literal[
    # Standard subtitle formats
    "srt",
    "vtt",
    "ass",
    "ssa",
    "sub",
    "sbv",
    "txt",
    "sami",
    "smi",
    # Tabular formats
    "csv",
    "tsv",
    "aud",
    "json",
    # Specialized formats
    "textgrid",  # Praat TextGrid
    "youtube_vtt",  # YouTube VTT with word-level timestamps
    "gemini",  # Gemini/YouTube transcript format
    # Professional NLE formats
    "avid_ds",
    "fcpxml",
    "premiere_xml",
    "audition_csv",
    # Special
    "auto",  # Auto-detect format
]

# Type alias for output caption formats (all formats with registered writers)
OutputCaptionFormat = Literal[
    # Standard subtitle formats
    "srt",
    "vtt",
    "ass",
    "ssa",
    "sub",
    "sbv",
    "txt",
    "sami",
    "smi",
    # Tabular formats
    "csv",
    "tsv",
    "aud",
    "json",
    # Specialized formats
    "textgrid",  # Praat TextGrid
    "youtube_vtt",  # YouTube VTT with word-level timestamps
    "gemini",  # Gemini/YouTube transcript format
    # TTML profiles (write-only)
    "ttml",  # Generic TTML
    "imsc1",  # IMSC1 (Netflix/streaming) TTML profile
    "ebu_tt_d",  # EBU-TT-D (European broadcast) TTML profile
    # Professional NLE formats
    "avid_ds",  # Avid Media Composer SubCap format
    "fcpxml",  # Final Cut Pro XML
    "premiere_xml",  # Adobe Premiere Pro XML (graphic clips)
    "audition_csv",  # Adobe Audition markers
    "edimarker_csv",  # Pro Tools (via EdiMarker) markers
]

# =============================================================================
# Runtime Format Lists (Derived from Type Definitions)
# =============================================================================

# Input caption formats list (derived from InputCaptionFormat)
INPUT_CAPTION_FORMATS: list[str] = list(get_args(InputCaptionFormat))

# Output caption formats list (derived from OutputCaptionFormat)
OUTPUT_CAPTION_FORMATS: list[str] = list(get_args(OutputCaptionFormat))

# Standard caption formats (formats with both reader and writer)
CAPTION_FORMATS: list[str] = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "sami", "smi"]

# All caption formats combined (for file detection, excludes "auto")
ALL_CAPTION_FORMATS: list[str] = list(set(INPUT_CAPTION_FORMATS + OUTPUT_CAPTION_FORMATS) - {"auto"})


@dataclass
class CaptionConfig:
    """
    Caption I/O configuration.

    Controls caption file reading, writing, and formatting options.
    """

    input_format: InputCaptionFormat = "auto"
    """Input caption format. Supports: 'auto' (detect),
        standard formats (srt, vtt, ass, ssa, sub, sbv, txt, sami, smi),
        tabular (csv, tsv, aud, json),
        specialized (textgrid, youtube_vtt, gemini),
        NLE (avid_ds, fcpxml, premiere_xml, audition_csv).
    """

    input_path: Optional[str] = None
    """Path to input caption file."""

    output_format: OutputCaptionFormat = "srt"
    """Output caption format. Supports: standard formats, tabular, specialized, TTML profiles (ttml, imsc1, ebu_tt_d),
    NLE (avid_ds, fcpxml, premiere_xml, audition_csv, edimarker_csv)."""

    output_path: Optional[str] = None
    """Path to output caption file."""

    include_speaker_in_text: bool = True
    """Preserve speaker labels in caption text content."""

    normalize_text: bool = True
    """Clean HTML entities and normalize whitespace in caption text."""

    split_sentence: bool = False
    """Re-segment captions intelligently based on punctuation and semantics."""

    word_level: bool = False
    """Include word-level timestamps in alignment results (useful for karaoke, dubbing)."""

    encoding: str = "utf-8"
    """Character encoding for reading/writing caption files (default: utf-8)."""

    source_lang: Optional[str] = None
    """Source language code for the caption content (e.g., 'en', 'zh', 'de')."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._normalize_paths()
        self._validate_formats()

    @property
    def need_alignment(self, trust_timestamps: bool) -> bool:
        """Determine if alignment is needed based on configuration."""
        if trust_timestamps and not self.split_sentence:
            if not self.word_level:
                return False
            if self.normalize_text:
                print(
                    "⚠️ Warning: Text normalization with 'trust_input_timestamps=True' and 'split_sentence=False'"
                    "💡 Recommended command:\n"
                    "   lai caption normalize input.srt normalized.srt\n"
                )

            return False

        return True

    def _normalize_paths(self) -> None:
        """Normalize and expand input/output paths."""
        # Expand and normalize input path if provided, but don't require it to exist yet
        # (it might be set later after downloading captions)
        if self.input_path is not None:
            self.input_path = str(Path(self.input_path).expanduser())

        if self.output_path is not None:
            self.output_path = str(Path(self.output_path).expanduser())
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_formats(self) -> None:
        """Validate input and output format fields."""
        if self.input_format not in INPUT_CAPTION_FORMATS:
            raise ValueError(f"input_format must be one of {INPUT_CAPTION_FORMATS}, got '{self.input_format}'")

        if self.output_format not in OUTPUT_CAPTION_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_CAPTION_FORMATS}, got '{self.output_format}'")

    def set_input_path(self, path: Pathlike) -> Path:
        """
        Set input caption path and validate it.

        Args:
            path: Path to input caption file (str or Path)

        Returns:
            Resolved path as Path object

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the path is not a file
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input caption file does not exist: '{resolved}'")
        if not resolved.is_file():
            raise ValueError(f"Input caption path is not a file: '{resolved}'")
        self.input_path = str(resolved)
        self.check_input_sanity()
        return resolved

    def set_output_path(self, path: Pathlike) -> Path:
        """
        Set output caption path and create parent directories if needed.

        Args:
            path: Path to output caption file (str or Path)

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
            raise ValueError("input_path is required but not set in CaptionConfig")

        input_file = Path(self.input_path).expanduser()
        if not input_file.exists():
            raise FileNotFoundError(
                f"Input caption file does not exist: '{input_file}'. " "Please check the path and try again."
            )
        if not input_file.is_file():
            raise ValueError(
                f"Input caption path is not a file: '{input_file}'. " "Expected a valid caption file path."
            )

    def check_sanity(self) -> bool:
        """Perform sanity checks on the configuration."""
        assert self.is_input_path_existed(), "Input caption path must be provided and exist."

    def is_input_path_existed(self) -> bool:
        """Check if input caption path is provided and exists."""
        if self.input_path is None:
            return False

        input_file = Path(self.input_path).expanduser()
        self.input_path = str(input_file)
        return input_file.exists() and input_file.is_file()
