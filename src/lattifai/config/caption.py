"""Caption I/O configuration for LattifAI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, get_args

from lhotse.utils import Pathlike

# =============================================================================
# Caption Style Configuration Classes
# =============================================================================


class CaptionFonts:
    """Common caption font constants.

    These are reference constants for popular fonts. You can use any
    system font name as the font_name parameter in CaptionStyle.
    """

    # Western fonts
    ARIAL = "Arial"
    IMPACT = "Impact"
    VERDANA = "Verdana"
    HELVETICA = "Helvetica"

    # Chinese fonts
    NOTO_SANS_SC = "Noto Sans SC"
    MICROSOFT_YAHEI = "Microsoft YaHei"
    PINGFANG_SC = "PingFang SC"
    SIMHEI = "SimHei"

    # Japanese fonts
    NOTO_SANS_JP = "Noto Sans JP"
    MEIRYO = "Meiryo"
    HIRAGINO_SANS = "Hiragino Sans"

    # Korean fonts
    NOTO_SANS_KR = "Noto Sans KR"
    MALGUN_GOTHIC = "Malgun Gothic"


@dataclass
class CaptionStyle:
    """Caption style configuration for ASS/TTML formats.

    Attributes:
        primary_color: Main text color (#RRGGBB)
        secondary_color: Secondary/highlight color (#RRGGBB)
        outline_color: Text outline color (#RRGGBB)
        back_color: Shadow color (#RRGGBB)
        font_name: Font family name (use CaptionFonts constants or any system font)
        font_size: Font size in points
        bold: Enable bold text
        italic: Enable italic text
        outline_width: Outline thickness
        shadow_depth: Shadow distance
        alignment: ASS alignment (1-9, numpad style), 2=bottom-center
        margin_l: Left margin in pixels
        margin_r: Right margin in pixels
        margin_v: Vertical margin in pixels
    """

    # Colors (#RRGGBB format)
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#00FFFF"
    outline_color: str = "#000000"
    back_color: str = "#000000"

    # Font
    font_name: str = CaptionFonts.ARIAL
    font_size: int = 48
    bold: bool = False
    italic: bool = False

    # Border and shadow
    outline_width: float = 2.0
    shadow_depth: float = 1.0

    # Position
    alignment: int = 2
    margin_l: int = 20
    margin_r: int = 20
    margin_v: int = 20


@dataclass
class KaraokeConfig:
    """Karaoke export configuration.

    Attributes:
        enabled: Whether karaoke mode is enabled
        effect: Karaoke effect type
            - "sweep": Gradual fill from left to right (ASS \\kf tag)
            - "instant": Instant highlight (ASS \\k tag)
            - "outline": Outline then fill (ASS \\ko tag)
        style: Caption style configuration (font, colors, position)
        lrc_precision: LRC time precision ("centisecond" or "millisecond")
        lrc_metadata: LRC metadata dict (ar, ti, al, etc.)
        ttml_timing_mode: TTML timing attribute ("Word" or "Line")
    """

    enabled: bool = False
    effect: Literal["sweep", "instant", "outline"] = "sweep"
    style: CaptionStyle = field(default_factory=CaptionStyle)

    # LRC specific
    lrc_precision: Literal["centisecond", "millisecond"] = "millisecond"
    lrc_metadata: Dict[str, str] = field(default_factory=dict)

    # TTML specific
    ttml_timing_mode: Literal["Word", "Line"] = "Word"


@dataclass
class StandardizationConfig:
    """Caption standardization configuration following broadcast guidelines.

    Reference Standards:
    - Netflix Timed Text Style Guide
    - BBC Subtitle Guidelines
    - EBU-TT-D Standard

    Attributes:
        min_duration: Minimum segment duration (seconds). Netflix recommends 5/6s, BBC 0.3s
        max_duration: Maximum segment duration (seconds). Netflix/BBC recommends 7s
        min_gap: Minimum gap between segments (seconds). 80ms prevents subtitle flicker
        max_lines: Maximum lines per segment. Broadcast standard is typically 2
        max_chars_per_line: Maximum characters per line. CJK auto-adjusted by ÷2 (e.g., 42 → 21)
        optimal_cps: Optimal reading speed (chars/sec). Netflix recommends 17-20 CPS
        start_margin: Start margin (seconds) before first word. None = no adjustment (default)
        end_margin: End margin (seconds) after last word. None = no adjustment (default)
        margin_collision_mode: How to handle collisions: 'trim' (reduce margin) or 'gap' (maintain min_gap)
    """

    min_duration: float = 0.8
    max_duration: float = 7.0
    min_gap: float = 0.08
    max_lines: int = 2
    max_chars_per_line: int = 42
    optimal_cps: float = 17.0
    start_margin: Optional[float] = None
    end_margin: Optional[float] = None
    margin_collision_mode: Literal["trim", "gap"] = "trim"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if self.max_duration <= self.min_duration:
            raise ValueError("max_duration must be greater than min_duration")
        if self.min_gap < 0:
            raise ValueError("min_gap cannot be negative")
        if self.max_lines < 1:
            raise ValueError("max_lines must be at least 1")
        if self.max_chars_per_line < 10:
            raise ValueError("max_chars_per_line must be at least 10")
        if self.start_margin is not None and self.start_margin < 0:
            raise ValueError("start_margin cannot be negative")
        if self.end_margin is not None and self.end_margin < 0:
            raise ValueError("end_margin cannot be negative")
        if self.margin_collision_mode not in ("trim", "gap"):
            raise ValueError("margin_collision_mode must be 'trim' or 'gap'")


# =============================================================================
# Format Type Definitions (Single Source of Truth)
# =============================================================================

# Type alias for input caption formats (all formats with registered readers)
InputCaptionFormat = Literal[
    # Standard subtitle formats
    "srt",
    "vtt",  # WebVTT (auto-detects YouTube VTT with word-level timestamps)
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
    "vtt",  # WebVTT (use karaoke_config.enabled=True for YouTube VTT style output)
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
        specialized (textgrid, gemini),
        NLE (avid_ds, fcpxml, premiere_xml, audition_csv).
        Note: VTT format auto-detects YouTube VTT with word-level timestamps.
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

    karaoke: Optional[KaraokeConfig] = None
    """Karaoke configuration when word_level=True (e.g., ASS \\kf tags, enhanced LRC).
    When None with word_level=True, outputs word-per-segment instead of karaoke styling.
    When provided, karaoke.enabled controls whether karaoke styling is applied."""

    encoding: str = "utf-8"
    """Character encoding for reading/writing caption files (default: utf-8)."""

    source_lang: Optional[str] = None
    """Source language code for the caption content (e.g., 'en', 'zh', 'de')."""

    standardization: Optional[StandardizationConfig] = None
    """Standardization configuration for broadcast-grade captions.
    When provided, captions will be standardized according to Netflix/BBC guidelines."""

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
        """Normalize and expand input/output paths.

        Uses Path.resolve() to get absolute paths and prevent path traversal issues.
        """
        # Expand and normalize input path if provided, but don't require it to exist yet
        # (it might be set later after downloading captions)
        if self.input_path is not None:
            self.input_path = str(Path(self.input_path).expanduser().resolve())

        if self.output_path is not None:
            self.output_path = str(Path(self.output_path).expanduser().resolve())
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

        input_file = Path(self.input_path).expanduser().resolve()
        if not input_file.exists():
            raise FileNotFoundError(
                f"Input caption file does not exist: '{input_file}'. " "Please check the path and try again."
            )
        if not input_file.is_file():
            raise ValueError(
                f"Input caption path is not a file: '{input_file}'. " "Expected a valid caption file path."
            )

    def check_sanity(self) -> None:
        """Perform sanity checks on the configuration.

        Raises:
            ValueError: If input path is not provided or does not exist.
        """
        if not self.is_input_path_existed():
            raise ValueError("Input caption path must be provided and exist.")

    def is_input_path_existed(self) -> bool:
        """Check if input caption path is provided and exists."""
        if self.input_path is None:
            return False

        input_file = Path(self.input_path).expanduser().resolve()
        self.input_path = str(input_file)
        return input_file.exists() and input_file.is_file()
