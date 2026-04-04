"""Caption I/O configuration for LattifAI SDK."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lattifai.caption.config import (
    INPUT_CAPTION_FORMATS,
    OUTPUT_CAPTION_FORMATS,
    CaptionStyle,
    InputCaptionFormat,
    KaraokeConfig,
    OutputCaptionFormat,
    StandardizationConfig,
)
from lattifai.caption.supervision import Pathlike


@dataclass
class CaptionInputConfig:
    """Caption input: source file, format, text preprocessing.

    Controls where to read captions from and how to preprocess the text
    before alignment or output.
    """

    path: Optional[str] = None
    """Path to input caption file."""

    format: InputCaptionFormat = "auto"
    """Input caption format. Supports: 'auto' (detect),
    standard (srt, vtt, ass, ssa, sub, sbv, txt, sami, smi),
    tabular (csv, tsv, aud, json), specialized (textgrid, gemini),
    NLE (avid_ds, fcpxml, premiere_xml, audition_csv)."""

    encoding: str = "utf-8"
    """Character encoding for reading caption files (default: utf-8)."""

    source_lang: Optional[str] = None
    """Source language code for the caption content (e.g., 'en', 'zh', 'de')."""

    normalize_text: bool = True
    """Clean HTML entities and normalize whitespace in caption text."""

    split_sentence: bool = False
    """Re-segment captions intelligently based on punctuation and semantics."""

    def __post_init__(self):
        """Validate input configuration."""
        self._normalize_path()
        self._validate_format()

    def _normalize_path(self) -> None:
        """Normalize and expand input path."""
        if self.path is not None:
            self.path = str(Path(self.path).expanduser().resolve())

    def _validate_format(self) -> None:
        """Validate input format."""
        if self.format not in INPUT_CAPTION_FORMATS:
            raise ValueError(f"input format must be one of {INPUT_CAPTION_FORMATS}, got '{self.format}'")

    def set_path(self, path: Pathlike) -> Path:
        """Set input caption path and validate it.

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
        self.path = str(resolved)
        return resolved

    def check_sanity(self) -> None:
        """Validate that path is properly configured and accessible.

        Raises:
            ValueError: If path is not set or is invalid
            FileNotFoundError: If path does not exist
        """
        if not self.path:
            raise ValueError("input path is required but not set")

        input_file = Path(self.path).expanduser().resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input caption file does not exist: '{input_file}'")
        if not input_file.is_file():
            raise ValueError(f"Input caption path is not a file: '{input_file}'")

    def is_path_existed(self) -> bool:
        """Check if input caption path is provided and exists."""
        if self.path is None:
            return False

        input_file = Path(self.path).expanduser().resolve()
        self.path = str(input_file)
        return input_file.exists() and input_file.is_file()


@dataclass
class CaptionOutputConfig:
    """Caption output: destination, format, content policy.

    Controls where to write captions to and what content to include.
    Visual rendering (font, colors, background) is in CaptionStyle, not here.
    """

    path: Optional[str] = None
    """Path to output caption file."""

    format: OutputCaptionFormat = "srt"
    """Output caption format. Supports: standard, tabular, specialized,
    TTML profiles (ttml, imsc1, ebu_tt_d),
    NLE (avid_ds, fcpxml, premiere_xml, audition_csv, edimarker_csv)."""

    include_speaker_in_text: bool = True
    """Preserve speaker labels in caption text content."""

    word_level: bool = False
    """Use word-level output (useful for karaoke, dubbing).
    JSON format includes 'words' field with word-level timestamps."""

    translation_first: bool = False
    """Place translation text above original text in bilingual output."""

    def __post_init__(self):
        """Validate output configuration."""
        self._normalize_path()
        self._validate_format()

    def _normalize_path(self) -> None:
        """Normalize and expand output path, creating parent dirs."""
        if self.path is not None:
            self.path = str(Path(self.path).expanduser().resolve())
            output_dir = Path(self.path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_format(self) -> None:
        """Validate output format."""
        if self.format not in OUTPUT_CAPTION_FORMATS:
            raise ValueError(f"output format must be one of {OUTPUT_CAPTION_FORMATS}, got '{self.format}'")

    def set_path(self, path: Pathlike) -> Path:
        """Set output caption path and create parent directories if needed.

        Args:
            path: Path to output caption file (str or Path)

        Returns:
            Resolved path as Path object
        """
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(resolved)
        return resolved


@dataclass
class CaptionConfig:
    """Caption pipeline configuration.

    Five clearly separated sub-configs:
    - input: source file, format, text preprocessing
    - output: destination, format, content policy
    - style: visual rendering (font, colors, background, speaker colors)
    - karaoke: karaoke behavior (effect, color scheme, format options)
    - standardization: broadcast compliance (Netflix/BBC guidelines)
    """

    _toml_section = "caption"

    input: CaptionInputConfig = field(default_factory=CaptionInputConfig)
    """Caption input: source file, format, text preprocessing."""

    output: CaptionOutputConfig = field(default_factory=CaptionOutputConfig)
    """Caption output: destination, format, content policy."""

    style: CaptionStyle = field(default_factory=CaptionStyle)
    """Visual rendering: font, colors, background, speaker colors, alignment.
    When karaoke.color_scheme is set, scheme colors override style colors."""

    karaoke: Optional[KaraokeConfig] = None
    """Karaoke behavior: effect type, color scheme, LRC/TTML options."""

    standardization: Optional[StandardizationConfig] = None
    """Broadcast compliance: Netflix/BBC guidelines for segment duration, CPS, etc."""

    # ── Convenience accessors (backward compatibility) ──

    @property
    def input_format(self) -> InputCaptionFormat:
        return self.input.format

    @property
    def input_path(self) -> Optional[str]:
        return self.input.path

    @input_path.setter
    def input_path(self, value: Optional[str]) -> None:
        self.input.path = value

    @property
    def output_format(self) -> OutputCaptionFormat:
        return self.output.format

    @property
    def output_path(self) -> Optional[str]:
        return self.output.path

    @output_path.setter
    def output_path(self, value: Optional[str]) -> None:
        self.output.path = value

    @property
    def include_speaker_in_text(self) -> bool:
        return self.output.include_speaker_in_text

    @property
    def normalize_text(self) -> bool:
        return self.input.normalize_text

    @property
    def split_sentence(self) -> bool:
        return self.input.split_sentence

    @property
    def word_level(self) -> bool:
        return self.output.word_level

    @word_level.setter
    def word_level(self, value: bool) -> None:
        self.output.word_level = value

    @property
    def translation_first(self) -> bool:
        return self.output.translation_first

    @property
    def encoding(self) -> str:
        return self.input.encoding

    @property
    def source_lang(self) -> Optional[str]:
        return self.input.source_lang

    @source_lang.setter
    def source_lang(self, value: Optional[str]) -> None:
        self.input.source_lang = value

    @property
    def speaker_color(self) -> str:
        return self.style.speaker_color

    def need_alignment(self, trust_timestamps: bool) -> bool:
        """Determine if alignment is needed based on configuration."""
        if trust_timestamps and not self.split_sentence:
            if not self.word_level:
                return False
            return False
        return True

    def set_input_path(self, path: Pathlike) -> Path:
        """Set input caption path and validate it."""
        return self.input.set_path(path)

    def set_output_path(self, path: Pathlike) -> Path:
        """Set output caption path and create parent directories if needed."""
        return self.output.set_path(path)

    def check_input_sanity(self) -> None:
        """Validate that input path is properly configured and accessible."""
        self.input.check_sanity()

    def check_sanity(self) -> None:
        """Perform sanity checks on the configuration."""
        if not self.is_input_path_existed():
            raise ValueError("Input caption path must be provided and exist.")

    def is_input_path_existed(self) -> bool:
        """Check if input caption path is provided and exists."""
        return self.input.is_path_existed()
