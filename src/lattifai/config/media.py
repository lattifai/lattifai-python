"""Media I/O configuration for LattifAI."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from lhotse.utils import Pathlike

# Supported media formats for both audio and video content
AUDIO_FORMATS = (
    "aac",
    "aiff",
    "alac",
    "flac",
    "m4a",
    "mp3",
    "ogg",
    "opus",
    "wav",
    "wma",
)

VIDEO_FORMATS = (
    "3gp",
    "avi",
    "flv",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "webm",
    "wmv",
)

MEDIA_FORMATS = tuple(sorted(set(AUDIO_FORMATS + VIDEO_FORMATS)))


@dataclass
class MediaConfig:
    """Unified configuration for audio/video input and output handling."""

    # Input configuration (local filesystem path or URL)
    input_path: Optional[str] = None
    """Local file path or URL to audio/video content."""

    media_format: str = "auto"
    """Media format (mp3, wav, mp4, etc.) or 'auto' for automatic detection."""

    sample_rate: Optional[int] = None
    """Audio sample rate in Hz (e.g., 16000, 44100)."""

    channels: Optional[int] = None
    """Number of audio channels (1=mono, 2=stereo)."""

    channel_selector: Optional[str | int] = "average"
    """Audio channel selection strategy: 'average', 'left', 'right', or channel index."""

    # Output / download configuration
    output_dir: Path = field(default_factory=lambda: Path.cwd())
    """Directory for output files (default: current working directory)."""

    output_path: Optional[str] = None
    """Full path for output file (overrides output_dir + filename)."""

    output_format: Optional[str] = None
    """Output media format (mp3, wav, mp4, etc.)."""

    prefer_audio: bool = True
    """Prefer audio format when 'auto' is specified."""

    default_audio_format: str = "mp3"
    """Default audio format when no format is specified."""

    default_video_format: str = "mp4"
    """Default video format when no format is specified."""

    force_overwrite: bool = False
    """Overwrite existing output files without prompting."""

    def __post_init__(self) -> None:
        """Validate configuration and normalize paths/formats."""
        self._setup_output_directory()
        self._validate_default_formats()
        self._normalize_media_format()
        self._process_input_path()
        self._process_output_path()

    def _setup_output_directory(self) -> None:
        """Ensure output directory exists and is valid."""
        resolved_output_dir = self._ensure_dir(self.output_dir)
        self.output_dir = resolved_output_dir

    def _validate_default_formats(self) -> None:
        """Validate default audio and video formats."""
        self.default_audio_format = self._normalize_format(self.default_audio_format)
        self.default_video_format = self._normalize_format(self.default_video_format)

    def _normalize_media_format(self) -> None:
        """Normalize media format, allowing 'auto' during initialization."""
        self.media_format = self._normalize_format(self.media_format, allow_auto=True)

    def _process_input_path(self) -> None:
        """Process and validate input path if provided."""
        if self.input_path is None:
            return

        if self._is_url(self.input_path):
            normalized_url = self._normalize_url(self.input_path)
            self.input_path = normalized_url
            if self.media_format == "auto":
                inferred_format = self._infer_format_from_source(normalized_url)
                if inferred_format:
                    self.media_format = self._normalize_format(inferred_format)
        else:
            # For local paths, normalize to string without validation here
            # Validation will be done in check_input_sanity()
            self.input_path = str(Path(self.input_path).expanduser())
            if self.media_format == "auto":
                inferred_format = Path(self.input_path).suffix.lstrip(".").lower()
                if inferred_format:
                    self.media_format = self._normalize_format(inferred_format)

        # Validate input after setting
        self.check_input_sanity()

    def _process_output_path(self) -> None:
        """Process output path and format."""
        if self.output_path is not None:
            self.set_output_path(self.output_path)
        elif self.output_format is not None:
            self.output_format = self._normalize_format(self.output_format)
        else:
            self.output_format = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clone(self, **updates: object) -> "MediaConfig":
        """Return a shallow copy of the config with optional overrides."""
        return replace(self, **updates)

    def normalize_format(self, media_format: Optional[str] = None, *, prefer_audio: Optional[bool] = None) -> str:
        """Resolve a media format (handling the special "auto" value)."""
        prefer_audio = self.prefer_audio if prefer_audio is None else prefer_audio
        candidate = (media_format or self.media_format or "auto").lower()
        if candidate == "auto":
            candidate = self.default_audio_format if prefer_audio else self.default_video_format
        return self._normalize_format(candidate)

    def is_audio_format(self, media_format: Optional[str] = None) -> bool:
        """Check whether the provided (or effective) format is an audio format."""
        return self.normalize_format(media_format) in AUDIO_FORMATS

    def is_video_format(self, media_format: Optional[str] = None) -> bool:
        """Check whether the provided (or effective) format is a video format."""
        return self.normalize_format(media_format) in VIDEO_FORMATS

    def set_media_format(self, media_format: Optional[str], *, prefer_audio: Optional[bool] = None) -> str:
        """Update media_format and return the normalized value."""
        normalized = self.normalize_format(media_format, prefer_audio=prefer_audio)
        self.media_format = normalized
        return normalized

    def set_input_path(self, path: Pathlike) -> Path | str:
        """Update the input path (local path or URL) and infer format if possible."""
        path = str(path)
        if self._is_url(path):
            normalized_url = self._normalize_url(path)
            self.input_path = normalized_url
            inferred_format = self._infer_format_from_source(normalized_url)
            if inferred_format:
                self.media_format = self._normalize_format(inferred_format)
            self.check_input_sanity()
            return normalized_url

        resolved = self._ensure_file(path)
        self.input_path = str(resolved)
        inferred_format = resolved.suffix.lstrip(".").lower()
        if inferred_format:
            self.media_format = self._normalize_format(inferred_format)
        self.check_input_sanity()
        return resolved

    def set_output_dir(self, output_dir: Pathlike) -> Path:
        """Update the output directory (creating it if needed)."""
        resolved = self._ensure_dir(output_dir)
        self.output_dir = resolved
        return resolved

    def set_output_path(self, output_path: Pathlike) -> Path:
        """Update the output path and synchronize output format and directory."""
        resolved = self._ensure_file(output_path, must_exist=False, create_parent=True)
        if not resolved.suffix:
            raise ValueError("output_path must include a filename with an extension.")
        fmt = resolved.suffix.lstrip(".").lower()
        self.output_path = str(resolved)
        self.output_dir = resolved.parent
        self.output_format = self._normalize_format(fmt)
        return resolved

    def prepare_output_path(self, stem: Optional[str] = None, format: Optional[str] = None) -> Path:
        """Return an output path, creating one if not set yet."""
        if self.output_path:
            return Path(self.output_path)

        effective_format = self.normalize_format(format or self.output_format or self.media_format)
        base_name = stem or (self._derive_input_stem() or "output")
        candidate = self.output_dir / f"{base_name}.{effective_format}"
        self.output_path = str(candidate)
        self.output_format = effective_format
        return candidate

    def is_input_remote(self) -> bool:
        """Return True if the configured input is a URL."""
        return bool(self.input_path and self._is_url(self.input_path))

    def check_input_sanity(self) -> None:
        """
        Validate that input_path is properly configured and accessible.

        Raises:
            ValueError: If input_path is not set or is invalid.
            FileNotFoundError: If input_path is a local file that does not exist.
        """
        if not self.input_path:
            raise ValueError("input_path is required but not set in MediaConfig")

        if self._is_url(self.input_path):
            # For URLs, validate that it's properly formatted
            try:
                parsed = urlparse(self.input_path)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(
                        f"Invalid URL format for input_path: '{self.input_path}'. "
                        "URL must include scheme (http/https) and domain."
                    )
            except (ValueError, AttributeError) as e:
                # ValueError: Invalid URL format
                # AttributeError: urlparse issues with malformed input
                raise ValueError(f"Failed to parse input_path as URL: {e}") from e
        else:
            # For local files, validate that the file exists and is accessible
            input_file = Path(self.input_path).expanduser()
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Input media file does not exist: '{input_file}'. " "Please check the path and try again."
                )
            if not input_file.is_file():
                raise ValueError(
                    f"Input media path is not a file: '{input_file}'. " "Expected a valid media file path."
                )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _ensure_dir(self, directory: Pathlike) -> Path:
        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(f"Output directory '{path}' is not a directory.")
        return path

    def _ensure_file(self, path: Pathlike, *, must_exist: bool = True, create_parent: bool = False) -> Path:
        file_path = Path(path).expanduser()
        if must_exist:
            if not file_path.exists():
                raise FileNotFoundError(f"Input media path '{file_path}' does not exist.")
            if not file_path.is_file():
                raise ValueError(f"Input media path '{file_path}' is not a file.")
        else:
            if create_parent:
                file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def _normalize_format(self, media_format: Optional[str], *, allow_auto: bool = False) -> str:
        if media_format is None:
            raise ValueError("media_format cannot be None")
        normalized = media_format.strip().lower()
        if not normalized:
            raise ValueError("media_format cannot be empty")
        if normalized == "auto":
            if allow_auto:
                return normalized
            normalized = self.default_audio_format if self.prefer_audio else self.default_video_format
        if normalized not in MEDIA_FORMATS:
            raise ValueError(
                "Unsupported media format '{fmt}'. Supported formats: {supported}".format(
                    fmt=media_format,
                    supported=", ".join(MEDIA_FORMATS),
                )
            )
        return normalized

    def _clean_url_escapes(self, url: str) -> str:
        """Remove shell escape backslashes from URL special characters."""
        return url.strip().replace(r"\?", "?").replace(r"\=", "=").replace(r"\&", "&")

    def _is_url(self, value: Pathlike) -> bool:
        if not isinstance(value, str):
            return False
        cleaned = self._clean_url_escapes(value)
        parsed = urlparse(cleaned)
        return bool(parsed.scheme and parsed.netloc)

    def _normalize_url(self, url: str) -> str:
        cleaned = self._clean_url_escapes(url)
        parsed = urlparse(cleaned)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("input_path must be an absolute URL when provided as a remote source.")
        return cleaned

    def _infer_format_from_source(self, source: str) -> Optional[str]:
        path_segment = Path(urlparse(source).path) if self._is_url(source) else Path(source)
        suffix = path_segment.suffix.lstrip(".").lower()
        return suffix or None

    def _derive_input_stem(self) -> Optional[str]:
        if not self.input_path:
            return None
        if self.is_input_remote():
            path_segment = Path(urlparse(self.input_path).path)
            stem = path_segment.stem
            return stem or None
        return Path(self.input_path).stem or None


__all__ = [
    "MediaConfig",
    "AUDIO_FORMATS",
    "VIDEO_FORMATS",
    "MEDIA_FORMATS",
]
