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
    media_format: str = "auto"
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # Output / download configuration
    output_dir: Path = field(default_factory=lambda: Path.cwd())
    output_path: Optional[str] = None
    output_format: Optional[str] = None
    prefer_audio: bool = True
    default_audio_format: str = "mp3"
    default_video_format: str = "mp4"
    force_overwrite: bool = False

    def __post_init__(self) -> None:
        """Validate configuration and normalize paths/formats."""
        resolved_output_dir = self._ensure_dir(self.output_dir)
        object.__setattr__(self, "output_dir", resolved_output_dir)

        # Validate defaults
        object.__setattr__(self, "default_audio_format", self._normalize_format(self.default_audio_format))
        object.__setattr__(self, "default_video_format", self._normalize_format(self.default_video_format))

        # Normalize media format (allow "auto" during initialization)
        object.__setattr__(self, "media_format", self._normalize_format(self.media_format, allow_auto=True))

        if self.input_path is not None:
            if self._is_url(self.input_path):
                normalized_url = self._normalize_url(self.input_path)
                object.__setattr__(self, "input_path", normalized_url)
                if self.media_format == "auto":
                    inferred_format = self._infer_format_from_source(normalized_url)
                    if inferred_format:
                        object.__setattr__(self, "media_format", self._normalize_format(inferred_format))
            else:
                resolved_input = self._ensure_file(self.input_path)
                object.__setattr__(self, "input_path", str(resolved_input))
                if self.media_format == "auto":
                    inferred_format = resolved_input.suffix.lstrip(".").lower()
                    if inferred_format:
                        object.__setattr__(self, "media_format", self._normalize_format(inferred_format))

        if self.output_path is not None:
            self.set_output_path(self.output_path)
        elif self.output_format is not None:
            object.__setattr__(self, "output_format", self._normalize_format(self.output_format))
        else:
            object.__setattr__(self, "output_format", None)

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
        object.__setattr__(self, "media_format", normalized)
        return normalized

    def set_input_path(self, path: Pathlike) -> Path | str:
        """Update the input path (local path or URL) and infer format if possible."""
        if self._is_url(path):
            normalized_url = self._normalize_url(str(path))
            object.__setattr__(self, "input_path", normalized_url)
            inferred_format = self._infer_format_from_source(normalized_url)
            if inferred_format:
                object.__setattr__(self, "media_format", self._normalize_format(inferred_format))
            return normalized_url

        resolved = self._ensure_file(path)
        object.__setattr__(self, "input_path", str(resolved))
        inferred_format = resolved.suffix.lstrip(".").lower()
        if inferred_format:
            object.__setattr__(self, "media_format", self._normalize_format(inferred_format))
        return resolved

    def set_output_dir(self, output_dir: Pathlike) -> Path:
        """Update the output directory (creating it if needed)."""
        resolved = self._ensure_dir(output_dir)
        object.__setattr__(self, "output_dir", resolved)
        return resolved

    def set_output_path(self, output_path: Pathlike) -> Path:
        """Update the output path and synchronize output format and directory."""
        resolved = self._ensure_file(output_path, must_exist=False, create_parent=True)
        if not resolved.suffix:
            raise ValueError("output_path must include a filename with an extension.")
        fmt = resolved.suffix.lstrip(".").lower()
        object.__setattr__(self, "output_path", str(resolved))
        object.__setattr__(self, "output_dir", resolved.parent)
        object.__setattr__(self, "output_format", self._normalize_format(fmt))
        return resolved

    def prepare_output_path(self, stem: Optional[str] = None, format: Optional[str] = None) -> Path:
        """Return an output path, creating one if not set yet."""
        if self.output_path:
            return Path(self.output_path)

        effective_format = self.normalize_format(format or self.output_format or self.media_format)
        base_name = stem or (self._derive_input_stem() or "output")
        candidate = self.output_dir / f"{base_name}.{effective_format}"
        object.__setattr__(self, "output_path", str(candidate))
        object.__setattr__(self, "output_format", effective_format)
        return candidate

    def is_input_remote(self) -> bool:
        """Return True if the configured input is a URL."""
        return bool(self.input_path and self._is_url(self.input_path))

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

    def _is_url(self, value: Pathlike) -> bool:
        if not isinstance(value, str):
            return False
        parsed = urlparse(value.strip())
        return bool(parsed.scheme and parsed.netloc)

    def _normalize_url(self, url: str) -> str:
        stripped = url.strip()
        parsed = urlparse(stripped)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("input_path must be an absolute URL when provided as a remote source.")
        return stripped

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
