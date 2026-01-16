"""Base classes for caption format readers and writers.

This module provides abstract base classes that all format handlers must implement,
ensuring a consistent interface across different caption formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from lhotse.utils import Pathlike

if TYPE_CHECKING:
    from ..supervision import Supervision


class FormatReader(ABC):
    """Abstract base class for caption format readers.

    All format readers must implement the `read` method to parse caption content
    and return a list of Supervision objects.

    Class Attributes:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")
        extensions: List of file extensions this reader handles (e.g., [".srt"])
        description: Human-readable description of the format
    """

    format_id: str = ""
    extensions: List[str] = []
    description: str = ""

    @classmethod
    @abstractmethod
    def read(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> List["Supervision"]:
        """Read caption content and return list of Supervision objects.

        Args:
            source: File path or string content
            normalize_text: Whether to normalize text (strip HTML, etc.)
            **kwargs: Format-specific options

        Returns:
            List of Supervision objects with timing and text
        """
        pass

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str]) -> Dict[str, str]:
        """Extract metadata from caption file or content.

        Args:
            source: File path or string content

        Returns:
            Dictionary of metadata key-value pairs
        """
        return {}

    @classmethod
    def can_read(cls, path: Union[Pathlike, str]) -> bool:
        """Check if this reader can handle the given file.

        Args:
            path: File path to check

        Returns:
            True if this reader supports the file format
        """
        path_str = str(path).lower()
        return any(path_str.endswith(ext.lower()) for ext in cls.extensions)

    @classmethod
    def is_content(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is string content rather than a file path.

        Args:
            source: Source to check

        Returns:
            True if source appears to be content, not a path
        """
        if not isinstance(source, str):
            return False
        # If it has newlines or is very long, it's likely content
        return "\n" in source or len(source) > 500


class FormatWriter(ABC):
    """Abstract base class for caption format writers.

    All format writers must implement `write` and `to_bytes` methods.

    Class Attributes:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")
        extensions: List of file extensions for this format
        description: Human-readable description of the format
    """

    format_id: str = ""
    extensions: List[str] = []
    description: str = ""

    @classmethod
    @abstractmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ) -> Path:
        """Write supervisions to a file.

        Args:
            supervisions: List of Supervision objects to write
            output_path: Path to output file
            include_speaker: Whether to include speaker labels in text
            **kwargs: Format-specific options

        Returns:
            Path to the written file
        """
        pass

    @classmethod
    @abstractmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to bytes in this format.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker labels
            **kwargs: Format-specific options

        Returns:
            Caption content as bytes
        """
        pass

    @classmethod
    def _should_include_speaker(cls, sup: Any, include_speaker: bool) -> bool:
        """Check if speaker should be included in output text.

        Considers both the global include_speaker flag and the segment-level
        'original_speaker' flag in custom metadata.
        """
        if not include_speaker or not getattr(sup, "speaker", None):
            return False
        custom = getattr(sup, "custom", None)
        if custom and not custom.get("original_speaker", True):
            return False
        return True


class FormatHandler(FormatReader, FormatWriter):
    """Combined reader and writer for formats that support both.

    Most caption formats support both reading and writing. This class
    combines both interfaces for convenience.
    """

    pass


# Type aliases for registration
ReaderType = type[FormatReader]
WriterType = type[FormatWriter]


def expand_to_word_supervisions(supervisions: List["Supervision"]) -> List["Supervision"]:
    """Expand supervisions with word alignment to one supervision per word.

    Used for word-per-segment output when word_level=True but karaoke=False.

    Args:
        supervisions: List of Supervision objects with optional alignment data

    Returns:
        List of Supervision objects, one per word if alignment exists,
        otherwise returns original supervisions unchanged.
    """
    from ..supervision import Supervision

    result = []
    for sup in supervisions:
        if sup.alignment and "word" in sup.alignment:
            for word in sup.alignment["word"]:
                result.append(
                    Supervision(
                        text=word.symbol,
                        start=word.start,
                        duration=word.duration,
                        speaker=sup.speaker,
                        id=f"{sup.id}_word" if sup.id else "",
                        recording_id=sup.recording_id if hasattr(sup, "recording_id") else "",
                    )
                )
        else:
            result.append(sup)
    return result
