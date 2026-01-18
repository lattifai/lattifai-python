"""Caption format handlers registry.

This module provides a central registry for all caption format readers and writers.
Formats are registered using decorators and can be looked up by format ID.

Example:
    >>> from lattifai.caption.formats import get_reader, get_writer
    >>> reader = get_reader("srt")
    >>> supervisions = reader.read("input.srt")
    >>> writer = get_writer("vtt")
    >>> writer.write(supervisions, "output.vtt")
"""

from typing import Dict, List, Optional, Type

from .base import FormatHandler, FormatReader, FormatWriter

# Global registries
_READERS: Dict[str, Type[FormatReader]] = {}
_WRITERS: Dict[str, Type[FormatWriter]] = {}


def register_reader(format_id: str):
    """Decorator to register a format reader.

    Args:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")

    Example:
        @register_reader("srt")
        class SRTReader(FormatReader):
            ...
    """

    def decorator(cls: Type[FormatReader]) -> Type[FormatReader]:
        cls.format_id = format_id
        _READERS[format_id.lower()] = cls
        return cls

    return decorator


def register_writer(format_id: str):
    """Decorator to register a format writer.

    Args:
        format_id: Unique identifier for the format

    Example:
        @register_writer("srt")
        class SRTWriter(FormatWriter):
            ...
    """

    def decorator(cls: Type[FormatWriter]) -> Type[FormatWriter]:
        cls.format_id = format_id
        _WRITERS[format_id.lower()] = cls
        return cls

    return decorator


def register_format(format_id: str):
    """Decorator to register both reader and writer for a format.

    Use this for classes that implement both FormatReader and FormatWriter.

    Args:
        format_id: Unique identifier for the format

    Example:
        @register_format("srt")
        class SRTFormat(FormatHandler):
            ...
    """

    def decorator(cls: Type[FormatHandler]) -> Type[FormatHandler]:
        cls.format_id = format_id
        _READERS[format_id.lower()] = cls
        _WRITERS[format_id.lower()] = cls
        return cls

    return decorator


def get_reader(format_id: str) -> Optional[Type[FormatReader]]:
    """Get a reader class by format ID.

    Args:
        format_id: Format identifier (case-insensitive)

    Returns:
        Reader class or None if not found
    """
    return _READERS.get(format_id.lower())


def get_writer(format_id: str) -> Optional[Type[FormatWriter]]:
    """Get a writer class by format ID.

    Args:
        format_id: Format identifier (case-insensitive)

    Returns:
        Writer class or None if not found
    """
    return _WRITERS.get(format_id.lower())


def list_readers() -> List[str]:
    """Get list of all registered reader format IDs."""
    return sorted(_READERS.keys())


def list_writers() -> List[str]:
    """Get list of all registered writer format IDs."""
    return sorted(_WRITERS.keys())


def detect_format(path: str) -> Optional[str]:
    """Detect format from file path by checking registered readers.

    Args:
        path: File path to check

    Returns:
        Format ID or None if no match found
    """
    path_str = str(path)

    # Check if it's content instead of a path
    is_content = "\n" in path_str or len(path_str) > 500

    # Prioritize specific formats that can detect by content
    # These often use shared extensions like .vtt, .txt, or .xml
    priority_formats = ["vtt", "gemini", "premiere_xml"]
    for format_id in priority_formats:
        reader_cls = _READERS.get(format_id)
        if reader_cls and reader_cls.can_read(path_str):
            return format_id

    if is_content:
        return None

    # Check each reader's extensions
    path_lower = path_str.lower()
    for format_id, reader_cls in _READERS.items():
        if format_id in priority_formats:
            continue
        if reader_cls.can_read(path_lower):
            return format_id

    # Fallback: try extension directly
    from pathlib import Path

    try:
        ext = Path(path_lower).suffix.lstrip(".")
        if ext in _READERS:
            return ext
    except (OSError, ValueError):
        # Likely content, not a path
        pass

    return None


# Import all format modules to trigger registration
# Standard formats
from . import gemini  # YouTube/Gemini markdown
from . import lrc  # Enhanced LRC with word-level timestamps
from . import pysubs2  # SRT, ASS, SSA, SUB, SAMI
from . import sbv  # SubViewer
from . import tabular  # CSV, TSV, AUD, TXT, JSON
from . import textgrid  # Praat TextGrid
from . import ttml  # TTML, IMSC1, EBU-TT-D
from . import vtt  # WebVTT with YouTube VTT word-level timestamp support

# Professional NLE formats
from .nle import audition  # Adobe Audition / Pro Tools markers
from .nle import avid  # Avid DS
from .nle import fcpxml  # Final Cut Pro XML
from .nle import premiere  # Adobe Premiere Pro XML

__all__ = [
    # Base classes
    "FormatReader",
    "FormatWriter",
    "FormatHandler",
    # Registration
    "register_reader",
    "register_writer",
    "register_format",
    # Lookup
    "get_reader",
    "get_writer",
    "list_readers",
    "list_writers",
    "detect_format",
]
