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
    path_lower = str(path).lower()

    # Check each reader's extensions
    for format_id, reader_cls in _READERS.items():
        if reader_cls.can_read(path_lower):
            return format_id

    # Fallback: try extension directly
    from pathlib import Path

    ext = Path(path_lower).suffix.lstrip(".")
    if ext in _READERS:
        return ext

    return None


# Import all format modules to trigger registration
# Standard formats
from . import gemini  # YouTube/Gemini markdown
from . import pysubs2  # SRT, VTT, ASS, SSA, SUB, SAMI
from . import sbv  # SubViewer
from . import tabular  # CSV, TSV, AUD, TXT, JSON
from . import textgrid  # Praat TextGrid
from . import ttml  # TTML, IMSC1, EBU-TT-D

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
