"""Common type definitions for LattifAI."""

from pathlib import Path
from typing import List, TypeAlias, Union

from lhotse.utils import Pathlike

from .caption import Supervision

# Path-like types
PathLike: TypeAlias = Pathlike  # Re-export for convenience (str | Path)

# Caption types
SupervisionList: TypeAlias = List[Supervision]
"""List of caption segments with timing and text information."""

# Media format types
MediaFormat: TypeAlias = str
"""Media format string (e.g., 'mp3', 'wav', 'mp4')."""

# URL types
URL: TypeAlias = str
"""String representing a URL."""

__all__ = [
    "PathLike",
    "SupervisionList",
    "MediaFormat",
    "URL",
]
