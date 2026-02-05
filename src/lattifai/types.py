"""Common type definitions for LattifAI."""

from pathlib import Path
from typing import List, TypeAlias, Union

from lattifai.caption import Supervision

# Path-like types (replaces lhotse.utils.Pathlike)
Pathlike: TypeAlias = Union[str, Path]
"""Type alias for path-like objects (str or Path)."""

PathLike: TypeAlias = Pathlike  # Re-export for convenience

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
