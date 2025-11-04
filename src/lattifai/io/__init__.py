from typing import List, Optional

from lhotse.utils import Pathlike

from .gemini_reader import GeminiReader, GeminiSegment
from .gemini_writer import GeminiWriter
from .reader import SubtitleFormat, SubtitleReader
from .supervision import Supervision
from .text_parser import normalize_html_text
from .utils import (
    ALL_SUBTITLE_FORMATS,
    INPUT_SUBTITLE_FORMATS,
    OUTPUT_SUBTITLE_FORMATS,
    SUBTITLE_FORMATS,
)
from .writer import SubtitleWriter

__all__ = [
    "SubtitleReader",
    "SubtitleWriter",
    "SubtitleIO",
    "Supervision",
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "SUBTITLE_FORMATS",
    "INPUT_SUBTITLE_FORMATS",
    "OUTPUT_SUBTITLE_FORMATS",
    "ALL_SUBTITLE_FORMATS",
]


class SubtitleIO:
    def __init__(self):
        pass

    @classmethod
    def read(cls, subtitle: Pathlike, format: Optional[SubtitleFormat] = None) -> List[Supervision]:
        return SubtitleReader.read(subtitle, format=format)

    @classmethod
    def write(cls, alignments: List[Supervision], output_path: Pathlike) -> Pathlike:
        return SubtitleWriter.write(alignments, output_path)
