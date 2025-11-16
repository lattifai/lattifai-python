from typing import List, Optional

from lhotse.utils import Pathlike

from ..config import SubtitleConfig
from .gemini_reader import GeminiReader, GeminiSegment
from .gemini_writer import GeminiWriter
from .reader import InputSubtitleFormat, SubtitleReader
from .supervision import Supervision
from .text_parser import normalize_text
from .writer import SubtitleWriter

__all__ = [
    "SubtitleReader",
    "SubtitleWriter",
    "SubtitleIO",
    "Supervision",
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "normalize_text",
    "Subtitler",
]


class SubtitleIO:
    def __init__(self):
        pass

    @classmethod
    def read(
        cls, subtitle: Pathlike, format: Optional[InputSubtitleFormat] = None, normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        return SubtitleReader.read(subtitle, format=format, normalize_text=normalize_text)

    @classmethod
    def write(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> Pathlike:
        return SubtitleWriter.write(alignments, output_path, include_speaker_in_text=include_speaker_in_text)


class Subtitler:
    def __init__(self, config: Optional[SubtitleConfig] = None):
        if config is None:
            config = SubtitleConfig()
        self.config = config

    def read(
        self,
        input_path: Optional[Pathlike] = None,
        format: Optional[InputSubtitleFormat] = None,
        normalize_text: Optional[bool] = False,
    ) -> List[Supervision]:
        if not input_path:
            input_path = self.config.input_path
            assert self.config.check_sanity() is True
        return SubtitleIO.read(
            input_path,
            format=format or self.config.input_format,
            normalize_text=normalize_text or self.config.normalize_text,
        )

    def write(
        self,
        alignments: List[Supervision],
        output_path: Optional[Pathlike] = None,
    ) -> Pathlike:
        output_path = output_path or self.config.output_path
        return SubtitleIO.write(
            alignments,
            output_path,
            include_speaker_in_text=self.config.include_speaker_in_text,
        )
