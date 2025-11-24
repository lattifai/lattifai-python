from typing import List, Optional

from lhotse.utils import Pathlike

from ..config import CaptionConfig
from .gemini_reader import GeminiReader, GeminiSegment
from .gemini_writer import GeminiWriter
from .reader import CaptionReader, InputCaptionFormat
from .supervision import Supervision
from .text_parser import normalize_text
from .writer import CaptionWriter

__all__ = [
    "CaptionReader",
    "CaptionWriter",
    "CaptionIO",
    "Supervision",
    "GeminiReader",
    "GeminiWriter",
    "GeminiSegment",
    "normalize_text",
    "Captioner",
]


class CaptionIO:
    def __init__(self):
        pass

    @classmethod
    def read(
        cls, caption: Pathlike, format: Optional[InputCaptionFormat] = None, normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        return CaptionReader.read(caption, format=format, normalize_text=normalize_text)

    @classmethod
    def write(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> Pathlike:
        return CaptionWriter.write(alignments, output_path, include_speaker_in_text=include_speaker_in_text)


class Captioner:
    def __init__(self, config: Optional[CaptionConfig] = None):
        if config is None:
            config = CaptionConfig()
        self.config = config

    def read(
        self,
        input_path: Optional[Pathlike] = None,
        format: Optional[InputCaptionFormat] = None,
        normalize_text: Optional[bool] = False,
    ) -> List[Supervision]:
        if not input_path:
            input_path = self.config.input_path
            assert self.config.check_sanity() is True
        return CaptionIO.read(
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
        return CaptionIO.write(
            alignments,
            output_path,
            include_speaker_in_text=self.config.include_speaker_in_text,
        )
