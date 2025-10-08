from typing import List, Optional

from lhotse.utils import Pathlike

from .reader import SubtitleFormat, SubtitleReader
from .supervision import Supervision
from .writer import SubtitleWriter

__all__ = ['SubtitleReader', 'SubtitleWriter', 'SubtitleIO', 'Supervision']


class SubtitleIO:
    def __init__(self):
        pass

    @classmethod
    def read(cls, subtitle: Pathlike, format: Optional[SubtitleFormat] = None) -> List[Supervision]:
        return SubtitleReader.read(subtitle, format=format)

    @classmethod
    def write(cls, alignments: List[Supervision], output_path: Pathlike) -> Pathlike:
        return SubtitleWriter.write(alignments, output_path)
