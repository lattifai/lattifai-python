from dataclasses import dataclass
from typing import Optional

from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds


@dataclass
class Supervision(SupervisionSegment):
    text: Optional[str] = None
    id: str = ''
    recording_id: str = ''
    start: Seconds = 0.0
    duration: Seconds = 0.0


__all__ = ['Supervision']
