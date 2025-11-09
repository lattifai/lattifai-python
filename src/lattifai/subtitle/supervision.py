from dataclasses import dataclass
from typing import Optional

from lhotse.supervision import SupervisionSegment
from lhotse.utils import Seconds


@dataclass
class Supervision(SupervisionSegment):
    """
    Extended SupervisionSegment with simplified initialization.

    Note: The `alignment` field is inherited from SupervisionSegment:
        alignment: Optional[Dict[str, List[AlignmentItem]]] = None

    Structure of alignment when return_details=True:
        {
            'word': [
                AlignmentItem(symbol='hello', start=0.0, duration=0.5, score=0.95),
                AlignmentItem(symbol='world', start=0.6, duration=0.4, score=0.92),
                ...
            ]
        }
    """

    text: Optional[str] = None
    id: str = ""
    recording_id: str = ""
    start: Seconds = 0.0
    duration: Seconds = 0.0


__all__ = ["Supervision"]
