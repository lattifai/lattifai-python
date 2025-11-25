"""LattifAI speaker diarization implementation."""

import logging
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from tgt import Interval, IntervalTier, TextGrid

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config.diarization import DiarizationConfig
from lattifai.logging import get_logger

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


NOT_KNOWN = "NotKnown"


class LattifAIDiarizer:
    """
    LattifAI Diarizer implementation using pyannote.audio.
    """

    def __init__(self, config: Optional[DiarizationConfig] = None):
        """
        Initialize LattifAI diarizer.

        Args:
            config: Diarization configuration
        """
        if config is None:
            config = DiarizationConfig()

        self.config = config
        self.logger = get_logger("diarization")

        self._diarizer = None

    @property
    def name(self) -> str:
        """Human-readable name of the diarizer."""
        return "LattifAI_Diarizer"

    @property
    def diarizer(self):
        """Lazy-load and return the diarization pipeline."""
        if self._diarizer is None:
            from lattifai_core.diarization import LattifAIDiarizer as CoreLattifAIDiarizer

            self._diarizer = CoreLattifAIDiarizer(config=self.config)

        return self._diarizer

    def diarize(
        self,
        input_media: AudioData,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TextGrid:
        """Perform speaker diarization on the input audio."""
        return self.diarizer.diarize(
            input_media,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def diarize_with_alignments(
        self,
        input_media: AudioData,
        alignments: List[Supervision],
        diarization: Optional[TextGrid] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> Tuple[TextGrid, List[Supervision]]:
        """Diarize the given media input and return alignments with refined speaker labels."""
        return self.diarizer.diarize_with_alignments(
            input_media,
            alignments=alignments,
            diarization=diarization,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
