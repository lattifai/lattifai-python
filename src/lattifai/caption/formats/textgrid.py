"""Praat TextGrid format handler.

TextGrid is Praat's native annotation format, commonly used in phonetics research.
"""

import tempfile
from pathlib import Path
from typing import List

from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("textgrid")
class TextGridFormat(FormatHandler):
    """Praat TextGrid format for phonetic analysis."""

    extensions = [".textgrid"]
    description = "Praat TextGrid - phonetics research format"

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read TextGrid format using tgt library."""
        from tgt import read_textgrid

        if cls.is_content(source):
            # Write to temp file for tgt library
            with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False, mode="w") as f:
                f.write(source)
                temp_path = f.name
            try:
                tgt = read_textgrid(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            tgt = read_textgrid(str(source))

        supervisions = [
            Supervision(
                text=interval.text,
                start=interval.start_time,
                duration=interval.end_time - interval.start_time,
                speaker=tier.name,
            )
            for tier in tgt.tiers
            for interval in tier.intervals
        ]

        return sorted(supervisions, key=lambda x: x.start)

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        **kwargs,
    ) -> Path:
        """Write TextGrid format using tgt library."""
        from lhotse.supervision import AlignmentItem
        from tgt import Interval, IntervalTier, TextGrid, write_to_file

        output_path = Path(output_path)
        tg = TextGrid()

        utterances = []
        words = []
        scores = {"utterances": [], "words": []}

        for sup in sorted(supervisions, key=lambda x: x.start):
            text = sup.text or ""
            if include_speaker and sup.speaker:
                text = f"{sup.speaker} {text}"

            utterances.append(Interval(sup.start, sup.end, text))

            # Extract word-level alignment if present
            alignment = getattr(sup, "alignment", None)
            if alignment and "word" in alignment:
                for item in alignment["word"]:
                    words.append(Interval(item.start, item.end, item.symbol))
                    if item.score is not None:
                        scores["words"].append(Interval(item.start, item.end, f"{item.score:.2f}"))

            if hasattr(sup, "custom") and sup.custom and "score" in sup.custom:
                scores["utterances"].append(Interval(sup.start, sup.end, f"{sup.custom['score']:.2f}"))

        tg.add_tier(IntervalTier(name="utterances", objects=utterances))

        if words:
            tg.add_tier(IntervalTier(name="words", objects=words))

        if scores["utterances"]:
            tg.add_tier(IntervalTier(name="utterance_scores", objects=scores["utterances"]))
        if scores["words"]:
            tg.add_tier(IntervalTier(name="word_scores", objects=scores["words"]))

        write_to_file(tg, str(output_path), format="long")
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert to TextGrid format bytes."""
        # TextGrid requires file I/O due to tgt library implementation
        with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cls.write(supervisions, tmp_path, include_speaker)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
