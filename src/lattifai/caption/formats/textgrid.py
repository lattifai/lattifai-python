"""Praat TextGrid format handler.

TextGrid is Praat's native annotation format, commonly used in phonetics research.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lhotse.utils import Pathlike

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
        """Read TextGrid format using tgt library.

        Preserves tier information in Supervision.custom:
        - textgrid_tier: Original tier name
        - textgrid_tier_index: Original tier index (for ordering)
        """
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

        supervisions = []
        for tier_idx, tier in enumerate(tgt.tiers):
            for interval in tier.intervals:
                supervisions.append(
                    Supervision(
                        text=interval.text,
                        start=interval.start_time,
                        duration=interval.end_time - interval.start_time,
                        speaker=tier.name,
                        custom={
                            "textgrid_tier": tier.name,
                            "textgrid_tier_index": tier_idx,
                        },
                    )
                )

        return sorted(supervisions, key=lambda x: x.start)

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Path:
        """Write TextGrid format using tgt library.

        Args:
            supervisions: List of supervisions to write
            output_path: Output file path
            include_speaker: Whether to include speaker in text
            metadata: Optional metadata (for API consistency)
        """
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
                # Check if speaker should be included
                include_this_speaker = True
                if hasattr(sup, "custom") and sup.custom and not sup.custom.get("original_speaker", True):
                    include_this_speaker = False

                if include_this_speaker:
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
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bytes:
        """Convert to TextGrid format bytes.

        Args:
            supervisions: List of supervisions to convert
            include_speaker: Whether to include speaker in text
            metadata: Optional metadata (currently unused, for API consistency)
        """
        # TextGrid requires file I/O due to tgt library implementation
        with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cls.write(supervisions, tmp_path, include_speaker, metadata=metadata, **kwargs)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str], **kwargs) -> Dict[str, Any]:
        """Extract metadata from TextGrid.

        Returns:
            Dict containing:
            - textgrid_xmin: Minimum time boundary
            - textgrid_xmax: Maximum time boundary
            - textgrid_tiers: List of tier names
        """
        import re
        from pathlib import Path

        metadata: Dict[str, Any] = {}
        if cls.is_content(source):
            content = source
        else:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                return {}

        match = re.search(r"xmin\s*=\s*([\d.]+)", content)
        if match:
            metadata["textgrid_xmin"] = float(match.group(1))
        match = re.search(r"xmax\s*=\s*([\d.]+)", content)
        if match:
            metadata["textgrid_xmax"] = float(match.group(1))

        # Extract tier names
        tier_names = re.findall(r'name\s*=\s*"([^"]+)"', content)
        if tier_names:
            metadata["textgrid_tiers"] = tier_names

        return metadata
