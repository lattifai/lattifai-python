"""Segmented alignment for long audio files."""

from typing import List, Optional, Tuple

import colorful

from lattifai.audio2 import AudioData
from lattifai.caption import Caption, Supervision
from lattifai.config import AlignmentConfig

from .tokenizer import END_PUNCTUATION


class Segmenter:
    """
    Handles segmented alignment for long audio/video files.

    Instead of aligning the entire audio at once (which can be slow and memory-intensive
    for long files), this class splits the alignment into manageable segments based on
    caption boundaries, time intervals, or an adaptive strategy.
    """

    def __init__(self, config: AlignmentConfig):
        """
        Initialize segmented aligner.

        Args:
            config: Alignment configuration with segmentation parameters
        """
        self.config = config

    def __call__(
        self,
        caption: Caption,
        max_duration: Optional[float] = None,
    ) -> List[Tuple[float, float, List[Supervision]]]:
        """
        Create segments based on caption boundaries and gaps.

        Splits when:
        1. Gap between captions exceeds segment_max_gap
        2. Duration approaches max_duration (adaptive mode only) and there's a reasonable break
        3. Duration significantly exceeds max_duration (adaptive mode only)

        Args:
            caption: Caption object with supervisions
            max_duration: Optional maximum segment duration (enables adaptive behavior)

        Returns:
            List of (start_time, end_time, supervisions) tuples for each segment
        """
        if not max_duration:
            max_duration = self.config.segment_duration

        if not caption.supervisions:
            return []

        supervisions = sorted(caption.supervisions, key=lambda s: s.start)

        segments = []
        current_segment_sups = []

        def should_skipalign(sups):
            return len(sups) == 1 and sups[0].text.strip().startswith("[") and sups[0].text.strip().endswith("]")

        for i, sup in enumerate(supervisions):
            if not current_segment_sups:
                current_segment_sups.append(sup)
                if should_skipalign(current_segment_sups):
                    # Single [APPLAUSE] caption, make its own segment
                    segments.append(
                        (current_segment_sups[0].start, current_segment_sups[-1].end, current_segment_sups, True)
                    )
                    current_segment_sups = []
                continue

            prev_sup = supervisions[i - 1]

            gap = max(sup.start - prev_sup.end, 0.0)
            # Always split on large gaps (natural breaks)
            exclude_max_gap = False
            if gap > self.config.segment_max_gap:
                exclude_max_gap = True

            endswith_punc = any(sup.text.endswith(punc) for punc in END_PUNCTUATION)

            # Adaptive duration control
            segment_duration = sup.end - current_segment_sups[0].start

            # Split if approaching duration limit and there's a reasonable break
            should_split = False
            if segment_duration >= max_duration * 0.8 and gap >= 1.0:
                should_split = True

            # Force split if duration exceeded significantly
            exclude_max_duration = False
            if segment_duration >= max_duration * 1.2:
                exclude_max_duration = True

            # [APPLAUSE] [APPLAUSE] [MUSIC]
            if sup.text.strip().startswith("[") and sup.text.strip().endswith("]"):
                # Close current segment
                if current_segment_sups:
                    segment_start = current_segment_sups[0].start
                    segment_end = current_segment_sups[-1].end + min(gap / 2.0, 2.0)
                    segments.append(
                        (segment_start, segment_end, current_segment_sups, should_skipalign(current_segment_sups))
                    )

                # Add current supervision as its own segment
                segments.append((sup.start, sup.end, [sup], True))

                # Update reset for new segment
                current_segment_sups = []
                continue

            if (should_split and endswith_punc) or exclude_max_gap or exclude_max_duration:
                # Close current segment
                if current_segment_sups:
                    segment_start = current_segment_sups[0].start
                    segment_end = current_segment_sups[-1].end + min(gap / 2.0, 2.0)
                    segments.append(
                        (segment_start, segment_end, current_segment_sups, should_skipalign(current_segment_sups))
                    )

                # Start new segment
                current_segment_sups = [sup]
            else:
                current_segment_sups.append(sup)

        # Add final segment
        if current_segment_sups:
            segment_start = current_segment_sups[0].start
            segment_end = current_segment_sups[-1].end + 2.0
            segments.append((segment_start, segment_end, current_segment_sups, should_skipalign(current_segment_sups)))

        return segments

    def print_segment_info(
        self,
        segments: List[Tuple[float, float, List[Supervision]]],
        verbose: bool = True,
    ) -> None:
        """
        Print information about created segments.

        Args:
            segments: List of segment tuples
            verbose: Whether to print detailed info
        """
        if not verbose:
            return

        total_sups = sum(len(sups) if isinstance(sups, list) else 1 for _, _, sups, _ in segments)

        print(colorful.cyan(f"📊 Created {len(segments)} alignment segments:"))
        for i, (start, end, sups, _) in enumerate(segments, 1):
            duration = end - start
            print(
                colorful.white(
                    f"   Segment {i:04d}: {start:8.2f}s - {end:8.2f}s "
                    f"(duration: {duration:8.2f}s, supervisions: {len(sups)if isinstance(sups, list) else 1:4d})"
                )
            )

        print(colorful.green(f"   Total: {total_sups} supervisions across {len(segments)} segments"))
