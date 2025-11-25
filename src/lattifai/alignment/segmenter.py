"""Segmented alignment for long audio files."""

from typing import List, Optional, Tuple

import colorful

from lattifai.audio2 import AudioData
from lattifai.caption import Caption, Supervision
from lattifai.config import CaptionConfig

from .tokenizer import END_PUNCTUATION


class Segmenter:
    """
    Handles segmented alignment for long audio/video files.

    Instead of aligning the entire audio at once (which can be slow and memory-intensive
    for long files), this class splits the alignment into manageable segments based on
    caption boundaries, time intervals, or an adaptive strategy.
    """

    def __init__(self, caption_config: CaptionConfig):
        """
        Initialize segmented aligner.

        Args:
            caption_config: Caption configuration with segmentation parameters
        """
        self.config = caption_config

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
        current_start = max(supervisions[0].start - 2.0, 0.0)

        for i, sup in enumerate(supervisions):
            if not current_segment_sups:
                current_segment_sups.append(sup)
                continue

            prev_sup = supervisions[i - 1]

            gap = sup.start - prev_sup.end
            next_gap = 0.0 if i + 1 >= len(supervisions) else supervisions[i + 1].start - sup.end
            # Always split on large gaps (natural breaks)
            exclude_max_gap = False
            if gap > self.config.segment_max_gap:
                exclude_max_gap = True

            endswith_punc = any(sup.text.endswith(punc) for punc in END_PUNCTUATION)  # and next_gap >= 0.24
            long_and_multisents = len(sup.text) > 20 and any(punc in sup.text for punc in END_PUNCTUATION)

            # Adaptive duration control
            segment_duration = sup.end - current_start

            # Split if approaching duration limit and there's a reasonable break
            should_split = False
            if segment_duration >= max_duration * 0.8 and gap >= 1.0:
                should_split = True

            # Force split if duration exceeded significantly
            exclude_max_duration = False
            if segment_duration >= max_duration * 1.2:
                exclude_max_duration = True

            if (should_split and endswith_punc) or exclude_max_gap or exclude_max_duration:
                # Close current segment
                segment_end = prev_sup.end + gap / 2.0
                segments.append((current_start, segment_end, current_segment_sups))

                if not exclude_max_gap and long_and_multisents and not endswith_punc:
                    segments.append(
                        (sup.start - min(gap / 2.0, 2.0), sup.end + min(next_gap / 2.0, 2.0), sup)
                    )  # will align this supervision to separately
                    current_start = sup.end
                    current_segment_sups = []
                else:
                    # Start new segment
                    current_start = sup.start - gap / 2.0
                    current_segment_sups = [sup]
            else:
                current_segment_sups.append(sup)

        # Add final segment
        if current_segment_sups:
            final_sup = current_segment_sups[-1]
            segments.append((current_start, final_sup.end + 2.0, current_segment_sups))

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

        total_sups = sum(len(sups) if isinstance(sups, list) else 1 for _, _, sups in segments)

        print(colorful.cyan(f"📊 Created {len(segments)} alignment segments:"))
        for i, (start, end, sups) in enumerate(segments, 1):
            duration = end - start
            print(
                colorful.white(
                    f"   Segment {i}: {start:8.2f}s - {end:8.2f}s "
                    f"(duration: {duration:8.2f}s, supervisions: {len(sups)if isinstance(sups, list) else 1:4d})"
                )
            )

        print(colorful.green(f"   Total: {total_sups} supervisions across {len(segments)} segments"))
