"""Segmented alignment for long audio files."""

from typing import List, Optional, Tuple

import colorful

from lattifai.audio2 import AudioData
from lattifai.caption import Caption, Supervision
from lattifai.config import CaptionConfig


class SegmentedAligner:
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

    def create_segments(
        self,
        caption: Caption,
        audio: Optional[AudioData] = None,
    ) -> List[Tuple[float, float, List[Supervision]]]:
        """
        Split caption into alignment segments.

        Args:
            caption: Caption object with supervisions to segment
            audio: Optional audio data for duration validation

        Returns:
            List of (start_time, end_time, supervisions) tuples for each segment
        """
        if not caption.supervisions:
            return []

        strategy = self.config.segment_strategy

        if strategy == "time":
            return self._create_time_based_segments(caption, audio)
        elif strategy == "caption":
            return self._create_caption_based_segments(caption)
        elif strategy == "adaptive":
            return self._create_adaptive_segments(caption, audio)
        else:
            raise ValueError(f"Unknown segment_strategy: {strategy}")

    def _create_caption_based_segments(
        self,
        caption: Caption,
    ) -> List[Tuple[float, float, List[Supervision]]]:
        """
        Create segments based on caption boundaries and gaps.

        Splits when:
        1. Gap between captions exceeds segment_max_gap
        2. Natural chapter/scene breaks are detected

        Args:
            caption: Caption object with supervisions

        Returns:
            List of segment tuples
        """
        segments = []
        current_segment_sups = []
        current_start = None

        supervisions = sorted(caption.supervisions, key=lambda s: s.start)

        for i, sup in enumerate(supervisions):
            if current_start is None:
                current_start = sup.start
                current_segment_sups.append(sup)
                continue

            # Check gap from previous supervision
            prev_sup = supervisions[i - 1]
            gap = sup.start - prev_sup.end

            # Split if gap is too large
            if gap > self.config.segment_max_gap:
                # Close current segment
                segment_end = prev_sup.end + self.config.segment_overlap
                segments.append((current_start, segment_end, current_segment_sups))

                # Start new segment
                current_start = max(sup.start - self.config.segment_overlap, 0)
                current_segment_sups = [sup]
            else:
                current_segment_sups.append(sup)

        # Add final segment
        if current_segment_sups:
            final_sup = current_segment_sups[-1]
            segments.append((current_start, final_sup.end, current_segment_sups))

        return segments

    def _create_time_based_segments(
        self,
        caption: Caption,
        audio: Optional[AudioData] = None,
    ) -> List[Tuple[float, float, List[Supervision]]]:
        """
        Create fixed-duration time segments.

        Args:
            caption: Caption object with supervisions
            audio: Optional audio for total duration

        Returns:
            List of segment tuples
        """
        if not self.config.segment_duration:
            raise ValueError("segment_duration must be set for time-based segmentation")

        segments = []
        supervisions = sorted(caption.supervisions, key=lambda s: s.start)

        if not supervisions:
            return []

        # Determine total duration
        if audio is not None:
            total_duration = audio.duration
        else:
            total_duration = supervisions[-1].end

        segment_dur = self.config.segment_duration
        overlap = self.config.segment_overlap

        current_time = 0.0
        while current_time < total_duration:
            segment_start = max(current_time - overlap, 0) if current_time > 0 else 0
            segment_end = current_time + segment_dur

            # Find supervisions in this time range
            segment_sups = [sup for sup in supervisions if sup.start < segment_end and sup.end > segment_start]

            if segment_sups:
                segments.append((segment_start, segment_end, segment_sups))

            current_time += segment_dur

        return segments

    def _create_adaptive_segments(
        self,
        caption: Caption,
        audio: Optional[AudioData] = None,
    ) -> List[Tuple[float, float, List[Supervision]]]:
        """
        Create segments adaptively based on both caption structure and time limits.

        Combines caption-based and time-based strategies:
        - Respects caption boundaries and natural breaks
        - Limits segment duration to avoid excessively long segments
        - Splits at natural boundaries when approaching duration limit

        Args:
            caption: Caption object with supervisions
            audio: Optional audio for validation

        Returns:
            List of segment tuples
        """
        if not self.config.segment_duration:
            # Fall back to caption-based if no duration limit
            return self._create_caption_based_segments(caption)

        segments = []
        supervisions = sorted(caption.supervisions, key=lambda s: s.start)

        if not supervisions:
            return []

        current_segment_sups = []
        current_start = supervisions[0].start

        for i, sup in enumerate(supervisions):
            if not current_segment_sups:
                current_segment_sups.append(sup)
                continue

            # Calculate current segment duration
            segment_duration = sup.end - current_start

            # Check if we should split
            should_split = False

            # Split if approaching duration limit and there's a natural break
            if segment_duration >= self.config.segment_duration * 0.9:
                prev_sup = supervisions[i - 1]
                gap = sup.start - prev_sup.end

                # Split if there's any reasonable gap
                if gap >= 1.0:  # 1 second gap is reasonable
                    should_split = True

            # Force split if duration exceeded significantly
            if segment_duration >= self.config.segment_duration * 1.2:
                should_split = True

            # Split on large gaps regardless of duration
            if i > 0:
                prev_sup = supervisions[i - 1]
                gap = sup.start - prev_sup.end
                if gap > self.config.segment_max_gap:
                    should_split = True

            if should_split:
                # Close current segment
                last_sup = current_segment_sups[-1]
                segment_end = last_sup.end + self.config.segment_overlap
                segments.append((current_start, segment_end, current_segment_sups))

                # Start new segment with overlap
                current_start = max(sup.start - self.config.segment_overlap, 0)
                current_segment_sups = [sup]
            else:
                current_segment_sups.append(sup)

        # Add final segment
        if current_segment_sups:
            final_sup = current_segment_sups[-1]
            segments.append((current_start, final_sup.end, current_segment_sups))

        return segments

    def merge_aligned_segments(
        self,
        segment_results: List[Tuple[float, float, List[Supervision]]],
    ) -> List[Supervision]:
        """
        Merge aligned supervisions from multiple segments.

        Handles overlapping regions by:
        1. Deduplicating supervisions in overlap zones
        2. Choosing the best alignment (based on confidence if available)
        3. Ensuring temporal continuity

        Args:
            segment_results: List of (start, end, aligned_supervisions) from each segment

        Returns:
            Merged list of aligned supervisions
        """
        if not segment_results:
            return []

        if len(segment_results) == 1:
            return segment_results[0][2]

        all_supervisions = []
        overlap = self.config.segment_overlap

        for i, (seg_start, seg_end, sups) in enumerate(segment_results):
            if i == 0:
                # First segment: take everything
                all_supervisions.extend(sups)
            else:
                # Subsequent segments: skip overlap region
                prev_end = segment_results[i - 1][1]
                overlap_start = prev_end - overlap

                # Only add supervisions that start after the overlap region
                non_overlap_sups = [sup for sup in sups if sup.start >= overlap_start]
                all_supervisions.extend(non_overlap_sups)

        # Sort by start time and remove any remaining duplicates
        all_supervisions.sort(key=lambda s: s.start)

        # Remove duplicates based on text and approximate timing
        deduplicated = []
        for sup in all_supervisions:
            # Check if this supervision is a duplicate of the last added one
            if deduplicated:
                last = deduplicated[-1]
                time_diff = abs(sup.start - last.start)

                # Consider it a duplicate if text matches and timing is very close
                if sup.text == last.text and time_diff < 0.1:
                    # Keep the one with better alignment (if score available)
                    if hasattr(sup, "score") and hasattr(last, "score"):
                        if sup.score > last.score:
                            deduplicated[-1] = sup
                    continue

            deduplicated.append(sup)

        return deduplicated

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

        total_sups = sum(len(sups) for _, _, sups in segments)

        print(colorful.cyan(f"📊 Created {len(segments)} alignment segments:"))
        for i, (start, end, sups) in enumerate(segments, 1):
            duration = end - start
            print(
                colorful.white(
                    f"   Segment {i}: {start:.1f}s - {end:.1f}s "
                    f"(duration: {duration:.1f}s, supervisions: {len(sups)})"
                )
            )

        print(colorful.green(f"   Total: {total_sups} supervisions across {len(segments)} segments"))
