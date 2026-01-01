"""Reader for YouTube transcript files with speaker labels and timestamps."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lhotse.utils import Pathlike

from .supervision import Supervision


@dataclass
class GeminiSegment:
    """Represents a segment in the Gemini transcript with metadata."""

    text: str
    timestamp: Optional[float] = None  # For backward compatibility (start time)
    end_timestamp: Optional[float] = None  # End time when timestamp is at the end
    speaker: Optional[str] = None
    section: Optional[str] = None
    segment_type: str = "dialogue"  # 'dialogue', 'event', or 'section_header'
    line_number: int = 0

    @property
    def start(self) -> float:
        """Return start time in seconds."""
        return self.timestamp if self.timestamp is not None else 0.0

    @property
    def end(self) -> Optional[float]:
        """Return end time in seconds if available."""
        return self.end_timestamp


class GeminiReader:
    """Parser for YouTube transcript format with speaker labels and timestamps."""

    # Regex patterns for parsing (supports both [HH:MM:SS] and [MM:SS] formats)
    TIMESTAMP_PATTERN = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})\]|\[(\d{1,2}):(\d{2})\]")
    SECTION_HEADER_PATTERN = re.compile(r"^##\s*\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.+)$")
    SPEAKER_PATTERN = re.compile(r"^\*\*(.+?[:：])\*\*\s*(.+)$")
    # Event pattern: [Event] [HH:MM:SS] or [Event] [MM:SS] - prioritize HH:MM:SS format
    EVENT_PATTERN = re.compile(r"^\[([^\]]+)\]\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]$")
    # Timestamp at the end indicates end time
    INLINE_TIMESTAMP_END_PATTERN = re.compile(r"^(.+?)\s*\[(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2}))\]$")
    # Timestamp at the beginning indicates start time
    INLINE_TIMESTAMP_START_PATTERN = re.compile(r"^\[(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2}))\]\s*(.+)$")

    # New patterns for YouTube link format: [[MM:SS](URL&t=seconds)]
    YOUTUBE_SECTION_PATTERN = re.compile(r"^##\s*\[\[(\d{1,2}):(\d{2})\]\([^)]*&t=(\d+)\)\]\s*(.+)$")
    YOUTUBE_INLINE_PATTERN = re.compile(r"^(.+?)\s*\[\[(\d{1,2}):(\d{2})\]\([^)]*&t=(\d+)\)\]$")

    @classmethod
    def parse_timestamp(cls, *args) -> float:
        """Convert timestamp to seconds.

        Supports both HH:MM:SS and MM:SS formats.
        Args can be (hours, minutes, seconds) or (minutes, seconds).
        Can also accept a single argument which is seconds.
        """
        if len(args) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = args
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        elif len(args) == 2:
            # MM:SS format
            minutes, seconds = args
            return int(minutes) * 60 + int(seconds)
        elif len(args) == 1:
            # Direct seconds (from YouTube &t= parameter)
            return int(args[0])
        else:
            raise ValueError(f"Invalid timestamp args: {args}")

    @classmethod
    def read(
        cls,
        transcript_path: Pathlike,
        include_events: bool = False,
        include_sections: bool = False,
    ) -> List[GeminiSegment]:
        """Parse YouTube transcript file and return list of transcript segments.

        Args:
                transcript_path: Path to the transcript file
                include_events: Whether to include event descriptions like [Applause]
                include_sections: Whether to include section headers

        Returns:
                List of GeminiSegment objects with all metadata
        """
        transcript_path = Path(transcript_path).expanduser().resolve()
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        segments: List[GeminiSegment] = []
        current_section = None
        current_speaker = None

        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            # Skip table of contents
            if line.startswith("* ["):
                continue
            if line.startswith("## Table of Contents"):
                continue

            # Parse section headers
            section_match = cls.SECTION_HEADER_PATTERN.match(line)
            if section_match:
                hours, minutes, seconds, section_title = section_match.groups()
                timestamp = cls.parse_timestamp(hours, minutes, seconds)
                current_section = section_title.strip()
                if include_sections:
                    segments.append(
                        GeminiSegment(
                            text=section_title.strip(),
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="section_header",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse YouTube format section headers: ## [[MM:SS](URL&t=seconds)] Title
            youtube_section_match = cls.YOUTUBE_SECTION_PATTERN.match(line)
            if youtube_section_match:
                minutes, seconds, url_seconds, section_title = youtube_section_match.groups()
                # Use the URL seconds for more accuracy
                timestamp = cls.parse_timestamp(url_seconds)
                current_section = section_title.strip()
                if include_sections:
                    segments.append(
                        GeminiSegment(
                            text=section_title.strip(),
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="section_header",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse event descriptions [event] [HH:MM:SS] or [MM:SS]
            event_match = cls.EVENT_PATTERN.match(line)
            if event_match:
                groups = event_match.groups()
                event_text = groups[0]
                # Parse timestamp - groups: (event_text, hours/minutes, minutes/seconds, seconds_optional)
                hours_or_minutes = groups[1]
                minutes_or_seconds = groups[2]
                seconds_optional = groups[3]

                if seconds_optional is not None:
                    # HH:MM:SS format
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, seconds_optional)
                else:
                    # MM:SS format
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds)

                if include_events and timestamp is not None:
                    segments.append(
                        GeminiSegment(
                            text=f"[{event_text.strip()}]",
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="event",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse speaker dialogue: **Speaker:** Text [HH:MM:SS] or [MM:SS]
            speaker_match = cls.SPEAKER_PATTERN.match(line)
            if speaker_match:
                speaker, text_with_timestamp = speaker_match.groups()
                current_speaker = speaker.strip()

                # Check for timestamp at the beginning (start time)
                start_match = cls.INLINE_TIMESTAMP_START_PATTERN.match(text_with_timestamp.strip())
                # Check for timestamp at the end (end time)
                end_match = cls.INLINE_TIMESTAMP_END_PATTERN.match(text_with_timestamp.strip())
                youtube_match = cls.YOUTUBE_INLINE_PATTERN.match(text_with_timestamp.strip())

                start_timestamp = None
                end_timestamp = None
                text = text_with_timestamp.strip()

                if start_match:
                    groups = start_match.groups()
                    # Parse timestamp - can be HH:MM:SS (groups 0,1,2) or MM:SS (groups 3,4)
                    if groups[0] is not None:  # HH:MM:SS format
                        start_timestamp = cls.parse_timestamp(groups[0], groups[1], groups[2])
                    elif groups[3] is not None:  # MM:SS format
                        start_timestamp = cls.parse_timestamp(groups[3], groups[4])
                    text = groups[5]  # Text is after timestamp
                elif end_match:
                    groups = end_match.groups()
                    text = groups[0]  # Text is before timestamp
                    # Parse timestamp - can be HH:MM:SS (groups 1,2,3) or MM:SS (groups 4,5)
                    if groups[1] is not None:  # HH:MM:SS format
                        end_timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                    elif groups[4] is not None:  # MM:SS format
                        end_timestamp = cls.parse_timestamp(groups[4], groups[5])
                elif youtube_match:
                    groups = youtube_match.groups()
                    text = groups[0]
                    # Extract seconds from URL parameter (treat as end time)
                    url_seconds = groups[3]
                    end_timestamp = cls.parse_timestamp(url_seconds)

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                current_speaker = None  # Reset speaker after use
                continue

            # Parse plain text with timestamp (check both positions)
            start_match = cls.INLINE_TIMESTAMP_START_PATTERN.match(line)
            end_match = cls.INLINE_TIMESTAMP_END_PATTERN.match(line)
            youtube_inline_match = cls.YOUTUBE_INLINE_PATTERN.match(line)

            start_timestamp = None
            end_timestamp = None
            text = None

            if start_match:
                groups = start_match.groups()
                # Parse timestamp - can be HH:MM:SS (groups 0,1,2) or MM:SS (groups 3,4)
                if groups[0] is not None:  # HH:MM:SS format
                    start_timestamp = cls.parse_timestamp(groups[0], groups[1], groups[2])
                elif groups[3] is not None:  # MM:SS format
                    start_timestamp = cls.parse_timestamp(groups[3], groups[4])
                text = groups[5]  # Text is after timestamp

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=start_timestamp,
                        end_timestamp=None,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                continue
            elif end_match:
                groups = end_match.groups()
                text = groups[0]  # Text is before timestamp
                # Parse timestamp - can be HH:MM:SS (groups 1,2,3) or MM:SS (groups 4,5)
                if groups[1] is not None:  # HH:MM:SS format
                    end_timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                elif groups[4] is not None:  # MM:SS format
                    end_timestamp = cls.parse_timestamp(groups[4], groups[5])

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=None,
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                continue
            elif youtube_inline_match:
                groups = youtube_inline_match.groups()
                text = groups[0]
                # Extract seconds from URL parameter (treat as end time)
                url_seconds = groups[3]
                end_timestamp = cls.parse_timestamp(url_seconds)

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=None,
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                continue

            # Skip markdown headers and other formatting
            if line.startswith("#"):
                continue

        return segments

    @classmethod
    def extract_for_alignment(
        cls,
        transcript_path: Pathlike,
        merge_consecutive: bool = False,
        min_duration: float = 0.1,
        merge_max_gap: float = 2.0,
    ) -> List[Supervision]:
        """Extract text segments for forced alignment.

        This extracts only dialogue segments (not events or section headers)
        and converts them to Supervision objects suitable for alignment.

        Args:
                transcript_path: Path to the transcript file
                merge_consecutive: Whether to merge consecutive segments from same speaker
                min_duration: Minimum duration for a segment
                merge_max_gap: Maximum time gap (seconds) to merge consecutive segments

        Returns:
                List of Supervision objects ready for alignment
        """
        segments = cls.read(transcript_path, include_events=True, include_sections=False)

        # Filter to dialogue and event segments with timestamps (either start or end)
        dialogue_segments = [
            s
            for s in segments
            if s.segment_type in ("dialogue", "event") and (s.timestamp is not None or s.end_timestamp is not None)
        ]

        if not dialogue_segments:
            raise ValueError(f"No dialogue segments with timestamps found in {transcript_path}")

        # Sort by timestamp (use start time if available, otherwise end time)
        dialogue_segments.sort(key=lambda x: x.timestamp if x.timestamp is not None else x.end_timestamp)

        # Convert to Supervision objects
        supervisions: List[Supervision] = []
        prev_end_time = 0.0

        for i, segment in enumerate(dialogue_segments):
            seg_start = None
            seg_end = None

            # Determine start and end times based on available timestamps
            if segment.timestamp is not None:
                # Has start time
                seg_start = segment.timestamp
                if segment.end_timestamp is not None:
                    # Has both start and end
                    seg_end = segment.end_timestamp
                else:
                    # Only has start, estimate end
                    if i < len(dialogue_segments) - 1:
                        # Use next segment's time
                        next_seg = dialogue_segments[i + 1]
                        if next_seg.timestamp is not None:
                            seg_end = next_seg.timestamp
                        elif next_seg.end_timestamp is not None:
                            # Next has only end, estimate its start and use that
                            words_next = len(next_seg.text.split())
                            estimated_duration_next = words_next * 0.3
                            seg_end = next_seg.end_timestamp - estimated_duration_next

                    if seg_end is None:
                        # Estimate based on text length
                        words = len(segment.text.split())
                        seg_end = seg_start + words * 0.3

            elif segment.end_timestamp is not None:
                # Only has end time, need to infer start
                seg_end = segment.end_timestamp
                # Use previous segment's end time as start, or estimate based on text
                if prev_end_time > 0:
                    seg_start = prev_end_time
                else:
                    # Estimate start based on text length
                    words = len(segment.text.split())
                    estimated_duration = words * 0.3
                    seg_start = seg_end - estimated_duration

            if seg_start is not None and seg_end is not None:
                duration = max(seg_end - seg_start, min_duration)
                if segment.segment_type == "dialogue":
                    supervisions.append(
                        Supervision(
                            text=segment.text,
                            start=seg_start,
                            duration=duration,
                            id=f"segment_{i:05d}",
                            speaker=segment.speaker,
                        )
                    )
                prev_end_time = seg_start + duration

        # Optionally merge consecutive segments from same speaker
        if merge_consecutive:
            merged = []
            current_speaker = None
            current_texts = []
            current_start = None
            last_end_time = None

            for i, (segment, sup) in enumerate(zip(dialogue_segments, supervisions)):
                # Check if we should merge with previous segment
                should_merge = False
                if segment.speaker == current_speaker and current_start is not None:
                    # Same speaker - check time gap
                    time_gap = sup.start - last_end_time if last_end_time else 0
                    if time_gap <= merge_max_gap:
                        should_merge = True

                if should_merge:
                    # Same speaker within time threshold, accumulate
                    current_texts.append(segment.text)
                    last_end_time = sup.start + sup.duration
                else:
                    # Different speaker or gap too large, save previous segment
                    if current_texts:
                        merged_text = " ".join(current_texts)
                        merged.append(
                            Supervision(
                                text=merged_text,
                                start=current_start,
                                duration=last_end_time - current_start,
                                id=f"merged_{len(merged):05d}",
                            )
                        )
                    current_speaker = segment.speaker
                    current_texts = [segment.text]
                    current_start = sup.start
                    last_end_time = sup.start + sup.duration

            # Add final segment
            if current_texts:
                merged_text = " ".join(current_texts)
                merged.append(
                    Supervision(
                        text=merged_text,
                        start=current_start,
                        duration=last_end_time - current_start,
                        id=f"merged_{len(merged):05d}",
                    )
                )

            supervisions = merged

        return supervisions


__all__ = ["GeminiReader", "GeminiSegment"]
