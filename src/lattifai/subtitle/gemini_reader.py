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
    timestamp: Optional[float] = None
    speaker: Optional[str] = None
    section: Optional[str] = None
    segment_type: str = "dialogue"  # 'dialogue', 'event', or 'section_header'
    line_number: int = 0

    @property
    def start(self) -> float:
        """Return start time in seconds."""
        return self.timestamp if self.timestamp is not None else 0.0


class GeminiReader:
    """Parser for YouTube transcript format with speaker labels and timestamps."""

    # Regex patterns for parsing (supports both [HH:MM:SS] and [MM:SS] formats)
    TIMESTAMP_PATTERN = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})\]|\[(\d{1,2}):(\d{2})\]")
    SECTION_HEADER_PATTERN = re.compile(r"^##\s*\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.+)$")
    SPEAKER_PATTERN = re.compile(r"^\*\*(.+?[:：])\*\*\s*(.+)$")
    EVENT_PATTERN = re.compile(r"^\[([^\]]+)\]\s*\[(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2}))\]$")
    INLINE_TIMESTAMP_PATTERN = re.compile(r"^(.+?)\s*\[(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2}))\]$")

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
                # Parse timestamp - can be HH:MM:SS (groups 1,2,3) or MM:SS (groups 4,5)
                if groups[1] is not None:  # HH:MM:SS format
                    timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                elif groups[4] is not None:  # MM:SS format
                    timestamp = cls.parse_timestamp(groups[4], groups[5])
                else:
                    timestamp = None

                if include_events and timestamp is not None:
                    segments.append(
                        GeminiSegment(
                            text=event_text.strip(),
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

                # Extract timestamp from the end of the text
                timestamp_match = cls.INLINE_TIMESTAMP_PATTERN.match(text_with_timestamp.strip())
                youtube_match = cls.YOUTUBE_INLINE_PATTERN.match(text_with_timestamp.strip())

                if timestamp_match:
                    groups = timestamp_match.groups()
                    text = groups[0]
                    # Parse timestamp - can be HH:MM:SS (groups 1,2,3) or MM:SS (groups 4,5)
                    if groups[1] is not None:  # HH:MM:SS format
                        timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                    elif groups[4] is not None:  # MM:SS format
                        timestamp = cls.parse_timestamp(groups[4], groups[5])
                    else:
                        timestamp = None
                elif youtube_match:
                    groups = youtube_match.groups()
                    text = groups[0]
                    # Extract seconds from URL parameter
                    url_seconds = groups[3]
                    timestamp = cls.parse_timestamp(url_seconds)
                else:
                    text = text_with_timestamp.strip()
                    timestamp = None

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                current_speaker = None  # Reset speaker after use
                continue

            # Parse plain text with timestamp at the end
            inline_match = cls.INLINE_TIMESTAMP_PATTERN.match(line)
            youtube_inline_match = cls.YOUTUBE_INLINE_PATTERN.match(line)

            if inline_match:
                groups = inline_match.groups()
                text = groups[0]
                # Parse timestamp - can be HH:MM:SS (groups 1,2,3) or MM:SS (groups 4,5)
                if groups[1] is not None:  # HH:MM:SS format
                    timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                elif groups[4] is not None:  # MM:SS format
                    timestamp = cls.parse_timestamp(groups[4], groups[5])
                else:
                    timestamp = None

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=timestamp,
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
                # Extract seconds from URL parameter
                url_seconds = groups[3]
                timestamp = cls.parse_timestamp(url_seconds)

                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=timestamp,
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
        segments = cls.read(transcript_path, include_events=False, include_sections=False)

        # Filter to only dialogue segments with timestamps
        dialogue_segments = [s for s in segments if s.segment_type == "dialogue" and s.timestamp is not None]

        if not dialogue_segments:
            raise ValueError(f"No dialogue segments with timestamps found in {transcript_path}")

        # Sort by timestamp
        dialogue_segments.sort(key=lambda x: x.timestamp)

        # Convert to Supervision objects
        supervisions: List[Supervision] = []

        for i, segment in enumerate(dialogue_segments):
            # Estimate duration based on next segment
            if i < len(dialogue_segments) - 1:
                duration = dialogue_segments[i + 1].timestamp - segment.timestamp
            else:
                # Last segment: estimate based on text length (rough heuristic)
                words = len(segment.text.split())
                duration = words * 0.3  # ~0.3 seconds per word

            supervisions.append(
                Supervision(
                    text=segment.text,
                    start=segment.timestamp,
                    duration=max(duration, min_duration),
                    id=f"segment_{i:05d}",
                    speaker=segment.speaker,
                )
            )

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
