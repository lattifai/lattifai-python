"""Gemini/YouTube transcript format handler.

Handles YouTube/Gemini markdown transcript format with timestamps like [HH:MM:SS].
Supports reading and writing transcript files with speaker labels, events, and sections.
"""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from lhotse.utils import Pathlike

from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


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
    # Standalone timestamp on its own line
    STANDALONE_TIMESTAMP_PATTERN = re.compile(r"^\[(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2}))\]$")

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
        transcript_path: Union[Pathlike, str],
        include_events: bool = False,
        include_sections: bool = False,
    ) -> List[GeminiSegment]:
        """Parse YouTube transcript file or content and return list of transcript segments.

        Args:
                transcript_path: Path to the transcript file or raw string content
                include_events: Whether to include event descriptions like [Applause]
                include_sections: Whether to include section headers

        Returns:
                List of GeminiSegment objects with all metadata
        """
        content = ""
        # Check if transcript_path is a multi-line string (content) or a short string (likely path)
        is_content = "\n" in str(transcript_path) or len(str(transcript_path)) > 1000

        if is_content:
            content = str(transcript_path)
        else:
            p = Path(transcript_path).expanduser().resolve()
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                # Fallback: treat as content if path doesn't exist
                content = str(transcript_path)

        segments: List[GeminiSegment] = []
        current_section = None
        current_speaker = None

        lines = content.splitlines()
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

            # Parse YouTube format section headers
            youtube_section_match = cls.YOUTUBE_SECTION_PATTERN.match(line)
            if youtube_section_match:
                minutes, seconds, url_seconds, section_title = youtube_section_match.groups()
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

            # Parse standalone timestamp [HH:MM:SS]
            # Often used as an end timestamp for the preceding block
            standalone_match = cls.STANDALONE_TIMESTAMP_PATTERN.match(line)
            if standalone_match:
                groups = standalone_match.groups()
                if groups[0] is not None:
                    ts = cls.parse_timestamp(groups[0], groups[1], groups[2])
                else:
                    ts = cls.parse_timestamp(groups[3], groups[4])

                # Assign to previous dialogue segment if it doesn't have an end time
                if segments and segments[-1].segment_type == "dialogue":
                    if segments[-1].end_timestamp is None:
                        segments[-1].end_timestamp = ts
                    elif segments[-1].timestamp is None:
                        # If it has an end but no start, this standalone might be its start?
                        # Usually standalone is end, but let's be flexible
                        segments[-1].timestamp = ts
                continue

            # Parse event descriptions [event] [HH:MM:SS]
            event_match = cls.EVENT_PATTERN.match(line)
            if event_match:
                groups = event_match.groups()
                event_text = groups[0]
                hours_or_minutes = groups[1]
                minutes_or_seconds = groups[2]
                seconds_optional = groups[3]

                if seconds_optional is not None:
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, seconds_optional)
                else:
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

            # Parse speaker dialogue: **Speaker:** Text [HH:MM:SS]
            speaker_match = cls.SPEAKER_PATTERN.match(line)
            if speaker_match:
                speaker, text_with_timestamp = speaker_match.groups()
                current_speaker = speaker.strip()

                start_match = cls.INLINE_TIMESTAMP_START_PATTERN.match(text_with_timestamp.strip())
                end_match = cls.INLINE_TIMESTAMP_END_PATTERN.match(text_with_timestamp.strip())
                youtube_match = cls.YOUTUBE_INLINE_PATTERN.match(text_with_timestamp.strip())

                start_timestamp = None
                end_timestamp = None
                text = text_with_timestamp.strip()

                if start_match:
                    groups = start_match.groups()
                    if groups[0] is not None:
                        start_timestamp = cls.parse_timestamp(groups[0], groups[1], groups[2])
                    elif groups[3] is not None:
                        start_timestamp = cls.parse_timestamp(groups[3], groups[4])
                    text = groups[5]
                elif end_match:
                    groups = end_match.groups()
                    text = groups[0]
                    if groups[1] is not None:
                        end_timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                    elif groups[4] is not None:
                        end_timestamp = cls.parse_timestamp(groups[4], groups[5])
                elif youtube_match:
                    groups = youtube_match.groups()
                    text = groups[0]
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
                current_speaker = None
                continue

            # Parse plain text (might contain inline timestamp or be a continuation)
            start_match = cls.INLINE_TIMESTAMP_START_PATTERN.match(line)
            end_match = cls.INLINE_TIMESTAMP_END_PATTERN.match(line)
            youtube_inline_match = cls.YOUTUBE_INLINE_PATTERN.match(line)

            if start_match:
                groups = start_match.groups()
                if groups[0] is not None:
                    start_timestamp = cls.parse_timestamp(groups[0], groups[1], groups[2])
                else:
                    start_timestamp = cls.parse_timestamp(groups[3], groups[4])
                text = groups[5]
                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        timestamp=start_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
            elif end_match:
                groups = end_match.groups()
                text = groups[0]
                if groups[1] is not None:
                    end_timestamp = cls.parse_timestamp(groups[1], groups[2], groups[3])
                else:
                    end_timestamp = cls.parse_timestamp(groups[4], groups[5])
                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
            elif youtube_inline_match:
                groups = youtube_inline_match.groups()
                text = groups[0]
                url_seconds = groups[3]
                segments.append(
                    GeminiSegment(
                        text=text.strip(),
                        end_timestamp=cls.parse_timestamp(url_seconds),
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
            else:
                # Plain text without any recognized markers
                # If it follows a speaker line or another dialogue line without end timestamp,
                # merge it into the last segment to support multi-line text blocks.
                if segments and segments[-1].segment_type == "dialogue" and segments[-1].end_timestamp is None:
                    segments[-1].text += " " + line.strip()
                else:
                    # Skip markdown headers and other formatting
                    if line.startswith("#"):
                        continue

                    segments.append(
                        GeminiSegment(
                            text=line.strip(),
                            speaker=current_speaker,
                            section=current_section,
                            segment_type="dialogue",
                            line_number=line_num,
                        )
                    )

        return segments

    @classmethod
    def extract_for_alignment(
        cls,
        transcript_path: Pathlike,
        merge_consecutive: bool = False,
        min_duration: float = 0.1,
        merge_max_gap: float = 2.0,
        normalize_text: bool = True,
        **kwargs,
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
                            text=segment.text.strip(),
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


class GeminiWriter:
    """Writer for updating YouTube transcript timestamps based on alignment results."""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to [HH:MM:SS] format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    @classmethod
    def update_timestamps(
        cls,
        original_transcript: Pathlike,
        aligned_supervisions: List[Supervision],
        output_path: Pathlike,
        timestamp_mapping: Optional[Dict[int, float]] = None,
    ) -> Pathlike:
        """Update transcript file with corrected timestamps from alignment.

        Args:
                original_transcript: Path to the original transcript file
                aligned_supervisions: List of aligned Supervision objects with corrected timestamps
                output_path: Path to write the updated transcript
                timestamp_mapping: Optional manual mapping from line_number to new timestamp

        Returns:
                Path to the output file
        """
        original_path = Path(original_transcript)
        output_path = Path(output_path)

        # Read original file
        with open(original_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse original segments to get line numbers
        original_segments = GeminiReader.read(original_transcript, include_events=True, include_sections=True)

        # Create mapping from line number to new timestamp
        if timestamp_mapping is None:
            timestamp_mapping = cls._create_timestamp_mapping(original_segments, aligned_supervisions)

        # Update timestamps in lines
        updated_lines = []
        for line_num, line in enumerate(lines, start=1):
            if line_num in timestamp_mapping:
                new_timestamp = timestamp_mapping[line_num]
                updated_line = cls._replace_timestamp(line, new_timestamp)
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

        # Write updated content
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        return output_path

    @classmethod
    def _create_timestamp_mapping(
        cls, original_segments: List[GeminiSegment], aligned_supervisions: List[Supervision]
    ) -> Dict[int, float]:
        """Create mapping from line numbers to new timestamps based on alignment.

        This performs text matching between original segments and aligned supervisions
        to determine which timestamps should be updated.
        """
        mapping = {}

        # Create a simple text-based matching
        dialogue_segments = [s for s in original_segments if s.segment_type == "dialogue"]

        # Try to match based on text content
        for aligned_sup in aligned_supervisions:
            aligned_text = aligned_sup.text.strip()

            # Find best matching original segment
            best_match = None
            best_score = 0

            for orig_seg in dialogue_segments:
                orig_text = orig_seg.text.strip()

                # Simple text similarity (could be improved with fuzzy matching)
                if aligned_text == orig_text:
                    best_match = orig_seg
                    best_score = 1.0
                    break
                elif aligned_text in orig_text or orig_text in aligned_text:
                    score = min(len(aligned_text), len(orig_text)) / max(len(aligned_text), len(orig_text))
                    if score > best_score:
                        best_score = score
                        best_match = orig_seg

            # If we found a good match, update the mapping
            if best_match and best_score > 0.8:
                mapping[best_match.line_number] = aligned_sup.start

        return mapping

    @classmethod
    def _replace_timestamp(cls, line: str, new_timestamp: float) -> str:
        """Replace timestamp in a line with new timestamp."""
        new_ts_str = cls.format_timestamp(new_timestamp)

        # Replace timestamp patterns
        # Pattern 1: [HH:MM:SS] at the end or in brackets
        line = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", new_ts_str, line)

        return line

    @classmethod
    def write_aligned_transcript(
        cls,
        aligned_supervisions: List[Supervision],
        output_path: Pathlike,
        include_word_timestamps: bool = False,
    ) -> Pathlike:
        """Write a new transcript file from aligned supervisions.

        This creates a simplified transcript format with accurate timestamps.

        Args:
                aligned_supervisions: List of aligned Supervision objects
                output_path: Path to write the transcript
                include_word_timestamps: Whether to include word-level timestamps if available

        Returns:
                Path to the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Aligned Transcript\n\n")

            for i, sup in enumerate(aligned_supervisions):
                # Write segment with timestamp
                start_ts = cls.format_timestamp(sup.start)
                f.write(f"{start_ts} {sup.text}\n")

                # Optionally write word-level timestamps
                if include_word_timestamps and hasattr(sup, "alignment") and sup.alignment:
                    if "word" in sup.alignment:
                        f.write("  Words: ")
                        word_parts = []
                        for word_info in sup.alignment["word"]:
                            word_ts = cls.format_timestamp(word_info["start"])
                            word_parts.append(f'{word_info["symbol"]}{word_ts}')
                        f.write(" ".join(word_parts))
                        f.write("\n")

                f.write("\n")

        return output_path

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        **kwargs,
    ) -> Path:
        """Alias for write_aligned_transcript for Caption API compatibility."""
        return Path(cls.write_aligned_transcript(supervisions, output_path, **kwargs))

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        **kwargs,
    ) -> bytes:
        """Convert aligned supervisions to Gemini format bytes."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cls.write_aligned_transcript(supervisions, tmp_path, **kwargs)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)


__all__ = ["GeminiWriter"]


@register_format("gemini")
class GeminiFormat(FormatHandler):
    """YouTube/Gemini markdown transcript format."""

    extensions = [".md"]
    description = "YouTube/Gemini transcript format with timestamps"

    @classmethod
    def can_read(cls, path) -> bool:
        """Check if this is a Gemini format file."""
        path_str = str(path).lower()
        return (
            path_str.endswith("gemini.md")
            or path_str.endswith("gemini3.md")
            or ("gemini" in path_str and path_str.endswith(".md"))
        )

    @classmethod
    def read(cls, path: Pathlike, **kwargs) -> List[Supervision]:
        """Read Gemini format file."""
        return GeminiReader.extract_for_alignment(path, **kwargs)

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        **kwargs,
    ) -> Path:
        """Write Gemini format file."""
        return GeminiWriter.write(supervisions, output_path, **kwargs)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        **kwargs,
    ) -> bytes:
        """Convert to Gemini format bytes."""
        return GeminiWriter.to_bytes(supervisions, **kwargs)
