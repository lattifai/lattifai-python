"""Utility functions for caption processing.

This module provides utility functions for:
- Timecode offset handling (for professional timelines starting at 01:00:00:00)
- Overlap/collision resolution (merge or trim modes)
- SRT format optimization (UTF-8 BOM, comma-separated milliseconds)
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .supervision import Supervision


class CollisionMode(Enum):
    """Mode for resolving overlapping captions."""

    MERGE = "merge"  # Merge overlapping lines with line break
    TRIM = "trim"  # Trim earlier caption to end before later starts
    KEEP = "keep"  # Keep overlaps as-is (may cause issues in some NLE)


@dataclass
class TimecodeOffset:
    """Configuration for timecode offset.

    Professional timelines often start at 01:00:00:00 instead of 00:00:00:00.
    This class handles the offset conversion.

    Attributes:
        hours: Hour offset (default 0)
        minutes: Minute offset (default 0)
        seconds: Second offset (default 0)
        frames: Frame offset (default 0)
        fps: Frame rate for frame-based offset calculation
    """

    hours: int = 0
    minutes: int = 0
    seconds: float = 0.0
    frames: int = 0
    fps: float = 25.0

    @property
    def total_seconds(self) -> float:
        """Calculate total offset in seconds."""
        return self.hours * 3600 + self.minutes * 60 + self.seconds + (self.frames / self.fps)

    @classmethod
    def from_timecode(cls, timecode: str, fps: float = 25.0) -> "TimecodeOffset":
        """Create offset from timecode string.

        Args:
            timecode: Timecode string (HH:MM:SS:FF or HH:MM:SS.mmm)
            fps: Frame rate

        Returns:
            TimecodeOffset instance
        """
        # Handle different separators
        if ";" in timecode:
            # Drop-frame format
            parts = timecode.replace(";", ":").split(":")
        else:
            parts = timecode.split(":")

        hours = int(parts[0]) if len(parts) > 0 else 0
        minutes = int(parts[1]) if len(parts) > 1 else 0

        # Handle seconds (may have frames or milliseconds)
        if len(parts) > 2:
            sec_part = parts[2]
            if "." in sec_part:
                # Millisecond format
                seconds = float(sec_part)
                frames = 0
            else:
                seconds = float(sec_part)
                frames = int(parts[3]) if len(parts) > 3 else 0
        else:
            seconds = 0.0
            frames = 0

        return cls(hours=hours, minutes=minutes, seconds=seconds, frames=frames, fps=fps)

    @classmethod
    def broadcast_start(cls, fps: float = 25.0) -> "TimecodeOffset":
        """Create standard broadcast start offset (01:00:00:00).

        Args:
            fps: Frame rate

        Returns:
            TimecodeOffset for broadcast start
        """
        return cls(hours=1, fps=fps)


def apply_timecode_offset(
    supervisions: List["Supervision"],
    offset: TimecodeOffset,
) -> List["Supervision"]:
    """Apply timecode offset to all supervisions.

    Args:
        supervisions: List of supervision segments
        offset: Timecode offset to apply

    Returns:
        New list of supervisions with offset applied
    """
    from .supervision import Supervision

    offset_seconds = offset.total_seconds
    result = []

    for sup in supervisions:
        new_sup = Supervision(
            text=sup.text,
            start=sup.start + offset_seconds,
            duration=sup.duration,
            speaker=sup.speaker,
            id=sup.id,
            language=sup.language,
            alignment=deepcopy(getattr(sup, "alignment", None)),
            custom=sup.custom.copy() if sup.custom else None,
        )

        # Also offset word-level alignments if present
        if new_sup.alignment and "word" in new_sup.alignment:
            from lhotse.supervision import AlignmentItem

            new_words = []
            for word in new_sup.alignment["word"]:
                new_words.append(
                    AlignmentItem(
                        symbol=word.symbol,
                        start=word.start + offset_seconds,
                        duration=word.duration,
                        score=word.score,
                    )
                )
            new_sup.alignment["word"] = new_words

        result.append(new_sup)

    return result


def resolve_overlaps(
    supervisions: List["Supervision"],
    mode: CollisionMode = CollisionMode.MERGE,
    gap_threshold: float = 0.05,
) -> List["Supervision"]:
    """Resolve overlapping supervisions.

    Args:
        supervisions: List of supervision segments (should be sorted by start time)
        mode: How to handle overlaps (MERGE, TRIM, or KEEP)
        gap_threshold: Minimum gap between captions in seconds (for TRIM mode)

    Returns:
        New list of supervisions with overlaps resolved
    """
    from .supervision import Supervision

    if not supervisions or mode == CollisionMode.KEEP:
        return supervisions

    # Sort by start time
    sorted_sups = sorted(supervisions, key=lambda x: x.start)
    result = []

    i = 0
    while i < len(sorted_sups):
        current = sorted_sups[i]

        # Find all overlapping supervisions
        overlapping = [current]
        j = i + 1
        while j < len(sorted_sups):
            next_sup = sorted_sups[j]
            # Check if next overlaps with any in our group
            current_end = max(s.end for s in overlapping)
            if next_sup.start < current_end:
                overlapping.append(next_sup)
                j += 1
            else:
                break

        if len(overlapping) == 1:
            # No overlap
            result.append(current)
            i += 1
        elif mode == CollisionMode.MERGE:
            # Merge all overlapping into one
            merged = _merge_supervisions(overlapping)
            result.append(merged)
            i = j
        elif mode == CollisionMode.TRIM:
            # Trim each to not overlap with next
            for k, sup in enumerate(overlapping[:-1]):
                next_sup = overlapping[k + 1]
                # Trim current to end before next starts
                new_duration = max(gap_threshold, next_sup.start - sup.start - gap_threshold)
                trimmed = Supervision(
                    text=sup.text,
                    start=sup.start,
                    duration=min(sup.duration, new_duration),
                    speaker=sup.speaker,
                    id=sup.id,
                    language=sup.language,
                    alignment=sup.alignment,
                    custom=sup.custom,
                )
                result.append(trimmed)
            # Add last one as-is
            result.append(overlapping[-1])
            i = j
        else:
            result.append(current)
            i += 1

    return result


def _merge_supervisions(supervisions: List["Supervision"]) -> "Supervision":
    """Merge multiple overlapping supervisions into one.

    Args:
        supervisions: List of overlapping supervisions

    Returns:
        Single merged supervision
    """
    from .supervision import Supervision

    if not supervisions:
        raise ValueError("Cannot merge empty supervision list")

    if len(supervisions) == 1:
        return supervisions[0]

    # Calculate merged timing
    start = min(s.start for s in supervisions)
    end = max(s.end for s in supervisions)

    # Merge text with line breaks, indicating speakers
    texts = []
    for sup in supervisions:
        text = sup.text.strip() if sup.text else ""
        if sup.speaker:
            texts.append(f"- {sup.speaker}: {text}")
        else:
            texts.append(f"- {text}")

    merged_text = "\n".join(texts)

    # Use first supervision's speaker or None for mixed speakers
    speakers = set(s.speaker for s in supervisions if s.speaker)
    speaker = supervisions[0].speaker if len(speakers) == 1 else None

    return Supervision(
        text=merged_text,
        start=start,
        duration=end - start,
        speaker=speaker,
        id=supervisions[0].id,
        language=supervisions[0].language,
    )


def format_srt_timestamp(seconds: float) -> str:
    """Format timestamp for SRT format.

    SRT uses comma as millisecond separator: HH:MM:SS,mmm

    Args:
        seconds: Time in seconds

    Returns:
        SRT-formatted timestamp string
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_content(
    supervisions: List["Supervision"],
    include_speaker: bool = True,
    use_bom: bool = True,
) -> bytes:
    """Generate SRT content with proper formatting.

    Args:
        supervisions: List of supervision segments
        include_speaker: Include speaker labels in text
        use_bom: Include UTF-8 BOM for Windows compatibility

    Returns:
        SRT content as bytes
    """
    lines = []

    for i, sup in enumerate(supervisions, 1):
        # Sequence number
        lines.append(str(i))

        # Timestamp line with comma separator
        start_ts = format_srt_timestamp(sup.start)
        end_ts = format_srt_timestamp(sup.end)
        lines.append(f"{start_ts} --> {end_ts}")

        # Text content
        text = sup.text.strip() if sup.text else ""
        if include_speaker and sup.speaker:
            # Check if speaker was originally in text
            if not (hasattr(sup, "custom") and sup.custom and not sup.custom.get("original_speaker", True)):
                text = f"{sup.speaker}: {text}"
        lines.append(text)

        # Blank line between entries
        lines.append("")

    content = "\n".join(lines)

    if use_bom:
        # UTF-8 with BOM for Windows compatibility
        return b"\xef\xbb\xbf" + content.encode("utf-8")
    else:
        return content.encode("utf-8")


def detect_overlaps(supervisions: List["Supervision"]) -> List[Tuple[int, int]]:
    """Detect all overlapping supervision pairs.

    Args:
        supervisions: List of supervision segments

    Returns:
        List of tuples (index1, index2) where supervisions overlap
    """
    overlaps = []
    sorted_sups = sorted(enumerate(supervisions), key=lambda x: x[1].start)

    for i in range(len(sorted_sups) - 1):
        idx1, sup1 = sorted_sups[i]
        for j in range(i + 1, len(sorted_sups)):
            idx2, sup2 = sorted_sups[j]
            if sup2.start >= sup1.end:
                break
            overlaps.append((idx1, idx2))

    return overlaps


def split_long_lines(
    supervisions: List["Supervision"],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> List["Supervision"]:
    """Split supervisions with long text into multiple segments.

    Useful for broadcast compliance where line length limits are strict.

    Args:
        supervisions: List of supervision segments
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per supervision

    Returns:
        New list with long supervisions split
    """
    from .supervision import Supervision

    result = []
    max_total_chars = max_chars_per_line * max_lines

    for sup in supervisions:
        text = sup.text.strip() if sup.text else ""

        if len(text) <= max_total_chars:
            # Text fits, just wrap lines if needed
            wrapped = _wrap_text(text, max_chars_per_line, max_lines)
            new_sup = Supervision(
                text=wrapped,
                start=sup.start,
                duration=sup.duration,
                speaker=sup.speaker,
                id=sup.id,
                language=sup.language,
                alignment=sup.alignment,
                custom=sup.custom,
            )
            result.append(new_sup)
        else:
            # Split into multiple supervisions
            chunks = _split_text_chunks(text, max_total_chars)
            chunk_duration = sup.duration / len(chunks)

            for i, chunk in enumerate(chunks):
                wrapped = _wrap_text(chunk, max_chars_per_line, max_lines)
                new_sup = Supervision(
                    text=wrapped,
                    start=sup.start + i * chunk_duration,
                    duration=chunk_duration,
                    speaker=sup.speaker if i == 0 else None,
                    id=f"{sup.id}_{i}" if sup.id else None,
                    language=sup.language,
                )
                result.append(new_sup)

    return result


def _wrap_text(text: str, max_chars: int, max_lines: int) -> str:
    """Wrap text to fit within character and line limits."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + (1 if current_line else 0) <= max_chars:
            current_line.append(word)
            current_length += word_len + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len

            if len(lines) >= max_lines:
                break

    if current_line and len(lines) < max_lines:
        lines.append(" ".join(current_line))

    return "\n".join(lines[:max_lines])


def _split_text_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks that fit within character limit."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(word)
            current_length += word_len + (1 if len(current_chunk) > 1 else 0)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
