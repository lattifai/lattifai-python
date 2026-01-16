"""YouTube VTT format reader with word-level timestamps.

YouTube auto-generated captions use a specific WebVTT variant with
word-level tags like: Word1<00:00:10.559><c> Word2</c>

This module provides READ-ONLY support for parsing YouTube VTT format.
For WRITING YouTube VTT format, use VTTFormat with karaoke=True:

    caption.to_string("vtt", word_level=True, karaoke=True)
"""

import re
from typing import Dict, List, Union

from lhotse.supervision import AlignmentItem

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("youtube_vtt")
class YouTubeVTTFormat(FormatHandler):
    """YouTube-specific WebVTT format with word-level timestamps."""

    extensions = [".vtt"]
    description = "YouTube WebVTT with word-level timestamps"

    @classmethod
    def can_read(cls, source) -> bool:
        """Check if content is YouTube VTT format with word-level timestamps."""
        if cls.is_content(source):
            content = source
        else:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read(4096)  # Just check first part
            except Exception:
                return False

        # Look for pattern like <00:00:10.559><c> word</c>
        return bool(re.search(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}><c>", content))

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict[str, str]:
        """Extract metadata from YouTube VTT."""
        if cls.is_content(source):
            content = source[:4096]
        else:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read(4096)
            except Exception:
                return {}

        metadata = {}
        lines = content.split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("Kind:"):
                metadata["kind"] = line.split(":", 1)[1].strip()
            elif line.startswith("Language:"):
                metadata["language"] = line.split(":", 1)[1].strip()

        return metadata

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Parse YouTube VTT format."""
        if cls.is_content(source):
            content = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

        supervisions = []

        # Pattern to match timestamp lines: 00:00:14.280 --> 00:00:17.269
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")

        # Pattern to match word-level timestamps: <00:00:10.559><c> word</c>
        word_timestamp_pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}[.,]\d{3})><c>\s*([^<]+)</c>")

        # Pattern to match the first word (before first timestamp)
        first_word_pattern = re.compile(r"^([^<\n]+?)<(\d{2}:\d{2}:\d{2}[.,]\d{3})>")

        def parse_timestamp(ts: str) -> float:
            """Convert timestamp string to seconds."""
            ts = ts.replace(",", ".")
            parts = ts.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        def has_word_timestamps(text: str) -> bool:
            """Check if text contains word-level timestamps."""
            return bool(word_timestamp_pattern.search(text) or first_word_pattern.match(text))

        lines = content.split("\n")
        i = 0

        # First pass: collect all cues with their content
        all_cues = []
        while i < len(lines):
            line = lines[i]
            ts_match = timestamp_pattern.search(line)
            if ts_match:
                cue_start = parse_timestamp(ts_match.group(1))
                cue_end = parse_timestamp(ts_match.group(2))

                cue_lines = []
                i += 1
                # Collect all lines until next timestamp or empty line after stripping
                while i < len(lines):
                    if timestamp_pattern.search(lines[i]):
                        break
                    stripped = lines[i].strip()
                    if not stripped and cue_lines and not lines[i - 1].strip():
                        # Two consecutive empty lines mark end of cue
                        break
                    if stripped:  # Only add non-empty lines
                        cue_lines.append(lines[i])
                    i += 1

                all_cues.append({"start": cue_start, "end": cue_end, "lines": cue_lines})
                continue
            i += 1

        # Second pass: identify cues to skip and cues to merge
        # Skip 0.010-duration cues without word timestamps if next cue starts at same time
        # Merge plain text cues that follow a skipped 0.010s cue with the previous cue
        cues_to_skip = set()
        cues_to_merge_text = {}  # Maps cue index to text to append

        for idx in range(len(all_cues) - 1):
            cue = all_cues[idx]
            duration = cue["end"] - cue["start"]

            # Check if duration is 0.010 seconds and has no word timestamps
            if abs(duration - 0.010) < 0.001:
                cue_text = "\n".join(cue["lines"])
                if not has_word_timestamps(cue_text):
                    # Check if next cue starts at same time as this one ends
                    next_cue = all_cues[idx + 1]
                    if abs(next_cue["start"] - cue["end"]) < 0.001:
                        cues_to_skip.add(idx)

                        # Check if next cue has plain text that should be merged
                        next_cue_text = "\n".join(next_cue["lines"])
                        if not has_word_timestamps(next_cue_text):
                            # Find the last non-skipped cue before this one
                            for prev_idx in range(idx - 1, -1, -1):
                                if prev_idx not in cues_to_skip:
                                    # Extract last line from next cue (skip context line)
                                    if len(next_cue["lines"]) > 1:
                                        append_text = next_cue["lines"][-1].strip()
                                        if append_text:
                                            cues_to_merge_text[prev_idx] = append_text
                                    cues_to_skip.add(idx + 1)
                                    break

        # Third pass: process remaining cues
        for idx, cue in enumerate(all_cues):
            if idx in cues_to_skip:
                continue

            cue_start = cue["start"]
            cue_end = cue["end"]
            cue_lines = cue["lines"]

            # Collect text with word timestamps only
            word_alignments = []
            text_parts = []

            for cue_line in cue_lines:
                cue_line = cue_line.strip()
                if not cue_line:
                    continue

                # Check for word-level timestamps
                word_matches = word_timestamp_pattern.findall(cue_line)
                first_match = first_word_pattern.match(cue_line)

                if word_matches or first_match:
                    # Line has word-level timestamps, extract them

                    # Get first word
                    if first_match:
                        first_word = first_match.group(1).strip()
                        first_word_next_ts = parse_timestamp(first_match.group(2))
                        if first_word:
                            text_parts.append(first_word)
                            word_alignments.append(
                                AlignmentItem(
                                    symbol=first_word,
                                    start=cue_start,
                                    duration=max(0.01, first_word_next_ts - cue_start),
                                )
                            )

                    # Process remaining words
                    for word_idx, (ts, word) in enumerate(word_matches):
                        word_start = parse_timestamp(ts)
                        word = word.strip()
                        if not word:
                            continue

                        text_parts.append(word)

                        if word_idx + 1 < len(word_matches):
                            next_ts = parse_timestamp(word_matches[word_idx + 1][0])
                            duration = next_ts - word_start
                        else:
                            duration = cue_end - word_start

                        word_alignments.append(
                            AlignmentItem(
                                symbol=word,
                                start=word_start,
                                duration=max(0.01, duration),
                            )
                        )

            # Skip if no text extracted
            if not text_parts:
                continue

            # Add merged text if applicable
            full_text = " ".join(text_parts)
            if idx in cues_to_merge_text:
                full_text += " " + cues_to_merge_text[idx]

            if normalize_text:
                full_text = normalize_text_fn(full_text)

            # Use word alignment times if available, otherwise use cue times
            if word_alignments:
                sup_start = word_alignments[0].start
                sup_end = word_alignments[-1].start + word_alignments[-1].duration
            else:
                sup_start = cue_start
                sup_end = cue_end

            supervisions.append(
                Supervision(
                    text=full_text,
                    start=sup_start,
                    duration=max(0.0, sup_end - sup_start),
                    alignment={"word": word_alignments} if word_alignments else None,
                )
            )

        return supervisions
