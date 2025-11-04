"""Writer for YouTube transcript files with corrected timestamps from alignment."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from lhotse.utils import Pathlike

from .gemini_reader import GeminiReader, GeminiSegment
from .supervision import Supervision


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


__all__ = ["GeminiWriter"]
