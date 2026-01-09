"""YouTube VTT format handler with word-level timestamps.

YouTube auto-generated captions use a specific WebVTT variant with
word-level tags like: Word1<00:00:10.559><c> Word2</c>
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike

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
    def extract_metadata(cls, source: Union[Pathlike, str], **kwargs) -> Dict[str, str]:
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

        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            ts_match = timestamp_pattern.search(line)
            if ts_match:
                cue_start = parse_timestamp(ts_match.group(1))
                cue_end = parse_timestamp(ts_match.group(2))

                cue_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not timestamp_pattern.search(lines[i]):
                    cue_lines.append(lines[i])
                    i += 1

                for cue_line in cue_lines:
                    cue_line = cue_line.strip()
                    if not cue_line:
                        continue

                    # Check for word-level timestamps
                    word_matches = word_timestamp_pattern.findall(cue_line)
                    if word_matches:
                        word_alignments = []

                        # Get first word
                        first_match = first_word_pattern.match(cue_line)
                        if first_match:
                            first_word = first_match.group(1).strip()
                            first_word_next_ts = parse_timestamp(first_match.group(2))
                            if first_word:
                                word_alignments.append(
                                    AlignmentItem(
                                        symbol=first_word,
                                        start=cue_start,
                                        duration=max(0.01, first_word_next_ts - cue_start),
                                    )
                                )

                        # Process remaining words
                        for idx, (ts, word) in enumerate(word_matches):
                            word_start = parse_timestamp(ts)
                            word = word.strip()
                            if not word:
                                continue

                            if idx + 1 < len(word_matches):
                                next_ts = parse_timestamp(word_matches[idx + 1][0])
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

                        if word_alignments:
                            full_text = " ".join(item.symbol for item in word_alignments)
                            if normalize_text:
                                full_text = normalize_text_fn(full_text)

                            sup_start = word_alignments[0].start
                            sup_end = word_alignments[-1].start + word_alignments[-1].duration

                            supervisions.append(
                                Supervision(
                                    text=full_text,
                                    start=sup_start,
                                    duration=max(0.0, sup_end - sup_start),
                                    alignment={"word": word_alignments},
                                )
                            )
                continue
            i += 1

        return cls._merge_supervisions(supervisions)

    @classmethod
    def _merge_supervisions(cls, supervisions: List[Supervision]) -> List[Supervision]:
        """Merge consecutive YouTube VTT supervisions into complete utterances."""
        if not supervisions:
            return supervisions

        merged = []
        current = supervisions[0]

        for next_sup in supervisions[1:]:
            gap = next_sup.start - (current.start + current.duration)

            if gap < 0.5 and current.alignment and next_sup.alignment:
                current_words = current.alignment.get("word", [])
                next_words = next_sup.alignment.get("word", [])
                merged_words = list(current_words) + list(next_words)

                merged_text = current.text + " " + next_sup.text
                merged_end = next_sup.start + next_sup.duration

                current = Supervision(
                    text=merged_text,
                    start=current.start,
                    duration=max(0.0, merged_end - current.start),
                    alignment={"word": merged_words},
                )
            else:
                merged.append(current)
                current = next_sup

        merged.append(current)
        return merged

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ) -> Path:
        """Write YouTube VTT format."""
        content = cls.to_bytes(supervisions, include_speaker=include_speaker, **kwargs)
        output_path = Path(str(output_path))
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to YouTube VTT format bytes."""

        def format_timestamp(seconds: float) -> str:
            """Format seconds into HH:MM:SS.mmm."""
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int(round((seconds % 1) * 1000))
            if ms == 1000:
                s += 1
                ms = 0
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        lines = ["WEBVTT", ""]

        for sup in sorted(supervisions, key=lambda x: x.start):
            lines.append(f"{format_timestamp(sup.start)} --> {format_timestamp(sup.end)}")

            text = sup.text or ""
            alignment = getattr(sup, "alignment", None)
            words = alignment.get("word") if alignment else None

            if words:
                text_parts = []
                for i, word in enumerate(words):
                    symbol = word.symbol
                    if i == 0 and include_speaker and sup.speaker:
                        symbol = f"{sup.speaker}: {symbol}"
                    text_parts.append(f"<{format_timestamp(word.start)}><c> {symbol}</c>")
                lines.append("".join(text_parts))
            else:
                if include_speaker and sup.speaker:
                    text = f"{sup.speaker}: {text}"
                lines.append(text)
            lines.append("")

        return "\n".join(lines).encode("utf-8")
