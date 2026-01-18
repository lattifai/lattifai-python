"""WebVTT format with YouTube VTT word-level timestamp support.

This module provides a unified VTT format handler that:
- Reads both standard VTT and YouTube VTT (with word-level timestamps)
- Writes standard VTT or YouTube VTT (when karaoke_config.enabled=True)

YouTube VTT format uses word-level tags like:
    Word1<00:00:10.559><c> Word2</c><00:00:11.000><c> Word3</c>
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pysubs2
from lhotse.supervision import AlignmentItem

from ...config.caption import KaraokeConfig
from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("vtt")
class VTTFormat(FormatHandler):
    """WebVTT format with YouTube VTT word-level timestamp support.

    Reading:
        - Auto-detects YouTube VTT format (with word-level timestamps)
        - Falls back to standard VTT parsing via pysubs2

    Writing:
        - Standard VTT by default
        - YouTube VTT style when word_level=True and karaoke_config.enabled=True
    """

    extensions = [".vtt"]
    description = "Web Video Text Tracks - HTML5 standard with YouTube VTT support"

    # Pattern to detect YouTube VTT word-level timestamps
    YOUTUBE_VTT_PATTERN = re.compile(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}><c>")

    @classmethod
    def can_read(cls, source) -> bool:
        """Check if source is a VTT file."""
        if cls.is_content(source):
            return source.strip().startswith("WEBVTT")
        try:
            path_str = str(source).lower()
            return path_str.endswith(".vtt")
        except Exception:
            return False

    @classmethod
    def _is_youtube_vtt(cls, content: str) -> bool:
        """Check if content is YouTube VTT format with word-level timestamps."""
        return bool(cls.YOUTUBE_VTT_PATTERN.search(content))

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read VTT format, auto-detecting YouTube VTT word-level timestamps.

        Args:
            source: File path or content string
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        if cls.is_content(source):
            content = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

        # Auto-detect YouTube VTT format
        if cls._is_youtube_vtt(content):
            return cls._read_youtube_vtt(content, normalize_text)
        else:
            return cls._read_standard_vtt(source if not cls.is_content(source) else content, normalize_text)

    @classmethod
    def _read_standard_vtt(cls, source, normalize_text: bool = True) -> List[Supervision]:
        """Read standard VTT using pysubs2."""
        try:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source, format_="vtt")
            else:
                subs = pysubs2.load(str(source), encoding="utf-8", format_="vtt")
        except Exception:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source)
            else:
                subs = pysubs2.load(str(source), encoding="utf-8")

        supervisions = []
        for event in subs.events:
            text = event.text
            if normalize_text:
                text = normalize_text_fn(text)

            speaker, text = parse_speaker_text(text)

            supervisions.append(
                Supervision(
                    text=text,
                    speaker=speaker or event.name or None,
                    start=event.start / 1000.0 if event.start is not None else 0,
                    duration=(event.end - event.start) / 1000.0 if event.end is not None else 0,
                )
            )

        return supervisions

    @classmethod
    def _read_youtube_vtt(cls, content: str, normalize_text: bool = True) -> List[Supervision]:
        """Parse YouTube VTT format with word-level timestamps."""
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
                while i < len(lines):
                    if timestamp_pattern.search(lines[i]):
                        break
                    stripped = lines[i].strip()
                    if not stripped and cue_lines and not lines[i - 1].strip():
                        break
                    if stripped:
                        cue_lines.append(lines[i])
                    i += 1

                all_cues.append({"start": cue_start, "end": cue_end, "lines": cue_lines})
                continue
            i += 1

        # Second pass: identify cues to skip and merge
        cues_to_skip = set()
        cues_to_merge_text = {}

        for idx in range(len(all_cues) - 1):
            cue = all_cues[idx]
            duration = cue["end"] - cue["start"]

            if abs(duration - 0.010) < 0.001:
                cue_text = "\n".join(cue["lines"])
                if not has_word_timestamps(cue_text):
                    next_cue = all_cues[idx + 1]
                    if abs(next_cue["start"] - cue["end"]) < 0.001:
                        cues_to_skip.add(idx)

                        next_cue_text = "\n".join(next_cue["lines"])
                        if not has_word_timestamps(next_cue_text):
                            for prev_idx in range(idx - 1, -1, -1):
                                if prev_idx not in cues_to_skip:
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

            word_alignments = []
            text_parts = []

            for cue_line in cue_lines:
                cue_line = cue_line.strip()
                if not cue_line:
                    continue

                word_matches = word_timestamp_pattern.findall(cue_line)
                first_match = first_word_pattern.match(cue_line)

                if word_matches or first_match:
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

            if not text_parts:
                continue

            full_text = " ".join(text_parts)
            if idx in cues_to_merge_text:
                full_text += " " + cues_to_merge_text[idx]

            if normalize_text:
                full_text = normalize_text_fn(full_text)

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

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict[str, str]:
        """Extract metadata from VTT header."""
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
            elif line.startswith("NOTE"):
                match = re.search(r"NOTE\s+(\w+):\s*(.+)", line)
                if match:
                    key, value = match.groups()
                    metadata[key.lower()] = value.strip()

        return metadata

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        **kwargs,
    ) -> Path:
        """Write VTT to file."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        fps: float = 25.0,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Convert to VTT bytes with optional karaoke and metadata preservation.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker in output
            fps: Frames per second (not used for VTT)
            word_level: If True and alignment exists, output word-per-segment or karaoke
            karaoke_config: Karaoke configuration. When enabled, output YouTube VTT
                style with word-level timestamps: <00:00:10.559><c> word</c>
            metadata: Optional metadata dict containing kind and language

        Returns:
            VTT content as bytes
        """
        from .base import expand_to_word_supervisions

        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # If karaoke enabled, output YouTube VTT style
        if word_level and karaoke_enabled:
            return cls._to_youtube_vtt_bytes(supervisions, include_speaker, metadata)

        # If word_level only (no karaoke), expand to word-per-segment
        if word_level:
            supervisions = expand_to_word_supervisions(supervisions)

        # Build VTT with metadata header
        return cls._to_vtt_bytes_with_metadata(supervisions, include_speaker, metadata)

    @classmethod
    def _to_vtt_bytes_with_metadata(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        metadata: Optional[Dict] = None,
    ) -> bytes:
        """Generate VTT with metadata header."""
        lines = ["WEBVTT"]

        if metadata:
            if metadata.get("kind"):
                lines.append(f"Kind: {metadata['kind']}")
            if metadata.get("language"):
                lines.append(f"Language: {metadata['language']}")

        lines.append("")

        subs = pysubs2.SSAFile()
        for sup in supervisions:
            text = sup.text or ""
            if cls._should_include_speaker(sup, include_speaker):
                text = f"{sup.speaker} {text}"
            subs.append(
                pysubs2.SSAEvent(
                    start=int(sup.start * 1000),
                    end=int(sup.end * 1000),
                    text=text,
                    name=sup.speaker or "",
                )
            )

        vtt_content = subs.to_string(format_="vtt")
        vtt_lines = vtt_content.split("\n")
        started = False
        for line in vtt_lines[1:]:
            if not started and not line.strip():
                continue
            started = True
            lines.append(line)

        return "\n".join(lines).encode("utf-8")

    @classmethod
    def _to_youtube_vtt_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        metadata: Optional[Dict] = None,
    ) -> bytes:
        """Generate YouTube VTT format with word-level timestamps.

        Format: <00:00:10.559><c> word</c>
        """

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

        lines = ["WEBVTT"]

        if metadata:
            if metadata.get("kind"):
                lines.append(f"Kind: {metadata['kind']}")
            if metadata.get("language"):
                lines.append(f"Language: {metadata['language']}")

        lines.append("")

        for sup in sorted(supervisions, key=lambda x: x.start):
            text = sup.text or ""
            alignment = getattr(sup, "alignment", None)
            words = alignment.get("word") if alignment else None

            if words:
                cue_start = words[0].start
                cue_end = words[-1].end
                lines.append(f"{format_timestamp(cue_start)} --> {format_timestamp(cue_end)}")

                text_parts = []
                for i, word in enumerate(words):
                    symbol = word.symbol
                    if i == 0 and include_speaker and sup.speaker:
                        symbol = f"{sup.speaker}: {symbol}"
                    text_parts.append(f"<{format_timestamp(word.start)}><c> {symbol}</c>")
                lines.append("".join(text_parts))
            else:
                lines.append(f"{format_timestamp(sup.start)} --> {format_timestamp(sup.end)}")
                if include_speaker and sup.speaker:
                    text = f"{sup.speaker}: {text}"
                lines.append(text)
            lines.append("")

        return "\n".join(lines).encode("utf-8")
