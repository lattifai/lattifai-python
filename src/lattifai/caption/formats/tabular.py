"""Tabular and plain text format handlers.

Handles: CSV, TSV, AUD (Audacity labels), TXT, JSON
"""

import csv
import json
from io import StringIO
from pathlib import Path
from typing import List

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text, parse_timestamp_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("csv")
class CSVFormat(FormatHandler):
    """CSV (Comma-Separated Values) format.

    Format: speaker,start,end,text (with header)
    Times are in milliseconds.
    """

    extensions = [".csv"]
    description = "CSV - tabular subtitle format"

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read CSV format."""
        if cls.is_content(source):
            lines = list(csv.reader(StringIO(source)))
        else:
            with open(source, "r", encoding="utf-8", newline="") as f:
                lines = list(csv.reader(f))

        if not lines:
            return []

        # Check for header
        first_line = [col.strip().lower() for col in lines[0]]
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker = "speaker" in first_line

        supervisions = []
        start_idx = 1 if has_header else 0

        for parts in lines[start_idx:]:
            if len(parts) < 3:
                continue
            try:
                if has_speaker and len(parts) >= 4:
                    speaker = parts[0].strip() or None
                    start = float(parts[1]) / 1000.0
                    end = float(parts[2]) / 1000.0
                    text = ",".join(parts[3:]).strip()
                else:
                    start = float(parts[0]) / 1000.0
                    end = float(parts[1]) / 1000.0
                    text = ",".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        return supervisions

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, include_speaker: bool = True, **kwargs) -> Path:
        """Write CSV format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], include_speaker: bool = True, **kwargs) -> bytes:
        """Convert to CSV format bytes."""
        output = StringIO()
        writer = csv.writer(output)

        if include_speaker:
            writer.writerow(["speaker", "start", "end", "text"])
            for sup in supervisions:
                if cls._should_include_speaker(sup, include_speaker):
                    text = f"{sup.speaker} {sup.text.strip()}"
                else:
                    text = sup.text.strip()
                writer.writerow([sup.speaker or "", round(1000 * sup.start), round(1000 * sup.end), text])
        else:
            writer.writerow(["start", "end", "text"])
            for sup in supervisions:
                writer.writerow([round(1000 * sup.start), round(1000 * sup.end), sup.text.strip()])

        return output.getvalue().encode("utf-8")


@register_format("tsv")
class TSVFormat(FormatHandler):
    """TSV (Tab-Separated Values) format.

    Format: speaker\tstart\tend\ttext (with header)
    Times are in milliseconds.
    """

    extensions = [".tsv"]
    description = "TSV - tab-separated subtitle format"

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read TSV format."""
        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = f.readlines()

        if not lines:
            return []

        first_line = lines[0].strip().lower()
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker = "speaker" in first_line

        supervisions = []
        start_idx = 1 if has_header else 0

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                if has_speaker and len(parts) >= 4:
                    speaker = parts[0].strip() or None
                    start = float(parts[1]) / 1000.0
                    end = float(parts[2]) / 1000.0
                    text = "\t".join(parts[3:]).strip()
                else:
                    start = float(parts[0]) / 1000.0
                    end = float(parts[1]) / 1000.0
                    text = "\t".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        return supervisions

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, include_speaker: bool = True, **kwargs) -> Path:
        """Write TSV format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], include_speaker: bool = True, **kwargs) -> bytes:
        """Convert to TSV format bytes."""
        lines = []
        if include_speaker:
            lines.append("speaker\tstart\tend\ttext")
            for sup in supervisions:
                speaker = sup.speaker if cls._should_include_speaker(sup, include_speaker) else ""
                text = sup.text.strip().replace("\t", " ")
                lines.append(f"{speaker}\t{round(1000 * sup.start)}\t{round(1000 * sup.end)}\t{text}")
        else:
            lines.append("start\tend\ttext")
            for sup in supervisions:
                text = sup.text.strip().replace("\t", " ")
                lines.append(f"{round(1000 * sup.start)}\t{round(1000 * sup.end)}\t{text}")

        return "\n".join(lines).encode("utf-8")


@register_format("aud")
class AUDFormat(FormatHandler):
    """Audacity Labels format.

    Format: start\tend\t[[speaker]]text
    Times are in seconds.
    """

    extensions = [".aud", ".txt"]
    description = "Audacity Labels format"

    @classmethod
    def can_read(cls, path) -> bool:
        """Only handle .aud extension for reading."""
        return str(path).lower().endswith(".aud")

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read AUD format."""
        import re

        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = f.readlines()

        supervisions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
                text = "\t".join(parts[2:]).strip()

                # Extract speaker from [[speaker]] prefix
                speaker = None
                speaker_match = re.match(r"^\[\[([^\]]+)\]\]\s*(.*)$", text)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        return supervisions

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, include_speaker: bool = True, **kwargs) -> Path:
        """Write AUD format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], include_speaker: bool = True, **kwargs) -> bytes:
        """Convert to AUD format bytes."""
        lines = []
        for sup in supervisions:
            text = sup.text.strip().replace("\t", " ")
            if cls._should_include_speaker(sup, include_speaker):
                text = f"{sup.speaker} {text}"
            lines.append(f"{sup.start}\t{sup.end}\t{text}")

        return "\n".join(lines).encode("utf-8")


@register_format("txt")
class TXTFormat(FormatHandler):
    """Plain text format with optional timestamps.

    Format: [start-end] text or [start-end] [speaker]: text
    """

    extensions = [".txt"]
    description = "Plain text with optional timestamps"

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read TXT format."""
        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]

        if normalize_text:
            lines = [normalize_text_fn(line) for line in lines]

        supervisions = []
        for line in lines:
            if not line:
                continue

            start, end, remaining_text = parse_timestamp_text(line)
            if start is not None and end is not None:
                speaker, text = parse_speaker_text(remaining_text)
                supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            else:
                speaker, text = parse_speaker_text(line)
                supervisions.append(Supervision(text=text, speaker=speaker))

        return supervisions

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, include_speaker: bool = True, **kwargs) -> Path:
        """Write TXT format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], include_speaker: bool = True, **kwargs) -> bytes:
        """Convert to TXT format bytes."""
        lines = []
        for sup in supervisions:
            text = sup.text or ""
            if cls._should_include_speaker(sup, include_speaker):
                text = f"{sup.speaker} {text}"
            lines.append(f"[{sup.start:.2f}-{sup.end:.2f}] {text}")

        return "\n".join(lines).encode("utf-8")


# JSON format moved to json.py for better organization
# Import here for backwards compatibility
from .json import JSONFormat  # noqa: F401
