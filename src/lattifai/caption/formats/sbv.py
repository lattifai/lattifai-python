"""SubViewer (SBV) format handler.

SBV is YouTube's native subtitle format with the following structure:
    0:00:00.000,0:00:02.000
    Text line 1

    0:00:02.000,0:00:04.000
    Text line 2
"""

from pathlib import Path
from typing import List

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


@register_format("sbv")
class SBVFormat(FormatHandler):
    """SubViewer (SBV) format - YouTube's native format."""

    extensions = [".sbv"]
    description = "SubViewer - YouTube native subtitle format"

    @classmethod
    def _parse_sbv_timestamp(cls, timestamp: str) -> float:
        """Parse SBV timestamp (H:MM:SS.mmm) to seconds."""
        parts = timestamp.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            s_parts = s.split(".")
            seconds = int(h) * 3600 + int(m) * 60 + int(s_parts[0])
            if len(s_parts) > 1:
                seconds += int(s_parts[1]) / 1000.0
            return seconds
        return 0.0

    @classmethod
    def _format_sbv_timestamp(cls, seconds: float) -> str:
        """Format seconds to SBV timestamp (H:MM:SS.mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h}:{m:02d}:{s:02d}.{ms:03d}"

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read SBV format."""
        # Get content
        if cls.is_content(source):
            content = source
        else:
            content = Path(source).read_text(encoding="utf-8")

        supervisions = []
        entries = content.strip().split("\n\n")

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue

            # First line: timestamp (H:MM:SS.mmm,H:MM:SS.mmm)
            timestamp_line = lines[0].strip()
            text_lines = lines[1:]

            if "," not in timestamp_line:
                continue

            try:
                start_str, end_str = timestamp_line.split(",", 1)
                start = cls._parse_sbv_timestamp(start_str)
                end = cls._parse_sbv_timestamp(end_str)

                text = " ".join(text_lines).strip()
                speaker, text = parse_speaker_text(text)

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(
                        Supervision(
                            text=text,
                            start=start,
                            duration=end - start,
                            speaker=speaker,
                        )
                    )
            except (ValueError, IndexError):
                continue

        return supervisions

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        **kwargs,
    ) -> Path:
        """Write SBV format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert to SBV format bytes."""
        lines = []

        for i, sup in enumerate(supervisions):
            start_time = cls._format_sbv_timestamp(sup.start)
            end_time = cls._format_sbv_timestamp(sup.end)
            lines.append(f"{start_time},{end_time}")

            text = sup.text.strip() if sup.text else ""
            if include_speaker and sup.speaker:
                # Check if speaker should be included
                include_this_speaker = True
                if hasattr(sup, "custom") and sup.custom and not sup.custom.get("original_speaker", True):
                    include_this_speaker = False

                if include_this_speaker:
                    text = f"{sup.speaker}: {text}"
            lines.append(text)

            if i < len(supervisions) - 1:
                lines.append("")

        return "\n".join(lines).encode("utf-8")
