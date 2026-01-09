"""Avid DS Subtitle format writer for Avid Media Composer integration.

This module provides functionality to export captions in Avid DS format,
which is the native format for Avid Media Composer's SubCap plugin.

Format specification:
- Header: "@ This file written with the Avid Caption plugin, version 1"
- Body: Tab-separated timecode (HH:MM:SS:FF) and text
- Timecodes are frame-based, not millisecond-based
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from lhotse.utils import Pathlike

from ...supervision import Supervision
from .. import register_writer
from ..base import FormatWriter


class FrameRate(Enum):
    """Standard video frame rates for professional editing."""

    FPS_23_976 = 23.976  # Film (24p pulldown)
    FPS_24 = 24.0  # Film
    FPS_25 = 25.0  # PAL / 25p
    FPS_29_97_NDF = 29.97  # NTSC Non-Drop Frame
    FPS_29_97_DF = 29.97  # NTSC Drop Frame (handled separately)
    FPS_30 = 30.0  # 30p
    FPS_50 = 50.0  # PAL 50p
    FPS_59_94 = 59.94  # NTSC 60p
    FPS_60 = 60.0  # 60p


@dataclass
class AvidDSConfig:
    """Configuration for Avid DS export.

    Attributes:
        fps: Frame rate for timecode calculation
        drop_frame: Whether to use drop-frame timecode (for 29.97fps)
        max_line_length: Maximum characters per line (Avid SubCap typically limits to 32-40)
        include_speaker: Whether to include speaker labels in text
    """

    fps: float = 25.0
    drop_frame: bool = False
    max_line_length: int = 40
    include_speaker: bool = True


class AvidDSWriter:
    """Writer for Avid DS subtitle format.

    This writer generates files compatible with Avid Media Composer's SubCap plugin.
    It handles frame-based timecode conversion and enforces broadcast-safe line lengths.

    Example:
        >>> from lattifai.caption import Caption
        >>> from lattifai.caption.formats.nle.avid import AvidDSWriter, AvidDSConfig
        >>> caption = Caption.read("input.srt")
        >>> config = AvidDSConfig(fps=25.0)
        >>> AvidDSWriter.write(caption.supervisions, "output_avid.txt", config)
    """

    # Avid DS file header (required for SubCap plugin)
    HEADER = "@ This file written with the Avid Caption plugin, version 1"

    @classmethod
    def seconds_to_timecode(
        cls,
        seconds: float,
        fps: float = 25.0,
        drop_frame: bool = False,
    ) -> str:
        """Convert seconds to SMPTE timecode (HH:MM:SS:FF).

        Args:
            seconds: Time in seconds
            fps: Frame rate (e.g., 23.976, 24, 25, 29.97, 30)
            drop_frame: Use drop-frame timecode (only for 29.97fps)

        Returns:
            Timecode string in HH:MM:SS:FF format (or HH:MM:SS;FF for drop-frame)

        Note:
            Drop-frame timecode skips frame numbers 0 and 1 at the start of each
            minute except every 10th minute to keep timecode in sync with real time
            for 29.97fps video.
        """
        if seconds < 0:
            seconds = 0

        if drop_frame and abs(fps - 29.97) < 0.01:
            # Drop-frame calculation for 29.97fps
            # Total frames at 29.97fps
            total_frames = int(round(seconds * 29.97))

            # Drop-frame adjustment
            # 2 frames dropped every minute except every 10th minute
            # = 2 * 9 = 18 frames dropped every 10 minutes
            d = total_frames // 17982  # Number of complete 10-minute chunks
            m = total_frames % 17982  # Remaining frames
            if m >= 2:
                # Add back dropped frames
                total_frames += 18 * d + 2 * ((m - 2) // 1798)

            frames = total_frames % 30
            total_seconds = total_frames // 30
            secs = total_seconds % 60
            total_minutes = total_seconds // 60
            mins = total_minutes % 60
            hours = total_minutes // 60

            # Drop-frame uses semicolon separator
            return f"{hours:02d}:{mins:02d}:{secs:02d};{frames:02d}"
        else:
            # Non-drop frame calculation
            total_frames = int(round(seconds * fps))
            frames = int(total_frames % fps)
            total_seconds = int(total_frames // fps)
            secs = total_seconds % 60
            total_minutes = total_seconds // 60
            mins = total_minutes % 60
            hours = total_minutes // 60

            return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

    @classmethod
    def wrap_text(cls, text: str, max_length: int = 40) -> List[str]:
        """Wrap text to fit within maximum line length.

        Args:
            text: Text to wrap
            max_length: Maximum characters per line

        Returns:
            List of wrapped lines
        """
        if len(text) <= max_length:
            return [text]

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            # +1 for space between words
            if current_length + word_length + (1 if current_line else 0) <= max_length:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        config: Optional[AvidDSConfig] = None,
    ) -> Path:
        """Write supervisions to Avid DS format file.

        Args:
            supervisions: List of supervision segments
            output_path: Output file path
            config: Avid DS export configuration

        Returns:
            Path to written file
        """
        if config is None:
            config = AvidDSConfig()

        output_path = Path(output_path)
        lines = [cls.HEADER, ""]  # Header + blank line

        for sup in supervisions:
            # Convert timestamps to timecode
            start_tc = cls.seconds_to_timecode(sup.start, config.fps, config.drop_frame)
            end_tc = cls.seconds_to_timecode(sup.end, config.fps, config.drop_frame)

            # Prepare text
            text = sup.text.strip() if sup.text else ""
            if config.include_speaker and sup.speaker:
                text = f"{sup.speaker}: {text}"

            # Wrap text to max line length
            wrapped_lines = cls.wrap_text(text, config.max_line_length)
            text = "\n".join(wrapped_lines)

            # Avid DS format: START_TC TAB END_TC TAB TEXT
            lines.append(f"{start_tc}\t{end_tc}\t{text}")

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        config: Optional[AvidDSConfig] = None,
    ) -> bytes:
        """Convert supervisions to Avid DS format bytes.

        Args:
            supervisions: List of supervision segments
            config: Avid DS export configuration

        Returns:
            Avid DS content as bytes
        """
        if config is None:
            config = AvidDSConfig()

        lines = [cls.HEADER, ""]

        for sup in supervisions:
            start_tc = cls.seconds_to_timecode(sup.start, config.fps, config.drop_frame)
            end_tc = cls.seconds_to_timecode(sup.end, config.fps, config.drop_frame)

            text = sup.text.strip() if sup.text else ""
            if config.include_speaker and sup.speaker:
                text = f"{sup.speaker}: {text}"

            wrapped_lines = cls.wrap_text(text, config.max_line_length)
            text = "\n".join(wrapped_lines)

            lines.append(f"{start_tc}\t{end_tc}\t{text}")

        return "\n".join(lines).encode("utf-8")


@register_writer("avid_ds")
class AvidDSFormat(FormatWriter):
    """Format handler for Avid DS caption format."""

    format_id = "avid_ds"
    extensions = [".txt"]
    description = "Avid DS Caption Format"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ):
        """Write supervisions to Avid DS format file.

        Args:
            supervisions: List of supervision segments
            output_path: Path to output file
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options (fps, drop_frame, etc.)

        Returns:
            Path to written file
        """
        config = AvidDSConfig(include_speaker=include_speaker, **kwargs)
        return AvidDSWriter.write(supervisions, output_path, config)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to Avid DS format bytes.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Avid DS content as bytes
        """
        config = AvidDSConfig(include_speaker=include_speaker, **kwargs)
        return AvidDSWriter.to_bytes(supervisions, config)
