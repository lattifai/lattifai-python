"""Adobe Audition marker CSV format writer for audio post-production.

This module provides functionality to export captions as Adobe Audition markers,
enabling audio editors to navigate and search transcripts in their audio editing workflow.

Format specification (Audition CSV):
- Header: Name,Start,Duration,Time Format,Type,Description
- Time Format: "decimal" (seconds with decimal)
- Type: "Cue" for markers
"""

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union

from lhotse.utils import Pathlike

from ...supervision import Supervision
from .. import register_writer
from ..base import FormatReader, FormatWriter


@dataclass
class AuditionCSVConfig:
    """Configuration for Adobe Audition CSV marker export.

    Attributes:
        time_format: Time format for markers ("decimal" or "samples")
        marker_type: Marker type ("Cue", "Subclip", "Track")
        include_speaker_in_name: Include speaker name in marker name
        use_description: Put text content in description field
        sample_rate: Sample rate (only used when time_format="samples")
    """

    time_format: str = "decimal"
    marker_type: str = "Cue"
    include_speaker_in_name: bool = True
    use_description: bool = True
    sample_rate: int = 48000


class AuditionCSVWriter:
    """Writer for Adobe Audition marker CSV format.

    Generates CSV files compatible with Adobe Audition's marker import feature,
    allowing transcripts to be imported as navigable markers in audio projects.

    Example:
        >>> from lattifai.caption import Caption
        >>> from lattifai.caption.formats.nle.audition_writer import AuditionCSVWriter, AuditionCSVConfig
        >>> caption = Caption.read("input.srt")
        >>> config = AuditionCSVConfig(include_speaker_in_name=True)
        >>> AuditionCSVWriter.write(caption.supervisions, "markers.csv", config)
    """

    # CSV header required by Adobe Audition
    HEADER = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]

    @classmethod
    def _format_time(
        cls,
        seconds: float,
        time_format: str = "decimal",
        sample_rate: int = 48000,
    ) -> str:
        """Format time value for Audition CSV.

        Args:
            seconds: Time in seconds
            time_format: "decimal" for seconds, "samples" for sample count
            sample_rate: Sample rate for sample-based timing

        Returns:
            Formatted time string
        """
        if time_format == "samples":
            return str(int(round(seconds * sample_rate)))
        else:
            # Decimal format with millisecond precision
            return f"{seconds:.3f}"

    @classmethod
    def _format_marker_name(
        cls,
        supervision: "Supervision",
        index: int,
        include_speaker: bool,
    ) -> str:
        """Format marker name from supervision.

        Args:
            supervision: Supervision segment
            index: Marker index (1-based)
            include_speaker: Whether to include speaker in name

        Returns:
            Formatted marker name
        """
        if include_speaker and supervision.speaker:
            return f"{supervision.speaker} - Marker {index:03d}"
        else:
            return f"Marker {index:03d}"

    @classmethod
    def _generate_csv_content(
        cls,
        supervisions: List["Supervision"],
        config: AuditionCSVConfig,
    ) -> str:
        """Generate CSV content string.

        Args:
            supervisions: List of supervision segments
            config: Export configuration

        Returns:
            CSV content as string
        """
        output = StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(cls.HEADER)

        # Write markers
        for i, sup in enumerate(supervisions, 1):
            name = cls._format_marker_name(sup, i, config.include_speaker_in_name)
            start = cls._format_time(sup.start, config.time_format, config.sample_rate)
            duration = cls._format_time(sup.duration, config.time_format, config.sample_rate)

            if config.use_description:
                description = sup.text.strip() if sup.text else ""
            else:
                description = ""

            writer.writerow(
                [
                    name,
                    start,
                    duration,
                    config.time_format,
                    config.marker_type,
                    description,
                ]
            )

        return output.getvalue()

    @classmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        config: Optional[AuditionCSVConfig] = None,
    ) -> Path:
        """Write supervisions to Audition CSV marker format.

        Args:
            supervisions: List of supervision segments
            output_path: Output file path
            config: Export configuration

        Returns:
            Path to written file
        """
        if config is None:
            config = AuditionCSVConfig()

        output_path = Path(output_path)
        content = cls._generate_csv_content(supervisions, config)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        config: Optional[AuditionCSVConfig] = None,
    ) -> bytes:
        """Convert supervisions to Audition CSV format bytes.

        Args:
            supervisions: List of supervision segments
            config: Export configuration

        Returns:
            CSV content as bytes
        """
        if config is None:
            config = AuditionCSVConfig()

        content = cls._generate_csv_content(supervisions, config)
        return content.encode("utf-8")


@dataclass
class EdiMarkerConfig:
    """Configuration for EdiMarker (Pro Tools) compatible CSV export.

    Attributes:
        include_speaker: Include speaker name in marker
        marker_prefix: Prefix for marker names
    """

    include_speaker: bool = True
    marker_prefix: str = "M"


class EdiMarkerWriter:
    """Writer for EdiMarker-compatible CSV format (Pro Tools bridge).

    EdiMarker is a third-party tool that converts CSV files to Pro Tools marker format.
    This writer generates CSV files compatible with EdiMarker's expected input format.

    Example:
        >>> from lattifai.caption import Caption
        >>> from lattifai.caption.formats.nle.audition_writer import EdiMarkerWriter
        >>> caption = Caption.read("input.srt")
        >>> EdiMarkerWriter.write(caption.supervisions, "markers_edimarker.csv")
    """

    # EdiMarker expected CSV header
    HEADER = ["Name", "Start", "End", "Text"]

    @classmethod
    def _seconds_to_timecode(cls, seconds: float, fps: float = 24.0) -> str:
        """Convert seconds to timecode format HH:MM:SS:FF.

        Args:
            seconds: Time in seconds
            fps: Frame rate

        Returns:
            Timecode string
        """
        total_frames = int(round(seconds * fps))
        frames = int(total_frames % fps)
        total_seconds = int(total_frames // fps)
        secs = total_seconds % 60
        total_minutes = total_seconds // 60
        mins = total_minutes % 60
        hours = total_minutes // 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"

    @classmethod
    def _generate_csv_content(
        cls,
        supervisions: List["Supervision"],
        config: EdiMarkerConfig,
        fps: float = 24.0,
    ) -> str:
        """Generate CSV content string.

        Args:
            supervisions: List of supervision segments
            config: Export configuration
            fps: Frame rate for timecode conversion

        Returns:
            CSV content as string
        """
        output = StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writerow(cls.HEADER)

        # Write markers
        for i, sup in enumerate(supervisions, 1):
            if config.include_speaker and sup.speaker:
                name = f"{config.marker_prefix}{i:03d}_{sup.speaker}"
            else:
                name = f"{config.marker_prefix}{i:03d}"

            start_tc = cls._seconds_to_timecode(sup.start, fps)
            end_tc = cls._seconds_to_timecode(sup.end, fps)
            text = sup.text.strip() if sup.text else ""

            writer.writerow([name, start_tc, end_tc, text])

        return output.getvalue()

    @classmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        config: Optional[EdiMarkerConfig] = None,
        fps: float = 24.0,
    ) -> Path:
        """Write supervisions to EdiMarker-compatible CSV format.

        Args:
            supervisions: List of supervision segments
            output_path: Output file path
            config: Export configuration
            fps: Frame rate for timecode conversion

        Returns:
            Path to written file
        """
        if config is None:
            config = EdiMarkerConfig()

        output_path = Path(output_path)
        content = cls._generate_csv_content(supervisions, config, fps)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(content)

        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        config: Optional[EdiMarkerConfig] = None,
        fps: float = 24.0,
    ) -> bytes:
        """Convert supervisions to EdiMarker CSV format bytes.

        Args:
            supervisions: List of supervision segments
            config: Export configuration
            fps: Frame rate for timecode conversion

        Returns:
            CSV content as bytes
        """
        if config is None:
            config = EdiMarkerConfig()

        content = cls._generate_csv_content(supervisions, config, fps)
        return content.encode("utf-8")


@register_writer("audition_csv")
class AuditionCSVFormat(FormatWriter):
    """Format handler for Adobe Audition CSV markers."""

    format_id = "audition_csv"
    extensions = [".csv"]
    description = "Adobe Audition CSV Marker Format"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ):
        """Write supervisions to Audition CSV format.

        Args:
            supervisions: List of supervision segments
            output_path: Path to output file
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Path to written file
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by Audition CSV)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = AuditionCSVConfig(include_speaker_in_name=include_speaker, **kwargs)
        return AuditionCSVWriter.write(supervisions, output_path, config)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to Audition CSV bytes.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Audition CSV content as bytes
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by Audition CSV)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = AuditionCSVConfig(include_speaker_in_name=include_speaker, **kwargs)
        return AuditionCSVWriter.to_bytes(supervisions, config)


class AuditionCSVReader:
    """Reader for Adobe Audition CSV markers."""

    @classmethod
    def read(cls, source: str, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read Audition CSV content and return supervisions."""
        supervisions = []

        # Use csv module to handle quoting correctly
        f = StringIO(source)
        reader = csv.DictReader(f)

        # Mapping for flexible header names if needed, but assuming standard Audition export
        # Standard: Name,Start,Duration,Time Format,Type,Description

        sample_rate = kwargs.get("sample_rate", 48000)

        for row in reader:
            # Check for required fields
            if "Start" not in row or "Duration" not in row:
                continue

            time_format = row.get("Time Format", "decimal")
            start_val = row["Start"]
            duration_val = row["Duration"]

            try:
                if time_format == "samples":
                    start_sec = float(start_val) / sample_rate
                    duration_sec = float(duration_val) / sample_rate
                else:
                    # decimal
                    start_sec = float(start_val)
                    duration_sec = float(duration_val)
            except ValueError:
                continue

            # Extract text from Description or Name
            description = row.get("Description", "")
            name = row.get("Name", "")

            # Logic: If description has content, prefer it as the caption text?
            # Or is Name the text? The Writer puts text in Description if configured,
            # and Name is "Speaker - Marker X".
            # So Description is the best candidate for caption text.
            text = description
            if not text and name:
                # Fallback to Name provided it doesn't look like generic "Marker 01"
                if not name.startswith("Marker "):
                    text = name

            if duration_sec > 0 and text:
                supervisions.append(
                    Supervision(
                        id=str(uuid.uuid4()),
                        recording_id="audition_import",
                        start=start_sec,
                        duration=duration_sec,
                        text=text.strip() if normalize_text else text,
                    )
                )

        return sorted(supervisions, key=lambda s: s.start)


import uuid

from .. import register_reader


@register_reader("audition_csv")
class AuditionCSVReaderHandler(FormatReader):
    """Reader handler for Audition CSV."""

    format_id = "audition_csv"
    extensions = [".csv"]

    @classmethod
    def can_read(cls, path: Union[Pathlike, str]) -> bool:
        # Check first line for "Time Format" or "Audition" specific headers
        if isinstance(path, (str, Path)) and not cls.is_content(path):
            # We rely on upstream detection because .csv is too generic
            return str(path).lower().endswith(".csv")
        return False

    @classmethod
    def read(cls, source: Union[Pathlike, str], normalize_text: bool = True, **kwargs) -> List[Supervision]:
        if isinstance(source, (str, Path)) and not cls.is_content(source):
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(source)

        return AuditionCSVReader.read(content, normalize_text=normalize_text, **kwargs)


@register_writer("edimarker_csv")
class EdiMarkerCSVFormat(FormatWriter):
    """Format handler for EdiMarker (Pro Tools) CSV markers."""

    format_id = "edimarker_csv"
    extensions = [".csv"]
    description = "EdiMarker (Pro Tools) CSV Marker Format"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        fps: float = 24.0,
        **kwargs,
    ):
        """Write supervisions to EdiMarker CSV format.

        Args:
            supervisions: List of supervision segments
            output_path: Path to output file
            include_speaker: Whether to include speaker labels
            fps: Frame rate for timecode conversion
            **kwargs: Additional config options

        Returns:
            Path to written file
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by EdiMarker)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = EdiMarkerConfig(include_speaker=include_speaker, **kwargs)
        return EdiMarkerWriter.write(supervisions, output_path, config, fps=fps)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        fps: float = 24.0,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to EdiMarker CSV bytes.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker labels
            fps: Frame rate for timecode conversion
            **kwargs: Additional config options

        Returns:
            EdiMarker CSV content as bytes
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by EdiMarker)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = EdiMarkerConfig(include_speaker=include_speaker, **kwargs)
        return EdiMarkerWriter.to_bytes(supervisions, config, fps=fps)
