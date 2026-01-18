"""FCPXML (Final Cut Pro XML) format writer for Final Cut Pro and DaVinci Resolve.

This module provides functionality to export captions in FCPXML v1.10 format,
which is compatible with Final Cut Pro and DaVinci Resolve.

Key features:
- Speaker diarization mapped to FCP Roles
- Text style definitions
- Bundle format support (.fcpxmld)
"""

import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from xml.dom import minidom

from lhotse.utils import Pathlike

from ...supervision import Supervision
from .. import register_writer
from ..base import FormatReader, FormatWriter


@dataclass
class FCPXMLStyle:
    """Text style configuration for FCPXML captions.

    Attributes:
        font: Font family name
        font_size: Font size in points
        font_color: Font color in RGBA format (e.g., "1 1 1 1" for white)
        background_color: Background color in RGBA format
        alignment: Text alignment ("left", "center", "right")
    """

    font: str = "Helvetica"
    font_size: int = 40
    font_color: str = "1 1 1 1"  # White
    background_color: Optional[str] = None
    alignment: str = "center"


@dataclass
class FCPXMLConfig:
    """Configuration for FCPXML export.

    Attributes:
        fps: Frame rate (affects duration calculations)
        map_speakers_to_roles: Map different speakers to FCP roles
        default_style: Default text style
        speaker_styles: Speaker-specific styles
        project_name: Name for the FCPXML project
        event_name: Name for the FCPXML event
        use_bundle: Export as .fcpxmld bundle (directory with Info.fcpxml)
    """

    fps: float = 25.0
    map_speakers_to_roles: bool = True
    default_style: FCPXMLStyle = field(default_factory=FCPXMLStyle)
    speaker_styles: Dict[str, FCPXMLStyle] = field(default_factory=dict)
    project_name: str = "LattifAI Captions"
    event_name: str = "LattifAI Import"
    use_bundle: bool = True


class FCPXMLWriter:
    """Writer for FCPXML (Final Cut Pro XML) format.

    Generates FCPXML v1.10 compatible files for Final Cut Pro and DaVinci Resolve.
    Supports speaker-to-role mapping for advanced editing workflows.

    Example:
        >>> from lattifai.caption import Caption
        >>> from lattifai.caption.formats.nle.fcpxml_writer import FCPXMLWriter, FCPXMLConfig
        >>> caption = Caption.read("input.srt")
        >>> config = FCPXMLConfig(map_speakers_to_roles=True)
        >>> FCPXMLWriter.write(caption.supervisions, "output.fcpxmld", config)
    """

    FCPXML_VERSION = "1.10"

    @classmethod
    def _seconds_to_fcpxml_time(cls, seconds: float, fps: float = 25.0) -> str:
        """Convert seconds to FCPXML time format.

        FCPXML uses rational time format: "numerator/denominator s"
        For simplicity, we use a large denominator for precision.

        Args:
            seconds: Time in seconds
            fps: Frame rate for calculation

        Returns:
            Time string in FCPXML format (e.g., "10/1s" or "1001/100s")
        """
        # Use 1000 as denominator for millisecond precision
        numerator = int(round(seconds * 1000))
        return f"{numerator}/1000s"

    @classmethod
    def _generate_uuid(cls) -> str:
        """Generate a unique identifier for FCPXML elements."""
        return str(uuid.uuid4()).upper()

    @classmethod
    def _create_text_style_def(
        cls,
        parent: ET.Element,
        style_id: str,
        style: FCPXMLStyle,
    ) -> ET.Element:
        """Create a text-style-def element.

        Args:
            parent: Parent XML element
            style_id: Unique style identifier
            style: Style configuration

        Returns:
            Created text-style-def element
        """
        style_def = ET.SubElement(parent, "text-style-def", id=style_id)
        text_style = ET.SubElement(
            style_def,
            "text-style",
            font=style.font,
            fontSize=str(style.font_size),
            fontColor=style.font_color,
            alignment=style.alignment,
        )
        if style.background_color:
            text_style.set("backgroundColor", style.background_color)
        return style_def

    @classmethod
    def _get_role_name(cls, speaker: Optional[str]) -> str:
        """Convert speaker name to FCP role format.

        Args:
            speaker: Speaker name or None

        Returns:
            Role name in FCP format
        """
        if not speaker:
            return "iTT?captionFormat=ITT.en"
        # Clean speaker name for role
        clean_name = speaker.replace(" ", "_").replace(".", "_")
        return f"iTT?role=Dialogue.{clean_name}"

    @classmethod
    def _build_fcpxml(
        cls,
        supervisions: List["Supervision"],
        config: FCPXMLConfig,
    ) -> ET.Element:
        """Build FCPXML document structure.

        Args:
            supervisions: List of supervision segments
            config: FCPXML configuration

        Returns:
            Root FCPXML element
        """
        # Create root element
        root = ET.Element("fcpxml", version=cls.FCPXML_VERSION)

        # Create resources section
        resources = ET.SubElement(root, "resources")

        # Add format resource (for timing calculations)
        format_id = "r1"
        # Frame duration as rational: 1/fps
        frame_duration = f"100/{int(config.fps * 100)}s"
        ET.SubElement(
            resources,
            "format",
            id=format_id,
            frameDuration=frame_duration,
            width="1920",
            height="1080",
        )

        # Create default style
        default_style_id = "ts1"
        cls._create_text_style_def(resources, default_style_id, config.default_style)

        # Create speaker-specific styles
        style_counter = 2
        speaker_style_ids = {}
        if config.map_speakers_to_roles:
            speakers = set(sup.speaker for sup in supervisions if sup.speaker)
            for speaker in speakers:
                style = config.speaker_styles.get(speaker, config.default_style)
                style_id = f"ts{style_counter}"
                cls._create_text_style_def(resources, style_id, style)
                speaker_style_ids[speaker] = style_id
                style_counter += 1

        # Create library structure
        library = ET.SubElement(root, "library")
        event = ET.SubElement(library, "event", name=config.event_name)
        project = ET.SubElement(project := ET.SubElement(event, "project", name=config.project_name), "sequence")

        # Calculate total duration
        if supervisions:
            total_duration = max(sup.end for sup in supervisions)
        else:
            total_duration = 3600  # Default 1 hour

        total_duration_str = cls._seconds_to_fcpxml_time(total_duration, config.fps)

        # Create spine (main timeline container)
        spine = ET.SubElement(project, "spine")

        # Create a gap element as the base for attaching captions
        gap = ET.SubElement(
            spine,
            "gap",
            name="Base",
            offset="0/1s",
            duration=total_duration_str,
            start="0/1s",
        )

        # Add captions to the gap
        for i, sup in enumerate(supervisions, 1):
            start_time = cls._seconds_to_fcpxml_time(sup.start, config.fps)
            duration = cls._seconds_to_fcpxml_time(sup.duration, config.fps)

            # Determine role based on speaker
            if config.map_speakers_to_roles and sup.speaker:
                role = cls._get_role_name(sup.speaker)
            else:
                role = cls._get_role_name(None)

            # Create caption element
            caption = ET.SubElement(
                gap,
                "caption",
                role=role,
                name=f"Caption {i}",
                offset=start_time,
                duration=duration,
                start=start_time,
            )

            # Add text content
            text_elem = ET.SubElement(caption, "text")
            text_elem.text = sup.text or ""

            # Add style reference
            style_id = speaker_style_ids.get(sup.speaker, default_style_id)
            caption.append(ET.Element("text-style-ref", ref=style_id))

        return root

    @classmethod
    def _prettify_xml(cls, element: ET.Element) -> str:
        """Convert XML element to pretty-printed string.

        Args:
            element: XML element to format

        Returns:
            Formatted XML string with proper indentation
        """
        rough_string = ET.tostring(element, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        # Remove extra whitespace and use 2-space indentation
        pretty = reparsed.toprettyxml(indent="    ")
        # Remove the XML declaration line and extra blank lines
        lines = [line for line in pretty.split("\n") if line.strip()]
        # Add proper XML declaration
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines[1:])

    @classmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        config: Optional[FCPXMLConfig] = None,
    ) -> Path:
        """Write supervisions to FCPXML format.

        Args:
            supervisions: List of supervision segments
            output_path: Output file path (.fcpxml or .fcpxmld)
            config: FCPXML export configuration

        Returns:
            Path to written file/bundle
        """
        if config is None:
            config = FCPXMLConfig()

        output_path = Path(output_path)
        root = cls._build_fcpxml(supervisions, config)
        xml_content = cls._prettify_xml(root)

        if config.use_bundle or output_path.suffix.lower() == ".fcpxmld":
            # Create bundle directory structure
            bundle_path = output_path.with_suffix(".fcpxmld")
            bundle_path.mkdir(parents=True, exist_ok=True)

            # Write Info.fcpxml inside bundle
            info_path = bundle_path / "Info.fcpxml"
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(xml_content)

            return bundle_path
        else:
            # Write single FCPXML file
            output_path = output_path.with_suffix(".fcpxml")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(xml_content)

            return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        config: Optional[FCPXMLConfig] = None,
    ) -> bytes:
        """Convert supervisions to FCPXML format bytes.

        Note: This returns a single FCPXML file content, not a bundle.

        Args:
            supervisions: List of supervision segments
            config: FCPXML export configuration

        Returns:
            FCPXML content as bytes
        """
        if config is None:
            config = FCPXMLConfig()

        root = cls._build_fcpxml(supervisions, config)
        xml_content = cls._prettify_xml(root)
        return xml_content.encode("utf-8")

    @classmethod
    def write_with_word_level(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        config: Optional[FCPXMLConfig] = None,
    ) -> Path:
        """Write supervisions with word-level timing to FCPXML.

        This creates individual caption elements for each word, enabling
        karaoke-style effects in Final Cut Pro.

        Args:
            supervisions: List of supervision segments with word-level alignment
            output_path: Output file path
            config: FCPXML export configuration

        Returns:
            Path to written file/bundle
        """
        if config is None:
            config = FCPXMLConfig()

        # Expand word-level alignments into individual supervisions
        from ...supervision import Supervision as SupClass

        expanded = []
        for sup in supervisions:
            alignment = getattr(sup, "alignment", None)
            if alignment and "word" in alignment:
                for word_item in alignment["word"]:
                    expanded.append(
                        SupClass(
                            text=word_item.symbol,
                            start=word_item.start,
                            duration=word_item.duration,
                            speaker=sup.speaker,
                        )
                    )
            else:
                expanded.append(sup)

        return cls.write(expanded, output_path, config)


@register_writer("fcpxml")
class FCPXMLFormat(FormatWriter):
    """Format handler for Final Cut Pro XML (FCPXML)."""

    format_id = "fcpxml"
    extensions = [".fcpxml", ".fcpxmld"]
    description = "Final Cut Pro XML Format"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ):
        """Write supervisions to FCPXML format.

        Args:
            supervisions: List of supervision segments
            output_path: Path to output file
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Path to written file
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by FCPXML)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = FCPXMLConfig(**kwargs)
        return FCPXMLWriter.write(supervisions, output_path, config)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to FCPXML bytes.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            FCPXML content as bytes
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by FCPXML)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = FCPXMLConfig(**kwargs)
        return FCPXMLWriter.to_bytes(supervisions, config)


class FCPXMLReader:
    """Reader for FCPXML format."""

    @classmethod
    def _parse_rational_time(cls, time_str: str) -> float:
        """Parse rational time string (e.g., "100/25s") to seconds."""
        if not time_str or not time_str.endswith("s"):
            return 0.0

        val_str = time_str[:-1]  # Remove 's'
        if "/" in val_str:
            num, den = val_str.split("/")
            return float(num) / float(den)
        else:
            return float(val_str)

    @classmethod
    def read(cls, source: str, normalize_text: bool = True) -> List[Supervision]:
        """Read FCPXML content and return supervisions."""
        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            return []

        supervisions = []

        # Traverse recursively to find caption elements
        # FCPXML structure is flexible, captions can be nested in spines, gaps, clips, etc.
        for caption in root.iter("caption"):
            # Get timing
            offset_str = caption.get("offset", "0s")
            # start_str = caption.get("start", "0s")
            duration_str = caption.get("duration", "0s")

            # In FCPXML, logic for absolute time is complex depending on parent containers.
            # Simplified approach: If direct child of a gap/spine in a simple project,
            # offset + start might be enough.
            # However, standard caption export usually puts them relative to the start of the project
            # or the 'offset' attribute is the absolute time on the timeline.
            # Let's assume 'offset' is the timeline start time for the caption clip.

            start_sec = cls._parse_rational_time(offset_str)
            duration_sec = cls._parse_rational_time(duration_str)

            # Get text
            text_elem = caption.find("text")
            text_content = ""
            if text_elem is not None:
                text_content = text_elem.text

            # Fallback if text element is empty or missing (some versions might differ)
            if not text_content:
                # Sometimes text is in 'name' attribute if it's a title?
                # But for 'caption' element, <text> child is standard.
                continue

            if duration_sec > 0:
                supervisions.append(
                    Supervision(
                        id=caption.get("name", str(uuid.uuid4())),
                        recording_id="fcpxml_import",
                        start=start_sec,
                        duration=duration_sec,
                        text=text_content.strip() if normalize_text else text_content,
                    )
                )

        return sorted(supervisions, key=lambda s: s.start)


from .. import register_reader


@register_reader("fcpxml")
class FCPXMLReaderHandler(FormatReader):
    """Reader handler for FCPXML."""

    format_id = "fcpxml"
    extensions = [".fcpxml", ".fcpxmld"]

    @classmethod
    def read(cls, source: Union[Pathlike, str], normalize_text: bool = True, **kwargs) -> List[Supervision]:
        if isinstance(source, (str, Path)) and not cls.is_content(source):
            # Check if it's a bundle directory
            p = Path(source)
            if p.is_dir() and p.suffix == ".fcpxmld":
                info_path = p / "Info.fcpxml"
                if info_path.exists():
                    p = info_path

            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(source)

        return FCPXMLReader.read(content, normalize_text=normalize_text)
