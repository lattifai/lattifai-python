"""Premiere Pro XML format writer for Adobe Premiere Pro integration.

This module provides functionality to export captions as Premiere Pro XML sequences,
enabling captions to be imported as animatable graphic clips rather than simple caption tracks.

Key features:
- Each caption/word becomes a separate text generator clip
- Supports word-level timing for karaoke-style effects
- Speaker separation to different video tracks
"""

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
class PremiereXMLConfig:
    """Configuration for Premiere Pro XML export.

    Attributes:
        fps: Frame rate for the sequence
        width: Sequence width in pixels
        height: Sequence height in pixels
        use_word_level: Export each word as separate clip (for karaoke effects)
        separate_speaker_tracks: Put different speakers on different video tracks
        font_name: Font name for text generators
        font_size: Font size for text generators
        sequence_name: Name for the sequence
    """

    fps: float = 25.0
    width: int = 1920
    height: int = 1080
    use_word_level: bool = False
    separate_speaker_tracks: bool = True
    font_name: str = "Arial"
    font_size: int = 60
    sequence_name: str = "LattifAI Captions"


class PremiereXMLWriter:
    """Writer for Premiere Pro XML (FCP7 XML) format.

    Generates XML sequences where captions are text generator clips on video tracks,
    allowing full animation and effects capabilities in Premiere Pro.

    Example:
        >>> from lattifai.caption import Caption
        >>> from lattifai.caption.formats.nle.premiere_xml_writer import PremiereXMLWriter, PremiereXMLConfig
        >>> caption = Caption.read("input.srt")
        >>> config = PremiereXMLConfig(use_word_level=True)
        >>> PremiereXMLWriter.write(caption.supervisions, "output.xml", config)
    """

    @classmethod
    def _seconds_to_frames(cls, seconds: float, fps: float) -> int:
        """Convert seconds to frame count."""
        return int(round(seconds * fps))

    @classmethod
    def _create_rate_element(cls, parent: ET.Element, fps: float) -> ET.Element:
        """Create a rate element with timebase and ntsc flag."""
        rate = ET.SubElement(parent, "rate")
        # Determine timebase and ntsc flag
        if abs(fps - 23.976) < 0.01:
            ET.SubElement(rate, "timebase").text = "24"
            ET.SubElement(rate, "ntsc").text = "TRUE"
        elif abs(fps - 29.97) < 0.01:
            ET.SubElement(rate, "timebase").text = "30"
            ET.SubElement(rate, "ntsc").text = "TRUE"
        elif abs(fps - 59.94) < 0.01:
            ET.SubElement(rate, "timebase").text = "60"
            ET.SubElement(rate, "ntsc").text = "TRUE"
        else:
            ET.SubElement(rate, "timebase").text = str(int(fps))
            ET.SubElement(rate, "ntsc").text = "FALSE"
        return rate

    @classmethod
    def _create_text_generator_clip(
        cls,
        parent: ET.Element,
        clip_id: str,
        text: str,
        start_frame: int,
        end_frame: int,
        fps: float,
        config: PremiereXMLConfig,
    ) -> ET.Element:
        """Create a text generator clipitem.

        Args:
            parent: Parent track element
            clip_id: Unique clip identifier
            text: Text content
            start_frame: Start frame number
            end_frame: End frame number
            fps: Frame rate
            config: Export configuration

        Returns:
            Created clipitem element
        """
        clipitem = ET.SubElement(parent, "clipitem", id=clip_id)

        ET.SubElement(clipitem, "name").text = text[:50]  # Truncate long names
        ET.SubElement(clipitem, "enabled").text = "TRUE"
        ET.SubElement(clipitem, "duration").text = str(end_frame - start_frame)

        cls._create_rate_element(clipitem, fps)

        ET.SubElement(clipitem, "start").text = str(start_frame)
        ET.SubElement(clipitem, "end").text = str(end_frame)
        ET.SubElement(clipitem, "in").text = "0"
        ET.SubElement(clipitem, "out").text = str(end_frame - start_frame)

        # Add generator item (text generator)
        file_elem = ET.SubElement(clipitem, "file", id=f"file-{clip_id}")
        ET.SubElement(file_elem, "name").text = "Text"
        ET.SubElement(file_elem, "pathurl").text = ""

        # Add media type
        media = ET.SubElement(file_elem, "media")
        video = ET.SubElement(media, "video")
        ET.SubElement(video, "duration").text = str(end_frame - start_frame)

        # Add filter for text generator parameters
        filter_elem = ET.SubElement(clipitem, "filter")
        effect = ET.SubElement(filter_elem, "effect")
        ET.SubElement(effect, "name").text = "Basic Text"
        ET.SubElement(effect, "effectid").text = "BasicText"
        ET.SubElement(effect, "effectcategory").text = "Text"
        ET.SubElement(effect, "effecttype").text = "generator"

        # Text content parameter
        param_text = ET.SubElement(effect, "parameter")
        ET.SubElement(param_text, "parameterid").text = "str"
        ET.SubElement(param_text, "name").text = "Text"
        value = ET.SubElement(param_text, "value")
        value.text = text

        # Font name parameter
        param_font = ET.SubElement(effect, "parameter")
        ET.SubElement(param_font, "parameterid").text = "font"
        ET.SubElement(param_font, "name").text = "Font"
        ET.SubElement(param_font, "value").text = config.font_name

        # Font size parameter
        param_size = ET.SubElement(effect, "parameter")
        ET.SubElement(param_size, "parameterid").text = "fontsize"
        ET.SubElement(param_size, "name").text = "Size"
        ET.SubElement(param_size, "valuemin").text = "1"
        ET.SubElement(param_size, "valuemax").text = "1000"
        ET.SubElement(param_size, "value").text = str(config.font_size)

        return clipitem

    @classmethod
    def _build_xml(
        cls,
        supervisions: List["Supervision"],
        config: PremiereXMLConfig,
    ) -> ET.Element:
        """Build Premiere Pro XML document structure.

        Args:
            supervisions: List of supervision segments
            config: Export configuration

        Returns:
            Root XML element
        """
        # Create root element
        root = ET.Element("xmeml", version="4")

        # Create sequence
        sequence = ET.SubElement(root, "sequence")
        ET.SubElement(sequence, "name").text = config.sequence_name

        # Calculate total duration
        if supervisions:
            total_duration_seconds = max(sup.end for sup in supervisions)
        else:
            total_duration_seconds = 60

        total_frames = cls._seconds_to_frames(total_duration_seconds, config.fps)
        ET.SubElement(sequence, "duration").text = str(total_frames)

        cls._create_rate_element(sequence, config.fps)

        # Timecode settings
        timecode = ET.SubElement(sequence, "timecode")
        cls._create_rate_element(timecode, config.fps)
        ET.SubElement(timecode, "string").text = "00:00:00:00"
        ET.SubElement(timecode, "frame").text = "0"
        displayformat = ET.SubElement(timecode, "displayformat")
        displayformat.text = "NDF"

        # Media section
        media = ET.SubElement(sequence, "media")

        # Video section
        video = ET.SubElement(media, "video")
        format_elem = ET.SubElement(video, "format")
        sample_characteristics = ET.SubElement(format_elem, "samplecharacteristics")
        cls._create_rate_element(sample_characteristics, config.fps)
        ET.SubElement(sample_characteristics, "width").text = str(config.width)
        ET.SubElement(sample_characteristics, "height").text = str(config.height)
        ET.SubElement(sample_characteristics, "anamorphic").text = "FALSE"
        ET.SubElement(sample_characteristics, "pixelaspectratio").text = "square"
        ET.SubElement(sample_characteristics, "fielddominance").text = "none"

        # Group supervisions by speaker if separate tracks are enabled
        if config.separate_speaker_tracks:
            speaker_groups: Dict[Optional[str], List["Supervision"]] = {}
            for sup in supervisions:
                speaker = sup.speaker or "_default"
                if speaker not in speaker_groups:
                    speaker_groups[speaker] = []
                speaker_groups[speaker].append(sup)
        else:
            speaker_groups = {"_default": supervisions}

        # Create video tracks for each speaker group
        clip_counter = 1
        for speaker, sups in speaker_groups.items():
            track = ET.SubElement(video, "track")

            # Expand to word level if configured
            if config.use_word_level:
                items_to_process = []
                for sup in sups:
                    alignment = getattr(sup, "alignment", None)
                    if alignment and "word" in alignment:
                        for word_item in alignment["word"]:
                            items_to_process.append(
                                {
                                    "text": word_item.symbol,
                                    "start": word_item.start,
                                    "duration": word_item.duration,
                                    "speaker": sup.speaker,
                                }
                            )
                    else:
                        items_to_process.append(
                            {
                                "text": sup.text,
                                "start": sup.start,
                                "duration": sup.duration,
                                "speaker": sup.speaker,
                            }
                        )
            else:
                items_to_process = [
                    {
                        "text": sup.text,
                        "start": sup.start,
                        "duration": sup.duration,
                        "speaker": sup.speaker,
                    }
                    for sup in sups
                ]

            # Create clipitems for each caption/word
            for item in items_to_process:
                if not item["text"]:
                    continue

                start_frame = cls._seconds_to_frames(item["start"], config.fps)
                end_frame = cls._seconds_to_frames(
                    item["start"] + item["duration"],
                    config.fps,
                )

                clip_id = f"clipitem-{clip_counter}"
                cls._create_text_generator_clip(
                    track,
                    clip_id,
                    item["text"],
                    start_frame,
                    end_frame,
                    config.fps,
                    config,
                )
                clip_counter += 1

        return root

    @classmethod
    def _prettify_xml(cls, element: ET.Element) -> str:
        """Convert XML element to pretty-printed string."""
        rough_string = ET.tostring(element, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="    ")
        lines = [line for line in pretty.split("\n") if line.strip()]
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines[1:])

    @classmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        config: Optional[PremiereXMLConfig] = None,
    ) -> Path:
        """Write supervisions to Premiere Pro XML format.

        Args:
            supervisions: List of supervision segments
            output_path: Output file path
            config: Export configuration

        Returns:
            Path to written file
        """
        if config is None:
            config = PremiereXMLConfig()

        output_path = Path(output_path).with_suffix(".xml")
        root = cls._build_xml(supervisions, config)
        xml_content = cls._prettify_xml(root)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        config: Optional[PremiereXMLConfig] = None,
    ) -> bytes:
        """Convert supervisions to Premiere Pro XML format bytes.

        Args:
            supervisions: List of supervision segments
            config: Export configuration

        Returns:
            Premiere XML content as bytes
        """
        if config is None:
            config = PremiereXMLConfig()

        root = cls._build_xml(supervisions, config)
        xml_content = cls._prettify_xml(root)
        return xml_content.encode("utf-8")


@register_writer("premiere_xml")
class PremiereXMLFormat(FormatWriter):
    """Format handler for Adobe Premiere Pro XML."""

    format_id = "premiere_xml"
    extensions = [".xml"]
    description = "Adobe Premiere Pro XML Format"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        include_speaker: bool = True,
        **kwargs,
    ):
        """Write supervisions to Premiere Pro XML format.

        Args:
            supervisions: List of supervision segments
            output_path: Path to output file
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Path to written file
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by Premiere XML)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = PremiereXMLConfig(**kwargs)
        return PremiereXMLWriter.write(supervisions, output_path, config)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to Premiere Pro XML bytes.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker labels
            **kwargs: Additional config options

        Returns:
            Premiere Pro XML content as bytes
        """
        # Filter out unsupported kwargs (word_level, karaoke, karaoke_config, metadata not supported by Premiere XML)
        kwargs.pop("word_level", None)
        kwargs.pop("karaoke", None)
        kwargs.pop("karaoke_config", None)
        kwargs.pop("metadata", None)
        config = PremiereXMLConfig(**kwargs)
        return PremiereXMLWriter.to_bytes(supervisions, config)


class PremiereXMLReader:
    """Reader for Premiere Pro XML format."""

    @classmethod
    def _frames_to_seconds(cls, frames: int, fps: float) -> float:
        """Convert frames to seconds."""
        if fps <= 0:
            return 0.0
        return frames / fps

    @classmethod
    def _get_fps_from_rate(cls, rate_elem: Optional[ET.Element]) -> float:
        """Extract FPS from rate element."""
        if rate_elem is None:
            return 25.0  # Default

        timebase = rate_elem.find("timebase")
        ntsc = rate_elem.find("ntsc")

        if timebase is None:
            return 25.0

        base = float(timebase.text)
        is_ntsc = ntsc is not None and ntsc.text == "TRUE"

        if is_ntsc:
            if base == 24:
                return 23.976
            elif base == 30:
                return 29.97
            elif base == 60:
                return 59.94

        return base

    @classmethod
    def read(cls, source: str, normalize_text: bool = True) -> List[Supervision]:
        """Read Premiere XML content and return supervisions."""
        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            # Handle potential encoding issues or invalid XML
            return []

        # Find sequence
        sequence = root.find("sequence")
        if sequence is None:
            # Maybe root is sequence?
            if root.tag == "sequence":
                sequence = root
            else:
                return []

        # Get frame rate
        rate = sequence.find("rate")
        fps = cls._get_fps_from_rate(rate)

        supervisions = []

        # Traverse video tracks for clipitems
        # Typically structure: sequence -> media -> video -> track -> clipitem
        media = sequence.find("media")
        if media is None:
            return []

        video = media.find("video")
        if video is None:
            return []

        for track in video.findall("track"):
            for clipitem in track.findall("clipitem"):
                # Check for filter/effect/name = Basic Text or similar
                # We look for text parameters
                text_content = ""

                # Check filter effects
                filter_elem = clipitem.find("filter")
                if filter_elem is not None:
                    effect = filter_elem.find("effect")
                    if effect is not None:
                        # Look for parameter with name "Text"
                        for param in effect.findall("parameter"):
                            name = param.find("name")
                            if name is not None and name.text == "Text":
                                val = param.find("value")
                                if val is not None:
                                    text_content = val.text
                                    break

                if not text_content:
                    # Alternative: check if name is the text (simplistic fallback)
                    # But often name is truncated.
                    pass

                if text_content:
                    start_frame = int(clipitem.find("start").text)
                    end_frame = int(clipitem.find("end").text)
                    # Clipitem timing is relative to sequence logic usually,
                    # but 'start' and 'end' in clipitem are often within the clip's local time?
                    # Wait, standard Premiere XML:
                    # <start> is start time in the sequence timeline (in frames)
                    # <end> is end time in the sequence timeline

                    # NOTE: Sometimes <start> is source start.
                    # We need to check if it's placed on timeline.
                    # Actually <start> and <end> inside clipitem usually define placement on timeline
                    # if NO <in>/<out> complexity overrides it.
                    # Let's assume standard usage from our Writer.

                    start_sec = cls._frames_to_seconds(start_frame, fps)
                    end_sec = cls._frames_to_seconds(end_frame, fps)
                    duration = end_sec - start_sec

                    if duration > 0:
                        supervisions.append(
                            Supervision(
                                id=clipitem.get("id", str(uuid.uuid4())),
                                recording_id="xml_import",
                                start=start_sec,
                                duration=duration,
                                text=text_content.strip() if normalize_text else text_content,
                            )
                        )

        return sorted(supervisions, key=lambda s: s.start)


from .. import register_reader


@register_reader("premiere_xml")
class PremiereXMLReaderHandler(FormatReader):
    """Reader handler for Premiere Pro XML."""

    format_id = "premiere_xml"
    extensions = [".xml"]

    @classmethod
    def can_read(cls, source) -> bool:
        """Check if source is Premiere Pro XML format.

        Premiere XML files contain <xmeml> root element.
        """
        # Check extension first
        if cls.is_content(source):
            content = source[:1024] if len(source) > 1024 else source
        else:
            path_str = str(source).lower()
            if not path_str.endswith(".xml"):
                return False
            # Read file content for detection
            try:
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read(1024)
            except Exception:
                return False

        # Check for xmeml root element
        return "<xmeml" in content.lower()

    @classmethod
    def read(cls, source: Union[Pathlike, str], normalize_text: bool = True, **kwargs) -> List[Supervision]:
        if isinstance(source, (str, Path)) and not cls.is_content(source):
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(source)

        return PremiereXMLReader.read(content, normalize_text=normalize_text)
