"""TTML/IMSC1/EBU-TT-D format handler.

TTML (Timed Text Markup Language) is a W3C standard used by:
- Netflix (IMSC1 profile)
- European broadcasters (EBU-TT-D profile)
- IMF workflows
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from xml.dom import minidom

from ..supervision import Supervision
from . import register_writer
from .base import FormatWriter

# XML namespaces
TTML_NS = "http://www.w3.org/ns/ttml"
TTML_STYLE_NS = "http://www.w3.org/ns/ttml#styling"
TTML_PARAM_NS = "http://www.w3.org/ns/ttml#parameter"
XML_NS = "http://www.w3.org/XML/1998/namespace"


@dataclass
class TTMLStyle:
    """Text style configuration for TTML."""

    font_family: str = "proportionalSansSerif"
    font_size: str = "100%"
    color: str = "#FFFFFF"
    background_color: Optional[str] = "#000000C0"
    text_align: str = "center"
    display_align: str = "after"


@dataclass
class TTMLRegion:
    """Region definition for TTML positioning."""

    id: str = "bottom"
    origin: str = "10% 80%"
    extent: str = "80% 15%"


@dataclass
class TTMLConfig:
    """Configuration for TTML export."""

    profile: str = "imsc1"  # "imsc1", "ebu-tt-d", or "basic"
    default_style: TTMLStyle = field(default_factory=TTMLStyle)
    default_region: TTMLRegion = field(default_factory=TTMLRegion)
    speaker_regions: Dict[str, TTMLRegion] = field(default_factory=dict)
    speaker_styles: Dict[str, TTMLStyle] = field(default_factory=dict)
    language: str = "en"


class TTMLFormatBase(FormatWriter):
    """Base TTML format writer."""

    @classmethod
    def _seconds_to_ttml_time(cls, seconds: float) -> str:
        """Convert seconds to TTML time format (HH:MM:SS.mmm)."""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @classmethod
    def _create_style_element(cls, parent: ET.Element, style_id: str, style: TTMLStyle) -> ET.Element:
        """Create a style element."""
        style_elem = ET.SubElement(parent, f"{{{TTML_NS}}}style")
        style_elem.set(f"{{{XML_NS}}}id", style_id)
        style_elem.set(f"{{{TTML_STYLE_NS}}}fontFamily", style.font_family)
        style_elem.set(f"{{{TTML_STYLE_NS}}}fontSize", style.font_size)
        style_elem.set(f"{{{TTML_STYLE_NS}}}color", style.color)
        style_elem.set(f"{{{TTML_STYLE_NS}}}textAlign", style.text_align)
        style_elem.set(f"{{{TTML_STYLE_NS}}}displayAlign", style.display_align)
        if style.background_color:
            style_elem.set(f"{{{TTML_STYLE_NS}}}backgroundColor", style.background_color)
        return style_elem

    @classmethod
    def _create_region_element(cls, parent: ET.Element, region: TTMLRegion) -> ET.Element:
        """Create a region element."""
        region_elem = ET.SubElement(parent, f"{{{TTML_NS}}}region")
        region_elem.set(f"{{{XML_NS}}}id", region.id)
        region_elem.set(f"{{{TTML_STYLE_NS}}}origin", region.origin)
        region_elem.set(f"{{{TTML_STYLE_NS}}}extent", region.extent)
        return region_elem

    @classmethod
    def _build_ttml(cls, supervisions: List[Supervision], config: TTMLConfig) -> ET.Element:
        """Build TTML document structure."""
        ET.register_namespace("", TTML_NS)
        ET.register_namespace("tts", TTML_STYLE_NS)
        ET.register_namespace("ttp", TTML_PARAM_NS)
        ET.register_namespace("xml", XML_NS)

        root = ET.Element(
            f"{{{TTML_NS}}}tt",
            attrib={
                f"{{{XML_NS}}}lang": config.language,
                f"{{{TTML_PARAM_NS}}}timeBase": "media",
            },
        )

        if config.profile == "imsc1":
            root.set(f"{{{TTML_PARAM_NS}}}profile", "http://www.w3.org/ns/ttml/profile/imsc1/text")
        elif config.profile == "ebu-tt-d":
            root.set(f"{{{TTML_PARAM_NS}}}profile", "urn:ebu:tt:distribution:2014-01")

        # Head section
        head = ET.SubElement(root, f"{{{TTML_NS}}}head")
        styling = ET.SubElement(head, f"{{{TTML_NS}}}styling")
        cls._create_style_element(styling, "default", config.default_style)

        for speaker, style in config.speaker_styles.items():
            style_id = f"speaker_{speaker.replace(' ', '_')}"
            cls._create_style_element(styling, style_id, style)

        layout = ET.SubElement(head, f"{{{TTML_NS}}}layout")
        cls._create_region_element(layout, config.default_region)

        for speaker, region in config.speaker_regions.items():
            cls._create_region_element(layout, region)

        # Body section
        body = ET.SubElement(root, f"{{{TTML_NS}}}body")
        div = ET.SubElement(body, f"{{{TTML_NS}}}div")

        for sup in supervisions:
            begin = cls._seconds_to_ttml_time(sup.start)
            end = cls._seconds_to_ttml_time(sup.end)

            p = ET.SubElement(div, f"{{{TTML_NS}}}p")
            p.set("begin", begin)
            p.set("end", end)

            if sup.speaker and sup.speaker in config.speaker_regions:
                p.set("region", config.speaker_regions[sup.speaker].id)
            else:
                p.set("region", config.default_region.id)

            if sup.speaker and sup.speaker in config.speaker_styles:
                style_id = f"speaker_{sup.speaker.replace(' ', '_')}"
                p.set("style", style_id)
            else:
                p.set("style", "default")

            if sup.speaker and config.profile != "basic":
                span = ET.SubElement(p, f"{{{TTML_NS}}}span")
                span.set(f"{{{TTML_STYLE_NS}}}fontWeight", "bold")
                span.text = f"{sup.speaker}: "
                span.tail = sup.text.strip() if sup.text else ""
            else:
                p.text = sup.text.strip() if sup.text else ""

        return root

    @classmethod
    def _prettify_xml(cls, element: ET.Element) -> str:
        """Convert XML element to pretty-printed string."""
        rough_string = ET.tostring(element, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="  ")
        lines = [line for line in pretty.split("\n") if line.strip()]
        return "\n".join(lines)


@register_writer("ttml")
class TTMLFormat(TTMLFormatBase):
    """Standard TTML format."""

    extensions = [".ttml", ".xml"]
    description = "Timed Text Markup Language - W3C standard"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        config: Optional[TTMLConfig] = None,
        **kwargs,
    ) -> Path:
        """Write TTML format."""
        if config is None:
            config = TTMLConfig()

        output_path = Path(output_path)
        if output_path.suffix.lower() not in [".ttml", ".xml"]:
            output_path = output_path.with_suffix(".ttml")

        root = cls._build_ttml(supervisions, config)
        xml_content = cls._prettify_xml(root)

        output_path.write_text(xml_content, encoding="utf-8")
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        config: Optional[TTMLConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to TTML format bytes."""
        if config is None:
            config = TTMLConfig()

        root = cls._build_ttml(supervisions, config)
        xml_content = cls._prettify_xml(root)
        return xml_content.encode("utf-8")

    @classmethod
    def write_imsc1(
        cls,
        supervisions: List[Supervision],
        output_path,
        language: str = "en",
        **kwargs,
    ) -> Path:
        """Convenience method to write IMSC1 format."""
        config = TTMLConfig(profile="imsc1", language=language)
        return cls.write(supervisions, output_path, config=config, **kwargs)

    @classmethod
    def write_ebu_tt_d(
        cls,
        supervisions: List[Supervision],
        output_path,
        language: str = "en",
        **kwargs,
    ) -> Path:
        """Convenience method to write EBU-TT-D format."""
        config = TTMLConfig(profile="ebu-tt-d", language=language)
        return cls.write(supervisions, output_path, config=config, **kwargs)


@register_writer("imsc1")
class IMSC1Format(TTMLFormatBase):
    """IMSC1 format - Netflix/streaming profile."""

    extensions = [".ttml"]
    description = "IMSC1 - Netflix/streaming TTML profile"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        language: str = "en",
        **kwargs,
    ) -> Path:
        """Write IMSC1 format."""
        config = TTMLConfig(profile="imsc1", language=language)
        return TTMLFormat.write(supervisions, output_path, include_speaker, config, **kwargs)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        language: str = "en",
        **kwargs,
    ) -> bytes:
        """Convert to IMSC1 format bytes."""
        config = TTMLConfig(profile="imsc1", language=language)
        return TTMLFormat.to_bytes(supervisions, include_speaker, config, **kwargs)


@register_writer("ebu_tt_d")
class EBUTD_Format(TTMLFormatBase):
    """EBU-TT-D format - European broadcast profile."""

    extensions = [".ttml"]
    description = "EBU-TT-D - European broadcast TTML profile"

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        language: str = "en",
        **kwargs,
    ) -> Path:
        """Write EBU-TT-D format."""
        config = TTMLConfig(profile="ebu-tt-d", language=language)
        return TTMLFormat.write(supervisions, output_path, include_speaker, config, **kwargs)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        language: str = "en",
        **kwargs,
    ) -> bytes:
        """Convert to EBU-TT-D format bytes."""
        config = TTMLConfig(profile="ebu-tt-d", language=language)
        return TTMLFormat.to_bytes(supervisions, include_speaker, config, **kwargs)


# Export config classes
__all__ = ["TTMLFormat", "IMSC1Format", "EBUTD_Format", "TTMLConfig", "TTMLStyle", "TTMLRegion"]
