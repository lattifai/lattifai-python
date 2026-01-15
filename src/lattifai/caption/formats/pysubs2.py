"""Standard subtitle formats using pysubs2 library.

Handles: SRT, VTT, ASS, SSA, SUB (MicroDVD), SAMI/SMI
"""

from pathlib import Path
from typing import Dict, List, Optional

import pysubs2

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler
from .karaoke import KaraokeConfig, KaraokeStyle


class Pysubs2Format(FormatHandler):
    """Base class for formats handled by pysubs2."""

    # Subclasses should set these
    pysubs2_format: str = ""

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read caption using pysubs2."""
        try:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source, format_=cls.pysubs2_format)
            else:
                subs = pysubs2.load(str(source), encoding="utf-8", format_=cls.pysubs2_format)
        except Exception:
            # Fallback: auto-detect format
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
    def extract_metadata(cls, source, **kwargs) -> Dict[str, str]:
        """Extract metadata from VTT or SRT."""
        import re
        from pathlib import Path

        metadata = {}
        if cls.is_content(source):
            content = source[:4096]
        else:
            path = Path(str(source))
            if not path.exists():
                return {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(4096)
            except Exception:
                return {}

        # WebVTT metadata extraction
        if cls.pysubs2_format == "vtt" or (isinstance(source, str) and source.startswith("WEBVTT")):
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

        # SRT doesn't have standard metadata, but check for BOM
        elif cls.pysubs2_format == "srt":
            if content.startswith("\ufeff"):
                metadata["encoding"] = "utf-8-sig"

        return metadata

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        fps: float = 25.0,
        **kwargs,
    ) -> Path:
        """Write caption using pysubs2."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker, fps=fps, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        fps: float = 25.0,
        **kwargs,
    ) -> bytes:
        """Convert to bytes using pysubs2."""
        subs = pysubs2.SSAFile()

        for sup in supervisions:
            # Extract word-level alignment if present
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

            if word_items:
                speaker = sup.speaker if cls._should_include_speaker(sup, include_speaker) else ""
                for word in word_items:
                    subs.append(
                        pysubs2.SSAEvent(
                            start=int(word.start * 1000),
                            end=int(word.end * 1000),
                            text=word.symbol,
                            name=speaker,
                        )
                    )
            else:
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

        # MicroDVD format requires framerate
        if cls.pysubs2_format == "microdvd":
            return subs.to_string(format_=cls.pysubs2_format, fps=fps).encode("utf-8")

        return subs.to_string(format_=cls.pysubs2_format).encode("utf-8")


@register_format("srt")
class SRTFormat(Pysubs2Format):
    """SRT (SubRip) format - the most widely used subtitle format."""

    extensions = [".srt"]
    pysubs2_format = "srt"
    description = "SubRip Subtitle format - universal compatibility"

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        use_bom: bool = False,
        **kwargs,
    ) -> bytes:
        """Generate SRT with proper formatting (comma for milliseconds)."""
        content = super().to_bytes(supervisions, include_speaker=include_speaker, **kwargs)

        # Add BOM if requested (for Windows compatibility)
        if use_bom:
            content = b"\xef\xbb\xbf" + content

        return content


@register_format("vtt")
class VTTFormat(Pysubs2Format):
    """WebVTT format for web video."""

    extensions = [".vtt"]
    pysubs2_format = "vtt"
    description = "Web Video Text Tracks - HTML5 standard"


@register_format("ass")
class ASSFormat(Pysubs2Format):
    """Advanced SubStation Alpha format with karaoke support."""

    extensions = [".ass"]
    pysubs2_format = "ass"
    description = "Advanced SubStation Alpha - rich styling support"

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        fps: float = 25.0,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to ASS bytes with optional karaoke tags.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker in output
            fps: Frames per second (not used for ASS)
            word_level: If True and alignment data exists, generate karaoke tags
            karaoke_config: Karaoke style configuration

        Returns:
            ASS content as bytes
        """
        # If word_level is False, use original behavior
        if not word_level:
            return super().to_bytes(supervisions, include_speaker=include_speaker, fps=fps, **kwargs)

        # Check if any supervision has word-level alignment
        has_alignment = any(getattr(sup, "alignment", None) and sup.alignment.get("word") for sup in supervisions)

        # If no alignment data, fallback to original behavior
        if not has_alignment:
            return super().to_bytes(supervisions, include_speaker=include_speaker, fps=fps, **kwargs)

        # Get karaoke config with defaults
        if karaoke_config is None:
            karaoke_config = KaraokeConfig()

        style = karaoke_config.style

        # Create ASS file with karaoke style
        subs = pysubs2.SSAFile()
        subs.styles["Karaoke"] = cls._create_karaoke_style(style)

        for sup in supervisions:
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

            if word_items:
                # Build karaoke text for the entire line
                karaoke_text = cls._build_karaoke_text(word_items, style.effect)

                # Create single event with karaoke tags
                subs.append(
                    pysubs2.SSAEvent(
                        start=int(sup.start * 1000),
                        end=int(sup.end * 1000),
                        text=karaoke_text,
                        style="Karaoke",
                    )
                )
            else:
                # No alignment for this supervision, use plain text
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

        return subs.to_string(format_="ass").encode("utf-8")

    @classmethod
    def _create_karaoke_style(cls, style: KaraokeStyle) -> pysubs2.SSAStyle:
        """Create pysubs2 SSAStyle from KaraokeStyle config.

        Args:
            style: KaraokeStyle configuration

        Returns:
            pysubs2.SSAStyle object
        """
        # Convert int alignment to pysubs2.Alignment enum
        alignment = pysubs2.Alignment(style.alignment)

        return pysubs2.SSAStyle(
            fontname=style.font_name,
            fontsize=style.font_size,
            primarycolor=cls._hex_to_ass_color(style.primary_color),
            secondarycolor=cls._hex_to_ass_color(style.secondary_color),
            outlinecolor=cls._hex_to_ass_color(style.outline_color),
            backcolor=cls._hex_to_ass_color(style.back_color),
            bold=style.bold,
            italic=style.italic,
            outline=style.outline_width,
            shadow=style.shadow_depth,
            alignment=alignment,
            marginl=style.margin_l,
            marginr=style.margin_r,
            marginv=style.margin_v,
        )

    @staticmethod
    def _hex_to_ass_color(hex_color: str) -> pysubs2.Color:
        """Convert #RRGGBB to pysubs2 Color.

        ASS uses &HAABBGGRR format (reversed RGB with alpha).

        Args:
            hex_color: Color in #RRGGBB format

        Returns:
            pysubs2.Color object
        """
        # Remove # prefix if present
        hex_color = hex_color.lstrip("#")

        # Parse RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return pysubs2.Color(r=r, g=g, b=b, a=0)

    @staticmethod
    def _build_karaoke_text(words: list, effect: str = "sweep") -> str:
        """Build karaoke tag text.

        Args:
            words: List of AlignmentItem objects
            effect: Karaoke effect type ("sweep", "instant", "outline")

        Returns:
            Text with karaoke tags, e.g. "{\\kf45}Hello {\\kf55}world"
        """
        tag_map = {"sweep": "kf", "instant": "k", "outline": "ko"}
        tag = tag_map.get(effect, "kf")

        parts = []
        for word in words:
            # Duration in centiseconds (multiply by 100)
            centiseconds = int(word.duration * 100)
            parts.append(f"{{\\{tag}{centiseconds}}}{word.symbol}")

        return " ".join(parts)


@register_format("ssa")
class SSAFormat(Pysubs2Format):
    """SubStation Alpha format (predecessor to ASS)."""

    extensions = [".ssa"]
    pysubs2_format = "ssa"
    description = "SubStation Alpha - legacy format"


@register_format("sub")
class MicroDVDFormat(Pysubs2Format):
    """MicroDVD format (frame-based)."""

    extensions = [".sub"]
    pysubs2_format = "microdvd"
    description = "MicroDVD - frame-based subtitle format"


@register_format("sami")
class SAMIFormat(Pysubs2Format):
    """SAMI (Synchronized Accessible Media Interchange) format."""

    extensions = [".smi", ".sami"]
    pysubs2_format = "sami"
    description = "SAMI - Microsoft format for accessibility"


# Register alias for SMI extension
@register_format("smi")
class SMIFormat(SAMIFormat):
    """SMI format (alias for SAMI)."""

    pass
