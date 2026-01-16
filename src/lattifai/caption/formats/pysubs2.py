"""Standard subtitle formats using pysubs2 library.

Handles: SRT, VTT, ASS, SSA, SUB (MicroDVD), SAMI/SMI
"""

from pathlib import Path
from typing import Dict, List, Optional

import pysubs2

from ...config.caption import CaptionStyle, KaraokeConfig
from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


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
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to bytes using pysubs2.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker in output
            fps: Frames per second (for MicroDVD format)
            word_level: If True and alignment exists, output word-per-segment
            karaoke_config: Karaoke configuration. When provided with enabled=True,
                use karaoke styling (format-specific)

        Returns:
            Subtitle content as bytes
        """
        from .base import expand_to_word_supervisions

        # Check if karaoke is enabled
        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # Expand to word-per-segment if word_level=True and karaoke is not enabled
        if word_level and not karaoke_enabled:
            supervisions = expand_to_word_supervisions(supervisions)

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
    """WebVTT format for web video with optional karaoke (YouTube VTT style)."""

    extensions = [".vtt"]
    pysubs2_format = "vtt"
    description = "Web Video Text Tracks - HTML5 standard"

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
        """Convert to VTT bytes with optional karaoke (YouTube VTT style).

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker in output
            fps: Frames per second (not used for VTT)
            word_level: If True and alignment exists, output word-per-segment or karaoke
            karaoke_config: Karaoke configuration. When enabled, output YouTube VTT
                style with word-level timestamps: <00:00:10.559><c> word</c>

        Returns:
            VTT content as bytes
        """
        from .base import expand_to_word_supervisions

        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # If karaoke enabled, output YouTube VTT style
        if word_level and karaoke_enabled:
            return cls._to_youtube_vtt_bytes(supervisions, include_speaker)

        # If word_level only (no karaoke), expand to word-per-segment
        if word_level:
            supervisions = expand_to_word_supervisions(supervisions)

        # Use pysubs2 for standard VTT output
        return super().to_bytes(supervisions, include_speaker=include_speaker, fps=fps, **kwargs)

    @classmethod
    def _to_youtube_vtt_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
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

        lines = ["WEBVTT", ""]

        for sup in sorted(supervisions, key=lambda x: x.start):
            text = sup.text or ""
            alignment = getattr(sup, "alignment", None)
            words = alignment.get("word") if alignment else None

            if words:
                # Use word timestamps for cue timing (more accurate)
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
                # Fallback to segment timing if no word alignment
                lines.append(f"{format_timestamp(sup.start)} --> {format_timestamp(sup.end)}")
                if include_speaker and sup.speaker:
                    text = f"{sup.speaker}: {text}"
                lines.append(text)
            lines.append("")

        return "\n".join(lines).encode("utf-8")


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
            word_level: If True and alignment exists, output word-per-segment or karaoke
            karaoke_config: Karaoke configuration. When provided with enabled=True,
                generate karaoke tags

        Returns:
            ASS content as bytes
        """
        # Check if karaoke is enabled
        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # If word_level is False or karaoke is not enabled, use base class behavior (word-per-segment)
        if not word_level or not karaoke_enabled:
            return super().to_bytes(
                supervisions,
                include_speaker=include_speaker,
                fps=fps,
                word_level=word_level,
                karaoke_config=karaoke_config,
                **kwargs,
            )

        # Check if any supervision has word-level alignment
        has_alignment = any(getattr(sup, "alignment", None) and sup.alignment.get("word") for sup in supervisions)

        # If no alignment data, fallback to base class behavior
        if not has_alignment:
            return super().to_bytes(
                supervisions,
                include_speaker=include_speaker,
                fps=fps,
                word_level=word_level,
                karaoke_config=karaoke_config,
                **kwargs,
            )

        style = karaoke_config.style

        # Create ASS file with karaoke style
        subs = pysubs2.SSAFile()
        subs.styles["Karaoke"] = cls._create_karaoke_style(style)

        for sup in supervisions:
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

            if word_items:
                # Build karaoke text for the entire line
                karaoke_text = cls._build_karaoke_text(word_items, karaoke_config.effect)

                # Use word timestamps for event timing (more accurate)
                event_start = int(word_items[0].start * 1000)
                event_end = int(word_items[-1].end * 1000)

                # Create single event with karaoke tags
                subs.append(
                    pysubs2.SSAEvent(
                        start=event_start,
                        end=event_end,
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
    def _create_karaoke_style(cls, style: CaptionStyle) -> pysubs2.SSAStyle:
        """Create pysubs2 SSAStyle from CaptionStyle config.

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
