import re
from abc import ABCMeta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from lhotse.utils import Pathlike

from ..config.caption import InputCaptionFormat, OutputCaptionFormat
from .caption import Caption
from .supervision import Supervision
from .text_parser import NORMALIZE_TEXT
from .text_parser import normalize_text as normalize_text_fn
from .text_parser import parse_speaker_text


class CaptionReader(ABCMeta):
    """Parser for converting different caption formats to Caption object."""

    @classmethod
    def read(
        cls, caption: Pathlike, format: Optional[InputCaptionFormat] = None, normalize_text: Optional[bool] = False
    ) -> Caption:
        """Parse caption file and convert to Caption object with metadata.

        Args:
            caption: Input caption to parse. Can be either:
                - str: Direct text content to parse or file path
                - Path: File path to read and parse
            format: Input caption format (txt, srt, vtt, ass, textgrid, gemini)
            normalize_text: Whether to normalize text during parsing

        Returns:
            Caption object containing supervisions and metadata
        """
        caption_path = Path(str(caption)) if not isinstance(caption, Path) else caption

        # Detect format if not provided
        if not format and caption_path.exists():
            format = caption_path.suffix.lstrip(".").lower()
        elif format:
            format = format.lower()

        # Extract metadata from file
        metadata = cls._extract_metadata(caption, format)

        # Parse supervisions and extract format-specific data (like ASS styles)
        supervisions, styles, script_info = cls._parse_supervisions(caption, format, normalize_text)

        # Create Caption object
        return Caption(
            supervisions=supervisions,
            language=metadata.get("language"),
            kind=metadata.get("kind"),
            source_format=format,
            source_path=str(caption_path) if caption_path.exists() else None,
            metadata=metadata,
            styles=styles,
            script_info=script_info,
        )

    @classmethod
    def _extract_metadata(cls, caption: Pathlike, format: Optional[str]) -> Dict[str, str]:
        """
        Extract metadata from caption file header.

        Args:
            caption: Caption file path or content
            format: Caption format

        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata = {}
        caption_path = Path(str(caption))

        if not caption_path.exists():
            return metadata

        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                content = f.read(2048)  # Read first 2KB for metadata

            # WebVTT metadata extraction
            if format == "vtt" or content.startswith("WEBVTT"):
                lines = content.split("\n")
                for line in lines[:10]:  # Check first 10 lines
                    line = line.strip()
                    if line.startswith("Kind:"):
                        metadata["kind"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Language:"):
                        metadata["language"] = line.split(":", 1)[1].strip()
                    elif line.startswith("NOTE"):
                        # Extract metadata from NOTE comments
                        match = re.search(r"NOTE\s+(\w+):\s*(.+)", line)
                        if match:
                            key, value = match.groups()
                            metadata[key.lower()] = value.strip()

            # SRT doesn't have standard metadata, but check for BOM
            elif format == "srt":
                if content.startswith("\ufeff"):
                    metadata["encoding"] = "utf-8-sig"

            # TextGrid metadata
            elif format == "textgrid" or caption_path.suffix.lower() == ".textgrid":
                match = re.search(r"xmin\s*=\s*([\d.]+)", content)
                if match:
                    metadata["xmin"] = match.group(1)
                match = re.search(r"xmax\s*=\s*([\d.]+)", content)
                if match:
                    metadata["xmax"] = match.group(1)

        except Exception:
            # If metadata extraction fails, continue with empty metadata
            pass

        return metadata

    @classmethod
    def _parse_supervisions(
        cls, caption: Pathlike, format: Optional[str], normalize_text: Optional[bool] = False
    ) -> Tuple[List[Supervision], Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
        """
        Parse supervisions from caption file and extract format-specific data.

        Args:
            caption: Caption file path or content
            format: Caption format
            normalize_text: Whether to normalize text

        Returns:
            Tuple of (supervisions, styles, script_info)
            - supervisions: List of Supervision objects
            - styles: Format-specific style information (e.g., ASS/SSA styles)
            - script_info: Format-specific script information (e.g., ASS/SSA script info)
        """
        styles = None
        script_info = None

        if format == "gemini" or str(caption).endswith("Gemini.md"):
            from .gemini_reader import GeminiReader

            supervisions = GeminiReader.extract_for_alignment(caption)
        elif format.lower() == "textgrid" or str(caption).lower().endswith("textgrid"):
            # Internel usage
            from tgt import read_textgrid

            tgt = read_textgrid(caption)
            supervisions = []
            for tier in tgt.tiers:
                supervisions.extend(
                    [
                        Supervision(
                            text=interval.text,
                            start=interval.start_time,
                            duration=interval.end_time - interval.start_time,
                            speaker=tier.name,
                        )
                        for interval in tier.intervals
                    ]
                )
            supervisions = sorted(supervisions, key=lambda x: x.start)
        elif format == "txt" or (format == "auto" and str(caption)[-4:].lower() == ".txt"):
            if not Path(str(caption)).exists():  # str
                lines = [line.strip() for line in str(caption).split("\n")]
            else:  # file
                path_str = str(caption)
                with open(path_str, encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
                    if NORMALIZE_TEXT or normalize_text:
                        lines = [normalize_text_fn(line) for line in lines]
            supervisions = [Supervision(text=line) for line in lines if line]
        else:
            try:
                supervisions, styles, script_info = cls._parse_caption(
                    caption, format=format, normalize_text=normalize_text
                )
            except Exception as e:
                print(f"Failed to parse caption with Format: {format}, Exception: {e}, trying 'gemini' parser.")
                from .gemini_reader import GeminiReader

                supervisions = GeminiReader.extract_for_alignment(caption)

        return supervisions, styles, script_info

    @classmethod
    def _parse_caption(
        cls, caption: Pathlike, format: Optional[OutputCaptionFormat], normalize_text: Optional[bool] = False
    ) -> Tuple[List[Supervision], Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
        """
        Parse caption using pysubs2 and extract styles.

        Args:
            caption: Caption file path or content
            format: Caption format
            normalize_text: Whether to normalize text

        Returns:
            Tuple of (supervisions, styles, script_info)
        """
        import pysubs2

        try:
            subs: pysubs2.SSAFile = pysubs2.load(
                caption, encoding="utf-8", format_=format if format != "auto" else None
            )  # file
        except IOError:
            try:
                subs: pysubs2.SSAFile = pysubs2.SSAFile.from_string(
                    caption, format_=format if format != "auto" else None
                )  # str
            except Exception as e:
                del e
                subs: pysubs2.SSAFile = pysubs2.load(caption, encoding="utf-8")  # auto detect format

        # Extract styles (ASS/SSA specific)
        styles = None
        script_info = None

        if hasattr(subs, "styles") and subs.styles:
            # Convert SSAStyle objects to dictionaries
            styles = {}
            for style_name, style in subs.styles.items():
                styles[style_name] = {
                    "fontname": style.fontname,
                    "fontsize": style.fontsize,
                    "primary_color": style.primarycolor,
                    "secondary_color": style.secondarycolor,
                    "outline_color": style.outlinecolor,
                    "back_color": style.backcolor,
                    "bold": style.bold,
                    "italic": style.italic,
                    "underline": style.underline,
                    "strikeout": style.strikeout,
                    "scale_x": style.scalex,
                    "scale_y": style.scaley,
                    "spacing": style.spacing,
                    "angle": style.angle,
                    "border_style": style.borderstyle,
                    "outline": style.outline,
                    "shadow": style.shadow,
                    "alignment": style.alignment,
                    "margin_l": style.marginl,
                    "margin_r": style.marginr,
                    "margin_v": style.marginv,
                    "encoding": style.encoding,
                }

        if hasattr(subs, "info") and subs.info:
            # Extract script info (ASS/SSA specific)
            script_info = dict(subs.info)

        # Parse supervisions
        supervisions = []
        for event in subs.events:
            if NORMALIZE_TEXT or normalize_text:
                event.text = normalize_text_fn(event.text)
            speaker, text = parse_speaker_text(event.text)
            supervisions.append(
                Supervision(
                    text=text,
                    speaker=speaker or event.name,
                    start=event.start / 1000.0 if event.start is not None else None,
                    duration=(event.end - event.start) / 1000.0 if event.end is not None else None,
                )
            )
        return supervisions, styles, script_info
