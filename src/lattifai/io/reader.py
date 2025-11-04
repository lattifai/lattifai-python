from abc import ABCMeta
from pathlib import Path
from typing import List, Literal, Optional, Union

from lhotse.utils import Pathlike

from .supervision import Supervision
from .text_parser import parse_speaker_text

SubtitleFormat = Literal["txt", "srt", "vtt", "ass", "auto"]


class SubtitleReader(ABCMeta):
    """Parser for converting different subtitle formats to List[Supervision]."""

    @classmethod
    def read(cls, subtitle: Pathlike, format: Optional[SubtitleFormat] = None) -> List[Supervision]:
        """Parse text and convert to Lhotse List[Supervision].

        Args:
            text: Input text to parse. Can be either:
                - str: Direct text content to parse
                - Path: File path to read and parse
            format: Input text format (txt, srt, vtt, ass, textgrid)

        Returns:
            Parsed text in Lhotse Cut
        """
        if not format and Path(str(subtitle)).exists():
            format = Path(str(subtitle)).suffix.lstrip(".").lower()
        elif format:
            format = format.lower()

        if format == "gemini" or str(subtitle).endswith("Gemini.md"):
            from .gemini_reader import GeminiReader

            supervisions = GeminiReader.extract_for_alignment(subtitle)
        elif format == "txt" or (format == "auto" and str(subtitle)[-4:].lower() == ".txt"):
            if not Path(str(subtitle)).exists():  # str
                lines = [line.strip() for line in str(subtitle).split("\n")]
            else:  # file
                path_str = str(subtitle)
                with open(path_str, encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
            supervisions = [Supervision(text=line) for line in lines if line]
        else:
            try:
                supervisions = cls._parse_subtitle(subtitle, format=format)
            except Exception as e:
                print(f"Failed to parse subtitle with Format: {format}, Exception: {e}, trying 'gemini' parser.")
                from .gemini_reader import GeminiReader

                supervisions = GeminiReader.extract_for_alignment(subtitle)

        return supervisions

    @classmethod
    def _parse_subtitle(cls, subtitle: Pathlike, format: Optional[SubtitleFormat]) -> List[Supervision]:
        import pysubs2

        try:
            subs: pysubs2.SSAFile = pysubs2.load(
                subtitle, encoding="utf-8", format_=format if format != "auto" else None
            )  # file
        except IOError:
            try:
                subs: pysubs2.SSAFile = pysubs2.SSAFile.from_string(
                    subtitle, format_=format if format != "auto" else None
                )  # str
            except Exception as e:
                del e
                subs: pysubs2.SSAFile = pysubs2.load(subtitle, encoding="utf-8")  # auto detect format

        supervisions = []
        for event in subs.events:
            # NOT apply text_parser.py:normalize_html_text here, to keep original text in subtitles
            speaker, text = parse_speaker_text(event.text)
            supervisions.append(
                Supervision(
                    text=text,
                    speaker=speaker,
                    start=event.start / 1000.0 if event.start is not None else None,
                    duration=(event.end - event.start) / 1000.0 if event.end is not None else None,
                )
            )
        return supervisions
