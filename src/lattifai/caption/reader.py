from abc import ABCMeta
from pathlib import Path
from typing import List, Literal, Optional, Union

from lhotse.utils import Pathlike

from ..config.caption import InputCaptionFormat, OutputCaptionFormat
from .supervision import Supervision
from .text_parser import NORMALIZE_TEXT
from .text_parser import normalize_text as normalize_text_fn
from .text_parser import parse_speaker_text


class CaptionReader(ABCMeta):
    """Parser for converting different caption formats to List[Supervision]."""

    @classmethod
    def read(
        cls, caption: Pathlike, format: Optional[InputCaptionFormat] = None, normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        """Parse text and convert to Lhotse List[Supervision].

        Args:
            text: Input text to parse. Can be either:
                - str: Direct text content to parse
                - Path: File path to read and parse
            format: Input text format (txt, srt, vtt, ass, textgrid)

        Returns:
            Parsed text in Lhotse Cut
        """
        if not format and Path(str(caption)).exists():
            format = Path(str(caption)).suffix.lstrip(".").lower()
        elif format:
            format = format.lower()

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
                supervisions = cls._parse_caption(caption, format=format, normalize_text=normalize_text)
            except Exception as e:
                print(f"Failed to parse caption with Format: {format}, Exception: {e}, trying 'gemini' parser.")
                from .gemini_reader import GeminiReader

                supervisions = GeminiReader.extract_for_alignment(caption)

        return supervisions

    @classmethod
    def _parse_caption(
        cls, caption: Pathlike, format: Optional[OutputCaptionFormat], normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
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
        return supervisions
