"""Caption data structure for storing subtitle information with metadata."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike
from tgt import TextGrid

from ..config.caption import InputCaptionFormat, OutputCaptionFormat
from .supervision import Supervision
from .text_parser import normalize_text as normalize_text_fn
from .text_parser import parse_speaker_text


@dataclass
class Caption:
    """
    Container for caption/subtitle data with metadata.

    This class encapsulates a list of supervisions (subtitle segments) along with
    metadata such as language, kind, format information, and source file details.

    Attributes:
        supervisions: List of supervision segments containing text and timing information
        language: Language code (e.g., 'en', 'zh', 'es')
        kind: Caption kind/type (e.g., 'captions', 'subtitles', 'descriptions')
        source_format: Original format of the caption file (e.g., 'vtt', 'srt', 'json')
        source_path: Path to the source caption file
        metadata: Additional custom metadata as key-value pairs
    """

    # read from subtitle file
    supervisions: List[Supervision] = field(default_factory=list)
    # Transcription results
    transcription: List[Supervision] = field(default_factory=list)
    # Audio Event Detection results
    audio_events: Optional[TextGrid] = None
    # Speaker Diarization results
    speaker_diarization: Optional[TextGrid] = None
    # Alignment results
    alignments: List[Supervision] = field(default_factory=list)

    language: Optional[str] = None
    kind: Optional[str] = None
    source_format: Optional[str] = None
    source_path: Optional[Pathlike] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of supervision segments."""
        return len(self.supervisions) or len(self.transcripts)

    def __iter__(self):
        """Iterate over supervision segments."""
        return iter(self.supervisions)

    def __getitem__(self, index):
        """Get supervision segment by index."""
        return self.supervisions[index]

    def __bool__(self) -> bool:
        """Return True if caption has supervisions."""
        return len(self.supervisions) > 0

    @property
    def is_empty(self) -> bool:
        """Check if caption has no supervisions."""
        return len(self.supervisions) == 0

    @property
    def duration(self) -> Optional[float]:
        """
        Get total duration of the caption in seconds.

        Returns:
            Total duration from first to last supervision, or None if empty
        """
        if not self.supervisions:
            return None
        return self.supervisions[-1].end - self.supervisions[0].start

    @property
    def start_time(self) -> Optional[float]:
        """Get start time of first supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[0].start

    @property
    def end_time(self) -> Optional[float]:
        """Get end time of last supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[-1].end

    def append(self, supervision: Supervision) -> None:
        """Add a supervision segment to the caption."""
        self.supervisions.append(supervision)

    def extend(self, supervisions: List[Supervision]) -> None:
        """Add multiple supervision segments to the caption."""
        self.supervisions.extend(supervisions)

    def filter_by_speaker(self, speaker: str) -> "Caption":
        """
        Create a new Caption with only supervisions from a specific speaker.

        Args:
            speaker: Speaker identifier to filter by

        Returns:
            New Caption instance with filtered supervisions
        """
        filtered_sups = [sup for sup in self.supervisions if sup.speaker == speaker]
        return Caption(
            supervisions=filtered_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def get_speakers(self) -> List[str]:
        """
        Get list of unique speakers in the caption.

        Returns:
            Sorted list of unique speaker identifiers
        """
        speakers = {sup.speaker for sup in self.supervisions if sup.speaker}
        return sorted(speakers)

    def shift_time(self, seconds: float) -> "Caption":
        """
        Create a new Caption with all timestamps shifted by given seconds.

        Args:
            seconds: Number of seconds to shift (positive delays, negative advances)

        Returns:
            New Caption instance with shifted timestamps
        """
        shifted_sups = [
            Supervision(
                text=sup.text,
                start=sup.start + seconds,
                duration=sup.duration,
                speaker=sup.speaker,
                id=sup.id,
                language=sup.language,
                alignment=sup.alignment if hasattr(sup, "alignment") else None,
                custom=sup.custom,
            )
            for sup in self.supervisions
        ]

        return Caption(
            supervisions=shifted_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict:
        """
        Convert Caption to dictionary representation.

        Returns:
            Dictionary with caption data and metadata
        """
        return {
            "supervisions": [sup.to_dict() for sup in self.supervisions],
            "language": self.language,
            "kind": self.kind,
            "source_format": self.source_format,
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
            "duration": self.duration,
            "num_segments": len(self.supervisions),
            "speakers": self.get_speakers(),
        }

    @classmethod
    def from_supervisions(
        cls,
        supervisions: List[Supervision],
        language: Optional[str] = None,
        kind: Optional[str] = None,
        source_format: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from a list of supervisions.

        Args:
            supervisions: List of supervision segments
            language: Language code
            kind: Caption kind/type
            source_format: Original format
            source_path: Source file path
            metadata: Additional metadata

        Returns:
            New Caption instance
        """
        return cls(
            supervisions=supervisions,
            language=language,
            kind=kind,
            source_format=source_format,
            source_path=source_path,
            metadata=metadata or {},
        )

    @classmethod
    def from_transcription_results(
        cls,
        transcription: List[Supervision],
        audio_events: Optional[TextGrid] = None,
        speaker_diarization: Optional[TextGrid] = None,
        language: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from transcription results including audio events and diarization.

        Args:
            transcription: List of transcription supervision segments
            audio_events: Optional TextGrid with audio event detection results
            speaker_diarization: Optional TextGrid with speaker diarization results
            language: Language code
            source_path: Source file path
            metadata: Additional metadata

        Returns:
            New Caption instance with transcription data
        """
        return cls(
            transcription=transcription,
            audio_events=audio_events,
            speaker_diarization=speaker_diarization,
            language=language,
            kind="transcription",
            source_format="asr",
            source_path=source_path,
            metadata=metadata or {},
        )

    @classmethod
    def read(
        cls,
        path: Pathlike,
        format: Optional[str] = None,
        normalize_text: bool = False,
    ) -> "Caption":
        """
        Read caption file and return Caption object.

        Args:
            path: Path to caption file
            format: Caption format (auto-detected if not provided)
            normalize_text: Whether to normalize text during reading

        Returns:
            Caption object containing supervisions and metadata

        Example:
            >>> caption = Caption.read("subtitles.srt")
            >>> print(f"Loaded {len(caption)} segments")
        """
        caption_path = Path(str(path)) if not isinstance(path, Path) else path

        # Detect format if not provided
        if not format and caption_path.exists():
            format = caption_path.suffix.lstrip(".").lower()
        elif format:
            format = format.lower()

        # Extract metadata from file
        metadata = cls._extract_metadata(path, format)

        # Parse supervisions
        supervisions = cls._parse_supervisions(path, format, normalize_text)

        # Create Caption object
        return cls(
            supervisions=supervisions,
            language=metadata.get("language"),
            kind=metadata.get("kind"),
            source_format=format,
            source_path=str(caption_path) if caption_path.exists() else None,
            metadata=metadata,
        )

    def write(
        self,
        path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> Pathlike:
        """
        Write caption to file.

        Args:
            path: Path to output caption file
            include_speaker_in_text: Whether to include speaker labels in text

        Returns:
            Path to the written file

        Example:
            >>> caption = Caption.read("input.srt")
            >>> caption.write("output.vtt", include_speaker_in_text=False)
        """
        if self.alignments:
            alignments = self.alignments
        else:
            alignments = self.supervisions

        if not alignments:
            alignments = self.transcription

        return self._write_caption(alignments, path, include_speaker_in_text)

    @staticmethod
    def _parse_alignment_from_supervision(supervision: Any) -> Optional[List[AlignmentItem]]:
        """
        Extract word-level alignment items from Supervision object.

        Args:
            supervision: Supervision object with potential alignment data

        Returns:
            List of AlignmentItem objects, or None if no alignment data present
        """
        if not hasattr(supervision, "alignment") or not supervision.alignment:
            return None

        if "word" not in supervision.alignment:
            return None

        return supervision.alignment["word"]

    @classmethod
    def _write_caption(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> Pathlike:
        """
        Write caption to file in various formats.

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output file
            include_speaker_in_text: Whether to include speaker in text

        Returns:
            Path to written file
        """
        if str(output_path)[-4:].lower() == ".txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for sup in alignments:
                    word_items = cls._parse_alignment_from_supervision(sup)
                    if word_items:
                        for item in word_items:
                            f.write(f"[{item.start:.2f}-{item.end:.2f}] {item.symbol}\n")
                    else:
                        if include_speaker_in_text and sup.speaker is not None:
                            text = f"{sup.speaker} {sup.text}"
                        else:
                            text = sup.text
                        f.write(f"[{sup.start:.2f}-{sup.end:.2f}] {text}\n")

        elif str(output_path)[-5:].lower() == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                # Enhanced JSON export with word-level alignment
                json_data = []
                for sup in alignments:
                    sup_dict = sup.to_dict()
                    json_data.append(sup_dict)
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        elif str(output_path).lower().endswith(".textgrid"):
            from tgt import Interval, IntervalTier, TextGrid, write_to_file

            tg = TextGrid()
            supervisions, words, scores = [], [], {"utterances": [], "words": []}
            for supervision in sorted(alignments, key=lambda x: x.start):
                if include_speaker_in_text and supervision.speaker is not None:
                    text = f"{supervision.speaker} {supervision.text}"
                else:
                    text = supervision.text
                supervisions.append(Interval(supervision.start, supervision.end, text or ""))
                # Extract word-level alignment using helper function
                word_items = cls._parse_alignment_from_supervision(supervision)
                if word_items:
                    for item in word_items:
                        words.append(Interval(item.start, item.end, item.symbol))
                        if item.score is not None:
                            scores["words"].append(Interval(item.start, item.end, f"{item.score:.2f}"))
                if supervision.has_custom("score"):
                    scores["utterances"].append(
                        Interval(supervision.start, supervision.end, f"{supervision.score:.2f}")
                    )

            tg.add_tier(IntervalTier(name="utterances", objects=supervisions))
            if words:
                tg.add_tier(IntervalTier(name="words", objects=words))

            if scores["utterances"]:
                tg.add_tier(IntervalTier(name="utterance_scores", objects=scores["utterances"]))
            if scores["words"]:
                tg.add_tier(IntervalTier(name="word_scores", objects=scores["words"]))

            write_to_file(tg, output_path, format="long")

        else:
            import pysubs2

            subs = pysubs2.SSAFile()
            for sup in alignments:
                # Add word-level timing as metadata in the caption text
                word_items = cls._parse_alignment_from_supervision(sup)
                if word_items:
                    for word in word_items:
                        subs.append(
                            pysubs2.SSAEvent(
                                start=int(word.start * 1000),
                                end=int(word.end * 1000),
                                text=word.symbol,
                                name=sup.speaker or "",
                            )
                        )
                else:
                    if include_speaker_in_text and sup.speaker is not None:
                        text = f"{sup.speaker} {sup.text}"
                    else:
                        text = sup.text
                    subs.append(
                        pysubs2.SSAEvent(
                            start=int(sup.start * 1000),
                            end=int(sup.end * 1000),
                            text=text or "",
                            name=sup.speaker or "",
                        )
                    )
            subs.save(output_path)

        return output_path

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
    ) -> List[Supervision]:
        """
        Parse supervisions from caption file.

        Args:
            caption: Caption file path or content
            format: Caption format
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        if format:
            format = format.lower()

        if format == "gemini" or str(caption).endswith("Gemini.md"):
            from .gemini_reader import GeminiReader

            supervisions = GeminiReader.extract_for_alignment(caption)
        elif format and (format == "textgrid" or str(caption).lower().endswith("textgrid")):
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
                    if normalize_text:
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
        """
        Parse caption using pysubs2.

        Args:
            caption: Caption file path or content
            format: Caption format
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
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

        # Parse supervisions
        supervisions = []
        for event in subs.events:
            if normalize_text:
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

    def __repr__(self) -> str:
        """String representation of Caption."""
        lang = f"lang={self.language}" if self.language else "lang=unknown"
        kind_str = f"kind={self.kind}" if self.kind else ""
        parts = [f"Caption({len(self.supervisions)} segments", lang]
        if kind_str:
            parts.append(kind_str)
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        return ", ".join(parts) + ")"
