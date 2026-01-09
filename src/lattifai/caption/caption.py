"""Caption data structure for storing subtitle information with metadata."""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

from lhotse.utils import Pathlike
from tgt import TextGrid

from ..config.caption import InputCaptionFormat, OutputCaptionFormat  # noqa: F401
from .formats import detect_format, get_reader, get_writer
from .supervision import Supervision

DiarizationOutput = TypeVar("DiarizationOutput")


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
    speaker_diarization: Optional[DiarizationOutput] = None
    # Alignment results
    alignments: List[Supervision] = field(default_factory=list)

    language: Optional[str] = None
    kind: Optional[str] = None
    source_format: Optional[str] = None
    source_path: Optional[Pathlike] = None
    word_level: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of supervision segments."""
        return len(self.supervisions or self.transcription)

    def __iter__(self):
        """Iterate over supervision segments."""
        return iter(self.supervisions)

    def __getitem__(self, index):
        """Get supervision segment by index."""
        return self.supervisions[index]

    def __bool__(self) -> bool:
        """Return True if caption has supervisions."""
        return self.__len__() > 0

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
                alignment=getattr(sup, "alignment", None),
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

    def to_string(self, format: str = "srt") -> str:
        """
        Return caption content in specified format.

        Args:
            format: Output format (e.g., 'srt', 'vtt', 'ass')

        Returns:
            String containing formatted captions
        """
        return self.to_bytes(output_format=format).decode("utf-8")

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
    def from_string(
        cls,
        content: str,
        format: str,
        normalize_text: bool = True,
    ) -> "Caption":
        """
        Create Caption from string content.

        Args:
            content: Caption content as string
            format: Caption format (e.g., 'srt', 'vtt', 'ass')
            normalize_text: Whether to normalize text during reading

        Returns:
            New Caption instance

        Example:
            >>> srt_content = \"\"\"1
            ... 00:00:00,000 --> 00:00:02,000
            ... Hello world\"\"\"
            >>> caption = Caption.from_string(srt_content, format=\"srt\")
        """
        buffer = io.StringIO(content)
        return cls.read(buffer, format=format, normalize_text=normalize_text)

    def to_bytes(self, output_format: Optional[str] = None, include_speaker_in_text: bool = True) -> bytes:
        """
        Convert caption to bytes.

        Args:
            output_format: Output format (e.g., 'srt', 'vtt', 'ass'). Defaults to source_format or 'srt'
            include_speaker_in_text: Whether to include speaker labels in text

        Returns:
            Caption content as bytes

        Example:
            >>> caption = Caption.read(\"input.srt\")
            >>> # Get as bytes in original format
            >>> data = caption.to_bytes()
            >>> # Get as bytes in specific format
            >>> vtt_data = caption.to_bytes(output_format=\"vtt\")
        """
        return self.write(None, output_format=output_format, include_speaker_in_text=include_speaker_in_text)

    @classmethod
    def from_transcription_results(
        cls,
        transcription: List[Supervision],
        audio_events: Optional[TextGrid] = None,
        speaker_diarization: Optional[DiarizationOutput] = None,
        language: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from transcription results including audio events and diarization.

        Args:
            transcription: List of transcription supervision segments
            audio_events: Optional TextGrid with audio event detection results
            speaker_diarization: Optional DiarizationOutput with speaker diarization results
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
        path: Union[Pathlike, io.BytesIO, io.StringIO],
        format: Optional[str] = None,
        normalize_text: bool = True,
    ) -> "Caption":
        """
        Read caption file or in-memory data and return Caption object.

        Args:
            path: Path to caption file, or BytesIO/StringIO object with caption content
            format: Caption format (auto-detected if not provided, required for in-memory data)
            normalize_text: Whether to normalize text during reading

        Returns:
            Caption object containing supervisions and metadata
        """
        # Detect format if not provided
        if not format:
            if isinstance(path, (io.BytesIO, io.StringIO)):
                raise ValueError("format parameter is required when reading from BytesIO/StringIO")
            format = detect_format(str(path))

        if not format:
            # Fallback to extension
            if not isinstance(path, (io.BytesIO, io.StringIO)):
                format = Path(str(path)).suffix.lstrip(".").lower()

        if not format:
            format = "srt"  # Last resort default

        # Get content if it's an in-memory buffer
        source = path
        if isinstance(path, io.BytesIO):
            source = path.read().decode("utf-8")
        elif isinstance(path, io.StringIO):
            source = path.read()

        # Reset buffer position if it was a stream
        if isinstance(path, (io.BytesIO, io.StringIO)):
            path.seek(0)

        # Get reader and perform extraction
        reader_cls = get_reader(format)
        if not reader_cls:
            # Use pysubs2 as a generic fallback if no specific reader exists
            from .formats.pysubs2 import Pysubs2Format

            reader_cls = Pysubs2Format

        supervisions = reader_cls.read(source, normalize_text=normalize_text)
        metadata = reader_cls.extract_metadata(source)

        # Create Caption object
        source_path = None
        if isinstance(path, (str, Path)) and not ("\n" in str(path) or len(str(path)) > 500):
            try:
                p = Path(str(path))
                if p.exists():
                    source_path = str(p)
            except (OSError, ValueError):
                pass

        return cls(
            supervisions=supervisions,
            language=metadata.get("language"),
            kind=metadata.get("kind"),
            source_format=format,
            source_path=source_path,
            metadata=metadata,
        )

    def write(
        self,
        path: Union[Pathlike, io.BytesIO, None] = None,
        output_format: Optional[str] = None,
        include_speaker_in_text: bool = True,
    ) -> Union[Pathlike, bytes]:
        """
        Write caption to file or return as bytes.

        Args:
            path: Path to output caption file, BytesIO object, or None to return bytes
            output_format: Output format (e.g., 'srt', 'vtt', 'ass')
            include_speaker_in_text: Whether to include speaker labels in text

        Returns:
            Path to the written file if path is a file path, or bytes if path is BytesIO/None
        """
        if self.alignments:
            supervisions = self.alignments
        elif self.supervisions:
            supervisions = self.supervisions
        else:
            supervisions = self.transcription

        # Determine output format
        if output_format:
            output_format = output_format.lower()
        elif isinstance(path, (io.BytesIO, type(None))):
            output_format = self.source_format or "srt"
        else:
            output_format = detect_format(str(path)) or Path(str(path)).suffix.lstrip(".").lower() or "srt"

        # Special casing for professional formats as before
        ext = output_format
        if isinstance(path, (str, Path)):
            path_str = str(path)
            if path_str.endswith("_avid.txt"):
                ext = "avid_ds"
            elif "audition" in path_str.lower() and path_str.endswith(".csv"):
                ext = "audition_csv"
            elif "edimarker" in path_str.lower() and path_str.endswith(".csv"):
                ext = "edimarker_csv"
            elif "imsc" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "imsc1"
            elif "ebu" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "ebu_tt_d"

        # Use YouTube VTT if word-level is requested for VTT output
        if ext == "vtt" and self.word_level:
            ext = "youtube_vtt"

        writer_cls = get_writer(ext)
        if not writer_cls:
            from .formats.pysubs2 import Pysubs2Format

            writer_cls = Pysubs2Format

        if isinstance(path, (str, Path)):
            return writer_cls.write(supervisions, path, include_speaker=include_speaker_in_text)

        content = writer_cls.to_bytes(supervisions, include_speaker=include_speaker_in_text)
        if isinstance(path, io.BytesIO):
            path.write(content)
            path.seek(0)
        return content

    def read_speaker_diarization(
        self,
        path: Pathlike,
    ) -> "DiarizationOutput":
        """
        Read speaker diarization TextGrid from file.
        """
        from lattifai_core.diarization import DiarizationOutput

        self.speaker_diarization = DiarizationOutput.read(path)
        return self.speaker_diarization

    def write_speaker_diarization(
        self,
        path: Pathlike,
    ) -> Pathlike:
        """
        Write speaker diarization TextGrid to file.
        """
        if not self.speaker_diarization:
            raise ValueError("No speaker diarization data to write.")

        self.speaker_diarization.write(path)
        return path

    def __repr__(self) -> str:
        """String representation of Caption."""
        lang = f"lang={self.language}" if self.language else "lang=unknown"
        kind_str = f"kind={self.kind}" if self.kind else ""
        parts = [f"Caption({len(self.supervisions or self.transcription)} segments", lang]
        if kind_str:
            parts.append(kind_str)
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        return ", ".join(parts) + ")"
