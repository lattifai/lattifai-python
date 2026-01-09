"""Caption data structure for storing subtitle information with metadata."""

import io
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike
from tgt import TextGrid

from ..config.caption import InputCaptionFormat, OutputCaptionFormat  # noqa: F401
from .parsers.text_parser import normalize_text as normalize_text_fn
from .parsers.text_parser import parse_speaker_text, parse_timestamp_text
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
        import pysubs2

        subs = pysubs2.SSAFile()

        if self.alignments:
            alignments = self.alignments
        else:
            alignments = self.supervisions

        if not alignments:
            alignments = self.transcription

        for sup in alignments:
            # Add word-level timing as metadata in the caption text
            word_items = self._parse_alignment_from_supervision(sup)
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
                subs.append(
                    pysubs2.SSAEvent(
                        start=int(sup.start * 1000),
                        end=int(sup.end * 1000),
                        text=sup.text or "",
                        name=sup.speaker or "",
                    )
                )

        return subs.to_string(format_=format)

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

        Example:
            >>> # From file
            >>> caption = Caption.read("subtitles.srt")
            >>> # From in-memory data
            >>> import io
            >>> data = io.BytesIO(b"1\n00:00:00,000 --> 00:00:02,000\nHello world")
            >>> caption = Caption.read(data, format="srt")
            >>> print(f"Loaded {len(caption)} segments")
        """
        # Handle in-memory data (BytesIO/StringIO)
        if isinstance(path, (io.BytesIO, io.StringIO)):
            if not format:
                raise ValueError("format parameter is required when reading from BytesIO/StringIO")

            # Read content from in-memory buffer
            if isinstance(path, io.BytesIO):
                content = path.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
            else:  # StringIO
                content = path.read()

            # Reset buffer position for potential re-reading
            path.seek(0)

            # Parse supervisions from string content
            supervisions = cls._parse_supervisions(content, format.lower(), normalize_text)

            return cls(
                supervisions=supervisions,
                language=None,
                kind=None,
                source_format=format.lower(),
                source_path=None,
                metadata={},
            )

        # Handle file path (existing logic)
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
        path: Union[Pathlike, io.BytesIO, None] = None,
        output_format: Optional[str] = None,
        include_speaker_in_text: bool = True,
    ) -> Union[Pathlike, bytes]:
        """
        Write caption to file or return as bytes.

        Args:
            path: Path to output caption file, BytesIO object, or None to return bytes
            output_format: Output format (e.g., 'srt', 'vtt', 'ass'). If not provided:
                - For file paths: detected from file extension
                - For BytesIO/None: uses source_format or defaults to 'srt'
            include_speaker_in_text: Whether to include speaker labels in text

        Returns:
            Path to the written file if path is a file path, or bytes if path is BytesIO/None

        Example:
            >>> caption = Caption.read("input.srt")
            >>> # Write to file
            >>> caption.write("output.vtt", include_speaker_in_text=False)
            >>> # Write to BytesIO with explicit format
            >>> import io
            >>> buffer = io.BytesIO()
            >>> caption.write(buffer, output_format="vtt")
            >>> # Get as bytes in specific format
            >>> data = caption.write(None, output_format="ass")
        """
        if self.alignments:
            alignments = self.alignments
        else:
            alignments = self.supervisions

        if not alignments:
            alignments = self.transcription

        # Determine output format
        if output_format:
            # Use explicitly provided format
            output_format = output_format.lower()
        elif isinstance(path, (io.BytesIO, type(None))):
            # Use source_format or default to srt
            output_format = self.source_format or "srt"
        else:
            # Get format from file path
            output_format = Path(str(path)).suffix.lstrip(".").lower() or "srt"

        # Handle in-memory operations without temporary files
        if isinstance(path, (io.BytesIO, type(None))):
            content = self._generate_caption_content(alignments, output_format, include_speaker_in_text)

            if isinstance(path, io.BytesIO):
                # Write to BytesIO
                path.write(content)
                path.seek(0)

            return content

        # Handle file path (existing logic)
        return self._write_caption(alignments, path, include_speaker_in_text, output_format)

    def read_speaker_diarization(
        self,
        path: Pathlike,
    ) -> TextGrid:
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

    @staticmethod
    def _format_sbv_timestamp(seconds: float) -> str:
        """Format timestamp for SBV format as H:MM:SS.mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h}:{m:02d}:{s:02d}.{ms:03d}"

    @classmethod
    def _generate_caption_content(
        cls,
        alignments: List[Supervision],
        output_format: str,
        include_speaker_in_text: bool = True,
    ) -> bytes:
        """
        Generate caption content in memory without temporary files.

        Args:
            alignments: List of supervision segments
            output_format: Output format (e.g., 'srt', 'vtt', 'ass')
            include_speaker_in_text: Whether to include speaker in text

        Returns:
            Caption content as bytes
        """
        output_format = output_format.lower()

        # Handle text format
        if output_format == "txt":
            lines = []
            for sup in alignments:
                word_items = cls._parse_alignment_from_supervision(sup)
                if word_items:
                    for item in word_items:
                        lines.append(f"[{item.start:.2f}-{item.end:.2f}] {item.symbol}")
                else:
                    if include_speaker_in_text and sup.speaker is not None:
                        if not sup.has_custom("original_speaker") or sup.custom["original_speaker"]:
                            text = f"[{sup.speaker}]: {sup.text}"
                        else:
                            text = sup.text
                    else:
                        text = sup.text
                    lines.append(f"[{sup.start:.2f}-{sup.end:.2f}] {text}")
            return "\n".join(lines).encode("utf-8")

        # Handle JSON format
        elif output_format == "json":
            json_data = [sup.to_dict() for sup in alignments]
            return json.dumps(json_data, ensure_ascii=False, indent=4).encode("utf-8")

        # Handle TextGrid format (requires file writing via tgt library)
        elif output_format == "textgrid":
            # TextGrid requires file I/O due to tgt library implementation
            with tempfile.NamedTemporaryFile(suffix=".textgrid", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                cls._write_textgrid(alignments, tmp_path, include_speaker_in_text)
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                tmp_path.unlink(missing_ok=True)

        # Handle TSV format
        elif output_format == "tsv":
            lines = []
            if include_speaker_in_text:
                lines.append("speaker\tstart\tend\ttext")
                for sup in alignments:
                    speaker = sup.speaker if cls._should_include_speaker(sup, True) else ""
                    start_ms = round(1000 * sup.start)
                    end_ms = round(1000 * sup.end)
                    text = sup.text.strip().replace("\t", " ")
                    lines.append(f"{speaker}\t{start_ms}\t{end_ms}\t{text}")
            else:
                lines.append("start\tend\ttext")
                for sup in alignments:
                    start_ms = round(1000 * sup.start)
                    end_ms = round(1000 * sup.end)
                    text = sup.text.strip().replace("\t", " ")
                    lines.append(f"{start_ms}\t{end_ms}\t{text}")
            return "\n".join(lines).encode("utf-8")

        # Handle CSV format
        elif output_format == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            if include_speaker_in_text:
                writer.writerow(["speaker", "start", "end", "text"])
                for sup in alignments:
                    speaker = sup.speaker if cls._should_include_speaker(sup, True) else ""
                    start_ms = round(1000 * sup.start)
                    end_ms = round(1000 * sup.end)
                    writer.writerow([speaker, start_ms, end_ms, sup.text.strip()])
            else:
                writer.writerow(["start", "end", "text"])
                for sup in alignments:
                    start_ms = round(1000 * sup.start)
                    end_ms = round(1000 * sup.end)
                    writer.writerow([start_ms, end_ms, sup.text.strip()])

            return output.getvalue().encode("utf-8")

        # Handle AUD format
        elif output_format == "aud":
            lines = []
            for sup in alignments:
                text = sup.text.strip().replace("\t", " ")
                if cls._should_include_speaker(sup, include_speaker_in_text):
                    text = f"[[{sup.speaker}]]{text}"
                lines.append(f"{sup.start}\t{sup.end}\t{text}")
            return "\n".join(lines).encode("utf-8")

        # Handle SBV format
        elif output_format == "sbv":
            lines = []
            for i, sup in enumerate(alignments):
                start_time = cls._format_sbv_timestamp(sup.start)
                end_time = cls._format_sbv_timestamp(sup.end)
                lines.append(f"{start_time},{end_time}")

                text = sup.text.strip()
                if cls._should_include_speaker(sup, include_speaker_in_text):
                    text = f"{sup.speaker}: {text}"
                lines.append(text)

                if i < len(alignments) - 1:
                    lines.append("")

            return "\n".join(lines).encode("utf-8")

        # Handle Avid DS format
        elif output_format == "avid_ds":
            from .formats.nle.avid import AvidDSConfig, AvidDSWriter

            config = AvidDSConfig(include_speaker=include_speaker_in_text)
            return AvidDSWriter.to_bytes(alignments, config)

        # Handle FCPXML format
        elif output_format == "fcpxml":
            from .formats.nle.fcpxml import FCPXMLConfig, FCPXMLWriter

            config = FCPXMLConfig()
            return FCPXMLWriter.to_bytes(alignments, config)

        # Handle Premiere Pro XML format
        elif output_format == "premiere_xml":
            from .formats.nle.premiere import PremiereXMLConfig, PremiereXMLWriter

            config = PremiereXMLConfig()
            return PremiereXMLWriter.to_bytes(alignments, config)

        # Handle Adobe Audition CSV format
        elif output_format == "audition_csv":
            from .formats.nle.audition import AuditionCSVConfig, AuditionCSVWriter

            config = AuditionCSVConfig(include_speaker_in_name=include_speaker_in_text)
            return AuditionCSVWriter.to_bytes(alignments, config)

        # Handle EdiMarker CSV format (Pro Tools)
        elif output_format == "edimarker_csv":
            from .formats.nle.audition import EdiMarkerConfig, EdiMarkerWriter

            config = EdiMarkerConfig(include_speaker=include_speaker_in_text)
            return EdiMarkerWriter.to_bytes(alignments, config)

        # Handle IMSC1 TTML format
        elif output_format == "imsc1":
            from .formats.ttml import TTMLConfig
            from .formats.ttml import TTMLFormat as TTMLWriter

            config = TTMLConfig(profile="imsc1")
            return TTMLWriter.to_bytes(alignments, config)

        # Handle EBU-TT-D TTML format
        elif output_format == "ebu_tt_d":
            from .formats.ttml import TTMLConfig
            from .formats.ttml import TTMLFormat as TTMLWriter

            config = TTMLConfig(profile="ebu-tt-d")
            return TTMLWriter.to_bytes(alignments, config)

        # Handle pysubs2 formats (srt, vtt, ass, ssa, ttml, json)
        else:
            import pysubs2

            subs = pysubs2.SSAFile()
            for sup in alignments:
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
                    text = sup.text
                    if cls._should_include_speaker(sup, include_speaker_in_text):
                        text = f"{sup.speaker} {sup.text}"
                    subs.append(
                        pysubs2.SSAEvent(
                            start=int(sup.start * 1000),
                            end=int(sup.end * 1000),
                            text=text or "",
                            name=sup.speaker or "",
                        )
                    )

            # Generate string content and convert to bytes
            content_str = subs.to_string(format_=output_format)
            return content_str.encode("utf-8")

    @staticmethod
    def _parse_alignment_from_supervision(supervision: Any) -> Optional[List[AlignmentItem]]:
        """
        Extract word-level alignment items from Supervision object.

        Args:
            supervision: Supervision object with potential alignment data

        Returns:
            List of AlignmentItem objects, or None if no alignment data present
        """
        alignment = getattr(supervision, "alignment", None)
        return alignment.get("word") if alignment else None

    @staticmethod
    def _get_file_extension(path: Pathlike) -> str:
        """Get lowercase file extension without dot."""
        return Path(path).suffix.lstrip(".").lower()

    @staticmethod
    def _should_include_speaker(supervision: Supervision, include_speaker_in_text: bool) -> bool:
        """Check if speaker should be included in output text."""
        if not include_speaker_in_text or not supervision.speaker:
            return False
        return not supervision.has_custom("original_speaker") or supervision.custom["original_speaker"]

    @classmethod
    def _write_caption(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
        output_format: Optional[str] = None,
    ) -> Pathlike:
        """
        Write caption to file in various formats.

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output file
            include_speaker_in_text: Whether to include speaker in text
            output_format: Explicit output format (overrides file extension detection)

        Returns:
            Path to written file
        """
        # Use explicit format if provided, otherwise detect from extension
        ext = output_format.lower() if output_format else cls._get_file_extension(output_path)

        if ext == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for sup in alignments:
                    word_items = cls._parse_alignment_from_supervision(sup)
                    if word_items:
                        for item in word_items:
                            f.write(f"[{item.start:.2f}-{item.end:.2f}] {item.symbol}\n")
                    else:
                        text = sup.text
                        if cls._should_include_speaker(sup, include_speaker_in_text):
                            text = f"[{sup.speaker}]: {sup.text}"
                        f.write(f"[{sup.start:.2f}-{sup.end:.2f}] {text}\n")

        elif ext == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json_data = [sup.to_dict() for sup in alignments]
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        elif ext == "textgrid":
            cls._write_textgrid(alignments, output_path, include_speaker_in_text)
        elif ext == "tsv":
            cls._write_tsv(alignments, output_path, include_speaker_in_text)
        elif ext == "csv":
            cls._write_csv(alignments, output_path, include_speaker_in_text)
        elif ext == "aud":
            cls._write_aud(alignments, output_path, include_speaker_in_text)
        elif ext == "sbv":
            cls._write_sbv(alignments, output_path, include_speaker_in_text)
        # Handle professional NLE formats
        elif ext == "avid_ds" or str(output_path).endswith("_avid.txt"):
            from .formats.nle.avid import AvidDSConfig, AvidDSWriter

            config = AvidDSConfig(include_speaker=include_speaker_in_text)
            AvidDSWriter.write(alignments, output_path, config)
        elif ext == "fcpxml" or ext == "fcpxmld":
            from .formats.nle.fcpxml import FCPXMLConfig, FCPXMLWriter

            config = FCPXMLConfig()
            FCPXMLWriter.write(alignments, output_path, config)
        elif ext == "premiere_xml" or (ext == "xml" and "premiere" in str(output_path).lower()):
            from .formats.nle.premiere import PremiereXMLConfig, PremiereXMLWriter

            config = PremiereXMLConfig()
            PremiereXMLWriter.write(alignments, output_path, config)
        elif ext == "audition_csv" or (ext == "csv" and "audition" in str(output_path).lower()):
            from .formats.nle.audition import AuditionCSVConfig, AuditionCSVWriter

            config = AuditionCSVConfig(include_speaker_in_name=include_speaker_in_text)
            AuditionCSVWriter.write(alignments, output_path, config)
        elif ext == "edimarker_csv" or (ext == "csv" and "edimarker" in str(output_path).lower()):
            from .formats.nle.audition import EdiMarkerConfig, EdiMarkerWriter

            config = EdiMarkerConfig(include_speaker=include_speaker_in_text)
            EdiMarkerWriter.write(alignments, output_path, config)
        elif ext == "imsc1" or (ext == "ttml" and "imsc" in str(output_path).lower()):
            from .formats.ttml import TTMLConfig
            from .formats.ttml import TTMLFormat as TTMLWriter

            config = TTMLConfig(profile="imsc1")
            TTMLWriter.write(alignments, output_path, config)
        elif ext == "ebu_tt_d" or (ext == "ttml" and "ebu" in str(output_path).lower()):
            from .formats.ttml import TTMLConfig
            from .formats.ttml import TTMLFormat as TTMLWriter

            config = TTMLConfig(profile="ebu-tt-d")
            TTMLWriter.write(alignments, output_path, config)
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
                    text = sup.text
                    if cls._should_include_speaker(sup, include_speaker_in_text):
                        text = f"{sup.speaker} {sup.text}"
                    subs.append(
                        pysubs2.SSAEvent(
                            start=int(sup.start * 1000),
                            end=int(sup.end * 1000),
                            text=text or "",
                            name=sup.speaker or "",
                        )
                    )

            # MicroDVD format requires framerate to be specified
            output_ext = cls._get_file_extension(output_path)
            if output_ext == "sub":
                # Default to 25 fps for MicroDVD format if not specified
                subs.save(output_path, fps=25.0)
            else:
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
    def _parse_youtube_vtt_with_word_timestamps(
        cls, content: str, normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        """
        Parse YouTube VTT format with word-level timestamps.

        YouTube auto-generated captions use this format:
        Word1<00:00:10.559><c> Word2</c><00:00:11.120><c> Word3</c>...

        Args:
            content: VTT file content
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects with word-level alignments
        """
        from lhotse.supervision import AlignmentItem

        supervisions = []

        # Pattern to match timestamp lines: 00:00:14.280 --> 00:00:17.269 align:start position:0%
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")

        # Pattern to match word-level timestamps: <00:00:10.559><c> word</c>
        word_timestamp_pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}[.,]\d{3})><c>\s*([^<]+)</c>")

        # Pattern to match the first word (before first timestamp)
        first_word_pattern = re.compile(r"^([^<\n]+?)<(\d{2}:\d{2}:\d{2}[.,]\d{3})>")

        def parse_timestamp(ts: str) -> float:
            """Convert timestamp string to seconds."""
            ts = ts.replace(",", ".")
            parts = ts.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp line
            ts_match = timestamp_pattern.search(line)
            if ts_match:
                cue_start = parse_timestamp(ts_match.group(1))
                cue_end = parse_timestamp(ts_match.group(2))

                # Read the next non-empty lines for cue content
                cue_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not timestamp_pattern.search(lines[i]):
                    cue_lines.append(lines[i])
                    i += 1

                # Process cue content
                for cue_line in cue_lines:
                    cue_line = cue_line.strip()
                    if not cue_line:
                        continue

                    # Check if this line has word-level timestamps
                    word_matches = word_timestamp_pattern.findall(cue_line)
                    if word_matches:
                        # This line has word-level timing
                        word_alignments = []

                        # Get the first word (before the first timestamp)
                        first_match = first_word_pattern.match(cue_line)
                        if first_match:
                            first_word = first_match.group(1).strip()
                            first_word_next_ts = parse_timestamp(first_match.group(2))
                            if first_word:
                                # First word starts at cue_start
                                word_alignments.append(
                                    AlignmentItem(
                                        symbol=first_word,
                                        start=cue_start,
                                        duration=first_word_next_ts - cue_start,
                                    )
                                )

                        # Process remaining words with timestamps
                        for idx, (ts, word) in enumerate(word_matches):
                            word_start = parse_timestamp(ts)
                            word = word.strip()
                            if not word:
                                continue

                            # Calculate duration based on next word's timestamp or cue end
                            if idx + 1 < len(word_matches):
                                next_ts = parse_timestamp(word_matches[idx + 1][0])
                                duration = next_ts - word_start
                            else:
                                duration = cue_end - word_start

                            word_alignments.append(
                                AlignmentItem(
                                    symbol=word,
                                    start=word_start,
                                    duration=max(0.01, duration),  # Ensure positive duration
                                )
                            )

                        if word_alignments:
                            # Create supervision with word-level alignment
                            full_text = " ".join(item.symbol for item in word_alignments)
                            if normalize_text:
                                full_text = normalize_text_fn(full_text)

                            sup_start = word_alignments[0].start
                            sup_end = word_alignments[-1].start + word_alignments[-1].duration

                            supervisions.append(
                                Supervision(
                                    text=full_text,
                                    start=sup_start,
                                    duration=sup_end - sup_start,
                                    alignment={"word": word_alignments},
                                )
                            )
                    else:
                        # Plain text line without word-level timing - skip duplicate lines
                        # (YouTube VTT often repeats the previous line without timestamps)
                        pass

                continue
            i += 1

        # Merge consecutive supervisions to form complete utterances
        if supervisions:
            supervisions = cls._merge_youtube_vtt_supervisions(supervisions)

        return supervisions

    @classmethod
    def _merge_youtube_vtt_supervisions(cls, supervisions: List[Supervision]) -> List[Supervision]:
        """
        Merge consecutive YouTube VTT supervisions into complete utterances.

        YouTube VTT splits utterances across multiple cues. This method merges
        cues that are close together in time.

        Args:
            supervisions: List of supervisions to merge

        Returns:
            List of merged supervisions
        """
        if not supervisions:
            return supervisions

        merged = []
        current = supervisions[0]

        for next_sup in supervisions[1:]:
            # Check if next supervision is close enough to merge (within 0.5 seconds)
            gap = next_sup.start - (current.start + current.duration)

            if gap < 0.5 and current.alignment and next_sup.alignment:
                # Merge alignments
                current_words = current.alignment.get("word", [])
                next_words = next_sup.alignment.get("word", [])
                merged_words = list(current_words) + list(next_words)

                # Create merged supervision
                merged_text = current.text + " " + next_sup.text
                merged_end = next_sup.start + next_sup.duration

                current = Supervision(
                    text=merged_text,
                    start=current.start,
                    duration=merged_end - current.start,
                    alignment={"word": merged_words},
                )
            else:
                merged.append(current)
                current = next_sup

        merged.append(current)
        return merged

    @classmethod
    def _is_youtube_vtt_with_word_timestamps(cls, content: str) -> bool:
        """
        Check if content is YouTube VTT format with word-level timestamps.

        Args:
            content: File content to check

        Returns:
            True if content contains YouTube-style word timestamps
        """
        # Look for pattern like <00:00:10.559><c> word</c>
        return bool(re.search(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}><c>", content))

    @classmethod
    def _parse_supervisions(
        cls, caption: Union[Pathlike, str], format: Optional[str], normalize_text: Optional[bool] = False
    ) -> List[Supervision]:
        """
        Parse supervisions from caption file or string content.

        Args:
            caption: Caption file path or string content
            format: Caption format
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        if format:
            format = format.lower()

        # Check if caption is string content (not a file path)
        is_string_content = isinstance(caption, str) and ("\n" in caption or len(caption) > 500)

        # Check for YouTube VTT with word-level timestamps first
        if is_string_content:
            content = caption
            if cls._is_youtube_vtt_with_word_timestamps(content):
                return cls._parse_youtube_vtt_with_word_timestamps(content, normalize_text)
        else:
            caption_path = Path(str(caption))
            if caption_path.exists():
                with open(caption_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if cls._is_youtube_vtt_with_word_timestamps(content):
                    return cls._parse_youtube_vtt_with_word_timestamps(content, normalize_text)

        # Match Gemini format: explicit format, or files ending with Gemini.md/Gemini3.md,
        # or files containing "gemini" in the name with .md extension
        if not is_string_content:
            caption_str = str(caption).lower()
            is_gemini_format = (
                format == "gemini"
                or str(caption).endswith("Gemini.md")
                or str(caption).endswith("Gemini3.md")
                or ("gemini" in caption_str and caption_str.endswith(".md"))
            )
        else:
            is_gemini_format = format == "gemini"

        if is_gemini_format:
            from .formats.gemini import GeminiReader

            supervisions = GeminiReader.extract_for_alignment(caption)
        elif format == "textgrid" or (not is_string_content and cls._get_file_extension(caption) == "textgrid"):
            # Internal usage
            from tgt import read_textgrid

            tgt = read_textgrid(caption)
            supervisions = [
                Supervision(
                    text=interval.text,
                    start=interval.start_time,
                    duration=interval.end_time - interval.start_time,
                    speaker=tier.name,
                )
                for tier in tgt.tiers
                for interval in tier.intervals
            ]
            supervisions = sorted(supervisions, key=lambda x: x.start)
        elif format == "tsv" or (not is_string_content and cls._get_file_extension(caption) == "tsv"):
            supervisions = cls._parse_tsv(caption, normalize_text)
        elif format == "csv" or (not is_string_content and cls._get_file_extension(caption) == "csv"):
            supervisions = cls._parse_csv(caption, normalize_text)
        elif format == "aud" or (not is_string_content and cls._get_file_extension(caption) == "aud"):
            supervisions = cls._parse_aud(caption, normalize_text)
        elif format == "sbv" or (not is_string_content and cls._get_file_extension(caption) == "sbv"):
            supervisions = cls._parse_sbv(caption, normalize_text)
        elif format == "txt" or (
            format == "auto" and not is_string_content and cls._get_file_extension(caption) == "txt"
        ):
            if is_string_content or not Path(str(caption)).exists():  # str content
                lines = [line.strip() for line in str(caption).split("\n")]
                if normalize_text:
                    lines = [normalize_text_fn(line) for line in lines]
            else:  # file
                path_str = str(caption)
                with open(path_str, encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
                    if normalize_text:
                        lines = [normalize_text_fn(line) for line in lines]
            supervisions = []
            for line in lines:
                if line:
                    # First try to parse timestamp format: [start-end] text
                    start, end, remaining_text = parse_timestamp_text(line)
                    if start is not None and end is not None:
                        # Has timestamp, now check for speaker in the remaining text
                        speaker, text = parse_speaker_text(remaining_text)
                        supervisions.append(
                            Supervision(
                                text=text,
                                start=start,
                                duration=end - start,
                                speaker=speaker,
                            )
                        )
                    else:
                        # No timestamp, just parse speaker and text
                        speaker, text = parse_speaker_text(line)
                        supervisions.append(Supervision(text=text, speaker=speaker))
        else:
            try:
                supervisions = cls._parse_caption(caption, format=format, normalize_text=normalize_text)
            except Exception as e:
                print(f"Failed to parse caption with Format: {format}, Exception: {e}, trying 'gemini' parser.")
                from .formats.gemini import GeminiReader

                supervisions = GeminiReader.extract_for_alignment(caption)

        return supervisions

    @classmethod
    def _parse_tsv(cls, caption: Pathlike, normalize_text: Optional[bool] = False) -> List[Supervision]:
        """
        Parse TSV (Tab-Separated Values) format caption file.

        Format specifications:
        - With speaker: speaker\tstart\tend\ttext
        - Without speaker: start\tend\ttext
        - Times are in milliseconds

        Args:
            caption: Caption file path
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        caption_path = Path(str(caption))
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption}")

        supervisions = []

        with open(caption_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if first line is a header
        first_line = lines[0].strip().lower()
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker_column = "speaker" in first_line

        start_idx = 1 if has_header else 0

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                if has_speaker_column and len(parts) >= 4:
                    # Format: speaker\tstart\tend\ttext
                    speaker = parts[0].strip() if parts[0].strip() else None
                    start = float(parts[1]) / 1000.0  # Convert milliseconds to seconds
                    end = float(parts[2]) / 1000.0
                    text = "\t".join(parts[3:]).strip()
                else:
                    # Format: start\tend\ttext
                    start = float(parts[0]) / 1000.0  # Convert milliseconds to seconds
                    end = float(parts[1]) / 1000.0
                    text = "\t".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                duration = end - start
                if duration < 0:
                    continue

                supervisions.append(
                    Supervision(
                        text=text,
                        start=start,
                        duration=duration,
                        speaker=speaker,
                    )
                )
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return supervisions

    @classmethod
    def _parse_csv(cls, caption: Pathlike, normalize_text: Optional[bool] = False) -> List[Supervision]:
        """
        Parse CSV (Comma-Separated Values) format caption file.

        Format specifications:
        - With speaker: speaker,start,end,text
        - Without speaker: start,end,text
        - Times are in milliseconds

        Args:
            caption: Caption file path
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        import csv

        caption_path = Path(str(caption))
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption}")

        supervisions = []

        with open(caption_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            lines = list(reader)

        if not lines:
            return supervisions

        # Check if first line is a header
        first_line = [col.strip().lower() for col in lines[0]]
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker_column = "speaker" in first_line

        start_idx = 1 if has_header else 0

        for parts in lines[start_idx:]:
            if len(parts) < 3:
                continue

            try:
                if has_speaker_column and len(parts) >= 4:
                    # Format: speaker,start,end,text
                    speaker = parts[0].strip() if parts[0].strip() else None
                    start = float(parts[1]) / 1000.0  # Convert milliseconds to seconds
                    end = float(parts[2]) / 1000.0
                    text = ",".join(parts[3:]).strip()
                else:
                    # Format: start,end,text
                    start = float(parts[0]) / 1000.0  # Convert milliseconds to seconds
                    end = float(parts[1]) / 1000.0
                    text = ",".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                duration = end - start
                if duration < 0:
                    continue

                supervisions.append(
                    Supervision(
                        text=text,
                        start=start,
                        duration=duration,
                        speaker=speaker,
                    )
                )
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return supervisions

    @classmethod
    def _parse_aud(cls, caption: Pathlike, normalize_text: Optional[bool] = False) -> List[Supervision]:
        """
        Parse AUD (Audacity Labels) format caption file.

        Format: start\tend\t[[speaker]]text
        - Times are in seconds (float)
        - Speaker is optional and enclosed in [[brackets]]

        Args:
            caption: Caption file path
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        caption_path = Path(str(caption))
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption}")

        supervisions = []

        with open(caption_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                # AUD format: start\tend\ttext (speaker in [[brackets]])
                start = float(parts[0])
                end = float(parts[1])
                text = "\t".join(parts[2:]).strip()

                # Extract speaker from [[speaker]] prefix
                speaker = None
                speaker_match = re.match(r"^\[\[([^\]]+)\]\]\s*(.*)$", text)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)

                if normalize_text:
                    text = normalize_text_fn(text)

                duration = end - start
                if duration < 0:
                    continue

                supervisions.append(
                    Supervision(
                        text=text,
                        start=start,
                        duration=duration,
                        speaker=speaker,
                    )
                )
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return supervisions

    @classmethod
    def _parse_sbv(cls, caption: Pathlike, normalize_text: Optional[bool] = False) -> List[Supervision]:
        """
        Parse SubViewer (SBV) format caption file.

        Format:
        0:00:00.000,0:00:02.000
        Text line 1

        0:00:02.000,0:00:04.000
        Text line 2

        Args:
            caption: Caption file path
            normalize_text: Whether to normalize text

        Returns:
            List of Supervision objects
        """
        caption_path = Path(str(caption))
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption}")

        supervisions = []

        with open(caption_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines to separate entries
        entries = content.strip().split("\n\n")

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 2:
                continue

            # First line: timestamp (H:MM:SS.mmm,H:MM:SS.mmm)
            timestamp_line = lines[0].strip()
            # Remaining lines: text
            text_lines = lines[1:]

            try:
                # Parse timestamp: 0:00:00.000,0:00:02.000
                if "," not in timestamp_line:
                    continue

                start_str, end_str = timestamp_line.split(",", 1)

                # Parse start time
                start_parts = start_str.strip().split(":")
                if len(start_parts) == 3:
                    h, m, s = start_parts
                    s_parts = s.split(".")
                    start = int(h) * 3600 + int(m) * 60 + int(s_parts[0])
                    if len(s_parts) > 1:
                        start += int(s_parts[1]) / 1000.0
                else:
                    continue

                # Parse end time
                end_parts = end_str.strip().split(":")
                if len(end_parts) == 3:
                    h, m, s = end_parts
                    s_parts = s.split(".")
                    end = int(h) * 3600 + int(m) * 60 + int(s_parts[0])
                    if len(s_parts) > 1:
                        end += int(s_parts[1]) / 1000.0
                else:
                    continue

                # Parse text and speaker
                text = " ".join(text_lines).strip()
                speaker, text = parse_speaker_text(text)

                if normalize_text:
                    text = normalize_text_fn(text)

                duration = end - start
                if duration < 0:
                    continue

                supervisions.append(
                    Supervision(
                        text=text,
                        start=start,
                        duration=duration,
                        speaker=speaker,
                    )
                )
            except (ValueError, IndexError):
                # Skip malformed entries
                continue

        return supervisions

    @classmethod
    def _write_textgrid(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> None:
        """
        Write caption to TextGrid format.

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output TextGrid file
            include_speaker_in_text: Whether to include speaker in text
        """
        from tgt import Interval, IntervalTier, TextGrid, write_to_file

        tg = TextGrid()
        supervisions, words, scores = [], [], {"utterances": [], "words": []}
        for supervision in sorted(alignments, key=lambda x: x.start):
            text = supervision.text
            if cls._should_include_speaker(supervision, include_speaker_in_text):
                text = f"{supervision.speaker} {supervision.text}"
            supervisions.append(Interval(supervision.start, supervision.end, text or ""))
            # Extract word-level alignment using helper function
            word_items = cls._parse_alignment_from_supervision(supervision)
            if word_items:
                for item in word_items:
                    words.append(Interval(item.start, item.end, item.symbol))
                    if item.score is not None:
                        scores["words"].append(Interval(item.start, item.end, f"{item.score:.2f}"))
            if supervision.has_custom("score"):
                scores["utterances"].append(Interval(supervision.start, supervision.end, f"{supervision.score:.2f}"))

        tg.add_tier(IntervalTier(name="utterances", objects=supervisions))
        if words:
            tg.add_tier(IntervalTier(name="words", objects=words))

        if scores["utterances"]:
            tg.add_tier(IntervalTier(name="utterance_scores", objects=scores["utterances"]))
        if scores["words"]:
            tg.add_tier(IntervalTier(name="word_scores", objects=scores["words"]))

        write_to_file(tg, output_path, format="long")

    @classmethod
    def _write_tsv(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> None:
        """
        Write caption to TSV format.

        Format: speaker\tstart\tend\ttext (with speaker)
        or: start\tend\ttext (without speaker)

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output TSV file
            include_speaker_in_text: Whether to include speaker column
        """
        with open(output_path, "w", encoding="utf-8") as file:
            # Write header
            if include_speaker_in_text:
                file.write("speaker\tstart\tend\ttext\n")
                for supervision in alignments:
                    speaker = supervision.speaker if cls._should_include_speaker(supervision, True) else ""
                    start_ms = round(1000 * supervision.start)
                    end_ms = round(1000 * supervision.end)
                    text = supervision.text.strip().replace("\t", " ")
                    file.write(f"{speaker}\t{start_ms}\t{end_ms}\t{text}\n")
            else:
                file.write("start\tend\ttext\n")
                for supervision in alignments:
                    start_ms = round(1000 * supervision.start)
                    end_ms = round(1000 * supervision.end)
                    text = supervision.text.strip().replace("\t", " ")
                    file.write(f"{start_ms}\t{end_ms}\t{text}\n")

    @classmethod
    def _write_csv(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> None:
        """
        Write caption to CSV format.

        Format: speaker,start,end,text (with speaker)
        or: start,end,text (without speaker)

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output CSV file
            include_speaker_in_text: Whether to include speaker column
        """
        import csv

        with open(output_path, "w", encoding="utf-8", newline="") as file:
            if include_speaker_in_text:
                writer = csv.writer(file)
                writer.writerow(["speaker", "start", "end", "text"])
                for supervision in alignments:
                    speaker = supervision.speaker if cls._should_include_speaker(supervision, True) else ""
                    start_ms = round(1000 * supervision.start)
                    end_ms = round(1000 * supervision.end)
                    text = supervision.text.strip()
                    writer.writerow([speaker, start_ms, end_ms, text])
            else:
                writer = csv.writer(file)
                writer.writerow(["start", "end", "text"])
                for supervision in alignments:
                    start_ms = round(1000 * supervision.start)
                    end_ms = round(1000 * supervision.end)
                    text = supervision.text.strip()
                    writer.writerow([start_ms, end_ms, text])

    @classmethod
    def _write_aud(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> None:
        """
        Write caption to AUD format.

        Format: start\tend\t[[speaker]]text
        or: start\tend\ttext (without speaker)

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output AUD file
            include_speaker_in_text: Whether to include speaker in [[brackets]]
        """
        with open(output_path, "w", encoding="utf-8") as file:
            for supervision in alignments:
                start = supervision.start
                end = supervision.end
                text = supervision.text.strip().replace("\t", " ")

                if cls._should_include_speaker(supervision, include_speaker_in_text):
                    text = f"[[{supervision.speaker}]]{text}"

                file.write(f"{start}\t{end}\t{text}\n")

    @classmethod
    def _write_sbv(
        cls,
        alignments: List[Supervision],
        output_path: Pathlike,
        include_speaker_in_text: bool = True,
    ) -> None:
        """
        Write caption to SubViewer (SBV) format.

        Format:
        0:00:00.000,0:00:02.000
        Text line 1

        0:00:02.000,0:00:04.000
        Text line 2

        Args:
            alignments: List of supervision segments to write
            output_path: Path to output SBV file
            include_speaker_in_text: Whether to include speaker in text
        """
        with open(output_path, "w", encoding="utf-8") as file:
            for i, supervision in enumerate(alignments):
                # Format timestamps as H:MM:SS.mmm
                start_time = cls._format_sbv_timestamp(supervision.start)
                end_time = cls._format_sbv_timestamp(supervision.end)

                # Write timestamp line
                file.write(f"{start_time},{end_time}\n")

                # Write text (with optional speaker)
                text = supervision.text.strip()
                if cls._should_include_speaker(supervision, include_speaker_in_text):
                    text = f"{supervision.speaker}: {text}"

                file.write(f"{text}\n")

                # Add blank line between entries (except after last one)
                if i < len(alignments) - 1:
                    file.write("\n")

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
        parts = [f"Caption({len(self.supervisions or self.transcription)} segments", lang]
        if kind_str:
            parts.append(kind_str)
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        return ", ".join(parts) + ")"
