"""Caption data structure for storing subtitle information with metadata."""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike
from tgt import TextGrid

from ..config.caption import InputCaptionFormat, OutputCaptionFormat  # noqa: F401
from .formats import detect_format, get_reader, get_writer
from .parsers.text_parser import normalize_text as normalize_text_fn
from .parsers.text_parser import parse_speaker_text
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
        if not format:
            format = detect_format(path)
        else:
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
        writer_cls = get_writer(output_format)

        if writer_cls:
            return writer_cls.to_bytes(alignments, include_speaker=include_speaker_in_text)

        # Fallback to pysubs2 for any other formats it might support
        import pysubs2

        subs = pysubs2.SSAFile()
        for sup in alignments:
            # Extract word-level alignment if present
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

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
        output_path = Path(str(output_path))
        # Use explicit format if provided, otherwise detect from extension
        ext = output_format.lower() if output_format else cls._get_file_extension(output_path)

        # Handle special professional NLE naming conventions
        if not output_format:
            if str(output_path).endswith("_avid.txt"):
                ext = "avid_ds"
            elif "audition" in str(output_path).lower() and ext == "csv":
                ext = "audition_csv"
            elif "edimarker" in str(output_path).lower() and ext == "csv":
                ext = "edimarker_csv"
            elif "imsc" in str(output_path).lower() and ext == "ttml":
                ext = "imsc1"
            elif "ebu" in str(output_path).lower() and ext == "ttml":
                ext = "ebu_tt_d"

        writer_cls = get_writer(ext)
        if writer_cls:
            writer_cls.write(alignments, output_path, include_speaker=include_speaker_in_text)
            return output_path

        # Fallback to manual writing via _generate_caption_content
        content = cls._generate_caption_content(alignments, ext, include_speaker_in_text)
        output_path.write_bytes(content)

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

        # Detect format if not provided
        fmt = format
        if not fmt and not is_string_content:
            fmt = detect_format(caption)

        if fmt:
            reader_cls = get_reader(fmt)
            if reader_cls:
                try:
                    return reader_cls.read(caption, normalize_text=normalize_text)
                except Exception as e:
                    print(f"Failed to parse caption with Reader: {fmt}, Exception: {e}, falling back.")

        # Fallback to generic pysubs2 parser or specialized Gemini parser if it failed
        try:
            return cls._parse_caption(caption, format=format, normalize_text=normalize_text)
        except Exception as e:
            # Final fallback to GeminiReader if it's potentially markdown
            if not is_string_content and str(caption).lower().endswith(".md"):
                from .formats.gemini import GeminiReader

                return GeminiReader.extract_for_alignment(caption)
            raise e

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
