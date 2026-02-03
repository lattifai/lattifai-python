"""Extended Caption class with transcription, alignment, and diarization support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar

from lattifai.caption import Caption as BaseCaption
from lattifai.caption import Pathlike, Supervision

if TYPE_CHECKING:
    from lattifai_core.event import LEDOutput

DiarizationOutput = TypeVar("DiarizationOutput")


@dataclass
class Caption(BaseCaption):
    """
    Extended Caption with transcription, alignment, and diarization support.

    Inherits from BaseCaption and adds fields for:
    - alignments: Post-alignment results
    - transcription: ASR results
    - event: LattifAI Event Detection results (LEDOutput)
    - diarization: Speaker diarization results


    These fields are used in the LattifAI pipeline for:
    - Forced alignment results
    - Storing intermediate transcription results
    - LattifAI Event Detection (music, applause, speech, etc.)
    - Speaker identification and separation

    """

    # Alignment results
    alignments: List[Supervision] = field(default_factory=list)

    # Transcription results
    transcription: List[Supervision] = field(default_factory=list)

    # LattifAI Event Detection results
    event: Optional["LEDOutput"] = None

    # Speaker Diarization results
    diarization: Optional[DiarizationOutput] = None

    def __len__(self) -> int:
        """Return the number of supervision segments."""
        return len(self.supervisions or self.transcription)

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

    def with_margins(
        self,
        start_margin: float = 0.08,
        end_margin: float = 0.20,
        min_gap: float = 0.08,
        collision_mode: str = "trim",
    ) -> "Caption":
        """
        Create a new Caption with segment boundaries adjusted based on word-level alignment.

        Uses supervision.alignment['word'] to recalculate segment start/end times
        with the specified margins applied around the actual speech boundaries.

        Prefers alignments > supervisions > transcription as source.

        Args:
            start_margin: Seconds to extend before the first word (default: 0.08)
            end_margin: Seconds to extend after the last word (default: 0.20)
            min_gap: Minimum gap between segments for collision handling (default: 0.08)
            collision_mode: How to handle segment overlap - 'trim' or 'gap' (default: 'trim')

        Returns:
            New Caption instance with adjusted timestamps

        Note:
            Segments without alignment data will keep their original timestamps.
        """
        from lattifai.caption.standardize import apply_margins_to_captions

        # Determine which supervisions to use (priority: alignments > supervisions > transcription)
        if self.alignments:
            source_sups = self.alignments
        elif self.supervisions:
            source_sups = self.supervisions
        else:
            source_sups = self.transcription

        adjusted_sups = apply_margins_to_captions(
            source_sups,
            start_margin=start_margin,
            end_margin=end_margin,
            min_gap=min_gap,
            collision_mode=collision_mode,
        )

        return Caption(
            supervisions=adjusted_sups,
            transcription=self.transcription,
            event=self.event,
            diarization=self.diarization,
            alignments=[],  # Clear alignments since we've applied them
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy() if self.metadata else {},
        )

    def write(
        self,
        path=None,
        output_format: Optional[str] = None,
        include_speaker_in_text: bool = True,
        word_level: bool = False,
        karaoke_config=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Write caption to file or return as bytes.

        Prefers alignments > supervisions > transcription as source.

        Args:
            path: Path to output caption file, BytesIO object, or None to return bytes
            output_format: Output format (e.g., 'srt', 'vtt', 'ass')
            include_speaker_in_text: Whether to include speaker labels in text
            word_level: Use word-level output format if supported
            karaoke_config: Karaoke configuration
            metadata: Optional metadata dict to pass to writer

        Returns:
            Path to the written file if path is a file path, or bytes if path is BytesIO/None
        """
        # Temporarily swap supervisions to use the priority order
        original_supervisions = self.supervisions

        if self.alignments:
            self.supervisions = self.alignments
        elif not self.supervisions and self.transcription:
            self.supervisions = self.transcription

        try:
            result = super().write(
                path=path,
                output_format=output_format,
                include_speaker_in_text=include_speaker_in_text,
                word_level=word_level,
                karaoke_config=karaoke_config,
                metadata=metadata,
            )
        finally:
            # Restore original supervisions
            self.supervisions = original_supervisions

        return result

    @classmethod
    def from_transcription_results(
        cls,
        transcription: List[Supervision],
        event: Optional["LEDOutput"] = None,
        diarization: Optional[DiarizationOutput] = None,
        language: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from transcription results including audio events and diarization.

        Args:
            transcription: List of transcription supervision segments
            event: Optional LEDOutput with event detection results
            diarization: Optional DiarizationOutput with speaker diarization results
            language: Language code
            source_path: Source file path
            metadata: Additional metadata

        Returns:
            New Caption instance with transcription data
        """
        return cls(
            transcription=transcription,
            event=event,
            diarization=diarization,
            language=language,
            kind="transcription",
            source_format="asr",
            source_path=source_path,
            metadata=metadata or {},
        )

    def read_diarization(
        self,
        path: Pathlike,
    ) -> "DiarizationOutput":
        """
        Read speaker diarization TextGrid from file.
        """
        from lattifai_core.diarization import DiarizationOutput

        self.diarization = DiarizationOutput.read(path)
        return self.diarization

    def write_diarization(
        self,
        path: Pathlike,
    ) -> Pathlike:
        """
        Write speaker diarization TextGrid to file.
        """
        if not self.diarization:
            raise ValueError("No speaker diarization data to write.")

        self.diarization.write(path)
        return path
