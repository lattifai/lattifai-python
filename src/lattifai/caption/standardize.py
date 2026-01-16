"""
Caption Standardization Module

Implements broadcast-grade caption standardization following Netflix/BBC guidelines:
- Timeline cleanup (min/max duration, gap checking)
- Smart text line breaking
- Quality validation

Reference Standards:
- Netflix Timed Text Style Guide
- BBC Subtitle Guidelines
- EBU-TT-D Standard
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

from lhotse.supervision import SupervisionSegment

from ..config.caption import StandardizationConfig
from .supervision import Supervision

__all__ = [
    "CaptionStandardizer",
    "CaptionValidator",
    "StandardizationConfig",
    "ValidationResult",
    "standardize_captions",
    "apply_margins_to_captions",
]


@dataclass
class ValidationResult:
    """Validation result."""

    valid: bool = True
    """Whether all validations passed"""

    warnings: List[str] = field(default_factory=list)
    """List of warning messages"""

    # Statistics
    avg_cps: float = 0.0
    """Average reading speed (chars/sec)"""

    max_cpl: int = 0
    """Maximum characters per line"""

    segments_too_short: int = 0
    """Number of segments too short"""

    segments_too_long: int = 0
    """Number of segments too long"""

    gaps_too_small: int = 0
    """Number of gaps too small"""


class CaptionStandardizer:
    """
    Caption standardization processor.

    Processing flow:
    1. Timeline cleanup - Adjust duration and gaps
    2. Text formatting - Smart line breaking
    3. Validation - Generate quality metrics

    Example:
        >>> standardizer = CaptionStandardizer(min_duration=0.8, max_chars_per_line=42)
        >>> processed = standardizer.process(supervisions)
    """

    # Chinese/Japanese punctuation (for line break priority)
    # Reference: alignment/punctuation.py
    CJK_PUNCTUATION = r"[，。、？！：；·…—～" "''（）【】〔〕〖〗《》〈〉「」『』〘〙〚〛]"

    # English/Western punctuation
    EN_PUNCTUATION = r"[,.!?;:\-–—«»‹›]"

    # All splittable punctuation (for line break search)
    ALL_PUNCTUATION = r"[，。、？！：；·…—～,.!?;:\-–—\s]"

    def __init__(
        self,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_lines: int = 2,
        max_chars_per_line: int = 42,
    ):
        """
        Initialize standardizer.

        Args:
            min_duration: Minimum duration (seconds)
            max_duration: Maximum duration (seconds)
            min_gap: Minimum gap (seconds)
            max_lines: Maximum number of lines
            max_chars_per_line: Maximum characters per line
        """
        self.config = StandardizationConfig(
            min_duration=min_duration,
            max_duration=max_duration,
            min_gap=min_gap,
            max_lines=max_lines,
            max_chars_per_line=max_chars_per_line,
        )

    def process(self, segments: List[Union[Supervision, SupervisionSegment]]) -> List[Supervision]:
        """
        Main processing entry point.

        Args:
            segments: List of original caption segments

        Returns:
            List of processed caption segments
        """
        if not segments:
            return []

        # 1. Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)

        # 2. Timeline cleanup
        processed = self._sanitize_timeline(sorted_segments)

        # 3. Text formatting
        processed = self._format_texts(processed)

        return processed

    def _sanitize_timeline(self, segments: List[Union[Supervision, SupervisionSegment]]) -> List[Supervision]:
        """
        Timeline cleanup.

        Processing logic:
        A. Gap check - Ensure sufficient gap between subtitles
        B. Min duration check - Extend too-short subtitles
        C. Max duration check - Truncate too-long subtitles

        Priority: Gap > Min duration (insufficient gap causes display issues)
        """
        result: List[Supervision] = []

        for i, seg in enumerate(segments):
            # Create new instance
            new_seg = self._copy_segment(seg)

            # A. Check gap with previous subtitle
            if result:
                prev_seg = result[-1]
                prev_end = prev_seg.start + prev_seg.duration
                gap = new_seg.start - prev_end

                if gap < self.config.min_gap:
                    # Gap too small or overlap
                    # Target: prev_end_new + min_gap = new_seg.start
                    # => prev_duration_new = new_seg.start - min_gap - prev_seg.start
                    target_prev_duration = new_seg.start - self.config.min_gap - prev_seg.start

                    if target_prev_duration >= self.config.min_duration:
                        # Safe to shorten previous subtitle (still meets min duration)
                        result[-1] = self._copy_segment(prev_seg, duration=target_prev_duration)
                    else:
                        # Shortening previous would go below min duration, delay current start
                        new_start = prev_end + self.config.min_gap
                        duration_diff = new_start - seg.start
                        new_duration = max(
                            0.1,  # Ensure at least some duration
                            new_seg.duration - duration_diff,
                        )
                        new_seg = self._copy_segment(new_seg, start=new_start, duration=new_duration)

            # B. Min duration check
            if new_seg.duration < self.config.min_duration:
                # Check if extending would overlap with next subtitle
                next_start = segments[i + 1].start if i + 1 < len(segments) else float("inf")
                max_extend = next_start - new_seg.start - self.config.min_gap
                new_duration = min(self.config.min_duration, max(max_extend, new_seg.duration))
                new_seg = self._copy_segment(new_seg, duration=new_duration)

            # C. Max duration check
            if new_seg.duration > self.config.max_duration:
                new_seg = self._copy_segment(new_seg, duration=self.config.max_duration)

            result.append(new_seg)

        return result

    def _format_texts(self, segments: List[Supervision]) -> List[Supervision]:
        """Apply text formatting to all subtitles."""
        return [self._copy_segment(seg, text=self._smart_split_text(seg.text or "")) for seg in segments]

    def _smart_split_text(self, text: str) -> str:
        """
        Smart text line breaking.

        Priority:
        1. CJK punctuation (，。！？ etc.)
        2. English punctuation (,.!? etc.)
        3. Whitespace
        4. Hard truncation

        Args:
            text: Original text

        Returns:
            Text with line breaks
        """
        # Clean text
        text = self._normalize_text(text)

        # Check if line break is needed
        if len(text) <= self.config.max_chars_per_line:
            return text

        lines: List[str] = []
        remaining = text

        for _ in range(self.config.max_lines):
            if len(remaining) <= self.config.max_chars_per_line:
                lines.append(remaining)
                remaining = ""
                break

            # Find best split point
            split_pos = self._find_split_point(remaining, self.config.max_chars_per_line)

            lines.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        # If remaining text exists and max lines reached, append to last line
        if remaining and lines:
            # Choose to append (may exceed char limit) rather than truncate
            lines[-1] = lines[-1] + " " + remaining if lines[-1] else remaining

        return "\n".join(lines)

    def _find_split_point(self, text: str, max_len: int) -> int:
        """
        Find best split point.

        Strategy: Find punctuation or whitespace near max_len
        Search range: 40% - 110% of max_len

        Args:
            text: Text to split
            max_len: Maximum length

        Returns:
            Split position index
        """
        search_start = int(max_len * 0.4)
        search_end = min(len(text), int(max_len * 1.1))

        best_pos = max_len
        best_priority = 999  # Lower is better

        # Search backwards, prefer split points closer to max_len
        for i in range(min(search_end, len(text)) - 1, search_start - 1, -1):
            char = text[i]
            priority = self._get_split_priority(char)

            if priority < best_priority:
                best_priority = priority
                best_pos = i + 1  # Split after punctuation/whitespace

                # Exit early if highest priority (CJK punctuation) found
                if priority == 1:
                    break

        return best_pos

    def _get_split_priority(self, char: str) -> int:
        """
        Get character split priority.

        Returns:
            1 = CJK punctuation (highest priority)
            2 = English punctuation
            3 = Whitespace
            999 = Other characters (not suitable for splitting)
        """
        if re.match(self.CJK_PUNCTUATION, char):
            return 1
        elif re.match(self.EN_PUNCTUATION, char):
            return 2
        elif char.isspace():
            return 3
        return 999

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text.

        - Remove excess whitespace
        - Remove existing newlines (will be reformatted)
        - Unify spaces
        """
        # Remove existing newlines
        text = text.replace("\n", " ")
        # Merge excess whitespace
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _copy_segment(
        self,
        seg: Union[Supervision, SupervisionSegment],
        **overrides,
    ) -> Supervision:
        """
        Create a copy of Supervision.

        Args:
            seg: Original segment
            **overrides: Fields to override

        Returns:
            New Supervision instance
        """
        return Supervision(
            id=overrides.get("id", seg.id),
            recording_id=overrides.get("recording_id", seg.recording_id),
            start=overrides.get("start", seg.start),
            duration=overrides.get("duration", seg.duration),
            channel=overrides.get("channel", getattr(seg, "channel", None)),
            text=overrides.get("text", seg.text),
            language=overrides.get("language", getattr(seg, "language", None)),
            speaker=overrides.get("speaker", getattr(seg, "speaker", None)),
            gender=overrides.get("gender", getattr(seg, "gender", None)),
            custom=overrides.get("custom", getattr(seg, "custom", None)),
            alignment=overrides.get("alignment", getattr(seg, "alignment", None)),
        )

    def apply_margins(
        self,
        segments: List[Union[Supervision, SupervisionSegment]],
        start_margin: Optional[float] = None,
        end_margin: Optional[float] = None,
    ) -> List[Supervision]:
        """
        Recalculate segment boundaries based on word-level alignment.

        Uses precise word-level timestamps from supervision.alignment['word']
        to recalculate segment start/end times.

        Args:
            segments: List of subtitles with alignment data
            start_margin: Start margin (overrides config default)
            end_margin: End margin (overrides config default)

        Returns:
            List of subtitles with new margins applied

        Note:
            - Segments without alignment data keep original timestamps
            - Automatically handles boundary collisions

        Example:
            >>> standardizer = CaptionStandardizer()
            >>> adjusted = standardizer.apply_margins(
            ...     supervisions, start_margin=0.05, end_margin=0.15
            ... )
        """
        if not segments:
            return []

        # Resolve margins: parameter > config > 0.0 (no adjustment)
        sm = start_margin if start_margin is not None else (self.config.start_margin or 0.0)
        em = end_margin if end_margin is not None else (self.config.end_margin or 0.0)

        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s.start)
        result: List[Supervision] = []

        for seg in sorted_segs:
            # Get word alignment
            words = self._get_word_alignment(seg)

            if not words:
                # No alignment data, keep original
                result.append(self._copy_segment(seg))
                continue

            # Calculate precise boundaries
            first_word_start = words[0].start
            last_word_end = words[-1].start + words[-1].duration

            # Apply margin (0.0 means no adjustment, just use word boundaries)
            new_start = max(0, first_word_start - sm)
            new_end = last_word_end + em

            # Collision detection (with previous segment)
            if result:
                prev_end = result[-1].start + result[-1].duration
                if new_start < prev_end + self.config.min_gap:
                    new_start = self._resolve_collision(prev_end, new_start, first_word_start, sm)

            new_duration = new_end - new_start
            result.append(self._copy_segment(seg, start=new_start, duration=new_duration))

        return result

    def _get_word_alignment(self, seg: Union[Supervision, SupervisionSegment]) -> List:
        """
        Safely get word alignment data.

        Args:
            seg: Subtitle segment

        Returns:
            Word alignment list, or empty list if not present
        """
        alignment = getattr(seg, "alignment", None)
        if alignment and "word" in alignment:
            return alignment["word"]
        return []

    def _resolve_collision(
        self,
        prev_end: float,
        new_start: float,
        first_word_start: float,
        start_margin: float,
    ) -> float:
        """
        Resolve collision with previous segment.

        Args:
            prev_end: End time of previous segment
            new_start: Currently calculated start time
            first_word_start: Start time of first word in current segment
            start_margin: Requested start_margin

        Returns:
            Adjusted start time
        """
        if self.config.margin_collision_mode == "gap":
            # Force maintain min_gap
            return prev_end + self.config.min_gap
        else:
            # Trim mode: preserve margin as much as possible, but not beyond speech start
            available_margin = first_word_start - (prev_end + self.config.min_gap)
            actual_margin = max(0, min(start_margin, available_margin))
            return first_word_start - actual_margin


class CaptionValidator:
    """
    Caption quality validator.

    Validates subtitles against broadcast standards and generates quality metrics report.

    Example:
        >>> validator = CaptionValidator()
        >>> result = validator.validate(supervisions)
        >>> if not result.valid:
        ...     print(result.warnings)
    """

    def __init__(
        self,
        config: Optional[StandardizationConfig] = None,
        min_duration: float = 0.8,
        max_duration: float = 7.0,
        min_gap: float = 0.08,
        max_chars_per_line: int = 42,
    ):
        """
        Initialize validator.

        Args:
            config: Standardization config (if provided, ignores other params)
            min_duration: Minimum duration
            max_duration: Maximum duration
            min_gap: Minimum gap
            max_chars_per_line: Maximum characters per line
        """
        if config:
            self.config = config
        else:
            self.config = StandardizationConfig(
                min_duration=min_duration,
                max_duration=max_duration,
                min_gap=min_gap,
                max_chars_per_line=max_chars_per_line,
            )

    def validate(self, segments: List[Union[Supervision, SupervisionSegment]]) -> ValidationResult:
        """
        Validate subtitles and return quality metrics.

        Args:
            segments: List of subtitle segments

        Returns:
            ValidationResult containing validation results and metrics
        """
        result = ValidationResult()

        if not segments:
            return result

        total_cps = 0.0
        prev_end = 0.0

        for i, seg in enumerate(segments):
            text = seg.text or ""
            duration = seg.duration

            # CPS calculation (excluding newlines)
            text_length = len(text.replace("\n", ""))
            cps = text_length / duration if duration > 0 else 0
            total_cps += cps

            # CPL calculation
            lines = text.split("\n")
            max_line_len = max((len(line) for line in lines), default=0)
            result.max_cpl = max(result.max_cpl, max_line_len)

            # Duration check
            if duration < self.config.min_duration:
                result.segments_too_short += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s < min {self.config.min_duration}s"
                )

            if duration > self.config.max_duration:
                result.segments_too_long += 1
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): duration {duration:.2f}s > max {self.config.max_duration}s"
                )

            # Gap check
            if i > 0:
                gap = seg.start - prev_end
                if gap < self.config.min_gap and gap >= 0:
                    result.gaps_too_small += 1
                    result.warnings.append(f"Segment {i} (id={seg.id}): gap {gap:.3f}s < min {self.config.min_gap}s")

            # CPL check
            if max_line_len > self.config.max_chars_per_line:
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): line length {max_line_len} > max {self.config.max_chars_per_line}"
                )

            # CPS check (reading speed too fast)
            if cps > self.config.optimal_cps * 1.5:  # Exceeds optimal by 50%
                result.warnings.append(
                    f"Segment {i} (id={seg.id}): CPS {cps:.1f} exceeds recommended {self.config.optimal_cps}"
                )

            prev_end = seg.start + seg.duration

        # Calculate average CPS
        result.avg_cps = total_cps / len(segments)

        # Determine if validation passed
        result.valid = result.segments_too_short == 0 and result.segments_too_long == 0 and result.gaps_too_small == 0

        return result


def standardize_captions(
    segments: List[Union[Supervision, SupervisionSegment]],
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    min_gap: float = 0.08,
    max_lines: int = 2,
    max_chars_per_line: int = 42,
) -> List[Supervision]:
    """
    Convenience function: Standardize caption list.

    Args:
        segments: List of original caption segments
        min_duration: Minimum duration (seconds)
        max_duration: Maximum duration (seconds)
        min_gap: Minimum gap (seconds)
        max_lines: Maximum number of lines
        max_chars_per_line: Maximum characters per line

    Returns:
        List of processed caption segments

    Example:
        >>> from lattifai.caption import standardize_captions
        >>> processed = standardize_captions(supervisions, max_chars_per_line=22)
    """
    standardizer = CaptionStandardizer(
        min_duration=min_duration,
        max_duration=max_duration,
        min_gap=min_gap,
        max_lines=max_lines,
        max_chars_per_line=max_chars_per_line,
    )
    return standardizer.process(segments)


def apply_margins_to_captions(
    segments: List[Union[Supervision, SupervisionSegment]],
    start_margin: float = 0.08,
    end_margin: float = 0.20,
    min_gap: float = 0.08,
    collision_mode: str = "trim",
) -> List[Supervision]:
    """
    Convenience function: Recalculate caption boundaries based on word-level alignment.

    Uses precise word-level timestamps from supervision.alignment['word']
    to recalculate segment start/end times.

    Args:
        segments: List of caption segments with alignment data
        start_margin: Start margin (seconds) - extends before first word
        end_margin: End margin (seconds) - extends after last word
        min_gap: Minimum gap (seconds) - for collision handling
        collision_mode: Collision mode 'trim' or 'gap'

    Returns:
        List of caption segments with new margins applied

    Example:
        >>> from lattifai.caption import apply_margins_to_captions
        >>> adjusted = apply_margins_to_captions(
        ...     supervisions, start_margin=0.05, end_margin=0.15
        ... )
    """
    standardizer = CaptionStandardizer(min_gap=min_gap)
    standardizer.config.start_margin = start_margin
    standardizer.config.end_margin = end_margin
    standardizer.config.margin_collision_mode = collision_mode
    return standardizer.apply_margins(segments, start_margin=start_margin, end_margin=end_margin)
