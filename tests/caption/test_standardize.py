#!/usr/bin/env python3
"""
Test suite for Caption Standardization Module

Tests for CaptionStandardizer and CaptionValidator classes.
"""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption import (
    Caption,
    CaptionStandardizer,
    CaptionValidator,
    StandardizationConfig,
    Supervision,
    ValidationResult,
    apply_margins_to_captions,
    standardize_captions,
)


class TestStandardizationConfig:
    """Test StandardizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StandardizationConfig()

        assert config.min_duration == 0.8
        assert config.max_duration == 7.0
        assert config.min_gap == 0.08
        assert config.max_lines == 2
        assert config.max_chars_per_line == 42
        assert config.optimal_cps == 17.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StandardizationConfig(
            min_duration=0.5,
            max_duration=5.0,
            min_gap=0.1,
            max_lines=3,
            max_chars_per_line=22,
        )

        assert config.min_duration == 0.5
        assert config.max_duration == 5.0
        assert config.min_gap == 0.1
        assert config.max_lines == 3
        assert config.max_chars_per_line == 22

    def test_validation_min_duration(self):
        """Test min_duration validation."""
        with pytest.raises(ValueError, match="min_duration must be positive"):
            StandardizationConfig(min_duration=0)

        with pytest.raises(ValueError, match="min_duration must be positive"):
            StandardizationConfig(min_duration=-1)

    def test_validation_max_duration(self):
        """Test max_duration validation."""
        with pytest.raises(ValueError, match="max_duration must be greater than min_duration"):
            StandardizationConfig(min_duration=5.0, max_duration=3.0)

    def test_validation_min_gap(self):
        """Test min_gap validation."""
        with pytest.raises(ValueError, match="min_gap cannot be negative"):
            StandardizationConfig(min_gap=-0.1)

    def test_validation_max_lines(self):
        """Test max_lines validation."""
        with pytest.raises(ValueError, match="max_lines must be at least 1"):
            StandardizationConfig(max_lines=0)

    def test_validation_max_chars_per_line(self):
        """Test max_chars_per_line validation."""
        with pytest.raises(ValueError, match="max_chars_per_line must be at least 10"):
            StandardizationConfig(max_chars_per_line=5)


class TestCaptionStandardizer:
    """Test CaptionStandardizer class."""

    def test_empty_input(self):
        """Test with empty input list."""
        standardizer = CaptionStandardizer()
        result = standardizer.process([])

        assert result == []

    def test_single_segment_no_change(self):
        """Test single segment that doesn't need changes."""
        standardizer = CaptionStandardizer()
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Hello world"),
        ]

        result = standardizer.process(segments)

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].duration == 2.0
        assert result[0].text == "Hello world"

    def test_gap_enforcement(self):
        """Test that gaps between segments are enforced."""
        standardizer = CaptionStandardizer(min_gap=0.08)
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="First"),
            Supervision(id="2", start=2.02, duration=2.0, text="Second"),  # Gap is only 0.02s
        ]

        result = standardizer.process(segments)

        # Check that gap is now at least 0.08s
        gap = result[1].start - (result[0].start + result[0].duration)
        assert gap >= 0.08, f"Gap {gap} should be >= 0.08"

    def test_min_duration_enforcement(self):
        """Test that minimum duration is enforced."""
        standardizer = CaptionStandardizer(min_duration=0.8)
        segments = [
            Supervision(id="1", start=0.0, duration=0.3, text="Short"),  # Below min
        ]

        result = standardizer.process(segments)

        assert result[0].duration >= 0.8

    def test_min_duration_with_next_segment(self):
        """Test min duration enforcement doesn't overlap next segment."""
        standardizer = CaptionStandardizer(min_duration=0.8, min_gap=0.08)
        segments = [
            Supervision(id="1", start=0.0, duration=0.3, text="Short"),
            Supervision(id="2", start=0.5, duration=2.0, text="Next"),  # Close to first
        ]

        result = standardizer.process(segments)

        # First segment should be extended but not overlap second
        first_end = result[0].start + result[0].duration
        gap = result[1].start - first_end
        assert gap >= 0.08, f"Gap {gap} should be >= 0.08"

    def test_max_duration_enforcement(self):
        """Test that maximum duration is enforced."""
        standardizer = CaptionStandardizer(max_duration=7.0)
        segments = [
            Supervision(id="1", start=0.0, duration=10.0, text="Too long"),  # Above max
        ]

        result = standardizer.process(segments)

        assert result[0].duration == 7.0

    def test_sorting_by_start_time(self):
        """Test that segments are sorted by start time."""
        standardizer = CaptionStandardizer()
        segments = [
            Supervision(id="2", start=5.0, duration=2.0, text="Second"),
            Supervision(id="1", start=0.0, duration=2.0, text="First"),
            Supervision(id="3", start=10.0, duration=2.0, text="Third"),
        ]

        result = standardizer.process(segments)

        assert result[0].id == "1"
        assert result[1].id == "2"
        assert result[2].id == "3"


class TestTextFormatting:
    """Test text formatting and line splitting."""

    def test_no_split_needed(self):
        """Test text that doesn't need splitting."""
        standardizer = CaptionStandardizer(max_chars_per_line=42)
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Short text"),
        ]

        result = standardizer.process(segments)

        assert "\n" not in result[0].text

    def test_english_text_split(self):
        """Test English text splitting at spaces."""
        standardizer = CaptionStandardizer(max_chars_per_line=30, max_lines=2)

        text = "This is a very long sentence that needs to be split"
        result = standardizer._smart_split_text(text)

        lines = result.split("\n")
        assert len(lines) <= 2
        # First line should be reasonable length
        assert len(lines[0]) <= 35  # Allow some flexibility

    def test_chinese_text_split(self):
        """Test Chinese text splitting at punctuation."""
        standardizer = CaptionStandardizer(max_chars_per_line=22, max_lines=2)

        text = "这是一段很长的中文字幕文本，需要被智能地分割成多行显示"
        result = standardizer._smart_split_text(text)

        lines = result.split("\n")
        assert len(lines) <= 2

    def test_split_at_cjk_punctuation(self):
        """Test that CJK punctuation is preferred for splitting."""
        standardizer = CaptionStandardizer(max_chars_per_line=15, max_lines=2)

        # Text with comma near middle
        text = "前面的文字，后面的文字"
        result = standardizer._smart_split_text(text)

        # Should split at the comma
        if "\n" in result:
            lines = result.split("\n")
            assert lines[0].endswith("，") or lines[0].endswith("字")

    def test_normalize_text(self):
        """Test text normalization."""
        standardizer = CaptionStandardizer()

        # Test multiple spaces
        assert standardizer._normalize_text("hello   world") == "hello world"

        # Test newlines
        assert standardizer._normalize_text("hello\nworld") == "hello world"

        # Test leading/trailing whitespace
        assert standardizer._normalize_text("  hello  ") == "hello"

    def test_existing_newlines_removed(self):
        """Test that existing newlines are removed and text is reformatted."""
        standardizer = CaptionStandardizer(max_chars_per_line=50)
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Line one\nLine two"),
        ]

        result = standardizer.process(segments)

        # Original newline should be replaced, text reformatted
        normalized = result[0].text.replace("\n", " ")
        assert "Line one" in normalized and "Line two" in normalized


class TestCaptionValidator:
    """Test CaptionValidator class."""

    def test_valid_segments(self):
        """Test validation of valid segments."""
        validator = CaptionValidator()
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Hello"),
            Supervision(id="2", start=2.5, duration=2.0, text="World"),
        ]

        result = validator.validate(segments)

        assert result.valid is True
        assert len(result.warnings) == 0
        assert result.segments_too_short == 0
        assert result.segments_too_long == 0
        assert result.gaps_too_small == 0

    def test_empty_input(self):
        """Test validation of empty input."""
        validator = CaptionValidator()
        result = validator.validate([])

        assert result.valid is True
        assert result.avg_cps == 0.0

    def test_detect_short_segments(self):
        """Test detection of segments that are too short."""
        validator = CaptionValidator(min_duration=0.8)
        segments = [
            Supervision(id="1", start=0.0, duration=0.3, text="Short"),  # Too short
            Supervision(id="2", start=2.0, duration=2.0, text="Normal"),
        ]

        result = validator.validate(segments)

        assert result.valid is False
        assert result.segments_too_short == 1
        assert any("duration" in w and "0.30s" in w for w in result.warnings)

    def test_detect_long_segments(self):
        """Test detection of segments that are too long."""
        validator = CaptionValidator(max_duration=7.0)
        segments = [
            Supervision(id="1", start=0.0, duration=10.0, text="Too long"),  # Too long
        ]

        result = validator.validate(segments)

        assert result.valid is False
        assert result.segments_too_long == 1

    def test_detect_small_gaps(self):
        """Test detection of gaps that are too small."""
        validator = CaptionValidator(min_gap=0.08)
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="First"),
            Supervision(id="2", start=2.02, duration=2.0, text="Second"),  # Gap is only 0.02s
        ]

        result = validator.validate(segments)

        assert result.valid is False
        assert result.gaps_too_small == 1

    def test_cps_calculation(self):
        """Test CPS (characters per second) calculation."""
        validator = CaptionValidator()
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="12345678"),  # 8 chars / 2s = 4 CPS
            Supervision(id="2", start=3.0, duration=1.0, text="123456"),  # 6 chars / 1s = 6 CPS
        ]

        result = validator.validate(segments)

        # Average CPS should be (4 + 6) / 2 = 5
        assert result.avg_cps == 5.0

    def test_cpl_calculation(self):
        """Test CPL (characters per line) calculation."""
        validator = CaptionValidator()
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Short"),
            Supervision(id="2", start=3.0, duration=2.0, text="A much longer line"),
        ]

        result = validator.validate(segments)

        assert result.max_cpl == len("A much longer line")

    def test_multiline_cpl(self):
        """Test CPL calculation with multiline text."""
        validator = CaptionValidator()
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="Short line\nA longer second line"),
        ]

        result = validator.validate(segments)

        assert result.max_cpl == len("A longer second line")

    def test_high_cps_warning(self):
        """Test warning for high CPS (fast reading speed)."""
        validator = CaptionValidator()  # optimal_cps = 17.0
        segments = [
            # 30 chars / 1s = 30 CPS, which is > 17 * 1.5 = 25.5
            Supervision(id="1", start=0.0, duration=1.0, text="A" * 30),
        ]

        result = validator.validate(segments)

        assert any("CPS" in w for w in result.warnings)


class TestConvenienceFunction:
    """Test the standardize_captions convenience function."""

    def test_standardize_captions_function(self):
        """Test the convenience function."""
        segments = [
            Supervision(id="1", start=0.0, duration=0.3, text="Short"),  # Will be extended
            Supervision(id="2", start=2.0, duration=2.0, text="Normal"),
        ]

        result = standardize_captions(segments, min_duration=0.8)

        assert len(result) == 2
        assert result[0].duration >= 0.8

    def test_standardize_captions_with_custom_params(self):
        """Test convenience function with custom parameters."""
        segments = [
            Supervision(id="1", start=0.0, duration=10.0, text="Long"),
        ]

        result = standardize_captions(segments, max_duration=5.0)

        assert result[0].duration == 5.0


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_overlapping_segments(self):
        """Test handling of overlapping segments."""
        standardizer = CaptionStandardizer(min_gap=0.08)
        segments = [
            Supervision(id="1", start=0.0, duration=3.0, text="First"),
            Supervision(id="2", start=2.0, duration=2.0, text="Second"),  # Overlaps with first
        ]

        result = standardizer.process(segments)

        # Should handle gracefully (first gets shortened)
        first_end = result[0].start + result[0].duration
        assert first_end <= result[1].start + 0.001  # Allow tiny floating point error

    def test_zero_duration_segment(self):
        """Test handling of zero-duration segment."""
        standardizer = CaptionStandardizer(min_duration=0.8)
        segments = [
            Supervision(id="1", start=0.0, duration=0.0, text="Empty"),
        ]

        result = standardizer.process(segments)

        # Should be extended to min_duration
        assert result[0].duration >= 0.8

    def test_preserve_segment_metadata(self):
        """Test that segment metadata is preserved."""
        standardizer = CaptionStandardizer()
        segments = [
            Supervision(
                id="seg1",
                recording_id="rec1",
                start=0.0,
                duration=2.0,
                text="Hello",
                speaker="Speaker1",
                language="en",
            ),
        ]

        result = standardizer.process(segments)

        assert result[0].id == "seg1"
        assert result[0].recording_id == "rec1"
        assert result[0].speaker == "Speaker1"
        assert result[0].language == "en"

    def test_very_long_text(self):
        """Test handling of very long text that exceeds max_lines."""
        standardizer = CaptionStandardizer(max_chars_per_line=20, max_lines=2)

        # Text much longer than 2 lines * 20 chars
        text = "This is a very very very long piece of text that cannot fit"
        result = standardizer._smart_split_text(text)

        lines = result.split("\n")
        assert len(lines) <= 2  # Should not exceed max_lines

    def test_text_with_only_punctuation(self):
        """Test text that is mostly punctuation."""
        standardizer = CaptionStandardizer()

        text = "...!!!"
        result = standardizer._smart_split_text(text)

        assert result == text  # Should not crash, return as-is

    def test_unicode_text(self):
        """Test handling of various Unicode characters."""
        standardizer = CaptionStandardizer(max_chars_per_line=50)
        segments = [
            Supervision(id="1", start=0.0, duration=2.0, text="日本語テキスト 한국어 текст"),
        ]

        result = standardizer.process(segments)

        # Should handle unicode gracefully
        assert "日本語" in result[0].text


class TestIntegration:
    """Integration tests combining standardization and validation."""

    def test_standardize_then_validate(self):
        """Test that standardized output passes validation."""
        # Create problematic segments (but with enough space to fix them)
        segments = [
            Supervision(id="1", start=0.0, duration=0.3, text="Short"),  # Too short
            Supervision(id="2", start=1.5, duration=2.0, text="Close"),  # Gap is fine after extending seg1
            Supervision(id="3", start=5.0, duration=10.0, text="Long"),  # Too long
        ]

        # Standardize
        standardizer = CaptionStandardizer(
            min_duration=0.8,
            max_duration=7.0,
            min_gap=0.08,
        )
        processed = standardizer.process(segments)

        # Validate
        validator = CaptionValidator(
            min_duration=0.8,
            max_duration=7.0,
            min_gap=0.08,
        )
        result = validator.validate(processed)

        # Should pass validation after standardization
        assert result.valid is True, f"Validation failed: {result.warnings}"


class TestStandardizationConfigMargins:
    """Test StandardizationConfig margin-related fields."""

    def test_default_margin_values(self):
        """Test default margin configuration values."""
        config = StandardizationConfig()

        assert config.start_margin == 0.08
        assert config.end_margin == 0.20
        assert config.margin_collision_mode == "trim"

    def test_custom_margin_values(self):
        """Test custom margin configuration values."""
        config = StandardizationConfig(
            start_margin=0.05,
            end_margin=0.15,
            margin_collision_mode="gap",
        )

        assert config.start_margin == 0.05
        assert config.end_margin == 0.15
        assert config.margin_collision_mode == "gap"

    def test_validation_start_margin(self):
        """Test start_margin validation."""
        with pytest.raises(ValueError, match="start_margin cannot be negative"):
            StandardizationConfig(start_margin=-0.1)

    def test_validation_end_margin(self):
        """Test end_margin validation."""
        with pytest.raises(ValueError, match="end_margin cannot be negative"):
            StandardizationConfig(end_margin=-0.1)

    def test_validation_margin_collision_mode(self):
        """Test margin_collision_mode validation."""
        with pytest.raises(ValueError, match="margin_collision_mode must be"):
            StandardizationConfig(margin_collision_mode="invalid")


class TestApplyMargins:
    """Test apply_margins functionality."""

    def test_apply_margins_basic(self):
        """Test basic margin application with word alignment."""
        seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="hello world",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.4, score=0.95),
                    AlignmentItem(symbol="world", start=1.6, duration=0.4, score=0.92),
                ]
            },
        )

        standardizer = CaptionStandardizer()
        result = standardizer.apply_margins([seg], start_margin=0.08, end_margin=0.20)

        # first_word_start=1.1, last_word_end=2.0
        # new_start = 1.1 - 0.08 = 1.02
        # new_end = 2.0 + 0.20 = 2.20
        assert len(result) == 1
        assert result[0].start == pytest.approx(1.02, abs=0.001)
        assert result[0].end == pytest.approx(2.20, abs=0.001)

    def test_apply_margins_no_alignment(self):
        """Test that segments without alignment data keep original timing."""
        seg = Supervision(id="1", start=1.0, duration=2.0, text="hello")

        standardizer = CaptionStandardizer()
        result = standardizer.apply_margins([seg], start_margin=0.08, end_margin=0.20)

        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].duration == 2.0

    def test_apply_margins_boundary_clamp_to_zero(self):
        """Test that start time is clamped to 0."""
        seg = Supervision(
            id="1",
            start=0.0,
            duration=2.0,
            text="hello",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=0.05, duration=0.5, score=0.95),
                ]
            },
        )

        standardizer = CaptionStandardizer()
        result = standardizer.apply_margins([seg], start_margin=0.10, end_margin=0.20)

        # first_word_start=0.05, start_margin=0.10
        # new_start = max(0, 0.05 - 0.10) = 0
        assert result[0].start == 0.0

    def test_apply_margins_collision_trim_mode(self):
        """Test collision handling in trim mode.

        When start_margin would cause overlap with previous segment,
        trim mode reduces the margin to fit within available space.
        """
        # Create segments where word boundaries are close but have some space
        segments = [
            Supervision(
                id="1",
                start=0.0,
                duration=1.0,
                text="first",
                alignment={
                    "word": [
                        AlignmentItem(symbol="first", start=0.1, duration=0.5, score=0.95),
                    ]
                },
            ),
            Supervision(
                id="2",
                start=1.0,
                duration=1.0,
                text="second",
                alignment={
                    "word": [
                        # Word starts at 1.0, so start_margin=0.10 would put start at 0.90
                        # First segment ends at 0.6 + 0.10 = 0.70 with end_margin
                        # This leaves room for second to start at 0.78 (with min_gap=0.08)
                        AlignmentItem(symbol="second", start=1.0, duration=0.5, score=0.92),
                    ]
                },
            ),
        ]

        standardizer = CaptionStandardizer(min_gap=0.08)
        standardizer.config.margin_collision_mode = "trim"
        result = standardizer.apply_margins(segments, start_margin=0.10, end_margin=0.10)

        # First segment: word 0.1-0.6, with margins → 0.0-0.70
        # Second segment: word 1.0-1.5, with margins → 0.90-1.60
        # Gap = 0.90 - 0.70 = 0.20, which is >= 0.08
        first_end = result[0].start + result[0].duration
        gap = result[1].start - first_end
        assert gap >= 0.08 - 0.001, f"Gap {gap} should be >= 0.08"

    def test_apply_margins_collision_gap_mode(self):
        """Test collision handling in gap mode."""
        segments = [
            Supervision(
                id="1",
                start=0.0,
                duration=1.0,
                text="first",
                alignment={
                    "word": [
                        AlignmentItem(symbol="first", start=0.1, duration=0.8, score=0.95),
                    ]
                },
            ),
            Supervision(
                id="2",
                start=1.0,
                duration=1.0,
                text="second",
                alignment={
                    "word": [
                        AlignmentItem(symbol="second", start=1.05, duration=0.8, score=0.92),
                    ]
                },
            ),
        ]

        standardizer = CaptionStandardizer(min_gap=0.08)
        standardizer.config.margin_collision_mode = "gap"
        result = standardizer.apply_margins(segments, start_margin=0.20, end_margin=0.20)

        # Check that min_gap is strictly enforced
        first_end = result[0].start + result[0].duration
        gap = result[1].start - first_end
        assert gap >= 0.08 - 0.001, f"Gap {gap} should be >= 0.08"

    def test_apply_margins_preserves_metadata(self):
        """Test that segment metadata is preserved."""
        seg = Supervision(
            id="seg1",
            recording_id="rec1",
            start=1.0,
            duration=2.0,
            text="hello",
            speaker="Speaker1",
            language="en",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        standardizer = CaptionStandardizer()
        result = standardizer.apply_margins([seg])

        assert result[0].id == "seg1"
        assert result[0].recording_id == "rec1"
        assert result[0].speaker == "Speaker1"
        assert result[0].language == "en"
        assert result[0].alignment is not None

    def test_apply_margins_empty_input(self):
        """Test with empty input list."""
        standardizer = CaptionStandardizer()
        result = standardizer.apply_margins([])

        assert result == []

    def test_apply_margins_config_defaults(self):
        """Test that config defaults are used when not overridden."""
        seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="hello",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        standardizer = CaptionStandardizer()
        standardizer.config.start_margin = 0.05
        standardizer.config.end_margin = 0.10

        result = standardizer.apply_margins([seg])  # No explicit margins

        # Should use config defaults: start=1.1-0.05=1.05, end=1.6+0.10=1.70
        assert result[0].start == pytest.approx(1.05, abs=0.001)
        assert result[0].end == pytest.approx(1.70, abs=0.001)


class TestApplyMarginsConvenienceFunction:
    """Test the apply_margins_to_captions convenience function."""

    def test_apply_margins_to_captions_basic(self):
        """Test the convenience function."""
        seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="hello",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        result = apply_margins_to_captions([seg], start_margin=0.05, end_margin=0.15)

        assert len(result) == 1
        # first_word_start=1.1, last_word_end=1.6
        # new_start = 1.1 - 0.05 = 1.05
        # new_end = 1.6 + 0.15 = 1.75
        assert result[0].start == pytest.approx(1.05, abs=0.001)
        assert result[0].end == pytest.approx(1.75, abs=0.001)


class TestCaptionWithMargins:
    """Test Caption.with_margins() method."""

    def test_caption_with_margins_basic(self):
        """Test Caption.with_margins() method."""
        seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="hello",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        caption = Caption(supervisions=[seg], language="en")
        adjusted = caption.with_margins(start_margin=0.05, end_margin=0.15)

        assert len(adjusted.supervisions) == 1
        assert adjusted.supervisions[0].start == pytest.approx(1.05, abs=0.001)
        assert adjusted.supervisions[0].end == pytest.approx(1.75, abs=0.001)
        assert adjusted.language == "en"

    def test_caption_with_margins_returns_new_instance(self):
        """Test that with_margins returns a new Caption instance."""
        seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="hello",
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        caption = Caption(supervisions=[seg])
        adjusted = caption.with_margins()

        assert caption is not adjusted
        assert caption.supervisions[0].start == 1.0  # Original unchanged

    def test_caption_with_margins_uses_alignments(self):
        """Test that with_margins prefers alignments over supervisions."""
        original_seg = Supervision(id="1", start=0.0, duration=5.0, text="original")
        aligned_seg = Supervision(
            id="1",
            start=1.0,
            duration=2.0,
            text="aligned",
            alignment={
                "word": [
                    AlignmentItem(symbol="aligned", start=1.1, duration=0.5, score=0.95),
                ]
            },
        )

        caption = Caption(supervisions=[original_seg], alignments=[aligned_seg])
        adjusted = caption.with_margins(start_margin=0.05, end_margin=0.10)

        # Should use aligned_seg, not original_seg
        assert adjusted.supervisions[0].text == "aligned"
        assert adjusted.supervisions[0].start == pytest.approx(1.05, abs=0.001)
