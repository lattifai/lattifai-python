"""Tests for subtitle shift functionality"""

import pytest  # noqa: F401

from lattifai.subtitle import Supervision


class TestSubtitleShift:
    """Test cases for subtitle timestamp shifting"""

    def test_shift_positive(self):
        """Test shifting timestamps forward (delay)"""
        supervisions = [
            Supervision(id="1", start=0.0, duration=2.0, text="First line"),
            Supervision(id="2", start=2.5, duration=1.5, text="Second line"),
            Supervision(id="3", start=5.0, duration=3.0, text="Third line"),
        ]

        shift_amount = 2.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert supervisions[0].start == 2.0
        assert supervisions[1].start == 4.5
        assert supervisions[2].start == 7.0

    def test_shift_negative(self):
        """Test shifting timestamps backward (advance)"""
        supervisions = [
            Supervision(id="1", start=5.0, duration=2.0, text="First line"),
            Supervision(id="2", start=7.5, duration=1.5, text="Second line"),
            Supervision(id="3", start=10.0, duration=3.0, text="Third line"),
        ]

        shift_amount = -2.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert supervisions[0].start == 3.0
        assert supervisions[1].start == 5.5
        assert supervisions[2].start == 8.0

    def test_shift_negative_with_clamp(self):
        """Test that negative shifts don't go below 0.0"""
        supervisions = [
            Supervision(id="1", start=1.0, duration=2.0, text="First line"),
            Supervision(id="2", start=3.0, duration=1.5, text="Second line"),
        ]

        shift_amount = -2.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert supervisions[0].start == 0.0  # Clamped to 0.0
        assert supervisions[1].start == 1.0

    def test_shift_zero(self):
        """Test that zero shift leaves timestamps unchanged"""
        supervisions = [
            Supervision(id="1", start=1.0, duration=2.0, text="First line"),
            Supervision(id="2", start=5.0, duration=1.5, text="Second line"),
        ]

        original_starts = [sup.start for sup in supervisions]

        shift_amount = 0.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        for i, sup in enumerate(supervisions):
            assert sup.start == original_starts[i]

    def test_shift_preserves_duration(self):
        """Test that shifting preserves duration values"""
        supervisions = [
            Supervision(id="1", start=0.0, duration=2.5, text="First line"),
            Supervision(id="2", start=3.0, duration=1.8, text="Second line"),
        ]

        original_durations = [sup.duration for sup in supervisions]

        shift_amount = 5.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        for i, sup in enumerate(supervisions):
            assert sup.duration == original_durations[i]

    def test_shift_preserves_text(self):
        """Test that shifting preserves text content"""
        supervisions = [
            Supervision(id="1", start=0.0, duration=2.0, text="Hello world"),
            Supervision(id="2", start=3.0, duration=1.5, text="Goodbye world"),
        ]

        original_texts = [sup.text for sup in supervisions]

        shift_amount = 1.5
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        for i, sup in enumerate(supervisions):
            assert sup.text == original_texts[i]

    def test_shift_preserves_speaker(self):
        """Test that shifting preserves speaker labels"""
        supervisions = [
            Supervision(id="1", start=0.0, duration=2.0, text="Line 1", speaker="Speaker A"),
            Supervision(id="2", start=3.0, duration=1.5, text="Line 2", speaker="Speaker B"),
        ]

        original_speakers = [sup.speaker for sup in supervisions]

        shift_amount = 2.5
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        for i, sup in enumerate(supervisions):
            assert sup.speaker == original_speakers[i]

    def test_shift_fractional_seconds(self):
        """Test shifting with fractional seconds"""
        supervisions = [
            Supervision(id="1", start=1.234, duration=2.0, text="First line"),
            Supervision(id="2", start=4.567, duration=1.5, text="Second line"),
        ]

        shift_amount = 0.111
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert abs(supervisions[0].start - 1.345) < 0.001
        assert abs(supervisions[1].start - 4.678) < 0.001

    def test_shift_empty_list(self):
        """Test shifting an empty list of supervisions"""
        supervisions = []

        shift_amount = 5.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert len(supervisions) == 0

    def test_shift_large_positive_value(self):
        """Test shifting with large positive offset"""
        supervisions = [
            Supervision(id="1", start=0.0, duration=2.0, text="First line"),
        ]

        shift_amount = 1000.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert supervisions[0].start == 1000.0

    def test_shift_large_negative_value(self):
        """Test shifting with large negative offset (clamped to 0)"""
        supervisions = [
            Supervision(id="1", start=5.0, duration=2.0, text="First line"),
        ]

        shift_amount = -1000.0
        for sup in supervisions:
            sup.start = max(0.0, sup.start + shift_amount)

        assert supervisions[0].start == 0.0
