"""Tests for Enhanced LRC format."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.lrc import LRCFormat
from lattifai.caption.supervision import Supervision
from lattifai.config.caption import KaraokeConfig


class TestLRCFormatWrite:
    """Test LRC format writing."""

    def test_standard_lrc_output(self):
        """Standard LRC should output line timestamps only."""
        sups = [
            Supervision(text="Hello world", start=15.2, duration=3.3),
            Supervision(text="This is karaoke", start=18.5, duration=2.0),
        ]
        result = LRCFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        assert "[00:15.200]Hello world" in content
        assert "[00:18.500]This is karaoke" in content
        # No word-level timestamps
        assert "<" not in content

    def test_enhanced_lrc_word_level(self):
        """Enhanced LRC should include word timestamps when karaoke_config.enabled=True."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        karaoke_config = KaraokeConfig(enabled=True)
        result = LRCFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "[00:15.200]" in content
        assert "<00:15.200>Hello" in content
        assert "<00:15.650>world" in content

    def test_word_per_line_output(self):
        """word_level=True without karaoke should output word per line."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = LRCFormat.to_bytes(sups, word_level=True)  # No karaoke_config
        content = result.decode("utf-8")

        # Each word should be on its own line
        assert "[00:15.200]Hello" in content
        assert "[00:15.650]world" in content
        # No enhanced LRC format
        assert "<" not in content

    def test_lrc_with_metadata(self):
        """LRC should include metadata when karaoke_config.enabled=True."""
        sups = [Supervision(text="Hello", start=0.0, duration=1.0)]
        config = KaraokeConfig(enabled=True, lrc_metadata={"ar": "Artist", "ti": "Title", "al": "Album"})
        result = LRCFormat.to_bytes(sups, word_level=False, karaoke_config=config)
        content = result.decode("utf-8")

        assert "[ar:Artist]" in content
        assert "[ti:Title]" in content
        assert "[al:Album]" in content

    def test_lrc_centisecond_precision(self):
        """LRC should support centisecond precision."""
        sups = [Supervision(text="Hello", start=15.234, duration=1.0)]
        config = KaraokeConfig(enabled=False, lrc_precision="centisecond")
        result = LRCFormat.to_bytes(sups, word_level=False, karaoke_config=config)
        content = result.decode("utf-8")

        # Centisecond: [00:15.23] not [00:15.234]
        assert "[00:15.23]" in content

    def test_lrc_fallback_without_alignment(self):
        """Word-level should fallback to line-level without alignment data."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        karaoke_config = KaraokeConfig(enabled=True)
        result = LRCFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "[00:10.000]No alignment" in content
        assert "<" not in content  # No word timestamps


class TestLRCFormatRead:
    """Test LRC format reading."""

    def test_read_standard_lrc(self):
        """Read standard LRC content."""
        content = """[00:15.20]Hello world
[00:18.50]This is karaoke
"""
        sups = LRCFormat.read(content)

        assert len(sups) == 2
        assert sups[0].text == "Hello world"
        assert abs(sups[0].start - 15.2) < 0.01
        assert sups[1].text == "This is karaoke"
        assert abs(sups[1].start - 18.5) < 0.01

    def test_read_enhanced_lrc(self):
        """Read enhanced LRC with word-level timestamps."""
        content = "[00:15.200]<00:15.200>Hello <00:15.650>world\n"
        sups = LRCFormat.read(content)

        assert len(sups) == 1
        assert sups[0].text == "Hello world"
        assert sups[0].alignment is not None
        assert "word" in sups[0].alignment
        assert len(sups[0].alignment["word"]) == 2
        assert sups[0].alignment["word"][0].symbol == "Hello"
        assert abs(sups[0].alignment["word"][0].start - 15.2) < 0.01
        assert sups[0].alignment["word"][1].symbol == "world"
        assert abs(sups[0].alignment["word"][1].start - 15.65) < 0.01

    def test_read_lrc_with_metadata(self):
        """Read LRC and ignore metadata lines."""
        content = """[ar:Artist Name]
[ti:Song Title]
[al:Album Name]

[00:00.00]First line
"""
        sups = LRCFormat.read(content)

        assert len(sups) == 1
        assert sups[0].text == "First line"

    def test_read_lrc_millisecond_precision(self):
        """Read LRC with millisecond precision."""
        content = "[00:15.234]Hello\n"
        sups = LRCFormat.read(content)

        assert len(sups) == 1
        assert abs(sups[0].start - 15.234) < 0.001


class TestLRCFormatRoundTrip:
    """Test LRC format roundtrip (write then read)."""

    def test_standard_roundtrip(self):
        """Standard LRC should survive roundtrip."""
        original = [
            Supervision(text="Hello world", start=15.2, duration=3.3),
            Supervision(text="This is karaoke", start=18.5, duration=2.0),
        ]
        content = LRCFormat.to_bytes(original, word_level=False).decode("utf-8")
        restored = LRCFormat.read(content)

        assert len(restored) == 2
        assert restored[0].text == original[0].text
        assert abs(restored[0].start - original[0].start) < 0.01
        assert restored[1].text == original[1].text
        assert abs(restored[1].start - original[1].start) < 0.01

    def test_enhanced_roundtrip(self):
        """Enhanced LRC with word-level should survive roundtrip."""
        original = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        karaoke_config = KaraokeConfig(enabled=True)
        content = LRCFormat.to_bytes(original, word_level=True, karaoke_config=karaoke_config).decode("utf-8")
        restored = LRCFormat.read(content)

        assert len(restored) == 1
        assert restored[0].text == "Hello world"
        assert restored[0].alignment is not None
        assert len(restored[0].alignment["word"]) == 2
        assert restored[0].alignment["word"][0].symbol == "Hello"
        assert abs(restored[0].alignment["word"][0].start - 15.2) < 0.01
