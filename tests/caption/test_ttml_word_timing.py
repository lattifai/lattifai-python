"""Tests for TTML word-level timing."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.ttml import TTMLFormat
from lattifai.caption.supervision import Supervision
from lattifai.config.caption import KaraokeConfig


class TestTTMLWordTiming:
    """Test TTML word-level timing output."""

    def test_word_timing_attribute(self):
        """Word-level with karaoke should add itunes:timing attribute."""
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
        result = TTMLFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert 'itunes:timing="Word"' in content or "timing" in content.lower()

    def test_word_spans_in_output(self):
        """Each word should be wrapped in span with begin/end when karaoke_config.enabled=True."""
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
        result = TTMLFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "<span" in content
        assert 'begin="00:00:15.200"' in content
        assert "Hello" in content
        assert "world" in content

    def test_paragraph_timing(self):
        """Paragraph should have overall begin/end."""
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
        result = TTMLFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "<p " in content
        assert 'begin="00:00:15.200"' in content

    def test_fallback_without_alignment(self):
        """Without alignment, should output normal TTML."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        karaoke_config = KaraokeConfig(enabled=True)
        result = TTMLFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "No alignment" in content
        assert content.count("<span") <= 1

    def test_word_per_paragraph_without_karaoke(self):
        """word_level=True without karaoke should output word per paragraph."""
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
        result = TTMLFormat.to_bytes(sups, word_level=True)  # No karaoke_config
        content = result.decode("utf-8")

        # Should have 2 paragraphs (one for each word)
        assert content.count("<p ") == 2
        # No itunes:timing attribute
        assert 'itunes:timing="Word"' not in content

    def test_word_level_false_uses_original(self):
        """word_level=False should use original behavior."""
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
        result = TTMLFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        assert 'itunes:timing="Word"' not in content
