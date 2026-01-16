"""Tests for ASS karaoke tag generation."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.supervision import Supervision
from lattifai.config.caption import CaptionStyle, KaraokeConfig


class TestASSKaraoke:
    """Test ASS karaoke tag generation."""

    def test_karaoke_sweep_effect(self):
        """Sweep effect should use \\kf tag."""
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
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        # \kf45 means 45 centiseconds (0.45s)
        assert "{\\kf45}Hello" in content
        assert "{\\kf285}world" in content

    def test_karaoke_instant_effect(self):
        """Instant effect should use \\k tag."""
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
        config = KaraokeConfig(enabled=True, effect="instant")
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\k45}Hello" in content
        assert "{\\k285}world" in content

    def test_karaoke_outline_effect(self):
        """Outline effect should use \\ko tag."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        config = KaraokeConfig(enabled=True, effect="outline")
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\ko50}Hello" in content

    def test_karaoke_style_in_output(self):
        """Karaoke style should be defined in ASS output."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        # Should have Karaoke style defined
        assert "Style: Karaoke" in content or "Karaoke," in content

    def test_fallback_without_alignment(self):
        """Without alignment, should output normal ASS."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "No alignment" in content
        assert "{\\k" not in content  # No karaoke tags

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
        result = ASSFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        # Should NOT have karaoke tags
        assert "{\\k" not in content
