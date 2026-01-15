"""Integration tests for word-level export across all formats."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats import get_writer
from lattifai.caption.formats.karaoke import KaraokeConfig, KaraokeFonts, KaraokeStyle
from lattifai.caption.supervision import Supervision


@pytest.fixture
def supervision_with_alignment():
    """Create test supervision with word alignment."""
    return Supervision(
        text="Hello beautiful world",
        start=15.2,
        duration=3.3,
        alignment={
            "word": [
                AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                AlignmentItem(symbol="beautiful", start=15.65, duration=0.75),
                AlignmentItem(symbol="world", start=16.4, duration=2.1),
            ]
        },
    )


class TestWordLevelIntegration:
    """Test word-level export across all supported formats."""

    def test_lrc_writer_registered(self):
        """LRC format should be registered."""
        writer = get_writer("lrc")
        assert writer is not None

    def test_all_formats_support_word_level(self, supervision_with_alignment):
        """All karaoke formats should support word_level parameter."""
        formats = ["lrc", "ass", "ttml"]
        for fmt in formats:
            writer = get_writer(fmt)
            assert writer is not None, f"Format {fmt} not registered"

            # Should not raise TypeError
            result = writer.to_bytes([supervision_with_alignment], word_level=True)
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_custom_karaoke_config(self, supervision_with_alignment):
        """Custom karaoke config should work across formats."""
        style = KaraokeStyle(
            effect="instant",
            primary_color="#FF00FF",
            font_name=KaraokeFonts.NOTO_SANS_SC,
        )
        config = KaraokeConfig(
            style=style,
            lrc_metadata={"ar": "Test Artist"},
        )

        # LRC
        lrc_writer = get_writer("lrc")
        lrc_result = lrc_writer.to_bytes([supervision_with_alignment], word_level=True, karaoke_config=config)
        assert b"[ar:Test Artist]" in lrc_result

        # ASS
        ass_writer = get_writer("ass")
        ass_result = ass_writer.to_bytes([supervision_with_alignment], word_level=True, karaoke_config=config)
        assert b"{\\k" in ass_result  # instant effect uses \k

    def test_graceful_fallback(self):
        """Formats should gracefully handle missing alignment."""
        sup_no_alignment = Supervision(text="No alignment data", start=0.0, duration=1.0)

        for fmt in ["lrc", "ass", "ttml"]:
            writer = get_writer(fmt)
            # Should not raise, should output line-level
            result = writer.to_bytes([sup_no_alignment], word_level=True)
            assert b"No alignment data" in result or b"No alignment" in result
