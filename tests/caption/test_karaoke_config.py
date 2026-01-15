"""Tests for karaoke configuration classes."""

import pytest

from lattifai.caption.formats.karaoke import KaraokeConfig, KaraokeFonts, KaraokeStyle


class TestKaraokeFonts:
    """Test KaraokeFonts constants."""

    def test_western_fonts_exist(self):
        """Western font constants should be defined."""
        assert KaraokeFonts.ARIAL == "Arial"
        assert KaraokeFonts.IMPACT == "Impact"
        assert KaraokeFonts.VERDANA == "Verdana"

    def test_chinese_fonts_exist(self):
        """Chinese font constants should be defined."""
        assert KaraokeFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert KaraokeFonts.MICROSOFT_YAHEI == "Microsoft YaHei"
        assert KaraokeFonts.PINGFANG_SC == "PingFang SC"

    def test_japanese_fonts_exist(self):
        """Japanese font constants should be defined."""
        assert KaraokeFonts.NOTO_SANS_JP == "Noto Sans JP"
        assert KaraokeFonts.MEIRYO == "Meiryo"


class TestKaraokeStyle:
    """Test KaraokeStyle dataclass."""

    def test_default_values(self):
        """Default style should have sensible defaults."""
        style = KaraokeStyle()
        assert style.effect == "sweep"
        assert style.primary_color == "#00FFFF"
        assert style.secondary_color == "#FFFFFF"
        assert style.font_name == KaraokeFonts.ARIAL
        assert style.font_size == 48
        assert style.bold is True

    def test_custom_values(self):
        """Custom values should override defaults."""
        style = KaraokeStyle(
            effect="instant",
            primary_color="#FF00FF",
            font_name=KaraokeFonts.NOTO_SANS_SC,
            font_size=56,
        )
        assert style.effect == "instant"
        assert style.primary_color == "#FF00FF"
        assert style.font_name == "Noto Sans SC"
        assert style.font_size == 56


class TestKaraokeConfig:
    """Test KaraokeConfig dataclass."""

    def test_default_config(self):
        """Default config should work."""
        config = KaraokeConfig()
        assert config.enabled is True
        assert isinstance(config.style, KaraokeStyle)
        assert config.lrc_precision == "millisecond"
        assert config.ttml_timing_mode == "Word"

    def test_lrc_metadata(self):
        """LRC metadata should be configurable."""
        config = KaraokeConfig(lrc_metadata={"ar": "Artist", "ti": "Title"})
        assert config.lrc_metadata["ar"] == "Artist"
        assert config.lrc_metadata["ti"] == "Title"
