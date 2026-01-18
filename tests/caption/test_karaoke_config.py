"""Tests for caption style and karaoke configuration classes."""

import pytest

from lattifai.config.caption import CaptionFonts, CaptionStyle, KaraokeConfig


class TestCaptionFonts:
    """Test CaptionFonts constants."""

    def test_western_fonts_exist(self):
        """Western font constants should be defined."""
        assert CaptionFonts.ARIAL == "Arial"
        assert CaptionFonts.IMPACT == "Impact"
        assert CaptionFonts.VERDANA == "Verdana"

    def test_chinese_fonts_exist(self):
        """Chinese font constants should be defined."""
        assert CaptionFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert CaptionFonts.MICROSOFT_YAHEI == "Microsoft YaHei"
        assert CaptionFonts.PINGFANG_SC == "PingFang SC"

    def test_japanese_fonts_exist(self):
        """Japanese font constants should be defined."""
        assert CaptionFonts.NOTO_SANS_JP == "Noto Sans JP"
        assert CaptionFonts.MEIRYO == "Meiryo"


class TestCaptionStyle:
    """Test CaptionStyle dataclass."""

    def test_default_values(self):
        """Default style should have sensible defaults."""
        style = CaptionStyle()
        assert style.primary_color == "#FFFFFF"
        assert style.secondary_color == "#00FFFF"
        assert style.font_name == CaptionFonts.ARIAL
        assert style.font_size == 48
        assert style.bold is False

    def test_custom_values(self):
        """Custom values should override defaults."""
        style = CaptionStyle(
            primary_color="#FF00FF",
            font_name=CaptionFonts.NOTO_SANS_SC,
            font_size=56,
            bold=True,
        )
        assert style.primary_color == "#FF00FF"
        assert style.font_name == "Noto Sans SC"
        assert style.font_size == 56
        assert style.bold is True


class TestKaraokeConfig:
    """Test KaraokeConfig dataclass."""

    def test_default_config(self):
        """Default config should work."""
        config = KaraokeConfig()
        assert config.enabled is False  # Default is False, must be explicitly enabled
        assert config.effect == "sweep"
        assert isinstance(config.style, CaptionStyle)
        assert config.lrc_precision == "millisecond"
        assert config.ttml_timing_mode == "Word"

    def test_effect_options(self):
        """Effect should support sweep, instant, outline."""
        config_sweep = KaraokeConfig(effect="sweep")
        assert config_sweep.effect == "sweep"

        config_instant = KaraokeConfig(effect="instant")
        assert config_instant.effect == "instant"

        config_outline = KaraokeConfig(effect="outline")
        assert config_outline.effect == "outline"

    def test_lrc_metadata(self):
        """LRC metadata should be configurable."""
        config = KaraokeConfig(lrc_metadata={"ar": "Artist", "ti": "Title"})
        assert config.lrc_metadata["ar"] == "Artist"
        assert config.lrc_metadata["ti"] == "Title"
