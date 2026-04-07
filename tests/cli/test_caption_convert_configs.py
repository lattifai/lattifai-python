"""Tests for caption convert format-config dispatch and type annotations."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lattifai.caption.config import ASSConfig, LRCConfig, RenderConfig, StandardizationConfig
from lattifai.caption.formats.nle.fcpxml import FCPXMLConfig
from lattifai.caption.formats.nle.premiere import PremiereXMLConfig
from lattifai.caption.formats.ttml import TTMLConfig
from lattifai.cli.caption import convert

SAMPLE_SRT = "tests/data/SA1.srt"


@pytest.fixture
def sample_srt():
    return SAMPLE_SRT


class TestConvertFormatConfigDispatch:
    """Verify convert() passes the correct format_config to Caption.write() based on output extension."""

    @patch("lattifai.data.Caption")
    def test_ass_extension_uses_ass_config(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(font_size=36)
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg)

        mock_caption.write.assert_called_once()
        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is ass_cfg

    @patch("lattifai.data.Caption")
    def test_ssa_extension_uses_ass_config(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(font_size=24)
        out = tmp_path / "out.ssa"
        convert(sample_srt, str(out), ass=ass_cfg)

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is ass_cfg

    @patch("lattifai.data.Caption")
    def test_lrc_extension_uses_lrc_config(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        lrc_cfg = LRCConfig()
        out = tmp_path / "out.lrc"
        convert(sample_srt, str(out), lrc=lrc_cfg)

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is lrc_cfg

    @patch("lattifai.data.Caption")
    def test_ttml_extension_uses_ttml_config(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ttml_cfg = TTMLConfig()
        out = tmp_path / "out.ttml"
        convert(sample_srt, str(out), ttml=ttml_cfg)

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is ttml_cfg

    @patch("lattifai.data.Caption")
    def test_fcpxml_extension_uses_fcpxml_config(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        fcpxml_cfg = FCPXMLConfig()
        out = tmp_path / "out.fcpxml"
        convert(sample_srt, str(out), fcpxml=fcpxml_cfg)

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is fcpxml_cfg

    @patch("lattifai.data.Caption")
    def test_srt_extension_fallback_to_none(self, mock_caption_cls, tmp_path, sample_srt):
        """When output is .srt and no ass config, format_config should be None."""
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        out = tmp_path / "out.srt"
        convert(sample_srt, str(out))

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["format_config"] is None

    @patch("lattifai.data.Caption")
    def test_render_and_standardization_passed_through(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        render_cfg = RenderConfig(word_level=True)
        std_cfg = StandardizationConfig()
        out = tmp_path / "out.vtt"
        convert(sample_srt, str(out), render=render_cfg, standardization=std_cfg)

        call_kwargs = mock_caption.write.call_args
        assert call_kwargs[1]["render"] is render_cfg
        assert call_kwargs[1]["standardization"] is std_cfg


class TestConvertKaraokeAutoRender:
    """Verify karaoke_effect auto-enables word_level in RenderConfig."""

    @patch("lattifai.data.Caption")
    def test_karaoke_creates_render_with_word_level(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(karaoke_effect="sweep")
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg)

        call_kwargs = mock_caption.write.call_args
        render = call_kwargs[1]["render"]
        assert render is not None
        assert render.word_level is True

    @patch("lattifai.data.Caption")
    def test_karaoke_sets_word_level_on_existing_render(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(karaoke_effect="instant")
        render_cfg = RenderConfig(word_level=False, include_speaker_in_text=True)
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg, render=render_cfg)

        call_kwargs = mock_caption.write.call_args
        render = call_kwargs[1]["render"]
        assert render.word_level is True
        assert render.include_speaker_in_text is True


class TestConvertReturnType:
    """Verify convert() returns a Pathlike value."""

    @patch("lattifai.data.Caption")
    def test_returns_output_path(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        out = tmp_path / "out.srt"
        result = convert(sample_srt, str(out))
        assert result == str(out)


class TestConvertE2E:
    """End-to-end convert tests using real Caption I/O (no mocks)."""

    def test_srt_to_vtt(self, tmp_path, sample_srt):
        out = tmp_path / "out.vtt"
        result = convert(sample_srt, str(out))
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "WEBVTT" in content

    def test_srt_to_json(self, tmp_path, sample_srt):
        out = tmp_path / "out.json"
        result = convert(sample_srt, str(out))
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_srt_to_lrc(self, tmp_path, sample_srt):
        out = tmp_path / "out.lrc"
        result = convert(sample_srt, str(out), lrc=LRCConfig())
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "[" in content  # LRC uses [mm:ss.xx] timestamps

    def test_srt_to_ass(self, tmp_path, sample_srt):
        out = tmp_path / "out.ass"
        result = convert(sample_srt, str(out), ass=ASSConfig(font_size=28))
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "[Script Info]" in content
