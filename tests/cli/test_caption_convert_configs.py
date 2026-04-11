"""Tests for caption convert format-config dispatch and type annotations."""

import importlib.resources
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lattifai.data.selftest as _selftest
from lattifai.caption.config import ASSConfig, LRCConfig, RenderConfig, StandardizationConfig
from lattifai.caption.formats.nle.fcpxml import FCPXMLConfig
from lattifai.caption.formats.nle.premiere import PremiereXMLConfig
from lattifai.caption.formats.ttml import TTMLConfig
from lattifai.cli.caption import convert

SAMPLE_SRT = str(importlib.resources.files(_selftest) / "test.srt")


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
    def test_karaoke_respects_user_provided_render_word_level_false(self, mock_caption_cls, tmp_path, sample_srt):
        """When the user explicitly passes render with word_level=False, the
        CLI must NOT auto-upgrade it to True — even if karaoke_effect is set.
        Auto-enable only kicks in when `render` is None (user didn't pass it).

        This is the regression the Phase 3 CLI fix addresses: previously the
        auto-enable branch unconditionally clobbered render.word_level=True
        whenever karaoke_effect was set, making line-scope kinetic impossible
        from the command line while still carrying a karaoke_effect argument."""
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(karaoke_effect="instant")
        render_cfg = RenderConfig(word_level=False, include_speaker_in_text=True)
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg, render=render_cfg)

        call_kwargs = mock_caption.write.call_args
        render = call_kwargs[1]["render"]
        assert render.word_level is False  # user's explicit value survives
        assert render.include_speaker_in_text is True


class TestConvertKineticStyle:
    """Verify kinetic_style flows through convert() with scope-aware behavior.

    Phase 2 architecture: kinetic_style does NOT auto-enable word_level.
    Line-scope (default) applies at event start; word-scope (explicit) fires
    per word. See lattifai-captions kinetic.py for the preset registry.
    """

    @patch("lattifai.data.Caption")
    def test_kinetic_alone_does_not_force_word_level(self, mock_caption_cls, tmp_path, sample_srt):
        """kinetic_style alone gives line-scope behavior; word_level stays default (False)."""
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(kinetic_style="fade")
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg)

        render = mock_caption.write.call_args[1]["render"]
        # word_level must NOT be auto-enabled
        assert render is None or render.word_level is False
        # karaoke_effect stays None — line-scope kinetic does not need it
        assert ass_cfg.karaoke_effect is None

    @patch("lattifai.data.Caption")
    def test_kinetic_with_explicit_word_level_auto_enables_sweep(self, mock_caption_cls, tmp_path, sample_srt):
        """When user sets word_level=True and kinetic_style, karaoke_effect auto-defaults to sweep."""
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(kinetic_style="bounce")
        render_cfg = RenderConfig(word_level=True)
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg, render=render_cfg)

        # karaoke_effect auto-sweeps so \kf tags exist for kinetic overrides
        assert ass_cfg.karaoke_effect == "sweep"

    @patch("lattifai.data.Caption")
    def test_explicit_word_level_false_is_respected(self, mock_caption_cls, tmp_path, sample_srt):
        """Regression: passing render.word_level=false with karaoke_effect set
        must NOT auto-upgrade word_level to True. Previously the CLI would
        unconditionally clobber render.word_level whenever karaoke_effect was
        set, which made it impossible to request line-scope kinetic from the
        command line while still carrying a karaoke_effect argument."""
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(karaoke_effect="sweep", kinetic_style="fade")
        render_cfg = RenderConfig(word_level=False)
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg, render=render_cfg)

        # Explicit False must survive the auto-enable branch.
        assert render_cfg.word_level is False

    @patch("lattifai.data.Caption")
    def test_kinetic_preserves_explicit_effect(self, mock_caption_cls, tmp_path, sample_srt):
        mock_caption = MagicMock()
        mock_caption.supervisions = [MagicMock(text="hello")]
        mock_caption_cls.read.return_value = mock_caption

        ass_cfg = ASSConfig(karaoke_effect="instant", kinetic_style="pop")
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ass_cfg)

        # Explicit instant is preserved, not overridden to sweep
        assert ass_cfg.karaoke_effect == "instant"

    def test_kinetic_invalid_rejected_at_config(self):
        """Fail-fast: unsupported kinetic_style raises at ASSConfig construction."""
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            ASSConfig(kinetic_style="totally_bogus")

    def test_kinetic_line_scope_e2e_srt_to_ass(self, tmp_path, sample_srt):
        """Line-scope fade applies at event start without needing word alignment."""
        out = tmp_path / "out.ass"
        convert(sample_srt, str(out), ass=ASSConfig(kinetic_style="fade"))
        content = Path(out).read_text()
        assert "[Script Info]" in content
        # Line-scope fade emits \fad(300,0) as an event-level override
        assert r"\fad(300,0)" in content

    def test_kinetic_word_scope_tags_emitted(self, tmp_path):
        """word_level=True + bounce emits per-word \\bord\\frz overrides (no \\fscy)."""
        from lattifai.caption.caption import Caption
        from lattifai.caption.supervision import AlignmentItem, Supervision

        sup = Supervision(
            text="Hello world",
            start=0.0,
            duration=1.0,
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    AlignmentItem(symbol="world", start=0.5, duration=0.5),
                ]
            },
        )
        caption = Caption(supervisions=[sup])
        src = tmp_path / "in.json"
        caption.write(str(src))

        out = tmp_path / "out.ass"
        convert(
            str(src),
            str(out),
            ass=ASSConfig(kinetic_style="bounce"),
            render=RenderConfig(word_level=True),
        )
        content = Path(out).read_text()
        # Phase 2 (revised): word-scope bounce is pure \bord+\blur impact.
        # \frz was dropped to avoid origin-displacement of off-center words,
        # and \fscx/\fscy is forbidden in word-scope to prevent line reflow.
        # Each word's block opens with \bord2\blur0 static reset to break
        # libass override inheritance from the previous word's animation.
        assert r"\fscx" not in content
        assert r"\fscy" not in content
        assert r"\frz" not in content
        assert r"\bord8\blur4" in content  # impact peak
        # Cumulative word offsets (word 2 starts at 500ms)
        assert r"\t(0,1,\bord8\blur4)" in content
        assert r"\t(500,501,\bord8\blur4)" in content


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
