"""Unit tests for CaptionConfig and sub-configs."""

import pytest

from lattifai.config import CaptionConfig, CaptionInputConfig, CaptionOutputConfig


class TestCaptionInputConfigValidation:
    """Test CaptionInputConfig validation."""

    def test_invalid_format_raises_error(self):
        """Test that invalid input format raises ValueError."""
        with pytest.raises(ValueError, match="input format must be one of"):
            CaptionInputConfig(format="invalid")

    def test_valid_formats(self):
        """Test that all valid input formats are accepted."""
        valid_formats = ["auto", "srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "gemini"]
        for fmt in valid_formats:
            config = CaptionInputConfig(format=fmt)
            assert config.format == fmt


class TestCaptionInputConfigDualSource:
    """Test the external-transcription dual-input path used by
    ``alignment.strategy='transcription'`` diff alignment workflows."""

    def test_transcription_path_defaults_to_none(self):
        """Default config does not opt into the external-transcription path."""
        config = CaptionInputConfig()
        assert config.transcription_path is None
        assert config.transcription_format == "auto"

    def test_transcription_path_normalized_when_set(self, tmp_path):
        """Path is expanded + resolved at construction time (mirrors ``path``)."""
        test_file = tmp_path / "external_asr.vtt"
        test_file.touch()
        config = CaptionInputConfig(transcription_path=str(test_file))
        assert config.transcription_path == str(test_file.resolve())

    def test_transcription_path_accepts_nonexistent_path(self, tmp_path):
        """``__post_init__`` only normalizes; existence checks belong to the loader.

        This matches the behavior of ``path`` — both fields can be configured
        before the referenced file is created (e.g. config built then file
        downloaded later in the pipeline).
        """
        non_existing = tmp_path / "not_yet_downloaded.vtt"
        config = CaptionInputConfig(transcription_path=str(non_existing))
        assert config.transcription_path == str(non_existing.resolve())

    def test_invalid_transcription_format_raises_error(self):
        """Same format vocabulary as ``format`` — invalid values rejected."""
        with pytest.raises(ValueError, match="transcription_format must be one of"):
            CaptionInputConfig(transcription_format="invalid")

    def test_valid_transcription_formats(self):
        """All standard input formats are valid for transcription too."""
        valid_formats = ["auto", "srt", "vtt", "ass", "txt", "json"]
        for fmt in valid_formats:
            config = CaptionInputConfig(transcription_format=fmt)
            assert config.transcription_format == fmt

    def test_both_path_and_transcription_path_independent(self, tmp_path):
        """Setting one does not affect the other — they are independent slots."""
        primary = tmp_path / "official.md"
        primary.touch()
        external = tmp_path / "yt.vtt"
        external.touch()

        config = CaptionInputConfig(
            path=str(primary),
            format="auto",
            transcription_path=str(external),
            transcription_format="vtt",
        )
        assert config.path == str(primary.resolve())
        assert config.transcription_path == str(external.resolve())
        assert config.format == "auto"
        assert config.transcription_format == "vtt"


class TestCaptionOutputConfigValidation:
    """Test CaptionOutputConfig validation."""

    def test_invalid_format_raises_error(self):
        """Test that invalid output format raises ValueError."""
        with pytest.raises(ValueError, match="output format must be one of"):
            CaptionOutputConfig(format="invalid")

    def test_valid_formats(self):
        """Test that all valid output formats are accepted."""
        valid_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "textgrid", "json"]
        for fmt in valid_formats:
            config = CaptionOutputConfig(format=fmt)
            assert config.format == fmt


class TestCaptionConfigPaths:
    """Test CaptionConfig path handling via sub-configs."""

    def test_set_input_path_with_valid_file(self, tmp_path):
        """Test setting input path with existing file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = CaptionConfig()
        result = config.set_input_path(test_file)

        assert result.exists()
        assert config.input.path == str(result)

    def test_set_input_path_with_nonexistent_file_raises_error(self, tmp_path):
        """Test that setting input path to nonexistent file raises FileNotFoundError."""
        config = CaptionConfig()
        nonexistent = tmp_path / "nonexistent.srt"

        with pytest.raises(FileNotFoundError):
            config.set_input_path(nonexistent)

    def test_set_output_path_creates_parent_dir(self, tmp_path):
        """Test that set_output_path creates parent directory."""
        config = CaptionConfig()
        output_file = tmp_path / "subdir" / "output.srt"

        result = config.set_output_path(output_file)

        assert result.parent.exists()
        assert config.output.path == str(result)

    def test_path_expansion_in_post_init(self, tmp_path):
        """Test that paths are expanded in __post_init__."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = CaptionInputConfig(path=str(test_file))
        assert config.path == str(test_file)


class TestCaptionConfigMethods:
    """Test CaptionConfig utility methods."""

    def test_check_input_sanity_with_no_input_raises_error(self):
        """Test that check_input_sanity raises error when input path is None."""
        config = CaptionConfig()
        with pytest.raises(ValueError, match="input path is required"):
            config.check_input_sanity()

    def test_check_input_sanity_with_valid_file(self, tmp_path):
        """Test check_input_sanity with valid file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = CaptionConfig(input=CaptionInputConfig(path=str(test_file)))
        config.check_input_sanity()

    def test_is_input_path_existed_returns_false_for_none(self):
        """Test is_input_path_existed returns False when input path is None."""
        config = CaptionConfig()
        assert config.is_input_path_existed() is False

    def test_is_input_path_existed_returns_true_for_valid_file(self, tmp_path):
        """Test is_input_path_existed returns True for existing file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = CaptionConfig(input=CaptionInputConfig(path=str(test_file)))
        assert config.is_input_path_existed() is True


class TestCaptionConfigDefaults:
    """Test CaptionConfig default values via sub-configs."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = CaptionConfig()

        assert config.input.format == "auto"
        assert config.output.format == "srt"
        assert config.render.include_speaker_in_text is True
        assert config.input.normalize_text is True
        assert config.input.split_sentence is False
        assert config.input.split_threshold == 0.35
        assert config.render.word_level is None  # tri-state: None=per-format default
        assert config.input.encoding == "utf-8"

    def test_backward_compat_properties(self):
        """Test backward-compatible property accessors."""
        config = CaptionConfig()

        assert config.input_format == "auto"
        assert config.output_format == "srt"
        assert config.include_speaker_in_text is True
        assert config.normalize_text is True
        assert config.split_sentence is False
        assert config.word_level is None  # tri-state: None=per-format default
        assert config.encoding == "utf-8"

    def test_custom_values(self):
        """Test that custom values override defaults."""
        from lattifai.caption.config import RenderConfig

        config = CaptionConfig(
            input=CaptionInputConfig(format="vtt", normalize_text=True),
            output=CaptionOutputConfig(format="json"),
            render=RenderConfig(word_level=True),
        )

        assert config.input.format == "vtt"
        assert config.output.format == "json"
        assert config.input.normalize_text is True
        assert config.render.word_level is True

    def test_split_threshold_override(self):
        """Custom split_threshold should override the 0.35 default."""
        # Aggressive cutting for experimental use.
        c1 = CaptionInputConfig(split_sentence=True, split_threshold=0.10)
        assert c1.split_threshold == 0.10

        # More aggressive than default but still within recommended range.
        c2 = CaptionInputConfig(split_sentence=True, split_threshold=0.20)
        assert c2.split_threshold == 0.20


class TestCaptionConfigStructure:
    """Test CaptionConfig structure."""

    def test_sub_configs(self):
        """CaptionConfig should have core sub-config fields."""
        config = CaptionConfig()
        assert hasattr(config, "input")
        assert hasattr(config, "output")
        assert hasattr(config, "render")
        assert hasattr(config, "standardization")
        assert hasattr(config, "ass")
        assert hasattr(config, "lrc")
        assert hasattr(config, "ttml")
        assert hasattr(config, "fcpxml")
        assert hasattr(config, "premiere")

    def test_speaker_color_in_ass(self):
        """speaker_color should be in ASSConfig, not a top-level field."""
        from lattifai.caption.config import ASSConfig

        config = CaptionConfig(ass=ASSConfig(speaker_color="auto"))
        assert config.ass.speaker_color == "auto"
        assert config.speaker_color == "auto"  # backward compat property
