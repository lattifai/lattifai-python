"""Unit tests for SubtitleConfig."""

import pytest

from lattifai.config import SubtitleConfig


class TestSubtitleConfigValidation:
    """Test SubtitleConfig validation."""

    def test_invalid_input_format_raises_error(self):
        """Test that invalid input format raises ValueError."""
        with pytest.raises(ValueError, match="input_format must be one of"):
            SubtitleConfig(input_format="invalid")

    def test_invalid_output_format_raises_error(self):
        """Test that invalid output format raises ValueError."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            SubtitleConfig(output_format="invalid")

    def test_valid_input_formats(self):
        """Test that all valid input formats are accepted."""
        valid_formats = ["auto", "srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "gemini"]
        for fmt in valid_formats:
            config = SubtitleConfig(input_format=fmt)
            assert config.input_format == fmt

    def test_valid_output_formats(self):
        """Test that all valid output formats are accepted."""
        valid_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "TextGrid", "json"]
        for fmt in valid_formats:
            config = SubtitleConfig(output_format=fmt)
            assert config.output_format == fmt


class TestSubtitleConfigPaths:
    """Test SubtitleConfig path handling."""

    def test_set_input_path_with_valid_file(self, tmp_path):
        """Test setting input path with existing file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = SubtitleConfig()
        result = config.set_input_path(test_file)

        assert result.exists()
        assert config.input_path == str(result)

    def test_set_input_path_with_nonexistent_file_raises_error(self, tmp_path):
        """Test that setting input path to nonexistent file raises FileNotFoundError."""
        config = SubtitleConfig()
        nonexistent = tmp_path / "nonexistent.srt"

        with pytest.raises(FileNotFoundError):
            config.set_input_path(nonexistent)

    def test_set_output_path_creates_parent_dir(self, tmp_path):
        """Test that set_output_path creates parent directory."""
        config = SubtitleConfig()
        output_file = tmp_path / "subdir" / "output.srt"

        result = config.set_output_path(output_file)

        assert result.parent.exists()
        assert config.output_path == str(result)

    def test_path_expansion_in_post_init(self, tmp_path):
        """Test that paths are expanded in __post_init__."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        # Use relative path with ~
        config = SubtitleConfig(input_path=str(test_file))
        assert config.input_path == str(test_file)


class TestSubtitleConfigMethods:
    """Test SubtitleConfig utility methods."""

    def test_check_input_sanity_with_no_input_raises_error(self):
        """Test that check_input_sanity raises error when input_path is None."""
        config = SubtitleConfig()
        with pytest.raises(ValueError, match="input_path is required"):
            config.check_input_sanity()

    def test_check_input_sanity_with_valid_file(self, tmp_path):
        """Test check_input_sanity with valid file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = SubtitleConfig(input_path=str(test_file))
        # Should not raise any error
        config.check_input_sanity()

    def test_is_input_path_existed_returns_false_for_none(self):
        """Test is_input_path_existed returns False when input_path is None."""
        config = SubtitleConfig()
        assert config.is_input_path_existed() is False

    def test_is_input_path_existed_returns_true_for_valid_file(self, tmp_path):
        """Test is_input_path_existed returns True for existing file."""
        test_file = tmp_path / "test.srt"
        test_file.touch()

        config = SubtitleConfig(input_path=str(test_file))
        assert config.is_input_path_existed() is True


class TestSubtitleConfigDefaults:
    """Test SubtitleConfig default values."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = SubtitleConfig()

        assert config.input_format == "auto"
        assert config.output_format == "srt"
        assert config.include_speaker_in_text is True
        assert config.normalize_text is False
        assert config.split_sentence is False
        assert config.word_level is False
        assert config.encoding == "utf-8"
        assert config.use_transcription is False

    def test_custom_values_override_defaults(self):
        """Test that custom values override defaults."""
        config = SubtitleConfig(
            input_format="vtt",
            output_format="json",
            normalize_text=True,
            word_level=True,
        )

        assert config.input_format == "vtt"
        assert config.output_format == "json"
        assert config.normalize_text is True
        assert config.word_level is True
