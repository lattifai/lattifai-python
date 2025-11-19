"""Unit tests for MediaConfig."""

import pytest

from lattifai.config import MediaConfig


class TestMediaConfigValidation:
    """Test MediaConfig input validation."""

    def test_invalid_local_path_raises_error(self):
        """Test that invalid local file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _ = MediaConfig(input_path="/nonexistent/file.wav")

    def test_valid_url_is_accepted(self):
        """Test that valid URL is accepted."""
        config = MediaConfig(input_path="https://example.com/video.mp4")
        assert config.is_input_remote() is True
        assert config.input_path == "https://example.com/video.mp4"

    def test_invalid_url_raises_error(self):
        """Test that string without scheme/netloc that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Input media file does not exist"):
            _ = MediaConfig(input_path="not-a-valid-url")

    def test_check_input_sanity_with_no_input_raises_error(self):
        """Test that check_input_sanity raises error when input_path is None."""
        config = MediaConfig()
        with pytest.raises(ValueError, match="input_path is required"):
            config.check_input_sanity()


class TestMediaConfigFormats:
    """Test MediaConfig format handling."""

    def test_default_formats_are_valid(self):
        """Test that default audio and video formats are valid."""
        config = MediaConfig()
        assert config.default_audio_format == "mp3"
        assert config.default_video_format == "mp4"

    def test_invalid_format_raises_error(self):
        """Test that invalid media format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported media format"):
            MediaConfig(media_format="invalid_format")

    def test_normalize_format_with_auto(self):
        """Test format normalization with 'auto' value."""
        config = MediaConfig(media_format="auto")
        normalized = config.normalize_format()
        assert normalized in ["mp3", "mp4"]  # Should default to one of these

    def test_is_audio_format(self):
        """Test audio format detection."""
        config = MediaConfig()
        assert config.is_audio_format("mp3") is True
        assert config.is_audio_format("wav") is True
        assert config.is_audio_format("mp4") is False

    def test_is_video_format(self):
        """Test video format detection."""
        config = MediaConfig()
        assert config.is_video_format("mp4") is True
        assert config.is_video_format("mkv") is True
        assert config.is_video_format("mp3") is False


class TestMediaConfigPaths:
    """Test MediaConfig path handling."""

    def test_set_output_dir_creates_directory(self, tmp_path):
        """Test that set_output_dir creates directory if it doesn't exist."""
        config = MediaConfig()
        new_dir = tmp_path / "test_output"
        result = config.set_output_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_set_output_path_with_extension(self, tmp_path):
        """Test setting output path with file extension."""
        config = MediaConfig()
        output_file = tmp_path / "output.mp3"
        config.set_output_path(output_file)
        assert config.output_path == str(output_file)
        assert config.output_format == "mp3"
        assert config.output_dir == tmp_path

    def test_set_output_path_without_extension_raises_error(self, tmp_path):
        """Test that setting output path without extension raises ValueError."""
        config = MediaConfig()
        output_file = tmp_path / "output"
        with pytest.raises(ValueError, match="must include a filename with an extension"):
            config.set_output_path(output_file)

    def test_prepare_output_path_creates_path(self, tmp_path):
        """Test prepare_output_path creates appropriate output path."""
        config = MediaConfig(output_dir=tmp_path, media_format="mp3")
        result = config.prepare_output_path(stem="test")
        assert result.parent == tmp_path
        assert result.name == "test.mp3"


class TestMediaConfigMethods:
    """Test MediaConfig utility methods."""

    def test_clone_creates_copy_with_updates(self):
        """Test that clone creates a copy with optional updates."""
        config = MediaConfig(media_format="mp3")
        cloned = config.clone(media_format="wav")
        assert cloned.media_format == "wav"
        assert config.media_format == "mp3"  # Original unchanged

    def test_is_input_remote_with_url(self):
        """Test is_input_remote returns True for URL."""
        config = MediaConfig(input_path="https://example.com/video.mp4")
        assert config.is_input_remote() is True

    def test_is_input_remote_with_local_file(self, tmp_path):
        """Test is_input_remote returns False for local file."""
        test_file = tmp_path / "test.wav"
        test_file.touch()
        config = MediaConfig(input_path=str(test_file))
        assert config.is_input_remote() is False
