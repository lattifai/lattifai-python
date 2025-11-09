"""Tests for configuration system."""

from pathlib import Path

import pytest

from lattifai.config import (
    AlignmentConfig,
    MediaConfig,
    SubtitleConfig,
    TranscriptionConfig,
    YouTubeWorkflowConfig,
)


class TestMediaConfig:
    """Test MediaConfig class."""

    def test_local_file_input(self, tmp_path):
        """Ensure local media paths are validated and formats inferred."""
        media_file = tmp_path / "sample.wav"
        media_file.write_bytes(b"")

        config = MediaConfig(input_path=str(media_file))

        assert config.input_path == str(media_file)
        assert config.media_format == "wav"
        assert config.is_input_remote() is False

    def test_url_input(self, tmp_path):
        """Ensure URL inputs skip filesystem checks and infer formats."""
        media_url = "https://cdn.example.com/audio/sample.mp3?token=abc"

        config = MediaConfig(input_path=media_url, output_dir=tmp_path)

        assert config.input_path == media_url
        assert config.media_format == "mp3"
        assert config.is_input_remote() is True

        output_path = config.prepare_output_path()
        assert output_path.parent == tmp_path
        assert output_path.name == "sample.mp3"


class TestAlignmentConfig:
    """Test AlignmentConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlignmentConfig()
        # API defaults
        assert config.api_key is None
        assert config.base_url == "https://api.lattifai.com/v1"
        assert config.timeout == 120.0
        assert config.max_retries == 2
        assert config.default_headers is None
        # Alignment defaults
        assert config.device == "cpu"
        assert config.model_name_or_path == "Lattifai/Lattice-1-Alpha"
        assert config.split_sentence is False
        assert config.word_level is False
        assert config.batch_size == 1

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loaded from environment."""
        monkeypatch.setenv("LATTIFAI_API_KEY", "env-key")
        config = AlignmentConfig()
        assert config.api_key == "env-key"

    def test_base_url_from_env(self, monkeypatch):
        """Test base URL loaded from environment."""
        monkeypatch.setenv("LATTIFAI_BASE_URL", "https://custom.api.com")
        config = AlignmentConfig(api_key="test-key")
        assert config.base_url == "https://custom.api.com"

    def test_invalid_timeout(self):
        """Test validation of timeout parameter."""
        with pytest.raises(ValueError, match="timeout must be greater than 0"):
            AlignmentConfig(api_key="test-key", timeout=0)

    def test_invalid_max_retries(self):
        """Test validation of max_retries parameter."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            AlignmentConfig(api_key="test-key", max_retries=-1)

    def test_custom_headers(self):
        """Test custom headers configuration."""
        headers = {"X-Custom": "value"}
        config = AlignmentConfig(api_key="test-key", default_headers=headers)
        assert config.default_headers == headers

    def test_custom_alignment_values(self):
        """Test custom configuration values."""
        config = AlignmentConfig(
            device="cuda",
            model_name_or_path="custom-model",
            split_sentence=True,
            word_level=True,
            batch_size=4,
        )
        assert config.device == "cuda"
        assert config.model_name_or_path == "custom-model"
        assert config.split_sentence is True
        assert config.word_level is True
        assert config.batch_size == 4

    def test_invalid_batch_size(self):
        """Test validation of batch_size parameter."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            AlignmentConfig(batch_size=0)

    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError, match="device must be one of"):
            AlignmentConfig(device="invalid")


class TestSubtitleConfig:
    """Test SubtitleConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SubtitleConfig()
        assert config.input_format == "auto"
        assert config.output_format == "srt"
        assert config.normalize_text is False
        assert config.output_dir == Path(".")
        assert config.include_speaker_in_text is True
        assert config.encoding == "utf-8"

    def test_custom_values(self, tmp_path):
        """Test custom configuration values."""
        output_dir = tmp_path / "output"
        config = SubtitleConfig(
            input_format="vtt",
            output_format="json",
            normalize_text=True,
            output_dir=output_dir,
            include_speaker_in_text=False,
            encoding="utf-16",
        )
        assert config.input_format == "vtt"
        assert config.output_format == "json"
        assert config.normalize_text is True
        assert config.output_dir == output_dir
        assert config.include_speaker_in_text is False
        assert config.encoding == "utf-16"
        # Should create output directory
        assert output_dir.exists()

    def test_invalid_input_format(self):
        """Test validation of input_format parameter."""
        with pytest.raises(ValueError, match="input_format must be one of"):
            SubtitleConfig(input_format="invalid")

    def test_invalid_output_format(self):
        """Test validation of output_format parameter."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            SubtitleConfig(output_format="invalid")


class TestTranscriptionConfig:
    """Test TranscriptionConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        assert config.gemini_api_key is None
        assert config.device == "cpu"
        assert config.media_format == "mp4"
        assert config.max_retries == 0
        assert config.force_overwrite is False
        assert config.verbose is False
        assert config.language is None
        assert config.model_name == "gemini-2.5-pro"

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loaded from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
        config = TranscriptionConfig()
        assert config.gemini_api_key == "gemini-env-key"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TranscriptionConfig(
            gemini_api_key="test-key",
            device="cuda",
            media_format="mp3",
            max_retries=3,
            force_overwrite=True,
            verbose=True,
            language="en",
            model_name="gemini-pro",
        )
        assert config.gemini_api_key == "test-key"
        assert config.device == "cuda"
        assert config.media_format == "mp3"
        assert config.max_retries == 3
        assert config.force_overwrite is True
        assert config.verbose is True
        assert config.language == "en"
        assert config.model_name == "gemini-pro"

    def test_invalid_max_retries(self):
        """Test validation of max_retries parameter."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            TranscriptionConfig(max_retries=-1)

    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError, match="device must be one of"):
            TranscriptionConfig(device="invalid")

    def test_invalid_media_format(self):
        """Test validation of media_format parameter."""
        with pytest.raises(ValueError, match="media_format must be one of"):
            TranscriptionConfig(media_format="invalid")


class TestYouTubeWorkflowConfig:
    """Test YouTubeWorkflowConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = YouTubeWorkflowConfig()
        assert isinstance(config.alignment, AlignmentConfig)
        assert isinstance(config.subtitle, SubtitleConfig)
        assert isinstance(config.transcription, TranscriptionConfig)
        assert config.output_dir == Path(".")

    def test_custom_values(self, tmp_path):
        """Test custom configuration values."""
        output_dir = tmp_path / "workflow_output"
        config = YouTubeWorkflowConfig(
            alignment=AlignmentConfig(api_key="test-key", device="cuda"),
            subtitle=SubtitleConfig(output_format="json"),
            transcription=TranscriptionConfig(media_format="mp3"),
            output_dir=output_dir,
        )
        assert config.alignment.api_key == "test-key"
        assert config.alignment.device == "cuda"
        assert config.subtitle.output_format == "json"
        assert config.transcription.media_format == "mp3"
        assert config.output_dir == output_dir
        # Should create output directory
        assert output_dir.exists()

    def test_subtitle_output_dir_sync(self, tmp_path):
        """Test that subtitle output_dir is synchronized with workflow output_dir."""
        output_dir = tmp_path / "sync_test"
        config = YouTubeWorkflowConfig(output_dir=output_dir)
        assert config.subtitle.output_dir == output_dir
