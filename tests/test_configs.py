"""Tests for configuration system."""

import pytest

from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    MediaConfig,
    TranscriptionConfig,
)


class TestMediaConfig:
    """Test MediaConfig class."""

    def test_local_file_input(self, tmp_path):
        """Ensure local media paths are validated and formats inferred."""
        media_file = tmp_path / "sample.wav"
        media_file.write_bytes(b"")

        config = MediaConfig(input_path=str(media_file))

        assert config.input_path == str(media_file)
        assert config.is_input_remote() is False

    def test_url_input(self, tmp_path):
        """Ensure URL inputs skip filesystem checks and infer formats."""
        media_url = "https://cdn.example.com/audio/sample.mp3?token=abc"

        config = MediaConfig(input_path=media_url, output_dir=tmp_path)

        assert config.input_path == media_url
        assert config.is_input_remote() is True

        output_path = config.prepare_output_path()
        assert output_path.parent == tmp_path
        assert output_path.name == "sample.mp3"


class TestAlignmentConfig:
    """Test AlignmentConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlignmentConfig()
        # Alignment defaults - device is auto-selected based on hardware
        assert config.device in ["cpu", "cuda", "mps"]
        assert config.model_name == "LattifAI/Lattice-1"
        assert config.batch_size == 1

    def test_api_key_from_env(self, monkeypatch):
        """Test API key loaded from environment."""
        monkeypatch.setenv("LATTIFAI_API_KEY", "env-key")
        config = ClientConfig()
        assert config.api_key == "env-key"

    def test_invalid_timeout(self):
        """Test validation of timeout parameter."""
        with pytest.raises(ValueError, match="timeout must be greater than 0"):
            ClientConfig(api_key="test-key", timeout=0)

    def test_invalid_max_retries(self):
        """Test validation of max_retries parameter."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ClientConfig(api_key="test-key", max_retries=-1)

    def test_custom_headers(self):
        """Test custom headers configuration."""
        headers = {"X-Custom": "value"}
        config = ClientConfig(api_key="test-key", default_headers=headers)
        assert config.default_headers == headers

    def test_custom_alignment_values(self):
        """Test custom configuration values."""
        config = AlignmentConfig(
            device="cuda",
            model_name="custom-model",
            batch_size=4,
        )
        assert config.device == "cuda"
        assert config.model_name == "custom-model"
        assert config.batch_size == 4

    def test_invalid_batch_size(self):
        """Test validation of batch_size parameter."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            AlignmentConfig(batch_size=0)

    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError, match="device must be one of"):
            AlignmentConfig(device="invalid")

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:7"],
    )
    def test_valid_device_values(self, device):
        """Test all valid device values including cuda:N format."""
        config = AlignmentConfig(device=device)
        assert config.device == device

    def test_auto_device_selection(self):
        """Test that 'auto' device gets converted to actual device."""
        config = AlignmentConfig(device="auto")
        # auto should be converted to one of the actual devices
        assert config.device in ["cpu", "cuda", "mps"]

    @pytest.mark.parametrize(
        "invalid_device",
        ["gpu", "tpu", "CUDA", "Cuda:0", "GPU", "invalid"],
    )
    def test_invalid_device_values(self, invalid_device):
        """Test various invalid device values."""
        with pytest.raises(ValueError, match="device must be one of"):
            AlignmentConfig(device=invalid_device)


class TestCaptionConfig:
    """Test CaptionConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CaptionConfig()
        assert config.input_format == "auto"
        assert config.output_format == "srt"
        assert config.normalize_text is True
        assert config.output_path is None
        assert config.include_speaker_in_text is True
        assert config.encoding == "utf-8"

    def test_custom_values(self, tmp_path):
        """Test custom configuration values."""
        config = CaptionConfig(
            input_format="vtt",
            output_format="json",
            normalize_text=True,
            include_speaker_in_text=False,
            encoding="utf-16",
        )
        assert config.input_format == "vtt"
        assert config.output_format == "json"
        assert config.normalize_text is True
        assert config.include_speaker_in_text is False
        assert config.encoding == "utf-16"

    def test_invalid_input_format(self):
        """Test validation of input_format parameter."""
        with pytest.raises(ValueError, match="input_format must be one of"):
            CaptionConfig(input_format="invalid")

    def test_invalid_output_format(self):
        """Test validation of output_format parameter."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            CaptionConfig(output_format="invalid")


class TestTranscriptionConfig:
    """Test TranscriptionConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        # Device is auto-selected based on hardware
        assert config.device in ["cpu", "cuda", "mps"]
        assert config.max_retries == 0
        assert config.force_overwrite is False
        assert config.verbose is False
        assert config.language is None
        assert config.model_name == "nvidia/parakeet-tdt-0.6b-v3"

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
            max_retries=3,
            force_overwrite=True,
            verbose=True,
            language="en",
            model_name="gemini-2.5-pro",
        )
        assert config.gemini_api_key == "test-key"
        assert config.device == "cuda"
        assert config.max_retries == 3
        assert config.force_overwrite is True
        assert config.verbose is True
        assert config.language == "en"
        assert config.model_name == "gemini-2.5-pro"

    def test_invalid_max_retries(self):
        """Test validation of max_retries parameter."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            TranscriptionConfig(max_retries=-1)

    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError, match="device must be one of"):
            TranscriptionConfig(device="invalid")

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:7"],
    )
    def test_valid_device_values(self, device):
        """Test all valid device values including cuda:N format."""
        config = TranscriptionConfig(device=device)
        assert config.device == device

    def test_auto_device_selection(self):
        """Test that 'auto' device gets converted to actual device."""
        config = TranscriptionConfig(device="auto")
        # auto should be converted to one of the actual devices
        assert config.device in ["cpu", "cuda", "mps"]

    @pytest.mark.parametrize(
        "invalid_device",
        ["gpu", "tpu", "CUDA", "Cuda:0", "GPU", "invalid"],
    )
    def test_invalid_device_values(self, invalid_device):
        """Test various invalid device values."""
        with pytest.raises(ValueError, match="device must be one of"):
            TranscriptionConfig(device=invalid_device)


class TestDiarizationConfig:
    """Test DiarizationConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DiarizationConfig()
        assert config.enabled is False
        # Device is auto-selected based on hardware
        assert config.device in ["cpu", "cuda", "mps"]
        assert config.num_speakers is None
        assert config.min_speakers is None
        assert config.max_speakers is None
        assert config.model_name == "pyannote/speaker-diarization-community-1"
        assert config.verbose is False
        assert config.debug is False
        assert config.min_claim_duration == 2.0
        assert config.min_claim_count == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DiarizationConfig(
            enabled=True,
            device="cuda",
            num_speakers=2,
            model_name="custom-diarization-model",
            verbose=True,
            debug=False,
        )
        assert config.enabled is True
        assert config.device == "cuda"
        assert config.num_speakers == 2
        assert config.model_name == "custom-diarization-model"
        assert config.verbose is True
        assert config.debug is False

    def test_speaker_range(self):
        """Test min_speakers and max_speakers configuration."""
        config = DiarizationConfig(
            min_speakers=2,
            max_speakers=5,
        )
        assert config.min_speakers == 2
        assert config.max_speakers == 5

    def test_invalid_num_speakers(self):
        """Test validation of num_speakers parameter."""
        with pytest.raises(ValueError, match="num_speakers must be at least 1"):
            DiarizationConfig(num_speakers=0)

    def test_invalid_min_speakers(self):
        """Test validation of min_speakers parameter."""
        with pytest.raises(ValueError, match="min_speakers must be at least 1"):
            DiarizationConfig(min_speakers=0)

    def test_invalid_max_speakers(self):
        """Test validation of max_speakers parameter."""
        with pytest.raises(ValueError, match="max_speakers must be at least 1"):
            DiarizationConfig(max_speakers=0)

    def test_invalid_speaker_range(self):
        """Test validation of speaker range."""
        with pytest.raises(ValueError, match="min_speakers cannot be greater than max_speakers"):
            DiarizationConfig(min_speakers=5, max_speakers=2)

    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError, match="device must be one of"):
            DiarizationConfig(device="invalid")

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda", "mps", "cuda:0", "cuda:1", "cuda:2", "cuda:7"],
    )
    def test_valid_device_values(self, device):
        """Test all valid device values including cuda:N format."""
        config = DiarizationConfig(device=device)
        assert config.device == device

    def test_auto_device_selection(self):
        """Test that 'auto' device gets converted to actual device."""
        config = DiarizationConfig(device="auto")
        # auto should be converted to one of the actual devices
        assert config.device in ["cpu", "cuda", "mps"]

    @pytest.mark.parametrize(
        "invalid_device",
        ["gpu", "tpu", "CUDA", "Cuda:0", "GPU", "invalid"],
    )
    def test_invalid_device_values(self, invalid_device):
        """Test various invalid device values."""
        with pytest.raises(ValueError, match="device must be one of"):
            DiarizationConfig(device=invalid_device)

    def test_custom_claim_thresholds(self):
        """Test custom min_claim_duration and min_claim_count."""
        config = DiarizationConfig(min_claim_duration=5.0, min_claim_count=3)
        assert config.min_claim_duration == 5.0
        assert config.min_claim_count == 3

    def test_invalid_min_claim_duration(self):
        """Test validation of min_claim_duration parameter."""
        with pytest.raises(ValueError, match="min_claim_duration must be non-negative"):
            DiarizationConfig(min_claim_duration=-1.0)

    def test_invalid_min_claim_count(self):
        """Test validation of min_claim_count parameter."""
        with pytest.raises(ValueError, match="min_claim_count must be at least 1"):
            DiarizationConfig(min_claim_count=0)
