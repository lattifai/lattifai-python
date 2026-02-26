"""Test transcriber factory method."""

import pytest

from lattifai.config import TranscriptionConfig
from lattifai.transcription import create_transcriber
from lattifai.transcription.gemini import GeminiTranscriber
from lattifai.transcription.lattifai import LattifAITranscriber


def test_create_gemini_transcriber_explicit():
    """Test creating Gemini transcriber with explicit model name."""
    config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key="test_key")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, GeminiTranscriber)
    assert transcriber.config.model_name == "gemini-2.5-pro"


def test_create_gemini_transcriber_preview():
    """Test creating Gemini transcriber with preview model."""
    config = TranscriptionConfig(model_name="gemini-3-pro-preview", gemini_api_key="test_key")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, GeminiTranscriber)
    assert transcriber.config.model_name == "gemini-3-pro-preview"


def test_create_lattifai_transcriber_nvidia():
    """Test creating LattifAI transcriber with NVIDIA model."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, LattifAITranscriber)
    assert transcriber.config.model_name == "nvidia/parakeet-tdt-0.6b-v3"


def test_create_lattifai_transcriber_iic():
    """Test creating LattifAI transcriber with IIC model."""
    config = TranscriptionConfig(model_name="iic/SenseVoiceSmall")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, LattifAITranscriber)


def test_create_transcriber_unsupported_model():
    """Test that ValueError is raised for unsupported model without API key."""

    with pytest.raises(ValueError) as exc_info:
        config = TranscriptionConfig(model_name="unsupported-model")
        del config

    assert "unsupported-model" in str(exc_info.value)


def test_create_transcriber_preserves_config():
    """Test that created transcriber has correct configuration."""
    config = TranscriptionConfig(
        model_name="gemini-2.5-pro",
        gemini_api_key="test_key",
        device="cpu",
        language="en",
        verbose=True,
    )
    transcriber = create_transcriber(config)

    assert transcriber.config.model_name == "gemini-2.5-pro"
    assert transcriber.config.gemini_api_key == "test_key"
    assert transcriber.config.device == "cpu"
    assert transcriber.config.language == "en"
    assert transcriber.config.verbose is True


def test_create_transcriber_supports_url_flag():
    """Test that created transcribers have correct supports_url flag."""
    # Gemini supports URL
    gemini_config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key="test_key")
    gemini = create_transcriber(gemini_config)
    assert gemini.supports_url is True

    # LattifAI does not support URL
    lattifai_config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3")
    lattifai = create_transcriber(lattifai_config)
    assert lattifai.supports_url is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
