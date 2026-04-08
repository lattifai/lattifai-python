"""Test transcriber factory method."""

import pytest

from lattifai.config import TranscriptionConfig
from lattifai.transcription import create_transcriber
from lattifai.transcription.gemini import GeminiTranscriber
from lattifai.transcription.lattifai import LattifAITranscriber
from lattifai.transcription.mlx import MLXTranscriber


def test_create_gemini_transcriber_explicit():
    """Test creating Gemini transcriber with explicit model name."""
    config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key="test_key")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, GeminiTranscriber)
    assert transcriber.config.model_name == "gemini-2.5-pro"


def test_create_gemini_transcriber_preview():
    """Test creating Gemini transcriber with preview model (auto-upgraded)."""
    config = TranscriptionConfig(model_name="gemini-3-pro-preview", gemini_api_key="test_key")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, GeminiTranscriber)
    # gemini-3-pro-preview is deprecated and auto-switches to gemini-3.1-pro-preview
    assert transcriber.config.model_name == "gemini-3.1-pro-preview"


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


def test_create_qwen_transcriber():
    """Test creating LattifAI transcriber with Qwen3-ASR-1.7B (non-MLX path)."""
    config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-1.7B", device="cpu")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, LattifAITranscriber)
    assert transcriber.config.model_name == "Qwen/Qwen3-ASR-1.7B"


def test_create_qwen_transcriber_0_6b():
    """Test creating LattifAI transcriber with Qwen3-ASR-0.6B (non-MLX path)."""
    config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B", device="cpu")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, LattifAITranscriber)
    assert transcriber.config.model_name == "Qwen/Qwen3-ASR-0.6B"


def test_create_mlx_transcriber_qwen_on_mps():
    """Qwen3-ASR on mps routes to MLXTranscriber."""
    config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B", device="mps")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, MLXTranscriber)
    assert transcriber._backend == "mlx-audio"


def test_create_mlx_transcriber_gemma_on_mps():
    """Gemma-4 on mps routes to MLXTranscriber."""
    config = TranscriptionConfig(model_name="google/gemma-4-E2B-it", device="mps")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, MLXTranscriber)
    assert transcriber._backend == "mlx-vlm"


def test_create_mlx_transcriber_voxtral_on_mps():
    """Voxtral on mps routes to MLXTranscriber."""
    config = TranscriptionConfig(model_name="mistralai/Voxtral-Mini-4B-2602", device="mps")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, MLXTranscriber)
    assert transcriber._backend == "mlx-audio"


def test_create_mlx_transcriber_mlx_community():
    """mlx-community/* always routes to MLXTranscriber."""
    config = TranscriptionConfig(model_name="mlx-community/Qwen3-ASR-0.6B-8bit", device="cpu")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, MLXTranscriber)


def test_create_lattifai_transcriber_qwen_on_cpu():
    """Qwen3-ASR on cpu still routes to LattifAITranscriber (not MLX)."""
    config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B", device="cpu")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, LattifAITranscriber)


def test_create_lattifai_transcriber_parakeet_on_mps():
    """Parakeet on mps still routes to LattifAITranscriber (not MLX)."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", device="mps")
    transcriber = create_transcriber(config)
    assert isinstance(transcriber, LattifAITranscriber)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
