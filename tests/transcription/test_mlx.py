"""Tests for MLX transcription support."""

import pytest

from lattifai.config import TranscriptionConfig


class TestMLXConfig:
    """Test TranscriptionConfig MLX-related fields."""

    def test_mlx_quantization_default_is_8bit(self):
        """mlx_quantization defaults to '8bit'."""
        config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3")
        assert config.mlx_quantization == "8bit"

    def test_mlx_quantization_accepts_4bit(self):
        """mlx_quantization accepts '4bit'."""
        config = TranscriptionConfig(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            mlx_quantization="4bit",
        )
        assert config.mlx_quantization == "4bit"

    def test_mlx_community_model_skips_validation(self):
        """mlx-community/* model names should not raise ValueError."""
        config = TranscriptionConfig(model_name="mlx-community/Qwen3-ASR-0.6B-8bit")
        assert config.model_name == "mlx-community/Qwen3-ASR-0.6B-8bit"

    def test_mlx_community_arbitrary_model_skips_validation(self):
        """Any mlx-community/* model name should pass validation."""
        config = TranscriptionConfig(model_name="mlx-community/some-future-model-4bit")
        assert config.model_name == "mlx-community/some-future-model-4bit"
