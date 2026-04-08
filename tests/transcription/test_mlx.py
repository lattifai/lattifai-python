"""Tests for MLX transcription support."""

import pytest

from lattifai.config import TranscriptionConfig
from lattifai.transcription.mlx import _MLX_MODEL_MAP, MLXTranscriber, _is_mlx_model


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


class TestMLXModelMapping:
    """Test model name mapping and routing logic."""

    def test_is_mlx_model_mlx_community_always(self):
        assert _is_mlx_model("mlx-community/Qwen3-ASR-0.6B-8bit", "mps") is True
        assert _is_mlx_model("mlx-community/Qwen3-ASR-0.6B-8bit", "cuda") is True
        assert _is_mlx_model("mlx-community/Qwen3-ASR-0.6B-8bit", "cpu") is True

    def test_is_mlx_model_mapped_on_mps(self):
        assert _is_mlx_model("Qwen/Qwen3-ASR-0.6B", "mps") is True
        assert _is_mlx_model("google/gemma-4-E2B-it", "mps") is True
        assert _is_mlx_model("mistralai/Voxtral-Mini-4B-2602", "mps") is True

    def test_is_mlx_model_mapped_not_on_cuda(self):
        assert _is_mlx_model("Qwen/Qwen3-ASR-0.6B", "cuda") is False
        assert _is_mlx_model("google/gemma-4-E2B-it", "cuda") is False

    def test_is_mlx_model_non_mlx(self):
        assert _is_mlx_model("nvidia/parakeet-tdt-0.6b-v3", "mps") is False
        assert _is_mlx_model("gemini-2.5-pro", "mps") is False

    def test_resolve_backend_qwen_is_mlx_audio(self):
        config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B")
        t = MLXTranscriber(config)
        assert t._backend == "mlx-audio"

    def test_resolve_backend_voxtral_is_mlx_audio(self):
        config = TranscriptionConfig(model_name="mistralai/Voxtral-Mini-4B-2602")
        t = MLXTranscriber(config)
        assert t._backend == "mlx-audio"

    def test_resolve_backend_gemma_is_mlx_vlm(self):
        config = TranscriptionConfig(model_name="google/gemma-4-E2B-it")
        t = MLXTranscriber(config)
        assert t._backend == "mlx-vlm"

    def test_resolve_backend_mlx_community_gemma_is_mlx_vlm(self):
        config = TranscriptionConfig(model_name="mlx-community/gemma-4-e4b-it-8bit")
        t = MLXTranscriber(config)
        assert t._backend == "mlx-vlm"

    def test_resolve_mlx_model_id_auto_map_8bit(self):
        config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B", mlx_quantization="8bit")
        t = MLXTranscriber(config)
        assert t._mlx_model_id == "mlx-community/Qwen3-ASR-0.6B-8bit"

    def test_resolve_mlx_model_id_auto_map_4bit(self):
        config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-1.7B", mlx_quantization="4bit")
        t = MLXTranscriber(config)
        assert t._mlx_model_id == "mlx-community/Qwen3-ASR-1.7B-4bit"

    def test_resolve_mlx_model_id_gemma_auto_map(self):
        config = TranscriptionConfig(model_name="google/gemma-4-E2B-it", mlx_quantization="8bit")
        t = MLXTranscriber(config)
        assert t._mlx_model_id == "mlx-community/gemma-4-E2B-it-8bit"

    def test_resolve_mlx_model_id_passthrough(self):
        config = TranscriptionConfig(model_name="mlx-community/Qwen3-ASR-0.6B-8bit")
        t = MLXTranscriber(config)
        assert t._mlx_model_id == "mlx-community/Qwen3-ASR-0.6B-8bit"

    def test_resolve_mlx_model_id_voxtral_auto_map(self):
        config = TranscriptionConfig(model_name="mistralai/Voxtral-Mini-4B-2602", mlx_quantization="8bit")
        t = MLXTranscriber(config)
        assert t._mlx_model_id == "mlx-community/Voxtral-Mini-4B-2602-8bit"
