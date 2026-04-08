"""Comprehensive tests for google/gemma-4-E2B and google/gemma-4-E4B support.

Tests cover:
- Config validation: models accepted in SUPPORTED_TRANSCRIPTION_MODELS
- Transcription factory: correct routing to LattifAITranscriber
- LattifAITranscriber: model loading and inference (mocked — no GPU / download)
- LLM factory: "transformers" provider routing
- TransformersClient: creation, provider_name, multimodal detection
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.config.transcription import SUPPORTED_TRANSCRIPTION_MODELS


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------
class TestGemma4Config:
    """Test that gemma-4 models are properly registered in config."""

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_model_in_supported_list(self, model_name):
        """Gemma-4 models must be in SUPPORTED_TRANSCRIPTION_MODELS."""
        assert model_name in SUPPORTED_TRANSCRIPTION_MODELS.__args__

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_config_creation(self, model_name):
        """TranscriptionConfig should accept gemma-4 model names without api_base_url."""
        config = TranscriptionConfig(model_name=model_name)
        assert config.model_name == model_name

    def test_unsupported_model_rejected(self):
        """Non-existent model without api_base_url should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model_name"):
            TranscriptionConfig(model_name="google/gemma-4-FAKE")

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_config_with_vllm(self, model_name):
        """Gemma-4 should also work with api_base_url (vLLM/SGLang mode)."""
        config = TranscriptionConfig(model_name=model_name, api_base_url="http://localhost:8000/v1")
        assert config.api_base_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# 2. Transcription factory routing
# ---------------------------------------------------------------------------
class TestGemma4TranscriberFactory:
    """Test that create_transcriber routes gemma-4 models correctly."""

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_routes_to_mlx_or_lattifai_transcriber(self, model_name):
        """Without api_base_url, gemma-4 routes based on device and MLX model map."""
        from lattifai.transcription import create_transcriber
        from lattifai.transcription.lattifai import LattifAITranscriber
        from lattifai.transcription.mlx import MLXTranscriber, _is_mlx_model

        # On mps: -it variants in _MLX_MODEL_MAP → MLXTranscriber, others → LattifAI
        config_mps = TranscriptionConfig(model_name=model_name, device="mps", lattice_model_path="disabled")
        config_mps.client_wrapper = MagicMock()
        config_mps.client_wrapper.check_permission = MagicMock()
        transcriber_mps = create_transcriber(config_mps)
        if _is_mlx_model(model_name, "mps"):
            assert isinstance(transcriber_mps, MLXTranscriber)
        else:
            assert isinstance(transcriber_mps, LattifAITranscriber)

        # On cpu → always LattifAITranscriber
        config_cpu = TranscriptionConfig(model_name=model_name, device="cpu", lattice_model_path="disabled")
        config_cpu.client_wrapper = MagicMock()
        config_cpu.client_wrapper.check_permission = MagicMock()
        transcriber_cpu = create_transcriber(config_cpu)
        assert isinstance(transcriber_cpu, LattifAITranscriber)
        assert transcriber_cpu.name == model_name

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_routes_to_vllm_transcriber(self, model_name):
        """With api_base_url, gemma-4 should route to VLLMTranscriber."""
        from lattifai.transcription import create_transcriber
        from lattifai.transcription.vllm import VLLMTranscriber

        config = TranscriptionConfig(
            model_name=model_name,
            api_base_url="http://localhost:8000/v1",
            lattice_model_path="disabled",
        )
        config.client_wrapper = MagicMock()
        config.client_wrapper.check_permission = MagicMock()

        transcriber = create_transcriber(config)
        assert isinstance(transcriber, VLLMTranscriber)

    def test_vllm_gemma_vad_chunk_30s(self):
        """VLLMTranscriber should auto-set 30s VAD chunk for gemma models."""
        from lattifai.transcription.vllm import VLLMTranscriber

        config = TranscriptionConfig(
            model_name="google/gemma-4-E4B",
            api_base_url="http://localhost:8000/v1",
        )
        transcriber = VLLMTranscriber(transcription_config=config)
        assert transcriber._get_vad_chunk_size() == 30.0

    def test_vllm_auto_switches_to_chat_mode(self):
        """Gemma-4 is not a dedicated ASR model; should auto-switch to chat API mode."""
        from lattifai.transcription.vllm import VLLMTranscriber

        config = TranscriptionConfig(
            model_name="google/gemma-4-E2B",
            api_base_url="http://localhost:8000/v1",
            api_mode="transcriptions",  # default
        )
        VLLMTranscriber(transcription_config=config)  # __init__ triggers auto-switch
        # Should have been auto-switched to "chat" for general-purpose model
        assert config.api_mode == "chat"


# ---------------------------------------------------------------------------
# 2b. Shared audio limits & segment splitting
# ---------------------------------------------------------------------------
class TestSharedAudioLimits:
    """Test _MAX_AUDIO_SECONDS, _get_max_audio_seconds, and _split_long_segments."""

    def test_gemma_max_audio_30s(self):
        """Gemma models should report 30s hard limit."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        config = TranscriptionConfig(model_name="google/gemma-4-E2B-it", device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)
        assert transcriber._get_max_audio_seconds() == 30.0

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("nvidia/parakeet-tdt-0.6b-v3", 1440.0),  # 24 min (model card)
            ("iic/SenseVoiceSmall", 30.0),  # 30s encoder
            ("Qwen/Qwen3-ASR-0.6B", 1200.0),  # 20 min (tech report)
            ("Qwen/Qwen3-ASR-1.7B", 1200.0),  # 20 min (tech report)
        ],
    )
    def test_model_audio_limits(self, model_name, expected):
        """Each model should have its verified audio limit."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        config = TranscriptionConfig(model_name=model_name, device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)
        assert transcriber._get_max_audio_seconds() == expected

    def test_split_long_segments_no_split_needed(self):
        """Segments within limit should pass through unchanged."""
        from lattifai.transcription.base import BaseTranscriber

        segments = [(0.0, 25.0), (30.0, 55.0)]
        result = BaseTranscriber._split_long_segments(segments, 30.0)
        assert result == segments

    def test_split_long_segments_splits_oversized(self):
        """A 90s segment should be split into 3 x 30s chunks."""
        from lattifai.transcription.base import BaseTranscriber

        segments = [(0.0, 90.0)]
        result = BaseTranscriber._split_long_segments(segments, 30.0)
        assert result == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

    def test_split_long_segments_remainder(self):
        """A 50s segment should be split into 30s + 20s."""
        from lattifai.transcription.base import BaseTranscriber

        segments = [(10.0, 60.0)]
        result = BaseTranscriber._split_long_segments(segments, 30.0)
        assert result == [(10.0, 40.0), (40.0, 60.0)]

    def test_split_mixed_segments(self):
        """Mix of short and long segments."""
        from lattifai.transcription.base import BaseTranscriber

        segments = [(0.0, 20.0), (25.0, 80.0), (85.0, 100.0)]
        result = BaseTranscriber._split_long_segments(segments, 30.0)
        assert result == [(0.0, 20.0), (25.0, 55.0), (55.0, 80.0), (85.0, 100.0)]

    def test_vllm_uses_shared_limit(self):
        """VLLMTranscriber._get_vad_chunk_size should use shared _MAX_AUDIO_SECONDS."""
        from lattifai.transcription.vllm import VLLMTranscriber

        config = TranscriptionConfig(
            model_name="google/gemma-4-E4B-it",
            api_base_url="http://localhost:8000/v1",
        )
        transcriber = VLLMTranscriber(transcription_config=config)
        assert transcriber._get_vad_chunk_size() == 30.0
        # Should come from the shared _MAX_AUDIO_SECONDS, not hardcoded
        assert transcriber._get_max_audio_seconds() == 30.0


# ---------------------------------------------------------------------------
# 3. LattifAITranscriber model loading (mocked)
# ---------------------------------------------------------------------------
class TestGemma4ModelLoading:
    """Test gemma-4 model loading logic in LattifAITranscriber."""

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_gemma4_branch_not_unsupported(self, model_name):
        """gemma-4 should NOT hit the 'Unsupported model_name' else branch."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        config = TranscriptionConfig(model_name=model_name, device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)

        # Verify the model_name matches the gemma-4 elif condition
        assert transcriber.config.model_name.startswith("google/gemma-4-")

    def test_unsupported_model_raises(self):
        """Non-gemma-4 unknown model should raise ValueError."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        config = TranscriptionConfig(model_name="google/gemma-4-E2B", device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)
        # Force a different model_name to hit the else branch
        transcriber.config.model_name = "unknown/fake-model"

        with pytest.raises(ValueError, match="Unsupported model_name"):
            transcriber._load_asr_model()

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_injected_model_processor_tuple(self, model_name):
        """Injecting (model, processor) tuple should work for _transcribe_impl."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        config = TranscriptionConfig(model_name=model_name, device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        import torch

        mock_model.generate.return_value = torch.tensor([[0] * 10 + [1, 2, 3]])

        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda s, k: torch.tensor([[0] * 10]) if k == "input_ids" else torch.ones(1, 10)
        mock_processor.apply_chat_template.return_value = mock_inputs
        mock_processor.decode.return_value = "Test output"
        mock_processor.parse_response.return_value = "Test output"

        # Inject directly (bypass _load_asr_model)
        transcriber._asr_model = (mock_model, mock_processor)

        audio = np.random.randn(16000).astype(np.float32)
        hypotheses, led = transcriber._transcribe_impl(audio, progress_bar=False)

        assert len(hypotheses) == 1
        assert hypotheses[0].text == "Test output"


# ---------------------------------------------------------------------------
# 4. LattifAITranscriber inference (mocked)
# ---------------------------------------------------------------------------
class TestGemma4Transcription:
    """Test gemma-4 transcription inference path (mocked)."""

    def _make_transcriber(self, model_name="google/gemma-4-E2B"):
        """Create a LattifAITranscriber with mocked gemma-4 model."""
        from lattifai.transcription.lattifai import LattifAITranscriber

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = MagicMock()

        mock_processor = MagicMock()
        # apply_chat_template returns a BatchEncoding-like object with .to()
        import torch

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, k: torch.tensor([[0] * 10]) if k == "input_ids" else torch.ones(1, 10)
        mock_inputs.to.return_value = mock_inputs
        mock_processor.apply_chat_template.return_value = mock_inputs

        # decode + parse_response returns transcribed text
        mock_processor.decode.return_value = "She had your dark suit in greasy wash water all year."
        mock_processor.parse_response.return_value = "She had your dark suit in greasy wash water all year."

        # generate returns token ids
        mock_model.generate.return_value = torch.tensor([[0] * 10 + [1, 2, 3, 4, 5]])

        config = TranscriptionConfig(model_name=model_name, device="cpu")
        transcriber = LattifAITranscriber(transcription_config=config)
        transcriber._asr_model = (mock_model, mock_processor)
        return transcriber, mock_model, mock_processor

    def test_transcribe_single_numpy(self):
        """Test transcription of a single numpy array."""
        transcriber, mock_model, mock_processor = self._make_transcriber()

        audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
        hypotheses, led = transcriber._transcribe_impl(audio, progress_bar=False)

        assert len(hypotheses) == 1
        assert isinstance(hypotheses[0], Supervision)
        assert hypotheses[0].text == "She had your dark suit in greasy wash water all year."
        assert hypotheses[0].duration == pytest.approx(1.0, abs=0.01)

        # Verify processor.apply_chat_template was called with audio content
        mock_processor.apply_chat_template.assert_called_once()
        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "user"
        content_types = [c["type"] for c in messages[0]["content"]]
        assert "audio" in content_types
        assert "text" in content_types

    def test_transcribe_batch_numpy(self):
        """Test transcription of a batch of numpy arrays."""
        transcriber, _, _ = self._make_transcriber()

        audio_list = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
        hypotheses, led = transcriber._transcribe_impl(audio_list, progress_bar=False)

        assert len(hypotheses) == 3
        for hyp in hypotheses:
            assert isinstance(hyp, Supervision)
            assert hyp.text is not None
            assert hyp.duration > 0

    def test_transcribe_with_language(self):
        """Test that language parameter is passed in the prompt."""
        transcriber, _, mock_processor = self._make_transcriber()

        audio = np.random.randn(16000).astype(np.float32)
        transcriber._transcribe_impl(audio, language="zh", progress_bar=False)

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        text_content = [c for c in messages[0]["content"] if c["type"] == "text"][0]
        # Official prompt uses full language name via get_language_name("zh") → "Chinese"
        assert "Chinese" in text_content["text"] or "zh" in text_content["text"]

    def test_transcribe_with_custom_prompt(self):
        """Test that custom prompt overrides the default."""
        transcriber, _, mock_processor = self._make_transcriber()
        transcriber.config.prompt = "Transcribe with punctuation."

        audio = np.random.randn(16000).astype(np.float32)
        transcriber._transcribe_impl(audio, progress_bar=False)

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        text_content = [c for c in messages[0]["content"] if c["type"] == "text"][0]
        assert "Transcribe with punctuation." in text_content["text"]

    def test_transcribe_2d_audio_squeezed(self):
        """Test that 2D audio (1, samples) is squeezed to 1D."""
        transcriber, _, mock_processor = self._make_transcriber()

        audio_2d = np.random.randn(1, 16000).astype(np.float32)
        transcriber._transcribe_impl(audio_2d, progress_bar=False)

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0]
        audio_content = [c for c in messages[0]["content"] if c["type"] == "audio"][0]
        assert audio_content["audio"].ndim == 1

    @pytest.mark.parametrize(
        "model_name", ["google/gemma-4-E2B", "google/gemma-4-E2B-it", "google/gemma-4-E4B", "google/gemma-4-E4B-it"]
    )
    def test_both_models(self, model_name):
        """Test that both E2B and E4B work through the same path."""
        transcriber, _, _ = self._make_transcriber(model_name=model_name)

        audio = np.random.randn(16000).astype(np.float32)
        hypotheses, _ = transcriber._transcribe_impl(audio, progress_bar=False)

        assert len(hypotheses) == 1
        assert hypotheses[0].text is not None


# ---------------------------------------------------------------------------
# 5. LLM factory: "transformers" provider
# ---------------------------------------------------------------------------
class TestTransformersLLMFactory:
    """Test LLM factory routing for transformers provider."""

    @pytest.mark.parametrize("provider", ["transformers", "huggingface", "hf"])
    def test_factory_creates_transformers_client(self, provider):
        """All provider aliases should create TransformersClient."""
        from lattifai.llm import create_client
        from lattifai.llm.transformers import TransformersClient

        client = create_client(provider, model="google/gemma-4-E2B")
        assert isinstance(client, TransformersClient)
        assert client.provider_name == "transformers"

    def test_factory_rejects_unknown_provider(self):
        """Unknown provider should raise ValueError."""
        from lattifai.llm import create_client

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_client("unknown_provider", model="some-model")


# ---------------------------------------------------------------------------
# 6. TransformersClient
# ---------------------------------------------------------------------------
class TestTransformersClient:
    """Test TransformersClient creation and properties."""

    def test_provider_name(self):
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="google/gemma-4-E2B")
        assert client.provider_name == "transformers"

    def test_multimodal_detection_gemma4(self):
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="google/gemma-4-E2B")
        assert client._is_multimodal("google/gemma-4-E2B") is True
        assert client._is_multimodal("google/gemma-4-E4B") is True

    def test_multimodal_detection_non_gemma(self):
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="meta-llama/Llama-3-8B")
        assert client._is_multimodal("meta-llama/Llama-3-8B") is False

    def test_lazy_loading(self):
        """Model should NOT be loaded at construction time."""
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="google/gemma-4-E2B")
        assert client._model is None
        assert client._tokenizer is None

    def test_resolve_model(self):
        """_resolve_model should return the configured model name."""
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="google/gemma-4-E4B")
        assert client._resolve_model() == "google/gemma-4-E4B"

    def test_resolve_model_raises_without_model(self):
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient()
        with pytest.raises(ValueError, match="No model specified"):
            client._resolve_model()

    def test_multimodal_uses_correct_branch(self):
        """Multimodal models should be detected and use the multimodal branch."""
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="google/gemma-4-E2B", device="cpu")
        assert client._is_multimodal("google/gemma-4-E2B") is True

        # Inject mock model/tokenizer directly (loading requires torchvision)
        client._model = MagicMock()
        client._tokenizer = MagicMock()
        assert client._model is not None
        assert client._tokenizer is not None

    def test_text_only_uses_correct_branch(self):
        """Non-multimodal models should NOT trigger multimodal detection."""
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="meta-llama/Llama-3-8B", device="cpu")
        assert client._is_multimodal("meta-llama/Llama-3-8B") is False

    def test_generate_sync(self):
        """Test synchronous generate path (mocked)."""
        import torch

        from lattifai.llm.transformers import TransformersClient

        # Mock model
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        mock_tokenizer.return_value = {"input_ids": mock_input_ids, "attention_mask": torch.ones(1, 5)}
        mock_tokenizer.decode.return_value = "Generated response"

        client = TransformersClient(model="test-model", device="cpu")
        # Inject mocked model/tokenizer directly (skip _load_model)
        client._model = mock_model
        client._tokenizer = mock_tokenizer

        result = client.generate_sync("Hello, world!")

        assert result == "Generated response"
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_model.generate.assert_called_once()

    def test_generate_json_sync(self):
        """Test synchronous generate_json path (mocked)."""
        import torch

        from lattifai.llm.transformers import TransformersClient

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        mock_tokenizer.return_value = {"input_ids": mock_input_ids, "attention_mask": torch.ones(1, 5)}
        mock_tokenizer.decode.return_value = '{"key": "value"}'

        client = TransformersClient(model="test-model", device="cpu")
        client._model = mock_model
        client._tokenizer = mock_tokenizer

        result = client.generate_json_sync("Return JSON")

        assert result == {"key": "value"}

    def test_dtype_options(self):
        """Test that dtype parameter is accepted."""
        from lattifai.llm.transformers import TransformersClient

        for dtype in ("bfloat16", "float16", "float32", None):
            client = TransformersClient(model="test-model", dtype=dtype)
            assert client._dtype_str == dtype

    def test_invalid_dtype_raises(self):
        """Invalid dtype should raise during model loading."""
        from lattifai.llm.transformers import TransformersClient

        client = TransformersClient(model="test-model", dtype="invalid")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            client._load_model()
