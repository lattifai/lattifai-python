"""Unit tests for VLLMTranscriber internal methods.

Tests _resolve_system_prompt(), _is_dedicated_asr_model(), and audio content type branching.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lattifai.config import TranscriptionConfig
from lattifai.transcription.vllm import VLLMTranscriber


@pytest.fixture
def make_transcriber():
    """Factory to create VLLMTranscriber with given config overrides."""

    def _make(**kwargs):
        defaults = {
            "model_name": "openai/whisper-large-v3",
            "api_base_url": "http://localhost:8080/v1",
            "api_mode": "transcriptions",
        }
        defaults.update(kwargs)
        cfg = TranscriptionConfig(**defaults)
        cfg.client_wrapper = MagicMock()
        cfg.client_wrapper.check_permission = MagicMock()
        return VLLMTranscriber(cfg)

    return _make


# ===========================================================================
# _is_dedicated_asr_model()
# ===========================================================================


class TestIsDedicatedASRModel:
    """Test ASR model keyword detection."""

    @pytest.mark.parametrize(
        "model",
        [
            "openai/whisper-large-v3",
            "Qwen/Qwen3-ASR-0.6B",
            "FunAudioLLM/Fun-ASR-Nano-2512",
            "SpeechIO/GLM-ASR-Large",
            "iic/SenseVoiceSmall",
            "mistralai/Voxtral-Mini-3B-2507",
            "nvidia/parakeet-tdt-0.6b-v3",
            "nvidia/canary-1b-v2",
        ],
    )
    def test_asr_models_detected(self, model, make_transcriber):
        t = make_transcriber(model_name=model)
        assert t._is_dedicated_asr_model() is True

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen3-235B-A22B",
            "mistralai/Mistral-Large-Instruct-2501",
        ],
    )
    def test_general_llms_not_detected(self, model, make_transcriber):
        t = make_transcriber(model_name=model)
        assert t._is_dedicated_asr_model() is False


# ===========================================================================
# _resolve_system_prompt()
# ===========================================================================


class TestResolveSystemPrompt:
    """Test system prompt resolution logic."""

    def test_none_for_asr_model(self, make_transcriber):
        """ASR models should get no system prompt by default."""
        t = make_transcriber(model_name="openai/whisper-large-v3")
        assert t._resolve_system_prompt() is None

    def test_default_for_general_llm(self, make_transcriber):
        """General-purpose LLMs get the default ASR system prompt."""
        t = make_transcriber(model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct")
        result = t._resolve_system_prompt()
        assert result is not None
        assert "transcription" in result.lower()

    def test_explicit_string(self, make_transcriber):
        """Explicit system_prompt string is used directly."""
        t = make_transcriber(model_name="openai/whisper-large-v3", system_prompt="Custom ASR prompt")
        assert t._resolve_system_prompt() == "Custom ASR prompt"

    def test_explicit_empty_disables(self, make_transcriber):
        """Empty string explicitly disables system prompt."""
        t = make_transcriber(model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct", system_prompt="")
        assert t._resolve_system_prompt() is None

    def test_file_path_reads_content(self, make_transcriber):
        """File path is read and content is used as system prompt."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  System prompt from file  \n")
            f.flush()
            t = make_transcriber(model_name="openai/whisper-large-v3", system_prompt=f.name)
            assert t._resolve_system_prompt() == "System prompt from file"
            Path(f.name).unlink()

    def test_nonexistent_file_used_as_string(self, make_transcriber):
        """Non-existent file path is treated as literal string."""
        t = make_transcriber(model_name="openai/whisper-large-v3", system_prompt="/nonexistent/path.txt")
        assert t._resolve_system_prompt() == "/nonexistent/path.txt"

    def test_explicit_overrides_asr_default(self, make_transcriber):
        """Explicit system_prompt on ASR model overrides None default."""
        t = make_transcriber(model_name="openai/whisper-large-v3", system_prompt="Force this prompt")
        assert t._resolve_system_prompt() == "Force this prompt"


# ===========================================================================
# TranscriptionConfig new fields defaults
# ===========================================================================


class TestTranscriptionConfigNewFields:
    """Test new TranscriptionConfig fields have correct defaults."""

    def test_system_prompt_default(self):
        cfg = TranscriptionConfig()
        assert cfg.system_prompt is None

    def test_audio_content_type_default(self):
        cfg = TranscriptionConfig()
        assert cfg.audio_content_type == "audio_url"

    def test_audio_content_type_values(self):
        for val in ("audio_url", "input_audio", "audio"):
            cfg = TranscriptionConfig(audio_content_type=val)
            assert cfg.audio_content_type == val

    def test_chat_audio_first_default(self):
        cfg = TranscriptionConfig()
        assert cfg.chat_audio_first is False

    def test_verbose_default(self):
        cfg = TranscriptionConfig()
        assert cfg.verbose is False
