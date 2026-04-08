"""Unit tests for AlignmentConfig, TranscriptionConfig, DiarizationConfig, TranslationConfig defaults and constraints."""

import pytest

from lattifai.config import AlignmentConfig, TranscriptionConfig
from lattifai.config.diarization import DiarizationConfig
from lattifai.config.translation import TranslationConfig


class TestAlignmentConfigDefaults:
    """Test AlignmentConfig default values and basic constraints."""

    def test_default_model(self):
        cfg = AlignmentConfig()
        assert cfg.model_name == "LattifAI/Lattice-1"

    def test_device_resolves(self):
        """device='auto' resolves to a concrete device (cpu/cuda/mps)."""
        cfg = AlignmentConfig()
        assert cfg.device in ("cpu", "cuda", "mps")

    def test_explicit_device(self):
        cfg = AlignmentConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_default_strategy(self):
        cfg = AlignmentConfig()
        assert cfg.strategy == "entire"

    def test_strategy_values(self):
        for s in ("entire", "caption", "transcription"):
            cfg = AlignmentConfig(strategy=s)
            assert cfg.strategy == s

    def test_default_trust_caption_timestamps(self):
        cfg = AlignmentConfig()
        assert cfg.trust_caption_timestamps is False

    def test_default_segment_duration(self):
        cfg = AlignmentConfig()
        assert cfg.segment_duration == 300.0

    def test_default_batch_size(self):
        cfg = AlignmentConfig()
        assert cfg.batch_size == 1


class TestTranscriptionConfigDefaults:
    """Test TranscriptionConfig default values and overrides."""

    def test_model_name_is_string(self):
        cfg = TranscriptionConfig()
        assert isinstance(cfg.model_name, str) and len(cfg.model_name) > 0

    def test_default_language_none(self):
        cfg = TranscriptionConfig()
        assert cfg.language is None

    def test_default_prompt_none(self):
        cfg = TranscriptionConfig()
        assert cfg.prompt is None

    def test_default_api_mode(self):
        cfg = TranscriptionConfig()
        assert cfg.api_mode in ("transcriptions", "chat", "realtime")

    def test_default_max_tokens(self):
        cfg = TranscriptionConfig()
        assert cfg.max_tokens is None

    def test_custom_model(self):
        cfg = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3")
        assert cfg.model_name == "nvidia/parakeet-tdt-0.6b-v3"

    def test_custom_language(self):
        cfg = TranscriptionConfig(language="zh")
        assert cfg.language == "zh"


class TestDiarizationConfigDefaults:
    """Test DiarizationConfig default values."""

    def test_default_enabled(self):
        cfg = DiarizationConfig()
        assert cfg.enabled is False

    def test_default_num_speakers_none(self):
        cfg = DiarizationConfig()
        assert cfg.num_speakers is None

    def test_custom_num_speakers(self):
        cfg = DiarizationConfig(num_speakers=3)
        assert cfg.num_speakers == 3


class TestTranslationConfigDefaults:
    """Test TranslationConfig overrides work correctly."""

    def test_target_lang_is_settable(self):
        cfg = TranslationConfig(target_lang="ja")
        assert cfg.target_lang == "ja"

    def test_bilingual_is_settable(self):
        cfg = TranslationConfig(bilingual=True)
        assert cfg.bilingual is True
        cfg2 = TranslationConfig(bilingual=False)
        assert cfg2.bilingual is False

    def test_mode_default(self):
        cfg = TranslationConfig()
        assert cfg.mode in ("quick", "normal", "refined")
