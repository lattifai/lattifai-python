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


class TestTranscriptionModelValidation:
    """Validate the built-in model whitelist + escape hatches.

    The whitelist (`SUPPORTED_TRANSCRIPTION_MODELS`) is the curated list of
    models we have explicitly tested. To avoid a new release every time Google
    ships a new Gemini variant, any `gemini-*` model id passes through; the
    Gemini API itself will reject typos at call time.
    """

    def test_gemini_3_5_flash_accepted(self):
        """v1.5.14: gemini-3.5-flash (Google I/O 2026-05-19 GA release) is in
        the curated whitelist and must not raise."""
        cfg = TranscriptionConfig(model_name="gemini-3.5-flash")
        assert cfg.model_name == "gemini-3.5-flash"

    def test_gemini_3_1_flash_lite_ga_accepted(self):
        """gemini-3.1-flash-lite (preview → GA) was missing from the whitelist
        before v1.5.14."""
        cfg = TranscriptionConfig(model_name="gemini-3.1-flash-lite")
        assert cfg.model_name == "gemini-3.1-flash-lite"

    def test_unlisted_gemini_prefix_accepted(self):
        """Forward-compat: any `gemini-*` model id passes the whitelist gate
        without a release bump. Google's API returns 404 on typos so the
        feedback loop is short enough."""
        cfg = TranscriptionConfig(model_name="gemini-4.0-flash-imaginary-preview")
        assert cfg.model_name == "gemini-4.0-flash-imaginary-preview"

    def test_non_gemini_unlisted_model_still_rejected(self):
        """The prefix escape hatch is gemini-only — random model ids from
        other providers must still raise so users get a clear error before
        the first API call."""
        with pytest.raises(ValueError, match="Unsupported model_name"):
            TranscriptionConfig(model_name="claude-4-opus-imaginary")

    def test_deprecated_gemini_3_pro_auto_switched(self):
        """Existing deprecation alias still works post-change: gemini-3-pro-preview
        gets rewritten to gemini-3.1-pro-preview."""
        cfg = TranscriptionConfig(model_name="gemini-3-pro-preview")
        assert cfg.model_name == "gemini-3.1-pro-preview"
