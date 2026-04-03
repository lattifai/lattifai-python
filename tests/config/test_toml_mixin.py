"""Tests for resolve_toml_raw_value and _toml_section declarations."""

import pytest

from lattifai.config.toml_mixin import resolve_toml_raw_value


class TestTomlSectionDeclarations:
    """Every Config class must declare _toml_section matching its TOML table name."""

    @pytest.mark.parametrize(
        "cls_path,expected_section",
        [
            ("lattifai.config.alignment.AlignmentConfig", "alignment"),
            ("lattifai.config.caption.CaptionConfig", "caption"),
            ("lattifai.config.client.ClientConfig", "client"),
            ("lattifai.config.diarization.DiarizationConfig", "diarization"),
            ("lattifai.config.event.EventConfig", "event"),
            ("lattifai.config.media.MediaConfig", "media"),
            ("lattifai.config.summarization.SummarizationConfig", "summarization"),
            ("lattifai.config.transcription.TranscriptionConfig", "transcription"),
            ("lattifai.config.translation.TranslationConfig", "translation"),
        ],
    )
    def test_toml_section_value(self, cls_path, expected_section):
        """Each Config class has the correct _toml_section for config.toml resolution."""
        module_path, cls_name = cls_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        assert hasattr(cls, "_toml_section"), f"{cls_name} missing _toml_section"
        assert cls._toml_section == expected_section

    def test_llm_subclasses_use_nested_sections(self):
        """LLM config subclasses use dotted sections for nested TOML tables."""
        from lattifai.config.diarization import DiarizationLLMConfig
        from lattifai.config.translation import TranslationLLMConfig

        assert TranslationLLMConfig().section == "translation.llm"
        assert DiarizationLLMConfig().section == "diarization.llm"


class TestResolveTomlRawValue:
    """resolve_toml_raw_value reads raw values from ~/.lattifai/config.toml.

    These tests hit the real config file (if present) or get None.
    The entrypoint injection tests (test_entrypoint.py) cover the mocked flow.
    """

    def test_returns_none_for_missing_section(self):
        """Non-existent section returns None without error."""
        result = resolve_toml_raw_value("__nonexistent_test_section__", "key")
        assert result is None

    def test_returns_none_for_missing_key(self):
        """Existing section with missing key returns None."""
        # "auth" section exists in most config.toml setups
        result = resolve_toml_raw_value("auth", "__nonexistent_key__")
        assert result is None

    def test_dotted_section_returns_none_for_missing(self):
        """Dotted (nested) section returns None when not present."""
        result = resolve_toml_raw_value("__nonexistent__.llm", "model_name")
        assert result is None
