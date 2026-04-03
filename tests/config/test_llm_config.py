"""Tests for LLMConfig resolution chain."""

from unittest.mock import patch

import pytest

from lattifai.config.llm import LLMConfig


class TestModelNameResolution:
    """model_name: explicit > config.toml [section] > fallback_model > raise."""

    def test_explicit_model_preserved(self):
        c = LLMConfig(model_name="my-model")
        assert c.model_name == "my-model"

    def test_fallback_model_used_when_none(self):
        c = LLMConfig(fallback_model="fb-model")
        assert c.model_name == "fb-model"

    def test_empty_string_treated_as_none(self):
        """Empty string should trigger config.toml lookup, then fallback."""
        c = LLMConfig(model_name="", fallback_model="fb-model")
        assert c.model_name == "fb-model"

    def test_raises_without_model_or_fallback(self):
        with pytest.raises(ValueError, match="No model_name provided"):
            LLMConfig()

    @patch("lattifai.config.llm.resolve_toml_value", return_value=None)
    def test_raises_with_section_hint(self, _):
        with pytest.raises(ValueError, match=r"No model configured for \[translation\]"):
            LLMConfig(section="translation")

    @patch("lattifai.config.llm.resolve_toml_value")
    def test_config_toml_resolution(self, mock_resolve):
        mock_resolve.return_value = "toml-model"
        c = LLMConfig(section="translation")
        assert c.model_name == "toml-model"
        mock_resolve.assert_any_call("translation", "model_name")

    @patch("lattifai.config.llm.resolve_toml_value")
    def test_explicit_not_overridden_by_toml(self, mock_resolve):
        mock_resolve.return_value = "toml-model"
        c = LLMConfig(model_name="explicit", section="translation")
        assert c.model_name == "explicit"

    @patch("lattifai.config.llm.resolve_toml_value", return_value=None)
    def test_fallback_when_toml_empty(self, _):
        c = LLMConfig(section="translation", fallback_model="fb-model")
        assert c.model_name == "fb-model"


class TestProviderInference:
    """provider: inferred from model_name prefix (read-only property)."""

    def test_gemini_model_infers_gemini(self):
        c = LLMConfig(model_name="gemini-3-flash-preview")
        assert c.provider == "gemini"

    def test_gemini_prefix_variants(self):
        for name in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"):
            assert LLMConfig._infer_provider(name) == "gemini"

    def test_non_gemini_infers_openai(self):
        c = LLMConfig(model_name="gpt-4o")
        assert c.provider == "openai"

    def test_openai_model_variants(self):
        for name in ("gpt-4o", "qwen3", "deepseek-r1", "llama-3.1-70b"):
            assert LLMConfig._infer_provider(name) == "openai"

    def test_none_model_infers_openai(self):
        assert LLMConfig._infer_provider(None) == "openai"

    def test_empty_model_infers_openai(self):
        assert LLMConfig._infer_provider("") == "openai"

    def test_provider_is_read_only_property(self):
        c = LLMConfig(model_name="gemini-3-flash-preview")
        assert c.provider == "gemini"
        # Changing model_name changes the inferred provider
        c.model_name = "gpt-4o"
        assert c.provider == "openai"


class TestApiKeyResolution:
    """api_key: explicit > env var > [section].api_key > global config."""

    def test_explicit_api_key_preserved(self):
        c = LLMConfig(model_name="m", api_key="explicit-key")
        assert c.api_key == "explicit-key"

    @patch("lattifai.config.llm.resolve_toml_value", return_value=None)
    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=False)
    def test_env_var_gemini(self, _):
        c = LLMConfig(model_name="gemini-3-flash-preview")
        assert c.api_key == "env-key"

    @patch("lattifai.config.llm.resolve_toml_value", return_value=None)
    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}, clear=False)
    def test_env_var_openai(self, _):
        c = LLMConfig(model_name="gpt-4o", api_base_url="http://x")
        assert c.api_key == "env-key"

    @patch("lattifai.config.llm.resolve_toml_value")
    @patch("dotenv.find_dotenv", return_value="")
    @patch("dotenv.load_dotenv", return_value=None)
    @patch.dict("os.environ", {}, clear=False)
    def test_section_api_key(self, _ld, _fd, mock_resolve):
        import os

        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

        def side_effect(section, key):
            return {"model_name": "m", "api_key": "section-key"}.get(key)

        mock_resolve.side_effect = side_effect
        c = LLMConfig(section="translation", fallback_model="m")
        assert c.api_key == "section-key"


class TestBaseUrlResolution:
    """api_base_url: explicit > env var > [section].api_base_url > global config."""

    @patch("lattifai.config.llm.resolve_toml_value", return_value=None)
    @patch.dict("os.environ", {"OPENAI_API_BASE_URL": "http://env"}, clear=False)
    def test_env_var(self, _):
        c = LLMConfig(model_name="gpt-4o")
        assert c.api_base_url == "http://env"

    @patch("lattifai.config.llm.resolve_toml_value")
    @patch("dotenv.find_dotenv", return_value="")
    @patch("dotenv.load_dotenv", return_value=None)
    @patch.dict("os.environ", {}, clear=False)
    def test_section_base_url(self, _ld, _fd, mock_resolve):
        import os

        os.environ.pop("OPENAI_API_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)

        def side_effect(section, key):
            return {
                "model_name": "gpt-4o",
                "api_key": "k",
                "api_base_url": "http://section",
            }.get(key)

        mock_resolve.side_effect = side_effect
        c = LLMConfig(section="translation")
        assert c.api_base_url == "http://section"


class TestSubclassSurvivesNemoRun:
    """Subclasses (TranslationLLMConfig, etc.) preserve section/fallback when nemo_run reconstructs."""

    def test_translation_subclass_defaults(self):
        from lattifai.config.translation import TranslationLLMConfig

        # nemo_run would construct: TranslationLLMConfig(model_name=None)
        # section="translation" and fallback_model survive as class defaults
        c = TranslationLLMConfig()
        assert c.section == "translation.llm"
        assert c.model_name is not None  # resolved from config.toml or fallback

    def test_translation_subclass_with_explicit_model(self):
        from lattifai.config.translation import TranslationLLMConfig

        c = TranslationLLMConfig(model_name="my-model")
        assert c.model_name == "my-model"
        assert c.section == "translation.llm"

    def test_translation_config_llm_type(self):
        from lattifai.config.translation import TranslationConfig, TranslationLLMConfig

        tc = TranslationConfig()
        assert isinstance(tc.llm, TranslationLLMConfig)
        assert tc.llm.section == "translation.llm"
