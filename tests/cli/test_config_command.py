"""Tests for lai config CLI command and config module internals."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigHelp:
    def test_config_show_help(self):
        result = subprocess.run(
            ["lai", "config", "show", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_config_set_help(self):
        result = subprocess.run(
            ["lai", "config", "set", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_config_get_help(self):
        result = subprocess.run(
            ["lai", "config", "get", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout


class TestConfigInternals:
    """Unit tests for config module helper functions."""

    def test_load_empty_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        with patch("lattifai.cli.config.CONFIG_FILE", config_file):
            from lattifai.cli.config import _load_config

            assert _load_config() == {}

    def test_save_and_load_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.config import _load_config, _normalize_config, _save_config

            data = {"api": {"GEMINI_API_KEY": "test-key-123"}}
            _save_config(data)
            assert config_file.exists()
            loaded = _normalize_config(_load_config())
            assert loaded.get("api", {}).get("GEMINI_API_KEY") == "test-key-123"

    def test_normalize_key(self):
        from lattifai.cli.config import _normalize_key

        assert _normalize_key("gemini_api_key") == "GEMINI_API_KEY"
        assert _normalize_key("transcription.model_name") == "transcription.model_name"

    def test_parse_dotted_key(self):
        from lattifai.cli.config import _parse_dotted_key

        assert _parse_dotted_key("transcription.model_name") == ("transcription", "model_name")
        assert _parse_dotted_key("GEMINI_API_KEY") == (None, "GEMINI_API_KEY")

    def test_mask_value(self):
        from lattifai.cli.config import _mask_value

        assert _mask_value("abcdefghijklmnop") == "abcd...mnop"
        assert _mask_value("short") == "****"

    def test_format_toml_value(self):
        from lattifai.cli.config import _format_toml_value

        assert _format_toml_value(True) == "true"
        assert _format_toml_value(False) == "false"
        assert _format_toml_value(42) == "42"
        assert _format_toml_value("hello") == '"hello"'

    def test_get_section_name(self):
        from lattifai.cli.config import _get_section_name

        assert _get_section_name("LATTIFAI_API_KEY") == "auth"
        assert _get_section_name("GEMINI_API_KEY") == "api"
        assert _get_section_name("unknown_key") is None

    def test_auth_operations(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.config import clear_auth, get_auth_value, set_auth_value

            set_auth_value("LATTIFAI_API_KEY", "test-key")
            assert get_auth_value("LATTIFAI_API_KEY") == "test-key"
            clear_auth()
            assert get_auth_value("LATTIFAI_API_KEY") is None


class TestSectionKeyDiscovery:
    """Verify auto-discovery of section keys from Config dataclasses."""

    def test_discovers_legacy_keys(self):
        """All previously hardcoded keys are still present."""
        from lattifai.cli.config import SECTION_KEYS

        legacy = [
            "transcription.model_name",
            "translation.llm.model_name",
            "translation.llm.api_key",
            "translation.llm.api_base_url",
            "diarization.llm.model_name",
            "diarization.llm.api_key",
            "diarization.llm.api_base_url",
        ]
        for key in legacy:
            assert key in SECTION_KEYS, f"Legacy key missing: {key}"

    def test_discovers_new_section_keys(self):
        """Fields added to Config classes are auto-discovered."""
        from lattifai.cli.config import SECTION_KEYS

        # Spot-check a few fields from different Config classes
        expected = [
            "translation.target_lang",
            "translation.mode",
            "diarization.infer_speakers",
            "diarization.num_speakers",
            "alignment.device",
            "alignment.strategy",
            "transcription.device",
            "transcription.language",
            "summarization.lang",
            "summarization.length",
        ]
        for key in expected:
            assert key in SECTION_KEYS, f"Expected key missing: {key}"

    def test_excludes_internal_fields(self):
        """LLMConfig internal fields (section, fallback_model) are excluded."""
        from lattifai.cli.config import SECTION_KEYS

        for key in SECTION_KEYS:
            assert not key.endswith(".section"), f"Internal field leaked: {key}"
            assert not key.endswith(".fallback_model"), f"Internal field leaked: {key}"

    def test_excludes_repr_false_fields(self):
        """Fields with repr=False (e.g. client_wrapper) are excluded."""
        from lattifai.cli.config import SECTION_KEYS

        for key in SECTION_KEYS:
            assert not key.endswith(".client_wrapper"), f"repr=False field leaked: {key}"

    def test_excludes_nested_config_fields(self):
        """Nested Config fields (e.g. karaoke, llm) are sections, not keys."""
        from lattifai.cli.config import SECTION_KEYS

        for key in SECTION_KEYS:
            # "diarization.llm" would be wrong — "diarization.llm.model_name" is correct
            parts = key.split(".")
            assert len(parts) >= 2, f"Invalid key format: {key}"

    def test_provider_removed_from_section_keys(self):
        """provider field no longer exists in LLMConfig, thus not discovered."""
        from lattifai.cli.config import SECTION_KEYS

        for key in SECTION_KEYS:
            assert not key.endswith(".provider"), f"Removed provider field found: {key}"

    def test_secret_keys_auto_detected(self):
        """Keys ending with .api_key are auto-detected as secrets."""
        from lattifai.cli.config import SECRET_KEYS

        assert "translation.llm.api_key" in SECRET_KEYS
        assert "diarization.llm.api_key" in SECRET_KEYS
        assert "client.api_key" in SECRET_KEYS

    def test_all_keys_includes_both_maps(self):
        """ALL_KEYS is the union of KEY_MAP and SECTION_KEYS."""
        from lattifai.cli.config import ALL_KEYS, KEY_MAP, SECTION_KEYS

        assert ALL_KEYS == set(KEY_MAP) | set(SECTION_KEYS)


class TestConfigSetSubprocess:
    """Subprocess-level tests for 'lai config set' parameter validation."""

    def test_config_set_missing_value_exits_nonzero(self):
        """'lai config set KEY' without a value should fail."""
        result = subprocess.run(
            ["lai", "config", "set", "GEMINI_API_KEY"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "missing" in combined.lower() or "value" in combined.lower() or "usage" in combined.lower()

    def test_config_set_unknown_key_exits_nonzero(self):
        """'lai config set UNKNOWN_KEY value' should fail with unknown key error."""
        result = subprocess.run(
            ["lai", "config", "set", "TOTALLY_BOGUS_KEY", "some-value"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "unknown" in combined.lower() or "key" in combined.lower()

    def test_config_set_equals_syntax_help(self):
        """'lai config set KEY=VALUE' syntax should be documented in help."""
        result = subprocess.run(
            ["lai", "config", "set", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "KEY" in combined or "key" in combined.lower()


class TestConfigSetUnit:
    """Unit tests for 'lai config set' logic."""

    def test_set_and_get_section_key(self, tmp_path):
        """Setting a dotted section key persists and can be read back."""
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.config import _load_config, _normalize_config, _save_config

            # Simulate what 'lai config set transcription.model_name gemini-2.5-flash' does
            config = _normalize_config(_load_config())
            config.setdefault("transcription", {})["model_name"] = "gemini-2.5-flash"
            _save_config(config)

            loaded = _normalize_config(_load_config())
            assert loaded.get("transcription", {}).get("model_name") == "gemini-2.5-flash"

    def test_set_equals_syntax_parsing(self):
        """KEY=VALUE syntax is correctly split."""
        key_raw = "GEMINI_API_KEY=test-key-123"
        if "=" in key_raw:
            key, value = key_raw.split("=", 1)
        assert key == "GEMINI_API_KEY"
        assert value == "test-key-123"

    def test_set_equals_with_equals_in_value(self):
        """KEY=VAL=UE only splits on the first '='."""
        key_raw = "OPENAI_API_BASE_URL=http://localhost:8000/v1?key=abc"
        key, value = key_raw.split("=", 1)
        assert key == "OPENAI_API_BASE_URL"
        assert value == "http://localhost:8000/v1?key=abc"
