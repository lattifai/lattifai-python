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
