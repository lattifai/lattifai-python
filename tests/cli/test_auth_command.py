"""Tests for lai auth CLI commands."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAuthHelp:
    def test_auth_login_help(self):
        result = subprocess.run(
            ["lai", "auth", "login", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_auth_logout_help(self):
        result = subprocess.run(
            ["lai", "auth", "logout", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_auth_whoami_help(self):
        result = subprocess.run(
            ["lai", "auth", "whoami", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_auth_trial_help(self):
        result = subprocess.run(
            ["lai", "auth", "trial", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout

    def test_auth_group_lists_subcommands(self):
        """'lai auth --help' should list all 4 subcommands."""
        result = subprocess.run(
            ["lai", "auth", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0
        for cmd in ("login", "logout", "whoami", "trial"):
            assert cmd in combined, f"Subcommand '{cmd}' not found in 'lai auth --help' output"


class TestAuthInternals:
    """Unit tests for auth helper functions."""

    def test_format_time_none(self):
        from lattifai.cli.auth import _format_time

        assert _format_time(None) == ""

    def test_format_time_valid_iso(self):
        from lattifai.cli.auth import _format_time

        result = _format_time("2026-01-01T00:00:00Z")
        assert "2026" in result
        assert "ago" in result or "future" in result

    def test_format_time_invalid(self):
        from lattifai.cli.auth import _format_time

        result = _format_time("not-a-date")
        assert result == "not-a-date"

    def test_now_iso_format(self):
        from lattifai.cli.auth import _now_iso

        iso = _now_iso()
        assert "T" in iso
        assert len(iso) > 10

    def test_local_callback_server_creates(self):
        from lattifai.cli.auth import LocalCallbackServer

        server = LocalCallbackServer(state="test-state", timeout=5.0)
        assert server.state == "test-state"
        assert server.code is None
        assert server.error is None

    def test_local_callback_server_bind_and_close(self):
        from lattifai.cli.auth import LocalCallbackServer

        server = LocalCallbackServer(state="test-state", timeout=5.0)
        server.start()
        assert server.port is not None
        assert server.port >= 49152
        server.close()

    def test_persist_auth(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.auth import _persist_auth
            from lattifai.cli.config import get_auth_value

            _persist_auth("test-api-key", {"user_email": "test@example.com", "key_name": "Test Device"})
            assert get_auth_value("USER_EMAIL") == "test@example.com"
            assert get_auth_value("KEY_NAME") == "Test Device"
            # API key should be obfuscated
            raw = get_auth_value("LATTIFAI_API_KEY")
            assert raw is not None
            assert raw != "test-api-key"  # Should be obfuscated
