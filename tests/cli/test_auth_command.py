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

    def test_persist_auth_preserves_comments(self, tmp_path):
        """_persist_auth should not destroy TOML comments (tomlkit round-trip)."""
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        # Write a config with comments
        config_file.write_text('[auth]\n# backup key\nLATTIFAI_API_KEY = "old"\nUSER_EMAIL = "old@test.com"\n')
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.auth import _persist_auth

            _persist_auth("new-key", {"user_email": "new@test.com", "key_name": "New"})
            content = config_file.read_text()
            assert "# backup key" in content, "Comment was destroyed during persist"

    def test_persist_trial_preserves_unrelated_keys(self, tmp_path):
        """_persist_trial_auth should preserve unrelated [auth] entries.

        It must clear OAuth-only fields (USER_EMAIL/KEY_NAME — see
        test_trial_login_do_not_coexist) but must NOT nuke other keys a user
        might have stored in [auth] (e.g. third-party tokens, API_KEY_ID).
        """
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        config_file.write_text('[auth]\nAPI_KEY_ID = "some-id-12345"\n')
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.auth import _persist_trial_auth
            from lattifai.cli.config import get_auth_value

            _persist_trial_auth({"api_key": "trial-key", "expires_at": "2026-12-31T00:00:00Z", "credits": 120})
            # Trial values written
            assert get_auth_value("IS_TRIAL") is not None
            # Unrelated key preserved (we don't wipe the whole section)
            assert get_auth_value("API_KEY_ID") == "some-id-12345"

    def test_login_after_trial_clears_trial_fields(self, tmp_path):
        """OAuth login after a trial must wipe IS_TRIAL/EXPIRES_AT/CREDITS.

        Regression: trial + login left both sets of fields in [auth],
        producing a misleading config where a logged-in session still
        reported IS_TRIAL=true and a stale EXPIRES_AT.
        """
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.auth import _persist_auth, _persist_trial_auth
            from lattifai.cli.config import get_auth_value

            # 1) Trial first
            _persist_trial_auth({"api_key": "trial-key", "expires_at": "2026-12-31T00:00:00Z", "credits": 120})
            assert get_auth_value("IS_TRIAL") is not None
            # 2) OAuth login next
            _persist_auth("real-key", {"user_email": "u@example.com", "key_name": "Mac-mini"})
            # Trial-only fields gone
            assert get_auth_value("IS_TRIAL") is None
            assert get_auth_value("EXPIRES_AT") is None
            assert get_auth_value("CREDITS") is None
            # Login fields present
            assert get_auth_value("USER_EMAIL") == "u@example.com"
            assert get_auth_value("KEY_NAME") == "Mac-mini"

    def test_trial_after_login_clears_login_fields(self, tmp_path):
        """Trial after OAuth login must wipe USER_EMAIL/KEY_NAME.

        Symmetric to test_login_after_trial_clears_trial_fields.
        """
        config_file = tmp_path / "config.toml"
        config_dir = tmp_path
        with (
            patch("lattifai.cli.config.CONFIG_FILE", config_file),
            patch("lattifai.cli.config.CONFIG_DIR", config_dir),
        ):
            from lattifai.cli.auth import _persist_auth, _persist_trial_auth
            from lattifai.cli.config import get_auth_value

            # 1) OAuth login first
            _persist_auth("real-key", {"user_email": "u@example.com", "key_name": "Mac-mini"})
            assert get_auth_value("USER_EMAIL") == "u@example.com"
            # 2) Trial next
            _persist_trial_auth({"api_key": "trial-key", "expires_at": "2026-12-31T00:00:00Z", "credits": 120})
            # Login-only fields gone
            assert get_auth_value("USER_EMAIL") is None
            assert get_auth_value("KEY_NAME") is None
            # Trial fields present
            assert get_auth_value("IS_TRIAL") is not None
            assert get_auth_value("EXPIRES_AT") == "2026-12-31T00:00:00Z"


class TestTrialExpiryWarning:
    """Tests for the warn_if_trial_expiring() pre-flight check."""

    def _reset_warning_flag(self):
        """Clear the once-per-process guard so each test sees a fresh state."""
        import lattifai.cli.auth as auth

        auth._EXPIRY_WARNING_SHOWN = False

    def test_silent_when_no_trial(self, capsys):
        self._reset_warning_flag()
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: None):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""

    def test_silent_when_far_future(self, capsys):
        self._reset_warning_flag()
        values = {"IS_TRIAL": True, "EXPIRES_AT": "2099-01-01T00:00:00Z"}
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: values.get(k)):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""

    def _normalize(self, s: str) -> str:
        """Collapse Rich's word-wrap whitespace so substring asserts survive line breaks."""
        return " ".join(s.split())

    def test_warns_when_expired(self, capsys):
        self._reset_warning_flag()
        values = {"IS_TRIAL": True, "EXPIRES_AT": "2020-01-01T00:00:00Z"}
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: values.get(k)):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
        captured = self._normalize(capsys.readouterr().out)
        assert "expired" in captured.lower()
        assert "lai auth trial" in captured

    def test_warns_when_expiring_soon(self, capsys):
        from datetime import datetime, timedelta, timezone

        self._reset_warning_flag()
        soon = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat().replace("+00:00", "Z")
        values = {"IS_TRIAL": True, "EXPIRES_AT": soon}
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: values.get(k)):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
        captured = self._normalize(capsys.readouterr().out)
        assert "expires in" in captured.lower()
        assert "lai auth login" in captured

    def test_warning_is_idempotent(self, capsys):
        self._reset_warning_flag()
        values = {"IS_TRIAL": True, "EXPIRES_AT": "2020-01-01T00:00:00Z"}
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: values.get(k)):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
            first = capsys.readouterr().out
            warn_if_trial_expiring()
            second = capsys.readouterr().out
        assert first.strip() != ""
        assert second == ""  # Second call must be silent

    def test_silent_on_malformed_expiry(self, capsys):
        self._reset_warning_flag()
        values = {"IS_TRIAL": True, "EXPIRES_AT": "not-a-date"}
        with patch("lattifai.cli.auth.get_auth_value", side_effect=lambda k: values.get(k)):
            from lattifai.cli.auth import warn_if_trial_expiring

            warn_if_trial_expiring()
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""


class TestDeviceAuthHeaders:
    """Tests for X-Device-Auth header injection in ClientConfig."""

    def test_client_config_injects_device_auth(self):
        """ClientConfig should auto-inject X-Device-Auth into default_headers."""
        from lattifai.config.client import ClientConfig

        config = ClientConfig(api_key="test-key-1234")
        assert config.default_headers is not None
        assert "X-Device-Auth" in config.default_headers

    def test_client_config_device_auth_contains_device_id(self):
        """X-Device-Auth payload should contain device_id."""
        from lattifai.config.client import ClientConfig

        config = ClientConfig(api_key="test-key-1234")
        payload = config.default_headers["X-Device-Auth"]
        assert "device_id" in payload

    def test_auth_headers_function(self):
        """auth.auth_headers() should include both Authorization and X-Device-Auth."""
        from lattifai.auth import auth_headers

        headers = auth_headers("test-key-5678")
        assert headers["Authorization"] == "Bearer test-key-5678"
        assert "X-Device-Auth" in headers
