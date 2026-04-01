"""Tests for lai update CLI command."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestUpdateHelp:
    def test_update_help(self):
        result = subprocess.run(
            ["lai", "update", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout


class TestAutoUpdater:
    """Unit tests for the AutoUpdater class."""

    def test_get_latest_version_success(self):
        from lattifai.cli.update import AutoUpdater

        updater = AutoUpdater()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "99.0.0"}}

        with patch("lattifai.cli.update.requests.get", return_value=mock_response):
            version = updater.get_latest_version()
        assert version == "99.0.0"

    def test_get_latest_version_network_error(self):
        from lattifai.cli.update import AutoUpdater

        updater = AutoUpdater()
        import requests

        with patch("lattifai.cli.update.requests.get", side_effect=requests.RequestException("timeout")):
            version = updater.get_latest_version()
        assert version is None

    def test_already_latest(self):
        from lattifai.cli.update import AutoUpdater

        updater = AutoUpdater()
        with (
            patch("importlib.metadata.version", return_value="1.0.0"),
            patch.object(updater, "get_latest_version", return_value="1.0.0"),
        ):
            code = updater.run(force=False)
        assert code == 0

    def test_pip_install_failure(self):
        from lattifai.cli.update import AutoUpdater

        updater = AutoUpdater()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: Could not install"
        with patch("subprocess.run", return_value=mock_result):
            code = updater._pip_install(force=False)
        assert code == 1

    def test_pip_install_timeout(self):
        from lattifai.cli.update import AutoUpdater

        updater = AutoUpdater()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=300)):
            code = updater._pip_install(force=False)
        assert code == 1
