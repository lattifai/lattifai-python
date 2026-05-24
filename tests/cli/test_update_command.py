"""Tests for lai update CLI command."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests as _requests


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


class TestCoreVersionCheck:
    """Tests for the new lattifai-core version check in `lai update`.

    Covers:
      - Local version display
      - PyPI latest version comparison
      - pyproject constraint verification (importlib.metadata.requires)
      - Graceful degradation on missing package / network failure
      - editable path also runs the check
    """

    def _make_updater(self):
        from lattifai.cli.update import AutoUpdater

        return AutoUpdater()

    def _capture_console_output(self, fn):
        """Run fn() with patched console and return all rendered strings (concatenated)."""
        import lattifai.cli.update as mod

        captured = []
        with patch.object(
            mod.console, "print", side_effect=lambda *a, **kw: captured.append(" ".join(str(x) for x in a))
        ):
            fn()
        return "\n".join(captured)

    def test_core_version_installed_latest(self):
        updater = self._make_updater()

        with (
            patch.object(updater, "_get_core_version", return_value="0.7.8"),
            patch.object(updater, "_get_latest_core_version", return_value="0.7.8"),
            patch.object(updater, "_get_core_constraint", return_value=">=0.7.8"),
        ):
            output = self._capture_console_output(lambda: updater._check_core_version())

        assert "lattifai-core" in output
        assert "0.7.8" in output
        assert "latest" in output.lower()
        assert "not satisfied" not in output.lower()

    def test_core_version_outdated(self):
        updater = self._make_updater()

        with (
            patch.object(updater, "_get_core_version", return_value="0.7.5"),
            patch.object(updater, "_get_latest_core_version", return_value="0.8.1"),
            patch.object(updater, "_get_core_constraint", return_value=None),
        ):
            output = self._capture_console_output(lambda: updater._check_core_version())

        assert "lattifai-core" in output
        assert "0.7.5" in output
        assert "0.8.1" in output
        assert "available" in output.lower()

    def test_core_constraint_violated(self):
        updater = self._make_updater()

        with (
            patch.object(updater, "_get_core_version", return_value="0.7.5"),
            patch.object(updater, "_get_latest_core_version", return_value="0.8.1"),
            patch.object(updater, "_get_core_constraint", return_value=">=0.7.8"),
        ):
            output = self._capture_console_output(lambda: updater._check_core_version())

        assert "0.7.5" in output
        assert ">=0.7.8" in output
        assert "not satisfied" in output.lower()
        assert "pip install" in output.lower()

    def test_core_not_installed(self):
        updater = self._make_updater()

        with (
            patch.object(updater, "_get_core_version", return_value=None),
            patch.object(updater, "_get_latest_core_version", return_value="0.7.8"),
            patch.object(updater, "_get_core_constraint", return_value=">=0.7.8"),
        ):
            output = self._capture_console_output(lambda: updater._check_core_version())

        assert "lattifai-core" in output
        assert "not installed" in output.lower()

    def test_core_pypi_unreachable(self):
        updater = self._make_updater()

        with (
            patch.object(updater, "_get_core_version", return_value="0.7.8"),
            patch.object(updater, "_get_latest_core_version", return_value=None),
            patch.object(updater, "_get_core_constraint", return_value=None),
        ):
            output = self._capture_console_output(lambda: updater._check_core_version())

        assert "lattifai-core" in output
        assert "0.7.8" in output

    def test_get_core_version_reads_metadata(self):
        updater = self._make_updater()

        with patch("importlib.metadata.version", return_value="0.7.8") as mock_v:
            result = updater._get_core_version()
        assert result == "0.7.8"
        mock_v.assert_called_with("lattifai-core")

    def test_get_core_version_returns_none_when_missing(self):
        import importlib.metadata as _meta

        updater = self._make_updater()

        with patch("importlib.metadata.version", side_effect=_meta.PackageNotFoundError("lattifai-core")):
            result = updater._get_core_version()
        assert result is None

    def test_get_latest_core_version_from_private_mirror(self):
        """Primary path: parse versions from private mirror simple index HTML."""
        updater = self._make_updater()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '<a href="../../wheels/lattifai_core-0.6.7-cp310-cp310-macosx.whl">x</a>'
            '<a href="../../wheels/lattifai_core-0.7.8-cp311-cp311-macosx.whl">x</a>'
            '<a href="../../wheels/lattifai_core-0.7.10-cp312-cp312-macosx.whl">x</a>'
        )

        with patch("lattifai.cli.update.requests.get", return_value=mock_response) as mock_get:
            result = updater._get_latest_core_version()

        # 0.7.10 > 0.7.8 > 0.6.7 by PEP 440
        assert result == "0.7.10"
        # ensure private mirror was queried
        args, _ = mock_get.call_args
        assert "lattifai.github.io/pypi/simple" in args[0]
        assert "lattifai-core" in args[0]

    def test_get_latest_core_version_network_error(self):
        updater = self._make_updater()
        with patch("lattifai.cli.update.requests.get", side_effect=_requests.RequestException("timeout")):
            result = updater._get_latest_core_version()
        assert result is None

    def test_get_latest_core_version_empty_index(self):
        """Empty simple index (no wheels) should return None, not crash."""
        updater = self._make_updater()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>No links here.</body></html>"

        with patch("lattifai.cli.update.requests.get", return_value=mock_response):
            result = updater._get_latest_core_version()
        assert result is None

    def test_get_core_constraint_parses_requires(self):
        updater = self._make_updater()
        fake_requires = [
            "k2py>=1.0",
            "lattifai-core>=0.7.8",
            "lhotse",
        ]
        with patch("importlib.metadata.requires", return_value=fake_requires):
            result = updater._get_core_constraint()
        assert result == ">=0.7.8"

    def test_get_core_constraint_with_extras(self):
        """Constraint with extras like `lattifai-core[event]>=0.7.8` should still parse."""
        updater = self._make_updater()
        fake_requires = [
            "lattifai-core[event]>=0.7.8 ; extra == 'event'",
        ]
        with patch("importlib.metadata.requires", return_value=fake_requires):
            result = updater._get_core_constraint()
        assert result == ">=0.7.8"

    def test_get_core_constraint_none_when_missing(self):
        updater = self._make_updater()
        with patch("importlib.metadata.requires", return_value=["k2py>=1.0"]):
            result = updater._get_core_constraint()
        assert result is None

    def test_post_check_invokes_core_check(self):
        """post_check() must call _check_core_version()."""
        updater = self._make_updater()
        with (
            patch("onnxruntime.get_available_providers", return_value=[]),
            patch.object(updater, "_check_core_version") as mock_check,
        ):
            updater.post_check()
        mock_check.assert_called_once()

    def test_refresh_editable_invokes_core_check_on_success(self):
        """Editable refresh success path must also run the core version check."""
        from pathlib import Path

        updater = self._make_updater()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with (
            patch("lattifai.cli.doctor._get_source_version", return_value="2.0.0"),
            patch("lattifai.cli.doctor._find_stale_egg_info", return_value=[]),
            patch("subprocess.run", return_value=mock_result),
            patch.object(updater, "_check_core_version") as mock_check,
        ):
            code = updater._refresh_editable(str(Path("/tmp/fake-src")), current_v="1.0.0")

        assert code == 0
        mock_check.assert_called_once()

    def test_refresh_editable_invokes_core_check_when_in_sync(self):
        """Editable 'already in sync' early-return path must also run the core check.

        Bug repro: when the editable install is already at the source version
        and no stale egg-info exists, _refresh_editable returned 0 without
        calling _check_core_version. That hides lattifai-core drift.
        """
        from pathlib import Path

        updater = self._make_updater()

        with (
            patch("lattifai.cli.doctor._get_source_version", return_value="1.5.14"),
            patch("lattifai.cli.doctor._find_stale_egg_info", return_value=[]),
            patch.object(updater, "_check_core_version") as mock_check,
        ):
            code = updater._refresh_editable(str(Path("/tmp/fake-src")), current_v="1.5.14")

        assert code == 0
        mock_check.assert_called_once()
