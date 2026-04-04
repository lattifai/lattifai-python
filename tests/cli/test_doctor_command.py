"""Tests for lai doctor CLI command."""

import subprocess

import pytest


class TestDoctorHelp:
    def test_doctor_runs(self):
        """Test that lai doctor executes without crashing."""
        result = subprocess.run(
            ["lai", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # doctor may return 0 (all pass) or 1 (some fail) — both are valid
        assert result.returncode in (0, 1)
        # Should contain some check output
        combined = result.stdout + result.stderr
        assert "LattifAI" in combined or "Doctor" in combined or "check" in combined.lower()


class TestDoctorInternals:
    """Unit tests for doctor check functions."""

    def test_check_os(self):
        from lattifai.cli.doctor import _check_os

        name, detail, status = _check_os()
        assert name == "OS"
        assert status == "OK"

    def test_check_python_version(self):
        from lattifai.cli.doctor import _check_python_version

        name, detail, status = _check_python_version()
        assert name == "Python version"
        assert status in ("OK", "FAIL")

    def test_check_gpu(self):
        from lattifai.cli.doctor import _check_gpu

        name, detail, status = _check_gpu()
        assert name == "GPU acceleration"
        assert status in ("OK", "WARN", "FAIL")

    def test_check_api_key_not_set(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.delenv("LATTIFAI_API_KEY", raising=False)
        from lattifai.cli.doctor import _check_api_key

        # Patch all three fallback sources to return nothing
        with (
            patch("lattifai.cli.doctor.os.environ.get", return_value=""),
            patch("lattifai.cli.config.get_auth_value", return_value=None),
            patch("dotenv.dotenv_values", return_value={}),
        ):
            name, detail, status = _check_api_key()
        assert status == "WARN"

    def test_check_api_key_set(self, monkeypatch):
        monkeypatch.setenv("LATTIFAI_API_KEY", "test-key-12345678")
        from lattifai.cli.doctor import _check_api_key

        name, detail, status = _check_api_key()
        assert status == "OK"
        assert "5678" in detail

    def test_check_dependencies(self):
        from lattifai.cli.doctor import _check_dependencies

        name, detail, status = _check_dependencies()
        assert name == "Dependencies"
        assert status in ("OK", "FAIL")

    def test_doctor_returns_exit_code(self):
        from lattifai.cli.doctor import doctor

        code = doctor()
        assert code in (0, 1)
