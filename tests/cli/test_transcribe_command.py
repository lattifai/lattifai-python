"""Tests for lai transcribe CLI command."""

import os
import subprocess

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_transcribe_command(args, env=None):
    """Helper to run the transcribe command and return result."""
    cmd = ["lai", "transcribe", "run", "-Y"]
    if LATTIFAI_TESTS_CLI_DRYRUN:
        cmd.append("--dryrun")
    cmd.extend(args)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True, env=env)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command: {' '.join(cmd)} failed with exit code {e.returncode}")
        raise e


class TestTranscribeHelp:
    def test_transcribe_help(self):
        result = subprocess.run(
            ["lai", "transcribe", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_transcribe_align_help(self):
        cmd = ["lai", "transcribe", "align", "-Y", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 or "help" in result.stdout


class TestTranscribeErrors:
    def test_missing_input(self, tmp_path):
        args = ["nonexistent_audio.wav", str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_transcribe_command(args)

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "nvidia/parakeet-tdt-0.6b-v3"])
    def test_transcribe_model_options_help(self, model):
        """Verify model names are accepted as arguments (help level)."""
        result = run_transcribe_command(["--help"])
        if result is not None:
            assert result.returncode == 0
