"""Tests for lai translate CLI command."""

import os
import subprocess

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_translate_command(args, env=None):
    """Helper to run the translate command and return result."""
    cmd = ["lai", "translate", "caption", "-Y"]
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


class TestTranslateHelp:
    def test_translate_caption_help(self):
        result = subprocess.run(
            ["lai", "translate", "caption", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_translate_youtube_help(self):
        cmd = ["lai", "translate", "youtube", "-Y", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 or "help" in result.stdout


class TestTranslateErrors:
    def test_missing_input_file(self, tmp_path):
        args = ["nonexistent_caption.srt", str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_translate_command(args)

    def test_empty_caption_file(self, tmp_path):
        empty_file = tmp_path / "empty.srt"
        empty_file.write_text("", encoding="utf-8")
        args = [str(empty_file), str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_translate_command(args)
