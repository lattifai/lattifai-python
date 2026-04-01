"""Tests for lai diarization CLI command."""

import os
import subprocess

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_diarize_command(args, env=None):
    """Helper to run the diarization command and return result."""
    cmd = ["lai", "diarize", "run", "-Y"]
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


class TestDiarizeHelp:
    def test_diarize_help(self):
        result = subprocess.run(
            ["lai", "diarize", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout


class TestDiarizeErrors:
    def test_missing_input_media(self, tmp_path):
        caption_file = tmp_path / "test.srt"
        caption_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        args = [
            "nonexistent_audio.wav",
            str(caption_file),
            str(tmp_path / "output.srt"),
        ]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_diarize_command(args)
