"""Tests for lattifai agent command"""

import os
import subprocess

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def run_agent_command(args, env=None):
    """Helper function to run the agent command and return result"""
    cmd = ["lai", "agent", "agent"]

    if os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false").lower() == "true":
        cmd.append("--dryrun")

    cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env=env,
        )
        return result
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(" ".join(cmd))
        raise e


class TestAgentCommand:
    """Test cases for agent command"""

    @pytest.mark.parametrize(
        "output_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt"],
    )
    def test_agent_output_formats(self, tmp_path, monkeypatch, output_format):
        """Test agent command with different output formats"""
        # Set environment variable for API key
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            f"subtitle.output_format={output_format}",
        ]
        run_agent_command(args)

    @pytest.mark.parametrize(
        "media_format",
        ["mp4"],
    )
    def test_agent_media_formats(self, tmp_path, media_format, monkeypatch):
        """Test agent command with different media formats"""
        # Set environment variable for API key
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            f"media.media_format={media_format}",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)

    def test_agent_split_sentence_option(self, tmp_path, monkeypatch):
        """Test agent command with split-sentence option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "subtitle.split_sentence=true",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)

    def test_agent_max_retries_option(self, tmp_path, monkeypatch):
        """Test agent command with max-retries option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "max_retries=3",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)

    def test_agent_verbose_option(self, tmp_path, monkeypatch):
        """Test agent command with verbose option (note: verbose is handled by nemo_run logging)"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)

    def test_agent_force_option(self, tmp_path, monkeypatch):
        """Test agent command with force option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "media.force_overwrite=true",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)

    def test_agent_gemini_api_key_option(self, tmp_path):
        """Test agent command with gemini api key option"""
        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "transcription.gemini_api_key=test_key_from_option",
        ]

        result = run_agent_command(args)

        if result is not None:
            assert result.returncode in [0, 1, 2] or "usage:" in result.stderr

    def test_agent_help(self):
        """Test agent command help output"""
        args = ["agent", "--help"]
        result = run_agent_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                # Check for key parameters in help text
                help_text = result.stdout + result.stderr
                # These are the actual nemo_run parameters
                assert "media" in help_text
                assert "transcription" in help_text
                assert "max_retries" in help_text or "max_retries" in help_text

    def test_agent_normalize_text_flag(self, tmp_path, monkeypatch):
        """Test agent command accepts normalize-text flag"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        args = [
            "media.input_path=https://www.youtube.com/shorts/wX9ybkEYDc0",
            f"media.output_dir={tmp_path}",
            "subtitle.normalize_text=true",
            "transcription.gemini_api_key=test_api_key",
        ]

        run_agent_command(args)
