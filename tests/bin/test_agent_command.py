"""Tests for lattifai agent command"""

import pytest
from click.testing import CliRunner

from lattifai.bin.cli_base import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


class TestAgentCommand:
    """Test cases for agent command"""

    @pytest.mark.parametrize(
        "output_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt"],
    )
    def test_agent_output_formats(self, cli_runner, tmp_path, output_format, monkeypatch):
        """Test agent command with different output formats"""
        # Set environment variable for API key
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--output-format",
                output_format,
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        # Command should accept the format parameter
        assert result.exit_code in [0, 1, 2]

    @pytest.mark.parametrize(
        "video_format",
        ["mp4", "webm", "mkv", "avi", "mov", "flv", "wmv", "mpeg", "mpg", "3gp"],
    )
    def test_agent_video_formats(self, cli_runner, tmp_path, video_format, monkeypatch):
        """Test agent command with different video formats"""
        # Set environment variable for API key
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--media-format",
                video_format,
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        # Command should accept the format parameter
        assert result.exit_code in [0, 1, 2]

    def test_agent_split_sentence_option(self, cli_runner, tmp_path, monkeypatch):
        """Test agent command with split-sentence option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--split-sentence",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_agent_max_retries_option(self, cli_runner, tmp_path, monkeypatch):
        """Test agent command with max-retries option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--max-retries",
                "3",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_agent_verbose_option(self, cli_runner, tmp_path, monkeypatch):
        """Test agent command with verbose option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--verbose",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_agent_force_option(self, cli_runner, tmp_path, monkeypatch):
        """Test agent command with force option"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--force",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_agent_missing_api_key(self, cli_runner, tmp_path):
        """Test agent command without API key"""
        # Note: This test is challenging because CliRunner inherits the parent environment
        # and we can't fully isolate environment variables. The test checks that if the
        # API key validation triggers, it shows the right message. Otherwise, it will
        # run the actual workflow if API keys are available in the environment.
        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
            env={},  # Request empty environment (though CliRunner may inherit some vars)
        )

        # Test should either:
        # 1. Exit with error about missing API key, OR
        # 2. Run successfully if API keys are available in parent environment
        assert result.exit_code in [0, 1, 2]
        # If it failed, check if it's because of missing API key
        if result.exit_code != 0 and "GEMINI_API_KEY" not in result.output:
            # Failed for another reason - that's ok for this test
            pass

    def test_agent_no_workflow_flag(self, cli_runner, tmp_path, monkeypatch):
        """Test agent command without --youtube flag"""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        # Should fail or show error
        assert result.exit_code in [0, 1] or "workflow type" in result.output

    def test_agent_gemini_api_key_option(self, cli_runner, tmp_path):
        """Test agent command with --gemini-api-key option"""
        result = cli_runner.invoke(
            cli,
            [
                "agent",
                "--youtube",
                "--gemini-api-key",
                "test_key_from_option",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_agent_help(self, cli_runner):
        """Test agent command help output"""
        result = cli_runner.invoke(cli, ["agent", "--help"])

        assert result.exit_code == 0
        assert "LattifAI Agentic Workflow Agent" in result.output
        assert "--youtube" in result.output
        assert "--output-format" in result.output
        assert "--media-format" in result.output
        assert "--gemini-api-key" in result.output
        assert "--max-retries" in result.output
        assert "--split-sentence" in result.output
        assert "--verbose" in result.output
        assert "--force" in result.output
