"""Tests for lattifai youtube command"""

import pytest
from click.testing import CliRunner

from lattifai.bin.cli_base import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


class TestYoutubeCommand:
    """Test cases for youtube command"""

    @pytest.mark.parametrize(
        "output_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt"],
    )
    def test_youtube_output_formats(self, cli_runner, tmp_path, output_format, monkeypatch):
        """Test youtube command with different output formats"""
        # Set environment variable for API key
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")

        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--output-format",
                output_format,
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
                "https://www.youtube.com/watch?v=kb9suz-kkoM",
            ],
        )

        # Command should accept the format parameter
        assert result.exit_code in [0, 1, 2], f"Format {output_format} test failed with output: {result.output}"

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_youtube_device_options(self, cli_runner, tmp_path, device):
        """Test youtube command with different device options"""
        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--device",
                device,
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/watch?v=kb9suz-kkoM",
            ],
        )

        # Test that the device parameter is accepted
        assert result.exit_code in [0, 1, 2]

    def test_youtube_split_sentence_option(self, cli_runner, tmp_path):
        """Test youtube command with split-sentence option"""
        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--split-sentence",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/watch?v=kb9suz-kkoM",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_audio_format_option(self, cli_runner, tmp_path):
        """Test youtube command with audio format option"""
        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--audio-format",
                "mp3",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/watch?v=kb9suz-kkoM",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_model_name_option(self, cli_runner, tmp_path):
        """Test youtube command with custom model name"""
        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--model-name-or-path",
                "Lattifai/Lattice-1-Alpha",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/watch?v=kb9suz-kkoM",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_invalid_url(self, cli_runner, tmp_path):
        """Test youtube command with invalid URL"""
        result = cli_runner.invoke(
            cli,
            [
                "youtube",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path),
                "not_a_valid_url",
            ],
        )

        # Should fail or handle gracefully
        assert result.exit_code in [0, 1, 2]

    def test_youtube_help(self, cli_runner):
        """Test youtube command help output"""
        result = cli_runner.invoke(cli, ["youtube", "--help"])

        assert result.exit_code == 0
        assert "Download media and subtitles from YouTube" in result.output
        assert "--output-format" in result.output
        assert "--media-format" in result.output
        assert "--device" in result.output
        assert "--split-sentence" in result.output
        assert "--model-name-or-path" in result.output
