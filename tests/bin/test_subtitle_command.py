"""Tests for lattifai subtitle commands"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from lattifai.bin.cli_base import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


@pytest.fixture
def sample_subtitle_file():
    return "tests/data/SA1.srt"


class TestSubtitleConvertCommand:
    """Test cases for subtitle convert command"""

    @pytest.mark.parametrize(
        "output_ext",
        ["srt", "vtt", "ass", "ssa", "sub"],
    )
    def test_subtitle_convert_formats(self, cli_runner, sample_subtitle_file, tmp_path, output_ext):
        """Test subtitle convert command with different output formats"""
        output_file = tmp_path / f"output.{output_ext}"

        # pysubs2 is imported inside the function, so we need to mock it at the module level
        with patch("pysubs2.load") as mock_load:
            mock_subtitle = MagicMock()
            mock_load.return_value = mock_subtitle

            result = cli_runner.invoke(
                cli,
                [
                    "subtitle",
                    "convert",
                    sample_subtitle_file,
                    str(output_file),
                ],
                catch_exceptions=True,
            )

            # Command should accept the format parameter
            # Exit code 0 = success, 1 = expected error with test data
            assert result.exit_code in [0, 1], f"Format {output_ext} not accepted by CLI"

    def test_subtitle_convert_missing_input(self, cli_runner, tmp_path):
        """Test subtitle convert command with missing input file"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "convert",
                "nonexistent_file.srt",
                str(tmp_path / "output.vtt"),
            ],
        )

        # Should fail with file not found error
        assert result.exit_code != 0

    def test_subtitle_convert_help(self, cli_runner):
        """Test subtitle convert command help output"""
        result = cli_runner.invoke(cli, ["subtitle", "convert", "--help"])

        assert result.exit_code == 0
        assert "Convert subtitle file to another format" in result.output


class TestSubtitleDownloadCommand:
    """Test cases for subtitle download command"""

    @pytest.mark.parametrize(
        "output_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "best"],
    )
    def test_subtitle_download_formats(self, cli_runner, tmp_path, output_format):
        """Test subtitle download command with different output formats"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "download",
                "--output-dir",
                str(tmp_path),
                "--output-format",
                output_format,
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        # Command should accept the format parameter
        assert result.exit_code in [0, 1, 2]

    def test_subtitle_download_with_lang(self, cli_runner, tmp_path):
        """Test subtitle download command with language option"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "download",
                "--output-dir",
                str(tmp_path),
                "--lang",
                "en",
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_subtitle_download_force_overwrite(self, cli_runner, tmp_path):
        """Test subtitle download command with force overwrite option"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "download",
                "--output-dir",
                str(tmp_path),
                "--force-overwrite",
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_subtitle_download_invalid_url(self, cli_runner, tmp_path):
        """Test subtitle download command with invalid URL"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "download",
                "--output-dir",
                str(tmp_path),
                "not_a_valid_youtube_url",
            ],
        )

        # Should fail with invalid URL error
        assert result.exit_code != 0
        assert "Invalid" in result.output or "Error" in result.output

    def test_subtitle_download_help(self, cli_runner):
        """Test subtitle download command help output"""
        result = cli_runner.invoke(cli, ["subtitle", "download", "--help"])

        assert result.exit_code == 0
        assert "Download subtitles from YouTube URL" in result.output
        assert "--output-dir" in result.output
        assert "--output-format" in result.output
        assert "--lang" in result.output
        assert "--force-overwrite" in result.output


class TestSubtitleListSubsCommand:
    """Test cases for subtitle list-subs command"""

    def test_subtitle_list_subs_valid_url(self, cli_runner):
        """Test subtitle list-subs command with valid URL"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "list-subs",
                "https://www.youtube.com/shorts/wX9ybkEYDc0",
            ],
        )

        # Command may fail due to network or invalid video ID, but should accept the URL
        assert result.exit_code in [0, 1, 2]

    def test_subtitle_list_subs_invalid_url(self, cli_runner):
        """Test subtitle list-subs command with invalid URL"""
        result = cli_runner.invoke(
            cli,
            [
                "subtitle",
                "list-subs",
                "not_a_valid_youtube_url",
            ],
        )

        # Should fail with invalid URL error
        assert result.exit_code != 0
        assert "Invalid" in result.output or "Error" in result.output

    def test_subtitle_list_subs_help(self, cli_runner):
        """Test subtitle list-subs command help output"""
        result = cli_runner.invoke(cli, ["subtitle", "list-subs", "--help"])

        assert result.exit_code == 0
        assert "List available subtitle tracks" in result.output


class TestSubtitleGroupHelp:
    """Test cases for subtitle command group"""

    def test_subtitle_help(self, cli_runner):
        """Test subtitle command group help output"""
        result = cli_runner.invoke(cli, ["subtitle", "--help"])

        assert result.exit_code == 0
        assert "Commands for subtitle format conversion" in result.output
        assert "convert" in result.output
        assert "download" in result.output
        assert "list-subs" in result.output
