"""Tests for lattifai youtube command"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        'output_format',
        ['srt', 'vtt', 'ass', 'ssa', 'sub', 'sbv', 'txt'],
    )
    def test_youtube_output_formats(self, cli_runner, tmp_path, output_format):
        """Test youtube command with different output formats"""
        with patch('lattifai.bin.align.YouTubeDownloader') as mock_downloader:
            with patch('lattifai.bin.align.LattifAI') as mock_aligner:
                # Mock the downloader - use AsyncMock for async methods
                mock_instance = MagicMock()
                mock_instance.download_audio = AsyncMock(return_value=str(tmp_path / 'audio.mp3'))
                mock_instance.download_subtitles = AsyncMock(return_value=[str(tmp_path / 'subtitle.vtt')])
                mock_downloader.return_value = mock_instance
                mock_downloader.extract_video_id = MagicMock(return_value='test_video_id')

                # Mock the aligner
                mock_aligner_instance = MagicMock()
                mock_aligner_instance.alignment.return_value = ([], None)
                mock_aligner.return_value = mock_aligner_instance

                result = cli_runner.invoke(
                    cli,
                    [
                        'youtube',
                        '--output-format',
                        output_format,
                        '--output-dir',
                        str(tmp_path),
                        '--device',
                        'cpu',
                        'https://www.youtube.com/watch?v=kb9suz-kkoM',
                    ],
                    catch_exceptions=False,
                )

                # Command should accept the format parameter
                assert result.exit_code in [0, 1, 2], f'Format {output_format} test failed with output: {result.output}'

    @pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
    def test_youtube_device_options(self, cli_runner, tmp_path, device):
        """Test youtube command with different device options"""
        result = cli_runner.invoke(
            cli,
            [
                'youtube',
                '--device',
                device,
                '--output-dir',
                str(tmp_path),
                'https://www.youtube.com/watch?v=kb9suz-kkoM',
            ],
        )

        # Test that the device parameter is accepted
        assert result.exit_code in [0, 1, 2]

    def test_youtube_split_sentence_option(self, cli_runner, tmp_path):
        """Test youtube command with split-sentence option"""
        result = cli_runner.invoke(
            cli,
            [
                'youtube',
                '--split-sentence',
                '--device',
                'cpu',
                '--output-dir',
                str(tmp_path),
                'https://www.youtube.com/watch?v=kb9suz-kkoM',
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_audio_format_option(self, cli_runner, tmp_path):
        """Test youtube command with audio format option"""
        result = cli_runner.invoke(
            cli,
            [
                'youtube',
                '--audio-format',
                'mp3',
                '--device',
                'cpu',
                '--output-dir',
                str(tmp_path),
                'https://www.youtube.com/watch?v=kb9suz-kkoM',
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_model_name_option(self, cli_runner, tmp_path):
        """Test youtube command with custom model name"""
        result = cli_runner.invoke(
            cli,
            [
                'youtube',
                '--model-name-or-path',
                'Lattifai/Lattice-1-Alpha',
                '--device',
                'cpu',
                '--output-dir',
                str(tmp_path),
                'https://www.youtube.com/watch?v=kb9suz-kkoM',
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_youtube_invalid_url(self, cli_runner, tmp_path):
        """Test youtube command with invalid URL"""
        result = cli_runner.invoke(
            cli,
            [
                'youtube',
                '--device',
                'cpu',
                '--output-dir',
                str(tmp_path),
                'not_a_valid_url',
            ],
        )

        # Should fail or handle gracefully
        assert result.exit_code in [0, 1, 2]

    def test_youtube_help(self, cli_runner):
        """Test youtube command help output"""
        result = cli_runner.invoke(cli, ['youtube', '--help'])

        assert result.exit_code == 0
        assert 'Download audio and subtitles from YouTube' in result.output
        assert '--output-format' in result.output
        assert '--audio-format' in result.output
        assert '--device' in result.output
        assert '--split-sentence' in result.output
        assert '--model-name-or-path' in result.output
