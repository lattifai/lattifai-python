"""Tests for lattifai align command"""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from lattifai.bin.align import align


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


@pytest.fixture
def sample_audio_file():
    return 'tests/data/SA1.wav'


@pytest.fixture
def sample_subtitle_file():
    return 'tests/data/SA1.srt'


class TestAlignCommand:
    """Test cases for align command"""

    @pytest.mark.parametrize(
        'input_format',
        ['srt', 'vtt', 'ass', 'ssa', 'sub', 'sbv', 'txt', 'auto', 'gemini'],
    )
    def test_align_input_formats(self, cli_runner, sample_audio_file, sample_subtitle_file, tmp_path, input_format):
        """Test align command with different subtitle formats"""
        output_file = tmp_path / f'output_{input_format}.srt'

        result = cli_runner.invoke(
            align,
            [
                '--input-format',
                input_format,
                '--device',
                'cpu',
                sample_audio_file,
                sample_subtitle_file,
                str(output_file),
            ],
            catch_exceptions=True,  # Catch exceptions to test CLI parameter acceptance
        )

        # Test that the CLI accepts the format parameter
        # Exit code 0 = success, 1 = error (expected for unsupported formats with test data)
        # Exit code 2 = CLI usage error (would indicate parameter not accepted)
        assert result.exit_code in [0, 1], f'Format {input_format} not accepted by CLI (exit code: {result.exit_code})'

    @pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
    def test_align_device_options(self, cli_runner, sample_audio_file, sample_subtitle_file, tmp_path, device):
        """Test align command with different device options"""
        output_file = tmp_path / f'output_{device}.srt'

        result = cli_runner.invoke(
            align,
            [
                '--device',
                device,
                sample_audio_file,
                sample_subtitle_file,
                str(output_file),
            ],
            catch_exceptions=True,  # Catch exceptions for device availability issues
        )

        # Test that the device parameter is accepted
        # May fail with exit code 1 if device is not available
        assert result.exit_code in [0, 1], f'Device {device} not accepted by CLI (exit code: {result.exit_code})'

    def test_align_split_sentence_option(self, cli_runner, sample_audio_file, sample_subtitle_file, tmp_path):
        """Test align command with split-sentence option"""
        output_file = tmp_path / 'output_split.srt'

        result = cli_runner.invoke(
            align,
            [
                '--split-sentence',
                '--device',
                'cpu',
                sample_audio_file,
                sample_subtitle_file,
                str(output_file),
            ],
            catch_exceptions=True,
        )

        assert result.exit_code in [0, 1], f'Split-sentence option not accepted (exit code: {result.exit_code})'

    def test_align_model_name_option(self, cli_runner, sample_audio_file, sample_subtitle_file, tmp_path):
        """Test align command with custom model name"""
        output_file = tmp_path / 'output_model.srt'

        result = cli_runner.invoke(
            align,
            [
                '--model-name-or-path',
                'Lattifai/Lattice-1-Alpha',
                '--device',
                'cpu',
                sample_audio_file,
                sample_subtitle_file,
                str(output_file),
            ],
            catch_exceptions=True,
        )

        assert result.exit_code in [0, 1], f'Model name option not accepted (exit code: {result.exit_code})'

    def test_align_missing_input_files(self, cli_runner, tmp_path):
        """Test align command with missing input files"""
        result = cli_runner.invoke(
            align,
            [
                'nonexistent_audio.wav',
                'nonexistent_subtitle.srt',
                str(tmp_path / 'output.srt'),
            ],
        )

        # Should fail with file not found error
        assert result.exit_code != 0

    def test_align_help(self, cli_runner):
        """Test align command help output"""
        result = cli_runner.invoke(align, ['--help'])

        assert result.exit_code == 0
        assert 'Command used to align media(audio/video) with subtitles' in result.output
        assert '--input-format' in result.output
        assert '--device' in result.output
        assert '--split-sentence' in result.output
        assert '--model-name-or-path' in result.output
