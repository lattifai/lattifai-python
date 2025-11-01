"""Integration tests for all CLI commands"""

import pytest
from click.testing import CliRunner

from lattifai.bin.cli_base import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner"""
    return CliRunner()


class TestCLIIntegration:
    """Integration tests for CLI commands"""

    def test_main_help(self, cli_runner):
        """Test main CLI help"""
        result = cli_runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'lattifai' in result.output.lower()
        assert 'align' in result.output
        assert 'youtube' in result.output
        assert 'agent' in result.output
        assert 'subtitle' in result.output

    def test_all_commands_have_help(self, cli_runner):
        """Test that all commands have help"""
        commands = ['align', 'youtube', 'agent', 'subtitle']

        for command in commands:
            result = cli_runner.invoke(cli, [command, '--help'])
            assert result.exit_code == 0, f'Command {command} --help failed'
            assert '--help' in result.output

    def test_all_format_options_are_valid(self, cli_runner):
        """Test that all format options are properly defined"""
        # Test align subtitle formats
        result = cli_runner.invoke(cli, ['align', '--help'])
        assert '--subtitle-format' in result.output
        assert 'srt' in result.output
        assert 'vtt' in result.output
        assert 'ass' in result.output
        assert 'ssa' in result.output
        assert 'sub' in result.output
        assert 'sbv' in result.output
        assert 'txt' in result.output
        assert 'auto' in result.output
        assert 'gemini' in result.output

        # Test youtube output formats
        result = cli_runner.invoke(cli, ['youtube', '--help'])
        assert '--output-format' in result.output
        assert 'srt' in result.output
        assert 'vtt' in result.output
        assert 'ass' in result.output
        assert 'ssa' in result.output
        assert 'sub' in result.output
        assert 'sbv' in result.output
        assert 'txt' in result.output

        # Test agent output formats
        result = cli_runner.invoke(cli, ['agent', '--help'])
        assert '--output-format' in result.output
        assert '--media-format' in result.output

        # Test subtitle download formats
        result = cli_runner.invoke(cli, ['subtitle', 'download', '--help'])
        assert '--output-format' in result.output
        assert 'best' in result.output

    def test_device_options_are_valid(self, cli_runner):
        """Test that device options are properly defined"""
        commands_with_device = ['align', 'youtube']

        for command in commands_with_device:
            result = cli_runner.invoke(cli, [command, '--help'])
            assert '--device' in result.output
            assert 'cpu' in result.output
            assert 'cuda' in result.output
            assert 'mps' in result.output

    def test_split_sentence_option_exists(self, cli_runner):
        """Test that split-sentence option exists in relevant commands"""
        commands_with_split = ['align', 'youtube', 'agent']

        for command in commands_with_split:
            result = cli_runner.invoke(cli, [command, '--help'])
            assert '--split-sentence' in result.output

    def test_model_name_option_exists(self, cli_runner):
        """Test that model-name-or-path option exists in relevant commands"""
        commands_with_model = ['align', 'youtube']

        for command in commands_with_model:
            result = cli_runner.invoke(cli, [command, '--help'])
            assert '--model-name-or-path' in result.output

    def test_api_key_option_exists(self, cli_runner):
        """Test that api-key option exists in relevant commands"""
        # align and youtube commands
        result = cli_runner.invoke(cli, ['align', '--help'])
        assert '--api-key' in result.output

        result = cli_runner.invoke(cli, ['youtube', '--help'])
        assert '--api-key' in result.output

        # agent command uses gemini-api-key
        result = cli_runner.invoke(cli, ['agent', '--help'])
        assert '--gemini-api-key' in result.output

    def test_output_dir_option_exists(self, cli_runner):
        """Test that output-dir option exists in relevant commands"""
        commands_with_output_dir = ['youtube', 'agent']

        for command in commands_with_output_dir:
            result = cli_runner.invoke(cli, [command, '--help'])
            assert '--output-dir' in result.output

        # subtitle download also has output-dir
        result = cli_runner.invoke(cli, ['subtitle', 'download', '--help'])
        assert '--output-dir' in result.output

    def test_no_json_csv_tsv_formats(self, cli_runner):
        """Test that csv and tsv formats are not in output formats (json is allowed)"""
        commands_to_check = [
            ['align', '--help'],
            ['youtube', '--help'],
            ['agent', '--help'],
        ]

        for command_args in commands_to_check:
            result = cli_runner.invoke(cli, command_args)
            # Check that they're not in the choice list (but might appear in other text)
            # We look for the pattern [srt|vtt|...] to ensure they're not in choices
            if '--output-format' in result.output:
                # Ensure csv, tsv are not in the format choices (json is allowed for programmatic access)
                format_section = result.output[result.output.find('--output-format') :]
                format_section = format_section[: format_section.find('\n', 200)]
                # JSON is allowed for programmatic use cases
                assert 'csv' not in format_section
                assert 'tsv' not in format_section
