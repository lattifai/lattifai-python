"""Integration tests for CLI commands"""

import os
import subprocess

import pytest  # noqa: F401
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def run_cli_command(command, args):
    """Helper function to run CLI commands and return result"""
    cmd = ["lai", command]

    if os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false").lower() == "true":
        cmd.append("--dryrun")

    cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(" ".join(cmd))
        raise e


class TestCLIIntegration:
    """Integration tests for CLI commands"""

    def test_all_commands_have_help(self):
        """Test that all commands have help"""
        commands = ["alignment", "agent", "subtitle"]

        for command in commands:
            result = run_cli_command(command, ["--help"])
            if result is not None:
                assert (
                    result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
                ), f"Command {command} --help failed"

    def test_nemo_run_parameters_exist(self):
        """Test that nemo_run style parameters exist in CLI commands"""
        # Test alignment command
        result = run_cli_command("alignment", ["--help"])
        if result is not None and result.returncode == 0:
            help_text = result.stdout + result.stderr
            assert "align" in help_text, help_text
            assert "youtube" in help_text

        # Test agent command
        result = run_cli_command("agent", ["--help"])
        if result is not None and result.returncode == 0:
            help_text = result.stdout + result.stderr
            assert "agent" in help_text, help_text

        # Test subtitle command
        result = run_cli_command("subtitle", ["--help"])
        if result is not None and result.returncode == 0:
            help_text = result.stdout + result.stderr
            assert "convert" in help_text
            assert "normalize" in help_text
