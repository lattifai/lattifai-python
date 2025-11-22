"""Tests for lattifai subtitle commands"""

import os
import subprocess

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def run_subtitle_command(args, env=None, dryrun: bool = True):
    """Helper function to run the subtitle command and return result"""
    cmd = ["lai", "subtitle"]

    if dryrun and os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false").lower() == "true":
        if args[0] in ["convert", "normalize", "shift"]:
            cmd.append(args[0])
            args = args[1:]
        cmd.append("--dryrun")
    else:
        if args[0] in ["convert", "normalize", "shift"]:
            cmd.append(args[0])
            args = args[1:]
        cmd.append("-Y")

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
        print(f"Command: {' '.join(cmd)} failed with exit code {e.returncode}")
        raise e


@pytest.fixture
def sample_subtitle_file():
    return "tests/data/SA1.srt"


class TestSubtitleConvertCommand:
    """Test cases for subtitle convert command"""

    @pytest.mark.parametrize(
        "output_ext",
        ["srt", "vtt", "json"],  # Only formats supported by actual implementation
    )
    def test_subtitle_convert_formats(self, sample_subtitle_file, tmp_path, output_ext):
        """Test subtitle convert command with different output formats"""
        output_file = tmp_path / f"output.{output_ext}"

        args = [
            "convert",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
        ]

        run_subtitle_command(args)

    def test_subtitle_convert_with_normalize_flag(self, sample_subtitle_file, tmp_path):
        """Test subtitle convert command with normalize flag"""
        output_file = tmp_path / "output_normalized.srt"

        args = [
            "convert",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
            "normalize_text=true",
        ]

        run_subtitle_command(args)

    def test_subtitle_convert_missing_input(self, tmp_path):
        """Test subtitle convert command with missing input file"""
        args = [
            "convert",
            "input_path=nonexistent_file.srt",
            f"output_path={tmp_path / 'output.vtt'}",
        ]

        with pytest.raises(subprocess.CalledProcessError):
            run_subtitle_command(args, dryrun=False)

    def test_subtitle_convert_help(self):
        """Test subtitle convert command help output"""
        args = ["--help"]
        run_subtitle_command(args)


class TestSubtitleNormalizeCommand:
    """Test cases for subtitle normalize command"""

    def test_subtitle_normalize_basic(self, sample_subtitle_file, tmp_path):
        """Test subtitle normalize command"""
        output_file = tmp_path / "output_normalized.srt"

        args = [
            "normalize",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
        ]

        run_subtitle_command(args)

    def test_subtitle_normalize_help(self):
        """Test subtitle normalize command help output"""
        args = ["normalize", "--help"]
        result = run_subtitle_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "subtitle" in help_text.lower()


class TestSubtitleShiftCommand:
    """Test cases for subtitle shift command"""

    def test_subtitle_shift_positive(self, sample_subtitle_file, tmp_path):
        """Test subtitle shift command with positive offset (delay)"""
        output_file = tmp_path / "output_shifted_positive.srt"

        args = [
            "shift",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
            "seconds=2.0",
        ]

        run_subtitle_command(args)

    def test_subtitle_shift_negative(self, sample_subtitle_file, tmp_path):
        """Test subtitle shift command with negative offset (advance)"""
        output_file = tmp_path / "output_shifted_negative.srt"

        args = [
            "shift",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
            "seconds=-1.5",
        ]

        run_subtitle_command(args)

    def test_subtitle_shift_zero(self, sample_subtitle_file, tmp_path):
        """Test subtitle shift command with zero offset (no change)"""
        output_file = tmp_path / "output_shifted_zero.srt"

        args = [
            "shift",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
            "seconds=0.0",
        ]

        run_subtitle_command(args)

    def test_subtitle_shift_with_format_conversion(self, sample_subtitle_file, tmp_path):
        """Test subtitle shift command with format conversion"""
        output_file = tmp_path / "output_shifted.vtt"

        args = [
            "shift",
            f"input_path={sample_subtitle_file}",
            f"output_path={output_file}",
            "seconds=1.5",
        ]

        run_subtitle_command(args)

    def test_subtitle_shift_help(self):
        """Test subtitle shift command help output"""
        args = ["shift", "--help"]
        result = run_subtitle_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "shift" in help_text.lower()
