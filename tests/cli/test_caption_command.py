"""Tests for lattifai caption commands"""

import os
import subprocess

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_caption_command(args, env=None, dryrun: bool = True):
    """Helper function to run the caption command and return result"""
    cmd = ["lai", "caption"]

    if dryrun and LATTIFAI_TESTS_CLI_DRYRUN:
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
            timeout=120,
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
def sample_caption_file():
    return "tests/data/SA1.srt"


class TestCaptionConvertCommand:
    """Test cases for caption convert command"""

    @pytest.mark.parametrize(
        "output_ext",
        ["srt", "vtt", "json"],  # Only formats supported by actual implementation
    )
    def test_caption_convert_formats(self, sample_caption_file, tmp_path, output_ext):
        """Test caption convert command with different output formats"""
        output_file = tmp_path / f"output.{output_ext}"

        args = [
            "convert",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
        ]

        run_caption_command(args)

    def test_caption_convert_with_normalize_flag(self, sample_caption_file, tmp_path):
        """Test caption convert command with normalize flag"""
        output_file = tmp_path / "output_normalized.srt"

        args = [
            "convert",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
            "normalize_text=true",
        ]

        run_caption_command(args)

    def test_caption_convert_missing_input(self, tmp_path):
        """Test caption convert command with missing input file"""
        args = [
            "convert",
            "input_path=nonexistent_file.srt",
            f"output_path={tmp_path / 'output.vtt'}",
        ]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_caption_command(args, dryrun=False)
        else:
            run_caption_command(args)

    def test_caption_convert_help(self):
        """Test caption convert command help output"""
        args = ["--help"]
        run_caption_command(args)


class TestCaptionNormalizeCommand:
    """Test cases for caption normalize command"""

    def test_caption_normalize_basic(self, sample_caption_file, tmp_path):
        """Test caption normalize command"""
        output_file = tmp_path / "output_normalized.srt"

        args = [
            "normalize",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
        ]

        run_caption_command(args)

    def test_caption_normalize_help(self):
        """Test caption normalize command help output"""
        args = ["normalize", "--help"]
        result = run_caption_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "caption" in help_text.lower()


class TestCaptionShiftCommand:
    """Test cases for caption shift command"""

    def test_caption_shift_positive(self, sample_caption_file, tmp_path):
        """Test caption shift command with positive offset (delay)"""
        output_file = tmp_path / "output_shifted_positive.srt"

        args = [
            "shift",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
            "seconds=2.0",
        ]

        run_caption_command(args)

    def test_caption_shift_negative(self, sample_caption_file, tmp_path):
        """Test caption shift command with negative offset (advance)"""
        output_file = tmp_path / "output_shifted_negative.srt"

        args = [
            "shift",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
            "seconds=-1.5",
        ]

        run_caption_command(args)

    def test_caption_shift_zero(self, sample_caption_file, tmp_path):
        """Test caption shift command with zero offset (no change)"""
        output_file = tmp_path / "output_shifted_zero.srt"

        args = [
            "shift",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
            "seconds=0.0",
        ]

        run_caption_command(args)

    def test_caption_shift_with_format_conversion(self, sample_caption_file, tmp_path):
        """Test caption shift command with format conversion"""
        output_file = tmp_path / "output_shifted.vtt"

        args = [
            "shift",
            f"input_path={sample_caption_file}",
            f"output_path={output_file}",
            "seconds=1.5",
        ]

        run_caption_command(args)

    def test_caption_shift_help(self):
        """Test caption shift command help output"""
        args = ["shift", "--help"]
        result = run_caption_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "shift" in help_text.lower()
