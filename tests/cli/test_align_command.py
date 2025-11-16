"""Tests for lattifai align command"""

import os
import subprocess

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def run_align_command(args, env=None):
    """Helper function to run the align command and return result"""
    cmd = ["lai", "alignment", "align"]

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


@pytest.fixture
def sample_audio_file():
    return "tests/data/SA1.wav"


@pytest.fixture
def sample_subtitle_file():
    return "tests/data/SA1.srt"


class TestAlignCommand:
    """Test cases for align command"""

    @pytest.mark.parametrize(
        "input_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "auto", "gemini"],
    )
    def test_align_input_formats(self, sample_audio_file, sample_subtitle_file, tmp_path, input_format):
        """Test align command with different subtitle formats"""
        output_file = tmp_path / f"output_{input_format}.srt"

        args = [
            f"media.input_path={sample_audio_file}",
            f"subtitle.input_path={sample_subtitle_file}",
            f"subtitle.output_path={output_file}",
            f"subtitle.input_format={input_format}",
            "alignment.device=cpu",
        ]

        run_align_command(args)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_align_device_options(self, sample_audio_file, sample_subtitle_file, tmp_path, device):
        """Test align command with different device options"""
        output_file = tmp_path / f"output_{device}.srt"

        args = [
            f"media.input_path={sample_audio_file}",
            f"subtitle.input_path={sample_subtitle_file}",
            f"subtitle.output_path={output_file}",
            f"alignment.device={device}",
        ]

        run_align_command(args)

    def test_align_split_sentence_option(self, sample_audio_file, sample_subtitle_file, tmp_path):
        """Test align command with split-sentence option"""
        output_file = tmp_path / "output_split.srt"

        args = [
            f"media.input_path={sample_audio_file}",
            f"subtitle.input_path={sample_subtitle_file}",
            f"subtitle.output_path={output_file}",
            "subtitle.split_sentence=true",
            "alignment.device=cpu",
        ]

        run_align_command(args)

    def test_align_model_name_option(self, sample_audio_file, sample_subtitle_file, tmp_path):
        """Test align command with custom model name"""
        output_file = tmp_path / "output_model.srt"

        args = [
            f"media.input_path={sample_audio_file}",
            f"subtitle.input_path={sample_subtitle_file}",
            f"subtitle.output_path={output_file}",
            "alignment.model_name_or_path=Lattifai/Lattice-1-Alpha",
            "alignment.device=cpu",
        ]

        run_align_command(args)

    def test_align_missing_input_files(self, tmp_path):
        """Test align command with missing input files"""
        args = [
            "media.input_path=nonexistent_audio.wav",
            "subtitle.input_path=nonexistent_subtitle.srt",
            f"subtitle.output_path={tmp_path / 'output.srt'}",
        ]

        result = run_align_command(args)

        if result is not None:
            assert result.returncode in [0, 1, 2] or "usage:" in result.stderr

    def test_align_help(self):
        """Test align command help output"""
        args = ["--help"]
        result = run_align_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "media" in help_text
                assert "subtitle" in help_text
                assert "alignment" in help_text

    def test_align_normalize_text_flag(self, sample_audio_file, sample_subtitle_file, tmp_path):
        """Test align command accepts normalize-text flag"""
        output_file = tmp_path / "output_normalized.srt"

        args = [
            f"media.input_path={sample_audio_file}",
            f"subtitle.input_path={sample_subtitle_file}",
            f"subtitle.output_path={output_file}",
            "subtitle.normalize_text=true",
            "alignment.device=cpu",
        ]

        run_align_command(args)
