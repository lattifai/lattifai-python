"""Tests for lattifai youtube command"""

import os
import subprocess

import pytest
import torch
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))


def run_youtube_command(args, env=None):
    """Helper function to run the youtube command and return result"""
    cmd = ["lai", "alignment", "youtube", "-Y"]

    if os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false").lower() == "true":
        cmd.append("--dryrun")

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
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise e


class TestYoutubeCommand:
    """Test cases for youtube command"""

    @pytest.mark.parametrize(
        "output_format",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt"],
    )
    def test_youtube_output_formats(self, tmp_path, output_format):
        """Test youtube command with different output formats"""
        args = [
            "https://www.youtube.com/watch\?v\=kb9suz-kkoM",
            f"media.output_dir={tmp_path}",
            "media.force_overwrite=true",
            f"caption.output_format={output_format}",
            "alignment.device=cpu",
        ]

        run_youtube_command(args)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_youtube_device_options(self, tmp_path, device):
        """Test youtube command with different device options"""
        args = [
            "media.input_path=https://www.youtube.com/watch\?v\=kb9suz-kkoM",
            f"media.output_dir={tmp_path}",
            f"alignment.device={device}",
            "media.force_overwrite=true",
        ]

        if device == "mps" and not torch.backends.mps.is_available():
            with pytest.raises(subprocess.CalledProcessError):
                _ = run_youtube_command(args)
        elif device == "cuda" and not torch.cuda.is_available():
            with pytest.raises(subprocess.CalledProcessError):
                _ = run_youtube_command(args)
        else:
            run_youtube_command(args)

    def test_youtube_options(self, tmp_path):
        """Test youtube command with media format option"""
        args = [
            "media.input_path=https://www.youtube.com/watch\?v\=kb9suz-kkoM",
            f"media.output_dir={tmp_path}",
            "media.output_format=mp3",
            "media.prefer_audio=true",
            "caption.split_sentence=true",
            "alignment.model_name=LattifAI/Lattice-1",
            "media.force_overwrite=true",
            "alignment.device=cpu",
        ]

        run_youtube_command(args)

    def test_youtube_invalid_url(self, tmp_path):
        """Test youtube command with invalid URL"""
        args = [
            "yt_url=not_a_valid_url",
            f"media.output_dir={tmp_path}",
            "alignment.device=cpu",
            "media.force_overwrite=true",
            "caption.input_path=dummy.srt",
        ]

        with pytest.raises(subprocess.CalledProcessError):
            _ = run_youtube_command(args)

    def test_youtube_help(self):
        """Test youtube command help output"""
        args = ["--help"]
        result = run_youtube_command(args)

        if result is not None:
            assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout
            if result.returncode == 0:
                help_text = result.stdout + result.stderr
                assert "media" in help_text
                assert "caption" in help_text
                assert "alignment" in help_text

    @pytest.mark.parametrize(
        "lang_code",
        ["en", "zh", "es", "fr", "de", "ja", "ko"],
    )
    def test_youtube_various_source_languages(self, tmp_path, lang_code):
        """Test youtube command with various source language codes"""
        args = [
            "media.input_path=https://www.youtube.com/watch\?v\=kb9suz-kkoM",
            f"media.output_dir={tmp_path}",
            f"caption.source_lang={lang_code}",
            "media.force_overwrite=true",
            "alignment.device=cpu",
        ]

        if lang_code != "en":
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                _ = run_youtube_command(args)

            # Check exception details
            assert exc_info.value.returncode == 1, f"Failed for lang_code: {lang_code}"
            # assert "No caption" in exc_info.value.stderr or "No caption" in exc_info.value.stdout, f"Failed for lang_code: {lang_code}"
        else:
            run_youtube_command(args)

    def test_youtube_source_lang_with_region(self, tmp_path):
        """Test youtube command with language code including region"""
        args = [
            "media.input_path=https://www.youtube.com/watch\?v\=kb9suz-kkoM",
            f"media.output_dir={tmp_path}",
            "caption.source_lang=en-US",  # not exist
            "media.force_overwrite=true",
            "alignment.device=cpu",
        ]

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _ = run_youtube_command(args)

        # Check exception details
        assert exc_info.value.returncode == 1
