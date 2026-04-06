"""Tests for lattifai youtube command"""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_youtube_command(args, env=None):
    """Helper function to run the youtube command and return result"""
    cmd = ["lai", "youtube", "align", "-Y"]

    if LATTIFAI_TESTS_CLI_DRYRUN:
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


def run_youtube_download_command(args, env=None):
    """Helper function to run the youtube download command and return result."""
    cmd = ["lai", "youtube", "download", "-Y"]

    if LATTIFAI_TESTS_CLI_DRYRUN:
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
            f"caption.output.format={output_format}",
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

        if device == "mps" and not torch.backends.mps.is_available() and not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                _ = run_youtube_command(args)
        elif device == "cuda" and not torch.cuda.is_available() and not LATTIFAI_TESTS_CLI_DRYRUN:
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
            "caption.input.split_sentence=true",
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
            "caption.input.path=dummy.srt",
        ]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                _ = run_youtube_command(args)
        else:
            _ = run_youtube_command(args)


class TestYoutubeDownloadCommand:
    """Test cases for youtube download command."""

    def test_youtube_download_help(self):
        result = subprocess.run(
            ["lai", "youtube", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_youtube_download_invalid_only(self, tmp_path):
        args = [
            "https://www.youtube.com/watch?v=kb9suz-kkoM",
            "only=invalid_value",
            f"media.output_dir={tmp_path}",
        ]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_youtube_download_command(args)

    def test_youtube_download_only_media(self, tmp_path):
        from lattifai.cli.youtube import youtube_download
        from lattifai.config import MediaConfig

        media = MediaConfig(input_path="https://www.youtube.com/watch?v=kb9suz-kkoM", output_dir=str(tmp_path))
        loop = Mock()
        loop.run_until_complete.side_effect = [
            {
                "title": "Video",
                "duration": 12,
                "uploader": "Uploader",
                "upload_date": "20250101",
                "view_count": 1,
                "description": "",
                "thumbnail": "",
                "webpage_url": media.input_path,
                "chapters": [],
            },
            str(tmp_path / "audio.mp3"),
        ]
        downloader = Mock()
        downloader.extract_video_id.return_value = "video123"
        downloader.download_media = Mock(return_value=str(tmp_path / "audio.mp3"))

        with (
            patch("asyncio.get_event_loop", return_value=loop),
            patch("lattifai.youtube.client.YouTubeDownloader", return_value=downloader),
        ):
            result = youtube_download(media=media, only="media")

        assert result == str(tmp_path / "audio.mp3")
        downloader.download_media.assert_called_once()
        downloader.download_captions.assert_not_called()

    def test_youtube_download_only_caption(self, tmp_path):
        from lattifai.cli.youtube import youtube_download
        from lattifai.config import MediaConfig

        media = MediaConfig(input_path="https://www.youtube.com/watch?v=kb9suz-kkoM", output_dir=str(tmp_path))
        loop = Mock()
        loop.run_until_complete.side_effect = [
            {
                "title": "Video",
                "duration": 12,
                "uploader": "Uploader",
                "upload_date": "20250101",
                "view_count": 1,
                "description": "",
                "thumbnail": "",
                "webpage_url": media.input_path,
                "chapters": [],
            },
            str(tmp_path / "captions.srt"),
        ]
        downloader = Mock()
        downloader.extract_video_id.return_value = "video123"
        downloader.download_captions = Mock(return_value=str(tmp_path / "captions.srt"))

        with (
            patch("asyncio.get_event_loop", return_value=loop),
            patch("lattifai.youtube.client.YouTubeDownloader", return_value=downloader),
        ):
            result = youtube_download(media=media, only="caption", source_lang="en")

        assert result == str(tmp_path / "captions.srt")
        downloader.download_media.assert_not_called()
        downloader.download_captions.assert_called_once()
        assert downloader.download_captions.call_args.kwargs["source_lang"] == "en"

    def test_youtube_download_only_meta(self, tmp_path):
        from lattifai.cli.youtube import youtube_download
        from lattifai.config import MediaConfig

        media = MediaConfig(input_path="https://www.youtube.com/watch?v=kb9suz-kkoM", output_dir=str(tmp_path))
        info = {
            "title": "Video",
            "duration": 3723,
            "uploader": "Uploader",
            "upload_date": "20250101",
            "view_count": 42,
            "description": "Metadata body",
            "thumbnail": "thumb.jpg",
            "webpage_url": media.input_path,
            "chapters": [{"title": "Intro", "start_time": 0}],
            "channel": "Uploader",
        }
        loop = Mock()
        loop.run_until_complete.return_value = info
        downloader = Mock()
        downloader.extract_video_id.return_value = "video123"

        with (
            patch("asyncio.get_event_loop", return_value=loop),
            patch("lattifai.youtube.client.YouTubeDownloader", return_value=downloader),
        ):
            result = youtube_download(media=media, only="meta")

        meta_path = tmp_path / "video123.meta.md"
        assert result is None
        assert meta_path.exists()
        content = meta_path.read_text(encoding="utf-8")
        assert 'title: "Video"' in content
        assert 'duration: "01:02:03"' in content
        assert "Metadata body" in content

    def test_youtube_download_missing_url(self):
        from lattifai.cli.youtube import youtube_download

        with pytest.raises(ValueError, match="YouTube URL is required"):
            youtube_download()

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
            f"caption.input.source_lang={lang_code}",
            "media.force_overwrite=true",
            "alignment.device=cpu",
        ]

        if lang_code != "en" and not LATTIFAI_TESTS_CLI_DRYRUN:
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
            "caption.input.source_lang=en-US",  # not exist
            "media.force_overwrite=true",
            "alignment.device=cpu",
        ]

        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                _ = run_youtube_command(args)

            # Check exception details
            assert exc_info.value.returncode == 1
        else:
            _ = run_youtube_command(args)
