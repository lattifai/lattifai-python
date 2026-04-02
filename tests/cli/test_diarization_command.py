"""Tests for lai diarization CLI command."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_diarize_command(args, env=None):
    """Helper to run the diarization command and return result."""
    cmd = ["lai", "diarize", "run", "-Y"]
    if LATTIFAI_TESTS_CLI_DRYRUN:
        cmd.append("--dryrun")
    cmd.extend(args)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True, env=env)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(f"Command: {' '.join(cmd)} failed with exit code {e.returncode}")
        raise e


class TestDiarizeHelp:
    def test_diarize_help(self):
        result = subprocess.run(
            ["lai", "diarize", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_diarize_help_run(self):
        result = subprocess.run(
            ["lai", "diarize", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout


class TestDiarizeErrors:
    def test_missing_input_media(self, tmp_path):
        caption_file = tmp_path / "test.srt"
        caption_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        args = [
            "nonexistent_audio.wav",
            str(caption_file),
            str(tmp_path / "output.srt"),
        ]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_diarize_command(args)

    def test_diarize_empty_caption_segments(self, tmp_path):
        from lattifai.cli.diarize import diarize
        from lattifai.config import CaptionConfig, MediaConfig

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        input_caption_path = tmp_path / "input.srt"
        input_caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        media = MediaConfig(input_path=str(tmp_path / "audio.wav"))
        caption = CaptionConfig(input_path=str(input_caption_path), output_path=str(tmp_path / "output.srt"))
        fake_client = Mock()
        fake_client.audio_loader = Mock(return_value="audio")
        fake_client._read_caption = Mock(return_value=SimpleNamespace(alignments=[], supervisions=[]))

        with patch("lattifai.cli.diarize.build_lattifai_client", return_value=fake_client):
            with pytest.raises(ValueError, match="Caption does not contain segments"):
                diarize(media=media, caption=caption)

    def test_diarize_infer_speakers_flag(self, tmp_path):
        from lattifai.cli.diarize import diarize
        from lattifai.config import CaptionConfig, DiarizationConfig, MediaConfig

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        input_caption_path = tmp_path / "input.srt"
        input_caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        media = MediaConfig(input_path=str(audio_path))
        caption = CaptionConfig(input_path=str(input_caption_path), output_path=str(tmp_path / "output.srt"))
        diarization_config = DiarizationConfig(infer_speakers=True)
        fake_caption = SimpleNamespace(alignments=[SimpleNamespace(text="hello")], supervisions=[])
        fake_client = Mock()
        fake_client.audio_loader = Mock(return_value="audio")
        fake_client._read_caption = Mock(return_value=fake_caption)
        fake_client.speaker_diarization = Mock(return_value="diarized")

        def _build_client(*, diarization=None, **kwargs):
            assert diarization is not None
            assert diarization.enabled is True
            assert diarization.infer_speakers is True
            return fake_client

        with patch("lattifai.cli.diarize.build_lattifai_client", side_effect=_build_client):
            result = diarize(media=media, caption=caption, diarization=diarization_config)

        assert result == "diarized"
        fake_client.speaker_diarization.assert_called_once()
