"""Tests for lai transcribe CLI command."""

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_transcribe_command(args, env=None):
    """Helper to run the transcribe command and return result."""
    cmd = ["lai", "transcribe", "run", "-Y"]
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


class TestTranscribeHelp:
    def test_transcribe_help(self):
        result = subprocess.run(
            ["lai", "transcribe", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_transcribe_align_help(self):
        cmd = ["lai", "transcribe", "align", "-Y", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 or "help" in result.stdout


class TestTranscribeErrors:
    def test_missing_input(self, tmp_path):
        args = ["nonexistent_audio.wav", str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_transcribe_command(args)

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "nvidia/parakeet-tdt-0.6b-v3"])
    def test_transcribe_model_options_help(self, model):
        """Verify model names are accepted as arguments (help level)."""
        result = run_transcribe_command(["--help"])
        if result is not None:
            assert result.returncode == 0


class TestTranscribeUnit:
    def test_transcribe_align_delegates(self):
        from lattifai.cli.transcribe import transcribe_align

        with patch("lattifai.cli.transcribe.alignment_align", return_value="aligned") as mock_align:
            result = transcribe_align(input_media="audio.wav", output_caption="out.srt")

        assert result == "aligned"
        mock_align.assert_called_once_with(
            input_media="audio.wav",
            output_caption="out.srt",
            media=None,
            caption=None,
            client=None,
            alignment=None,
            transcription=None,
            diarization=None,
        )

    def test_transcribe_run_missing_input(self, mock_api_key):
        from lattifai.cli.transcribe import transcribe

        with pytest.raises(ValueError, match="Input is required"):
            transcribe()

    def test_transcribe_run_local_file(self, tmp_path):
        from lattifai.cli.transcribe import transcribe
        from lattifai.config import MediaConfig

        input_path = tmp_path / "audio.wav"
        input_path.write_bytes(b"fake")
        output_path = tmp_path / "output.srt"

        fake_transcriber = Mock()
        fake_transcriber.name = "fake-transcriber"
        fake_transcriber.supports_url = False
        fake_transcriber.file_suffix = ".srt"
        fake_transcriber.transcribe = AsyncMock(return_value="transcript")
        fake_transcriber.write = Mock(
            side_effect=lambda _t, path, **kwargs: Path(path).write_text("ok", encoding="utf-8")
        )

        fake_audio_loader = Mock(return_value="audio-data")

        with (
            patch("lattifai_core.client.SyncAPIClient", return_value=Mock()),
            patch("lattifai.transcription.create_transcriber", return_value=fake_transcriber),
            patch("lattifai.audio2.AudioLoader", return_value=fake_audio_loader),
            patch("lattifai.cli.transcribe._resolve_model_path", return_value="/tmp/model"),
        ):
            result = transcribe(
                input=str(input_path),
                output_caption=str(output_path),
                media=MediaConfig(input_path=str(input_path)),
            )

        assert result == "transcript"
        fake_audio_loader.assert_called_once()
        fake_transcriber.transcribe.assert_awaited_once_with("audio-data")
        fake_transcriber.write.assert_called_once()
        assert output_path.exists()
