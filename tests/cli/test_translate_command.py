"""Tests for lai translate CLI command."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_translate_command(args, env=None):
    """Helper to run the translate command and return result."""
    cmd = ["lai", "translate", "caption", "-Y"]
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


class TestTranslateHelp:
    def test_translate_caption_help(self):
        result = subprocess.run(
            ["lai", "translate", "caption", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_translate_youtube_help(self):
        cmd = ["lai", "translate", "youtube", "-Y", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 or "help" in result.stdout


class TestTranslateErrors:
    def test_missing_input_file(self, tmp_path):
        args = ["nonexistent_caption.srt", str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_translate_command(args)

    def test_empty_caption_file(self, tmp_path):
        empty_file = tmp_path / "empty.srt"
        empty_file.write_text("", encoding="utf-8")
        args = [str(empty_file), str(tmp_path / "output.srt")]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_translate_command(args)

    def test_translate_youtube_missing_url(self):
        cmd = ["lai", "translate", "youtube", "-Y"]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)


class TestTranslateUnit:
    def test_translate_caption_success_mocked(self, tmp_path):
        from lattifai.cli.translate import translate
        from lattifai.config.caption import CaptionConfig
        from lattifai.config.translation import TranslationConfig

        input_path = tmp_path / "input.srt"
        input_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        output_path = tmp_path / "translated.srt"

        cap = Mock()
        cap.supervisions = [SimpleNamespace(text="Hello")]
        cap.source_path = str(input_path)

        def _write(path, **kwargs):
            Path(path).write_text("Translated content", encoding="utf-8")

        cap.write.side_effect = _write

        translator = Mock()
        translator.name = "fake:translator"
        translator.translate_captions = AsyncMock(return_value=cap.supervisions)

        with (
            patch("lattifai.caption.Caption.read", return_value=cap),
            patch("lattifai.translation.create_translator", return_value=translator),
            patch("lattifai.cli.translate._should_continue_with_refined", return_value=False),
        ):
            result = translate(
                input=str(input_path),
                output=str(output_path),
                translation=TranslationConfig(target_lang="zh"),
                caption=CaptionConfig(),
            )

        assert result is cap
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "Translated content"
        translator.translate_captions.assert_awaited_once()

    def test_translate_caption_implicit_output_path(self, tmp_path):
        from lattifai.cli.translate import _resolve_translation_output_path

        input_path = tmp_path / "input.srt"
        output_path = _resolve_translation_output_path(
            input_path=input_path,
            explicit_output=None,
            source_path=None,
            target_lang="zh",
        )

        assert output_path.parent == tmp_path
        assert output_path.suffix == ".srt"
        assert output_path.stem.startswith("input_")

    def test_should_continue_with_refined_auto(self):
        from lattifai.cli.translate import _should_continue_with_refined
        from lattifai.config.translation import TranslationConfig

        config = TranslationConfig(mode="normal", auto_refine_after_normal=True)
        assert _should_continue_with_refined(config) is True

    def test_should_continue_with_refined_non_normal(self):
        from lattifai.cli.translate import _should_continue_with_refined
        from lattifai.config.translation import TranslationConfig

        config = TranslationConfig(mode="quick")
        assert _should_continue_with_refined(config) is False
