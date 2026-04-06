"""Tests for lai diarization CLI command."""

import os
import subprocess
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from lattifai.cli.diarize import _resolve_context

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


class TestResolveContext:
    """Unit tests for _resolve_context() — file path vs inline string detection."""

    def test_none_returns_none(self):
        assert _resolve_context(None) is None

    def test_empty_string_returns_none(self):
        assert _resolve_context("") is None

    def test_inline_string_passthrough(self):
        ctx = "podcast, host is Alice, guest is Bob"
        assert _resolve_context(ctx) == ctx

    def test_nonexistent_path_treated_as_inline(self):
        result = _resolve_context("/nonexistent/path/to/meta.md")
        assert result == "/nonexistent/path/to/meta.md"

    def test_file_without_frontmatter(self, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_text("Just some plain text about speakers.", encoding="utf-8")
        assert _resolve_context(str(f)) == "Just some plain text about speakers."

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        assert _resolve_context(str(f)) is None

    def test_meta_md_with_speakers(self, tmp_path):
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: AI and Mathematics
                speakers:
                  - name: Terence Tao
                    role: guest
                  - name: Dwarkesh Patel
                    role: host
                ---
                A deep conversation about AI.
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert "Channel/Host: Dwarkesh Patel" in result
        assert "Guests: Terence Tao" in result
        assert "Title: AI and Mathematics" in result
        assert "A deep conversation about AI" in result

    def test_meta_md_channel_fallback(self, tmp_path):
        """Channel field used when no speakers with host role."""
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: Great Talk
                channel: TechChannel
                ---
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert "Channel/Host: TechChannel" in result
        assert "Title: Great Talk" in result

    def test_meta_md_uploader_fallback(self, tmp_path):
        """'uploader' used as channel fallback."""
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: My Video
                uploader: SomeCreator
                ---
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert "Channel/Host: SomeCreator" in result

    def test_meta_md_channel_not_duplicated_with_host(self, tmp_path):
        """Channel is NOT added when speakers already provide a host."""
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: Interview
                channel: MyChannel
                speakers:
                  - name: Alice
                    role: host
                ---
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert result.count("Channel/Host:") == 1
        assert "Channel/Host: Alice" in result

    def test_description_filters_urls_and_timestamps(self, tmp_path):
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: Talk
                ---
                https://example.com/sponsor
                00:00 Intro
                This is the real description.
                Another meaningful line.
                #hashtag should be skipped
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert "This is the real description." in result
        assert "Another meaningful line." in result
        assert "https://example.com" not in result
        assert "00:00" not in result
        assert "#hashtag" not in result

    def test_description_limited_to_3_lines(self, tmp_path):
        f = tmp_path / "video.meta.md"
        f.write_text(
            textwrap.dedent(
                """\
                ---
                title: Long
                ---
                Line one.
                Line two.
                Line three.
                Line four should not appear.
            """
            ),
            encoding="utf-8",
        )
        result = _resolve_context(str(f))
        assert "Line three." in result
        assert "Line four" not in result

    def test_unclosed_frontmatter_returns_none(self, tmp_path):
        f = tmp_path / "broken.meta.md"
        f.write_text("---\ntitle: Broken\nno closing delimiter", encoding="utf-8")
        assert _resolve_context(str(f)) is None

    def test_invalid_yaml_falls_back_to_raw(self, tmp_path):
        f = tmp_path / "bad.meta.md"
        f.write_text("---\n: [invalid yaml\n---\nbody\n", encoding="utf-8")
        result = _resolve_context(str(f))
        assert result is not None  # Falls back to raw frontmatter text


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

    def test_diarize_naming_help(self):
        result = subprocess.run(
            ["lai", "diarize", "naming", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "help" in result.stdout
        # Verify command description is present
        combined = result.stdout + result.stderr
        assert "speaker" in combined.lower() or "naming" in combined.lower() or "help" in combined.lower()


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
        from lattifai.config import CaptionConfig, CaptionInputConfig, CaptionOutputConfig, MediaConfig

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        input_caption_path = tmp_path / "input.srt"
        input_caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        media = MediaConfig(input_path=str(tmp_path / "audio.wav"))
        caption = CaptionConfig(
            input=CaptionInputConfig(path=str(input_caption_path)),
            output=CaptionOutputConfig(path=str(tmp_path / "output.srt")),
        )
        fake_client = Mock()
        fake_client.audio_loader = Mock(return_value="audio")
        fake_client._read_caption = Mock(return_value=SimpleNamespace(alignments=[], supervisions=[]))

        with patch("lattifai.cli.diarize.build_lattifai_client", return_value=fake_client):
            with pytest.raises(ValueError, match="Caption does not contain segments"):
                diarize(media=media, caption=caption)

    def test_diarize_infer_speakers_flag(self, tmp_path):
        from lattifai.cli.diarize import diarize
        from lattifai.config import (
            CaptionConfig,
            CaptionInputConfig,
            CaptionOutputConfig,
            DiarizationConfig,
            MediaConfig,
        )

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        input_caption_path = tmp_path / "input.srt"
        input_caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        media = MediaConfig(input_path=str(audio_path))
        caption = CaptionConfig(
            input=CaptionInputConfig(path=str(input_caption_path)),
            output=CaptionOutputConfig(path=str(tmp_path / "output.srt")),
        )
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


class TestNamingUnit:
    """Unit tests for the lai diarize naming command."""

    def test_naming_missing_input_raises(self):
        """Naming requires an input caption path."""
        from lattifai.cli.diarize import naming

        with pytest.raises(ValueError, match="Input caption path required"):
            naming()

    def test_naming_empty_caption_raises(self, tmp_path):
        """Naming raises when caption has no segments."""
        from lattifai.cli.diarize import naming

        input_path = tmp_path / "empty.srt"
        input_path.write_text("", encoding="utf-8")

        cap = SimpleNamespace(supervisions=[], source_path=str(input_path))

        with patch("lattifai.caption.Caption.read", return_value=cap):
            with pytest.raises(ValueError, match="no segments"):
                naming(input_caption=str(input_path))

    def test_naming_single_speaker_returns_early(self, tmp_path):
        """Naming returns early when fewer than 2 speakers are found."""
        from lattifai.cli.diarize import naming

        input_path = tmp_path / "single_speaker.srt"
        input_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        cap = SimpleNamespace(
            supervisions=[SimpleNamespace(text="Hello", speaker="SPEAKER_00")],
            source_path=str(input_path),
        )

        with patch("lattifai.caption.Caption.read", return_value=cap):
            result = naming(input_caption=str(input_path))

        assert result is cap  # Returns unchanged

    def test_naming_multi_speaker_calls_infer(self, tmp_path):
        """Naming calls infer_speaker_names when 2+ speakers are present."""
        from lattifai.cli.diarize import naming
        from lattifai.config import DiarizationConfig

        input_path = tmp_path / "multi_speaker.json"
        input_path.write_text("{}", encoding="utf-8")
        output_path = tmp_path / "named.json"

        sup0 = SimpleNamespace(text="Hello there", speaker="SPEAKER_00")
        sup1 = SimpleNamespace(text="Hey, how are you?", speaker="SPEAKER_01")
        cap = Mock()
        cap.supervisions = [sup0, sup1]
        cap.source_path = str(input_path)

        diarization_config = DiarizationConfig()
        fake_llm_client = Mock()

        with (
            patch("lattifai.caption.Caption.read", return_value=cap),
            patch(
                "lattifai.diarization.speaker.infer_speaker_names",
                return_value={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
            ) as mock_infer,
            patch.object(diarization_config, "llm") as mock_llm_cfg,
        ):
            mock_llm_cfg.create_client.return_value = fake_llm_client
            mock_llm_cfg.model_name = "test-model"
            result = naming(
                input_caption=str(input_path),
                output_caption=str(output_path),
                diarization=diarization_config,
            )

        mock_infer.assert_called_once()
        # Verify speaker labels were updated
        assert sup0.speaker == "Alice"
        assert sup1.speaker == "Bob"
        cap.write.assert_called_once()

    def test_naming_no_inferred_names_returns_cap(self, tmp_path):
        """Naming returns cap unchanged when LLM cannot infer names."""
        from lattifai.cli.diarize import naming
        from lattifai.config import DiarizationConfig

        input_path = tmp_path / "multi.srt"
        input_path.write_text("placeholder", encoding="utf-8")

        sup0 = SimpleNamespace(text="...", speaker="SPEAKER_00")
        sup1 = SimpleNamespace(text="...", speaker="SPEAKER_01")
        cap = Mock()
        cap.supervisions = [sup0, sup1]
        cap.source_path = str(input_path)

        diarization_config = DiarizationConfig()

        with (
            patch("lattifai.caption.Caption.read", return_value=cap),
            patch("lattifai.diarization.speaker.infer_speaker_names", return_value={}),
            patch.object(diarization_config, "llm") as mock_llm_cfg,
        ):
            mock_llm_cfg.create_client.return_value = Mock()
            mock_llm_cfg.model_name = "test-model"
            result = naming(
                input_caption=str(input_path),
                diarization=diarization_config,
            )

        assert result is cap
        # Speaker labels should NOT have changed
        assert sup0.speaker == "SPEAKER_00"
        assert sup1.speaker == "SPEAKER_01"
        cap.write.assert_not_called()
