"""Tests for lai summarize caption CLI command."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

LATTIFAI_TESTS_CLI_DRYRUN = bool(os.environ.get("LATTIFAI_TESTS_CLI_DRYRUN", "false"))


def run_summarize_command(args, env=None):
    """Helper to run the summarize command and return result."""
    cmd = ["lai", "summarize", "caption", "-Y"]
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
        print(f"Command: {' '.join(cmd)} failed with exit code {e.returncode}")
        raise e


class TestSummarizeHelp:
    """Test summarize command help output."""

    def test_summarize_help(self):
        result = subprocess.run(
            ["lai", "summarize", "caption", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "usage:" in result.stdout or "help" in result.stdout

    def test_summarize_group_lists_caption_subcommand(self):
        """'lai summarize --help' should list the 'caption' subcommand."""
        result = subprocess.run(
            ["lai", "summarize", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0
        assert "caption" in combined.lower()


class TestSummarizeErrors:
    """Test summarize command error handling."""

    def test_missing_input_file(self, tmp_path):
        args = ["nonexistent_file.srt"]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_summarize_command(args)

    def test_empty_caption_file(self, tmp_path):
        empty_file = tmp_path / "empty.srt"
        empty_file.write_text("", encoding="utf-8")
        args = [str(empty_file)]
        if not LATTIFAI_TESTS_CLI_DRYRUN:
            with pytest.raises(subprocess.CalledProcessError):
                run_summarize_command(args)


class TestSummarizeCLIUnit:
    def test_summarize_caption_success_mocked(self, tmp_path):
        from lattifai.cli.summarize import summarize_caption
        from lattifai.config.summarization import SummarizationConfig
        from lattifai.summarization.schema import SummaryConfidence, SummaryResult

        input_path = tmp_path / "input.srt"
        input_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        cap = SimpleNamespace(
            supervisions=[SimpleNamespace(text="Hello"), SimpleNamespace(text="World")],
            source_path=str(input_path),
            language="en",
        )
        config = SummarizationConfig(lang="en", output_format="markdown")
        fake_client = object()
        config.llm.create_client = Mock(return_value=fake_client)
        fake_summarizer = Mock()
        fake_summarizer.summarize = AsyncMock(
            return_value=SummaryResult(
                title="input",
                summary="Short summary",
                key_points=["Point 1"],
                confidence=SummaryConfidence(score=0.8, rationale="good"),
            )
        )

        with (
            patch("lattifai.caption.Caption.read", return_value=cap),
            patch("lattifai.summarization.ContentSummarizer", return_value=fake_summarizer),
        ):
            output_path = summarize_caption(input=str(input_path), summarization=config)

        output_file = Path(output_path)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "## Summary" in content
        assert "Short summary" in content

    def test_summarize_caption_json_output(self, tmp_path):
        from lattifai.cli.summarize import summarize_caption
        from lattifai.config.summarization import SummarizationConfig
        from lattifai.summarization.schema import SummaryResult

        input_path = tmp_path / "input.srt"
        input_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        cap = SimpleNamespace(
            supervisions=[SimpleNamespace(text="Hello")],
            source_path=str(input_path),
            language="en",
        )
        config = SummarizationConfig(lang="zh", output_format="json")
        config.llm.create_client = Mock(return_value=object())
        fake_summarizer = Mock()
        fake_summarizer.summarize = AsyncMock(return_value=SummaryResult(title="input", summary="JSON summary"))

        with (
            patch("lattifai.caption.Caption.read", return_value=cap),
            patch("lattifai.summarization.ContentSummarizer", return_value=fake_summarizer),
        ):
            output_path = summarize_caption(input=str(input_path), summarization=config)

        output_file = Path(output_path)
        assert output_file.suffix == ".json"
        parsed = json.loads(output_file.read_text(encoding="utf-8"))
        assert parsed["summary"] == "JSON summary"


class TestSummarizationUnit:
    """Unit tests for summarization module with mocked LLM."""

    def test_summary_result_from_dict(self):
        from lattifai.summarization.schema import summary_result_from_dict

        data = {
            "title": "Test Video",
            "summary": "This is a test summary.",
            "key_points": ["Point 1", "Point 2"],
            "entities": [{"name": "Python", "type": "language", "description": "Programming language"}],
            "actionable_insights": ["Learn Python"],
            "confidence": {"score": 0.85, "rationale": "Good source"},
        }
        result = summary_result_from_dict(data)
        assert result.title == "Test Video"
        assert result.summary == "This is a test summary."
        assert len(result.key_points) == 2
        assert len(result.entities) == 1
        assert result.entities[0].name == "Python"
        assert result.confidence.score == 0.85

    def test_summary_result_from_dict_missing_fields(self):
        from lattifai.summarization.schema import summary_result_from_dict

        result = summary_result_from_dict({})
        assert result.title == "Untitled"
        assert result.summary == ""
        assert result.key_points == []
        assert result.entities == []

    def test_summary_result_from_dict_scalar_confidence(self):
        from lattifai.summarization.schema import summary_result_from_dict

        result = summary_result_from_dict({"confidence": 0.9})
        assert result.confidence.score == 0.9

    def test_summary_result_to_dict(self):
        from lattifai.summarization.schema import (
            SummaryConfidence,
            SummaryEntity,
            SummaryResult,
            summary_result_to_dict,
        )

        result = SummaryResult(
            title="Test",
            summary="Summary text",
            key_points=["A"],
            entities=[SummaryEntity(name="X", type="concept")],
            confidence=SummaryConfidence(score=0.8, rationale="test"),
        )
        d = summary_result_to_dict(result)
        assert d["title"] == "Test"
        assert d["confidence"]["score"] == 0.8
        assert len(d["entities"]) == 1

    def test_render_markdown(self):
        from lattifai.summarization.renderer import render_markdown
        from lattifai.summarization.schema import SummaryConfidence, SummaryResult

        result = SummaryResult(
            title="Test Title",
            summary="A summary.",
            key_points=["Point 1"],
            confidence=SummaryConfidence(score=0.85, rationale="Good", source_quality="high"),
        )
        md = render_markdown(result)
        assert "## Summary" in md
        assert "## Key Points" in md
        assert "A summary." in md
        assert "Point 1" in md
        assert "0.85" in md

    def test_render_json(self):
        import json

        from lattifai.summarization.renderer import render_json
        from lattifai.summarization.schema import SummaryResult

        result = SummaryResult(title="Test", summary="Summary")
        output = render_json(result)
        parsed = json.loads(output)
        assert parsed["title"] == "Test"

    def test_render_dispatch(self):
        from lattifai.summarization.renderer import render
        from lattifai.summarization.schema import SummaryResult

        result = SummaryResult(title="Test", summary="Summary")
        assert "## Summary" in render(result, "markdown")
        assert '"title"' in render(result, "json")

    def test_prompt_builder(self):
        from lattifai.summarization.prompts import build_summary_user_prompt
        from lattifai.summarization.schema import SummaryInput

        inp = SummaryInput(
            title="Test Video",
            text="Hello world this is a test transcript.",
            source_type="captions",
            source_lang="en",
        )
        prompt = build_summary_user_prompt(inp, lang="en", length="short")
        assert "Output language: en" in prompt
        assert "Source type: captions" in prompt
        assert "Hello world" in prompt
        assert "2-4 sentences" in prompt

    def test_prompt_builder_chunk(self):
        from lattifai.summarization.prompts import build_summary_user_prompt
        from lattifai.summarization.schema import SummaryInput

        inp = SummaryInput(title="Test", text="Content here.")
        prompt = build_summary_user_prompt(inp, lang="zh", length="medium", chunk_index=0, total_chunks=3)
        assert "chunk 1 of 3" in prompt

    def test_reduce_prompt(self):
        from lattifai.summarization.prompts import build_reduce_user_prompt

        partials = [{"summary": "Part 1"}, {"summary": "Part 2"}]
        prompt = build_reduce_user_prompt(partials, title="Test", lang="en", length="medium", source_type="captions")
        assert "Merge" in prompt
        assert "Part 1" in prompt

    def test_config_validation(self):
        from lattifai.config.summarization import SummarizationConfig

        with pytest.raises(ValueError, match="max_input_chars"):
            SummarizationConfig(max_input_chars=100)
        with pytest.raises(ValueError, match="chunk_chars"):
            SummarizationConfig(chunk_chars=500)
        with pytest.raises(ValueError, match="max_chunks"):
            SummarizationConfig(max_chunks=0)
        with pytest.raises(ValueError, match="temperature"):
            SummarizationConfig(temperature=2.0)
        with pytest.raises(ValueError, match="overlap_chars"):
            SummarizationConfig(overlap_chars=15000)

    def test_config_defaults(self):
        from lattifai.config.summarization import SummarizationConfig

        config = SummarizationConfig()
        assert config.lang == "en"
        assert config.length == "medium"
        assert config.output_format == "markdown"
        assert config.temperature == 0.2

    def test_chunking_logic(self):
        from lattifai.config.summarization import SummarizationConfig
        from lattifai.summarization.summarizer import ContentSummarizer

        config = SummarizationConfig(max_input_chars=2000)
        summarizer = ContentSummarizer(config, client=None)
        assert summarizer._needs_chunking("x" * 2001) is True
        assert summarizer._needs_chunking("x" * 1000) is False

    def test_split_text(self):
        from lattifai.config.summarization import SummarizationConfig
        from lattifai.summarization.summarizer import ContentSummarizer

        config = SummarizationConfig(chunk_chars=1000, overlap_chars=100, max_chunks=10)
        summarizer = ContentSummarizer(config, client=None)
        text = "\n".join([f"Line {i} with some content here for testing chunk splitting." for i in range(100)])
        chunks = summarizer._split_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_confidence_adjustment(self):
        from lattifai.config.summarization import SummarizationConfig
        from lattifai.summarization.schema import SummaryResult
        from lattifai.summarization.summarizer import ContentSummarizer

        config = SummarizationConfig()
        summarizer = ContentSummarizer(config, client=None)
        result = SummaryResult(title="Test", summary="Summary")

        adjusted = summarizer._adjust_confidence(result, source_type="captions", text_length=1000, used_chunking=False)
        assert adjusted.confidence.score == 0.85
        assert adjusted.confidence.source_quality == "high"

        adjusted = summarizer._adjust_confidence(result, source_type="metadata", text_length=100, used_chunking=True)
        assert adjusted.confidence.score < 0.45
        assert adjusted.confidence.source_quality == "low"

    def test_output_path_resolution(self):
        from lattifai.cli.summarize import _resolve_output_path
        from lattifai.config.summarization import SummarizationConfig

        config = SummarizationConfig(lang="zh")
        path = _resolve_output_path(Path("/tmp/test.srt"), None, config)
        assert path == Path("/tmp/test.summary.zh.md")

        config_json = SummarizationConfig(lang="en", output_format="json")
        path = _resolve_output_path(Path("/tmp/test.srt"), None, config_json)
        assert path == Path("/tmp/test.summary.en.json")

        path = _resolve_output_path(Path("/tmp/test.srt"), "/custom/output.md", config)
        assert path == Path("/custom/output.md")
