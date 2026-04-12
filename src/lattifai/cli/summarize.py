"""Summarisation CLI entry point with nemo_run."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import CaptionConfig
from lattifai.config.summarization import SummarizationConfig


def _parse_meta_md(meta_path: Path) -> dict[str, Any]:
    """Parse YAML frontmatter + body from a .meta.md file.

    Returns a dict with keys: title, channel, duration, upload_date,
    speakers (list of dicts), chapters (list of dicts with title/start).
    """
    import yaml

    content = meta_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
    if not match:
        return {}

    fm = yaml.safe_load(match.group(1)) or {}
    body = match.group(2) or ""

    result: dict[str, Any] = {}
    if fm.get("title"):
        result["title"] = fm["title"]
    if fm.get("channel"):
        result["channel"] = fm["channel"]
    if fm.get("duration"):
        result["duration"] = str(fm["duration"])
    if fm.get("upload_date"):
        result["upload_date"] = str(fm["upload_date"])
    if fm.get("url"):
        result["url"] = fm["url"]

    # Speakers
    speakers = fm.get("speakers")
    if isinstance(speakers, list) and speakers:
        result["speakers"] = [
            {"name": s["name"], **({"role": s["role"]} if s.get("role") else {})}
            for s in speakers
            if isinstance(s, dict) and s.get("name")
        ]

    # YouTube chapters from body text (e.g. "0:00 Intro\n5:30 Topic")
    chapter_pattern = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", re.MULTILINE)
    matches = chapter_pattern.findall(body)
    if len(matches) >= 2:
        chapters = []
        for ts_str, title in matches:
            parts = ts_str.split(":")
            if len(parts) == 3:
                secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                secs = int(parts[0]) * 60 + int(parts[1])
            chapters.append({"title": title.strip(), "start": float(secs)})
        result["chapters"] = chapters

    return result


def _build_speaker_text(cap: Any) -> str:
    """Build text with speaker labels and timestamps for richer context.

    Format: [MM:SS] [Speaker] text
    Falls back to plain text if no speakers or timestamps are present.
    """
    lines = []
    has_speakers = any(getattr(s, "speaker", None) for s in cap.supervisions)
    has_times = any(getattr(s, "start", None) is not None for s in cap.supervisions)

    if not has_speakers and not has_times:
        return "\n".join(s.text for s in cap.supervisions if s.text)

    for sup in cap.supervisions:
        text = sup.text or ""
        if not text:
            continue

        parts = []
        start = getattr(sup, "start", None)
        if start is not None:
            m, sec = divmod(int(start), 60)
            parts.append(f"[{m:02d}:{sec:02d}]")

        speaker = getattr(sup, "speaker", None)
        if speaker:
            parts.append(f"[{speaker}]")

        parts.append(text)
        lines.append(" ".join(parts))

    return "\n".join(lines)


@run.cli.entrypoint(name="caption", namespace="summarize", entrypoint_cls=LattifAIEntrypoint)
def summarize_caption(
    input: Optional[str] = None,
    output: Optional[str] = None,
    meta: Optional[str] = None,
    summarization: Annotated[Optional[SummarizationConfig], run.Config[SummarizationConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
):
    """
    Summarise a caption or transcript file into a structured document.

    Reads any supported caption format (SRT, VTT, ASS, Gemini MD, …),
    extracts the text content, and produces a structured summary using
    an LLM.  Output is Markdown (default) or JSON.

    Args:
        input: Path to input caption file (positional argument).
        output: Path for the output summary file.  If omitted the file
            is written next to the input with a ``.summary.<lang>.md``
            suffix.
        meta: Path to a .meta.md file for video metadata. Populates
            title, channel, speakers, duration, and YouTube chapters
            for richer, more accurate summaries.
        summarization: Summarisation configuration.
            Fields: llm.model_name, llm.api_base_url, lang, length,
                    output_format,
                    source_lang, max_input_chars, chunk_chars,
                    max_chunks, overlap_chars, include_chapters,
                    include_metadata, temperature, verbose
        caption: Caption I/O configuration.
            Fields: input_format

    Examples:
        # Basic summary (Markdown, medium length, English)
        lai summarize caption input.srt

        # With video metadata for richer context
        lai summarize caption input.json meta=video.meta.md

        # Chinese short summary
        lai summarize caption input.srt summarization.lang=zh summarization.length=short

        # JSON output
        lai summarize caption input.srt summarization.output_format=json

        # Long summary with custom output path
        lai summarize caption input.srt output=my_summary.md summarization.length=long

        # Use OpenAI-compatible endpoint (provider inferred from model_name)
        lai summarize caption input.srt \\
            summarization.llm.model_name=qwen3 \\
            summarization.llm.api_base_url=http://localhost:8000/v1
    """
    from lattifai.caption import Caption
    from lattifai.summarization import ContentSummarizer, SummaryInput
    from lattifai.summarization.renderer import render
    from lattifai.theme import theme
    from lattifai.utils import safe_print

    config = summarization or SummarizationConfig()

    if not input:
        raise ValueError("Input caption file is required.")

    input_path = Path(input)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input}")

    # Read caption
    safe_print(theme.step(f"Loading: {input_path}"))
    cap = Caption.read(str(input_path))

    if not cap.supervisions:
        raise ValueError(f"No caption segments found in: {input}")

    # Build text with speaker labels and timestamps when available
    text = _build_speaker_text(cap)
    if not text.strip():
        raise ValueError("Caption file contains no text content.")

    # Parse .meta.md if provided
    meta_data: dict[str, Any] = {}
    chapters: list[dict[str, Any]] = []
    video_title = input_path.stem

    if meta:
        meta_path = Path(meta)
        if meta_path.exists():
            meta_data = _parse_meta_md(meta_path)
            safe_print(theme.step(f"Metadata: {meta_path.name}"))
            if meta_data.get("title"):
                video_title = meta_data["title"]
            if meta_data.get("chapters"):
                chapters = meta_data["chapters"]
                safe_print(theme.step(f"  Chapters: {len(chapters)} from description"))
            if meta_data.get("speakers"):
                names = [s["name"] for s in meta_data["speakers"]]
                safe_print(theme.step(f"  Speakers: {', '.join(names)}"))

    safe_print(
        theme.step(
            f"Summarising {len(cap.supervisions)} segments "
            f"({len(text)} chars) [length={config.length}, lang={config.lang}]"
        )
    )

    # Build SummaryInput with enriched metadata
    summary_metadata: dict = {}
    if cap.source_path:
        summary_metadata["source_file"] = cap.source_path
    if cap.language:
        summary_metadata["source_lang"] = cap.language
    if meta_data.get("channel"):
        summary_metadata["channel"] = meta_data["channel"]
    if meta_data.get("url"):
        summary_metadata["url"] = meta_data["url"]
    if meta_data.get("duration"):
        summary_metadata["duration"] = meta_data["duration"]
    if meta_data.get("speakers"):
        summary_metadata["speakers"] = ", ".join(
            f"{s['name']} ({s.get('role', 'speaker')})" for s in meta_data["speakers"]
        )

    summary_input = SummaryInput(
        title=video_title,
        text=text,
        metadata=summary_metadata,
        chapters=chapters,
        source_type="captions",
        source_lang=config.source_lang or cap.language,
    )

    # Summarise (retry once on JSON parse failure — LLM can produce
    # malformed JSON on the first attempt)
    client = config.llm.create_client()
    summarizer = ContentSummarizer(config, client)
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            result = asyncio.run(summarizer.summarize(summary_input))
            break
        except (json.JSONDecodeError, RuntimeError, Exception) as exc:
            if attempt < max_attempts:
                safe_print(theme.warn(f"LLM returned invalid JSON, retrying ({attempt}/{max_attempts})..."))
                continue
            safe_print(theme.err(f"Summary failed after {max_attempts} attempts: {exc}"))
            raise

    # Carry metadata through to rendered output
    if meta_data.get("channel"):
        result.metadata["channel"] = meta_data["channel"]
    if meta_data.get("url"):
        result.metadata["url"] = meta_data["url"]
    if meta_data.get("duration"):
        result.metadata["duration"] = meta_data["duration"]

    # Render
    rendered = render(result, config.output_format)

    # Resolve output path
    output_path = _resolve_output_path(input_path, output, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    safe_print(theme.ok(f"Summary saved: {output_path}"))
    return str(output_path)


def _resolve_output_path(input_path: Path, explicit_output: Optional[str], config: SummarizationConfig) -> Path:
    """Determine the output file path."""
    if explicit_output:
        return Path(explicit_output)

    ext = ".json" if config.output_format == "json" else ".md"
    return input_path.with_name(f"{input_path.stem}.summary.{config.lang}{ext}")


def main():
    """Entry point for lai-summarize command."""
    run.cli.main(summarize_caption)


if __name__ == "__main__":
    main()
