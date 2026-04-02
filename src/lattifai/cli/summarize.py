"""Summarisation CLI entry point with nemo_run."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import CaptionConfig
from lattifai.config.summarization import SummarizationConfig


@run.cli.entrypoint(name="caption", namespace="summarize", entrypoint_cls=LattifAIEntrypoint)
def summarize_caption(
    input: Optional[str] = None,
    output: Optional[str] = None,
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
        summarization: Summarisation configuration.
            Fields: llm.provider, llm.model_name, llm.api_key,
                    llm.api_base_url, lang, length, output_format,
                    source_lang, max_input_chars, chunk_chars,
                    max_chunks, overlap_chars, include_chapters,
                    include_metadata, temperature, verbose
        caption: Caption I/O configuration.
            Fields: input_format

    Examples:
        # Basic summary (Markdown, medium length, English)
        lai summarize caption input.srt

        # Chinese short summary
        lai summarize caption input.srt summarization.lang=zh summarization.length=short

        # JSON output
        lai summarize caption input.srt summarization.output_format=json

        # Long summary with custom output path
        lai summarize caption input.srt output=my_summary.md summarization.length=long

        # Use OpenAI-compatible endpoint
        lai summarize caption input.srt \\
            summarization.llm.provider=openai \\
            summarization.llm.api_base_url=http://localhost:8000/v1 \\
            summarization.llm.model_name=qwen3
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

    # Build plain text from supervisions
    text = "\n".join(sup.text for sup in cap.supervisions if sup.text)
    if not text.strip():
        raise ValueError("Caption file contains no text content.")

    safe_print(
        theme.step(
            f"Summarising {len(cap.supervisions)} segments "
            f"({len(text)} chars) [length={config.length}, lang={config.lang}]"
        )
    )

    # Build SummaryInput
    metadata: dict = {}
    if cap.source_path:
        metadata["source_file"] = cap.source_path
    if cap.language:
        metadata["source_lang"] = cap.language

    summary_input = SummaryInput(
        title=input_path.stem,
        text=text,
        metadata=metadata,
        source_type="captions",
        source_lang=config.source_lang or cap.language,
    )

    # Summarise
    client = config.llm.create_client()
    summarizer = ContentSummarizer(config, client)
    result = asyncio.run(summarizer.summarize(summary_input))

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
