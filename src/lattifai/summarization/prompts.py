"""Prompt templates for the summarisation pipeline."""

from __future__ import annotations

import json
from typing import Any

from lattifai.summarization.schema import SummaryInput

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a factual content summarisation engine.

Rules:
1. Be faithful to the source. Never invent facts absent from the input.
2. Prefer explicit claims over inferred ones.
3. Maximise information density — no filler, no generic phrasing.
4. Write the output in the requested language.
5. Return **valid JSON only** — no markdown, no commentary.
6. If the source is weak or incomplete, reflect that in the confidence rationale.
7. Entities must come from the source, not from guesswork.
8. If there are no genuine actionable insights, return an empty list.

Return a JSON object with exactly these fields:
{
  "title": "string",
  "summary": "string",
  "key_points": ["string", ...],
  "entities": [{"name": "string", "type": "string", "description": "string"}, ...],
  "actionable_insights": ["string", ...],
  "confidence": {"score": 0.0-1.0, "rationale": "string"}
}"""

# ---------------------------------------------------------------------------
# Length instructions
# ---------------------------------------------------------------------------

LENGTH_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Produce a concise summary:\n"
        "- summary: 2-4 sentences\n"
        "- key_points: 3-5 items\n"
        "- entities: up to 5\n"
        "- actionable_insights: 0-3"
    ),
    "medium": (
        "Produce a moderately detailed summary:\n"
        "- summary: 1-3 paragraphs\n"
        "- key_points: 5-8 items\n"
        "- entities: up to 8\n"
        "- actionable_insights: 2-5"
    ),
    "long": (
        "Produce a comprehensive summary:\n"
        "- summary: 3-6 paragraphs\n"
        "- key_points: 8-12 items\n"
        "- entities: up to 12\n"
        "- actionable_insights: 3-8"
    ),
}


def get_length_instruction(length: str) -> str:
    """Return length-specific guidance for the user prompt."""
    return LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"])


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_summary_user_prompt(
    summary_input: SummaryInput,
    *,
    lang: str,
    length: str,
    include_metadata: bool = True,
    include_chapters: bool = True,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Assemble the user prompt for a single summarisation call."""
    parts: list[str] = []

    parts.append("Summarise the following content.\n")
    parts.append(f"Output language: {lang}")
    parts.append(f"Source type: {summary_input.source_type}")
    if summary_input.source_lang:
        parts.append(f"Source language: {summary_input.source_lang}")
    parts.append("")

    # Length guidance
    parts.append(get_length_instruction(length))
    parts.append("")

    # Chunk context
    if chunk_index is not None and total_chunks is not None:
        parts.append(f"This is chunk {chunk_index + 1} of {total_chunks}.")
        parts.append("Summarise only this chunk; a merge step follows.\n")

    # Title
    if summary_input.title:
        parts.append(f"Title:\n{summary_input.title}\n")

    # Metadata
    if include_metadata and summary_input.metadata:
        meta_lines = [f"  {k}: {v}" for k, v in summary_input.metadata.items() if v]
        if meta_lines:
            parts.append("Metadata:")
            parts.extend(meta_lines)
            parts.append("")

    # Chapters
    if include_chapters and summary_input.chapters:
        parts.append("Chapters:")
        for ch in summary_input.chapters:
            title = ch.get("title", "")
            start = ch.get("start", "")
            parts.append(f"  - [{start}] {title}")
        parts.append("")

    # Content
    parts.append("Source content:")
    parts.append(summary_input.text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reduce prompt builder
# ---------------------------------------------------------------------------


def build_reduce_user_prompt(
    partial_results: list[dict[str, Any]],
    *,
    title: str,
    lang: str,
    length: str,
    source_type: str,
) -> str:
    """Assemble the user prompt for the map-reduce merge step."""
    parts: list[str] = [
        "Merge the following partial summaries into one final structured JSON summary.\n",
        f"Output language: {lang}",
        f"Source type: {source_type}",
        f"Title: {title}",
        "",
        get_length_instruction(length),
        "",
        "Rules:",
        "1. Deduplicate repeated points.",
        "2. Preserve facts that appear consistently across chunks.",
        "3. If chunks conflict, prefer the more cautious phrasing.",
        "4. Keep the final result faithful and concise.",
        "5. Return valid JSON only.\n",
        "Partial summaries:",
        json.dumps(partial_results, ensure_ascii=False, indent=2),
    ]
    return "\n".join(parts)
