"""Prompt templates for the summarisation pipeline."""

from __future__ import annotations

import json
from typing import Any

from lattifai.summarization.schema import SummaryInput

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a factual podcast/video content summarisation engine.

Rules:
1. Be faithful to the source. Never invent facts absent from the input.
2. Prefer explicit claims over inferred ones.
3. Maximise information density — no filler, no generic phrasing.
4. Write the output in the requested language.
5. Return **valid JSON only** — no markdown, no commentary.
6. If the source is weak or incomplete, reflect that in the confidence rationale.
7. Entities must come from the source, not from guesswork.
8. tags must be lowercase English keywords (kebab-case), regardless of output language.
9. seo_title must be under 60 characters, seo_description under 155 characters.
10. Chapters are the CORE structure — they ARE the summary. Each chapter must have \
a narrative paragraph (summary) and optionally 1-2 direct quotes with timestamps.
11. CHAPTER STRUCTURE RULES:
    - If the user prompt contains a "LOCKED CHAPTERS" section, you MUST output \
exactly those chapters in the SAME ORDER with IDENTICAL title and start values \
copied verbatim. Do NOT merge, split, rename, reorder, or drop any chapter. \
Generate ONLY the `summary` and `quotes` for each locked chapter.
    - If chapters are provided as plain "Chapters:" (not locked), use them as the \
suggested section structure.
    - If no chapters are provided, segment the content into 3-12 coherent topic \
sections using [MM:SS] timestamps from the source transcript.
12. Preserve the speaker's voice: include exact short quotes (max 25 words) that \
capture key insights. Format quotes as: "exact words" [MM:SS]

Return a JSON object with exactly these fields:
{
  "title": "string",
  "overview": "1-2 sentence high-level hook — why should someone listen to this?",
  "chapters": [
    {
      "title": "Chapter title",
      "start": total_seconds_as_float (e.g. [01:09] = 69.0, [00:30] = 30.0, NOT 1.09),
      "summary": "1-3 sentence narrative of this section with speaker attribution",
      "quotes": ["exact words from speaker [MM:SS]"]
    }
  ],
  "entities": [{"name": "string", "type": "string", "description": "string"}, ...],
  "tags": ["lowercase-tag", ...],
  "seo_title": "string (under 60 chars)",
  "seo_description": "string (under 155 chars)",
  "confidence": {"score": 0.0-1.0, "rationale": "string"}
}"""

# ---------------------------------------------------------------------------
# Length instructions
# ---------------------------------------------------------------------------

LENGTH_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Produce a concise summary:\n"
        "- overview: 1-2 sentences\n"
        "- chapters: 2-4 sections, each with 1-2 sentence summary\n"
        "- entities: up to 5\n"
        "- quotes: 0-1 per chapter"
    ),
    "medium": (
        "Produce a moderately detailed summary:\n"
        "- overview: 2-3 sentences\n"
        "- chapters: 3-6 sections, each with 2-3 sentence summary\n"
        "- entities: up to 8\n"
        "- quotes: 1 per chapter when available"
    ),
    "long": (
        "Produce a comprehensive summary:\n"
        "- overview: 3-4 sentences\n"
        "- chapters: 4-10 sections, each with 3-5 sentence summary\n"
        "- entities: up to 12\n"
        "- quotes: 1-2 per chapter"
    ),
}


def resolve_auto_length(text_chars: int) -> str:
    """Pick short/medium/long based on input text length.

    Rough mapping (assuming ~130 chars/min for spoken English):
      < 10 min (~13k chars) → short
      10-45 min (~13k-60k)  → medium
      > 45 min (~60k+)      → long
    """
    if text_chars < 13000:
        return "short"
    if text_chars < 60000:
        return "medium"
    return "long"


def get_length_instruction(length: str, *, text_chars: int = 0) -> str:
    """Return length-specific guidance for the user prompt.

    When *length* is ``"auto"``, resolves to short/medium/long based
    on *text_chars*.
    """
    if length == "auto":
        length = resolve_auto_length(text_chars)
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
    lock_chapters: bool = False,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Assemble the user prompt for a single summarisation call.

    When *lock_chapters* is True and chapters are present, renders them as a
    "LOCKED CHAPTERS" block the model is required to preserve verbatim.
    """
    parts: list[str] = []

    parts.append("Summarise the following content.\n")
    parts.append(f"Output language: {lang}")
    parts.append(f"Source type: {summary_input.source_type}")
    if summary_input.source_lang:
        parts.append(f"Source language: {summary_input.source_lang}")
    parts.append("")

    # Length guidance
    parts.append(get_length_instruction(length, text_chars=len(summary_input.text)))
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

    # Chapters — either LOCKED (hard constraint) or suggested
    if include_chapters and summary_input.chapters:
        if lock_chapters:
            parts.append("LOCKED CHAPTERS (REQUIRED — copy verbatim, do NOT modify count/order/title/start):")
            for ch in summary_input.chapters:
                title = ch.get("title", "")
                start = ch.get("start", "")
                parts.append(f"  - start={start}  title={title!r}")
            parts.append("")
            parts.append(
                "You MUST output exactly these chapters in the same order, "
                "with identical `title` and `start` values. Generate only the "
                "`summary` and `quotes` for each chapter."
            )
            parts.append("")
        else:
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
    locked_chapters: list[dict[str, Any]] | None = None,
) -> str:
    """Assemble the user prompt for the map-reduce merge step.

    When *locked_chapters* is provided, the reduce step must output exactly
    that chapter list (count/order/title/start copied verbatim); it only
    fills in summary + quotes from the partials.
    """
    parts: list[str] = [
        "Merge the following partial summaries into one final structured JSON summary.\n",
        f"Output language: {lang}",
        f"Source type: {source_type}",
        f"Title: {title}",
        "",
        get_length_instruction(length),
        "",
    ]

    if locked_chapters:
        parts.append("LOCKED CHAPTERS (REQUIRED — copy verbatim, do NOT modify count/order/title/start):")
        for ch in locked_chapters:
            title_val = ch.get("title", "")
            start_val = ch.get("start", "")
            parts.append(f"  - start={start_val}  title={title_val!r}")
        parts.append("")
        parts.append(
            "Your merged output MUST contain exactly these chapters in the same "
            "order, with identical `title` and `start` values. Synthesise the "
            "`summary` and `quotes` for each chapter from the partial summaries."
        )
        parts.append("")

    parts.extend(
        [
            "Rules:",
            "1. Deduplicate repeated points.",
            "2. Preserve facts that appear consistently across chunks.",
            "3. If chunks conflict, prefer the more cautious phrasing.",
            "4. Keep the final result faithful and concise.",
            "5. Return valid JSON only.\n",
            "Partial summaries:",
            json.dumps(partial_results, ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(parts)
