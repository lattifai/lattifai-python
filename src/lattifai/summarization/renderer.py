"""Render SummaryResult to Markdown or JSON."""

from __future__ import annotations

import json

from lattifai.summarization.schema import SummaryResult, summary_result_to_dict


def _fmt_ts(seconds: float) -> str:
    """Format seconds as [MM:SS] or [HH:MM:SS]."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"[{h}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def render_markdown(result: SummaryResult) -> str:
    """Render a SummaryResult as a Markdown document.

    YAML frontmatter: all structured fields for machine parsing (D1 publish).
    Body: chapter-based narrative with timestamps and quotes for human reading.
    """
    lines: list[str] = []

    # ── YAML Frontmatter ──
    lines.append("---")
    lines.append(f'title: "{_escape_yaml(result.title)}"')

    if result.seo_title:
        lines.append(f'seo_title: "{_escape_yaml(result.seo_title)}"')
    if result.seo_description:
        lines.append(f'seo_description: "{_escape_yaml(result.seo_description)}"')

    if result.tags:
        tags_str = json.dumps(result.tags, ensure_ascii=False)
        lines.append(f"tags: {tags_str}")

    if result.chapters:
        lines.append("chapters:")
        for ch in result.chapters:
            lines.append(f'  - title: "{_escape_yaml(ch.title)}"')
            lines.append(f"    start: {ch.start}")
            if ch.end:
                lines.append(f"    end: {ch.end}")

    if result.confidence:
        lines.append(f"confidence: {result.confidence.score:.2f}")
        lines.append(f"source_quality: {result.confidence.source_quality}")

    for key in ("channel", "url", "duration", "source_type"):
        val = result.metadata.get(key)
        if val:
            lines.append(f'{key}: "{_escape_yaml(str(val))}"')

    lines.append("---")
    lines.append("")

    # ── Overview ──
    if result.overview:
        lines.append(result.overview)
        lines.append("")

    # ── Chapters (the narrative core) ──
    for ch in result.chapters:
        ts = _fmt_ts(ch.start)
        lines.append(f"## {ts} {ch.title}")
        lines.append("")
        if ch.summary:
            lines.append(ch.summary)
        if ch.quotes:
            lines.append("")
            for quote in ch.quotes:
                lines.append(f"> *{quote}*")
        lines.append("")

    # ── Entities ──
    if result.entities:
        lines.append("## Entities")
        lines.append("")
        for entity in result.entities:
            desc = f": {entity.description}" if entity.description else ""
            lines.append(f"- **{entity.name}** ({entity.type}){desc}")
        lines.append("")

    return "\n".join(lines)


def render_json(result: SummaryResult) -> str:
    """Render a SummaryResult as a pretty-printed JSON string."""
    return json.dumps(summary_result_to_dict(result), ensure_ascii=False, indent=2)


def render(result: SummaryResult, output_format: str = "markdown") -> str:
    """Dispatch to the appropriate renderer."""
    if output_format == "json":
        return render_json(result)
    return render_markdown(result)


def _escape_yaml(text: str) -> str:
    """Escape characters that break YAML double-quoted strings."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
