"""Render SummaryResult to Markdown or JSON."""

from __future__ import annotations

import json

from lattifai.summarization.schema import SummaryResult, summary_result_to_dict


def render_markdown(result: SummaryResult) -> str:
    """Render a SummaryResult as a stable Markdown document.

    The structure and field order are deterministic — the model never
    controls Markdown formatting, only the content values.
    """
    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "{_escape_yaml(result.title)}"')
    if result.confidence:
        lines.append(f"confidence: {result.confidence.score:.2f}")
        lines.append(f"source_quality: {result.confidence.source_quality}")
    for key in ("channel", "url", "duration", "source_type"):
        val = result.metadata.get(key)
        if val:
            lines.append(f'{key}: "{_escape_yaml(str(val))}"')
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(result.summary)
    lines.append("")

    # Key Points
    if result.key_points:
        lines.append("## Key Points")
        lines.append("")
        for point in result.key_points:
            lines.append(f"- {point}")
        lines.append("")

    # Entities
    if result.entities:
        lines.append("## Entities")
        lines.append("")
        for entity in result.entities:
            desc = f": {entity.description}" if entity.description else ""
            lines.append(f"- **{entity.name}** ({entity.type}){desc}")
        lines.append("")

    # Actionable Insights
    if result.actionable_insights:
        lines.append("## Actionable Insights")
        lines.append("")
        for insight in result.actionable_insights:
            lines.append(f"- {insight}")
        lines.append("")

    # Confidence
    if result.confidence:
        lines.append("## Confidence")
        lines.append("")
        lines.append(f"- **Score**: {result.confidence.score:.2f}")
        lines.append(f"- **Source quality**: {result.confidence.source_quality}")
        if result.confidence.rationale:
            lines.append(f"- **Rationale**: {result.confidence.rationale}")
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
