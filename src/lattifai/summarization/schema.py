"""Data structures for the summarization pipeline."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SummaryEntity:
    """Named entity extracted from content."""

    name: str
    type: str
    description: str = ""


@dataclass
class SummaryConfidence:
    """Confidence score with rationale and source quality indicator."""

    score: float
    rationale: str
    source_quality: Literal["high", "medium", "low"] = "medium"


@dataclass
class SummaryInput:
    """Normalised summarisation input — source-agnostic."""

    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    source_type: Literal["captions", "metadata", "mixed"] = "metadata"
    source_lang: str | None = None


@dataclass
class SummaryResult:
    """Structured summary output produced by ContentSummarizer."""

    title: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    entities: list[SummaryEntity] = field(default_factory=list)
    actionable_insights: list[str] = field(default_factory=list)
    confidence: SummaryConfidence | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def summary_result_from_dict(data: dict[str, Any], *, fallback_title: str = "Untitled") -> SummaryResult:
    """Build a *SummaryResult* from a raw dict (typically LLM JSON output).

    Missing fields are filled with safe defaults so that downstream
    rendering never fails even if the model omits optional keys.
    """
    entities_raw = data.get("entities") or []
    entities = [
        SummaryEntity(
            name=e.get("name", ""),
            type=e.get("type", ""),
            description=e.get("description", ""),
        )
        for e in entities_raw
        if isinstance(e, dict)
    ]

    conf_raw = data.get("confidence")
    confidence = None
    if isinstance(conf_raw, dict):
        confidence = SummaryConfidence(
            score=float(conf_raw.get("score", 0.5)),
            rationale=conf_raw.get("rationale", ""),
        )
    elif isinstance(conf_raw, (int, float)):
        confidence = SummaryConfidence(score=float(conf_raw), rationale="")

    return SummaryResult(
        title=data.get("title") or fallback_title,
        summary=data.get("summary", ""),
        key_points=data.get("key_points") or [],
        entities=entities,
        actionable_insights=data.get("actionable_insights") or [],
        confidence=confidence,
        metadata=data.get("metadata") or {},
    )


def summary_result_to_dict(result: SummaryResult) -> dict[str, Any]:
    """Serialise a *SummaryResult* to a plain dict for JSON output."""
    d: dict[str, Any] = {
        "title": result.title,
        "summary": result.summary,
        "key_points": result.key_points,
        "entities": [{"name": e.name, "type": e.type, "description": e.description} for e in result.entities],
        "actionable_insights": result.actionable_insights,
    }
    if result.confidence:
        d["confidence"] = {
            "score": result.confidence.score,
            "rationale": result.confidence.rationale,
            "source_quality": result.confidence.source_quality,
        }
    if result.metadata:
        d["metadata"] = result.metadata
    return d
