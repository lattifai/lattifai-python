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
class SummaryChapter:
    """A topic segment with narrative summary and optional quotes."""

    title: str
    start: float
    end: float = 0.0
    summary: str = ""
    quotes: list[str] = field(default_factory=list)


@dataclass
class SummaryResult:
    """Structured summary output produced by ContentSummarizer."""

    title: str
    overview: str = ""
    chapters: list[SummaryChapter] = field(default_factory=list)
    entities: list[SummaryEntity] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    seo_title: str = ""
    seo_description: str = ""
    confidence: SummaryConfidence | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Backward-compatible: join chapter summaries into a flat summary."""
        return "\n\n".join(ch.summary for ch in self.chapters if ch.summary)


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

    # Tags
    tags_raw = data.get("tags") or []
    tags = [str(t).lower() for t in tags_raw if isinstance(t, str)]

    # Chapters (now the core structure with narrative content)
    chapters_raw = data.get("chapters") or []
    chapters = [
        SummaryChapter(
            title=ch.get("title", ""),
            start=float(ch.get("start", 0)),
            end=float(ch.get("end", 0)),
            summary=ch.get("summary", ""),
            quotes=ch.get("quotes") or [],
        )
        for ch in chapters_raw
        if isinstance(ch, dict)
    ]
    # Fill end times: each chapter ends where the next begins
    for i in range(len(chapters) - 1):
        if chapters[i].end == 0:
            chapters[i].end = chapters[i + 1].start

    return SummaryResult(
        title=data.get("title") or fallback_title,
        overview=data.get("overview", ""),
        chapters=chapters,
        entities=entities,
        tags=tags,
        seo_title=data.get("seo_title") or "",
        seo_description=data.get("seo_description") or "",
        confidence=confidence,
        metadata=data.get("metadata") or {},
    )


def summary_result_to_dict(result: SummaryResult) -> dict[str, Any]:
    """Serialise a *SummaryResult* to a plain dict for JSON output."""
    d: dict[str, Any] = {
        "title": result.title,
        "overview": result.overview,
        "chapters": [
            {
                "title": c.title,
                "start": c.start,
                "end": c.end,
                "summary": c.summary,
                **({"quotes": c.quotes} if c.quotes else {}),
            }
            for c in result.chapters
        ],
        "entities": [{"name": e.name, "type": e.type, "description": e.description} for e in result.entities],
        "tags": result.tags,
        "seo_title": result.seo_title,
        "seo_description": result.seo_description,
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
