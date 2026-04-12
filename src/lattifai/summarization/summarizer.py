"""Generic content summariser backed by an LLM."""

from __future__ import annotations

import logging
from typing import Any

from lattifai.config.summarization import SummarizationConfig
from lattifai.llm.base import BaseLLMClient
from lattifai.summarization.prompts import (
    SYSTEM_PROMPT,
    build_reduce_user_prompt,
    build_summary_user_prompt,
    resolve_auto_length,
)
from lattifai.summarization.schema import SummaryConfidence, SummaryInput, SummaryResult, summary_result_from_dict

logger = logging.getLogger(__name__)


class ContentSummarizer:
    """Source-agnostic structured summariser.

    Accepts a :class:`SummaryInput` and produces a :class:`SummaryResult`
    by calling an LLM via ``generate_json``.  Long inputs are handled with
    a map-reduce strategy: each chunk is summarised independently, then the
    partial results are merged into a single output.
    """

    def __init__(self, config: SummarizationConfig, client: BaseLLMClient) -> None:
        self._config = config
        self._client = client

    @property
    def name(self) -> str:
        return f"ContentSummarizer[{self._client.provider_name}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarize(self, summary_input: SummaryInput) -> SummaryResult:
        """Summarise *summary_input* and return a validated result."""
        # Resolve "auto" length once based on input size.
        if self._config.length == "auto":
            self._resolved_length = resolve_auto_length(len(summary_input.text))
            logger.info("Auto length: %d chars → %s", len(summary_input.text), self._resolved_length)
        else:
            self._resolved_length = self._config.length

        if self._needs_chunking(summary_input.text):
            logger.info("Input exceeds %d chars — using map-reduce", self._config.max_input_chars)
            result = await self._summarize_map_reduce(summary_input)
        else:
            result = await self._summarize_single(summary_input)

        return self._adjust_confidence(
            result,
            source_type=summary_input.source_type,
            text_length=len(summary_input.text),
            used_chunking=self._needs_chunking(summary_input.text),
        )

    # ------------------------------------------------------------------
    # Single-pass summarisation
    # ------------------------------------------------------------------

    async def _summarize_single(self, summary_input: SummaryInput) -> SummaryResult:
        prompt = build_summary_user_prompt(
            summary_input,
            lang=self._config.lang,
            length=self._resolved_length,
            include_metadata=self._config.include_metadata,
            include_chapters=self._config.include_chapters,
        )
        data = await self._call_llm(prompt)
        return summary_result_from_dict(data, fallback_title=summary_input.title)

    # ------------------------------------------------------------------
    # Map-reduce summarisation
    # ------------------------------------------------------------------

    async def _summarize_map_reduce(self, summary_input: SummaryInput) -> SummaryResult:
        chunks = self._split_text(summary_input.text)
        total = len(chunks)
        logger.info("Split into %d chunks (chunk_chars=%d)", total, self._config.chunk_chars)

        partial_results: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_input = SummaryInput(
                title=summary_input.title,
                text=chunk_text,
                metadata=summary_input.metadata,
                chapters=summary_input.chapters,
                source_type=summary_input.source_type,
                source_lang=summary_input.source_lang,
            )
            prompt = build_summary_user_prompt(
                chunk_input,
                lang=self._config.lang,
                length=self._resolved_length,
                include_metadata=self._config.include_metadata and idx == 0,
                include_chapters=self._config.include_chapters and idx == 0,
                chunk_index=idx,
                total_chunks=total,
            )
            data = await self._call_llm(prompt)
            partial_results.append(data)
            if self._config.verbose:
                logger.info("Chunk %d/%d summarised", idx + 1, total)

        # Reduce
        reduce_prompt = build_reduce_user_prompt(
            partial_results,
            title=summary_input.title,
            lang=self._config.lang,
            length=self._resolved_length,
            source_type=summary_input.source_type,
        )
        merged = await self._call_llm(reduce_prompt)
        result = summary_result_from_dict(merged, fallback_title=summary_input.title)
        result.metadata["chunked"] = True
        result.metadata["chunks_used"] = total
        return result

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM and return the parsed JSON dict."""
        data = await self._client.generate_json(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self._config.temperature,
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected dict from LLM, got {type(data).__name__}")
        return data

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _needs_chunking(self, text: str) -> bool:
        return len(text) > self._config.max_input_chars

    def _split_text(self, text: str) -> list[str]:
        """Split *text* into overlapping chunks respecting paragraph boundaries."""
        chunk_size = self._config.chunk_chars
        overlap = self._config.overlap_chars
        max_chunks = self._config.max_chunks

        paragraphs = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 1  # +1 for newline
            if current_len + para_len > chunk_size and current:
                chunks.append("\n".join(current))
                # Keep overlap from the tail of current chunk
                overlap_lines: list[str] = []
                overlap_len = 0
                for line in reversed(current):
                    if overlap_len + len(line) + 1 > overlap:
                        break
                    overlap_lines.insert(0, line)
                    overlap_len += len(line) + 1
                current = overlap_lines
                current_len = overlap_len
            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n".join(current))

        # Safety cap
        if len(chunks) > max_chunks:
            logger.warning(
                "Chunk count %d exceeds max_chunks %d — sampling evenly",
                len(chunks),
                max_chunks,
            )
            step = len(chunks) / max_chunks
            chunks = [chunks[int(i * step)] for i in range(max_chunks)]

        return chunks

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def _adjust_confidence(
        self,
        result: SummaryResult,
        *,
        source_type: str,
        text_length: int,
        used_chunking: bool,
    ) -> SummaryResult:
        """Override the LLM-reported confidence with rule-based adjustments."""
        # Base score by source type
        base_scores = {"captions": 0.85, "mixed": 0.75, "metadata": 0.45}
        score = base_scores.get(source_type, 0.60)

        # Adjustments
        adjustments: list[str] = []
        if used_chunking:
            score -= 0.05
            adjustments.append("chunked")
        if text_length < 500:
            score -= 0.15
            adjustments.append("very short input")
        if result.metadata.get("truncated"):
            score -= 0.10
            adjustments.append("truncated")

        score = max(0.05, min(0.95, score))

        # Build rationale
        quality_map = {
            "captions": "high" if text_length >= 500 else "low",
            "mixed": "medium",
            "metadata": "low",
        }
        source_quality = quality_map.get(source_type, "medium")

        rationale_parts = [f"Based on {source_type} source"]
        if adjustments:
            rationale_parts.append(f"adjusted for: {', '.join(adjustments)}")
        rationale = "; ".join(rationale_parts) + "."

        # Preserve any model rationale as a suffix
        if result.confidence and result.confidence.rationale:
            rationale += f" Model note: {result.confidence.rationale}"

        result.confidence = SummaryConfidence(
            score=round(score, 2),
            rationale=rationale,
            source_quality=source_quality,
        )
        return result
