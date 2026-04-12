"""Generic content summarisation module for LattifAI."""

from lattifai.summarization.schema import (
    SummaryChapter,
    SummaryConfidence,
    SummaryEntity,
    SummaryInput,
    SummaryResult,
    summary_result_from_dict,
    summary_result_to_dict,
)
from lattifai.summarization.summarizer import ContentSummarizer

__all__ = [
    "ContentSummarizer",
    "SummaryChapter",
    "SummaryConfidence",
    "SummaryEntity",
    "SummaryInput",
    "SummaryResult",
    "summary_result_from_dict",
    "summary_result_to_dict",
]
