"""Content analyzer for translation preparation."""

import json
import logging
from typing import TYPE_CHECKING, Optional

from lattifai.translation.prompts import build_analysis_prompt

if TYPE_CHECKING:
    from lattifai.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# Maximum characters to send for analysis (avoid token limits)
MAX_ANALYSIS_CHARS = 50000


class ContentAnalyzer:
    """Analyzes caption content to extract terminology, style, and register.

    Used in normal and refined translation modes to improve consistency.
    """

    def __init__(self, translator: "BaseTranslator"):
        self.translator = translator

    async def analyze(
        self,
        texts: list[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        approach: str = "rewrite",
    ) -> Optional[dict]:
        """Analyze full caption text to prepare for translation/rewriting.

        Args:
            texts: All caption segment texts.
            source_lang: Source language code (optional).
            target_lang: Target language code (optional, improves analysis quality).
            approach: 'rewrite' for natural expression, 'translate' for source fidelity.

        Returns:
            Analysis dict with terminology, metaphor_map, cultural_notes, style, register, speakers, notes.
            Returns None if analysis fails.
        """
        full_text = "\n".join(texts)

        # Truncate if too long
        if len(full_text) > MAX_ANALYSIS_CHARS:
            logger.info("Text too long for analysis (%d chars), truncating to %d", len(full_text), MAX_ANALYSIS_CHARS)
            full_text = full_text[:MAX_ANALYSIS_CHARS]

        prompt = build_analysis_prompt(full_text, source_lang, target_lang, approach=approach)

        try:
            response_text = await self.translator._call_llm(prompt)
            analysis = json.loads(response_text)

            term_count = len(analysis.get("terminology", []))
            style = analysis.get("style", "unknown")
            logger.info("Content analysis complete: %d terms, style=%s", term_count, style)

            return analysis
        except Exception as e:
            logger.warning("Content analysis failed (continuing without): %s", e)
            return None
