"""Translation reviewer for refined mode."""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from lattifai.translation.prompts import build_review_prompt

if TYPE_CHECKING:
    from lattifai.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# Maximum segments per review batch
REVIEW_BATCH_SIZE = 100


@dataclass
class ReviewOutcome:
    """Review result with revised texts and per-line critique notes."""

    revised_texts: list[str]
    critiques: list[str]


class TranslationReviewer:
    """Reviews translations for consistency and quality.

    Used in refined mode as a post-translation quality pass.
    Checks terminology consistency, referential coherence, and natural expression.
    """

    def __init__(self, translator: "BaseTranslator"):
        self.translator = translator

    async def review(
        self,
        original_texts: list[str],
        translated_texts: list[str],
        target_lang: str,
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> ReviewOutcome:
        """Review and revise translations.

        Args:
            original_texts: Source language texts.
            translated_texts: Current translations.
            target_lang: Target language code.
            analysis: Content analysis from analyzer.
            glossary: Merged glossary.

        Returns:
            ReviewOutcome with revised translations and critique notes.
        """
        if len(original_texts) != len(translated_texts):
            raise ValueError("original_texts and translated_texts must have the same length")

        all_revised: list[str] = []
        all_critiques: list[str] = []

        # Process in batches to avoid token limits
        for i in range(0, len(original_texts), REVIEW_BATCH_SIZE):
            batch_originals = original_texts[i : i + REVIEW_BATCH_SIZE]
            batch_translations = translated_texts[i : i + REVIEW_BATCH_SIZE]

            prompt = build_review_prompt(
                original_texts=batch_originals,
                translated_texts=batch_translations,
                target_lang=target_lang,
                analysis=analysis,
                glossary=glossary,
            )

            try:
                response_text = await self.translator._call_llm(prompt)
                revised, critiques = self._parse_review_response(response_text, batch_translations)

                if len(revised) != len(batch_originals):
                    logger.warning(
                        "Review batch size mismatch: expected %d, got %d. Using originals for missing.",
                        len(batch_originals),
                        len(revised),
                    )
                    while len(revised) < len(batch_originals):
                        revised.append(batch_translations[len(revised)])
                    while len(critiques) < len(batch_originals):
                        critiques.append("")

                all_revised.extend(revised[: len(batch_originals)])
                all_critiques.extend(critiques[: len(batch_originals)])

            except Exception as e:
                logger.warning("Review failed for batch %d (keeping original translations): %s", i, e)
                all_revised.extend(batch_translations)
                all_critiques.extend(["Review failed; kept draft translation."] * len(batch_translations))

        logger.info("Translation review complete: %d segments", len(all_revised))
        return ReviewOutcome(revised_texts=all_revised, critiques=all_critiques)

    @staticmethod
    def _parse_review_response(response_text: str, fallback_translations: list[str]) -> tuple[list[str], list[str]]:
        """Parse review output with backward compatibility for legacy schemas."""
        parsed = json.loads(response_text)

        # Legacy schema: ["revised-1", "revised-2", ...]
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed, [""] * len(parsed)

        items = None
        if isinstance(parsed, dict):
            items = parsed.get("items")
        elif isinstance(parsed, list):
            items = parsed

        if not isinstance(items, list):
            raise ValueError("Invalid review response schema: expected list or object with 'items'")

        revised: list[str] = []
        critiques: list[str] = []

        for idx, item in enumerate(items):
            if isinstance(item, str):
                revised.append(item)
                critiques.append("")
                continue

            if not isinstance(item, dict):
                fallback = fallback_translations[idx] if idx < len(fallback_translations) else ""
                revised.append(fallback)
                critiques.append("Invalid item schema; kept draft translation.")
                continue

            revised_text = (
                item.get("revised")
                or item.get("translated")
                or item.get("translation")
                or (fallback_translations[idx] if idx < len(fallback_translations) else "")
            )
            critique = item.get("critique") or item.get("reason") or ""
            revised.append(str(revised_text))
            critiques.append(str(critique))

        return revised, critiques
