"""Translation reviewer for refined mode."""

import json
import logging
from typing import TYPE_CHECKING, Optional

from lattifai.translation.prompts import build_review_prompt

if TYPE_CHECKING:
    from lattifai.translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# Maximum segments per review batch
REVIEW_BATCH_SIZE = 100


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
    ) -> list[str]:
        """Review and revise translations.

        Args:
            original_texts: Source language texts.
            translated_texts: Current translations.
            target_lang: Target language code.
            analysis: Content analysis from analyzer.
            glossary: Merged glossary.

        Returns:
            Revised translations (same length as input).
        """
        if len(original_texts) != len(translated_texts):
            raise ValueError("original_texts and translated_texts must have the same length")

        all_revised: list[str] = []

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
                revised = json.loads(response_text)

                if len(revised) != len(batch_originals):
                    logger.warning(
                        "Review batch size mismatch: expected %d, got %d. Using originals for missing.",
                        len(batch_originals),
                        len(revised),
                    )
                    while len(revised) < len(batch_originals):
                        revised.append(batch_translations[len(revised)])

                all_revised.extend(revised[: len(batch_originals)])

            except Exception as e:
                logger.warning("Review failed for batch %d (keeping original translations): %s", i, e)
                all_revised.extend(batch_translations)

        logger.info("Translation review complete: %d segments", len(all_revised))
        return all_revised
