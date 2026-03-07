"""Base translator abstraction for LattifAI."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from lattifai.caption import Supervision
from lattifai.config.translation import TranslationConfig

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """
    Base class for LLM-based caption translators.

    Subclasses implement the LLM API call; the base class handles
    batching, context windows, and the quick/normal/refined pipeline.
    """

    def __init__(self, config: TranslationConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the translator."""

    @abstractmethod
    async def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and return raw text response.

        The response is expected to be valid JSON.
        """

    async def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        bilingual: bool,
        style: str = "storytelling",
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
        context_before: Optional[list[str]] = None,
        context_after: Optional[list[str]] = None,
    ) -> list:
        """Translate a batch of texts using the LLM.

        Returns:
            List of translated strings (monolingual) or dicts with
            "original" and "translated" keys (bilingual).
        """
        import json

        from lattifai.translation.prompts import build_translate_prompt

        prompt = build_translate_prompt(
            texts=texts,
            target_lang=target_lang,
            bilingual=bilingual,
            style=style,
            analysis=analysis,
            glossary=glossary,
            context_before=context_before,
            context_after=context_after,
        )

        response_text = await self._call_llm(prompt)
        result = json.loads(response_text)

        if len(result) != len(texts):
            logger.warning("Batch size mismatch: expected %d, got %d. Padding/truncating.", len(texts), len(result))
            if len(result) < len(texts):
                # Pad with originals
                for i in range(len(result), len(texts)):
                    result.append({"original": texts[i], "translated": texts[i]} if bilingual else texts[i])
            else:
                result = result[: len(texts)]

        return result

    async def translate_captions(
        self,
        supervisions: list[Supervision],
        config: Optional[TranslationConfig] = None,
    ) -> list[Supervision]:
        """Main translation pipeline: dispatches to quick/normal/refined.

        Args:
            supervisions: Input caption segments.
            config: Override config (uses self.config if None).

        Returns:
            Updated supervisions with translations applied.
        """
        cfg = config or self.config

        texts = [sup.text or "" for sup in supervisions]

        if not texts:
            return supervisions

        # Step 1: Analysis (normal/refined only)
        analysis = None
        glossary_terms = None
        if cfg.mode in ("normal", "refined"):
            from lattifai.translation.analyzer import ContentAnalyzer

            analyzer = ContentAnalyzer(self)
            analysis = await analyzer.analyze(texts, source_lang=cfg.source_lang)

            if cfg.save_artifacts and analysis:
                self._save_artifact(cfg, "01-analysis.json", analysis)

            # Extract translated terms from analysis if available
            if analysis and analysis.get("terminology"):
                glossary_terms = {}
                for term in analysis["terminology"]:
                    if "translation" in term:
                        glossary_terms[term["source"]] = term["translation"]

        # Load and merge glossary
        from lattifai.translation.glossary import load_glossary, merge_glossaries

        user_glossary = load_glossary(cfg.glossary_file)
        merged_glossary = merge_glossaries(user_glossary, glossary_terms) if (user_glossary or glossary_terms) else None

        # Step 2: Batch translation
        translated = await self._translate_all_batches(
            texts=texts,
            config=cfg,
            analysis=analysis,
            glossary=merged_glossary,
        )

        # Step 3: Review (refined only)
        if cfg.mode == "refined":
            from lattifai.translation.reviewer import TranslationReviewer

            reviewer = TranslationReviewer(self)

            # Extract plain translated texts for review
            if cfg.bilingual:
                plain_translations = [
                    item.get("translated", "") if isinstance(item, dict) else item for item in translated
                ]
            else:
                plain_translations = translated

            revised = await reviewer.review(
                original_texts=texts,
                translated_texts=plain_translations,
                target_lang=cfg.target_lang,
                analysis=analysis,
                glossary=merged_glossary,
            )

            if cfg.save_artifacts:
                self._save_artifact(cfg, "03-review.json", {"revised": revised})

            # Update translated with revised versions
            if cfg.bilingual:
                translated = [{"original": orig, "translated": rev} for orig, rev in zip(texts, revised)]
            else:
                translated = revised

        # Apply translations to supervisions
        for idx, sup in enumerate(supervisions):
            if idx >= len(translated):
                break
            if cfg.bilingual:
                item = translated[idx]
                if isinstance(item, dict):
                    sup.translation = item.get("translated", "")
                    sup.target_lang = cfg.target_lang
                else:
                    sup.translation = str(item)
                    sup.target_lang = cfg.target_lang
            else:
                sup.text = str(translated[idx]) if not isinstance(translated[idx], str) else translated[idx]

        return supervisions

    async def _translate_all_batches(
        self,
        texts: list[str],
        config: TranslationConfig,
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> list:
        """Translate all texts in batches with concurrency control."""
        batch_size = config.batch_size
        context_lines = config.context_lines
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def _process_batch(start_idx: int) -> tuple[int, list]:
            async with semaphore:
                batch = texts[start_idx : start_idx + batch_size]
                ctx_before = texts[max(0, start_idx - context_lines) : start_idx] if start_idx > 0 else None
                batch_end = start_idx + batch_size
                ctx_after = texts[batch_end : batch_end + context_lines] if batch_end < len(texts) else None

                if config.verbose:
                    total_batches = (len(texts) - 1) // batch_size + 1
                    batch_num = start_idx // batch_size + 1
                    logger.info("Translating batch %d/%d (%d segments)...", batch_num, total_batches, len(batch))

                result = await self.translate_batch(
                    texts=batch,
                    target_lang=config.target_lang,
                    bilingual=config.bilingual,
                    style=config.style,
                    analysis=analysis,
                    glossary=glossary,
                    context_before=ctx_before,
                    context_after=ctx_after,
                )
                return start_idx, result

        # Create tasks for all batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            tasks.append(_process_batch(i))

        # Run concurrently and collect results in order
        batch_results = await asyncio.gather(*tasks)

        # Sort by start index and flatten
        batch_results.sort(key=lambda x: x[0])
        results = []
        for _, batch_result in batch_results:
            results.extend(batch_result)

        return results

    @staticmethod
    def _save_artifact(config: TranslationConfig, filename: str, data) -> None:
        """Save an artifact to the artifacts directory."""
        import json
        from pathlib import Path

        artifacts_dir = Path(config.artifacts_dir) if config.artifacts_dir else Path(".")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        filepath = artifacts_dir / filename
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved artifact: %s", filepath)
