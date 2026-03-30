"""Base translator abstraction for LattifAI."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lattifai.caption import Supervision
from lattifai.config.translation import TranslationConfig
from lattifai.llm import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class TranslationPipelineState:
    """In-memory state for continuing from normal mode to refined mode."""

    original_texts: list[str]
    analysis: Optional[dict]
    glossary: Optional[dict[str, str]]
    draft_translations: list


class BaseTranslator:
    """
    LLM-based caption translator.

    Uses a BaseLLMClient for LLM calls; handles batching, context windows,
    and the quick/normal/refined pipeline.
    """

    def __init__(self, config: TranslationConfig, client: BaseLLMClient):
        self.config = config
        self.client = client
        self._last_pipeline_state: Optional[TranslationPipelineState] = None

    @property
    def name(self) -> str:
        """Human-readable name of the translator."""
        return f"{self.client.provider_name}:{self.config.llm.model}"

    async def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and return raw JSON text response."""
        result = await self.client.generate_json(prompt, model=self.config.llm.model)
        # generate_json returns parsed object; re-serialize for backward compat
        return json.dumps(result, ensure_ascii=False)

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
        shared_prompt: Optional[str] = None,
    ) -> list:
        """Translate a batch of texts using the LLM.

        Returns:
            List of translated strings (monolingual) or dicts with
            "original" and "translated" keys (bilingual).
        """
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
            shared_prompt=shared_prompt,
        )

        response_text = await self._call_llm(prompt)
        result = json.loads(response_text)

        if len(result) != len(texts):
            logger.warning("Batch size mismatch: expected %d, got %d. Padding/truncating.", len(texts), len(result))
            if len(result) < len(texts):
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
        from lattifai.translation.prompts import build_shared_translate_prompt

        cfg = config or self.config
        texts = [sup.text or "" for sup in supervisions]
        if not texts:
            return supervisions

        analysis = await self._maybe_analyze(texts, cfg)
        glossary_terms = self._extract_glossary_terms(analysis)
        merged_glossary = self._load_and_merge_glossaries(cfg, glossary_terms)

        shared_prompt = build_shared_translate_prompt(
            target_lang=cfg.target_lang,
            bilingual=cfg.bilingual,
            style=cfg.style,
            analysis=analysis,
            glossary=merged_glossary,
            approach=cfg.approach,
        )

        if cfg.save_artifacts:
            if analysis:
                self._save_artifact(cfg, "01-analysis.json", analysis)
                self._save_artifact(cfg, "01-analysis.md", self._format_analysis_markdown(analysis))
            self._save_artifact(cfg, "02-prompt.md", shared_prompt)

        translated = await self._translate_all_batches(
            texts=texts,
            config=cfg,
            analysis=analysis,
            glossary=merged_glossary,
            shared_prompt=shared_prompt,
        )

        draft_plain = self._extract_plain_translations(translated, cfg.bilingual)
        if cfg.save_artifacts:
            self._save_artifact(cfg, "03-draft.json", {"draft": draft_plain})
            self._save_artifact(cfg, "03-draft.md", self._format_draft_markdown(texts, draft_plain))

        self._last_pipeline_state = TranslationPipelineState(
            original_texts=texts,
            analysis=analysis,
            glossary=merged_glossary,
            draft_translations=translated,
        )

        if cfg.mode == "refined":
            revised_texts, _ = await self._review_draft(
                original_texts=texts,
                draft_translations=draft_plain,
                config=cfg,
                analysis=analysis,
                glossary=merged_glossary,
            )
            translated = self._wrap_translations_for_output(texts, revised_texts, cfg.bilingual)

        self._apply_translations(supervisions, translated, cfg)

        if cfg.save_artifacts:
            final_plain = self._extract_plain_translations(translated, cfg.bilingual)
            self._save_artifact(
                cfg, "translation.md", self._format_draft_markdown(texts, final_plain, title="Final Translation")
            )

        return supervisions

    async def refine_existing_draft(
        self,
        supervisions: list[Supervision],
        config: Optional[TranslationConfig] = None,
        source_texts: Optional[list[str]] = None,
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> list[Supervision]:
        """Run refined review on an existing draft without retranslating."""
        cfg = config or self.config
        state = self._last_pipeline_state

        if source_texts is not None:
            original_texts = source_texts
        elif state and len(state.original_texts) == len(supervisions):
            original_texts = state.original_texts
        elif cfg.bilingual:
            original_texts = [sup.text or "" for sup in supervisions]
        else:
            raise ValueError("source_texts is required for monolingual refinement when no prior pipeline state exists.")

        if cfg.bilingual:
            draft_translations = [str(getattr(sup, "translation", "") or "") for sup in supervisions]
        else:
            draft_translations = [sup.text or "" for sup in supervisions]

        review_analysis = analysis or (state.analysis if state else None)
        if review_analysis is None:
            review_analysis = await self._maybe_analyze(original_texts, cfg)

        glossary_terms = self._extract_glossary_terms(review_analysis)
        review_glossary = (
            glossary or (state.glossary if state else None) or self._load_and_merge_glossaries(cfg, glossary_terms)
        )

        revised_texts, _ = await self._review_draft(
            original_texts=original_texts,
            draft_translations=draft_translations,
            config=cfg,
            analysis=review_analysis,
            glossary=review_glossary,
        )

        translated = self._wrap_translations_for_output(original_texts, revised_texts, cfg.bilingual)
        self._apply_translations(supervisions, translated, cfg)

        if cfg.save_artifacts:
            self._save_artifact(
                cfg,
                "translation.md",
                self._format_draft_markdown(original_texts, revised_texts, title="Final Translation"),
            )

        return supervisions

    async def _maybe_analyze(self, texts: list[str], config: TranslationConfig) -> Optional[dict]:
        """Run analysis only when the mode requires it."""
        if config.mode not in ("normal", "refined"):
            return None

        from lattifai.translation.analyzer import ContentAnalyzer

        analyzer = ContentAnalyzer(self)
        return await analyzer.analyze(
            texts, source_lang=config.source_lang, target_lang=config.target_lang, approach=config.approach
        )

    async def _review_draft(
        self,
        original_texts: list[str],
        draft_translations: list[str],
        config: TranslationConfig,
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[str]]:
        """Run global review and optionally save critique/revision artifacts."""
        from lattifai.translation.reviewer import TranslationReviewer

        reviewer = TranslationReviewer(self)
        outcome = await reviewer.review(
            original_texts=original_texts,
            translated_texts=draft_translations,
            target_lang=config.target_lang,
            analysis=analysis,
            glossary=glossary,
            approach=config.approach,
        )

        if config.save_artifacts:
            self._save_artifact(config, "04-critique.json", {"critiques": outcome.critiques})
            self._save_artifact(
                config,
                "04-critique.md",
                self._format_critique_markdown(
                    original_texts=original_texts,
                    draft_translations=draft_translations,
                    revised_translations=outcome.revised_texts,
                    critiques=outcome.critiques,
                ),
            )
            self._save_artifact(config, "05-revision.json", {"revised": outcome.revised_texts})
            self._save_artifact(
                config,
                "05-revision.md",
                self._format_draft_markdown(original_texts, outcome.revised_texts, title="Revised Translation"),
            )

        return outcome.revised_texts, outcome.critiques

    async def _translate_all_batches(
        self,
        texts: list[str],
        config: TranslationConfig,
        analysis: Optional[dict] = None,
        glossary: Optional[dict[str, str]] = None,
        shared_prompt: Optional[str] = None,
    ) -> list:
        """Translate all texts in batches with concurrency control."""
        from tqdm import tqdm

        batch_size = config.batch_size
        context_lines = config.context_lines
        total_batches = (len(texts) - 1) // batch_size + 1
        semaphore = asyncio.Semaphore(config.max_concurrent)
        pbar = tqdm(total=len(texts), desc="Translating", unit="seg")

        async def _process_batch(start_idx: int) -> tuple[int, list]:
            async with semaphore:
                batch = texts[start_idx : start_idx + batch_size]
                ctx_before = texts[max(0, start_idx - context_lines) : start_idx] if start_idx > 0 else None
                batch_end = start_idx + batch_size
                ctx_after = texts[batch_end : batch_end + context_lines] if batch_end < len(texts) else None

                if config.verbose:
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
                    shared_prompt=shared_prompt,
                )
                pbar.update(len(batch))
                return start_idx, result

        try:
            tasks = [_process_batch(i) for i in range(0, len(texts), batch_size)]
            batch_results = await asyncio.gather(*tasks)
        finally:
            pbar.close()

        batch_results.sort(key=lambda item: item[0])

        results = []
        for _, batch_result in batch_results:
            results.extend(batch_result)
        return results

    @staticmethod
    def _extract_glossary_terms(analysis: Optional[dict]) -> Optional[dict[str, str]]:
        """Extract source->translation pairs from analysis output."""
        if not analysis:
            return None

        glossary_terms: dict[str, str] = {}
        for term in analysis.get("terminology", []):
            if not isinstance(term, dict):
                continue
            source = str(term.get("source", "")).strip()
            translation = str(term.get("translation", "")).strip()
            if source and translation:
                glossary_terms[source] = translation

        return glossary_terms or None

    @staticmethod
    def _load_and_merge_glossaries(
        config: TranslationConfig, analysis_terms: Optional[dict[str, str]]
    ) -> Optional[dict[str, str]]:
        from lattifai.translation.glossary import load_glossary, merge_glossaries

        user_glossary = load_glossary(config.glossary_file)
        if user_glossary or analysis_terms:
            return merge_glossaries(user_glossary, analysis_terms)
        return None

    @staticmethod
    def _extract_plain_translations(translated: list, bilingual: bool) -> list[str]:
        """Normalize translation output into a plain list of translated strings."""
        if not bilingual:
            return [str(item) for item in translated]

        plain: list[str] = []
        for item in translated:
            if isinstance(item, dict):
                plain.append(str(item.get("translated", "")))
            else:
                plain.append(str(item))
        return plain

    @staticmethod
    def _wrap_translations_for_output(original_texts: list[str], translated_texts: list[str], bilingual: bool) -> list:
        """Wrap plain translated text into the output schema expected by apply stage."""
        if not bilingual:
            return translated_texts
        return [{"original": orig, "translated": trans} for orig, trans in zip(original_texts, translated_texts)]

    @staticmethod
    def _apply_translations(supervisions: list[Supervision], translated: list, config: TranslationConfig) -> None:
        """Apply translated values onto supervision objects in-place."""
        for idx, sup in enumerate(supervisions):
            if idx >= len(translated):
                break
            if config.bilingual:
                item = translated[idx]
                if isinstance(item, dict):
                    sup.translation = item.get("translated", "")
                else:
                    sup.translation = str(item)
                sup.target_lang = config.target_lang
            else:
                sup.text = str(translated[idx]) if not isinstance(translated[idx], str) else translated[idx]

    @staticmethod
    def _format_analysis_markdown(analysis: dict) -> str:
        """Render analysis object into a readable markdown artifact."""
        lines = ["# Analysis", ""]

        style = analysis.get("style")
        register = analysis.get("register")
        if style:
            lines.append(f"- Style: {style}")
        if register:
            lines.append(f"- Register: {register}")
        if style or register:
            lines.append("")

        terminology = analysis.get("terminology", [])
        lines.append("## Terminology")
        if terminology:
            for item in terminology:
                if not isinstance(item, dict):
                    continue
                source = item.get("source", "")
                translation = item.get("translation", "")
                context = item.get("context", "")
                bullet = f'- "{source}"'
                if translation:
                    bullet += f' -> "{translation}"'
                if context:
                    bullet += f" ({context})"
                lines.append(bullet)
        else:
            lines.append("- None")
        lines.append("")

        metaphor_map = analysis.get("metaphor_map", [])
        lines.append("## Metaphor Map")
        if metaphor_map:
            for item in metaphor_map:
                if not isinstance(item, dict):
                    continue
                source = item.get("source", "")
                intent = item.get("intent", "")
                strategy = item.get("strategy", "")
                lines.append(f'- "{source}"')
                if intent:
                    lines.append(f"  - intent: {intent}")
                if strategy:
                    lines.append(f"  - strategy: {strategy}")
        else:
            lines.append("- None")
        lines.append("")

        notes = analysis.get("notes")
        if notes:
            lines.append("## Notes")
            lines.append(str(notes))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _format_draft_markdown(
        original_texts: list[str], translated_texts: list[str], title: str = "Draft Translation"
    ) -> str:
        """Render aligned source/target lines as markdown."""
        lines = [f"# {title}", ""]
        for idx, (source, target) in enumerate(zip(original_texts, translated_texts), 1):
            lines.append(f"## Segment {idx}")
            lines.append(f"- Source: {source}")
            lines.append(f"- Translation: {target}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _format_critique_markdown(
        original_texts: list[str],
        draft_translations: list[str],
        revised_translations: list[str],
        critiques: list[str],
    ) -> str:
        """Render review critique details into markdown."""
        lines = ["# Critique", ""]
        for idx, (source, draft, revised, critique) in enumerate(
            zip(original_texts, draft_translations, revised_translations, critiques), 1
        ):
            changed = "yes" if draft != revised else "no"
            lines.append(f"## Segment {idx}")
            lines.append(f"- Source: {source}")
            lines.append(f"- Draft: {draft}")
            lines.append(f"- Revised: {revised}")
            lines.append(f"- Changed: {changed}")
            lines.append(f"- Note: {critique or 'No issue found.'}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _save_artifact(config: TranslationConfig, filename: str, data) -> None:
        """Save an artifact to the artifacts directory."""
        artifacts_dir = Path(config.artifacts_dir) if config.artifacts_dir else Path(".")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        filepath = artifacts_dir / filename

        if isinstance(data, str):
            filepath.write_text(data, encoding="utf-8")
        else:
            filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Saved artifact: %s", filepath)
