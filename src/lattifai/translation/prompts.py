"""Translation prompt templates for LattifAI.

Core principles adapted from baoyu-translate:
- Rewrite, don't translate -- express in the target language naturally
- Paraphrase over literal -- translate the author's intent, not word-for-word
- Interpret metaphors -- translate by intent, not surface imagery
- Preserve emotion -- keep the emotional coloring of the original
- Sound natural -- use idiomatic expressions in the target language

Caption-specific principles:
- Keep it concise -- captions must convey meaning in limited space
- Stay conversational -- captions represent spoken language
- Speaker consistency -- same speaker should have consistent translation style

Reference: https://x.com/dotey/status/2029969547927658673
"""

import json
from typing import Optional

from lattifai.languages import LANGUAGE_NAMES, get_language_name  # noqa: F401


def build_analysis_prompt(full_text: str, source_lang: Optional[str] = None) -> str:
    """Build prompt for content analysis (normal/refined modes).

    Extracts terminology, speaking style, and register from the full caption text.
    """
    lang_hint = f"The source language is {get_language_name(source_lang)}. " if source_lang else ""

    return f"""Analyze this caption text for translation preparation. {lang_hint}

Extract and return a JSON object with these fields:
1. "terminology": Array of objects with "source" and "context" keys \
-- technical terms, proper nouns, brand names, person names that need consistent translation
2. "style": One of "formal", "casual", "mixed", "technical", "narrative" -- the dominant speaking style
3. "register": Brief description of the language register (e.g. "academic lecture", "casual vlog", "news broadcast")
4. "speakers": Array of distinct speaker styles detected (if any), each with "id" and "style" description
5. "notes": Any other observations relevant for translation quality

Caption text:
{full_text}"""


def build_translate_prompt(
    texts: list[str],
    target_lang: str,
    bilingual: bool,
    style: str = "storytelling",
    analysis: Optional[dict] = None,
    glossary: Optional[dict[str, str]] = None,
    context_before: Optional[list[str]] = None,
    context_after: Optional[list[str]] = None,
) -> str:
    """Build the translation prompt for a batch of caption segments.

    Args:
        texts: Lines to translate.
        target_lang: Target language code or name.
        bilingual: If True, return both original and translated text.
        style: Translation style hint.
        analysis: Content analysis result (from analyzer).
        glossary: Term -> translation mapping.
        context_before: Preceding lines for context (not translated).
        context_after: Following lines for context (not translated).

    Returns:
        Fully assembled prompt string.
    """
    lang_name = get_language_name(target_lang)

    # Core translation principles
    principles = [
        "Rewrite in the target language rather than translate word-for-word",
        "Translate the speaker's intent, not literal words",
        "Interpret metaphors and idioms by meaning, not surface imagery",
        "Preserve the emotional tone and coloring of the original",
        "Use natural, idiomatic expressions in the target language",
        "Keep captions concise -- they must fit limited display space",
        "Maintain conversational tone for spoken content",
        "Keep the exact same order and count as input lines",
    ]

    # Build instruction
    if bilingual:
        output_format = 'Return a JSON array of objects, each with "original" and "translated" keys.'
    else:
        output_format = "Return a JSON array of translated strings."

    instruction = f"Translate captions to {lang_name}. {output_format}\n"
    instruction += f"\nStyle: {style}\n"
    instruction += "\nTranslation principles:\n"
    for i, p in enumerate(principles, 1):
        instruction += f"{i}. {p}\n"

    # Add analysis context if available
    if analysis:
        if analysis.get("terminology"):
            instruction += "\nKey terminology (translate consistently):\n"
            for term in analysis["terminology"]:
                instruction += f'- "{term["source"]}": {term.get("context", "")}\n'
        if analysis.get("style"):
            instruction += f'\nDetected style: {analysis["style"]}\n'
        if analysis.get("register"):
            instruction += f'Register: {analysis["register"]}\n'

    # Add glossary if available
    if glossary:
        instruction += "\nGlossary (use these exact translations):\n"
        for source, target in glossary.items():
            instruction += f'- "{source}" -> "{target}"\n'

    instruction += '\nIMPORTANT: ONLY translate the lines in "to_translate". Context lines are for reference only.'

    # Build input data
    input_data: dict = {"to_translate": texts}
    if context_before:
        input_data["context_before"] = context_before
    if context_after:
        input_data["context_after"] = context_after

    return f"{instruction}\n\nInput:\n{json.dumps(input_data, ensure_ascii=False)}"


def build_review_prompt(
    original_texts: list[str],
    translated_texts: list[str],
    target_lang: str,
    analysis: Optional[dict] = None,
    glossary: Optional[dict[str, str]] = None,
) -> str:
    """Build prompt for translation review (refined mode).

    Reviews the full translation for consistency and quality.
    """
    lang_name = get_language_name(target_lang)

    instruction = (
        f"Review these {lang_name} caption translations for quality. "
        "Return a JSON array of revised translations (same length as input)."
    )
    instruction += """

Review criteria:
1. Terminology consistency -- same source term should have the same translation throughout
2. Referential coherence -- pronouns and references should be clear across segments
3. Natural expression -- translations should sound like native captions
4. Conciseness -- keep translations brief and suitable for caption display
5. Emotional fidelity -- translations should preserve the original tone

If a translation is already good, keep it unchanged. Only revise where improvement is needed.
"""

    if analysis and analysis.get("terminology"):
        instruction += "\nKey terminology for consistency check:\n"
        for term in analysis["terminology"]:
            instruction += f'- "{term["source"]}": {term.get("context", "")}\n'

    if glossary:
        instruction += "\nRequired glossary translations:\n"
        for source, target in glossary.items():
            instruction += f'- "{source}" -> "{target}"\n'

    # Build paired input
    pairs = []
    for orig, trans in zip(original_texts, translated_texts):
        pairs.append({"original": orig, "translated": trans})

    return f"{instruction}\n\nTranslations to review:\n{json.dumps(pairs, ensure_ascii=False)}"
