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


def build_analysis_prompt(
    full_text: str,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    approach: str = "rewrite",
) -> str:
    """Build prompt for content analysis (normal/refined modes).

    Extracts terminology, metaphor handling hints, speaking style, and register.

    Args:
        approach: 'rewrite' frames as natural expression preparation;
                  'translate' frames as accuracy-oriented preparation.
    """
    lang_hint = f"The source language is {get_language_name(source_lang)}. " if source_lang else ""
    target_hint = f" in {get_language_name(target_lang)}" if target_lang else ""

    if approach == "rewrite":
        task_desc = f"Analyze this caption text to prepare for rewriting it{target_hint}. {lang_hint}\n"
        task_desc += (
            "Focus on elements that would cause problems if translated literally: metaphors, "
            "culturally-specific references, emotional nuances, and domain-specific terms."
        )
        term_guidance = (
            "only include terms the model might get wrong or translate inconsistently. "
            'Skip obvious terms (e.g. "machine learning" -> "机器学习"). '
            "Focus on ambiguous, domain-specific, or culturally loaded terms."
        )
        metaphor_guidance = (
            "identify idioms/metaphors where literal translation would sound unnatural. "
            "For each, describe the speaker's actual intent and suggest a rewriting strategy."
        )
        quality_label = "rewriting"
    else:
        task_desc = f"Analyze this caption text to prepare for translating it{target_hint}. {lang_hint}\n"
        task_desc += (
            "Focus on elements that require careful handling: technical terms, proper nouns, "
            "ambiguous phrases, and domain-specific vocabulary."
        )
        term_guidance = (
            "include all technical terms, proper nouns, brand names, and person names "
            "that need consistent translation throughout the text."
        )
        metaphor_guidance = (
            "identify idioms and figurative expressions. "
            "For each, note the literal meaning and the intended meaning to ensure accurate translation."
        )
        quality_label = "translation"

    return f"""{task_desc}

Extract and return a JSON object with these fields:
1. "terminology": Array of objects with "source", "translation", and "context" keys \
-- {term_guidance}
2. "metaphor_map": Array of objects with "source", "intent", "literal_risk", and "strategy" keys \
-- {metaphor_guidance}
3. "cultural_notes": Array of short notes for references that may need adaptation or \
explanation for the target audience
4. "style": One of "formal", "casual", "mixed", "technical", "narrative" -- the dominant speaking style
5. "register": Brief description of the language register (e.g. "academic lecture", "casual vlog", "news broadcast")
6. "speakers": Array of distinct speaker styles detected (if any), each with "id" and "style" description
7. "notes": Any other observations relevant for {quality_label} quality

Caption text:
{full_text}"""


_REWRITE_PRINCIPLES = [
    "Rewrite in the target language rather than translate word-for-word",
    "Translate the speaker's intent, not literal words",
    "Interpret metaphors and idioms by meaning, not surface imagery",
    "Preserve the emotional tone and coloring of the original",
    "Use natural, idiomatic expressions in the target language",
    "Keep captions concise -- they must fit limited display space",
    "Maintain conversational tone for spoken content",
    "Keep the exact same order and count as input lines",
    "Keep well-known technical brand and product names in their original English form "
    "(e.g. Wolfram Alpha, Mathematica, GitHub, PyTorch) -- do NOT transliterate them",
    "Output ONLY the translation -- never add translator notes, annotations, or "
    "meta-commentary such as '(原文可能存在拼写错误)', '(译者注:…)' or '[unclear in source]'; "
    "if the source is ambiguous, make the most reasonable interpretation and translate directly",
]

_TRANSLATE_PRINCIPLES = [
    "Translate accurately, preserving the original meaning and structure",
    "Keep technical terms and proper nouns consistent throughout",
    "Preserve the sentence structure when it doesn't hinder clarity",
    "Translate idioms and metaphors faithfully, adding brief clarification if needed",
    "Maintain the original tone and register",
    "Keep captions concise -- they must fit limited display space",
    "Preserve speaker distinctions and conversational markers",
    "Keep the exact same order and count as input lines",
    "Keep well-known technical brand and product names in their original English form "
    "(e.g. Wolfram Alpha, Mathematica, GitHub, PyTorch) -- do NOT transliterate them",
    "Output ONLY the translation -- never add translator notes, annotations, or "
    "meta-commentary such as '(原文可能存在拼写错误)', '(译者注:…)' or '[unclear in source]'; "
    "if the source is ambiguous, make the most reasonable interpretation and translate directly",
]


def _get_principles(approach: str = "rewrite") -> list[str]:
    return _REWRITE_PRINCIPLES if approach == "rewrite" else _TRANSLATE_PRINCIPLES


def build_shared_translate_prompt(
    target_lang: str,
    bilingual: bool,
    style: str = "storytelling",
    analysis: Optional[dict] = None,
    glossary: Optional[dict[str, str]] = None,
    approach: str = "rewrite",
) -> str:
    """Build the shared translation context (without task-specific input lines).

    Args:
        approach: 'rewrite' for natural expression, 'translate' for source fidelity.
    """
    lang_name = get_language_name(target_lang)
    output_format = (
        'Return a JSON array of objects, each with "original" and "translated" keys.'
        if bilingual
        else "Return a JSON array of translated strings."
    )

    if approach == "rewrite":
        instruction = f"Rewrite these captions in {lang_name}. {output_format}\n"
        principles_label = "Rewriting principles"
    else:
        instruction = f"Translate these captions into {lang_name}. {output_format}\n"
        principles_label = "Translation principles"

    instruction += f"\nStyle: {style}\n"
    instruction += f"\n{principles_label}:\n"
    for i, principle in enumerate(_get_principles(approach), 1):
        instruction += f"{i}. {principle}\n"

    if analysis:
        if analysis.get("terminology"):
            instruction += "\nKey terminology (translate consistently):\n"
            for term in analysis["terminology"]:
                instruction += f'- "{term.get("source", "")}"'
                if term.get("translation"):
                    instruction += f' -> "{term["translation"]}"'
                if term.get("context"):
                    instruction += f' ({term["context"]})'
                instruction += "\n"
        if analysis.get("metaphor_map"):
            instruction += "\nMetaphor and idiom handling guidance:\n"
            for item in analysis["metaphor_map"]:
                source = item.get("source", "")
                intent = item.get("intent", "")
                strategy = item.get("strategy", "")
                instruction += f'- "{source}": intent={intent}; strategy={strategy}\n'
        if analysis.get("style"):
            instruction += f'\nDetected style: {analysis["style"]}\n'
        if analysis.get("register"):
            instruction += f'Register: {analysis["register"]}\n'

    if glossary:
        instruction += "\nGlossary (use these exact translations):\n"
        for source, target in glossary.items():
            instruction += f'- "{source}" -> "{target}"\n'

    return instruction.strip()


def build_translate_prompt(
    texts: list[str],
    target_lang: str,
    bilingual: bool,
    style: str = "storytelling",
    analysis: Optional[dict] = None,
    glossary: Optional[dict[str, str]] = None,
    context_before: Optional[list[str]] = None,
    context_after: Optional[list[str]] = None,
    shared_prompt: Optional[str] = None,
    approach: str = "rewrite",
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
        shared_prompt: Optional pre-built shared prompt context.
        approach: 'rewrite' for natural expression, 'translate' for source fidelity.

    Returns:
        Fully assembled prompt string.
    """
    instruction = shared_prompt or build_shared_translate_prompt(
        target_lang=target_lang,
        bilingual=bilingual,
        style=style,
        analysis=analysis,
        glossary=glossary,
        approach=approach,
    )
    verb = "rewrite" if approach == "rewrite" else "translate"
    n = len(texts)
    instruction += f'\n\nIMPORTANT: ONLY {verb} the lines in "to_translate". Context lines are for reference only.'
    instruction += (
        f"\n\nYou MUST return a JSON array with EXACTLY {n} elements — one per input line, "
        f"in the same order. Do NOT skip, merge, or omit any line."
    )

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
    approach: str = "rewrite",
) -> str:
    """Build prompt for translation review (refined mode).

    Reviews the full translation for consistency and quality.

    Args:
        approach: 'rewrite' emphasizes natural expression; 'translate' emphasizes accuracy.
    """
    lang_name = get_language_name(target_lang)

    task_word = "rewrites" if approach == "rewrite" else "translations"
    instruction = (
        f"Review these {lang_name} caption {task_word} for quality. " 'Return a JSON object with one key: "items".'
    )

    if approach == "rewrite":
        instruction += """

Review criteria:
1. Terminology consistency -- same source term should have the same translation throughout
2. Referential coherence -- pronouns and references should be clear across segments
3. Cross-segment coherence -- read segments as a continuous narrative; \
check that tone, tense, and referring expressions flow naturally from one segment to the next
4. Natural expression -- translations should sound like native captions, not translated text
5. Conciseness -- keep translations brief and suitable for caption display
6. Emotional fidelity -- translations should preserve the original tone and coloring; \
"alarming" is not just "惊人的" but carries unease
7. Metaphor handling -- flag any literal translations of idioms/metaphors that sound unnatural; \
rewrite by the speaker's intent, not the surface imagery

If a translation is already good, keep it unchanged. Only revise where improvement is needed."""
    else:
        instruction += """

Review criteria:
1. Accuracy -- translations must faithfully convey the original meaning without omissions or additions
2. Terminology consistency -- same source term should have the same translation throughout
3. Referential coherence -- pronouns and references should be clear across segments
4. Cross-segment coherence -- read segments as a continuous narrative; \
check that references and context flow correctly from one segment to the next
5. Completeness -- no information from the original should be lost
6. Conciseness -- keep translations brief and suitable for caption display
7. Register fidelity -- maintain the same level of formality as the original

If a translation is already good, keep it unchanged. Only revise where improvement is needed."""

    instruction += """

Output schema:
{
  "items": [
    {
      "revised": "final translation text",
      "critique": "brief note about issue or why unchanged",
      "changed": true
    }
  ]
}
The number of items must exactly match the input length and preserve order.
"""

    if analysis and analysis.get("terminology"):
        instruction += "\nKey terminology for consistency check:\n"
        for term in analysis["terminology"]:
            source = term.get("source", "")
            context = term.get("context", "")
            translation = term.get("translation")
            if translation:
                instruction += f'- "{source}" -> "{translation}" ({context})\n'
            else:
                instruction += f'- "{source}": {context}\n'

    if analysis and analysis.get("metaphor_map"):
        instruction += "\nMetaphor checks:\n"
        for item in analysis["metaphor_map"]:
            source = item.get("source", "")
            intent = item.get("intent", "")
            instruction += f'- "{source}" should preserve intent: {intent}\n'

    if glossary:
        instruction += "\nRequired glossary translations:\n"
        for source, target in glossary.items():
            instruction += f'- "{source}" -> "{target}"\n'

    # Build paired input
    pairs = []
    for orig, trans in zip(original_texts, translated_texts):
        pairs.append({"original": orig, "translated": trans})

    return f"{instruction}\n\nTranslations to review:\n{json.dumps(pairs, ensure_ascii=False)}"
