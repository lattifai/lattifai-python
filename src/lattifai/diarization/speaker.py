"""LLM-based speaker name inference for diarization.

Uses the LattifAI LLM client abstraction to infer real speaker names
from transcript text, following the same strategy as YouTube transcript
speaker identification: metadata > content cues > fallback labels.
"""

import logging
from typing import Dict, List, Optional

from lattifai.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert at identifying speakers in transcripts.

Given text samples grouped by speaker label (SPEAKER_00, SPEAKER_01, ...),
infer the real name or role of each speaker.

## Identification Strategy

1. **Self-introductions**: "I'm Zhang San", "My name is...", "大家好我是..."
2. **How others address them**: "So Zhang San, what do you think?"
3. **Roles from context**: interviewer asks questions → Host; expert answers → Guest
4. **Content cues**: "In my lab we..." → Researcher; "Our company..." → Executive

## Rules

- Only return mappings you are confident about (>80% certainty)
- If you cannot determine a name, use a descriptive role: "Host", "Guest", "Interviewer"
- If you truly cannot identify a speaker, omit them from the result
- Keep names consistent with the language of the transcript
- Return valid JSON: {"SPEAKER_00": "Real Name", "SPEAKER_01": "Host"}
"""


class SpeakerNameInferrer:
    """Infer real speaker names from diarized transcript text via LLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: Optional[str] = None,
    ):
        self.llm = llm_client
        self.model = model

    def __call__(
        self,
        speaker_texts: Dict[str, List[str]],
        context: Optional[str] = None,
    ) -> Dict[str, str]:
        """Infer speaker names from text samples.

        Args:
            speaker_texts: Text samples grouped by speaker label.
            context: Optional user-provided context about the content.

        Returns:
            Mapping from speaker label to inferred real name.
        """
        prompt = self._build_prompt(speaker_texts, context)

        try:
            result = self.llm.generate_json_sync(
                prompt,
                system=SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"LLM speaker inference failed: {e}")
            return {}

        if not isinstance(result, dict):
            logger.warning(f"LLM returned non-dict: {type(result)}")
            return {}

        # Validate: only keep entries mapping existing speaker labels to non-empty strings
        validated = {}
        for k, v in result.items():
            if k in speaker_texts and isinstance(v, str) and v.strip():
                validated[k] = v.strip()

        return validated

    def _build_prompt(
        self,
        speaker_texts: Dict[str, List[str]],
        context: Optional[str],
    ) -> str:
        parts = []

        if context:
            parts.append(f"## Context\n{context}\n")

        parts.append("## Transcript Samples by Speaker\n")

        for speaker, texts in sorted(speaker_texts.items()):
            parts.append(f"### {speaker} ({len(texts)} samples)\n")
            for i, text in enumerate(texts, 1):
                # Truncate very long texts to save tokens
                display = text[:300] + "..." if len(text) > 300 else text
                parts.append(f"{i}. {display}")
            parts.append("")

        parts.append(
            "Based on the above, return a JSON object mapping speaker labels "
            "to their inferred real names. Only include speakers you can identify."
        )

        return "\n".join(parts)
