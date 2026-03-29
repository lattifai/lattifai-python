"""LLM-based speaker name inference for diarization.

Uses the LattifAI LLM client abstraction to infer real speaker names
from transcript text, following the same strategy as YouTube transcript
speaker identification: metadata > content cues > fallback labels.
"""

import logging
import re
from typing import Dict, List, Optional

from lattifai.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert at identifying speakers in podcast/interview transcripts.

Given text samples grouped by speaker label (SPEAKER_00, SPEAKER_01, ...),
infer the real name of each speaker.

## Identification Strategy (in priority order)

1. **Candidate names from metadata**: If a "Candidate Names" section is
   provided, you MUST assign those names to the matching SPEAKER labels.
   The host/interviewer asks shorter questions; the guest gives longer answers.
2. **Self-introductions**: "I'm Zhang San", "My name is...", "大家好我是..."
3. **How others address them**: "So Zhang San, what do you think?"
4. **Role signals**: The speaker who asks questions is typically the host;
   the one who gives longer, detailed answers is the guest.

## Rules

- You MUST use real names from metadata/candidates instead of "Host"/"Guest"
- Map EVERY speaker label to a candidate name when candidates are provided
- Only fall back to "Host"/"Guest" if NO candidate names are available
- Return valid JSON: {"SPEAKER_00": "Real Name", "SPEAKER_01": "Guest Name"}
"""


def extract_candidate_names(context: Optional[str]) -> Dict[str, List[str]]:
    """Extract candidate host/guest names from metadata context.

    Returns {"host": [...], "guest": [...]} with extracted person names.
    Focuses on high-precision extraction — avoids titles, roles, or URLs.
    """
    if not context:
        return {}

    candidates: Dict[str, List[str]] = {"host": [], "guest": []}

    def _add(key: str, name: str):
        name = name.strip().rstrip("。.，,")
        # Remove parenthetical role suffixes: "Name (Title)"
        name = re.sub(r"\s*[（(].+?[)）]$", "", name).strip()
        # Reject if too short, too long, or looks like a role/title
        if not name or len(name) < 2 or len(name) > 30:
            return
        if re.search(r"(创始|CEO|CTO|COO|总监|主席|顾问|教授|博士|Director|Founder|President)", name):
            return
        if name not in candidates[key]:
            candidates[key].append(name)

    # 1. Channel/Host name (often the host for interview podcasts)
    m = re.search(r"Channel/Host:\s*(.+?)(?:\n|$)", context)
    if m:
        channel = m.group(1).strip()
        if not re.search(r"(?:podcast|show|radio|channel|播客|节目|频道|Priors|Space|MLST)", channel, re.IGNORECASE):
            _add("host", channel)

    # 2. Title pattern: "Guest Name — topic" or "Guest Name: topic"
    m = re.search(r"Title:\s*(.+?)(?:\n|$)", context)
    if m:
        title = m.group(1).strip()
        # Strip leading episode numbers
        title = re.sub(r"^(?:#?\d+\.?\s*|E\d+[｜|]?\s*)", "", title)
        tm = re.match(r"^(.+?)\s*[—–|]\s*", title)
        if tm:
            name_part = tm.group(1).strip()
            if not re.match(r"^\d", name_part):
                _add("guest", name_part)

    # 3. Chinese structured blocks: 【主播】Name / 【嘉宾】Name
    for m in re.finditer(r"[【\[](?:主播|主持)[】\]][^\n]*\n(.+?)(?:\n[【\[]|\n\n|\Z)", context, re.DOTALL):
        # First line after header is the name (before comma = role)
        first_line = m.group(1).strip().split("\n")[0]
        name = first_line.split("，")[0].split(",")[0].strip()
        _add("host", name)

    for m in re.finditer(r"[【\[](?:嘉宾|guest)[】\]][^\n]*\n(.+?)(?:\n[【\[]|\n\n|\Z)", context, re.DOTALL | re.I):
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            name = line.split("，")[0].split(",")[0].strip()
            _add("guest", name)

    # 4. "X and Y are joined by Z" pattern
    m = re.search(
        r"(\b[A-Z][\w]+(?:\s+[A-Z][\w]+)+(?:\s+and\s+[A-Z][\w]+(?:\s+[A-Z][\w]+)+)?)"
        r"\s+(?:are\s+)?joined\s+by\s+"
        r"(\b[A-Z][\w]+(?:\s+[A-Z][\w]+)+)",
        context,
    )
    if m:
        for h in re.split(r"\s+and\s+", m.group(1)):
            _add("host", h.strip())
        _add("guest", m.group(2).strip())

    return {k: v for k, v in candidates.items() if v}


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

        # Extract and present candidate names prominently
        candidates = extract_candidate_names(context)
        if candidates:
            parts.append("## Candidate Names (from metadata)\n")
            if candidates.get("host"):
                parts.append(f"- Host(s): {', '.join(candidates['host'])}")
            if candidates.get("guest"):
                parts.append(f"- Guest(s): {', '.join(candidates['guest'])}")
            parts.append("")

        parts.append("## Transcript Samples by Speaker\n")

        for speaker, texts in sorted(speaker_texts.items()):
            # Compute avg length to help LLM distinguish host (short) vs guest (long)
            avg_len = sum(len(t) for t in texts) // max(len(texts), 1)
            parts.append(f"### {speaker} ({len(texts)} samples, avg {avg_len} chars)\n")
            for i, text in enumerate(texts, 1):
                display = text[:300] + "..." if len(text) > 300 else text
                parts.append(f"{i}. {display}")
            parts.append("")

        parts.append(
            "Based on the above, return a JSON object mapping ALL speaker labels "
            "to their real names from the candidate list. "
            "The host typically asks shorter questions; the guest gives longer answers."
        )

        return "\n".join(parts)
