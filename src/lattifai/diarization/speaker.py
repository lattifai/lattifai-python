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
2. **Conversation flow**: If a "Conversation Excerpt" is provided, analyze
   turn-taking patterns and how speakers address each other by name.
3. **Self-introductions**: "I'm Zhang San", "My name is...", "大家好我是..."
4. **How others address them**: "So Zhang San, what do you think?"
5. **Role signals**: The speaker who asks questions is typically the host;
   the one who gives longer, detailed answers is the guest.

## Rules

- You MUST use real names from metadata/candidates instead of "Host"/"Guest"
- Map EVERY speaker label to a candidate name when candidates are provided
- **Always use the person's full legal name**, not nicknames or aliases
  (e.g. use "Shawn Wang" not "Swyx", use "张伟" not "小张")
- For anonymous speakers (e.g. audience Q&A), use descriptive labels like
  "Audience" or "Questioner", not "Host"/"Guest"
- Only fall back to "Host"/"Guest" if NO name can be determined at all
- Return valid JSON: {"SPEAKER_00": "Real Name", "SPEAKER_01": "Guest Name"}
"""

# Pre-compiled pattern for rejecting pure role/title names
_TITLE_ONLY_RE = re.compile(
    r"(?:创始.+|管理.+|执行.+|联合.+|高级.+|首席.+|"
    r"CEO|CTO|COO|CFO|总监|主席|顾问|合伙人|总裁|总经理|"
    r"Director|Founder|President|Chairman|Advisor)",
    re.IGNORECASE,
)


# Words that never start a person name — articles, pronouns, conjunctions, common verbs
_NON_NAME_STARTERS = frozenset(
    "the a an we he she it they this that how what why when where who which "
    "is are was were will can do does did not no and or but if so".split()
)


def _looks_like_person_name(text: str) -> bool:
    """Heuristic: does *text* look like a Western or CJK person name?

    Accepts: "Jeff Dean", "Blaise Agüera y Arcas", "张三", "Shawn Wang (Swyx)"
    Rejects: "We are near the end of the exponential", "AI-Powered Future"
    """
    # CJK names: 2-4 characters
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", text):
        return True
    # Western names: 2-6 words, first word capitalized, no sentence starters
    words = text.split()
    if len(words) < 2 or len(words) > 6:
        return False
    if words[0].lower() in _NON_NAME_STARTERS:
        return False
    # First significant word must start with uppercase
    if not words[0][0].isupper():
        return False
    return True


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
        # Strip common title prefixes (CJK: no space needed; English: require space after)
        name = re.sub(r"^(?:教授|博士)", "", name).strip()
        name = re.sub(r"^(?:Dr\.?|Prof\.?)\s+", "", name, flags=re.IGNORECASE).strip()
        # Strip common title suffixes
        name = re.sub(r"(?:教授|博士|老师)$", "", name).strip()
        # Reject if too short, too long
        if not name or len(name) < 2 or len(name) > 30:
            return
        # Reject if what remains is purely a role/title (fullmatch, not search)
        if _TITLE_ONLY_RE.fullmatch(name):
            return
        if name not in candidates[key]:
            candidates[key].append(name)

    # 1. Channel/Host name (often the host for interview podcasts)
    m = re.search(r"Channel/Host:\s*(.+?)(?:\n|$)", context)
    if m:
        channel = m.group(1).strip()
        if not re.search(
            r"(?:podcast|show|radio|channel|talk|street|clips|shorts|highlights|播客|节目|频道|Priors|Space|MLST)",
            channel,
            re.IGNORECASE,
        ):
            _add("host", channel)
        else:
            # Strip common sub-channel suffixes to recover host name
            # e.g. "Dwarkesh Clips" → "Dwarkesh", "Lex Fridman Clips" → "Lex Fridman"
            stripped = re.sub(
                r"\s+(?:Clips|Shorts|Highlights|Podcast|Show|Channel|Radio|TV)$",
                "",
                channel,
                flags=re.IGNORECASE,
            ).strip()
            if stripped and stripped != channel:
                _add("host", stripped)

    # 2. Title pattern: "Guest Name — topic" or "topic — Guest Name"
    m = re.search(r"Title:\s*(.+?)(?:\n|$)", context)
    if m:
        title = m.group(1).strip()
        # Strip leading episode numbers
        title = re.sub(r"^(?:#?\d+\.?\s*|E\d+[｜|]?\s*)", "", title)
        # 2a. Name BEFORE separator: "Guest Name — topic"
        found_leading_name = False
        leading_match = re.match(r"^(.+?)\s*[—–|]\s*", title)
        if leading_match:
            leading_name = leading_match.group(1).strip()
            if not re.match(r"^\d", leading_name) and _looks_like_person_name(leading_name):
                _add("guest", leading_name)
                found_leading_name = True
        # 2b. Name AFTER last separator: "topic — Guest Name" or "topic - Guest Name"
        #     Only if 2a did not already find a valid name (avoid double extraction)
        #     Matches em-dash, en-dash, and spaced-hyphen (" - ") but NOT inline hyphens ("AI-powered")
        if not found_leading_name:
            trailing_name_match = re.search(r"(?:[—–]|\s-\s)\s*([^—–|]+)$", title)
            if trailing_name_match:
                trailing_text = trailing_name_match.group(1).strip()
                # Strip common podcast/episode suffixes
                trailing_text = re.sub(
                    r"\s*(?:Podcast|Episode|EP|节目|播客)\s*#?\d*$", "", trailing_text, flags=re.IGNORECASE
                ).strip()
                if trailing_text and _looks_like_person_name(trailing_text):
                    _add("guest", trailing_text)

    # 3. Chinese structured blocks: 【主播】Name / 【嘉宾】Name
    #    Handles both single-line (【主播】张三) and multi-line (【主播】\n张三) formats.
    for m in re.finditer(r"[【\[](?:主播|主持)[】\]]\s*(.+?)(?:\n[【\[]|\n\n|\Z)", context, re.DOTALL):
        first_line = m.group(1).strip().split("\n")[0]
        name = first_line.split("，")[0].split(",")[0].strip()
        _add("host", name)

    for m in re.finditer(r"[【\[](?:嘉宾|guest)[】\]]\s*(.+?)(?:\n[【\[]|\n\n|\Z)", context, re.DOTALL | re.I):
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            name = line.split("，")[0].split(",")[0].strip()
            _add("guest", name)

    # 4. "X and Y are joined by Z" — intentionally narrow (high-precision extractor).
    #    Only matches "ProperName joined by ProperName" with capitalized words.
    #    Broadening risks false positives on descriptive text.
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
    """Infer real speaker names from diarized transcript text via LLM.

    Supports two complementary input modes:
    - ``speaker_texts``: grouped text samples per speaker (legacy)
    - ``dialogue_turns``: ordered (speaker, text) pairs preserving conversation flow

    When ``dialogue_turns`` is provided, the prompt includes a conversation excerpt
    that reveals turn-taking patterns and cross-speaker references (e.g. "So Jeff,
    what do you think?"), which significantly improves identification accuracy.

    Multi-pass voting (``voting_rounds > 1``) runs inference multiple times and
    picks the majority answer per speaker to reduce temperature-induced randomness.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: Optional[str] = None,
        voting_rounds: int = 1,
    ):
        self.llm = llm_client
        self.model = model
        self.voting_rounds = max(1, voting_rounds)

    def __call__(
        self,
        speaker_texts: Dict[str, List[str]],
        context: Optional[str] = None,
        dialogue_turns: Optional[List[tuple]] = None,
    ) -> Dict[str, str]:
        """Infer speaker names from text samples and/or dialogue turns.

        Args:
            speaker_texts: Text samples grouped by speaker label.
            context: Optional user-provided context about the content.
            dialogue_turns: Optional ordered list of (speaker_label, text) pairs
                representing the actual conversation flow.

        Returns:
            Mapping from speaker label to inferred real name.
        """
        candidates = extract_candidate_names(context)
        prompt = self._build_prompt(speaker_texts, context, candidates, dialogue_turns)

        if self.voting_rounds <= 1:
            result = self._single_inference(prompt, speaker_texts)
        else:
            # Multi-pass voting: run N times, take majority per speaker
            from collections import Counter

            vote_counts: Dict[str, Counter] = {speaker: Counter() for speaker in speaker_texts}

            for _ in range(self.voting_rounds):
                round_result = self._single_inference(prompt, speaker_texts)
                for speaker_label, predicted_name in round_result.items():
                    vote_counts[speaker_label][predicted_name] += 1

            # Pick the most common prediction per speaker
            result = {}
            for speaker_label, counter in vote_counts.items():
                if counter:
                    winner, count = counter.most_common(1)[0]
                    result[speaker_label] = winner
                    if count < self.voting_rounds:
                        logger.debug(
                            "  %s: majority=%r (%d/%d votes)",
                            speaker_label,
                            winner,
                            count,
                            self.voting_rounds,
                        )

        # Post-LLM correction: relabel audience members in talk/presentation format
        # Use dialogue_turns for true segment counts (unaffected by sampling),
        # falling back to speaker_texts counts when dialogue_turns unavailable.
        true_segment_counts = self._compute_segment_counts(speaker_texts, dialogue_turns)
        result = self._correct_talk_format_labels(true_segment_counts, result)
        return result

    @staticmethod
    def _compute_segment_counts(
        speaker_texts: Dict[str, List[str]],
        dialogue_turns: Optional[List[tuple]] = None,
    ) -> Dict[str, int]:
        """Compute per-speaker segment counts from the best available source.

        ``dialogue_turns`` reflects the true distribution (before sampling),
        while ``speaker_texts`` may be truncated by the sampling strategy
        (first 5 + top 15), which flattens the dominant speaker's count.
        """
        if dialogue_turns:
            from collections import Counter

            return dict(Counter(speaker for speaker, _ in dialogue_turns))
        return {speaker: len(texts) for speaker, texts in speaker_texts.items()}

    @staticmethod
    def _correct_talk_format_labels(segment_counts: Dict[str, int], result: Dict[str, str]) -> Dict[str, str]:
        """Post-LLM correction: relabel audience in presentation/talk format.

        When one speaker dominates (>75% segments) and 3+ speakers exist, this
        is likely a talk/presentation. Non-dominant speakers with generic labels
        like "Host" or "Guest" are relabeled as "Audience" — they are audience
        members asking questions, not hosts or guests.

        Only relabels speakers with <10% of total segments to avoid misclassifying
        moderators who may legitimately have 10-20% of the conversation.
        """
        total_segments = sum(segment_counts.values())
        if total_segments == 0 or len(segment_counts) < 3:
            return result

        dominant_speaker = max(segment_counts, key=segment_counts.get)
        dominant_ratio = segment_counts[dominant_speaker] / total_segments
        if dominant_ratio < 0.75:
            return result

        # Talk format confirmed: correct generic labels for minor speakers
        generic_labels = {"host", "guest", "moderator", "interviewer", "co-host"}
        corrected = dict(result)
        for speaker, name in corrected.items():
            if speaker == dominant_speaker:
                continue
            speaker_ratio = segment_counts.get(speaker, 0) / total_segments
            if speaker_ratio < 0.10 and name.lower().strip() in generic_labels:
                corrected[speaker] = "Audience"
                logger.debug(
                    "Talk format correction: %s %r -> 'Audience' (%.1f%% of segments)",
                    speaker,
                    name,
                    speaker_ratio * 100,
                )
        return corrected

    def _single_inference(self, prompt: str, speaker_texts: Dict[str, List[str]]) -> Dict[str, str]:
        """Run a single LLM inference and validate the result."""
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

        validated = {}
        for speaker_label, name in result.items():
            if speaker_label in speaker_texts and isinstance(name, str) and name.strip():
                validated[speaker_label] = name.strip()
        return validated

    def _build_prompt(
        self,
        speaker_texts: Dict[str, List[str]],
        context: Optional[str],
        candidates: Optional[Dict[str, List[str]]] = None,
        dialogue_turns: Optional[List[tuple]] = None,
    ) -> str:
        parts = []

        if context:
            parts.append(f"## Context\n{context}\n")

        # Present candidate names prominently
        if candidates:
            parts.append("## Candidate Names (from metadata)\n")
            if candidates.get("host"):
                parts.append(f"- Host(s): {', '.join(candidates['host'])}")
            if candidates.get("guest"):
                parts.append(f"- Guest(s): {', '.join(candidates['guest'])}")
            parts.append("")

        # Dialogue turns: show conversation flow for turn-taking and cross-references
        if dialogue_turns:
            parts.append("## Conversation Excerpt (first 30 turns)\n")
            parts.append("Pay attention to how speakers address each other by name.\n")
            for speaker_label, text in dialogue_turns[:30]:
                display = text[:150] + "..." if len(text) > 150 else text
                parts.append(f"**{speaker_label}:** {display}")
            parts.append("")

        # Content format analysis: detect presentation/talk vs interview
        # Use dialogue_turns for true counts (pre-sampling), fall back to speaker_texts
        segment_counts = self._compute_segment_counts(speaker_texts, dialogue_turns)
        total_segments = sum(segment_counts.values())
        is_talk_format = False
        dominant_speaker = None
        if total_segments > 0 and len(speaker_texts) >= 3:
            dominant_speaker = max(segment_counts, key=segment_counts.get)
            dominant_ratio = segment_counts[dominant_speaker] / total_segments
            if dominant_ratio > 0.75:
                is_talk_format = True
                minor_speakers = [s for s in segment_counts if s != dominant_speaker]
                parts.append("## Content Format Analysis\n")
                parts.append("**FORMAT: PRESENTATION / TALK** (not a standard interview)\n")
                parts.append(
                    f"- {dominant_speaker} is the **presenter/speaker** "
                    f"({dominant_ratio:.0%} of all segments, "
                    f"{segment_counts[dominant_speaker]}/{total_segments})"
                )
                for minor in minor_speakers:
                    minor_count = segment_counts[minor]
                    minor_pct = minor_count / total_segments * 100
                    parts.append(
                        f"- {minor} has only {minor_count} segment(s) "
                        f"({minor_pct:.1f}%) → **audience member / questioner**"
                    )
                parts.append(
                    "\n**IMPORTANT**: There is NO host in this content. "
                    f"{dominant_speaker} is the presenter. "
                    "ALL other speakers are audience members asking questions. "
                    'Label them as "Audience" or "Questioner", '
                    'NEVER as "Host" or "Guest".\n'
                )

        parts.append("## Transcript Samples by Speaker\n")

        for speaker, texts in sorted(speaker_texts.items()):
            average_length = sum(len(t) for t in texts) // max(len(texts), 1)
            # Add role hint for talk format
            if is_talk_format and speaker == dominant_speaker:
                role_hint = " [PRESENTER]"
            elif is_talk_format:
                role_hint = " [AUDIENCE]"
            else:
                role_hint = ""
            parts.append(f"### {speaker} ({len(texts)} samples, avg {average_length} chars){role_hint}\n")
            for i, text in enumerate(texts, 1):
                display = text[:200] + "..." if len(text) > 200 else text
                parts.append(f"{i}. {display}")
            parts.append("")

        if is_talk_format:
            parts.append(
                "Based on the above, return a JSON object mapping ALL speaker labels "
                "to their real names. The presenter's name should come from metadata or "
                "self-introduction. ALL other speakers are audience members — "
                'label them as "Audience" or "Questioner". '
                'Do NOT use "Host" or "Guest".'
            )
        elif candidates:
            parts.append(
                "Based on the above, return a JSON object mapping ALL speaker labels "
                "to their real names from the candidate list. "
                "The host typically asks shorter questions; the guest gives longer answers."
            )
        else:
            parts.append(
                "Based on the above, return a JSON object mapping ALL speaker labels "
                "to their inferred real names. Use self-introductions, how others address them, "
                "or role signals (host asks short questions; guest gives longer answers). "
                'If no name can be determined, use "Host" or "Guest".'
            )

        return "\n".join(parts)


def infer_speaker_names(
    supervisions: list,
    context: Optional[str] = None,
    llm_client: Optional[BaseLLMClient] = None,
    model: Optional[str] = None,
    voting_rounds: int = 1,
) -> Dict[str, str]:
    """Infer speaker names from diarized caption supervisions.

    Standalone entry point that extracts speaker_texts and dialogue_turns
    from supervisions, then delegates to SpeakerNameInferrer.

    Args:
        supervisions: List of Supervision objects with speaker labels.
        context: Optional metadata context (title, channel, description).
        llm_client: LLM client instance. If None, auto-created from config.
        model: Override LLM model name.
        voting_rounds: Number of inference passes for majority voting.

    Returns:
        Mapping from speaker label (e.g. "SPEAKER_00") to inferred name.
    """
    from collections import defaultdict

    # Build speaker_texts and dialogue_turns from supervisions
    speaker_texts: Dict[str, List[str]] = defaultdict(list)
    dialogue_turns: List[tuple] = []

    for sup in supervisions:
        label = sup.speaker or "UNKNOWN"
        text = sup.text or ""
        if not text.strip():
            continue
        speaker_texts[label].append(text)
        dialogue_turns.append((label, text))

    speaker_texts = dict(speaker_texts)

    if not speaker_texts:
        return {}

    # Auto-create LLM client if not provided
    if llm_client is None:
        from lattifai.config.diarization import DiarizationLLMConfig

        llm_config = DiarizationLLMConfig()
        if model:
            llm_config.model_name = model
        llm_client = llm_config.create_client()

    inferrer = SpeakerNameInferrer(llm_client, model=model, voting_rounds=voting_rounds)
    return inferrer(speaker_texts, context=context, dialogue_turns=dialogue_turns)
