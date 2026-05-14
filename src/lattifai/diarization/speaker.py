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
2. **Speaker background match**: If a "Speaker Background" section lists
   affiliations / aliases / bios, use them as strong identity anchors:
   - **Affiliation**: a candidate's affiliation (e.g. "Stanford AI Lab") that
     appears in a speaker's self-introduction ("I work at Stanford", "我在
     斯坦福") or in their topical references ("in our lab at Stanford") is a
     strong assignment signal.
   - **Aliases**: when a speaker is addressed by an alias (e.g. "Swyx"), map
     that to the candidate's full legal name, not the alias.
   - **Bio**: episode-specific expertise from the bio aligns with content a
     speaker brings up (e.g. an RLHF bio → speaker who discusses RLHF).
3. **Conversation flow**: If a "Conversation Excerpt" is provided, analyze
   turn-taking patterns and how speakers address each other by name.
4. **Self-introductions**: "I'm Zhang San", "My name is...", "大家好我是..."
5. **How others address them**: "So Zhang San, what do you think?"
6. **Role signals**: The speaker who asks questions is typically the host;
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


# Role-keyword bank — drives both "which role bucket does this candidate map to"
# and "what does this role typically say in a transcript". When a guest's role
# string contains any of the bucket keys, the corresponding signal phrases are
# used to mine role-revealing samples from that guest's speech, surfaced to the
# LLM as deterministic evidence. This sidesteps long-context dilution and
# ordering bias.
_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "engineering": [
        "engineering",
        "engineer",
        "infra",
        "infrastructure",
        "code",
        "codebase",
        "system",
        "architecture",
        "stack",
        "deploy",
        "build",
        "review the pr",
        "reviewing the pr",
        "pr review",
        "approve the pr",
        "merge",
        "scaling",
        "latency",
        "ops",
        "uptime",
        "reliability",
    ],
    "product": [
        "product",
        "users",
        "user-facing",
        "customer",
        "customers",
        "roadmap",
        "prioritize",
        "feature",
        "release",
        "ship",
        "go-to-market",
        "gtm",
        "pricing",
        "positioning",
        "segment",
        "use case",
        "user research",
    ],
    "research": [
        "research",
        "experiment",
        "paper",
        "model training",
        "training",
        "rlhf",
        "eval",
        "scaling laws",
        "ablation",
        "benchmark",
    ],
    "design": [
        "design",
        "designer",
        "ux",
        "ui",
        "user experience",
        "wireframe",
        "mock",
        "prototype",
    ],
    "sales": [
        "sales",
        "deal",
        "pipeline",
        "enterprise",
        "account",
        "quota",
        "rep",
        "outbound",
    ],
    "marketing": [
        "marketing",
        "campaign",
        "brand",
        "comms",
        "pr team",
        "press",
        "social",
        "growth marketing",
    ],
    "ceo": [
        "fundrais",
        "board",
        "investor",
        "strategy",
        "vision",
        "hiring",
        "company-wide",
        "leadership team",
    ],
    "legal": [
        "legal",
        "compliance",
        "contract",
        "policy",
        "regulation",
        "regulatory",
    ],
}


def _role_bucket(role: str) -> Optional[str]:
    """Map a free-form role string to a keyword bucket via heuristic match.

    Examples:
      "head of engineering for the Claude platform" → "engineering"
      "head of product" → "product"
      "VP of design" → "design"
    Returns None if no bucket matches confidently.
    """
    if not role:
        return None
    lower = role.lower()
    # Prefer the bucket whose own key appears literally in the role string;
    # this beats keyword-set overlap when the role itself names the function.
    for bucket in _ROLE_KEYWORDS:
        if bucket in lower:
            return bucket
    # Soft fallback: longest-keyword-overlap. Helps "VP, infrastructure" map
    # to "engineering" via the "infrastructure" keyword.
    best = None
    best_len = 0
    for bucket, kws in _ROLE_KEYWORDS.items():
        for kw in kws:
            if len(kw) > best_len and kw in lower:
                best = bucket
                best_len = len(kw)
    return best


def _mine_role_evidence(
    speaker_texts: Dict[str, List[str]],
    candidate_roles: Dict[str, str],
) -> Dict[str, List[tuple]]:
    """Mine role-revealing utterances from each speaker.

    Returns {speaker_label: [(role_bucket, snippet), ...]} ranked by signal
    strength. Empty when no candidate has a recognizable role bucket — keeps
    the prompt clean for non-role-aware episodes.
    """
    # Resolve which role buckets are in play for this episode.
    active_buckets = {bucket for role in candidate_roles.values() if (bucket := _role_bucket(role)) is not None}
    if not active_buckets:
        return {}

    evidence: Dict[str, List[tuple]] = {}
    for speaker, texts in speaker_texts.items():
        hits: List[tuple] = []
        for text in texts:
            lower = text.lower()
            for bucket in active_buckets:
                if any(kw in lower for kw in _ROLE_KEYWORDS[bucket]):
                    hits.append((bucket, text.strip()))
                    break  # one bucket per snippet to avoid double-counting
        if hits:
            # Dedup by (bucket, first 80 chars of snippet) and cap per speaker.
            seen = set()
            ranked: List[tuple] = []
            for b, s in hits:
                key = (b, s[:80])
                if key in seen:
                    continue
                seen.add(key)
                ranked.append((b, s))
                if len(ranked) >= 8:
                    break
            evidence[speaker] = ranked
    return evidence


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


_STRUCTURED_ROLE_KEYWORDS = ("host", "guest", "moderator", "interviewer", "co-host")


def _parse_speakers_block(context: str) -> List[Dict[str, object]]:
    """Parse the structured `Speakers:` block emitted by `_resolve_context`.

    Expected format::

        Speakers:
        - Alice Chen (host) @ Anthropic (research engineer)
          aliases: Alice
          bio: Host of the show. Background in distributed systems.
        - Bob Smith (guest) @ Stanford AI Lab

    Returns a list of dicts with keys ``name`` / ``role`` / ``affiliation`` /
    ``aliases`` / ``bio``. Only the parenthetical *role* keywords from
    ``_STRUCTURED_ROLE_KEYWORDS`` are treated as roles — any other
    parenthetical on the name line (e.g. ``Shawn Wang (Swyx)``) is captured
    as an implicit alias so the LLM can fold short forms back to the legal
    name without confusing them with a role.

    Block-terminator rule: blank lines are tolerated *between* entries (a
    common readability pattern). The block only ends when we encounter a
    non-blank, non-indented line that does not start with ``- ``.
    """
    m = re.search(r"^Speakers:\s*$", context, re.MULTILINE)
    if not m:
        return []

    rest = context[m.end() :]
    entries: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    role_re = re.compile(rf"\s*\(({'|'.join(_STRUCTURED_ROLE_KEYWORDS)})\)\s*$", re.IGNORECASE)
    paren_re = re.compile(r"\s*\(([^)]+)\)\s*$")

    def _commit() -> None:
        nonlocal current
        if current and current.get("name"):
            entries.append(current)
        current = None

    for raw in rest.split("\n"):
        if not raw.strip():
            # Blank lines are readability separators inside the block; commit
            # the current entry but do NOT terminate — the next non-indented,
            # non-bullet line is what ends the block.
            _commit()
            continue

        if raw.startswith("- "):
            _commit()
            content = raw[2:].strip()
            affiliation = ""
            if " @ " in content:
                left, aff = content.split(" @ ", 1)
                affiliation = aff.strip()
                content = left.strip()
            role = ""
            rm = role_re.search(content)
            if rm:
                role = rm.group(1).strip().lower()
                content = content[: rm.start()].strip()
            # Any remaining trailing parenthetical on the name is an implicit
            # alias (e.g. "Shawn Wang (Swyx)"), not a role.
            implicit_aliases: List[str] = []
            pm = paren_re.search(content)
            if pm:
                aliases_text = pm.group(1).strip()
                content = content[: pm.start()].strip()
                # Reject URLs / handles
                if aliases_text and not aliases_text.startswith(("@", "http")) and "/" not in aliases_text:
                    implicit_aliases = [a.strip() for a in re.split(r"[，,]", aliases_text) if a.strip()]
            current = {
                "name": content,
                "role": role,
                "affiliation": affiliation,
                "aliases": list(implicit_aliases),
                "bio": "",
            }
            continue

        if raw.startswith("  ") and current is not None:
            sub = raw.strip()
            if sub.lower().startswith("aliases:"):
                aliases_str = sub.split(":", 1)[1].strip()
                explicit = [a.strip() for a in aliases_str.split(",") if a.strip()]
                # Merge explicit aliases with any implicit alias captured from
                # the bullet line, preserving order and deduping.
                seen = set()
                merged: List[str] = []
                for a in list(current.get("aliases") or []) + explicit:
                    if a and a not in seen:
                        seen.add(a)
                        merged.append(a)
                current["aliases"] = merged
            elif sub.lower().startswith("bio:"):
                current["bio"] = sub.split(":", 1)[1].strip()
            elif current.get("bio"):
                current["bio"] = f"{current['bio']} {sub}".strip()
            continue

        # Non-indented, non-bullet, non-blank line — block ended.
        _commit()
        break

    _commit()
    return entries


def extract_candidate_names(context: Optional[str]) -> Dict[str, object]:
    """Extract candidate host/guest names from metadata context.

    Returns ``{"host": [name, ...], "guest": [name, ...], "roles": {name: role},
    "affiliations": {name: aff}, "aliases": {name: [alias, ...]}, "bios": {name: bio}}``
    with extracted person names. Auxiliary keys are only present when the
    underlying signal was captured — the ``roles`` / ``affiliations`` /
    ``aliases`` / ``bios`` maps are the crucial signals that let the LLM
    disambiguate multiple guests sharing the same conversation.

    Focuses on high-precision extraction — avoids titles, roles, or URLs.
    """
    if not context:
        return {}

    candidates: Dict[str, object] = {
        "host": [],
        "guest": [],
        "roles": {},
        "affiliations": {},
        "aliases": {},
        "bios": {},
    }

    def _clean_role(role: str) -> str:
        role = role.strip().rstrip(".,;:").strip()
        # Drop trailing connective fragments
        role = re.sub(r"\s+(?:and|or)$", "", role, flags=re.IGNORECASE).strip()
        # Cap at a reasonable length so we don't paste a whole sentence.
        return role[:80]

    def _add(
        key: str,
        name: str,
        role: Optional[str] = None,
        affiliation: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        bio: Optional[str] = None,
    ):
        name = name.strip().rstrip("。.，,")
        # Remove parenthetical role suffixes: "Name (Title)"
        # Preserve the parenthetical content as a role hint when no explicit
        # role was passed in (e.g. "Angela Jiang (head of product)").
        paren_match = re.search(r"\s*[（(](.+?)[)）]$", name)
        paren_role: Optional[str] = None
        if paren_match:
            paren_text = paren_match.group(1).strip()
            # Reject pure handle / URL captures
            if not paren_text.startswith("@") and "/" not in paren_text and not paren_text.startswith("http"):
                paren_role = paren_text
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
        if name not in candidates[key]:  # type: ignore[operator]
            candidates[key].append(name)  # type: ignore[union-attr]
        chosen_role = role if role else paren_role
        if chosen_role:
            cleaned = _clean_role(chosen_role)
            if cleaned and name not in candidates["roles"]:  # type: ignore[operator]
                candidates["roles"][name] = cleaned  # type: ignore[index]
        if affiliation:
            aff_clean = affiliation.strip()[:120]
            if aff_clean and name not in candidates["affiliations"]:  # type: ignore[operator]
                candidates["affiliations"][name] = aff_clean  # type: ignore[index]
        if aliases:
            cleaned_aliases = [a.strip() for a in aliases if a and a.strip() and a.strip() != name]
            if cleaned_aliases and name not in candidates["aliases"]:  # type: ignore[operator]
                candidates["aliases"][name] = cleaned_aliases  # type: ignore[index]
        if bio:
            bio_clean = bio.strip()[:400]
            if bio_clean and name not in candidates["bios"]:  # type: ignore[operator]
                candidates["bios"][name] = bio_clean  # type: ignore[index]

    # 0. Structured Speakers block (emitted by `_resolve_context`). High-precision
    #    direct read — fills affiliation/aliases/bio that regex paths can't capture.
    for entry in _parse_speakers_block(context):
        role = str(entry.get("role") or "").lower()
        key = "host" if role in ("host", "moderator", "co-host") else "guest"
        _add(
            key,
            str(entry.get("name") or ""),
            role=role or None,
            affiliation=str(entry.get("affiliation") or "") or None,
            aliases=entry.get("aliases") or None,  # type: ignore[arg-type]
            bio=str(entry.get("bio") or "") or None,
        )

    # 1. Channel/Host name (often the host for interview podcasts)
    m = re.search(r"Channel/Host:\s*(.+?)(?:\n|$)", context)
    if m:
        channel = m.group(1).strip()
        if not re.search(
            r"(?:podcast|show|radio|channel|talk|street|clips|shorts|highlights|播客|节目|频道|Priors|Space|MLST)",
            channel,
            re.IGNORECASE,
        ):
            # Validate looks like a person name to reject descriptive channel
            # names that don't trigger the show-keyword blacklist (e.g.
            # "The Diary Of A CEO", "Apolas").
            if _looks_like_person_name(channel):
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
            if stripped and stripped != channel and _looks_like_person_name(stripped):
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
                    r"\s*(?:Podcast|Episode|EP|节目|播客)\s*#?\d*$",
                    "",
                    trailing_text,
                    flags=re.IGNORECASE,
                ).strip()
                if trailing_text and _looks_like_person_name(trailing_text):
                    _add("guest", trailing_text)

    # 3. Chinese structured blocks: 【主播】Name / 【嘉宾】Name
    #    Handles both single-line (【主播】张三) and multi-line (【主播】\n张三) formats.
    for m in re.finditer(r"[【\[](?:主播|主持)[】\]]\s*(.+?)(?:\n[【\[]|\n\n|\Z)", context, re.DOTALL):
        first_line = m.group(1).strip().split("\n")[0]
        name = first_line.split("，")[0].split(",")[0].strip()
        _add("host", name)

    for m in re.finditer(
        r"[【\[](?:嘉宾|guest)[】\]]\s*(.+?)(?:\n[【\[]|\n\n|\Z)",
        context,
        re.DOTALL | re.I,
    ):
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

    # 5. Free-form host/guest intro with optional role: extracts the speakers
    #    introduced by a host in narrative prose. This covers podcast
    #    descriptions like:
    #      "I talk with Angela Jiang (@angjiang), head of product for the
    #       Claude platform, and Katelyn Lesse (@katelyn_lesse), head of
    #       engineering for the Claude platform, about ..."
    #    Captures (name, role) so the LLM can disambiguate multiple guests
    #    sharing the same conversation. High precision via require:
    #      - intro verb (talk/chat/interview/sit down with, joined by ...)
    #      - capitalized two-word name immediately after
    #      - optional parenthesized handle/role
    #      - optional role clause delimited by comma / "—" / colon
    _INTRO_VERB = (
        r"(?:talk(?:s|ing|ed)?\s+with|interview(?:s|ing|ed)?\s+(?:with\s+)?|"
        r"joined\s+by|chat(?:s|ting|ted)?\s+with|sit(?:s|ting|s\s+down)?\s+(?:down\s+)?with|"
        r"sat\s+down\s+with|spoke?\s+with|speak(?:s|ing)?\s+with)"
    )
    _NAME_PAREN_ROLE = (
        r"(?P<name>[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,3})"
        r"(?:\s*\([^)]{1,60}\))?"  # optional handle/short paren
        r"(?:\s*[,，—–:-]\s*(?P<role>[^,，.\n()]{4,80}?))?"  # optional role
        r"(?=\s*(?:,|，|\.|\n|$|\s+and\s+|\s+about\s+|\s+on\s+))"
    )
    for intro in re.finditer(rf"\b{_INTRO_VERB}\s+", context, re.IGNORECASE):
        tail = context[intro.end() : intro.end() + 400]
        # 拆 "and" 把多嘉宾分开
        for chunk in re.split(r"\s+and\s+(?=[A-Z])", tail, flags=re.IGNORECASE):
            chunk = chunk.lstrip(", \n").rstrip()
            mm = re.match(_NAME_PAREN_ROLE, chunk)
            if not mm:
                continue
            nm = mm.group("name").strip()
            role = (mm.group("role") or "").strip()
            if not _looks_like_person_name(nm):
                continue
            _add("guest", nm, role=role or None)

    # 6. Standalone "Name (role)" — fallback when no intro verb matched.
    #    Restricted to lines NOT containing URLs / handles to avoid catching
    #    "@danshipper (host of Every)" style noise. Picks up "head of X" /
    #    "CTO / founder / VP / ..." role hints.
    if not candidates["guest"] and not candidates["host"]:  # type: ignore[operator]
        for mm in re.finditer(
            r"\b(?P<name>[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,3})"
            r"\s*[,，]?\s*"
            r"(?P<role>(?:head|chief|director|vp|cto|ceo|founder|co-?founder|"
            r"president|engineer|engineering manager|product manager|"
            r"senior|staff|principal|lead|host|guest)\b[^.,()\n]{0,60})",
            context,
            re.IGNORECASE,
        ):
            nm = mm.group("name").strip()
            role = mm.group("role").strip()
            if _looks_like_person_name(nm):
                _add("guest", nm, role=role)

    # Drop empty buckets (legacy contract — callers check `if candidates:`)
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

        # Present candidate names prominently. When `roles` is available,
        # render each name with its descriptive role on its own line — this is
        # the strongest signal for disambiguating multiple guests in the same
        # conversation (e.g. head of product vs head of engineering).
        if candidates:
            parts.append("## Candidate Names (from metadata)\n")
            roles = candidates.get("roles") or {}

            def _format_entry(name: str) -> str:
                role = roles.get(name)
                return f"{name} — {role}" if role else name

            if candidates.get("host"):
                if any(roles.get(n) for n in candidates["host"]):
                    parts.append("- Host(s):")
                    for n in candidates["host"]:
                        parts.append(f"  - {_format_entry(n)}")
                else:
                    parts.append(f"- Host(s): {', '.join(candidates['host'])}")
            if candidates.get("guest"):
                if any(roles.get(n) for n in candidates["guest"]):
                    parts.append("- Guest(s):")
                    for n in candidates["guest"]:
                        parts.append(f"  - {_format_entry(n)}")
                else:
                    parts.append(f"- Guest(s): {', '.join(candidates['guest'])}")
            parts.append("")

            # Speaker Background: render affiliation / aliases / bio per candidate.
            # Strong identity anchors for matching self-introductions, cross-
            # addressing by alias, and topical expertise — only emit when at
            # least one signal is non-empty to keep the prompt lean.
            affiliations = candidates.get("affiliations") or {}
            aliases_map = candidates.get("aliases") or {}
            bios = candidates.get("bios") or {}
            if affiliations or aliases_map or bios:
                all_names = list(candidates.get("host") or []) + list(candidates.get("guest") or [])
                bg_lines: List[str] = []
                for nm in all_names:
                    nm_role = roles.get(nm)
                    nm_aff = affiliations.get(nm)
                    nm_aliases = aliases_map.get(nm) or []
                    nm_bio = bios.get(nm)
                    if not (nm_aff or nm_aliases or nm_bio):
                        continue
                    bg_lines.append(f"### {nm}")
                    if nm_role:
                        bg_lines.append(f"- Role: {nm_role}")
                    if nm_aff:
                        bg_lines.append(f"- Affiliation: {nm_aff}")
                    if nm_aliases:
                        bg_lines.append(f"- Aliases: {', '.join(nm_aliases)}")
                    if nm_bio:
                        bg_lines.append(f"- Bio: {nm_bio}")
                    bg_lines.append("")
                if bg_lines:
                    parts.append("## Speaker Background\n")
                    parts.append(
                        "Strong identity anchors. Match a speaker's self-introduction "
                        "or topical references against the affiliation/bio of each "
                        "candidate; aliases tell you which short form refers to which "
                        "full legal name.\n"
                    )
                    parts.extend(bg_lines)

            # Role evidence: mine transcript for role-revealing utterances and
            # surface them per-speaker. This is the deterministic disambiguator
            # for multi-guest episodes — without it, the LLM tends to assign
            # guests in the order they appear in the candidate list.
            evidence = _mine_role_evidence(speaker_texts, roles)
            if evidence:
                parts.append("## Role Evidence by Speaker\n")
                parts.append(
                    "These transcript samples contain role-specific signals. "
                    "Use them to match each speaker to the candidate whose "
                    "role (above) aligns with their evidence:\n"
                )
                for speaker in sorted(evidence.keys()):
                    parts.append(f"### {speaker}")
                    for bucket, snippet in evidence[speaker]:
                        display = snippet[:240] + "…" if len(snippet) > 240 else snippet
                        parts.append(f"- [{bucket}] {display}")
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
            instr = [
                "Based on the above, return a JSON object mapping ALL speaker labels "
                "to their real names from the candidate list. "
                "The host typically asks shorter questions; the guest gives longer answers."
            ]
            # Multi-guest disambiguation: when 2+ guest candidates share the
            # same conversation, role hints are the only reliable signal.
            guest_count = len(candidates.get("guest") or [])
            has_roles = bool(candidates.get("roles"))
            if guest_count >= 2 and has_roles:
                instr.append(
                    "\nIMPORTANT — multiple guests share this conversation. Each "
                    "guest's role is listed above. Match each guest to the speaker "
                    "label whose transcript contains role-specific signals: a 'head "
                    "of engineering' will talk about systems, code, infrastructure, "
                    "PR reviews; a 'head of product' will talk about users, "
                    "roadmap, prioritization, customers. Do NOT assign guests based "
                    "on speaking order or the order they appear in the candidate "
                    "list — only on substantive role evidence from the transcript."
                )
            parts.append("\n".join(instr))
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
