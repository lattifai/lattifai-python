"""Speaker identification for podcast transcripts.

Uses metadata + Gemini/heuristic analysis to map diarization tiers (SPEAKER_00, etc.)
to real speaker names.
"""

import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional

from .types import EpisodeMetadata, SpeakerIdentification

logger = logging.getLogger(__name__)

# Regex patterns for speaker labels in Gemini transcript output
_SPEAKER_LABEL_RE = re.compile(r"\*\*(.+?):\*\*")

# Patterns for heuristic intro detection
# NOTE: No re.IGNORECASE — we require capitalized proper names (English)
_INTRO_PATTERNS = [
    re.compile(r"(?:I'm|I am|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"(?:our guest|my guest|today's guest)[^,]*?\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"(?:joining us|joined by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    re.compile(r"speaking with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
    # Chinese patterns
    re.compile(r"(?:我是|大家好我是)\s*(.{2,8})"),
    re.compile(r"(?:嘉宾|请到了|邀请到)\s*(.{2,8})"),
]


class SpeakerIdentifier:
    """Identify speaker names in podcast transcripts.

    Combines metadata-based extraction, Gemini-based NER, and heuristic
    pattern matching to map diarization tier labels to real speaker names.

    Workflow:
    1. Extract candidate names from metadata (RSS author, title patterns)
    2. Optionally use Gemini to analyze intro text + metadata
    3. Map identified names to diarization tiers using transcript speaker labels
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        intro_words: int = 500,
        llm_client=None,
    ):
        """Initialize SpeakerIdentifier.

        Args:
            gemini_api_key: Gemini API key for AI-based identification.
            gemini_model: Gemini model to use for identification.
            intro_words: Number of words from transcript start for analysis.
            llm_client: Optional BaseLLMClient instance. If None, creates one from api_key.
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.intro_words = intro_words
        self._llm_client = llm_client

    def identify(
        self,
        transcript_text: str,
        episode: Optional[EpisodeMetadata] = None,
        method: str = "gemini",
        host_names: Optional[List[str]] = None,
        guest_names: Optional[List[str]] = None,
        intro_words: Optional[int] = None,
    ) -> List[SpeakerIdentification]:
        """Identify speakers in a podcast transcript.

        Args:
            transcript_text: Full transcript text (Gemini markdown format).
            episode: Episode metadata for context.
            method: Identification method ('gemini' or 'heuristic').
            host_names: Pre-known host names (skip detection).
            guest_names: Pre-known guest names (skip detection).
            intro_words: Override intro_words for this call (uses instance default if None).

        Returns:
            List of SpeakerIdentification results.
        """
        # Phase 1: Collect candidates from metadata
        candidates = self._extract_from_metadata(episode, host_names, guest_names)

        # If all speakers are pre-identified, skip detection
        if host_names and guest_names:
            logger.info("All speakers pre-identified from config, skipping detection")
            return candidates

        # Phase 2: Extract from transcript
        effective_intro_words = intro_words if intro_words is not None else self.intro_words
        intro_text = self._get_intro_text(transcript_text, effective_intro_words)

        if method == "gemini" and self.gemini_api_key:
            gemini_results = self._extract_via_gemini(intro_text, episode, candidates)
            candidates = self._merge_candidates(candidates, gemini_results)
        else:
            heuristic_results = self._extract_via_heuristic(intro_text)
            candidates = self._merge_candidates(candidates, heuristic_results)

        return candidates

    def map_to_diarization_tiers(
        self,
        speakers: List[SpeakerIdentification],
        transcript_text: str,
        diarization_tiers: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Map identified speaker names to diarization tier labels.

        Strategy:
        1. Extract **Speaker Name:** labels from Gemini transcript
        2. For each named speaker, find which tier they appear in most often
        3. Unmatched tiers keep their original label

        Args:
            speakers: Identified speakers.
            transcript_text: Transcript text with **Speaker:** labels.
            diarization_tiers: List of tier names (e.g., ['SPEAKER_00', 'SPEAKER_01']).

        Returns:
            Mapping of {tier_label: speaker_name} (e.g., {'SPEAKER_00': 'Alice'}).
        """
        if not speakers or not diarization_tiers:
            return {}

        # Extract speaker labels from transcript
        transcript_speakers = _SPEAKER_LABEL_RE.findall(transcript_text)
        if not transcript_speakers:
            # No speaker labels in transcript - map by order (host = SPEAKER_00)
            return self._map_by_order(speakers, diarization_tiers)

        # Count occurrences of each speaker name
        name_counts = Counter(transcript_speakers)
        speaker_names = [s.name for s in speakers]

        # Simple ordering: most frequent speaker = host = first tier
        # This works because hosts typically speak more in intros
        sorted_names = sorted(name_counts.keys(), key=lambda n: name_counts[n], reverse=True)

        mapping = {}
        used_tiers = set()
        for name in sorted_names:
            if name in speaker_names:
                # Find first unused tier
                for tier in diarization_tiers:
                    if tier not in used_tiers:
                        mapping[tier] = name
                        used_tiers.add(tier)
                        break

        return mapping

    def _extract_from_metadata(
        self,
        episode: Optional[EpisodeMetadata],
        host_names: Optional[List[str]] = None,
        guest_names: Optional[List[str]] = None,
    ) -> List[SpeakerIdentification]:
        """Extract speaker candidates from episode metadata."""
        results = []

        # Add pre-configured names
        for name in host_names or []:
            results.append(SpeakerIdentification(name=name, role="host", confidence=1.0, source="config"))
        for name in guest_names or []:
            results.append(SpeakerIdentification(name=name, role="guest", confidence=1.0, source="config"))

        if not episode:
            return results

        existing_names = {s.name for s in results}

        # Add from episode metadata
        for name in episode.host_names:
            if name and name not in existing_names:
                results.append(SpeakerIdentification(name=name, role="host", confidence=0.8, source="metadata"))
                existing_names.add(name)

        for name in episode.guest_names:
            if name and name not in existing_names:
                results.append(SpeakerIdentification(name=name, role="guest", confidence=0.7, source="metadata"))
                existing_names.add(name)

        return results

    def _get_llm_client(self):
        """Get or create the LLM client for speaker identification."""
        if self._llm_client is None:
            from lattifai.llm import create_client

            self._llm_client = create_client("gemini", api_key=self.gemini_api_key, model=self.gemini_model)
        return self._llm_client

    def _extract_via_gemini(
        self,
        intro_text: str,
        episode: Optional[EpisodeMetadata],
        existing_candidates: List[SpeakerIdentification],
    ) -> List[SpeakerIdentification]:
        """Use Gemini to identify speakers from intro text + metadata."""
        if not self.gemini_api_key and self._llm_client is None:
            logger.warning("No Gemini API key, falling back to heuristic")
            return self._extract_via_heuristic(intro_text)

        # Build context for Gemini
        context_parts = []
        if episode:
            if episode.show_notes:
                context_parts.append(f"Show Notes:\n{episode.show_notes}")
            if episode.podcast and episode.podcast.author:
                context_parts.append(f"Podcast Author: {episode.podcast.author}")
        if existing_candidates:
            context_parts.append("Known speakers: " + ", ".join(f"{s.name} ({s.role})" for s in existing_candidates))

        prompt = f"""Analyze this podcast transcript intro and metadata to identify all speakers.

Context:
{chr(10).join(context_parts)}

Transcript (first {self.intro_words} words):
{intro_text}

Return a JSON array of speakers found. Each speaker object must have:
- "name": Full name of the speaker
- "role": "host" or "guest"
- "confidence": 0.0 to 1.0
- "source": "gemini"

Return ONLY the JSON array, no other text. Example:
[{{"name": "Alice Johnson", "role": "host", "confidence": 0.9, "source": "gemini"}}]
"""

        try:
            client = self._get_llm_client()
            speakers_data = client.generate_json_sync(prompt, temperature=0.1)

            results = []
            if not isinstance(speakers_data, list):
                speakers_data = [speakers_data]
            for sp in speakers_data:
                results.append(
                    SpeakerIdentification(
                        name=sp.get("name", "Unknown"),
                        role=sp.get("role", "unknown"),
                        confidence=float(sp.get("confidence", 0.5)),
                        source="gemini",
                    )
                )

            logger.info(f"Gemini identified {len(results)} speakers: {[s.name for s in results]}")
            return results

        except Exception as e:
            logger.warning(f"Gemini speaker identification failed: {e}")
            return self._extract_via_heuristic(intro_text)

    def _extract_via_heuristic(self, intro_text: str) -> List[SpeakerIdentification]:
        """Extract speaker names using regex patterns on intro text."""
        results = []
        seen_names = set()

        # Extract from **Speaker Name:** patterns in transcript
        speaker_labels = _SPEAKER_LABEL_RE.findall(intro_text)
        for name in speaker_labels:
            name = name.strip()
            if name and name not in seen_names and not name.startswith("Speaker"):
                # First unique speaker is likely host
                role = "host" if not results else "guest"
                results.append(SpeakerIdentification(name=name, role=role, confidence=0.6, source="heuristic"))
                seen_names.add(name)

        # Extract from intro phrases
        for pattern in _INTRO_PATTERNS:
            for match in pattern.finditer(intro_text):
                name = match.group(1).strip()
                if name and name not in seen_names and len(name) > 1:
                    results.append(SpeakerIdentification(name=name, role="unknown", confidence=0.4, source="heuristic"))
                    seen_names.add(name)

        return results

    def _get_intro_text(self, transcript_text: str, intro_words: Optional[int] = None) -> str:
        """Extract first N words from transcript for analysis."""
        n = intro_words if intro_words is not None else self.intro_words
        words = transcript_text.split()
        return " ".join(words[:n])

    def _merge_candidates(
        self,
        existing: List[SpeakerIdentification],
        new: List[SpeakerIdentification],
    ) -> List[SpeakerIdentification]:
        """Merge new candidates into existing list, preferring higher confidence."""
        existing_names = {s.name: s for s in existing}

        for sp in new:
            if sp.name in existing_names:
                # Keep higher confidence
                if sp.confidence > existing_names[sp.name].confidence:
                    existing_names[sp.name] = sp
            else:
                existing_names[sp.name] = sp

        return list(existing_names.values())

    def _map_by_order(
        self,
        speakers: List[SpeakerIdentification],
        tiers: List[str],
    ) -> Dict[str, str]:
        """Map speakers to tiers by order: hosts first, then guests."""
        hosts = [s for s in speakers if s.role == "host"]
        guests = [s for s in speakers if s.role == "guest"]
        others = [s for s in speakers if s.role not in ("host", "guest")]
        ordered = hosts + guests + others

        mapping = {}
        for i, speaker in enumerate(ordered):
            if i < len(tiers):
                mapping[tiers[i]] = speaker.name

        return mapping
