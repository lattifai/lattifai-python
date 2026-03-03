"""Podcast workflow configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class PodcastConfig:
    """Configuration for podcast transcription and speaker identification.

    Attributes:
        enabled: Enable podcast processing features.
        identify_speakers: Enable speaker name identification from metadata + Gemini.
        host_names: Known host names (skips detection for these).
        guest_names: Known guest names (skips detection for these).
        num_speakers: Expected number of speakers (passed to diarization).
        show_notes: Show notes text for context injection.
        rss_feed_url: RSS feed URL (overrides auto-detection from URL).
        intro_words: Number of words from transcript start used for speaker identification.
        identification_method: Method for speaker identification ('gemini' or 'heuristic').
    """

    enabled: bool = True
    """Enable podcast processing features."""

    identify_speakers: bool = True
    """Enable speaker name identification from metadata and/or Gemini."""

    host_names: List[str] = field(default_factory=list)
    """Known host names. When provided, skips host detection."""

    guest_names: List[str] = field(default_factory=list)
    """Known guest names. When provided, skips guest detection."""

    num_speakers: Optional[int] = None
    """Expected number of speakers. Passed to diarization config if set."""

    show_notes: Optional[str] = None
    """Show notes text for speaker identification context."""

    rss_feed_url: Optional[str] = None
    """RSS feed URL. Overrides auto-detection from platform URL."""

    intro_words: int = 500
    """Number of words from transcript start used for Gemini speaker identification."""

    identification_method: Literal["gemini", "heuristic"] = "gemini"
    """Method for speaker identification. 'gemini' uses Gemini API, 'heuristic' uses regex patterns."""

    def __post_init__(self):
        """Validate configuration."""
        if self.intro_words < 50:
            raise ValueError("intro_words must be at least 50")
        if self.identification_method not in ("gemini", "heuristic"):
            raise ValueError(
                f"identification_method must be 'gemini' or 'heuristic', got '{self.identification_method}'"
            )

    @property
    def has_known_speakers(self) -> bool:
        """Return True if any speaker names are pre-configured."""
        return bool(self.host_names or self.guest_names)
