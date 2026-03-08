"""Podcast data types for LattifAI podcast module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PodcastPlatform(Enum):
    """Supported podcast platforms."""

    APPLE = "apple"
    SPOTIFY = "spotify"
    XIAOYUZHOU = "xiaoyuzhou"
    YOUTUBE = "youtube"
    RSS = "rss"
    UNKNOWN = "unknown"


@dataclass
class PodcastMetadata:
    """Podcast show-level metadata from RSS or platform API.

    Attributes:
        title: Podcast show title.
        author: Show author / host name (from itunes:author or RSS author).
        description: Show description.
        rss_feed_url: RSS feed URL if available.
        platform: Source platform.
        podcast_id: Platform-specific podcast ID.
        image_url: Podcast artwork URL.
        language: Language code (e.g., 'en', 'zh').
        categories: iTunes categories.
    """

    title: str = ""
    author: str = ""
    description: str = ""
    rss_feed_url: str = ""
    platform: PodcastPlatform = PodcastPlatform.UNKNOWN
    podcast_id: str = ""
    image_url: str = ""
    language: str = ""
    categories: List[str] = field(default_factory=list)


@dataclass
class EpisodeMetadata:
    """Single episode metadata extracted from RSS or platform.

    Attributes:
        title: Episode title.
        description: Episode description / show notes.
        audio_url: Direct URL to episode audio file.
        duration: Episode duration in seconds (0 if unknown).
        publish_date: Publication date string.
        episode_id: Platform-specific episode ID.
        podcast: Parent podcast metadata.
        host_names: Known host names extracted from metadata.
        guest_names: Known guest names extracted from metadata.
        show_notes: Full show notes text (may contain speaker info).
    """

    title: str = ""
    description: str = ""
    audio_url: str = ""
    duration: float = 0.0
    publish_date: str = ""
    episode_id: str = ""
    podcast: Optional[PodcastMetadata] = None
    host_names: List[str] = field(default_factory=list)
    guest_names: List[str] = field(default_factory=list)
    show_notes: str = ""


@dataclass
class SpeakerIdentification:
    """Result of speaker identification.

    Attributes:
        name: Speaker name.
        role: Speaker role ('host', 'guest', 'unknown').
        confidence: Confidence score (0.0 to 1.0).
        source: Source of identification ('metadata', 'gemini', 'heuristic').
        tier: Diarization tier assignment (e.g., 'SPEAKER_00').
    """

    name: str
    role: str = "unknown"
    confidence: float = 0.0
    source: str = "unknown"
    tier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "confidence": self.confidence,
            "source": self.source,
            "tier": self.tier,
        }
