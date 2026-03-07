"""Podcast Module for LattifAI.

This module provides podcast metadata extraction, audio download,
and speaker identification for podcast transcription workflows.

Key Components:
    PodcastLoader: Fetches episode metadata and audio from various
        podcast platforms (Apple Podcasts, RSS, Xiaoyuzhou, Spotify).

    SpeakerIdentifier: Identifies speaker names using metadata,
        Gemini AI, and heuristic pattern matching.

    PodcastPlatform: Enum of supported podcast platforms.

    EpisodeMetadata: Dataclass containing episode information
        (title, audio URL, show notes, speaker info).

Example:
    >>> from lattifai.podcast import PodcastLoader, SpeakerIdentifier
    >>> loader = PodcastLoader()
    >>> episode = loader.get_episode_metadata("https://podcasts.apple.com/...")
    >>> print(episode.title, episode.audio_url)
"""

from .client import PodcastLoader
from .speaker_identifier import SpeakerIdentifier
from .types import EpisodeMetadata, PodcastMetadata, PodcastPlatform, SpeakerIdentification

__all__ = [
    "PodcastLoader",
    "SpeakerIdentifier",
    "PodcastPlatform",
    "PodcastMetadata",
    "EpisodeMetadata",
    "SpeakerIdentification",
]
