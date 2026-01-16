"""YouTube Data Acquisition Module.

This module provides YouTube video metadata extraction, media download,
and caption retrieval functionality powered by yt-dlp.

Key Components:
    YoutubeLoader: Lightweight loader for fetching video metadata and
        caption content in memory. Use this for quick metadata lookups
        or when you don't need to save files to disk.

    YouTubeDownloader: Full-featured downloader for media files and
        captions with disk persistence. Supports various output formats
        and quality settings.

    VideoMetadata: Dataclass containing video information (title, duration,
        channel, upload date, available captions, etc.).

    CaptionTrack: Represents a single caption track with language code,
        format, and content retrieval methods.

Features:
    - Proxy and cookie support for geo-restricted content
    - Automatic caption format detection (manual vs auto-generated)
    - Multiple audio/video format options
    - Async and sync download APIs

Example:
    >>> from lattifai.youtube import YoutubeLoader, VideoMetadata
    >>> loader = YoutubeLoader()
    >>> metadata = loader.get_metadata("https://youtube.com/watch?v=...")
    >>> print(metadata.title, metadata.duration)

Requirements:
    yt-dlp must be installed: `pip install yt-dlp`

See Also:
    - lattifai.client.LattifAI.youtube: High-level YouTube workflow method
"""

from .client import YouTubeDownloader, YoutubeLoader
from .types import CaptionTrack, VideoMetadata

__all__ = ["YoutubeLoader", "YouTubeDownloader", "VideoMetadata", "CaptionTrack"]
