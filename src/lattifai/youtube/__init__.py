"""
YouTube Data Acquisition Module (powered by yt-dlp)
"""

from .client import YouTubeDownloader, YoutubeLoader
from .types import CaptionTrack, VideoMetadata

__all__ = ["YoutubeLoader", "YouTubeDownloader", "VideoMetadata", "CaptionTrack"]
