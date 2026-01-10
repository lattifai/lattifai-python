"""
YouTube Data Acquisition Module (powered by yt-dlp)
"""

from .downloader import YouTubeDownloader
from .loader import YoutubeLoader
from .types import CaptionSegment, CaptionTrack, VideoMetadata

__all__ = ["YoutubeLoader", "YouTubeDownloader", "VideoMetadata", "CaptionTrack", "CaptionSegment"]
