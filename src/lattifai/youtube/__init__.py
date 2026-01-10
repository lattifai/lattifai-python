"""
YouTube Data Acquisition Module (powered by yt-dlp)
"""

from .loader import YoutubeLoader
from .types import CaptionSegment, CaptionTrack, VideoMetadata

__all__ = ["YoutubeLoader", "VideoMetadata", "CaptionTrack", "CaptionSegment"]
