from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    description: str
    duration: float  # seconds
    thumbnail_url: str
    channel_name: str
    view_count: int
    upload_date: Optional[str] = None


@dataclass
class CaptionTrack:
    language_code: str
    language_name: str
    kind: str  # 'manual' | 'asr'
    ext: str  # 'vtt', 'srv3' etc
    url: Optional[str] = None
