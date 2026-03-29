"""
YouTube client for metadata extraction and media download using yt-dlp
"""

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

from lattifai.caption.config import CAPTION_FORMATS

from ..errors import LattifAIError
from ..workflow.base import setup_workflow_logger
from ..workflow.file_manager import TRANSCRIBE_CHOICE, FileExistenceManager
from .types import CaptionTrack, VideoMetadata

logger = logging.getLogger(__name__)


class YouTubeError(LattifAIError):
    """Base error for YouTube operations"""

    pass


class VideoUnavailableError(YouTubeError):
    """Video is not available (private, deleted, etc)"""

    pass


class YoutubeLoader:
    """Lightweight YouTube metadata and caption content loader

    Use this class when you need to:
    - Fetch video metadata quickly
    - Get caption content in memory (not save to disk)
    - Support proxy and cookies configuration
    """

    def __init__(self, proxy: Optional[str] = None, cookies: Optional[str] = None):
        if yt_dlp is None:
            raise ImportError("yt-dlp is required. Install with `pip install yt-dlp`")

        # Auto-load from environment if not specified
        if proxy is None:
            proxy = os.getenv("YOUTUBE_PROXY")
        if cookies is None:
            cookies = os.getenv("YOUTUBE_COOKIE_FILE") or os.getenv("YOUTUBE_COOKIE_BROWSER")

        self.proxy = proxy
        self.cookies = cookies

        # Base configuration for metadata extraction
        self._base_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,  # Need full info for captions
            "youtube_include_dash_manifest": False,
            "youtube_include_hls_manifest": False,
        }

        if self.proxy:
            self._base_opts["proxy"] = self.proxy
            logger.info(f"🌐 Using proxy: {self.proxy}")

        # Cookie configuration
        if self.cookies:
            # Check if it's a browser name (chrome, firefox, safari, etc.)
            browser_names = ["chrome", "firefox", "safari", "edge", "opera", "brave"]
            if self.cookies.lower() in browser_names:
                # Use cookies from browser directly
                self._base_opts["cookiesfrombrowser"] = (self.cookies.lower(),)
                logger.info(f"🍪 Using cookies from browser: {self.cookies}")
            else:
                # Use cookie file
                cookie_path = Path(self.cookies).expanduser()
                if cookie_path.exists():
                    self._base_opts["cookiefile"] = str(cookie_path)
                    logger.info(f"🍪 Using cookie file: {cookie_path}")
                else:
                    logger.warning(f"⚠️ Cookie file not found: {cookie_path}")
                    logger.warning("💡 Tip: Run 'yt-dlp --cookies-from-browser chrome' to extract cookies")

        # Note: player_client configuration is removed to avoid format availability issues
        # with certain videos. Let yt-dlp automatically select the best client.
        # Previous config caused "Requested format is not available" errors for some videos.

    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Fetch basic video metadata and list of available captions.
        Returns a dict with 'metadata' (VideoMetadata) and 'captions' (List[CaptionTrack]).
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        opts = {
            **self._base_opts,
            "writesubtitles": True,
            "writeautomaticsub": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Parse metadata
                metadata = VideoMetadata(
                    video_id=info.get("id", video_id),
                    title=info.get("title", "Unknown"),
                    description=info.get("description", ""),
                    duration=float(info.get("duration", 0)),
                    thumbnail_url=info.get("thumbnail", ""),
                    channel_name=info.get("uploader", "Unknown"),
                    view_count=info.get("view_count", 0),
                    upload_date=info.get("upload_date"),
                )

                # Parse captions
                tracks: List[CaptionTrack] = []

                # Manual captions
                subtitles = info.get("subtitles", {})
                for lang, formats in subtitles.items():
                    for fmt in formats:
                        tracks.append(
                            CaptionTrack(
                                language_code=lang,
                                language_name=self._get_lang_name(formats),
                                kind="manual",
                                ext=fmt.get("ext", ""),
                                url=fmt.get("url"),
                            )
                        )

                # Auto captions
                auto_subs = info.get("automatic_captions", {})
                for lang, formats in auto_subs.items():
                    for fmt in formats:
                        tracks.append(
                            CaptionTrack(
                                language_code=lang,
                                language_name=self._get_lang_name(formats),
                                kind="asr",
                                ext=fmt.get("ext", ""),
                                url=fmt.get("url"),
                            )
                        )

                return {"metadata": metadata, "captions": tracks}

        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "Sign in to confirm" in msg or "not a bot" in msg:
                # Bot detection error - provide helpful guidance
                error_msg = (
                    f"🤖 YouTube Bot Detection: Video {video_id} requires authentication.\n\n"
                    "Solutions:\n"
                    "1. Use browser cookies (recommended):\n"
                    "   loader = YoutubeLoader(cookies='chrome')  # or 'firefox', 'safari'\n\n"
                    "2. Export cookie file:\n"
                    "   yt-dlp --cookies-from-browser chrome --cookies cookies.txt <video_url>\n"
                    "   loader = YoutubeLoader(cookies='cookies.txt')\n\n"
                    "3. Environment variable:\n"
                    "   export YOUTUBE_COOKIE_BROWSER=chrome\n\n"
                    f"Original error: {msg}"
                )
                raise VideoUnavailableError(error_msg) from e
            elif "Private video" in msg:
                raise VideoUnavailableError(f"Video {video_id} is private") from e
            raise YouTubeError(f"yt-dlp failed: {msg}") from e
        except Exception as e:
            raise YouTubeError(f"Unexpected error: {str(e)}") from e

    def get_caption(self, video_id: str, lang: str = "en") -> Dict[str, str]:
        """
        Fetch transcript for a specific language.
        Returns a dict with 'content' (raw string) and 'fmt' (format extension).
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        # We need to download json3 or vtt to parse.
        # Ideally we want json3 for precision, but yt-dlp prefers vtt/srv3

        opts = {
            **self._base_opts,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang],
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Look for the requested language in subtitles or automatic_captions
                subs = info.get("subtitles", {}).get(lang)
                if not subs:
                    subs = info.get("automatic_captions", {}).get(lang)

                if not subs:
                    raise YouTubeError(f"No captions found for language: {lang}")

                # Sort to find best format (json3 > vtt > ttml > srv3)
                best_fmt = self._find_best_format(subs)
                if not best_fmt or not best_fmt.get("url"):
                    raise YouTubeError("Could not find a download URL for captions")

                caption_url = best_fmt["url"]
                ext = best_fmt.get("ext")
                content = self._fetch_caption(caption_url)

                return {"content": content, "fmt": ext}

        except Exception as e:
            raise YouTubeError(f"Failed to fetch transcript: {str(e)}") from e

    def _get_lang_name(self, formats: List[Dict]) -> str:
        if formats and "name" in formats[0]:
            return formats[0]["name"]
        return "Unknown"

    def _find_best_format(self, formats: List[Dict]) -> Optional[Dict]:
        # Prefer json3 (best precision), srv3 (word-level timing), then vtt
        priority = ["json3", "srv3", "vtt", "ttml", "srv2", "srv1"]

        for fmt_ext in priority:
            for f in formats:
                if f.get("ext") == fmt_ext:
                    return f
        return formats[0] if formats else None

    def _fetch_caption(self, url: str) -> str:
        import requests

        try:
            resp = requests.get(url, proxies={"https": self.proxy} if self.proxy else None)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.error(f"Error fetching caption: {e}")
            raise YouTubeError("Failed to fetch caption content") from e

    def get_audio_url(
        self,
        video_id: str,
        format_preference: str = "m4a",
        quality: str = "best",
        audio_track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get direct audio-only stream URL for a YouTube video.

        Args:
            video_id: YouTube video ID
            format_preference: Preferred audio format (m4a, webm, opus)
            quality: Audio quality - "best" (highest bitrate), "medium" (~128kbps),
                    "low" (~50kbps), or specific bitrate like "128", "64"
            audio_track_id: Specific audio track ID for multi-language videos (e.g., "en.2")

        Returns:
            Dict with url, mime_type, bitrate, content_length, format_id, ext
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Use base opts (includes proxy and cookie config) + DASH manifest
        opts = {
            **self._base_opts,
            "youtube_include_dash_manifest": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Get all formats and filter for audio-only (no video track)
                formats = info.get("formats", [])

                def is_direct_url(url: str) -> bool:
                    """Check if URL is a direct stream URL (not HLS manifest)"""
                    if not url:
                        return False
                    # HLS manifests contain these patterns
                    hls_patterns = ["manifest.googlevideo.com", "/hls_playlist/", ".m3u8"]
                    return not any(p in url for p in hls_patterns)

                audio_formats = [
                    f
                    for f in formats
                    if f.get("acodec") not in (None, "none")
                    and f.get("vcodec") in (None, "none")
                    and f.get("url")  # Must have a direct URL
                    and is_direct_url(f.get("url"))  # Exclude HLS manifests
                ]

                if not audio_formats:
                    # Fallback: If no audio-only formats, use lowest resolution video with audio
                    # This happens with HLS-only videos (e.g., protected content)
                    logger.warning("No audio-only formats found. Falling back to lowest resolution video with audio.")
                    audio_formats = [
                        f
                        for f in formats
                        if f.get("acodec") not in (None, "none")
                        and f.get("vcodec") not in (None, "none")
                        and f.get("url")
                        and is_direct_url(f.get("url"))  # Exclude HLS manifests
                    ]
                    # Sort by resolution (lowest first) for minimal bandwidth
                    audio_formats.sort(key=lambda f: f.get("height") or f.get("width") or 9999)

                if not audio_formats:
                    # Check if there are HLS-only formats (common for Shorts)
                    # HLS can still work with server-side streaming (same IP)
                    hls_with_audio = [f for f in formats if f.get("acodec") not in (None, "none") and f.get("url")]
                    if hls_with_audio:
                        logger.warning("Only HLS streams available. Returning HLS URL for server-side streaming.")
                        # Sort: prefer audio-only, then by resolution (lowest first)
                        hls_with_audio.sort(
                            key=lambda f: (
                                0 if f.get("vcodec") in (None, "none") else 1,
                                f.get("height") or f.get("width") or 9999,
                            )
                        )
                        audio_formats = hls_with_audio
                    else:
                        raise YouTubeError(
                            "No formats with audio available. YouTube may require authentication for this video."
                        )

                # Filter by audio_track_id if specified (for multi-language audio)
                if audio_track_id:
                    # yt-dlp uses format_id patterns like "251-0" or "audio_track" field
                    # Try matching by format_id suffix or audio_track field
                    track_filtered = [
                        f
                        for f in audio_formats
                        if f.get("audio_track", {}).get("id") == audio_track_id
                        or (f.get("format_id") and audio_track_id in f.get("format_id", ""))
                        or f.get("language") == audio_track_id.split(".")[0]  # e.g., "en" from "en.2"
                    ]
                    if track_filtered:
                        audio_formats = track_filtered
                        logger.info(f"Filtered to {len(audio_formats)} formats for audio_track_id={audio_track_id}")

                # Parse quality parameter
                # "best" = highest bitrate, "medium" ~128kbps, "low" ~50kbps
                quality_tier = quality.lower()
                if quality_tier == "best":
                    max_bitrate = float("inf")
                elif quality_tier == "medium":
                    max_bitrate = 160  # Allow up to 160kbps for "medium"
                elif quality_tier == "low":
                    max_bitrate = 70  # Allow up to 70kbps for "low"
                elif quality_tier.isdigit():
                    max_bitrate = int(quality_tier) + 20  # Allow some tolerance
                else:
                    max_bitrate = float("inf")  # Default to best

                # Sort by preference: format match > bitrate (within limit)
                def score_format(f: Dict) -> tuple:
                    ext = f.get("ext", "")
                    ext_match = 2 if ext == format_preference else 0
                    # Prefer m4a/webm over other formats
                    common_format = 1 if ext in ("m4a", "webm", "opus") else 0
                    bitrate = f.get("abr") or f.get("tbr") or 0

                    # For quality tiers, filter then maximize
                    if bitrate <= max_bitrate:
                        quality_score = bitrate  # Higher is better within limit
                    else:
                        quality_score = -1000  # Exclude formats exceeding limit

                    return (ext_match, common_format, quality_score)

                audio_formats.sort(key=score_format, reverse=True)
                best = audio_formats[0]

                # Check if selected format is HLS (requires server-side streaming)
                best_url = best.get("url", "")
                is_hls = not is_direct_url(best_url)

                return {
                    "url": best_url,
                    "mime_type": best.get("ext", format_preference),
                    "bitrate": best.get("abr") or best.get("tbr"),
                    "sample_rate": best.get("asr"),  # Audio sample rate
                    "content_length": best.get("filesize") or best.get("filesize_approx"),
                    "format_id": best.get("format_id"),
                    "ext": best.get("ext"),
                    "is_hls": is_hls,  # True = use server streaming, False = use proxy
                }

        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "Sign in to confirm" in msg or "not a bot" in msg:
                raise YouTubeError(
                    f"🤖 YouTube Bot Detection: Cookie configuration required to access this video. "
                    f"Reference: YoutubeLoader(cookies='chrome') or set environment variable YOUTUBE_COOKIE_BROWSER=chrome. "
                    f"Original error: {msg}"
                ) from e
            raise YouTubeError(f"Failed to get audio URL: {msg}") from e
        except Exception as e:
            raise YouTubeError(f"Unexpected error getting audio URL: {str(e)}") from e

    def get_video_url(self, video_id: str, format_preference: str = "mp4", quality: str = "best") -> Dict[str, Any]:
        """
        Get direct video stream URL for a YouTube video.

        Args:
            video_id: YouTube video ID
            format_preference: Preferred video format (mp4, webm)
            quality: Video quality (best, 1080, 720, 480, 360)

        Returns:
            Dict with url, mime_type, width, height, fps, vcodec, acodec, bitrate, content_length, format_id, ext

        Note:
            Prioritizes formats that include both video AND audio to avoid silent videos.
            YouTube separates high-quality video and audio streams; we prefer pre-muxed formats.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Use base opts (includes proxy and cookie config) + DASH and HLS manifests
        opts = {
            **self._base_opts,
            "youtube_include_dash_manifest": True,
            "youtube_include_hls_manifest": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Get all formats
                formats = info.get("formats", [])

                def is_direct_url(url: str) -> bool:
                    """Check if URL is a direct stream URL (not HLS manifest)"""
                    if not url:
                        return False
                    hls_patterns = ["manifest.googlevideo.com", "/hls_playlist/", ".m3u8"]
                    return not any(p in url for p in hls_patterns)

                # Filter for video formats:
                # - Must have video codec
                # - Must have a URL
                # - Prefer direct URLs (DASH) over HLS manifests
                def is_usable_video(f: Dict) -> bool:
                    if f.get("vcodec") in (None, "none"):
                        return False
                    if not f.get("url"):
                        return False
                    return True

                # First try: direct URLs only (exclude HLS)
                video_formats = [f for f in formats if is_usable_video(f) and is_direct_url(f.get("url", ""))]

                # Fallback: include HLS if no direct formats
                if not video_formats:
                    logger.warning("No direct video URLs found. Falling back to HLS formats.")
                    video_formats = [f for f in formats if is_usable_video(f)]

                if not video_formats:
                    raise YouTubeError("No video formats available")

                # Parse target height from quality parameter
                target_height = None
                if quality != "best" and quality.isdigit():
                    target_height = int(quality)

                # Sort by preference: has_audio (MOST IMPORTANT) > format match > resolution > bitrate
                # YouTube high-quality streams are often video-only; we MUST prefer formats with audio
                def score_format(f: Dict) -> tuple:
                    ext = f.get("ext", "")
                    ext_match = 1 if ext == format_preference else 0
                    height = f.get("height") or 0
                    bitrate = f.get("tbr") or f.get("vbr") or 0
                    # has_audio is now the HIGHEST priority - video without audio is useless for most users
                    has_audio = 10 if f.get("acodec") not in (None, "none") else 0

                    # For quality filtering, penalize formats exceeding target
                    height_score = height
                    if target_height and height > target_height:
                        height_score = -1000  # Heavily penalize exceeding target

                    return (has_audio, ext_match, height_score, bitrate)

                video_formats.sort(key=score_format, reverse=True)
                best = video_formats[0]

                # Check if selected format is HLS
                best_url = best.get("url", "")
                is_hls = not is_direct_url(best_url)

                # Log selection for debugging
                logger.info(
                    f"Selected video format: {best.get('format_id')} "
                    f"({best.get('width')}x{best.get('height')}, "
                    f"vcodec={best.get('vcodec')}, acodec={best.get('acodec')}, is_hls={is_hls})"
                )

                return {
                    "url": best_url,
                    "mime_type": best.get("ext", format_preference),
                    "width": best.get("width"),
                    "height": best.get("height"),
                    "fps": best.get("fps"),
                    "vcodec": best.get("vcodec"),
                    "acodec": best.get("acodec"),
                    "bitrate": best.get("tbr") or best.get("vbr"),
                    "content_length": best.get("filesize") or best.get("filesize_approx"),
                    "format_id": best.get("format_id"),
                    "ext": best.get("ext"),
                    "is_hls": is_hls,
                }

        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "Sign in to confirm" in msg or "not a bot" in msg:
                raise YouTubeError(
                    f"🤖 YouTube Bot Detection: Cookie configuration required to access this video. "
                    f"Reference: YoutubeLoader(cookies='chrome') or set environment variable YOUTUBE_COOKIE_BROWSER=chrome. "
                    f"Original error: {msg}"
                ) from e
            raise YouTubeError(f"Failed to get video URL: {msg}") from e
        except Exception as e:
            raise YouTubeError(f"Unexpected error getting video URL: {str(e)}") from e


class YouTubeDownloader:
    """YouTube media and caption file downloader using yt-dlp

    Use this class when you need to:
    - Download audio/video files to disk
    - Download caption files to disk
    - Manage file existence and overwrite options
    - Async download support
    """

    def __init__(self):
        if yt_dlp is None:
            raise ImportError("yt-dlp is required. Install with `pip install yt-dlp`")

        self.logger = setup_workflow_logger("youtube")
        self.logger.info(f"yt-dlp version: {yt_dlp.version.__version__}")

    def _normalize_audio_quality(self, quality: str) -> str:
        """
        Normalize quality parameter for audio downloads.

        Handles cross-type quality values (e.g., video resolution used for audio).

        Args:
            quality: Raw quality string

        Returns:
            Normalized audio quality string
        """
        quality_lower = quality.lower()

        # Direct audio quality values
        if quality_lower in ("best", "medium", "low"):
            return quality_lower

        # Numeric values need interpretation
        if quality_lower.isdigit():
            value = int(quality_lower)
            # Values > 320 are likely video resolutions, not audio bitrates
            if value > 320:
                self.logger.warning(f"⚠️ Quality '{quality}' looks like video resolution, using 'best' for audio")
                return "best"
            # Values <= 320 are reasonable audio bitrates
            return quality_lower

        # Unknown value, default to best
        return "best"

    def _normalize_video_quality(self, quality: str) -> str:
        """
        Normalize quality parameter for video downloads.

        Handles cross-type quality values (e.g., audio bitrate/quality used for video).

        Args:
            quality: Raw quality string

        Returns:
            Normalized video quality string
        """
        quality_lower = quality.lower()

        # Map audio quality terms to video equivalents
        if quality_lower == "low":
            self.logger.info("🎬 Mapping audio quality 'low' to video 360p")
            return "360"
        elif quality_lower == "medium":
            self.logger.info("🎬 Mapping audio quality 'medium' to video 720p")
            return "720"
        elif quality_lower == "best":
            return "best"

        # Numeric values
        if quality_lower.isdigit():
            value = int(quality_lower)
            # Values <= 320 are likely audio bitrates, not video resolutions
            if value <= 320:
                self.logger.warning(f"⚠️ Quality '{quality}' looks like audio bitrate, using 'best' for video")
                return "best"
            # Values > 320 are reasonable video resolutions
            return quality_lower

        # Unknown value, default to best
        return "best"

    def _build_audio_format_selector(self, audio_track_id: Optional[str], quality: str = "best") -> str:
        """
        Build yt-dlp format selector string for audio track and quality selection.

        Args:
            audio_track_id: Audio track selection:
                - "original": Select the original audio track (format_id contains "drc")
                - Language code (e.g., "en", "ja"): Select by language
                - Format ID (e.g., "251-drc"): Select specific format
                - None: No filtering
            quality: Audio quality:
                - "best": Highest bitrate (default)
                - "medium": ~128 kbps
                - "low": ~50 kbps
                - Numeric string (e.g., "128"): Target bitrate in kbps

        Returns:
            yt-dlp format selector string
        """
        # Normalize quality for audio context
        quality_lower = self._normalize_audio_quality(quality)

        # Build quality filter
        quality_filter = ""
        if quality_lower == "medium":
            quality_filter = "[abr<=160]"
            self.logger.info("🎵 Audio quality: medium (~128 kbps)")
        elif quality_lower == "low":
            quality_filter = "[abr<=70]"
            self.logger.info("🎵 Audio quality: low (~50 kbps)")
        elif quality_lower.isdigit():
            max_bitrate = int(quality_lower) + 20  # Allow some tolerance
            quality_filter = f"[abr<={max_bitrate}]"
            self.logger.info(f"🎵 Audio quality: ~{quality_lower} kbps")
        # "best" = no filter, use bestaudio

        # Build track filter
        if audio_track_id is None:
            return f"bestaudio{quality_filter}/bestaudio/best"

        if audio_track_id.lower() == "original":
            self.logger.info("🎵 Selecting original audio track (format_id contains 'drc')")
            return f"bestaudio[format_id*=drc]{quality_filter}/bestaudio{quality_filter}/bestaudio/best"

        # Check if it looks like a format_id (contains hyphen or is numeric)
        if "-" in audio_track_id or audio_track_id.isdigit():
            self.logger.info(f"🎵 Selecting audio by format_id: {audio_track_id}")
            return f"bestaudio[format_id={audio_track_id}]{quality_filter}/bestaudio{quality_filter}/bestaudio/best"

        # Assume it's a language code
        self.logger.info(f"🎵 Selecting audio by language: {audio_track_id}")
        return f"bestaudio[language^={audio_track_id}]{quality_filter}/bestaudio{quality_filter}/bestaudio/best"

    def _build_video_format_selector(self, audio_format_selector: str, quality: str = "best") -> str:
        """
        Build yt-dlp format selector string for video with quality selection.

        Args:
            audio_format_selector: Audio format selector from _build_audio_format_selector
            quality: Video quality:
                - "best": Highest resolution (default)
                - "low": 360p
                - "medium": 720p
                - "1080", "720", "480", "360": Target resolution

        Returns:
            yt-dlp format selector string
        """
        # Normalize quality for video context
        quality_lower = self._normalize_video_quality(quality)

        if quality_lower.isdigit():
            height = int(quality_lower)
            self.logger.info(f"🎬 Video quality: {height}p")
            return f"bestvideo[height<={height}]+{audio_format_selector}/best[height<={height}]/best"

        # "best" or fallback
        return f"bestvideo*+{audio_format_selector}/best"

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extract video ID from YouTube URL

        Supports various YouTube URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - https://m.youtube.com/watch?v=VIDEO_ID

        Returns:
            Video ID (e.g., 'cprOj8PWepY')
        """
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
            r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return "youtube_media"

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video metadata without downloading"""
        self.logger.info(f"🔍 Extracting video info for: {url}")

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _extract_info():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(url, download=False)

            metadata = await loop.run_in_executor(None, _extract_info)

            # Extract relevant info
            info = {
                "title": metadata.get("title", "Unknown"),
                "duration": metadata.get("duration", 0),
                "uploader": metadata.get("uploader", "Unknown"),
                "upload_date": metadata.get("upload_date", "Unknown"),
                "view_count": metadata.get("view_count", 0),
                "description": metadata.get("description", ""),
                "thumbnail": metadata.get("thumbnail", ""),
                "webpage_url": metadata.get("webpage_url", url),
                "chapters": metadata.get("chapters") or [],
                "categories": metadata.get("categories") or [],
                "channel": metadata.get("channel", ""),
            }

            self.logger.info(f'✅ Video info extracted: {info["title"]}')
            return info

        except yt_dlp.utils.DownloadError as e:
            self.logger.error(f"Failed to extract video info: {str(e)}")
            raise RuntimeError(f"Failed to extract video info: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to parse video metadata: {str(e)}")
            raise RuntimeError(f"Failed to parse video metadata: {str(e)}")

    async def download_media(
        self,
        url: str,
        output_dir: Optional[str] = None,
        media_format: Optional[str] = None,
        force_overwrite: bool = False,
        audio_track_id: Optional[str] = "original",
        quality: str = "best",
    ) -> str:
        """
        Download media (audio or video) from YouTube URL based on format

        This is a unified method that automatically selects between audio and video
        download based on the media format extension.

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            media_format: Media format - audio (mp3, wav, m4a, aac, opus, ogg, flac, aiff)
                         or video (mp4, webm, mkv, avi, mov, etc.) (default: mp3)
            force_overwrite: Skip user confirmation and overwrite existing files
            audio_track_id: Audio track selection for multi-language videos:
                - "original": Select the original audio track (default)
                - Language code (e.g., "en", "ja"): Select by language
                - Format ID (e.g., "251-drc"): Select specific format
                - None: No filtering, use yt-dlp default
            quality: Media quality selection:
                For audio: "best", "medium", "low", or bitrate like "128"
                For video: "best", "1080", "720", "480", "360"

        Returns:
            Path to downloaded media file
        """
        media_format = media_format or "mp3"

        # Determine if format is audio or video
        audio_formats = ["mp3", "wav", "m4a", "aac", "opus", "ogg", "flac", "aiff"]
        is_audio = media_format.lower() in audio_formats

        if is_audio:
            self.logger.info(f"🎵 Detected audio format: {media_format}")
            return await self.download_audio(
                url=url,
                output_dir=output_dir,
                media_format=media_format,
                force_overwrite=force_overwrite,
                audio_track_id=audio_track_id,
                quality=quality,
            )
        else:
            self.logger.info(f"🎬 Detected video format: {media_format}")
            return await self.download_video(
                url=url,
                output_dir=output_dir,
                video_format=media_format,
                force_overwrite=force_overwrite,
                audio_track_id=audio_track_id,
                quality=quality,
            )

    async def _download_media_internal(
        self,
        url: str,
        output_dir: str,
        media_format: str,
        is_audio: bool,
        force_overwrite: bool = False,
        audio_track_id: Optional[str] = "original",
        quality: str = "best",
    ) -> str:
        """
        Internal unified method for downloading audio or video from YouTube

        Args:
            url: YouTube URL
            output_dir: Output directory
            media_format: Media format (audio or video extension)
            is_audio: True for audio download, False for video download
            force_overwrite: Skip user confirmation and overwrite existing files
            audio_track_id: Audio track selection for multi-language videos:
                - "original": Select the original audio track (default)
                - Language code (e.g., "en", "ja"): Select by language
                - Format ID (e.g., "251-drc"): Select specific format
                - None: No filtering, use yt-dlp default
            quality: Media quality selection:
                For audio: "best", "medium", "low", or bitrate like "128"
                For video: "best", "1080", "720", "480", "360"

        Returns:
            Path to downloaded media file
        """
        target_dir = Path(output_dir).expanduser()
        media_type = "audio" if is_audio else "video"
        emoji = "🎵" if is_audio else "🎬"

        self.logger.info(f"{emoji} Downloading {media_type} from: {url}")
        self.logger.info(f"📁 Output directory: {target_dir}")
        self.logger.info(f'{"🎶" if is_audio else "🎥"} Media format: {media_format}')

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing files
        video_id = self.extract_video_id(url)
        existing_files = FileExistenceManager.check_existing_files(video_id, str(target_dir), [media_format])

        # Handle existing files
        if existing_files["media"] and not force_overwrite:
            if FileExistenceManager.is_interactive_mode():
                user_choice = FileExistenceManager.prompt_user_confirmation(
                    {"media": existing_files["media"]}, "media download"
                )

                if user_choice == "cancel":
                    raise RuntimeError("Media download cancelled by user")
                elif user_choice == "overwrite":
                    # Continue with download
                    pass
                elif user_choice in existing_files["media"]:
                    # User selected a specific file
                    return user_choice
                else:
                    # Fallback: use first file
                    self.logger.info(f'✅ Using existing media file: {existing_files["media"][0]}')
                    return existing_files["media"][0]
            else:
                # Non-interactive mode: use existing file
                self.logger.info(f'✅ Using existing media file: {existing_files["media"][0]}')
                return existing_files["media"][0]

        # Generate output filename template
        output_template = str(target_dir / f"{video_id}.%(ext)s")

        # Build format selector with audio track and quality filtering
        audio_format_selector = self._build_audio_format_selector(audio_track_id, quality)

        # Build yt-dlp options based on media type
        if is_audio:
            opts = {
                "format": audio_format_selector,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": media_format,
                        "preferredquality": "0",  # Best quality for conversion
                    }
                ],
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": True,
            }
        else:
            # For video, combine video with selected audio track
            video_format_selector = self._build_video_format_selector(audio_format_selector, quality)
            opts = {
                "format": video_format_selector,
                "merge_output_format": media_format,
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": True,
            }

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _download():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])

            await loop.run_in_executor(None, _download)

            self.logger.info(f"✅ {media_type.capitalize()} download completed")

            # Check for expected file format
            expected_file = target_dir / f"{video_id}.{media_format}"
            if expected_file.exists():
                self.logger.info(f"{emoji} Downloaded {media_type}: {expected_file}")
                return str(expected_file)

            # Fallback: search for media files with this video_id
            if is_audio:
                fallback_extensions = [media_format, "mp3", "wav", "m4a", "aac"]
            else:
                fallback_extensions = [media_format, "mp4", "webm", "mkv"]

            for ext in fallback_extensions:
                files = list(target_dir.glob(f"{video_id}*.{ext}"))
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    self.logger.info(f"{emoji} Found {media_type} file: {latest_file}")
                    return str(latest_file)

            raise RuntimeError(f"Downloaded {media_type} file not found")

        except yt_dlp.utils.DownloadError as e:
            self.logger.error(f"Failed to download {media_type}: {str(e)}")
            raise RuntimeError(f"Failed to download {media_type}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to download {media_type}: {str(e)}")
            raise RuntimeError(f"Failed to download {media_type}: {str(e)}")

    async def download_audio(
        self,
        url: str,
        output_dir: Optional[str] = None,
        media_format: Optional[str] = None,
        force_overwrite: bool = False,
        audio_track_id: Optional[str] = "original",
        quality: str = "best",
    ) -> str:
        """
        Download audio from YouTube URL

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            media_format: Audio format (default: mp3)
            force_overwrite: Skip user confirmation and overwrite existing files
            audio_track_id: Audio track selection for multi-language videos
            quality: Audio quality ("best", "medium", "low", or bitrate like "128")

        Returns:
            Path to downloaded audio file
        """
        target_dir = output_dir or tempfile.gettempdir()
        media_format = media_format or "mp3"
        return await self._download_media_internal(
            url,
            target_dir,
            media_format,
            is_audio=True,
            force_overwrite=force_overwrite,
            audio_track_id=audio_track_id,
            quality=quality,
        )

    async def download_video(
        self,
        url: str,
        output_dir: Optional[str] = None,
        video_format: str = "mp4",
        force_overwrite: bool = False,
        audio_track_id: Optional[str] = "original",
        quality: str = "best",
    ) -> str:
        """
        Download video from YouTube URL

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            video_format: Video format
            force_overwrite: Skip user confirmation and overwrite existing files
            audio_track_id: Audio track selection for multi-language videos
            quality: Video quality ("best", "1080", "720", "480", "360")

        Returns:
            Path to downloaded video file
        """
        target_dir = output_dir or tempfile.gettempdir()
        return await self._download_media_internal(
            url,
            target_dir,
            video_format,
            is_audio=False,
            force_overwrite=force_overwrite,
            audio_track_id=audio_track_id,
            quality=quality,
        )

    @staticmethod
    def _extract_transcript_url_from_description(description: str) -> Optional[str]:
        """Extract a transcript URL from YouTube video description.

        Looks for patterns like:
          *Transcript:*
          https://lexfridman.com/some-guest-transcript

        Or inline:
          Transcript: https://example.com/transcript
          Substack Article w/Show Notes: https://www.latent.space/p/jeffdean

        Returns:
            Full transcript URL or None
        """
        if not description:
            return None

        # Keywords that indicate a transcript or show notes URL
        _LABEL_KEYWORDS = r"transcript|show\s*notes|rescript"

        lines = description.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            # Check if this line is a standalone label (e.g., "*Transcript:*", "Show Notes:")
            if re.match(rf"^[\*_]*(?:{_LABEL_KEYWORDS})[\s:]*[\*_]*\s*$", stripped):
                # Look at the next non-empty line for URL
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith("http"):
                        return next_line
            # Check if URL is on the same line after a keyword
            # Matches: "Transcript: URL", "Show Notes: URL", "Substack Article w/Show Notes: URL"
            m = re.search(rf"(?:{_LABEL_KEYWORDS})\s*[:\-–]\s*(https?://\S+)", stripped)
            if m:
                # Re-extract from original case line to preserve URL case
                m2 = re.search(r"https?://\S+", line)
                if m2:
                    return m2.group(0)

        return None

    async def _find_podscripts_url(self, video_title: str, channel_name: str) -> Optional[str]:
        """Search podscripts.co for a transcript matching the video title.

        Podscripts.co hosts third-party transcripts for popular podcasts.
        This method searches the podcast's episode listing page and fuzzy-matches
        the video title to find the correct episode URL.
        """
        import urllib.request

        if not video_title:
            return None

        # Map known channels to podscripts slugs
        _CHANNEL_SLUGS = {
            "No Priors: AI, Machine Learning, Tech, & Startups": "no-priors-artificial-intelligence-technology-startups",
            "No Priors": "no-priors-artificial-intelligence-technology-startups",
            "Machine Learning Street Talk": "machine-learning-street-talk",
            "Latent Space": "latent-space-the-ai-engineer-podcast",
            "Dwarkesh Patel": "the-dwarkesh-patel-podcast",
            "Lex Fridman": "lex-fridman-podcast",
        }

        slug = _CHANNEL_SLUGS.get(channel_name)
        if not slug:
            # Try to find slug by normalizing channel name
            normalized = channel_name.lower().replace(" ", "-").replace(".", "")
            slug = normalized

        listing_url = f"https://podscripts.co/podcasts/{slug}"
        self.logger.info(f"🔍 Searching podscripts.co: {listing_url}")

        try:
            loop = asyncio.get_event_loop()

            def _fetch_listing():
                req = urllib.request.Request(
                    listing_url,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode("utf-8")

            html = await loop.run_in_executor(None, _fetch_listing)

            # Extract all episode links
            episode_links = re.findall(rf'href="(/podcasts/{re.escape(slug)}/[^"]+)"', html)
            episode_links = list(set(episode_links))  # deduplicate

            if not episode_links:
                return None

            # Fuzzy match: extract keywords from video title and find best match
            title_words = set(
                w.lower() for w in re.findall(r"[a-zA-Z]+", video_title) if len(w) > 3  # skip short words
            )

            best_match = None
            best_score = 0
            for link in episode_links:
                # Extract slug words from URL
                slug_part = link.split("/")[-1]
                slug_words = set(slug_part.replace("-", " ").split())
                # Score by word overlap
                score = len(title_words & slug_words)
                if score > best_score:
                    best_score = score
                    best_match = link

            if best_match and best_score >= 2:
                full_url = f"https://podscripts.co{best_match}"
                self.logger.info(f"✅ Matched podscripts episode (score={best_score}): {full_url}")
                return full_url

            self.logger.info(f"No matching episode found on podscripts.co (best score={best_score})")
            return None

        except Exception as e:
            self.logger.debug(f"podscripts.co search failed: {e}")
            return None

    async def _download_external_transcript(
        self,
        transcript_url: str,
        output_dir: str,
        video_id: str,
        youtube_url: Optional[str] = None,
        video_info: Optional[Dict[str, Any]] = None,
        force_overwrite: bool = False,
    ) -> Optional[str]:
        """Download and parse a transcript from an external URL.

        Fetches the webpage, extracts text content with speaker labels and timestamps.
        Saves as {video_id}.transcript.md in markdown transcript format compatible
        with lattifai-captions MarkdownReader.

        Uses urllib first (fast, no dependencies). If the result looks like a
        JS-rendered SPA (very little text content), falls back to headless Chrome
        ``--dump-dom`` which executes JavaScript before returning the DOM.

        Returns:
            Path to saved transcript file, or None on failure
        """
        output_path = Path(output_dir).expanduser() / f"{video_id}.transcript.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded (unless force_overwrite)
        if output_path.exists() and not force_overwrite:
            self.logger.info(f"✅ Using existing external transcript: {output_path}")
            return str(output_path)

        self.logger.info(f"📥 Downloading external transcript from: {transcript_url}")

        try:
            loop = asyncio.get_event_loop()

            # Rescript API shortcut: SPA site with a JSON API
            rescript_match = re.match(r"https://app\.rescript\.info/public/share/(.+)", transcript_url)
            if rescript_match:
                transcript_text = await self._fetch_rescript_transcript(transcript_url, loop)
                if transcript_text:
                    frontmatter = self._build_transcript_frontmatter(video_info, youtube_url, transcript_url)
                    output_path.write_text(frontmatter + transcript_text, encoding="utf-8")
                    self.logger.info(f"✅ Saved Rescript transcript: {output_path} ({len(transcript_text)} chars)")
                    return str(output_path)

            # Quick reachability check — avoid wasting 100s+ on unreachable hosts
            from urllib.parse import urlparse

            host = urlparse(transcript_url).hostname
            reachable = await loop.run_in_executor(None, self._is_host_reachable, host)

            html = None
            if reachable:
                # Direct fetch with proxy support from environment
                def _fetch():
                    import urllib.request

                    proxy_handler = urllib.request.ProxyHandler()
                    opener = urllib.request.build_opener(proxy_handler)
                    req = urllib.request.Request(
                        transcript_url,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                    )
                    with opener.open(req, timeout=15) as resp:
                        return resp.read().decode("utf-8")

                try:
                    html = await loop.run_in_executor(None, _fetch)
                except Exception as e:
                    self.logger.info(f"🔄 urllib failed ({e}), trying fallbacks...")
            else:
                self.logger.info(f"🔄 Host {host} unreachable, skipping direct fetch")

            # Fallback 1: headless Chrome (only if host is reachable — Chrome also connects directly)
            if not html and reachable:
                html = await loop.run_in_executor(None, self._fetch_with_headless_chrome, transcript_url, 20)

            # Fallback 2: Jina Reader API (server-side fetch, bypasses local network blocks)
            if not html:
                self.logger.info("🔄 Trying Jina Reader API (server-side fetch)...")
                jina_md = await loop.run_in_executor(None, self._fetch_with_jina_reader, transcript_url, 15)
                if jina_md:
                    frontmatter = self._build_transcript_frontmatter(video_info, youtube_url, transcript_url)
                    output_path.write_text(frontmatter + jina_md, encoding="utf-8")
                    self.logger.info(f"✅ Saved transcript via Jina Reader: {output_path} ({len(jina_md)} chars)")
                    return str(output_path)

            if not html:
                if not reachable:
                    self.logger.warning(
                        f"Host {host} is unreachable and Jina Reader also failed. "
                        "Try setting HTTPS_PROXY or use a VPN."
                    )
                else:
                    self.logger.warning("Failed to fetch transcript page")
                return None

            # Parse HTML to extract transcript in markdown format
            transcript_text = self._parse_transcript_html(html, youtube_url=youtube_url)

            # Detect SPA: if parsed text is too short but HTML is large, content is JS-rendered
            if (not transcript_text or len(transcript_text) < 200) and len(html) > 2000:
                spa_indicators = ['<div id="root"></div>', '<div id="app"></div>', "noscript>You need to enable"]
                if any(indicator in html for indicator in spa_indicators):
                    self.logger.info("🔄 SPA detected, falling back to headless Chrome...")
                    html = await loop.run_in_executor(None, self._fetch_with_headless_chrome, transcript_url)
                    if html:
                        transcript_text = self._parse_transcript_html(html, youtube_url=youtube_url)

            if not transcript_text:
                # Last resort: Jina Reader API
                self.logger.info("🔄 HTML parsing failed, trying Jina Reader API...")
                jina_md = await loop.run_in_executor(None, self._fetch_with_jina_reader, transcript_url)
                if jina_md:
                    frontmatter = self._build_transcript_frontmatter(video_info, youtube_url, transcript_url)
                    output_path.write_text(frontmatter + jina_md, encoding="utf-8")
                    self.logger.info(f"✅ Saved transcript via Jina Reader: {output_path} ({len(jina_md)} chars)")
                    return str(output_path)
                self.logger.warning("Failed to extract transcript content from page")
                return None

            frontmatter = self._build_transcript_frontmatter(video_info, youtube_url, transcript_url)
            output_path.write_text(frontmatter + transcript_text, encoding="utf-8")
            self.logger.info(f"✅ Saved external transcript: {output_path} ({len(transcript_text)} chars)")
            return str(output_path)

        except Exception as e:
            self.logger.warning(f"Failed to download external transcript: {e}")
            return None

    @staticmethod
    def _is_host_reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
        """Quick TCP connect check to determine if a host is reachable."""
        import socket

        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, OSError):
            return False

    @staticmethod
    def _build_transcript_frontmatter(
        video_info: Optional[Dict[str, Any]],
        youtube_url: Optional[str],
        transcript_url: str,
    ) -> str:
        """Build YAML frontmatter string for transcript files."""
        if not video_info:
            return ""
        fm_fields = []
        for key in ["title", "duration", "upload_date"]:
            if video_info.get(key):
                fm_fields.append(f"{key}: {video_info[key]}")
        if video_info.get("uploader"):
            fm_fields.append(f"channel: {video_info['uploader']}")
        if youtube_url:
            fm_fields.append(f'url: "{youtube_url}"')
        fm_fields.append(f"transcript_source: {transcript_url}")
        desc = video_info.get("description", "")
        if desc:
            for marker in ["*SPONSORS:", "*CONTACT ", "*EPISODE LINKS:", "*PODCAST LINKS:", "*SOCIAL LINKS:"]:
                pos = desc.find(marker)
                if pos > 0:
                    desc = desc[:pos].rstrip()
                    break
            desc = desc.replace("\n\n", "\n")
            fm_fields.append("description: |")
            for line in desc.split("\n"):
                fm_fields.append(f"  {line}")
        return "---\n" + "\n".join(fm_fields) + "\n---\n\n"

    async def _fetch_rescript_transcript(self, transcript_url: str, loop) -> Optional[str]:
        """Fetch transcript from Rescript API (SPA with JSON backend).

        Rescript (app.rescript.info) stores transcripts as JSON accessible via
        /api/public/share/{share_id}. The response contains `final_transcript`
        with speaker labels and timestamps.
        """
        import json
        import urllib.request

        api_url = transcript_url.replace("/public/share/", "/api/public/share/")
        self.logger.info(f"🔗 Fetching Rescript API: {api_url}")

        def _fetch_api():
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            data = await loop.run_in_executor(None, _fetch_api)
        except Exception as e:
            self.logger.warning(f"Rescript API failed: {e}")
            return None

        session = data.get("session", {})
        transcript = session.get("final_transcript") or session.get("transcript", "")
        if not transcript:
            self.logger.warning("Rescript API returned no transcript content")
            return None

        # Parse Rescript format: "Speaker [HH:MM:SS]:\n  text\n\n  [HH:MM:SS]\n\n  text continued"
        # into markdown format: **Speaker:** text [HH:MM:SS]
        lines = transcript.split("\n")
        segments = []
        current_speaker = ""
        current_hms = ""
        current_text = []

        speaker_pattern = re.compile(r"^(.+?)\s*\[(\d{1,2}:\d{2}:\d{2})\]\s*:?\s*$")
        ts_pattern = re.compile(r"^\s*\[(\d{1,2}:\d{2}:\d{2})\]\s*$")

        for line in lines:
            sp_m = speaker_pattern.match(line)
            ts_m = ts_pattern.match(line)
            if sp_m:
                # Save previous segment
                if current_text:
                    text = " ".join(current_text).strip()
                    if text:
                        segments.append({"speaker": current_speaker, "hms": current_hms, "text": text})
                current_speaker = sp_m.group(1).strip()
                current_hms = sp_m.group(2)
                current_text = []
            elif ts_m:
                # Mid-segment timestamp: save current text as segment, start new one
                if current_text:
                    text = " ".join(current_text).strip()
                    if text:
                        segments.append({"speaker": current_speaker, "hms": current_hms, "text": text})
                current_hms = ts_m.group(1)
                current_text = []
            else:
                stripped = line.strip()
                if stripped:
                    current_text.append(stripped)

        # Don't forget the last segment
        if current_text:
            text = " ".join(current_text).strip()
            if text:
                segments.append({"speaker": current_speaker, "hms": current_hms, "text": text})

        if not segments:
            return transcript  # Return raw text as fallback

        # Extract chapters from _narrativeData (title + time range)
        narrative = session.get("_narrativeData") or {}
        chapters = []
        for title, chapter_data in narrative.items():
            if not isinstance(chapter_data, dict):
                continue
            spans = chapter_data.get("spans", [])
            if spans:
                # Use the earliest span start as chapter start time
                start_sec = min(s.get("start", 0) for s in spans if isinstance(s, dict))
                chapters.append({"title": title, "start": start_sec})
        chapters.sort(key=lambda c: c["start"])

        # Build TOC if chapters exist
        toc_lines = []
        if chapters:

            def _secs_to_hms(secs):
                h, r = divmod(int(secs), 3600)
                m, s = divmod(r, 60)
                return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

            toc_lines.append("## Table of Contents")
            for ch in chapters:
                toc_lines.append(f"- {_secs_to_hms(ch['start'])} – {ch['title']}")
            toc_lines.append("")

        # Convert chapter start times to HH:MM:SS for insertion into transcript
        chapter_by_hms = {}
        if chapters:
            for ch in chapters:
                # Find the segment whose timestamp is closest to (but >= ) chapter start
                chapter_by_hms[ch["title"]] = ch["start"]

        # Format as markdown, inserting chapter headings at the right positions
        md_lines = list(toc_lines)
        chapter_idx = 0
        for seg in segments:
            # Insert chapter heading before this segment if its time matches
            seg_secs = self._hms_to_secs(seg["hms"]) if seg["hms"] else 0
            while chapter_idx < len(chapters) and chapters[chapter_idx]["start"] <= seg_secs:
                md_lines.append(f"## {chapters[chapter_idx]['title']}")
                md_lines.append("")
                chapter_idx += 1

            speaker = seg["speaker"]
            prefix = f"**{speaker}:** " if speaker else ""
            suffix = f" [{seg['hms']}]" if seg["hms"] else ""
            md_lines.append(f"{prefix}{seg['text']}{suffix}")
            md_lines.append("")

        self.logger.info(f"✅ Parsed {len(segments)} segments, {len(chapters)} chapters from Rescript API")
        return "\n".join(md_lines)

    @staticmethod
    def _hms_to_secs(hms: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = hms.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return float(parts[0])

    @staticmethod
    def _fetch_with_jina_reader(url: str, timeout: int = 15) -> Optional[str]:
        """Fetch page content via Jina Reader API (r.jina.ai).

        Jina Reader renders JS-heavy pages server-side and returns clean markdown.
        Useful as a fallback when both urllib and headless Chrome fail.
        """
        import urllib.request

        logger = logging.getLogger(__name__)
        jina_url = f"https://r.jina.ai/{url}"
        try:
            req = urllib.request.Request(
                jina_url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8")
            if content and len(content) > 500:
                # Strip Jina metadata header (Title:, URL Source:, etc.)
                lines = content.split("\n")
                body_start = 0
                for i, line in enumerate(lines):
                    if line.startswith("Markdown Content:"):
                        body_start = i + 1
                        break
                if body_start:
                    content = "\n".join(lines[body_start:]).strip()
                # Normalize Substack transcript format to standard markdown
                content = YouTubeDownloader._normalize_substack_transcript(content)
                logger.info(f"✅ Jina Reader returned {len(content)} chars")
                return content
            logger.warning(f"Jina Reader returned insufficient content ({len(content) if content else 0} chars)")
            return None
        except Exception as e:
            logger.warning(f"Jina Reader failed: {e}")
            return None

    @staticmethod
    def _normalize_substack_transcript(content: str) -> str:
        """Normalize Substack/Dwarkesh transcript to standard markdown format.

        Converts:
          - Chapter links: [(HH:MM:SS) - Title](url) → ## Title
          - Standalone bold speaker: **Name**\\n\\ntext → **Name:** text [HH:MM:SS]
          - Strips markdown links: [text](url) → text

        The output matches the format expected by lattifai-captions MarkdownReader:
            **Speaker:** text [HH:MM:SS]
        """
        lines = content.split("\n")
        output = []
        # Collect chapters for TOC and timestamp assignment
        chapters = []  # [(seconds, title)]
        chapter_pattern = re.compile(r"^\[\((\d{1,2}:\d{2}:\d{2})\)\s*[-–—]\s*(.+?)\]\(https?://[^)]+\)\s*$")
        standalone_speaker = re.compile(r"^\*\*([^*:：]+)\*\*\s*$")
        md_link = re.compile(r"\[([^\]]+)\]\([^)]+\)")

        # First pass: detect if this is a Substack transcript
        has_standalone_speakers = False
        has_chapter_links = False
        for line in lines:
            if standalone_speaker.match(line):
                has_standalone_speakers = True
            if chapter_pattern.match(line):
                has_chapter_links = True
            if has_standalone_speakers and has_chapter_links:
                break

        if not has_standalone_speakers:
            return content  # Not a Substack transcript, return as-is

        # Collect all chapters
        for line in lines:
            ch_m = chapter_pattern.match(line)
            if ch_m:
                hms = ch_m.group(1)
                title = ch_m.group(2).strip()
                parts = hms.split(":")
                secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                chapters.append((secs, title, hms))

        # Build TOC
        if chapters:
            output.append("## Table of Contents")
            for secs, title, hms in chapters:
                output.append(f"- {hms} – {title}")
            output.append("")

        # Second pass: convert speaker blocks
        current_speaker = None
        current_text = []
        in_preamble = True  # Skip sponsor/intro content before first speaker

        def _flush_segment():
            nonlocal current_text
            if current_speaker and current_text:
                text = " ".join(current_text)
                # Strip markdown links
                text = md_link.sub(r"\1", text)
                output.append(f"**{current_speaker}:** {text}")
                output.append("")
            current_text = []

        for line in lines:
            # Skip chapter link lines (already in TOC)
            if chapter_pattern.match(line):
                continue

            sp_m = standalone_speaker.match(line)
            if sp_m:
                # Flush previous segment
                _flush_segment()
                current_speaker = sp_m.group(1).strip()
                in_preamble = False

                # Insert chapter heading if this speaker's position matches
                # (use chapter_idx to track progress through chapters)
                continue

            if in_preamble:
                continue

            stripped = line.strip()
            if stripped:
                current_text.append(stripped)

        # Flush last segment
        _flush_segment()

        return "\n".join(output)

    @staticmethod
    def _find_chrome() -> Optional[str]:
        """Find Chrome/Chromium executable on the system."""
        import shutil

        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ]
        # Allow override via environment variable
        env_path = os.environ.get("CHROME_PATH") or os.environ.get("URL_CHROME_PATH")
        if env_path:
            candidates.insert(0, env_path)

        for candidate in candidates:
            if os.path.isfile(candidate) or shutil.which(candidate):
                return candidate
        return None

    @staticmethod
    def _fetch_with_headless_chrome(url: str, timeout: int = 45) -> Optional[str]:
        """Fetch rendered page content using headless Chrome via CDP.

        Zero-dependency fallback for SPA/CSR sites. Uses a lightweight TypeScript
        script (fetch_rendered_html.ts) that launches system Chrome in headless mode,
        waits for JS execution and network idle, then returns the rendered DOM.

        Requires: system Chrome + bun (``npx -y bun``).
        """
        import shutil
        import subprocess

        logger = logging.getLogger(__name__)

        # Locate the CDP fetch script (shipped alongside this module)
        script = Path(__file__).parent / "fetch_rendered_html.ts"
        if not script.exists():
            logger.warning(f"CDP fetch script not found at {script}")
            return None

        # Check bun availability
        if not shutil.which("bun") and not shutil.which("npx"):
            logger.warning("Neither bun nor npx found. Cannot run headless Chrome CDP script.")
            return None

        runner = ["bun"] if shutil.which("bun") else ["npx", "-y", "bun"]

        try:
            result = subprocess.run(
                [*runner, str(script), url, str(timeout * 1000)],
                capture_output=True,
                text=True,
                timeout=timeout + 10,  # subprocess timeout slightly longer than script timeout
            )
            html = result.stdout
            if result.stderr:
                logger.debug(f"CDP fetch: {result.stderr.strip()}")
            if html and len(html) > 500:
                return html
            logger.warning(f"CDP fetch returned insufficient content ({len(html) if html else 0} bytes)")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Headless Chrome CDP timed out after {timeout}s for {url}")
            return None
        except Exception as e:
            logger.warning(f"Headless Chrome CDP failed: {e}")
            return None

    @staticmethod
    def _parse_transcript_html(html: str, youtube_url: Optional[str] = None) -> Optional[str]:
        """Parse transcript HTML into markdown transcript format.

        Output is compatible with lattifai-captions MarkdownReader:

            **Speaker Name:** Transcript text... [HH:MM:SS]

        Also supports the legacy podcast-transcript format:

            Speaker Name
            [(HH:MM:SS)](youtube_url&t=N)
            Transcript text...
        """
        try:
            from html.parser import HTMLParser

            class TranscriptParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.lines = []
                    self.current_text = []
                    self.in_body = False
                    self.skip_depth = 0

                def handle_starttag(self, tag, attrs):
                    if tag == "body":
                        self.in_body = True
                    elif tag in ("script", "style", "nav", "header", "footer", "noscript"):
                        self.skip_depth += 1
                    elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                        self._flush()

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "header", "footer", "noscript"):
                        self.skip_depth = max(0, self.skip_depth - 1)
                    elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                        self._flush()

                def handle_data(self, data):
                    if self.in_body and self.skip_depth == 0:
                        text = data.strip()
                        if text:
                            self.current_text.append(text)

                def _flush(self):
                    if self.current_text:
                        line = " ".join(self.current_text).strip()
                        if line:
                            self.lines.append(line)
                        self.current_text = []

                def get_text(self):
                    self._flush()
                    return "\n".join(self.lines)

            # If input is not HTML (e.g. markdown from url-to-markdown), use as-is
            if "<body" not in html.lower() and "<html" not in html.lower():
                text = html
            else:
                parser = TranscriptParser()
                parser.feed(html)
                text = parser.get_text()
            lines = text.split("\n")

            # Patterns that indicate post-transcript UI noise (footer, social, nav).
            # Must be precise to avoid matching spoken dialogue (e.g. "Subscribe to our YouTube...").
            _ui_noise_re = re.compile(
                r"^(\d+\s*Likes?\s*(∙|·)?\s*\d*\s*Restacks?"  # "36 Likes ∙ 7 Restacks"
                r"|Discussion about this"
                r"|Comments?\s*Restacks?"  # "Comments Restacks" nav
                r"|Ready for more\?"
                r"|Share$|Reply$|Like$|Subscribe$"  # standalone button labels
                r"|©\s*\d{4}|Privacy\s*∙\s*Terms|Start your Substack|Get the app$"
                r"|Show Topics$|See all$|Sign in$"
                r"|Click on any sentence in the transcript"
                r"|\d{1,2}月\d{1,2}日$"  # Chinese date (comment timestamps)
                r"|OK: \d+ bytes)"  # CDP stderr leak
            )

            # Patterns to truncate from the end of the last segment's text content
            _inline_noise_re = re.compile(
                r"\s*(?:"
                r"There aren't comments yet.*"
                r"|Click on any sentence in the transcript.*"
                r"|DO NOT SELL OR SHARE MY PERSONAL INFORMATION.*"
                r"|What is this\?.*Report Ad.*"
                r"|OK: \d+ bytes.*"
                r")$"
            )

            def _clean_trailing_noise(md_text: str) -> str:
                """Remove UI noise lines and inline noise from parsed transcript."""
                out_lines = md_text.rstrip().split("\n")
                # Remove trailing noise lines
                while out_lines:
                    last = out_lines[-1].strip()
                    if not last or _ui_noise_re.match(last):
                        out_lines.pop()
                    else:
                        break
                # Also truncate inline noise from the last text line
                if out_lines:
                    out_lines[-1] = _inline_noise_re.sub("", out_lines[-1])
                return "\n".join(out_lines) + "\n"

            def _hms_to_secs(hms: str) -> int:
                parts = hms.split(":")
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return int(parts[0])

            def _format_segments(segments: list) -> str:
                """Format segments into markdown transcript format: **Speaker:** text [HH:MM:SS]"""
                md_lines = []
                for seg in segments:
                    # Chapter heading marker
                    if "_chapter" in seg:
                        md_lines.append(f"## {seg['_chapter']}")
                        md_lines.append("")
                        continue
                    speaker = seg.get("speaker") or ""
                    text = seg.get("text", "").strip()
                    hms = seg.get("hms")
                    # Skip empty speaker or "Unknown"
                    prefix = f"**{speaker}:** " if speaker and speaker != "Unknown" else ""
                    suffix = f" [{hms}]" if hms else ""
                    md_lines.append(f"{prefix}{text}{suffix}")
                    md_lines.append("")
                return _clean_trailing_noise("\n".join(md_lines))

            # Strategy 1: timestamped lines with speaker names
            # Supports: "Speaker (HH:MM:SS) text"       — lexfridman.com
            #           "Speaker [HH:MM:SS]: text"      — latent.space / Substack
            # Convert to markdown format:
            #   **Speaker Name:** text [HH:MM:SS]
            ts_pattern = re.compile(r"^(.+?)\s+[\(\[](\d{1,2}:\d{2}:\d{2})[\)\]]:?\s*(.*)")
            # Chapter line: "0:00 – Topic" or "1:30:00 - Topic"
            chapter_pattern = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[–—-]\s*(.+)")
            ts_segments = []
            in_transcript = False
            current_seg = None

            # Collect pre-transcript content: intro text and table of contents
            preamble_lines = []
            chapters = []
            in_toc = False
            # UI noise lines to skip entirely
            _preamble_skip_re = re.compile(
                r"^(Skip to|Go back to|Watch the full|Useful links"
                r"|Please note that the transcript"
                r"|Here are some useful links)"
            )

            # Set of chapter titles collected from TOC (populated in first pass below)
            chapter_titles = set()

            for line in lines:
                m = ts_pattern.match(line)
                if m:
                    in_transcript = True
                    in_toc = False
                    if current_seg:
                        ts_segments.append(current_seg)
                    current_seg = {"speaker": m.group(1), "hms": m.group(2), "text": m.group(3).strip()}
                elif in_transcript and current_seg:
                    # Skip navigation / UI noise lines
                    if re.match(r"^(Skip to|Go back|Watch the|Useful links|Table of Contents)", line):
                        continue
                    # Stop appending when we hit post-transcript UI elements
                    if re.match(
                        r"^(\d+\s*Likes?|Discussion|Comments|Restacks?|Subscribe|Ready for"
                        r"|Share|Reply|Like$|©|Privacy|Terms|Start your|Substack"
                        r"|Show Topics|See all|\d{1,2}月\d{1,2}日)",
                        line,
                    ):
                        in_transcript = False
                        continue
                    # Detect chapter heading lines (match TOC titles or standalone short lines
                    # that don't look like transcript text)
                    stripped = line.strip()
                    if stripped and stripped in chapter_titles:
                        ts_segments.append(current_seg)
                        ts_segments.append({"_chapter": stripped})
                        current_seg = None
                        continue
                    if current_seg and len(line) > 10:
                        current_seg["text"] += " " + line.strip()
                elif not in_transcript:
                    # Pre-transcript: collect intro and chapters
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.lower().startswith("table of contents") or stripped.lower().startswith("here are the"):
                        in_toc = True
                        continue
                    if in_toc:
                        ch_m = chapter_pattern.match(stripped)
                        if ch_m:
                            chapter_title = ch_m.group(2).strip()
                            chapters.append(f"- {ch_m.group(1)} – {chapter_title}")
                            chapter_titles.add(chapter_title)
                            continue
                        # Exit TOC only after we've seen at least one chapter entry
                        if chapters:
                            in_toc = False
                        else:
                            # Still in TOC preamble (e.g. "Here are the loose chapters...")
                            continue
                    if _preamble_skip_re.match(stripped):
                        continue
                    # Chapter title appearing before first transcript segment
                    if stripped in chapter_titles:
                        ts_segments.append({"_chapter": stripped})
                        continue
                    # Keep meaningful intro lines (description, episode info)
                    if len(stripped) > 20:
                        preamble_lines.append(stripped)

            if current_seg:
                ts_segments.append(current_seg)

            if ts_segments:
                header_parts = []
                if preamble_lines:
                    header_parts.append("\n".join(preamble_lines))
                if chapters:
                    header_parts.append("## Table of Contents\n" + "\n".join(chapters))
                header = "\n\n".join(header_parts) + "\n\n" if header_parts else ""
                return header + _format_segments(ts_segments)

            # Split text into blocks for block-based strategies
            blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]

            # Strategy 2a: Dwarkesh/Substack article transcript (most specific — check first)
            # Looks for "## Transcript" section with "**Speaker**" labels and
            # "### HH:MM:SS - Topic" chapter headers.
            # Within each chapter, interpolates timestamps proportionally by text length.
            if any(b == "## Transcript" or b.startswith("## Transcript\n") for b in blocks):
                tx_start = next(
                    (i for i, b in enumerate(blocks) if b == "## Transcript" or b.startswith("## Transcript\n")),
                    None,
                )
                if tx_start is not None:
                    # Match "**Speaker**" with optional trailing timestamp like "**Speaker** 1:20:33"
                    bold_speaker = re.compile(r"^\*\*([^*]+)\*\*(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
                    chapter_ts = re.compile(r"^###?\s+(\d{1,2}:\d{2}:\d{2})\s*[-–—]?\s*(.*)")
                    art_segments = []
                    current_speaker = "Unknown"
                    current_hms = "0:00:00"

                    for bi2 in range(tx_start + 1, len(blocks)):
                        b = blocks[bi2]
                        ch_m = chapter_ts.match(b)
                        if ch_m:
                            current_hms = ch_m.group(1)
                            continue
                        sp_m = bold_speaker.match(b)
                        if sp_m:
                            current_speaker = sp_m.group(1)
                            continue
                        if len(b) < 15:
                            continue
                        if b.startswith("[") and b.endswith(")") and len(b) < 200:
                            continue
                        # Stop at post-transcript comment/discussion sections
                        if re.match(
                            r"^(####?\s+Discussion|CommentsRestacks"
                            r"|\d+\s*Likes?\s*(∙|·|\[)"
                            r"|Like\s*(\(\d+\))?\s*Reply)",
                            b,
                        ):
                            break
                        # Skip markdown headers that aren't chapter timestamps
                        if b.startswith("#") and not chapter_ts.match(b):
                            continue
                        # Stop at post-transcript UI elements
                        if _ui_noise_re.match(b):
                            break
                        art_segments.append({"speaker": current_speaker, "hms": current_hms, "text": b})

                    if len(art_segments) >= 5:
                        # Interpolate timestamps within each chapter based on text length.
                        # Group consecutive segments by chapter timestamp, then distribute
                        # time proportionally so each segment gets a unique start time.
                        chapter_boundaries = []
                        for idx, seg in enumerate(art_segments):
                            if idx == 0 or seg["hms"] != art_segments[idx - 1]["hms"]:
                                chapter_boundaries.append(idx)

                        # Estimate chars-per-second from known chapters for last chapter extrapolation
                        _total_known_chars = 0
                        _total_known_secs = 0
                        if len(chapter_boundaries) >= 2:
                            first_secs = _hms_to_secs(art_segments[chapter_boundaries[0]]["hms"])
                            last_known_secs = _hms_to_secs(art_segments[chapter_boundaries[-1]]["hms"])
                            _total_known_secs = last_known_secs - first_secs
                            for ci2 in range(len(chapter_boundaries) - 1):
                                s2 = chapter_boundaries[ci2]
                                e2 = chapter_boundaries[ci2 + 1]
                                _total_known_chars += sum(len(seg["text"]) for seg in art_segments[s2:e2])

                        for ci, ch_start_idx in enumerate(chapter_boundaries):
                            ch_end_idx = (
                                chapter_boundaries[ci + 1] if ci + 1 < len(chapter_boundaries) else len(art_segments)
                            )
                            ch_start_secs = _hms_to_secs(art_segments[ch_start_idx]["hms"])
                            if ci + 1 < len(chapter_boundaries):
                                ch_end_secs = _hms_to_secs(art_segments[chapter_boundaries[ci + 1]]["hms"])
                            else:
                                # Last chapter: estimate duration from chars/sec ratio of known chapters
                                last_ch_chars = sum(len(seg["text"]) for seg in art_segments[ch_start_idx:])
                                if _total_known_chars > 0 and _total_known_secs > 0:
                                    cps = _total_known_chars / _total_known_secs
                                    ch_end_secs = ch_start_secs + last_ch_chars / cps
                                else:
                                    ch_end_secs = ch_start_secs + 600
                            ch_duration = ch_end_secs - ch_start_secs
                            ch_segs = art_segments[ch_start_idx:ch_end_idx]
                            total_chars = sum(len(s["text"]) for s in ch_segs)
                            if total_chars == 0:
                                total_chars = 1
                            elapsed = 0.0
                            for seg in ch_segs:
                                seg_secs = ch_start_secs + elapsed
                                h = int(seg_secs // 3600)
                                m = int((seg_secs % 3600) // 60)
                                s = int(seg_secs % 60)
                                seg["hms"] = f"{h:02d}:{m:02d}:{s:02d}"
                                elapsed += (len(seg["text"]) / total_chars) * ch_duration

                        return _format_segments(art_segments)

            # Strategy 2b: block-based format (Rescript, podcast apps)
            # Supports two block orderings:
            #   A) Speaker → HH:MM:SS → text  (Rescript)
            #   B) M:SS → SPEAKER N → text    (Dwarkesh audio player preview)
            block_segments = []
            bi = 0
            while bi < len(blocks) - 2:
                b0, b1, b2 = blocks[bi], blocks[bi + 1], blocks[bi + 2]
                # Order A: [speaker, timestamp, text]
                ts_a = re.match(r"^(\d{1,2}:\d{2}:\d{2})$", b1)
                if (
                    ts_a
                    and len(b0) < 80
                    and not b0.startswith(("#", "[", "!", "http", "---"))
                    and not re.match(r"^\d", b0)
                ):
                    block_segments.append({"speaker": b0, "hms": ts_a.group(1), "text": b2})
                    bi += 3
                    continue
                # Order B: Dwarkesh/Substack — timestamps + optional speaker labels
                # Format: "M:SS\n\n[SPEAKER N]\n\ntext" or "M:SS\n\ntext" (same speaker)
                ts_b = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)$", b0)
                if ts_b:
                    hms = ts_b.group(1)
                    if hms.count(":") == 1:
                        hms = "0:" + hms  # M:SS → 0:MM:SS
                    # Check if b1 is a speaker label
                    if re.match(r"^(SPEAKER\s*\d+)$", b1) and bi + 2 < len(blocks):
                        block_segments.append({"speaker": b1, "hms": hms, "text": b2})
                        bi += 3
                        continue
                    # No speaker label → text directly after timestamp, reuse last speaker
                    elif len(b1) > 20 and not re.match(r"^\d{1,2}:\d{2}", b1):
                        last_speaker = block_segments[-1]["speaker"] if block_segments else "Unknown"
                        block_segments.append({"speaker": last_speaker, "hms": hms, "text": b1})
                        bi += 2
                        continue
                bi += 1

            if len(block_segments) >= 3:
                return _format_segments(block_segments)

            # Strategy 3: dialogue lines "Speaker Name: text" (Substack/Dwarkesh style)
            dialogue_pattern = re.compile(
                r"^([A-Z][a-zA-Z\u00C0-\u024F'.\-]+(?: [A-Z][a-zA-Z\u00C0-\u024F'.\-]+){0,3}):\s+(.+)"
            )
            dialogue_segments = []
            for line in lines:
                m = dialogue_pattern.match(line)
                if m:
                    dialogue_segments.append({"speaker": m.group(1), "text": m.group(2).strip()})

            # Require ≥10 segments with ≥2 distinct speakers to avoid false positives
            # from navigation text like "Site Name: Subtitle"
            dialogue_speakers = {s["speaker"] for s in dialogue_segments}
            if len(dialogue_segments) >= 10 and len(dialogue_speakers) >= 2:
                return _format_segments(dialogue_segments)

            # Strategy 4: "Starting point is HH:MM:SS" blocks (podscripts.co)
            # In markdown: timestamp and text on separate lines
            # In HTML:     timestamp and text may be on the same line
            sp_pattern = re.compile(r"^Starting point is (\d{1,2}:\d{2}:\d{2})\s*(.*)")
            sp_segments = []
            current_sp = None
            for line in lines:
                m = sp_pattern.match(line)
                if m:
                    if current_sp:
                        sp_segments.append(current_sp)
                    inline_text = m.group(2).strip()
                    current_sp = {"hms": m.group(1), "lines": [inline_text] if inline_text else []}
                elif current_sp and line.strip():
                    current_sp["lines"].append(line.strip())
            if current_sp:
                sp_segments.append(current_sp)

            if len(sp_segments) >= 3:
                formatted = []
                for seg in sp_segments:
                    seg_text = " ".join(seg["lines"])
                    if not seg_text:
                        continue
                    formatted.append({"speaker": "Unknown", "hms": seg["hms"], "text": seg_text})
                return _format_segments(formatted)

            # Fallback: return all body text if no structured transcript found
            return text if len(text) > 200 else None

        except Exception:
            return None

    async def download_captions(
        self,
        url: str,
        output_dir: str,
        force_overwrite: bool = False,
        source_lang: Optional[str] = None,
        transcriber_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download video captions using yt-dlp

        Args:
            url: YouTube URL
            output_dir: Output directory
            force_overwrite: Skip user confirmation and overwrite existing files
            source_lang: Specific caption language/track to download (e.g., 'en')
                          If None, downloads all available captions
            transcriber_name: Name of the transcriber (for user prompts)
        Returns:
            Path to downloaded transcript file or None if not available
        """
        target_dir = Path(output_dir).expanduser()

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID
        video_id = self.extract_video_id(url)

        # --- Phase 1: External transcript (high-quality, with speaker labels) ---
        # Always attempt; result is stored for later return but does NOT skip YT download.
        external_transcript_path = None
        try:
            transcript_md = target_dir / f"{video_id}.transcript.md"
            if transcript_md.exists() and not force_overwrite:
                self.logger.info(f"✅ Found existing external transcript: {transcript_md}")
                external_transcript_path = str(transcript_md)
            else:
                info = await self.get_video_info(url)
                description = info.get("description", "")
                transcript_url = self._extract_transcript_url_from_description(description)
                if transcript_url:
                    self.logger.info(f"🔗 Found transcript URL in description: {transcript_url}")
                    ext_path = await self._download_external_transcript(
                        transcript_url, output_dir, video_id, youtube_url=url, video_info=info
                    )
                    if ext_path:
                        external_transcript_path = ext_path
        except Exception as e:
            self.logger.debug(f"External transcript check skipped: {e}")

        # --- Phase 2: YouTube captions (accurate timestamps) ---
        # Always download YT captions so both sources are available on disk.
        yt_caption_exists = False
        if not force_overwrite:
            existing_files = FileExistenceManager.check_existing_files(
                video_id, str(target_dir), caption_formats=CAPTION_FORMATS
            )
            if existing_files["caption"]:
                yt_caption_exists = True
                self.logger.info(f"🔍 Found existing YT caption: {existing_files['caption'][0]}")

        # If external transcript exists and YT captions already on disk, return transcript
        if external_transcript_path and yt_caption_exists:
            self.logger.info(f"✅ Both sources available. Using external transcript: {external_transcript_path}")
            return external_transcript_path

        # If YT captions exist but no external transcript, use YT captions
        if yt_caption_exists and not external_transcript_path:
            if not force_overwrite:
                if FileExistenceManager.is_interactive_mode():
                    user_choice = FileExistenceManager.prompt_user_confirmation(
                        {"caption": existing_files["caption"]}, "caption download", transcriber_name=transcriber_name
                    )

                    if user_choice == "cancel":
                        raise RuntimeError("Caption download cancelled by user")
                    elif user_choice == "overwrite":
                        pass  # Continue with download below
                    elif user_choice == TRANSCRIBE_CHOICE:
                        return TRANSCRIBE_CHOICE
                    elif user_choice in existing_files["caption"]:
                        caption_file = Path(user_choice)
                        self.logger.info(f"✅ Using selected caption file: {caption_file}")
                        return str(caption_file)
                    else:
                        caption_file = Path(existing_files["caption"][0])
                        self.logger.info(f"✅ Using existing caption file: {caption_file}")
                        return str(caption_file)
                else:
                    caption_file = Path(existing_files["caption"][0])
                    return str(caption_file)

        self.logger.info(f"📥 Downloading caption for: {url}")
        if source_lang:
            self.logger.info(f"🎯 Targeting specific caption track: {source_lang}")

        # Auto-detect original language from video info when source_lang is not specified
        if not source_lang:
            try:
                info = await self.get_video_info(url)
                # Prefer manually created subtitles keys as the most reliable signal
                manual_subs = info.get("subtitles", {})
                video_lang = info.get("language")

                if manual_subs:
                    # Use all manually created subtitle languages
                    sub_langs = list(manual_subs.keys())
                    source_lang = ",".join(sub_langs)
                    self.logger.info(f"🌐 Auto-detected manual caption languages: {sub_langs}")
                elif video_lang:
                    source_lang = video_lang
                    self.logger.info(f"🌐 Auto-detected video language: {source_lang}")
            except Exception as e:
                self.logger.debug(f"Language auto-detection skipped: {e}")

        output_template = str(target_dir / f"{video_id}.%(ext)s")

        # Configure yt-dlp options for caption download
        opts = {
            "skip_download": True,  # Don't download video/audio
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "best",
            "outtmpl": output_template,
            "quiet": False,
            "no_warnings": True,
        }

        # Add caption language selection if specified (or auto-detected)
        if source_lang:
            opts["subtitleslangs"] = [f"{lang}*" for lang in source_lang.split(",")]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _download_subs():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])

            await loop.run_in_executor(None, _download_subs)

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)

            # Check for specific error conditions
            if "No automatic or manual captions found" in error_msg:
                self.logger.warning("No captions available for this video")
            elif "HTTP Error 429" in error_msg or "Too Many Requests" in error_msg:
                self.logger.error("YouTube rate limit exceeded. Please try again later or use a different method.")
                self.logger.error(
                    "YouTube rate limit exceeded (HTTP 429). "
                    "Try again later or use --cookies option with authenticated cookies. "
                    "See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
                )
            else:
                self.logger.error(f"Failed to download transcript: {error_msg}")
        except Exception as e:
            self.logger.error(f"Failed to download transcript: {str(e)}")

        # Find the downloaded YT caption files
        caption_patterns = [
            f"{video_id}.*vtt",
            f"{video_id}.*srt",
            f"{video_id}.*sub",
            f"{video_id}.*sbv",
            f"{video_id}.*ssa",
            f"{video_id}.*ass",
        ]

        caption_files = []
        for pattern in caption_patterns:
            _caption_files = list(target_dir.glob(pattern))
            for caption_file in _caption_files:
                self.logger.info(f"📥 YT caption on disk: {caption_file}")
            caption_files.extend(_caption_files)

        # --- Phase 3: Choose best source ---
        # External transcript (speaker labels + metadata) is preferred over YT captions
        if external_transcript_path:
            if caption_files:
                self.logger.info(f"✅ Both sources available: external transcript + {len(caption_files)} YT caption(s)")
            self.logger.info(f"✅ Using external transcript: {external_transcript_path}")
            return external_transcript_path

        # No external transcript — fall back to YT captions
        if len(caption_files) == 1:
            self.logger.info(f"✅ Using YT caption: {caption_files[0]}")
            return str(caption_files[0])

        if caption_files and FileExistenceManager.is_interactive_mode():
            self.logger.info(f"📋 Found {len(caption_files)} YT caption files")
            caption_choice = FileExistenceManager.prompt_file_selection(
                file_type="caption",
                files=[str(f) for f in caption_files],
                operation="use",
                transcriber_name=transcriber_name,
            )

            if caption_choice == "cancel":
                raise RuntimeError("Caption selection cancelled by user")
            elif caption_choice == TRANSCRIBE_CHOICE:
                return caption_choice
            elif caption_choice:
                self.logger.info(f"✅ Selected caption: {caption_choice}")
                return caption_choice
            else:
                self.logger.info(f"✅ Using first caption: {caption_files[0]}")
                return str(caption_files[0])
        elif caption_files:
            self.logger.info(f"✅ Using first YT caption: {caption_files[0]}")
            return str(caption_files[0])
        else:
            self.logger.warning("No caption files available after download")
            return None

    async def list_available_captions(self, url: str) -> List[Dict[str, Any]]:
        """
        List all available caption tracks for a YouTube video

        Args:
            url: YouTube URL

        Returns:
            List of caption track information dictionaries
        """
        self.logger.info(f"📋 Listing available captions for: {url}")

        opts = {
            "skip_download": True,
            "listsubtitles": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _get_info():
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(url, download=False)

            info = await loop.run_in_executor(None, _get_info)

            caption_info = []

            # Parse manual captions
            subtitles = info.get("subtitles", {})
            for lang, formats in subtitles.items():
                if formats:
                    format_names = [f.get("ext", "") for f in formats]
                    lang_name = formats[0].get("name", lang) if formats else lang
                    caption_info.append(
                        {"language": lang, "name": lang_name, "formats": format_names, "kind": "manual"}
                    )

            # Parse automatic captions
            auto_subs = info.get("automatic_captions", {})
            for lang, formats in auto_subs.items():
                if formats:
                    format_names = [f.get("ext", "") for f in formats]
                    lang_name = formats[0].get("name", lang) if formats else lang
                    caption_info.append({"language": lang, "name": lang_name, "formats": format_names, "kind": "asr"})

            self.logger.info(f"✅ Found {len(caption_info)} caption tracks")
            return caption_info

        except yt_dlp.utils.DownloadError as e:
            self.logger.error(f"Failed to list captions: {str(e)}")
            raise RuntimeError(f"Failed to list captions: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to list captions: {str(e)}")
            raise RuntimeError(f"Failed to list captions: {str(e)}")
