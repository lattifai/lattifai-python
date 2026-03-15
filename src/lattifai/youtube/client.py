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
        _LABEL_KEYWORDS = r"transcript|show\s*notes"

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

    async def _download_external_transcript(
        self,
        transcript_url: str,
        output_dir: str,
        video_id: str,
        youtube_url: Optional[str] = None,
    ) -> Optional[str]:
        """Download and parse a transcript from an external URL.

        Fetches the webpage, extracts text content with speaker labels and timestamps.
        Saves as {video_id}.transcript.md in podcast-transcript format compatible
        with lattifai-captions PodcastTranscriptReader.

        Uses urllib first (fast, no dependencies). If the result looks like a
        JS-rendered SPA (very little text content), falls back to headless Chrome
        ``--dump-dom`` which executes JavaScript before returning the DOM.

        Returns:
            Path to saved transcript file, or None on failure
        """
        output_path = Path(output_dir).expanduser() / f"{video_id}.transcript.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded
        if output_path.exists():
            self.logger.info(f"✅ Using existing external transcript: {output_path}")
            return str(output_path)

        self.logger.info(f"📥 Downloading external transcript from: {transcript_url}")

        try:
            loop = asyncio.get_event_loop()

            def _fetch():
                import urllib.request

                # Build opener with proxy support from environment variables
                proxy_handler = urllib.request.ProxyHandler()  # reads HTTP_PROXY/HTTPS_PROXY from env
                opener = urllib.request.build_opener(proxy_handler)
                req = urllib.request.Request(
                    transcript_url,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                )
                with opener.open(req, timeout=30) as resp:
                    return resp.read().decode("utf-8")

            html = await loop.run_in_executor(None, _fetch)

            # Parse HTML to extract transcript in podcast-transcript markdown format
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
                self.logger.warning("Failed to extract transcript content from page")
                return None

            output_path.write_text(transcript_text, encoding="utf-8")
            self.logger.info(f"✅ Saved external transcript: {output_path} ({len(transcript_text)} chars)")
            return str(output_path)

        except Exception as e:
            self.logger.warning(f"Failed to download external transcript: {e}")
            return None

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
        """Parse transcript HTML into podcast-transcript Markdown format.

        Output is compatible with lattifai-captions PodcastTranscriptReader:

            Speaker Name
            [(HH:MM:SS)](youtube_url&t=N)
            Transcript text...

        For pages without timestamps (Substack/Dwarkesh), outputs dialogue format:

            Speaker Name
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
                    elif tag in ("p", "br", "h1", "h2", "h3", "h4", "h5", "h6"):
                        self._flush()

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "header", "footer", "noscript"):
                        self.skip_depth = max(0, self.skip_depth - 1)
                    elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
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

            parser = TranscriptParser()
            parser.feed(html)
            text = parser.get_text()
            lines = text.split("\n")

            base_yt = youtube_url.split("?")[0] if youtube_url else None

            def _hms_to_secs(hms: str) -> int:
                parts = hms.split(":")
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return int(parts[0])

            # Strategy 1: timestamped lines "Speaker Name (HH:MM:SS) text"
            # Convert to podcast-transcript format:
            #   Speaker Name
            #   [(HH:MM:SS)](youtube_url&t=N)
            #   text
            ts_pattern = re.compile(r"^(.+?)\s+\((\d{1,2}:\d{2}:\d{2})\)\s*(.*)")
            ts_segments = []
            in_transcript = False
            current_seg = None

            for line in lines:
                m = ts_pattern.match(line)
                if m:
                    in_transcript = True
                    if current_seg:
                        ts_segments.append(current_seg)
                    current_seg = {"speaker": m.group(1), "hms": m.group(2), "text": m.group(3).strip()}
                elif in_transcript and current_seg:
                    if re.match(r"^(Skip to|Go back|Watch the|Useful links|Table of Contents)", line):
                        continue
                    if len(line) > 10:
                        current_seg["text"] += " " + line.strip()

            if current_seg:
                ts_segments.append(current_seg)

            if ts_segments:
                md_lines = []
                for seg in ts_segments:
                    md_lines.append(seg["speaker"])
                    secs = _hms_to_secs(seg["hms"])
                    if base_yt:
                        md_lines.append(f"[({seg['hms']})]({base_yt}?t={secs})")
                    else:
                        md_lines.append(f"[({seg['hms']})](#{secs})")
                    md_lines.append(seg["text"])
                    md_lines.append("")
                return "\n".join(md_lines)

            # Strategy 2: dialogue lines "Speaker Name: text" (Substack/Dwarkesh style)
            dialogue_pattern = re.compile(
                r"^([A-Z][a-zA-Z\u00C0-\u024F'.\-]+(?: [A-Z][a-zA-Z\u00C0-\u024F'.\-]+){0,3}):\s+(.+)"
            )
            dialogue_segments = []
            for line in lines:
                m = dialogue_pattern.match(line)
                if m:
                    dialogue_segments.append({"speaker": m.group(1), "text": m.group(2).strip()})

            if len(dialogue_segments) >= 3:
                md_lines = []
                for seg in dialogue_segments:
                    md_lines.append(seg["speaker"])
                    md_lines.append(seg["text"])
                    md_lines.append("")
                return "\n".join(md_lines)

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

        # Priority 1: Check for external transcript URL in video description
        # This provides higher-quality human-edited transcripts (e.g., Lex Fridman, Dwarkesh)
        try:
            # Quick check: if .transcript.md already exists, use it directly
            transcript_md = target_dir / f"{video_id}.transcript.md"
            if transcript_md.exists() and not force_overwrite:
                self.logger.info(f"✅ Using existing external transcript: {transcript_md}")
                return str(transcript_md)

            info = await self.get_video_info(url)
            description = info.get("description", "")
            transcript_url = self._extract_transcript_url_from_description(description)
            if transcript_url:
                self.logger.info(f"🔗 Found transcript URL in description: {transcript_url}")
                ext_path = await self._download_external_transcript(
                    transcript_url, output_dir, video_id, youtube_url=url
                )
                if ext_path:
                    return ext_path
        except Exception as e:
            self.logger.debug(f"External transcript check skipped: {e}")

        # Priority 2: Check for existing caption files (vtt, srt, etc.)
        if not force_overwrite:
            existing_files = FileExistenceManager.check_existing_files(
                video_id, str(target_dir), caption_formats=CAPTION_FORMATS
            )

            # Handle existing caption files
            if existing_files["caption"] and not force_overwrite:
                if FileExistenceManager.is_interactive_mode():
                    user_choice = FileExistenceManager.prompt_user_confirmation(
                        {"caption": existing_files["caption"]}, "caption download", transcriber_name=transcriber_name
                    )

                    if user_choice == "cancel":
                        raise RuntimeError("Caption download cancelled by user")
                    elif user_choice == "overwrite":
                        # Continue with download
                        pass
                    elif user_choice == TRANSCRIBE_CHOICE:
                        return TRANSCRIBE_CHOICE
                    elif user_choice in existing_files["caption"]:
                        # User selected a specific file
                        caption_file = Path(user_choice)
                        self.logger.info(f"✅ Using selected caption file: {caption_file}")
                        return str(caption_file)
                    else:
                        # Fallback: use first file
                        caption_file = Path(existing_files["caption"][0])
                        self.logger.info(f"✅ Using existing caption file: {caption_file}")
                        return str(caption_file)
                else:
                    caption_file = Path(existing_files["caption"][0])
                    self.logger.info(f"🔍 Found existing caption: {caption_file}")
                    return str(caption_file)

        self.logger.info(f"📥 Downloading caption for: {url}")
        if source_lang:
            self.logger.info(f"🎯 Targeting specific caption track: {source_lang}")

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

        # Add caption language selection if specified
        if source_lang:
            opts["subtitleslangs"] = [f"{source_lang}*"]

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

        # Find the downloaded transcript file
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
                self.logger.info(f"📥 Downloaded caption: {caption_file}")
            caption_files.extend(_caption_files)

        # If only one caption file, return it directly
        if len(caption_files) == 1:
            self.logger.info(f"✅ Using caption: {caption_files[0]}")
            return str(caption_files[0])

        # Multiple caption files found, let user choose
        if FileExistenceManager.is_interactive_mode():
            self.logger.info(f"📋 Found {len(caption_files)} caption files")
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
            elif caption_files:
                # Fallback to first file
                self.logger.info(f"✅ Using first caption: {caption_files[0]}")
                return str(caption_files[0])
            else:
                self.logger.warning("No caption files available after download")
                return None
        elif caption_files:
            # Non-interactive mode: use first file
            self.logger.info(f"✅ Using first caption: {caption_files[0]}")
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
