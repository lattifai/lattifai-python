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

from ..config.caption import CAPTION_FORMATS
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

        if self.cookies:
            self._base_opts["cookiefile"] = self.cookies

        # Strategy: Prefer Android client to avoid PO Token issues on Web
        # But for captions, sometimes Web is needed.
        # We start with a robust default.
        self._base_opts["extractor_args"] = {"youtube": {"player_client": ["android", "web"]}}

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
            if "Sign in to confirm" in msg or "Private video" in msg:
                raise VideoUnavailableError(f"Video {video_id} is unavailable: {msg}")
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
        # Prefer json3, then vtt
        priority = ["json3", "vtt", "ttml", "srv3", "srv2", "srv1"]

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
                url=url, output_dir=output_dir, media_format=media_format, force_overwrite=force_overwrite
            )
        else:
            self.logger.info(f"🎬 Detected video format: {media_format}")
            return await self.download_video(
                url=url, output_dir=output_dir, video_format=media_format, force_overwrite=force_overwrite
            )

    async def _download_media_internal(
        self,
        url: str,
        output_dir: str,
        media_format: str,
        is_audio: bool,
        force_overwrite: bool = False,
    ) -> str:
        """
        Internal unified method for downloading audio or video from YouTube

        Args:
            url: YouTube URL
            output_dir: Output directory
            media_format: Media format (audio or video extension)
            is_audio: True for audio download, False for video download
            force_overwrite: Skip user confirmation and overwrite existing files

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

        # Build yt-dlp options based on media type
        if is_audio:
            opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": media_format,
                        "preferredquality": "0",  # Best quality
                    }
                ],
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": True,
            }
        else:
            opts = {
                "format": "bestvideo*+bestaudio/best",
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
    ) -> str:
        """
        Download audio from YouTube URL

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            media_format: Audio format (default: mp3)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded audio file
        """
        target_dir = output_dir or tempfile.gettempdir()
        media_format = media_format or "mp3"
        return await self._download_media_internal(
            url, target_dir, media_format, is_audio=True, force_overwrite=force_overwrite
        )

    async def download_video(
        self, url: str, output_dir: Optional[str] = None, video_format: str = "mp4", force_overwrite: bool = False
    ) -> str:
        """
        Download video from YouTube URL

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            video_format: Video format
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded video file
        """
        target_dir = output_dir or tempfile.gettempdir()
        return await self._download_media_internal(
            url, target_dir, video_format, is_audio=False, force_overwrite=force_overwrite
        )

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

        # Extract video ID and check for existing caption files
        video_id = self.extract_video_id(url)
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
