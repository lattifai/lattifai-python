"""
YouTube downloader module using yt-dlp and Agent
"""

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.caption import CAPTION_FORMATS
from .base import setup_workflow_logger
from .file_manager import TRANSCRIBE_CHOICE, FileExistenceManager


class YouTubeDownloader:
    """YouTube video/audio downloader using yt-dlp

    Configuration (in __init__):
        - None (stateless downloader)

    Runtime parameters (in __call__ or methods):
        - url: YouTube URL to download
        - output_dir: Where to save files
        - media_format: Format to download (mp3, mp4, etc.)
        - force_overwrite: Whether to overwrite existing files
    """

    def __init__(self):
        self.logger = setup_workflow_logger("youtube")
        # Check if yt-dlp is available
        self._check_ytdlp()

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

    def _check_ytdlp(self):
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, check=True)
            self.logger.info(f"yt-dlp version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "yt-dlp is not installed or not found in PATH. Please install it with: pip install yt-dlp"
            )

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video metadata without downloading"""
        self.logger.info(f"🔍 Extracting video info for: {url}")

        cmd = ["yt-dlp", "--dump-json", "--no-download", url]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            import json

            metadata = json.loads(result.stdout)

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

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to extract video info: {e.stderr}")
            raise RuntimeError(f"Failed to extract video info: {e.stderr}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse video metadata: {e}")
            raise RuntimeError(f"Failed to parse video metadata: {e}")

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
                         or video (mp4, webm, mkv, avi, mov, etc.) (default: instance format)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded media file
        """
        media_format = media_format or self.media_format

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
                    # self.logger.info(f"✅ Using selected media file: {user_choice}")
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

        # Build yt-dlp command based on media type
        if is_audio:
            cmd = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format",
                media_format,
                "--audio-quality",
                "0",  # Best quality
                "--output",
                output_template,
                "--no-playlist",
                url,
            ]
        else:
            cmd = [
                "yt-dlp",
                "--format",
                "bestvideo*+bestaudio/best",
                "--merge-output-format",
                media_format,
                "--output",
                output_template,
                "--no-playlist",
                url,
            ]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            self.logger.info(f"✅ {media_type.capitalize()} download completed")

            # Find the downloaded file
            # Try to parse from yt-dlp output first
            if is_audio:
                output_lines = result.stderr.strip().split("\n")
                for line in reversed(output_lines):
                    if "Destination:" in line or "has already been downloaded" in line:
                        parts = line.split()
                        filename = " ".join(parts[1:]) if "Destination:" in line else parts[0]
                        file_path = target_dir / filename
                        if file_path.exists():
                            self.logger.info(f"{emoji} Downloaded {media_type} file: {file_path}")
                            return str(file_path)

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

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download {media_type}: {e.stderr}")
            raise RuntimeError(f"Failed to download {media_type}: {e.stderr}")

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
            media_format: Audio format (default: instance format)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded audio file
        """
        target_dir = output_dir or tempfile.gettempdir()
        media_format = media_format or self.media_format
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
        ytdlp_options = [
            "yt-dlp",
            "--skip-download",  # Don't download video/audio
            "--output",
            output_template,
            "--sub-format",
            "best",  # Prefer best available format
        ]

        # Add caption language selection if specified
        if source_lang:
            ytdlp_options.extend(["--write-sub", "--write-auto-sub", "--sub-langs", f"{source_lang}*"])
        else:
            # Download only manual captions (not auto-generated) in English to avoid rate limiting
            ytdlp_options.extend(["--write-sub", "--write-auto-sub"])

        ytdlp_options.append(url)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(ytdlp_options, capture_output=True, text=True, check=True)
            )

            self.logger.info(f"yt-dlp transcript output: {result.stdout.strip()}")

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

            if not caption_files:
                self.logger.warning("No caption available for this video")
                return None

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
                else:
                    # Fallback to first file
                    self.logger.info(f"✅ Using first caption: {caption_files[0]}")
                    return str(caption_files[0])
            else:
                # Non-interactive mode: use first file
                self.logger.info(f"✅ Using first caption: {caption_files[0]}")
                return str(caption_files[0])

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            if "No automatic or manual captions found" in error_msg:
                self.logger.warning("No captions available for this video")
                return None
            else:
                self.logger.error(f"Failed to download transcript: {error_msg}")
                raise RuntimeError(f"Failed to download transcript: {error_msg}")

    async def list_available_captions(self, url: str) -> List[Dict[str, Any]]:
        """
        List all available caption tracks for a YouTube video

        Args:
            url: YouTube URL

        Returns:
            List of caption track information dictionaries
        """
        self.logger.info(f"📋 Listing available captions for: {url}")

        cmd = ["yt-dlp", "--list-subs", "--no-download", url]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            # Parse the caption list output
            caption_info = []
            lines = result.stdout.strip().split("\n")

            # Look for the caption section (not automatic captions)
            in_caption_section = False
            for line in lines:
                if "Available captions for" in line:
                    in_caption_section = True
                    continue
                elif "Available automatic captions for" in line:
                    in_caption_section = False
                    continue
                elif in_caption_section and line.strip():
                    # Skip header lines
                    if "Language" in line and "Name" in line and "Formats" in line:
                        continue

                    # Parse caption information
                    # Format: "Language Name Formats" where formats are comma-separated
                    # Example: "en-uYU-mmqFLq8 English - CC1    vtt, srt, ttml, srv3, srv2, srv1, json3"

                    if line.strip() and not line.startswith("["):
                        # Split by multiple spaces to separate language, name, and formats
                        import re

                        parts = re.split(r"\s{2,}", line.strip())

                        if len(parts) >= 2:
                            # First part is language, last part is formats
                            language_and_name = parts[0]
                            formats_str = parts[-1]

                            # Split language and name - language is first word
                            lang_name_parts = language_and_name.split(" ", 1)
                            language = lang_name_parts[0]
                            name = lang_name_parts[1] if len(lang_name_parts) > 1 else ""

                            # If there are more than 2 parts, middle parts are also part of name
                            if len(parts) > 2:
                                name = " ".join([name] + parts[1:-1]).strip()

                            # Parse formats - they are comma-separated
                            formats = [f.strip() for f in formats_str.split(",") if f.strip()]

                            caption_info.append({"language": language, "name": name, "formats": formats})

            self.logger.info(f"✅ Found {len(caption_info)} caption tracks")
            return caption_info

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list captions: {e.stderr}")
            raise RuntimeError(f"Failed to list captions: {e.stderr}")
