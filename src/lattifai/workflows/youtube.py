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

from ..client import AsyncLattifAI
from ..io import SUBTITLE_FORMATS, GeminiWriter, SubtitleIO
from .base import WorkflowAgent, WorkflowStep, setup_workflow_logger
from .file_manager import FileExistenceManager
from .gemini import GeminiTranscriber


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
                    self.logger.info(f"✅ Using selected media file: {user_choice}")
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

    async def download_subtitles(
        self,
        url: str,
        output_dir: str,
        force_overwrite: bool = False,
        subtitle_lang: Optional[str] = None,
        enable_gemini_option: bool = False,
    ) -> Optional[str]:
        """
        Download video subtitles using yt-dlp

        Args:
            url: YouTube URL
            output_dir: Output directory
            force_overwrite: Skip user confirmation and overwrite existing files
            subtitle_lang: Specific subtitle language/track to download (e.g., 'en')
                          If None, downloads all available subtitles
            enable_gemini_option: Whether to show Gemini transcription as an option in interactive mode

        Returns:
            Path to downloaded transcript file or None if not available
        """
        target_dir = Path(output_dir).expanduser()

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing subtitle files
        video_id = self.extract_video_id(url)
        if not force_overwrite:
            existing_files = FileExistenceManager.check_existing_files(
                video_id, str(target_dir), subtitle_formats=SUBTITLE_FORMATS
            )

            # Handle existing subtitle files
            if existing_files["subtitle"] and not force_overwrite:
                if FileExistenceManager.is_interactive_mode():
                    user_choice = FileExistenceManager.prompt_user_confirmation(
                        {"subtitle": existing_files["subtitle"]}, "subtitle download"
                    )

                    if user_choice == "cancel":
                        raise RuntimeError("Subtitle download cancelled by user")
                    elif user_choice == "overwrite":
                        # Continue with download
                        pass
                    elif user_choice in existing_files["subtitle"]:
                        # User selected a specific file
                        subtitle_file = Path(user_choice)
                        self.logger.info(f"✅ Using selected subtitle file: {subtitle_file}")
                        return str(subtitle_file)
                    else:
                        # Fallback: use first file
                        subtitle_file = Path(existing_files["subtitle"][0])
                        self.logger.info(f"✅ Using existing subtitle file: {subtitle_file}")
                        return str(subtitle_file)
                else:
                    subtitle_file = Path(existing_files["subtitle"][0])
                    self.logger.info(f"🔍 Found existing subtitle: {subtitle_file}")
                    return str(subtitle_file)

        self.logger.info(f"📥 Downloading subtitle for: {url}")
        if subtitle_lang:
            self.logger.info(f"🎯 Targeting specific subtitle track: {subtitle_lang}")

        output_template = str(target_dir / f"{video_id}.%(ext)s")

        # Configure yt-dlp options for subtitle download
        ytdlp_options = [
            "yt-dlp",
            "--skip-download",  # Don't download video/audio
            "--output",
            output_template,
            "--sub-format",
            "best",  # Prefer best available format
        ]

        # Add subtitle language selection if specified
        if subtitle_lang:
            ytdlp_options.extend(["--write-sub", "--write-auto-sub", "--sub-langs", f"{subtitle_lang}.*"])
        else:
            # Download only manual subtitles (not auto-generated) in English to avoid rate limiting
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
            subtitle_patterns = [
                f"{video_id}.*vtt",
                f"{video_id}.*srt",
                f"{video_id}.*sub",
                f"{video_id}.*sbv",
                f"{video_id}.*ssa",
                f"{video_id}.*ass",
            ]

            subtitle_files = []
            for pattern in subtitle_patterns:
                _subtitle_files = list(target_dir.glob(pattern))
                for subtitle_file in _subtitle_files:
                    self.logger.info(f"📥 Downloaded subtitle: {subtitle_file}")
                subtitle_files.extend(_subtitle_files)

            if not subtitle_files:
                self.logger.warning("No subtitle available for this video")
                return None

            # If only one subtitle file, return it directly
            if len(subtitle_files) == 1:
                self.logger.info(f"✅ Using subtitle: {subtitle_files[0]}")
                return str(subtitle_files[0])

            # Multiple subtitle files found, let user choose
            if FileExistenceManager.is_interactive_mode():
                self.logger.info(f"📋 Found {len(subtitle_files)} subtitle files")
                # Use the enable_gemini_option parameter passed by caller
                subtitle_choice = FileExistenceManager.prompt_file_selection(
                    file_type="subtitle",
                    files=[str(f) for f in subtitle_files],
                    operation="use",
                    enable_gemini=enable_gemini_option,
                )

                if subtitle_choice == "cancel":
                    raise RuntimeError("Subtitle selection cancelled by user")
                elif subtitle_choice == "gemini":
                    # User chose to transcribe with Gemini instead of using downloaded subtitles
                    self.logger.info("✨ User selected Gemini transcription")
                    return "gemini"  # Return special value to indicate Gemini transcription
                elif subtitle_choice:
                    self.logger.info(f"✅ Selected subtitle: {subtitle_choice}")
                    return subtitle_choice
                else:
                    # Fallback to first file
                    self.logger.info(f"✅ Using first subtitle: {subtitle_files[0]}")
                    return str(subtitle_files[0])
            else:
                # Non-interactive mode: use first file
                self.logger.info(f"✅ Using first subtitle: {subtitle_files[0]}")
                return str(subtitle_files[0])

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            if "No automatic or manual subtitles found" in error_msg:
                self.logger.warning("No subtitles available for this video")
                return None
            else:
                self.logger.error(f"Failed to download transcript: {error_msg}")
                raise RuntimeError(f"Failed to download transcript: {error_msg}")

    async def list_available_subtitles(self, url: str) -> List[Dict[str, Any]]:
        """
        List all available subtitle tracks for a YouTube video

        Args:
            url: YouTube URL

        Returns:
            List of subtitle track information dictionaries
        """
        self.logger.info(f"📋 Listing available subtitles for: {url}")

        cmd = ["yt-dlp", "--list-subs", "--no-download", url]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            # Parse the subtitle list output
            subtitle_info = []
            lines = result.stdout.strip().split("\n")

            # Look for the subtitle section (not automatic captions)
            in_subtitle_section = False
            for line in lines:
                if "Available subtitles for" in line:
                    in_subtitle_section = True
                    continue
                elif "Available automatic captions for" in line:
                    in_subtitle_section = False
                    continue
                elif in_subtitle_section and line.strip():
                    # Skip header lines
                    if "Language" in line and "Name" in line and "Formats" in line:
                        continue

                    # Parse subtitle information
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

                            subtitle_info.append({"language": language, "name": name, "formats": formats})

            self.logger.info(f"✅ Found {len(subtitle_info)} subtitle tracks")
            return subtitle_info

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list subtitles: {e.stderr}")
            raise RuntimeError(f"Failed to list subtitles: {e.stderr}")


class YouTubeSubtitleAgent(WorkflowAgent):
    """Agent for YouTube URL to aligned subtitles workflow

    Configuration (in __init__):
        - downloader, transcriber, aligner: Component instances (dependency injection)
        - max_retries: Max retry attempts for workflow steps

    Runtime parameters (in __call__ or process_youtube_url):
        - url: YouTube URL to process
        - output_dir: Where to save files
        - media_format: Video/audio format (mp3, mp4, etc.)
        - force_overwrite: Whether to overwrite existing files
        - output_format: Subtitle output format (srt, vtt, etc.)
        - split_sentence: Re-segment subtitles semantically
        - word_level: Include word-level timestamps
    """

    def __init__(
        self,
        downloader: YouTubeDownloader,
        transcriber: GeminiTranscriber,
        aligner: AsyncLattifAI,
        max_retries: int = 0,
    ):
        super().__init__("YouTube Subtitle Agent", max_retries)

        # Components (injected)
        self.downloader = downloader
        self.transcriber = transcriber
        self.aligner = aligner

    def define_steps(self) -> List[WorkflowStep]:
        """Define the workflow steps"""
        return [
            WorkflowStep(
                name="Process YouTube URL", description="Extract video info and download video/audio", required=True
            ),
            WorkflowStep(
                name="Transcribe Media",
                description="Download subtitle if available or transcribe the media file",
                required=True,
            ),
            WorkflowStep(name="Align Subtitle", description="Align Subtitle with media using LattifAI", required=True),
            WorkflowStep(
                name="Export Results", description="Export aligned subtitles in specified formats", required=True
            ),
        ]

    async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""

        if step.name == "Process YouTube URL":
            return await self._process_youtube_url(context)

        elif step.name == "Transcribe Media":
            return await self._transcribe_media(context)

        elif step.name == "Align Subtitle":
            return await self._align_subtitle(context)

        elif step.name == "Export Results":
            return await self._export_results(context)

        else:
            raise ValueError(f"Unknown step: {step.name}")

    async def _process_youtube_url(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Process YouTube URL and download video"""
        url = context.get("url")
        if not url:
            raise ValueError("YouTube URL is required")

        output_dir = context.get("output_dir") or tempfile.gettempdir()
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        media_format = context.get("media_format", "mp4")
        force_overwrite = context.get("force_overwrite", False)

        self.logger.info(f"🎥 Processing YouTube URL: {url}")
        self.logger.info(f"📦 Media format: {media_format}")

        # Download media (audio or video) with runtime parameters
        media_path = await self.downloader.download_media(
            url=url,
            output_dir=str(output_dir),
            media_format=media_format,
            force_overwrite=force_overwrite,
        )

        # Try to download subtitles if available
        subtitle_path = None
        try:
            subtitle_path = await self.downloader.download_subtitles(
                url=url,
                output_dir=str(output_dir),
                force_overwrite=force_overwrite,
                enable_gemini_option=bool(self.transcriber.api_key),
            )
            if subtitle_path:
                self.logger.info(f"✅ Subtitle downloaded: {subtitle_path}")
            else:
                self.logger.info("ℹ️  No subtitles available for this video")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to download subtitles: {e}")
            # Continue without subtitles - will transcribe later if needed

        # Get video metadata
        metadata = await self.downloader.get_video_info(url)

        result = {
            "url": url,
            "video_path": media_path,  # Keep 'video_path' key for backward compatibility
            "audio_path": media_path,  # Also add 'audio_path' for clarity
            "metadata": metadata,
            "video_format": media_format,
            "output_dir": output_dir,
            "force_overwrite": force_overwrite,
            "downloaded_subtitle_path": subtitle_path,  # Store downloaded subtitle path
        }

        self.logger.info(f"✅ Media downloaded: {media_path}")
        return result

    async def _transcribe_media(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Transcribe video using Gemini 2.5 Pro or use downloaded subtitle"""
        url = context.get("url")
        result = context.get("process_youtube_url_result", {})
        video_path = result.get("video_path")
        output_dir = result.get("output_dir")
        force_overwrite = result.get("force_overwrite", False)
        downloaded_subtitle_path = result.get("downloaded_subtitle_path")

        if not url or not video_path:
            raise ValueError("URL and video path not found in context")

        video_id = self.downloader.extract_video_id(url)

        # If subtitle was already downloaded in step 1 and user selected it, use it directly
        if downloaded_subtitle_path and downloaded_subtitle_path != "gemini":
            self.logger.info(f"📥 Using subtitle: {downloaded_subtitle_path}")
            return {"subtitle_path": downloaded_subtitle_path}

        # Check for existing subtitles if subtitle was not downloaded yet
        self.logger.info("📥 Checking for existing subtitles...")

        # Check for existing subtitle files (all formats including Gemini transcripts)
        existing_files = FileExistenceManager.check_existing_files(
            video_id,
            str(output_dir),
            subtitle_formats=SUBTITLE_FORMATS,  # Check all subtitle formats including Markdown
        )

        # Prompt user if subtitle exists and force_overwrite is not set
        if existing_files["subtitle"] and not force_overwrite:
            # Let user choose which subtitle file to use
            # Enable Gemini option if API key is available (check transcriber's api_key)
            has_gemini_key = bool(self.transcriber.api_key)
            subtitle_choice = FileExistenceManager.prompt_file_selection(
                file_type="subtitle",
                files=existing_files["subtitle"],
                operation="transcribe",
                enable_gemini=has_gemini_key,
            )

            if subtitle_choice == "cancel":
                raise RuntimeError("Transcription cancelled by user")
            elif subtitle_choice in ("overwrite", "gemini"):
                # Continue to transcription below
                # For 'gemini', user explicitly chose to transcribe with Gemini
                pass
            elif subtitle_choice == "use":
                # User chose to use existing subtitle files (use first one)
                subtitle_path = Path(existing_files["subtitle"][0])
                self.logger.info(f"🔁 Using existing subtitle: {subtitle_path}")
                return {"subtitle_path": str(subtitle_path)}
            elif subtitle_choice:  # User selected a specific file path
                # Use selected subtitle
                subtitle_path = Path(subtitle_choice)
                self.logger.info(f"🔁 Using existing subtitle: {subtitle_path}")
                return {"subtitle_path": str(subtitle_path)}
            # If user_choice == 'overwrite' or 'gemini', continue to transcription below

        # TODO: support other Transcriber options
        self.logger.info("✨ Transcribing URL with Gemini 2.5 Pro...")
        transcript = await self.transcriber.transcribe_url(url)
        subtitle_path = output_dir / f"{video_id}_Gemini.md"
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        result = {"subtitle_path": str(subtitle_path)}
        self.logger.info(f"✅   Transcript generated: {len(transcript)} characters")
        return result

    async def _align_subtitle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Align transcript with video using LattifAI"""
        result = context["process_youtube_url_result"]
        media_path = result.get("video_path", result.get("audio_path"))
        subtitle_path = context.get("transcribe_media_result", {}).get("subtitle_path")

        if not media_path or not subtitle_path:
            raise ValueError("Video path and subtitle path are required")

        self.logger.info("🎯 Aligning subtitle with video...")

        if subtitle_path.endswith("_Gemini.md"):
            is_gemini_format = True
        else:
            is_gemini_format = False
        subtitle_path = Path(subtitle_path)

        self.logger.info(f'📄 Subtitle format: {"Gemini" if is_gemini_format else f"{subtitle_path.suffix}"}')

        original_subtitle_path = subtitle_path
        output_dir = result.get("output_dir")
        split_sentence = context.get("split_sentence", False)
        word_level = context.get("word_level", False)
        output_path = output_dir / f"{Path(media_path).stem}_aligned.ass"

        # Perform alignment with LattifAI (split_sentence and word_level passed as function parameters)
        aligned_result = await self.aligner.alignment(
            audio=media_path,
            subtitle=str(subtitle_path),  # Use dialogue text for YouTube format, original for plain text
            format="gemini" if is_gemini_format else "auto",
            split_sentence=split_sentence,
            return_details=word_level,
            output_subtitle_path=str(output_path),
        )

        result = {
            "aligned_path": output_path,
            "alignment_result": aligned_result,
            "original_subtitle_path": original_subtitle_path,
            "is_gemini_format": is_gemini_format,
        }

        self.logger.info("✅ Alignment completed")
        return result

    async def _export_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Export results in specified format and update subtitle file"""
        align_result = context.get("align_subtitle_result", {})
        aligned_path = align_result.get("aligned_path")
        original_subtitle_path = align_result.get("original_subtitle_path")
        is_gemini_format = align_result.get("is_gemini_format", False)
        metadata = context.get("process_youtube_url_result", {}).get("metadata", {})

        if not aligned_path:
            raise ValueError("Aligned subtitle path not found")

        output_format = context.get("output_format", "srt")
        self.logger.info(f"📤 Exporting results in format: {output_format}")

        supervisions = SubtitleIO.read(aligned_path, format="ass")
        exported_files = {}

        # Update original transcript file with aligned timestamps if YouTube format
        if is_gemini_format:
            assert Path(original_subtitle_path).exists(), "Original subtitle path not found"
            self.logger.info("📝 Updating original transcript with aligned timestamps...")

            try:
                # Generate updated transcript file path
                original_path = Path(original_subtitle_path)
                updated_subtitle_path = original_path.parent / f"{original_path.stem}_LattifAI.md"

                # Update timestamps in original transcript
                GeminiWriter.update_timestamps(
                    original_transcript=original_subtitle_path,
                    aligned_supervisions=supervisions,
                    output_path=str(updated_subtitle_path),
                )

                exported_files["updated_transcript"] = str(updated_subtitle_path)
                self.logger.info(f"✅ Updated transcript: {updated_subtitle_path}")

            except Exception as e:
                self.logger.warning(f"⚠️  Failed to update transcript timestamps: {e}")

        # Export to requested subtitle format
        output_path = str(aligned_path).replace(
            "_aligned.ass", f'{"_Gemini" if is_gemini_format else ""}_LattifAI.{output_format}'
        )
        SubtitleIO.write(supervisions, output_path=output_path)
        exported_files[output_format] = output_path
        self.logger.info(f"✅ Exported {output_format.upper()}: {output_path}")

        result = {
            "exported_files": exported_files,
            "metadata": metadata,
            "subtitle_count": len(supervisions),
            "is_gemini_format": is_gemini_format,
            "original_subtitle_path": original_subtitle_path,
        }

        return result

    async def __call__(
        self,
        url: str,
        output_dir: Optional[str] = None,
        media_format: str = "mp4",
        force_overwrite: bool = False,
        output_format: str = "srt",
        split_sentence: bool = False,
        word_level: bool = False,
    ) -> Dict[str, Any]:
        """Main entry point - callable interface"""
        return await self.process_youtube_url(
            url=url,
            output_dir=output_dir,
            media_format=media_format,
            force_overwrite=force_overwrite,
            output_format=output_format,
            split_sentence=split_sentence,
            word_level=word_level,
        )

    async def process_youtube_url(
        self,
        url: str,
        output_dir: Optional[str] = None,
        media_format: str = "mp4",
        force_overwrite: bool = False,
        output_format: str = "srt",
        split_sentence: bool = False,
        word_level: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a YouTube URL

        Args:
            url: YouTube URL to process
            output_dir: Directory to save output files
            media_format: Media format for download (mp3, mp4, etc.)
            force_overwrite: Force overwrite existing files
            output_format: Subtitle output format (srt, vtt, ass, etc.)
            split_sentence: Re-segment subtitles by semantics
            word_level: Include word-level alignment timestamps

        Returns:
            Dictionary containing results and exported file paths
        """
        # Execute the workflow with parameters
        result = await self.execute(
            url=url,
            output_dir=output_dir,
            media_format=media_format,
            force_overwrite=force_overwrite,
            output_format=output_format,
            split_sentence=split_sentence,
            word_level=word_level,
        )

        if result.is_success:
            return result.data.get("export_results_result", {})
        else:
            # Re-raise the original exception if available to preserve error type and context
            if result.exception:
                raise result.exception
            else:
                raise Exception(f"Workflow failed: {result.error}")
