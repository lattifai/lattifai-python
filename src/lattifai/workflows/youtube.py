"""
YouTube downloader module using yt-dlp
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import setup_workflow_logger
from .file_manager import VideoFileManager


class YouTubeDownloader:
    """YouTube video/audio downloader using yt-dlp"""

    def __init__(self, audio_format: str = 'mp3'):
        self.audio_format = audio_format
        self.logger = setup_workflow_logger('youtube')
        self.file_manager = VideoFileManager(platform='youtube')

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
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return 'youtube_media'

    def _check_ytdlp(self):
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, check=True)
            self.logger.info(f'yt-dlp version: {result.stdout.strip()}')
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                'yt-dlp is not installed or not found in PATH. Please install it with: pip install yt-dlp'
            )

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video metadata without downloading"""
        self.logger.info(f'🔍 Extracting video info for: {url}')

        cmd = ['yt-dlp', '--dump-json', '--no-download', url]

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
                'title': metadata.get('title', 'Unknown'),
                'duration': metadata.get('duration', 0),
                'uploader': metadata.get('uploader', 'Unknown'),
                'upload_date': metadata.get('upload_date', 'Unknown'),
                'view_count': metadata.get('view_count', 0),
                'description': metadata.get('description', ''),
                'thumbnail': metadata.get('thumbnail', ''),
                'webpage_url': metadata.get('webpage_url', url),
            }

            self.logger.info(f'✅ Video info extracted: {info["title"]}')
            return info

        except subprocess.CalledProcessError as e:
            self.logger.error(f'Failed to extract video info: {e.stderr}')
            raise RuntimeError(f'Failed to extract video info: {e.stderr}')
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse video metadata: {e}')
            raise RuntimeError(f'Failed to parse video metadata: {e}')

    async def download_audio(
        self,
        url: str,
        output_dir: Optional[str] = None,
        audio_format: Optional[str] = None,
        force_overwrite: bool = False,
    ) -> str:
        """
        Download audio from YouTube URL

        Args:
            url: YouTube URL
            output_dir: Output directory (default: temp directory)
            audio_format: Audio format (default: instance format)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded audio file
        """
        target_dir = Path(output_dir or tempfile.gettempdir()).expanduser()
        audio_format = audio_format or self.audio_format

        self.logger.info(f'🎵 Downloading audio from: {url}')
        self.logger.info(f'📁       Output directory: {target_dir}')
        self.logger.info(f'🎶           Audio format: {audio_format}')

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing files
        video_id = self.extract_video_id(url)
        existing_files = self.file_manager.check_existing_files(video_id, str(target_dir), audio_format)

        # Handle existing files
        if existing_files['audio'] and not force_overwrite:
            if self.file_manager.is_interactive_mode():
                user_choice = self.file_manager.prompt_user_confirmation(
                    {'audio': existing_files['audio']}, 'audio download'
                )

                if user_choice == 'use':
                    self.logger.info(f'✅ Using existing audio file: {existing_files["audio"][0]}')
                    return existing_files['audio'][0]
                elif user_choice == 'cancel':
                    raise RuntimeError('Audio download cancelled by user')
                # For 'overwrite', continue with download
            else:
                # Non-interactive mode: use existing file
                self.logger.info(f'✅ Using existing audio file: {existing_files["audio"][0]}')
                return existing_files['audio'][0]

        # Generate output filename template
        output_template = str(target_dir / f'{video_id}.%(ext)s')

        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format',
            audio_format,
            '--audio-quality',
            '0',  # Best quality
            '--output',
            output_template,
            '--no-playlist',
            url,
        ]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            self.logger.info('✅ Audio download completed')

            # Find the downloaded file
            # yt-dlp outputs the filename in the last line of stdout
            output_lines = result.stderr.strip().split('\n')
            for line in reversed(output_lines):
                if 'Destination:' in line or 'has already been downloaded' in line:
                    # Extract filename from the line
                    parts = line.split()
                    if 'Destination:' in line:
                        filename = ' '.join(parts[1:])
                    else:
                        # "file.ext has already been downloaded"
                        filename = parts[0]

                    file_path = target_dir / filename
                    if file_path.exists():
                        self.logger.info(f'📄 Downloaded file: {file_path}')
                        return str(file_path)

            # Fallback: search for audio files in output directory
            audio_extensions = [audio_format, 'mp3', 'wav', 'm4a', 'aac']
            for ext in audio_extensions:
                pattern = f'*.{ext}'
                files = list(target_dir.glob(pattern))
                if files:
                    # Return the most recently created file
                    latest_file = max(files, key=os.path.getctime)
                    self.logger.info(f'📄 Found audio file: {latest_file}')
                    return str(latest_file)

            raise RuntimeError('Downloaded audio file not found')

        except subprocess.CalledProcessError as e:
            self.logger.error(f'Failed to download audio: {e.stderr}')
            raise RuntimeError(f'Failed to download audio: {e.stderr}')

    async def download_video(
        self, url: str, output_dir: Optional[str] = None, video_format: str = 'mp4', force_overwrite: bool = False
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
        target_dir = Path(output_dir or tempfile.gettempdir()).expanduser()

        self.logger.info(f'🎬 Downloading video from: {url}')
        self.logger.info(f'📁 Output directory: {target_dir}')
        self.logger.info(f'🎥 Video format: {video_format}')

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing files
        video_id = self.extract_video_id(url)
        video_file = target_dir / f'{video_id}.{video_format}'

        # Handle existing files
        if video_file.exists() and not force_overwrite:
            if self.file_manager.is_interactive_mode():
                user_choice = self.file_manager.prompt_user_confirmation({'video': [str(video_file)]}, 'video download')

                if user_choice == 'use':
                    self.logger.info(f'✅ Using existing video file: {video_file}')
                    return str(video_file)
                elif user_choice == 'cancel':
                    raise RuntimeError('Video download cancelled by user')
                # For 'overwrite', continue with download
            else:
                # Non-interactive mode: use existing file
                self.logger.info(f'✅ Using existing video file: {video_file}')
                return str(video_file)

        # Generate output filename template
        output_template = str(target_dir / f'{video_id}.%(ext)s')

        # Use flexible format selection and merge if needed
        # 'bestvideo*+bestaudio/best' will merge best video and audio, or download best single file
        # --merge-output-format ensures the final format matches our requirement
        cmd = [
            'yt-dlp',
            '--format',
            'bestvideo*+bestaudio/best',
            '--merge-output-format',
            video_format,
            '--output',
            output_template,
            '--no-playlist',
            url,
        ]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )
            # TODO: check result
            del result
            self.logger.info('✅ Video download completed')

            # Find the downloaded file - check for the expected format first
            video_file = target_dir / f'{video_id}.{video_format}'
            if video_file.exists():
                self.logger.info(f'📄 Downloaded video: {video_file}')
                return str(video_file)

            # Fallback: search for any video files with this video_id
            video_patterns = [f'{video_id}.{video_format}', f'{video_id}.*']
            for pattern in video_patterns:
                video_files = list(target_dir.glob(pattern))
                if video_files:
                    # Return the most recently created file
                    latest_file = max(video_files, key=os.path.getctime)
                    self.logger.info(f'📄 Downloaded video: {latest_file}')
                    return str(latest_file)

            raise RuntimeError('Downloaded video file not found')

        except subprocess.CalledProcessError as e:
            self.logger.error(f'Failed to download video: {e.stderr}')
            raise RuntimeError(f'Failed to download video: {e.stderr}')

    async def download_transcript(
        self, url: str, output_dir: str, force_overwrite: bool = False, transcript_format: str = 'txt'
    ) -> Optional[str]:
        """
        Download video transcript/subtitles using yt-dlp

        Args:
            url: YouTube URL
            output_dir: Output directory
            force_overwrite: Skip user confirmation and overwrite existing files
            transcript_format: Format for transcript (txt, vtt, srt)

        Returns:
            Path to downloaded transcript file or None if not available
        """
        target_dir = Path(output_dir).expanduser()

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing transcript files
        video_id = self.extract_video_id(url)
        existing_files = self.file_manager.check_existing_media_files(
            video_id, str(target_dir), transcript_formats=[transcript_format, 'txt', 'vtt', 'srt']
        )

        # Handle existing transcript files
        if existing_files['transcript'] and not force_overwrite:
            if self.file_manager.is_interactive_mode():
                user_choice = self.file_manager.prompt_user_confirmation(
                    {'transcript': existing_files['transcript']}, 'transcript download'
                )

                if user_choice == 'use':
                    transcript_file = Path(existing_files['transcript'][0])
                    self.logger.info(f'✅ Using existing transcript file: {transcript_file}')
                    return str(transcript_file)
                elif user_choice == 'cancel':
                    raise RuntimeError('Transcript download cancelled by user')
                # For 'overwrite', continue with download
            else:
                transcript_file = Path(existing_files['transcript'][0])
                self.logger.info(f'🔍 Found existing transcript: {transcript_file}')
                return str(transcript_file)

        self.logger.info(f'📄 Downloading transcript for: {url}')

        output_template = str(target_dir / f'{video_id}_transcript.%(ext)s')

        # Configure yt-dlp options for transcript download
        ytdlp_options = [
            'yt-dlp',
            '--write-auto-sub',  # Download automatic subtitles
            '--write-sub',  # Download manual subtitles if available
            '--skip-download',  # Don't download video/audio
            '--output',
            output_template,
            '--sub-format',
            'vtt/srt/best',  # Prefer vtt or srt format
            url,
        ]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(ytdlp_options, capture_output=True, text=True, check=True)
            )

            self.logger.info(f'yt-dlp transcript output: {result.stdout.strip()}')

            # Find the downloaded transcript file
            transcript_patterns = [
                f'{video_id}_transcript*.vtt',
                f'{video_id}_transcript*.srt',
                f'{video_id}_transcript*.txt',
            ]

            for pattern in transcript_patterns:
                transcript_files = list(target_dir.glob(pattern))
                if transcript_files:
                    # Convert to desired format if needed
                    transcript_file = transcript_files[0]

                    if transcript_format == 'txt' and transcript_file.suffix != '.txt':
                        # Convert to plain text
                        txt_file = target_dir / f'{video_id}_transcript.txt'
                        self._convert_subtitle_to_text(transcript_file, txt_file)
                        transcript_file = txt_file

                    self.logger.info(f'📄 Downloaded transcript: {transcript_file}')
                    return str(transcript_file)

            self.logger.warning('No transcript available for this video')
            return None

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            if 'No automatic or manual subtitles found' in error_msg:
                self.logger.warning('No transcript/subtitles available for this video')
                return None
            else:
                self.logger.error(f'Failed to download transcript: {error_msg}')
                raise RuntimeError(f'Failed to download transcript: {error_msg}')

    def _convert_subtitle_to_text(self, subtitle_file: Path, output_file: Path) -> None:
        """Convert subtitle file (vtt/srt) to plain text"""
        try:
            content = subtitle_file.read_text(encoding='utf-8')

            # Remove VTT/SRT formatting and extract text
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                # Skip timestamps, numbers, and empty lines
                if not line or line.isdigit() or '-->' in line or line.startswith('WEBVTT'):
                    continue
                # Remove HTML-like tags
                import re

                line = re.sub(r'<[^>]*>', '', line)
                if line:
                    lines.append(line)

            # Write cleaned text
            output_file.write_text('\n'.join(lines), encoding='utf-8')
            self.logger.info(f'✅ Converted to text format: {output_file}')

        except Exception as e:
            self.logger.error(f'Failed to convert subtitle to text: {e}')
            # Fallback: just copy the file
            import shutil

            shutil.copy2(subtitle_file, output_file)


class YouTubeWorkflow:
    """High-level YouTube workflow orchestrator"""

    def __init__(self, audio_format: str = 'mp3'):
        self.downloader = YouTubeDownloader(audio_format=audio_format)
        self.logger = setup_workflow_logger('youtube')

    async def process_url(
        self,
        url: str,
        output_dir: Optional[str] = None,
        download_video: bool = False,
        download_transcript: bool = True,
        force_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Process YouTube URL to extract metadata and download media

        Args:
            url: YouTube URL
            output_dir: Output directory
            download_video: Whether to download video in addition to audio
            download_transcript: Whether to download transcript/subtitles
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Dictionary with metadata and file paths
        """
        self.logger.info(f'🚀 Processing YouTube URL: {url}')

        # Get video metadata
        metadata = await self.downloader.get_video_info(url)

        # Download audio
        audio_path = await self.downloader.download_audio(url, output_dir, force_overwrite=force_overwrite)

        result = {'url': url, 'metadata': metadata, 'audio_path': audio_path}

        # Download video if requested
        if download_video:
            video_path = await self.downloader.download_video(url, output_dir, force_overwrite=force_overwrite)
            result['video_path'] = video_path

        # Download transcript if requested
        if download_transcript:
            transcript_path = await self.downloader.download_transcript(
                url, output_dir, force_overwrite=force_overwrite
            )
            if transcript_path:
                result['transcript_path'] = transcript_path

        self.logger.info('✅ YouTube processing completed')
        return result
