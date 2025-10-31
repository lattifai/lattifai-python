"""
YouTube downloader module using yt-dlp and Agent
"""

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..client import LattifAI
from ..io import GeminiReader, GeminiWriter, SubtitleIO
from .base import WorkflowAgent, WorkflowStep, setup_workflow_logger
from .file_manager import VideoFileManager
from .gemini import GeminiTranscriber


class YouTubeDownloader:
    """YouTube video/audio downloader using yt-dlp"""

    def __init__(self, media_format: str = 'mp3'):
        self.media_format = media_format
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
            media_format: Media format - audio (mp3, wav, m4a, aac, opus, ogg, flac)
                         or video (mp4, webm, mkv, avi, etc.) (default: instance format)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded media file
        """
        media_format = media_format or self.media_format

        # Determine if format is audio or video
        audio_formats = ['mp3', 'wav', 'm4a', 'aac', 'opus', 'ogg', 'flac']
        is_audio = media_format.lower() in audio_formats

        if is_audio:
            self.logger.info(f'🎵 Detected audio format: {media_format}')
            return await self.download_audio(
                url=url, output_dir=output_dir, media_format=media_format, force_overwrite=force_overwrite
            )
        else:
            self.logger.info(f'🎬 Detected video format: {media_format}')
            return await self.download_video(
                url=url, output_dir=output_dir, video_format=media_format, force_overwrite=force_overwrite
            )

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
            media_format: Media format (default: instance format)
            force_overwrite: Skip user confirmation and overwrite existing files

        Returns:
            Path to downloaded audio file
        """
        target_dir = Path(output_dir or tempfile.gettempdir()).expanduser()
        media_format = media_format or self.media_format

        self.logger.info(f'🎵 Downloading audio from: {url}')
        self.logger.info(f'📁       Output directory: {target_dir}')
        self.logger.info(f'🎶           Media format: {media_format}')

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing files
        video_id = self.extract_video_id(url)
        existing_files = self.file_manager.check_existing_files(video_id, str(target_dir), media_format)

        # Handle existing files
        if existing_files['media'] and not force_overwrite:
            if self.file_manager.is_interactive_mode():
                user_choice = self.file_manager.prompt_user_confirmation(
                    {'media': existing_files['media']}, 'media download'
                )

                if user_choice == 'use':
                    self.logger.info(f'✅ Using existing media file: {existing_files["media"][0]}')
                    return existing_files['media'][0]
                elif user_choice == 'cancel':
                    raise RuntimeError('Media download cancelled by user')
                # For 'overwrite', continue with download
            else:
                # Non-interactive mode: use existing file
                self.logger.info(f'✅ Using existing media file: {existing_files["media"][0]}')
                return existing_files['media'][0]

        # Generate output filename template
        output_template = str(target_dir / f'{video_id}.%(ext)s')

        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format',
            media_format,
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
            audio_extensions = [media_format, 'mp3', 'wav', 'm4a', 'aac']
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
                user_choice = self.file_manager.prompt_user_confirmation({'media': [str(video_file)]}, 'media download')

                if user_choice == 'use':
                    self.logger.info(f'✅ Using existing media file: {video_file}')
                    return str(video_file)
                elif user_choice == 'cancel':
                    raise RuntimeError('Media download cancelled by user')
                # For 'overwrite', continue with download
            else:
                # Non-interactive mode: use existing file
                self.logger.info(f'✅ Using existing media file: {video_file}')
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

    async def download_subtitles(
        self, url: str, output_dir: str, force_overwrite: bool = False, subtitle_lang: Optional[str] = None
    ) -> Optional[str]:
        """
        Download video subtitles using yt-dlp

        Args:
            url: YouTube URL
            output_dir: Output directory
            force_overwrite: Skip user confirmation and overwrite existing files
            subtitle_lang: Specific subtitle language/track to download (e.g., 'en')
                          If None, downloads all available subtitles

        Returns:
            Path to downloaded transcript file or None if not available
        """
        target_dir = Path(output_dir).expanduser()

        # Create output directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract video ID and check for existing subtitle files
        video_id = self.extract_video_id(url)
        if not force_overwrite:
            existing_files = self.file_manager.check_existing_media_files(
                video_id, str(target_dir), subtitle_formats=['vtt', 'srt']
            )

            # Handle existing subtitle files
            if existing_files['subtitle'] and not force_overwrite:
                if self.file_manager.is_interactive_mode():
                    user_choice = self.file_manager.prompt_user_confirmation(
                        {'subtitle': existing_files['subtitle']}, 'subtitle download'
                    )

                    if user_choice == 'use':
                        subtitle_file = Path(existing_files['subtitle'][0])
                        self.logger.info(f'✅ Using existing subtitle file: {subtitle_file}')
                        return str(subtitle_file)
                    elif user_choice == 'cancel':
                        raise RuntimeError('Subtitle download cancelled by user')
                    # For 'overwrite', continue with download
                else:
                    subtitle_file = Path(existing_files['subtitle'][0])
                    self.logger.info(f'🔍 Found existing subtitle: {subtitle_file}')
                    return str(subtitle_file)

        self.logger.info(f'📄 Downloading subtitle for: {url}')
        if subtitle_lang:
            self.logger.info(f'🎯 Targeting specific subtitle track: {subtitle_lang}')

        output_template = str(target_dir / f'{video_id}.%(ext)s')

        # Configure yt-dlp options for subtitle download
        ytdlp_options = [
            'yt-dlp',
            '--skip-download',  # Don't download video/audio
            '--output',
            output_template,
            '--sub-format',
            'best',  # Prefer best available format
        ]

        # Add subtitle language selection if specified
        if subtitle_lang:
            ytdlp_options.extend(['--write-sub', '--write-auto-sub', '--sub-langs', f'{subtitle_lang}.*'])
        else:
            # Download only manual subtitles (not auto-generated) in English to avoid rate limiting
            ytdlp_options.extend(['--write-sub', '--write-auto-sub'])

        ytdlp_options.append(url)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(ytdlp_options, capture_output=True, text=True, check=True)
            )

            self.logger.info(f'yt-dlp transcript output: {result.stdout.strip()}')

            # Find the downloaded transcript file
            subtitle_patterns = [
                f'{video_id}.*vtt',
                f'{video_id}.*srt',
            ]

            subtitle_files = []
            for pattern in subtitle_patterns:
                _subtitle_files = list(target_dir.glob(pattern))
                for subtitle_file in _subtitle_files:
                    self.logger.info(f'📄 Downloaded subtitle: {subtitle_file}')
                subtitle_files.extend(_subtitle_files)

            if not subtitle_files:
                self.logger.warning('No subtitle available for this video')
            return subtitle_files

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            if 'No automatic or manual subtitles found' in error_msg:
                self.logger.warning('No subtitles available for this video')
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

    async def list_available_subtitles(self, url: str) -> List[Dict[str, Any]]:
        """
        List all available subtitle tracks for a YouTube video

        Args:
            url: YouTube URL

        Returns:
            List of subtitle track information dictionaries
        """
        self.logger.info(f'📋 Listing available subtitles for: {url}')

        cmd = ['yt-dlp', '--list-subs', '--no-download', url]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True)
            )

            # Parse the subtitle list output
            subtitle_info = []
            lines = result.stdout.strip().split('\n')

            # Look for the subtitle section (not automatic captions)
            in_subtitle_section = False
            for line in lines:
                if 'Available subtitles for' in line:
                    in_subtitle_section = True
                    continue
                elif 'Available automatic captions for' in line:
                    in_subtitle_section = False
                    continue
                elif in_subtitle_section and line.strip():
                    # Skip header lines
                    if 'Language' in line and 'Name' in line and 'Formats' in line:
                        continue

                    # Parse subtitle information
                    # Format: "Language Name Formats" where formats are comma-separated
                    # Example: "en-uYU-mmqFLq8 English - CC1    vtt, srt, ttml, srv3, srv2, srv1, json3"

                    if line.strip() and not line.startswith('['):
                        # Split by multiple spaces to separate language, name, and formats
                        import re

                        parts = re.split(r'\s{2,}', line.strip())

                        if len(parts) >= 2:
                            # First part is language, last part is formats
                            language_and_name = parts[0]
                            formats_str = parts[-1]

                            # Split language and name - language is first word
                            lang_name_parts = language_and_name.split(' ', 1)
                            language = lang_name_parts[0]
                            name = lang_name_parts[1] if len(lang_name_parts) > 1 else ''

                            # If there are more than 2 parts, middle parts are also part of name
                            if len(parts) > 2:
                                name = ' '.join([name] + parts[1:-1]).strip()

                            # Parse formats - they are comma-separated
                            formats = [f.strip() for f in formats_str.split(',') if f.strip()]

                            subtitle_info.append({'language': language, 'name': name, 'formats': formats})

            self.logger.info(f'✅ Found {len(subtitle_info)} subtitle tracks')
            return subtitle_info

        except subprocess.CalledProcessError as e:
            self.logger.error(f'Failed to list subtitles: {e.stderr}')
            raise RuntimeError(f'Failed to list subtitles: {e.stderr}')


class YouTubeSubtitleAgent(WorkflowAgent):
    """Agent for YouTube URL to aligned subtitles workflow"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        video_format: str = 'mp4',
        output_format: str = 'srt',
        max_retries: int = 0,
        split_sentence: bool = False,
        output_dir: Optional[Path] = None,
        force_overwrite: bool = False,
    ):
        super().__init__('YouTube Subtitle Agent', max_retries)

        # Configuration
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.video_format = video_format
        self.output_format = output_format
        self.split_sentence = split_sentence
        self.output_dir = output_dir or Path(tempfile.gettempdir())
        self.force_overwrite = force_overwrite

        # Initialize components
        self.downloader = YouTubeDownloader(media_format='mp3')  # Keep for backward compatibility
        self.transcriber = GeminiTranscriber(api_key=self.gemini_api_key)
        self.aligner = LattifAI()

        # Validate configuration
        if not self.gemini_api_key:
            raise ValueError(
                'Gemini API key is required. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.'
            )

    def define_steps(self) -> List[WorkflowStep]:
        """Define the workflow steps"""
        return [
            WorkflowStep(
                name='Process YouTube URL', description='Extract video info and download video/audio', required=True
            ),
            WorkflowStep(
                name='Transcribe Media',
                description='Download subtitle if available or transcribe the media file',
                required=True,
            ),
            WorkflowStep(name='Align Subtitle', description='Align Subtitle with media using LattifAI', required=True),
            WorkflowStep(
                name='Export Results', description='Export aligned subtitles in specified formats', required=True
            ),
        ]

    async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""

        if step.name == 'Process YouTube URL':
            return await self._process_youtube_url(context)

        elif step.name == 'Transcribe Media':
            return await self._transcribe_media(context)

        elif step.name == 'Align Subtitle':
            return await self._align_subtitle(context)

        elif step.name == 'Export Results':
            return await self._export_results(context)

        else:
            raise ValueError(f'Unknown step: {step.name}')

    async def _process_youtube_url(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Process YouTube URL and download video"""
        url = context.get('url')
        if not url:
            raise ValueError('YouTube URL is required')

        self.logger.info(f'🎥 Processing YouTube URL: {url}')

        # Download video (no conversion needed for Gemini)
        video_path = await self.downloader.download_video(
            url=url, output_dir=self.output_dir, video_format=self.video_format, force_overwrite=self.force_overwrite
        )

        # Get video metadata
        metadata = await self.downloader.get_video_info(url)

        result = {'url': url, 'video_path': video_path, 'metadata': metadata, 'video_format': self.video_format}

        self.logger.info(f'✅ Video downloaded: {video_path}')
        return result

    async def _transcribe_media(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Transcribe video using Gemini 2.5 Pro"""
        url = context.get('url')
        video_path = context.get('process_youtube_url_result', {}).get('video_path')

        if not url or not video_path:
            raise ValueError('URL and video path not found in context')

        video_id = self.downloader.extract_video_id(url)

        # Download subtitle if available
        self.logger.info('📥 Checking for existing subtitles...')

        # Check for existing subtitle files (all formats including Gemini transcripts)
        existing_files = self.downloader.file_manager.check_existing_media_files(
            video_id,
            str(self.output_dir),
            subtitle_formats=['md', 'vtt', 'srt', 'ass'],  # Check all subtitle formats including Markdown
        )

        # Prompt user if subtitle exists and force_overwrite is not set
        if existing_files['subtitle'] and not self.force_overwrite:
            from .file_manager import FileExistenceManager

            # Let user choose which subtitle file to use
            subtitle_choice = FileExistenceManager.prompt_file_selection(
                file_type='subtitle file', files=existing_files['subtitle'], operation='transcribe'
            )

            if subtitle_choice == 'cancel':
                raise RuntimeError('Transcription cancelled by user')
            elif subtitle_choice == 'overwrite':
                # Continue to transcription below
                pass
            elif subtitle_choice:  # User selected a specific file
                # Use selected subtitle
                subtitle_path = Path(subtitle_choice)
                self.logger.info(f'🔁 Using existing subtitle: {subtitle_path}')

                return {'subtitle_path': str(subtitle_path)}
            # If user_choice == 'overwrite', continue to transcription below

        # TODO: support other Transcriber options
        self.logger.info('🎤 Transcribing URL with Gemini 2.5 Pro...')
        transcript = await self.transcriber.transcribe_url(url)
        subtitle_path = self.output_dir / f'{video_id}_gemini.md'
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        result = {'subtitle_path': str(subtitle_path)}
        self.logger.info(f'✅   Transcript generated: {len(transcript)} characters')
        return result

    async def _align_subtitle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Align transcript with video using LattifAI"""
        result = context['process_youtube_url_result']
        media_path = result.get('video_path', result.get('audio_path'))
        subtitle_path = context.get('transcribe_media_result', {}).get('subtitle_path')

        if not media_path or not subtitle_path:
            raise ValueError('Video path and subtitle path are required')

        self.logger.info('🎯 Aligning subtitle with video...')

        # Detect transcript format using GeminiReader
        if subtitle_path.endswith('_gemini.md'):
            # segments = GeminiReader.read(subtitle_path, include_events=True, include_sections=True)
            # supervisions = GeminiReader.extract_for_alignment(
            #     subtitle_path, merge_consecutive=False, min_duration=0.1
            # )
            is_gemini_format = True
        else:
            # segments = SubtitleIO.read(subtitle_path, format='auto')
            is_gemini_format = False

        subtitle_path = Path(subtitle_path)

        self.logger.info(f'📄 Subtitle format: {"Gemini" if is_gemini_format else f"{subtitle_path.suffix}"}')

        original_subtitle_path = subtitle_path

        # Create temporary output file
        output_path = Path(self.output_dir) / f'{Path(media_path).stem}_aligned.ass'

        # Perform alignment with LattifAI using extracted/original text
        aligned_result = self.aligner.alignment(
            audio=media_path,
            subtitle=str(subtitle_path),  # Use dialogue text for YouTube format, original for plain text
            format='gemini' if is_gemini_format else 'auto',
            split_sentence=self.split_sentence,
            output_subtitle_path=str(output_path),
        )

        result = {
            'aligned_path': output_path,
            'alignment_result': aligned_result,
            'original_subtitle_path': original_subtitle_path,
            'is_gemini_format': is_gemini_format,
        }

        self.logger.info('✅ Alignment completed')
        return result

    async def _export_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Export results in specified format and update subtitle file"""
        align_result = context.get('align_subtitle_result', {})
        aligned_path = align_result.get('aligned_path')
        original_subtitle_path = align_result.get('original_subtitle_path')
        is_gemini_format = align_result.get('is_gemini_format', False)
        metadata = context.get('process_youtube_url_result', {}).get('metadata', {})

        if not aligned_path:
            raise ValueError('Aligned subtitle path not found')

        self.logger.info(f'📤 Exporting results in format: {self.output_format}')

        supervisions = SubtitleIO.read(aligned_path, format='ass')
        exported_files = {}

        # Update original transcript file with aligned timestamps if YouTube format
        if is_gemini_format and original_subtitle_path:
            self.logger.info('📝 Updating original transcript with aligned timestamps...')

            try:
                # Generate updated transcript file path
                original_path = Path(original_subtitle_path)
                updated_subtitle_path = original_path.parent / f'{original_path.stem}_aligned.md'

                # Update timestamps in original transcript
                GeminiWriter.update_timestamps(
                    original_transcript=original_subtitle_path,
                    aligned_supervisions=supervisions,
                    output_path=str(updated_subtitle_path),
                )

                exported_files['updated_transcript'] = str(updated_subtitle_path)
                self.logger.info(f'✅ Updated transcript: {updated_subtitle_path}')

            except Exception as e:
                self.logger.warning(f'⚠️  Failed to update transcript timestamps: {e}')

        # Export to requested subtitle format
        output_path = str(aligned_path).replace('_aligned.ass', f'_lattifai.{self.output_format}')
        SubtitleIO.write(supervisions, output_path=output_path)
        exported_files[self.output_format] = output_path
        self.logger.info(f'✅ Exported {self.output_format.upper()}: {output_path}')

        result = {
            'exported_files': exported_files,
            'metadata': metadata,
            'subtitle_count': len(supervisions),
            'is_gemini_format': is_gemini_format,
            'original_subtitle_path': original_subtitle_path,
        }

        return result

    async def process_youtube_url(
        self, url: str, output_dir: Optional[str] = None, output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a YouTube URL

        Args:
            url: YouTube URL to process
            output_dir: Directory to save output files (optional)
            output_format: Output format (optional, uses instance default)

        Returns:
            Dictionary containing results and exported file paths
        """
        if output_format:
            self.output_format = output_format

        if output_dir:
            expanded_dir = Path(output_dir).expanduser()
            expanded_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = expanded_dir

        # Execute the workflow
        result = await self.execute(url=url)

        if result.is_success:
            return result.data.get('export_results_result', {})
        else:
            raise Exception(f'Workflow failed: {result.error}')
