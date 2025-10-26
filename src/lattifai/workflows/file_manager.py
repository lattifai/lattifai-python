"""
File existence management utilities for video processing workflows
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import colorful


class FileExistenceManager:
    """Utility class for handling file existence checks and user confirmations"""

    @staticmethod
    def check_existing_files(video_id: str, output_dir: str, audio_format: str = 'mp3') -> Dict[str, List[str]]:
        """
        Check for existing audio and subtitle files for a given video ID

        Args:
            video_id: Video ID (e.g., YouTube video ID)
            output_dir: Output directory to check
            audio_format: Audio format to check for

        Returns:
            Dictionary with 'audio' and 'subtitle' keys containing lists of existing files
        """
        output_path = Path(output_dir).expanduser()
        existing_files = {'audio': [], 'subtitle': []}

        if not output_path.exists():
            return existing_files

        # Check for audio files
        audio_extensions = [audio_format, 'mp3', 'wav', 'm4a', 'aac']
        checked_extensions = set()  # Avoid duplicates
        for ext in audio_extensions:
            if ext not in checked_extensions:
                audio_file = output_path / f'{video_id}.{ext}'
                if audio_file.exists():
                    existing_files['audio'].append(str(audio_file))
                checked_extensions.add(ext)

        # Check for subtitle files
        subtitle_extensions = ['srt', 'vtt', 'ass', 'txt']
        for ext in subtitle_extensions:
            subtitle_file = output_path / f'{video_id}.{ext}'
            if subtitle_file.exists():
                existing_files['subtitle'].append(str(subtitle_file))

        return existing_files

    @staticmethod
    def check_existing_media_files(
        video_id: str,
        output_dir: str,
        audio_formats: List[str] = None,
        video_formats: List[str] = None,
        subtitle_formats: List[str] = None,
        transcript_formats: List[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Enhanced version to check for existing media files with customizable formats

        Args:
            video_id: Video ID from any platform
            output_dir: Output directory to check
            audio_formats: List of audio formats to check
            video_formats: List of video formats to check
            subtitle_formats: List of subtitle formats to check
            transcript_formats: List of transcript formats to check

        Returns:
            Dictionary with 'audio', 'video', 'subtitle', 'transcript' keys containing lists of existing files
        """
        output_path = Path(output_dir).expanduser()
        existing_files = {'audio': [], 'video': [], 'subtitle': [], 'transcript': []}

        if not output_path.exists():
            return existing_files

        # Default formats
        audio_formats = audio_formats or ['mp3', 'wav', 'm4a', 'aac', 'opus']
        video_formats = video_formats or ['mp4', 'webm', 'mkv', 'avi']
        subtitle_formats = subtitle_formats or ['srt', 'vtt', 'ass', 'txt']
        transcript_formats = transcript_formats or ['txt', 'transcript']

        # Check for audio files
        for ext in set(audio_formats):  # Remove duplicates
            audio_file = output_path / f'{video_id}.{ext}'
            if audio_file.exists():
                existing_files['audio'].append(str(audio_file))

        # Check for video files
        for ext in set(video_formats):  # Remove duplicates
            video_file = output_path / f'{video_id}.{ext}'
            if video_file.exists():
                existing_files['video'].append(str(video_file))

        # Check for subtitle files
        for ext in set(subtitle_formats):  # Remove duplicates
            subtitle_file = output_path / f'{video_id}.{ext}'
            if subtitle_file.exists():
                existing_files['subtitle'].append(str(subtitle_file))

        # Check for transcript files
        for ext in set(transcript_formats):  # Remove duplicates
            transcript_file = output_path / f'{video_id}_transcript.{ext}'
            if transcript_file.exists():
                existing_files['transcript'].append(str(transcript_file))

        return existing_files

    @staticmethod
    def prompt_user_confirmation(existing_files: Dict[str, List[str]], operation: str = 'download') -> str:
        """
        Prompt user for confirmation when files already exist (legacy, confirms all files together)

        Args:
            existing_files: Dictionary of existing files
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            User choice: 'use' (use existing), 'overwrite' (regenerate), or 'cancel'
        """
        has_audio = bool(existing_files.get('audio', []))
        has_video = bool(existing_files.get('video', []))
        has_subtitle = bool(existing_files.get('subtitle', []))
        has_transcript = bool(existing_files.get('transcript', []))

        if not has_audio and not has_video and not has_subtitle and not has_transcript:
            return 'proceed'  # No existing files, proceed normally

        # Header with warning color
        print(f'\n{colorful.bold_yellow("⚠️  Existing files found:")}')

        if has_audio:
            print(f'{colorful.cyan("📱 Audio files:")}')
            for file_path in existing_files['audio']:
                print(f'   {colorful.green("•")} {file_path}')

        if has_video:
            print(f'{colorful.cyan("🎬 Video files:")}')
            for file_path in existing_files['video']:
                print(f'   {colorful.green("•")} {file_path}')

        if has_subtitle:
            print(f'{colorful.cyan("📝 Subtitle files:")}')
            for file_path in existing_files['subtitle']:
                print(f'   {colorful.green("•")} {file_path}')

        if has_transcript:
            print(f'{colorful.cyan("📄 Transcript files:")}')
            for file_path in existing_files['transcript']:
                print(f'   {colorful.green("•")} {file_path}')

        # Options with colors
        print(f'\n{colorful.bold_black_on_cyan("What would you like to do?")}')
        print(f'{colorful.bold_green("1.")} {colorful.green("Use existing files")} (skip download/generation)')
        print(f'{colorful.bold_yellow("2.")} {colorful.yellow("Overwrite existing files")} (regenerate)')
        print(f'{colorful.bold_red("3.")} {colorful.red("Cancel operation")}')

        while True:
            try:
                choice = input(f'\n{colorful.bold_magenta("Enter your choice (1-3):")} ').strip()

                if choice == '1':
                    print(f'{colorful.green("✅ Using existing files")}')
                    return 'use'
                elif choice == '2':
                    print(f'{colorful.yellow("🔄 Overwriting existing files")}')
                    return 'overwrite'
                elif choice == '3':
                    print(f'{colorful.red("❌ Operation cancelled")}')
                    return 'cancel'
                else:
                    print(f'{colorful.red("Invalid choice. Please enter 1, 2, or 3.")}')

            except (EOFError, KeyboardInterrupt):
                print(f'\n{colorful.red("Operation cancelled by user.")}')
                return 'cancel'

    @staticmethod
    def prompt_file_type_confirmation(file_type: str, files: List[str], operation: str = 'download') -> str:
        """
        Prompt user for confirmation for a specific file type

        Args:
            file_type: Type of file ('audio', 'video', 'subtitle', 'transcript')
            files: List of existing files of this type
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            User choice: 'use' (use existing), 'overwrite' (regenerate), or 'cancel'
        """
        if not files:
            return 'proceed'

        # File type emoji and label mapping
        type_info = {
            'audio': ('📱', 'Audio'),
            'video': ('🎬', 'Video'),
            'subtitle': ('📝', 'Subtitle'),
            'transcript': ('📄', 'Transcript'),
        }

        emoji, label = type_info.get(file_type, ('📄', file_type.capitalize()))

        # Header with warning color
        print(f'\n{colorful.bold_yellow(f"⚠️  Existing {label} files found:")}')

        for file_path in files:
            print(f'   {colorful.green("•")} {file_path}')

        # Options with colors
        print(f'\n{colorful.bold_black_on_cyan(f"What would you like to do with {label.lower()} files?")}')
        print(f'{colorful.bold_green("1.")} {colorful.green(f"Use existing {label.lower()} files")} (skip {operation})')
        print(f'{colorful.bold_yellow("2.")} {colorful.yellow(f"Overwrite {label.lower()} files")} (re-{operation})')
        print(f'{colorful.bold_red("3.")} {colorful.red("Cancel operation")}')

        while True:
            try:
                choice = input(f'\n{colorful.bold_magenta("Enter your choice (1-3):")} ').strip()

                if choice == '1':
                    print(f'{colorful.green(f"✅ Using existing {label.lower()} files")}')
                    return 'use'
                elif choice == '2':
                    print(f'{colorful.yellow(f"🔄 Overwriting {label.lower()} files")}')
                    return 'overwrite'
                elif choice == '3':
                    print(f'{colorful.red("❌ Operation cancelled")}')
                    return 'cancel'
                else:
                    print(f'{colorful.red("Invalid choice. Please enter 1, 2, or 3.")}')

            except (EOFError, KeyboardInterrupt):
                print(f'\n{colorful.red("Operation cancelled by user.")}')
                return 'cancel'

    @staticmethod
    def prompt_per_file_type_confirmation(
        existing_files: Dict[str, List[str]], operation: str = 'download'
    ) -> Dict[str, str]:
        """
        Prompt user for confirmation separately for each file type

        Args:
            existing_files: Dictionary of existing files by type
            operation: Type of operation (e.g., "download", "generate")

        Returns:
            Dictionary mapping file type to user choice ('use', 'overwrite', or 'cancel')
        """
        choices = {}

        # Process each file type separately
        for file_type in ['audio', 'video', 'subtitle', 'transcript']:
            files = existing_files.get(file_type, [])
            if files:
                choice = FileExistenceManager.prompt_file_type_confirmation(file_type, files, operation)
                choices[file_type] = choice

                # If user cancels, stop asking and return
                if choice == 'cancel':
                    return choices
            else:
                choices[file_type] = 'proceed'

        return choices

    @staticmethod
    def is_interactive_mode() -> bool:
        """Check if we're running in interactive mode (TTY available)"""
        return sys.stdin.isatty() and sys.stdout.isatty()


class VideoFileManager:
    """Platform-agnostic video file management utilities"""

    def __init__(self, platform: str = 'generic'):
        """
        Initialize video file manager

        Args:
            platform: Platform identifier (e.g., 'youtube', 'bilibili', 'vimeo')
        """
        self.platform = platform
        self.file_manager = FileExistenceManager()

    def get_video_filename(self, video_id: str, extension: str) -> str:
        """
        Generate standardized filename for video files

        Args:
            video_id: Video identifier
            extension: File extension

        Returns:
            Standardized filename
        """
        # Could be extended to support platform-specific naming conventions
        return f'{video_id}.{extension}'

    def check_existing_files(self, video_id: str, output_dir: str, audio_format: str = 'mp3') -> Dict[str, List[str]]:
        """
        Check for existing files using the platform-agnostic file manager

        Args:
            video_id: Video ID
            output_dir: Output directory
            audio_format: Primary audio format to check

        Returns:
            Dictionary of existing files
        """
        return self.file_manager.check_existing_files(video_id, output_dir, audio_format)

    def check_existing_media_files(
        self,
        video_id: str,
        output_dir: str,
        audio_formats: List[str] = None,
        video_formats: List[str] = None,
        subtitle_formats: List[str] = None,
        transcript_formats: List[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Check for existing media files with platform-specific format support

        Args:
            video_id: Video ID
            output_dir: Output directory
            audio_formats: Supported audio formats
            video_formats: Supported video formats
            subtitle_formats: Supported subtitle formats
            transcript_formats: Supported transcript formats

        Returns:
            Dictionary of existing files
        """
        return self.file_manager.check_existing_media_files(
            video_id, output_dir, audio_formats, video_formats, subtitle_formats, transcript_formats
        )

    def prompt_user_confirmation(self, existing_files: Dict[str, List[str]], operation: str = 'download') -> str:
        """
        Prompt user for confirmation using the file manager (legacy, all files together)

        Args:
            existing_files: Dictionary of existing files
            operation: Operation description

        Returns:
            User choice
        """
        return self.file_manager.prompt_user_confirmation(existing_files, operation)

    def prompt_file_type_confirmation(self, file_type: str, files: List[str], operation: str = 'download') -> str:
        """
        Prompt user for confirmation for a specific file type

        Args:
            file_type: Type of file ('audio', 'video', 'subtitle', 'transcript')
            files: List of existing files of this type
            operation: Operation description

        Returns:
            User choice for this file type
        """
        return self.file_manager.prompt_file_type_confirmation(file_type, files, operation)

    def prompt_per_file_type_confirmation(
        self, existing_files: Dict[str, List[str]], operation: str = 'download'
    ) -> Dict[str, str]:
        """
        Prompt user for confirmation separately for each file type

        Args:
            existing_files: Dictionary of existing files by type
            operation: Operation description

        Returns:
            Dictionary mapping file type to user choice
        """
        return self.file_manager.prompt_per_file_type_confirmation(existing_files, operation)

    def is_interactive_mode(self) -> bool:
        """Check if running in interactive mode"""
        return self.file_manager.is_interactive_mode()
