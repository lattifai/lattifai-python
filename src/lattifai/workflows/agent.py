"""
YouTube Alignment Agent

An agentic workflow for processing YouTube videos through:
1. URL processing and audio download
2. Gemini 2.5 Pro transcription
3. LattifAI alignment
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..client import LattifAI
from ..io import GeminiReader, GeminiWriter
from .base import WorkflowAgent, WorkflowStep
from .gemini import GeminiTranscriber
from .youtube import YouTubeDownloader


class YouTubeAlignmentAgent(WorkflowAgent):
    """Agent for YouTube URL to aligned subtitles workflow"""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        video_format: str = 'mp4',
        output_formats: List[str] = None,
        max_retries: int = 0,
        output_dir: Optional[Path] = None,
        force_overwrite: bool = False,
    ):
        super().__init__('YouTube Alignment Agent', max_retries)

        # Configuration
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.video_format = video_format
        self.output_formats = output_formats or ['srt']
        self.output_dir = output_dir or Path(tempfile.gettempdir())
        self.force_overwrite = force_overwrite

        # Initialize components
        self.downloader = YouTubeDownloader(audio_format='mp3')  # Keep for backward compatibility
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
                name='Process YouTube URL', description='Extract video info and download video', required=True
            ),
            WorkflowStep(
                name='Transcribe Media',
                description='Generate transcript from video using Gemini 2.5 Pro',
                required=True,
            ),
            WorkflowStep(
                name='Align Transcript', description='Align transcript with video using LattifAI', required=True
            ),
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

        elif step.name == 'Align Transcript':
            return await self._align_transcript(context)

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

        # Check for existing transcript files
        existing_transcripts = self.downloader.file_manager.check_existing_media_files(
            video_id, str(self.output_dir), transcript_formats=['txt', 'vtt', 'srt', 'vtt', 'ass']
        )

        # Prompt user if transcript exists and force_overwrite is not set
        if existing_transcripts['transcript'] and not self.force_overwrite:
            from .file_manager import FileExistenceManager

            user_choice = FileExistenceManager.prompt_file_type_confirmation(
                file_type='transcript', files=existing_transcripts['transcript'], operation='transcription'
            )

            if user_choice == 'cancel':
                raise RuntimeError('Transcription cancelled by user')
            elif user_choice == 'use':
                # Use existing transcript
                existing_path = Path(existing_transcripts['transcript'][0])
                self.logger.info(f'🔁 Using existing transcript: {existing_path}')

                if existing_path.suffix != '.txt':
                    transcript_path = self.output_dir / f'{video_id}_transcript.txt'
                    if transcript_path != existing_path:
                        self.logger.info('🧹 Converting transcript to plain text format')
                        self.downloader._convert_subtitle_to_text(existing_path, transcript_path)
                else:
                    transcript_path = existing_path

                transcript = Path(transcript_path).read_text(encoding='utf-8')
                return {'transcript': transcript, 'transcript_path': str(transcript_path)}
            # If user_choice == 'overwrite', continue to transcription below

        self.logger.info('🎤 Transcribing URL with Gemini 2.5 Pro...')

        transcript = await self.transcriber.transcribe_url(url)

        # Save transcript to temporary file
        transcript_path = self.output_dir / f'{video_id}_transcript.txt'
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        result = {'transcript': transcript, 'transcript_path': str(transcript_path)}

        self.logger.info(f'✅   Transcript generated: {len(transcript)} characters')
        return result

    async def _align_transcript(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Align transcript with video using LattifAI"""
        video_path = context.get('process_youtube_url_result', {}).get('video_path')
        transcript_path = context.get('transcribe_media_result', {}).get('transcript_path')

        if not video_path or not transcript_path:
            raise ValueError('Video path and transcript path are required')

        self.logger.info('🎯 Aligning transcript with video...')

        transcript_path_obj = Path(transcript_path)

        # Step 1: Detect transcript format using GeminiReader
        try:
            segments = GeminiReader.read(transcript_path_obj, include_events=True, include_sections=True)
            is_youtube_format = any(seg.speaker for seg in segments) or any(
                seg.segment_type == 'section_header' for seg in segments
            )
        except Exception as e:
            self.logger.warning(f'Failed to parse transcript, assuming plain text format: {e}')
            is_youtube_format = False

        self.logger.info(f'📄 Transcript format: {"YouTube" if is_youtube_format else "Plain text"}')

        alignment_text_path = transcript_path
        original_transcript_path = transcript_path

        # Step 2: Extract dialogue text for alignment if YouTube format
        if is_youtube_format:
            self.logger.info('🎭 Extracting dialogue from YouTube format transcript...')

            # Extract dialogue segments using GeminiReader
            supervisions = GeminiReader.extract_for_alignment(
                transcript_path_obj, merge_consecutive=False, min_duration=0.1
            )

            # Create dialogue-only text file for alignment
            dialogue_texts = [sup.text for sup in supervisions]
            dialogue_content = '\n'.join(dialogue_texts)

            # Save dialogue text to temporary file
            dialogue_path = transcript_path_obj.parent / f'{transcript_path_obj.stem}_text_for_alignment.txt'
            dialogue_path.write_text(dialogue_content, encoding='utf-8')
            alignment_text_path = str(dialogue_path)

            self.logger.info(f'💬 Extracted {len(supervisions)} segments for alignment')
            self.logger.info(f'📝 Text saved: {dialogue_path}')
        else:
            self.logger.info('📝 Using plain text transcript directly for alignment')

        # Create temporary output file
        output_base = Path(self.output_dir) / f'{Path(video_path).stem}_aligned'

        # Step 3: Perform alignment with LattifAI using extracted/original text
        aligned_result = self.aligner.alignment(
            audio=video_path,
            subtitle=alignment_text_path,  # Use dialogue text for YouTube format, original for plain text
            format='txt',
            split_sentence=True,
            output_subtitle_path=f'{output_base}.srt',
        )

        result = {
            'aligned_path': f'{output_base}.srt',
            'alignment_result': aligned_result,
            'original_transcript_path': original_transcript_path,
            'is_youtube_format': is_youtube_format,
            'alignment_text_path': alignment_text_path,
        }

        self.logger.info('✅ Alignment completed')
        return result

    async def _export_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Export results in specified formats and update transcript file"""
        align_result = context.get('align_transcript_result', {})
        aligned_path = align_result.get('aligned_path')
        original_transcript_path = align_result.get('original_transcript_path')
        is_youtube_format = align_result.get('is_youtube_format', False)
        metadata = context.get('process_youtube_url_result', {}).get('metadata', {})

        if not aligned_path:
            raise ValueError('Aligned subtitle path not found')

        self.logger.info(f'📤 Exporting results in formats: {self.output_formats}')

        # Read aligned subtitles
        from ..io import SubtitleIO

        supervisions = SubtitleIO.read(aligned_path)
        exported_files = {}

        # Step 1: Update original transcript file with aligned timestamps if YouTube format
        if is_youtube_format and original_transcript_path:
            self.logger.info('📝 Updating original transcript with aligned timestamps...')

            try:
                # Generate updated transcript file path
                original_path = Path(original_transcript_path)
                updated_transcript_path = original_path.parent / f'{original_path.stem}_aligned.txt'

                # Update timestamps in original transcript
                GeminiWriter.update_timestamps(
                    original_transcript_path=original_transcript_path,
                    aligned_supervisions=supervisions,
                    output_path=str(updated_transcript_path),
                )

                exported_files['updated_transcript'] = str(updated_transcript_path)
                self.logger.info(f'✅ Updated transcript: {updated_transcript_path}')

            except Exception as e:
                self.logger.warning(f'⚠️  Failed to update transcript timestamps: {e}')

        # Step 2: Generate simplified aligned transcript
        if original_transcript_path:
            self.logger.info('📄 Generating simplified aligned transcript...')

            try:
                original_path = Path(original_transcript_path)
                simple_transcript_path = original_path.parent / f'{original_path.stem}_simple_aligned.txt'

                # Write simplified aligned transcript
                GeminiWriter.write_aligned_transcript(
                    aligned_supervisions=supervisions,
                    output_path=str(simple_transcript_path),
                    include_word_timestamps=False,
                )

                exported_files['simple_transcript'] = str(simple_transcript_path)
                self.logger.info(f'✅ Simple transcript: {simple_transcript_path}')

            except Exception as e:
                self.logger.warning(f'⚠️  Failed to generate simple transcript: {e}')

        # Step 3: Export to requested subtitle formats
        for format_name in self.output_formats:
            output_path = aligned_path.replace('.srt', f'.{format_name}')
            SubtitleIO.write(supervisions, output_path=output_path)
            exported_files[format_name] = output_path
            self.logger.info(f'✅ Exported {format_name.upper()}: {output_path}')

        result = {
            'exported_files': exported_files,
            'metadata': metadata,
            'subtitle_count': len(supervisions),
            'is_youtube_format': is_youtube_format,
            'original_transcript_path': original_transcript_path,
        }

        return result

    async def process_youtube_url(
        self, url: str, output_dir: Optional[str] = None, output_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing a YouTube URL

        Args:
            url: YouTube URL to process
            output_dir: Directory to save output files (optional)
            output_formats: List of output formats (optional, uses instance default)

        Returns:
            Dictionary containing results and exported file paths
        """
        if output_formats:
            self.output_formats = output_formats

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
