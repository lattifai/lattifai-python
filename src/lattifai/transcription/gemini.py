"""Gemini 2.5 Pro transcription module with config-driven architecture."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from lattifai.config import TranscriptionConfig


class GeminiTranscriber:
    """
    Gemini 2.5 Pro audio transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    """

    # The specific Gem URL
    GEM_URL = "https://gemini.google.com/gem/1870ly7xvW2hU_umtv-LedGsjywT0sQiN"

    def __init__(
        self,
        transcription_config: Optional[TranscriptionConfig] = None,
    ):
        """
        Initialize Gemini transcriber.

        Args:
            transcription_config: Transcription configuration. If None, uses default.
        """
        # Initialize config with default if not provided
        if transcription_config is None:
            transcription_config = TranscriptionConfig()

        self.config = transcription_config
        self.logger = logging.getLogger(__name__)

        # Warn if API key not available
        if not self.config.gemini_api_key:
            self.logger.warning(
                "⚠️ Gemini API key not provided. API key will be required when calling transcription methods."
            )

    async def __call__(self, youtube_url: str) -> str:
        """Main entry point for transcription."""
        return await self.transcribe_url(youtube_url)

    async def transcribe_url(self, url: str) -> str:
        """
        Transcribe audio from URL using Gemini 2.5 Pro.

        Args:
            url: URL to transcribe (e.g., YouTube)

        Returns:
            Transcribed text

        Raises:
            ValueError: If API key not provided
            RuntimeError: If transcription fails
        """
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key is required for transcription")

        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for: {url}")

        try:
            # Initialize client
            client = genai.Client(api_key=self.config.gemini_api_key)

            # Load prompt (using simple transcription prompt for now)
            system_prompt = self._get_transcription_prompt()

            # Generate transcription
            if self.config.verbose:
                self.logger.info(f"🔄 Sending request to {self.config.model_name}...")

            config = GenerateContentConfig(
                system_instruction=system_prompt,
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=-1,
                ),
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.config.model_name,
                    contents=Part.from_uri(file_uri=url, mime_type="video/*"),
                    config=config,
                ),
            )

            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            transcript = response.text.strip()

            if self.config.verbose:
                self.logger.info(f"✅ Transcription completed: {len(transcript)} characters")

            return transcript

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    async def transcribe_file(self, media_file_path: Union[str, Path]) -> str:
        """
        Transcribe audio/video from local file using Gemini 2.5 Pro.

        Args:
            media_file_path: Path to local audio/video file

        Returns:
            Transcribed text

        Raises:
            ValueError: If API key not provided
            RuntimeError: If transcription fails
        """
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key is required for transcription")

        media_file_path = Path(media_file_path)

        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for file: {media_file_path}")

        try:
            # Initialize client
            client = genai.Client(api_key=self.config.gemini_api_key)

            # Load prompt
            system_prompt = self._get_transcription_prompt()

            # Upload audio file
            if self.config.verbose:
                self.logger.info("📤 Uploading audio file to Gemini...")
            media_file = client.files.upload(path=str(media_file_path))

            # Generate transcription
            if self.config.verbose:
                self.logger.info(f"🔄 Sending transcription request to {self.config.model_name}...")

            config = GenerateContentConfig(
                system_instruction=system_prompt,
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=-1,
                ),
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.config.model_name,
                    contents=Part.from_uri(file_uri=media_file.uri, mime_type=media_file.mime_type),
                    config=config,
                ),
            )

            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            transcript = response.text.strip()

            if self.config.verbose:
                self.logger.info(f"✅ Transcription completed: {len(transcript)} characters")

            return transcript

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    async def transcribe(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> dict:
        """
        Transcribe audio file (implements BaseTranscriber interface).

        Args:
            audio_path: Path to audio file or URL
            output_dir: Directory for output files

        Returns:
            dict: Transcription results with metadata
        """
        audio_path_str = str(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if it's a URL or file path
        if audio_path_str.startswith(("http://", "https://")):
            transcript = await self.transcribe_url(audio_path_str)
        else:
            transcript = await self.transcribe_file(audio_path)

        # Save transcript to file if output_dir provided
        output_file = output_dir / f"{Path(audio_path).stem}_transcript.txt"
        output_file.write_text(transcript, encoding="utf-8")

        return {
            "transcript": transcript,
            "output_file": str(output_file),
            "model": self.config.model_name,
            "language": self.config.language,
        }

    def _get_transcription_prompt(self) -> str:
        """Get transcription system prompt."""
        base_prompt = """You are an expert audio transcription assistant.
Your task is to accurately transcribe the audio content from the provided media file.

Guidelines:
- Transcribe exactly what is spoken
- Include punctuation and proper formatting
- Preserve speaker changes if multiple speakers
- Indicate unclear audio with [inaudible]
- Do NOT add commentary or interpretation
"""
        if self.config.language:
            base_prompt += f"\n- Transcribe in {self.config.language} language"

        return base_prompt

    def get_gem_info(self) -> dict:
        """Get information about the Gem being used."""
        return {
            "gem_name": "Audio Transcription Gem",
            "gem_url": self.GEM_URL,
            "model": self.config.model_name,
            "description": "Specialized Gem for media content transcription",
        }
