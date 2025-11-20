"""Gemini 2.5 Pro transcription module with config-driven architecture."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from lattifai.config import TranscriptionConfig
from lattifai.transcription.base import BaseTranscriber
from lattifai.transcription.prompts import get_prompt_loader


class GeminiTranscriber(BaseTranscriber):
    """
    Gemini 2.5/3 Pro audio transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    """

    # Transcriber metadata
    name = "Gemini"
    file_suffix = ".md"

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
        self._client: Optional[genai.Client] = None
        self._generation_config: Optional[GenerateContentConfig] = None
        self._system_prompt: Optional[str] = None

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
        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for: {url}")

        try:
            contents = Part.from_uri(file_uri=url, mime_type="video/*")
            return await self._run_generation(contents, source=url)

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
        media_file_path = Path(media_file_path)

        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for file: {media_file_path}")

        try:
            client = self._get_client()

            # Upload audio file
            if self.config.verbose:
                self.logger.info("📤 Uploading audio file to Gemini...")
            media_file = client.files.upload(path=str(media_file_path))

            contents = Part.from_uri(file_uri=media_file.uri, mime_type=media_file.mime_type)
            return await self._run_generation(contents, source=str(media_file_path), client=client)

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    def _get_transcription_prompt(self) -> str:
        """Get (and cache) transcription system prompt from prompts module."""
        if self._system_prompt is not None:
            return self._system_prompt

        # Load prompt from prompts/gemini/transcription_gem.txt
        prompt_loader = get_prompt_loader()
        base_prompt = prompt_loader.get_gemini_transcription_prompt()

        # Add language-specific instruction if configured
        if self.config.language:
            base_prompt += f"\n\n* Use {self.config.language} language for transcription."

        self._system_prompt = base_prompt
        return self._system_prompt

    def get_gem_info(self) -> dict:
        """Get information about the Gem being used."""
        return {
            "gem_name": "Media Transcription Gem",
            "gem_url": self.GEM_URL,
            "model": self.config.model_name,
            "description": "Specialized Gem for media content transcription",
        }

    def _build_result(self, transcript: str, output_file: Path) -> dict:
        """Augment the base result with Gemini-specific metadata."""
        base_result = super()._build_result(transcript, output_file)
        base_result.update({"model": self.config.model_name, "language": self.config.language})
        return base_result

    def _get_client(self) -> genai.Client:
        """Lazily create the Gemini client when first needed."""
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key is required for transcription")

        if self._client is None:
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    def _get_generation_config(self) -> GenerateContentConfig:
        """Lazily build the generation config since it rarely changes."""
        if self._generation_config is None:
            self._generation_config = GenerateContentConfig(
                system_instruction=self._get_transcription_prompt(),
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=-1,
                    # thinking_level="high",  # "low", "medium"
                ),
            )
        return self._generation_config

    async def _run_generation(
        self,
        contents: Part,
        *,
        source: str,
        client: Optional[genai.Client] = None,
    ) -> str:
        """
        Shared helper for sending generation requests and handling the response.
        """
        client = client or self._get_client()
        config = self._get_generation_config()

        if self.config.verbose:
            self.logger.info(f"🔄 Sending transcription request to {self.config.model_name} ({source})...")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=config,
            ),
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini API")

        transcript = response.text.strip()

        if self.config.verbose:
            self.logger.info(f"✅ Transcription completed ({source}): {len(transcript)} characters")

        return transcript
