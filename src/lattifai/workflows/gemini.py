"""
Gemini 2.5 Pro transcription module
"""

import asyncio
from typing import Optional

# Import Google GenAI SDK
from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from .base import setup_workflow_logger
from .prompts import get_prompt_loader


class GeminiTranscriber:
    """Gemini 2.5 Pro audio transcription using the specified Gem

    Configuration (in __init__):
        - api_key: Gemini API key (required)

    Runtime parameters (in __call__):
        - youtube_url: YouTube URL to transcribe
    """

    # The specific Gem URL provided by the user
    GEM_URL = "https://gemini.google.com/gem/1870ly7xvW2hU_umtv-LedGsjywT0sQiN"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = setup_workflow_logger("gemini")
        self.prompt_loader = get_prompt_loader()

        if not self.api_key:
            self.logger.warning(
                "⚠️ Gemini API key not provided. API key will be required when calling transcription methods."
            )

    async def __call__(self, youtube_url: str) -> str:
        """Main entry point for transcription"""
        return await self.transcribe_url(youtube_url)

    async def transcribe_url(self, youtube_url: str) -> str:
        """
        Transcribe audio from YouTube URL using Gemini 2.5 Pro Gem

        Args:
            youtube_url: YouTube URL to transcribe

        Returns:
            Transcribed text
        """
        if not self.api_key:
            raise ValueError("Gemini API key is required for transcription")

        self.logger.info(f"🎤 Starting Gemini transcription for: {youtube_url}")

        try:
            # Initialize client
            client = genai.Client(api_key=self.api_key)

            # Load prompt from Gem configuration
            system_prompt = self.prompt_loader.get_gemini_transcription_prompt()

            # Generate transcription with extended thinking
            self.logger.info("🔄 Sending request to Gemini 2.5 Pro...")
            config = GenerateContentConfig(
                system_instruction=system_prompt,
                # Enable thinking by including it in response modalities
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=-1,
                ),
            )
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=Part.from_uri(file_uri=youtube_url, mime_type="video/*"),
                    config=config,
                ),
            )

            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            transcript = response.text.strip()

            self.logger.info(f"✅ Transcription completed: {len(transcript)} characters")
            return transcript

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    async def transcribe_file(self, media_file_path: str) -> str:
        """
        Transcribe audio/video from local file using Gemini 2.5 Pro

        Args:
            media_file_path: Path to local audio file

        Returns:
            Transcribed text
        """
        if not self.api_key:
            raise ValueError("Gemini API key is required for transcription")

        self.logger.info(f"🎤 Starting Gemini transcription for file: {media_file_path}")

        try:
            # Initialize client
            client = genai.Client(api_key=self.api_key)

            # Load prompt from Gem configuration
            system_prompt = self.prompt_loader.get_gemini_transcription_prompt()

            # Upload audio file
            self.logger.info("📤 Uploading audio file to Gemini...")
            media_file = client.files.upload(path=media_file_path)

            # Generate transcription with extended thinking
            # Note: For thinking mode, you may want to use 'gemini-2.0-flash-thinking-exp' or similar models
            self.logger.info("🔄 Sending transcription request...")
            config = GenerateContentConfig(
                system_instruction=system_prompt,
                # Enable thinking by including it in response modalities
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=-1,
                ),
            )
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=Part.from_uri(file_uri=media_file.uri, mime_type=media_file.mime_type),
                    config=config,
                ),
            )

            if not response.text:
                raise RuntimeError("Empty response from Gemini API")

            transcript = response.text.strip()

            self.logger.info(f"✅ Transcription completed: {len(transcript)} characters")
            return transcript

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    def get_gem_info(self) -> dict:
        """Get information about the Gem being used"""
        return {
            "gem_name": "Audio Transcription Gem",
            "gem_url": self.GEM_URL,
            "model": "Gemini 2.5 Pro",
            "description": "Specialized Gem for media content transcribe",
        }
