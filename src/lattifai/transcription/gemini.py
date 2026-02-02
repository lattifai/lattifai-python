"""Gemini 2.5 Pro transcription module with config-driven architecture."""

import asyncio
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from google import genai
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.transcription.base import BaseTranscriber
from lattifai.transcription.prompts import get_prompt_loader


class GeminiTranscriber(BaseTranscriber):
    """
    Gemini 2.5/3 Pro audio transcription with config-driven architecture.

    Uses TranscriptionConfig for all behavioral settings.
    """

    # Transcriber metadata
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
        super().__init__(config=transcription_config)

        self._client: Optional[genai.Client] = None
        self._generation_config: Optional[GenerateContentConfig] = None
        self._system_prompt: Optional[str] = None

        # Warn if API key not available
        if not self.config.gemini_api_key:
            self.logger.warning(
                "⚠️ Gemini API key not provided. API key will be required when calling transcription methods."
            )

    @property
    def name(self) -> str:
        """Human-readable name of the transcriber."""
        return f"{self.config.model_name}"

    async def transcribe_url(self, url: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio from URL using Gemini 2.5 Pro.

        Args:
            url: URL to transcribe (e.g., YouTube)
            language: Optional language code for transcription (overrides config)

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

    async def transcribe_file(self, media_file: Union[str, Path, AudioData], language: Optional[str] = None) -> str:
        """
        Transcribe audio/video from local file using Gemini 2.5 Pro.

        Args:
            media_file: Path to local audio/video file
            language: Optional language code for transcription (overrides config)

        Returns:
            Transcribed text

        Raises:
            ValueError: If API key not provided
            RuntimeError: If transcription fails
        """
        media_file = str(media_file)

        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for file: {media_file}")

        try:
            client = self._get_client()

            # Upload audio file
            if self.config.verbose:
                self.logger.info("📤 Uploading audio file to Gemini...")
            media_file = client.files.upload(file=media_file)

            contents = Part.from_uri(file_uri=media_file.uri, mime_type=media_file.mime_type)
            return await self._run_generation(contents, source=media_file, client=client)

        except ImportError:
            raise RuntimeError("Google GenAI SDK not installed. Please install with: pip install google-genai")
        except Exception as e:
            self.logger.error(f"Gemini transcription failed: {str(e)}")
            raise RuntimeError(f"Gemini transcription failed: {str(e)}")

    def transcribe_numpy(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        language: Optional[str] = None,
    ) -> Union[Supervision, List[Supervision]]:
        """
        Transcribe audio from a numpy array (or list of arrays) and return Supervision.

        Note: Gemini API does not support word-level alignment. The returned
        Supervision will contain only the full transcription text without alignment.

        Args:
            audio: Audio data as numpy array (shape: [samples]),
                   or a list of such arrays for batch processing.
            language: Optional language code for transcription.

        Returns:
            Supervision object (or list of Supervision objects) with transcription text (no alignment).

        Raises:
            ValueError: If API key not provided
            RuntimeError: If transcription fails
        """
        # Handle batch processing
        if isinstance(audio, list):
            return [self.transcribe_numpy(arr, language=language) for arr in audio]

        audio_array = audio
        # Use default sample rate of 16000 Hz
        sample_rate = 16000

        if self.config.verbose:
            self.logger.info(f"🎤 Starting Gemini transcription for numpy array (sample_rate={sample_rate})")

        # Ensure audio is in the correct shape
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(1, -1)
        elif audio_array.ndim > 2:
            raise ValueError(f"Audio array must be 1D or 2D, got shape {audio_array.shape}")

        # Save numpy array to temporary file
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Transpose to (samples, channels) for soundfile
            sf.write(tmp_file.name, audio_array.T, sample_rate)
            tmp_path = Path(tmp_file.name)

        try:
            # Transcribe using simple ASR prompt
            import asyncio

            transcript = asyncio.run(self._transcribe_with_simple_prompt(tmp_path, language=language))

            # Create Supervision object from transcript
            duration = audio_array.shape[-1] / sample_rate
            supervision = Supervision(
                id="gemini-transcription",
                recording_id="numpy-array",
                start=0.0,
                duration=duration,
                text=transcript,
                speaker=None,
                alignment=None,  # Gemini does not provide word-level alignment
            )

            return supervision

        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    async def _transcribe_with_simple_prompt(self, media_file: Path, language: Optional[str] = None) -> str:
        """
        Transcribe audio using a simple ASR prompt instead of complex instructions.

        Args:
            media_file: Path to audio file
            language: Optional language code

        Returns:
            Transcribed text
        """
        client = self._get_client()

        # Upload audio file
        if self.config.verbose:
            self.logger.info("📤 Uploading audio file to Gemini...")
        uploaded_file = client.files.upload(file=str(media_file))

        # Simple ASR prompt
        system_prompt = "Transcribe the audio."
        if language:
            system_prompt = f"Transcribe the audio in {language}."

        # Create simple generation config
        simple_config = GenerateContentConfig(
            system_instruction=system_prompt,
            response_modalities=["TEXT"],
        )

        contents = Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=simple_config,
            ),
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini API")

        transcript = response.text.strip()

        if self.config.verbose:
            self.logger.info(f"✅ Transcription completed: {len(transcript)} characters")

        return transcript

    def _get_transcription_prompt(self) -> str:
        """Get (and cache) transcription system prompt.

        Priority:
        1. Custom prompt from config.prompt (file path or text)
        2. Default prompt from prompts/gemini/transcription_gem.txt
        """
        if self._system_prompt is not None:
            return self._system_prompt

        # Check for custom prompt
        if self.config.prompt:
            prompt_path = Path(self.config.prompt)
            if prompt_path.exists() and prompt_path.is_file():
                # Load from file
                base_prompt = prompt_path.read_text(encoding="utf-8").strip()
                if self.config.verbose:
                    self.logger.info(f"📝 Using custom prompt from file: {prompt_path}")
            else:
                # Use as direct text
                base_prompt = self.config.prompt
                if self.config.verbose:
                    self.logger.info("📝 Using custom prompt text")
        else:
            # Load default prompt from prompts/gemini/transcription_gem.txt
            prompt_loader = get_prompt_loader()
            base_prompt = prompt_loader.get_gemini_transcription_prompt()

        # Add language-specific instruction if configured
        if self.config.language:
            base_prompt += f"\n\n* Use {self.config.language} language for transcription."

        # Add media description context if available
        if self.config.description:
            base_prompt += f"\n\n## Media Context\n\n{self.config.description}"

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
            # Only include thinking_config if thinking mode is enabled
            thinking_config = None
            if self.config.thinking:
                thinking_config = ThinkingConfig(
                    include_thoughts=self.config.include_thoughts,
                    thinking_budget=-1,
                )

            self._generation_config = GenerateContentConfig(
                system_instruction=self._get_transcription_prompt(),
                response_modalities=["TEXT"],
                thinking_config=thinking_config,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
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

        # Extract content based on include_thoughts setting
        if self.config.include_thoughts:
            transcript = self._extract_with_thoughts(response)
        else:
            if not response.text:
                raise RuntimeError("Empty response from Gemini API")
            transcript = response.text.strip()

        if self.config.verbose:
            self.logger.info(f"✅ Transcription completed ({source}): {len(transcript)} characters")

        return transcript

    def _extract_with_thoughts(self, response) -> str:
        """Extract response content including thinking process and metadata."""
        output_parts = []
        thoughts = []
        text_parts = []

        # Iterate through all parts in the response
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                    # This is a thinking part
                    if hasattr(part, "text") and part.text:
                        thoughts.append(part.text)
                elif hasattr(part, "text") and part.text:
                    # This is a regular text part
                    text_parts.append(part.text)

        # Extract metadata
        metadata_lines = self._extract_response_metadata(response)
        if metadata_lines:
            output_parts.append("---")
            output_parts.extend(metadata_lines)
            output_parts.append("---\n")

        # Format output with thoughts section if present
        if thoughts:
            output_parts.append("<thinking>")
            output_parts.extend(thoughts)
            output_parts.append("</thinking>\n")

        output_parts.extend(text_parts)

        result = "\n".join(output_parts).strip()
        if not result:
            raise RuntimeError("Empty response from Gemini API")

        return result

    def _extract_response_metadata(self, response) -> list:
        """Extract useful metadata from Gemini response as YAML frontmatter."""
        lines = []

        # Model version
        if hasattr(response, "model_version") and response.model_version:
            lines.append(f"model_version: {response.model_version}")

        # Usage metadata (token counts)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                lines.append(f"prompt_tokens: {usage.prompt_token_count}")
            if hasattr(usage, "candidates_token_count"):
                lines.append(f"output_tokens: {usage.candidates_token_count}")
            if hasattr(usage, "total_token_count"):
                lines.append(f"total_tokens: {usage.total_token_count}")
            # Thinking tokens if available
            if hasattr(usage, "thoughts_token_count") and usage.thoughts_token_count:
                lines.append(f"thinking_tokens: {usage.thoughts_token_count}")

        # Candidate-level metadata
        if response.candidates:
            candidate = response.candidates[0]

            # Finish reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                lines.append(f"finish_reason: {candidate.finish_reason}")

            # Average log probability (confidence indicator)
            if hasattr(candidate, "avg_logprobs") and candidate.avg_logprobs is not None:
                lines.append(f"avg_logprobs: {candidate.avg_logprobs:.4f}")

            # Citation metadata
            if hasattr(candidate, "citation_metadata") and candidate.citation_metadata:
                citations = getattr(candidate.citation_metadata, "citations", [])
                if citations:
                    lines.append("citations:")
                    for cite in citations:
                        uri = getattr(cite, "uri", "")
                        start = getattr(cite, "start_index", "")
                        end = getattr(cite, "end_index", "")
                        if uri:
                            lines.append(f"  - uri: {uri}")
                            if start or end:
                                lines.append(f"    range: [{start}, {end}]")

        return lines

    def write(self, transcript: str, output_file: Path, encoding: str = "utf-8", cache_event: bool = True) -> Path:
        """
        Persist transcript text to disk and return the file path.
        """
        if isinstance(output_file, str):
            output_file = Path(output_file)
        output_file.write_text(transcript, encoding=encoding)
        return output_file
