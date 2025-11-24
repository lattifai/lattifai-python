"""Base client classes for LattifAI SDK."""

import tempfile
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Union  # noqa: F401

import httpx
from lhotse.utils import Pathlike

from lattifai.audio2 import AudioData

from .config import ClientConfig

# Import from errors module for consistency
from .errors import APIError, ConfigurationError

if TYPE_CHECKING:
    from .config import AlignmentConfig, SubtitleConfig, TranscriptionConfig


class LattifAIClientMixin:
    """
    Mixin class providing shared functionality for LattifAI clients.

    This mixin contains common logic for transcription and downloading that is
    used by both synchronous and asynchronous client implementations.
    """

    # Shared docstring templates for class, __init__, alignment, and youtube methods
    _CLASS_DOC = """
    {sync_or_async} LattifAI client for audio/video-subtitle alignment.

    This client provides {sync_or_async_lower} methods for aligning audio/video files with subtitle/transcript
    text using the Lattice-1 forced alignment model. It supports multiple subtitle formats
    (SRT, VTT, ASS, TXT) and provides word-level alignment with configurable sentence splitting.

    The client uses a config-driven architecture with four main configuration objects:
    - ClientConfig: API connection settings (API key, base URL, timeout, retries)
    - AlignmentConfig: Model and alignment behavior settings
    - SubtitleConfig: Subtitle I/O format and processing settings
    - TranscriptionConfig: Transcription service settings (optional, for YouTube workflow)

    Example:
        >>> from lattifai import {client_class}, ClientConfig
        >>>
        >>> # Initialize with default settings
        >>> client = {client_class}()
        >>>
        >>> # Or with custom configuration
        >>> config = ClientConfig(api_key="your-api-key")
        >>> client = {client_class}(config=config)
        >>>
        >>> # Perform alignment
        >>> {await_keyword}alignments, output_path = {await_keyword}client.alignment(
        ...     input_media_path="audio.wav",
        ...     input_subtitle_path="subtitle.srt",
        ...     output_subtitle_path="aligned.srt"
        ... )

    Attributes:
        aligner: Lattice1Aligner instance for performing forced alignment{async_note}
        subtitler: Subtitler instance for reading/writing subtitle files
        transcriber: Optional transcriber instance for audio transcription{transcriber_note}
    """

    _INIT_DOC = """
        Initialize {client_class} {sync_or_async_lower} client.

        Args:
            client_config: Client configuration for API connection settings. If None, uses defaults
                          (reads API key from LATTIFAI_API_KEY environment variable).
            alignment_config: Alignment {config_desc}
                            If None, uses {default_desc}.
            subtitle_config: Subtitle I/O configuration for format handling and processing.
                           If None, uses default settings{subtitle_note}.
            transcription_config: Transcription service configuration{transcription_note}.

        Raises:
            ConfigurationError: If API key is not provided {api_key_source}.
        """

    _ALIGNMENT_DOC = """
        Perform {async_prefix}forced alignment on audio and subtitle/text.

        This {async_word}method aligns subtitle text with audio by finding the precise timing of {timing_desc}
        and subtitle segment. {concurrency_note}

        The alignment process consists of five steps:
        1. Parse the input subtitle file into segments{async_suffix1}
        2. Generate a lattice graph from subtitle text{async_suffix2}
        3. Search the lattice using audio features{async_suffix3}
        4. Decode results to extract word-level timings{async_suffix4}
        5. Export aligned subtitles (if output path provided{async_suffix5})

        Args:
            input_media_path: Path to audio/video file (WAV, MP3, FLAC, MP4, etc.). Must be readable by ffmpeg.
            input_subtitle_path: Path to subtitle or plain text file to align with audio.
            input_subtitle_format: Input subtitle format ('srt', 'vtt', 'ass', 'txt'). If None, {format_default}
                   from file extension or uses config default.
            split_sentence: Enable automatic sentence re-splitting for better alignment accuracy.
                          If None, uses config default (typically False).
            output_subtitle_path: Optional path to write aligned subtitle file. If provided,
                                exports results{export_note}.

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information{timing_note}
                - Output subtitle path (same as input parameter, or None if not provided)

        Raises:
            SubtitleProcessingError: If subtitle file cannot be parsed or output cannot be written.
            LatticeEncodingError: If lattice graph generation fails (invalid text format).
            AlignmentError: If audio alignment fails (audio processing or model inference error).
            LatticeDecodingError: If lattice decoding fails (invalid results from model).

        Example:
            >>> {example_imports}
            >>> {example_code}
        """

    _YOUTUBE_METHOD_DOC = """
        Download and align YouTube video with subtitles or transcription.

        This end-to-end method handles the complete YouTube alignment workflow:
        1. Downloads media from YouTube in specified format
        2. Downloads subtitles OR transcribes audio (based on config)
        3. Performs forced alignment with Lattice-1 model
        4. Exports aligned subtitles

        Args:
            url: YouTube video URL (e.g., https://youtube.com/watch?v=VIDEO_ID)
            output_dir: Directory for downloaded files. If None, uses temporary directory.
            media_format: Media format to download (mp3, mp4, wav, etc.). If None, uses config default.
            subtitle_lang: Specific subtitle language to download (e.g., 'en', 'zh'). If None, downloads all.
            force_overwrite: Skip confirmation prompts and overwrite existing files.
            output_subtitle_path: Path for aligned subtitle output. If None, auto-generates.
            **alignment_kwargs: Additional arguments passed to alignment() method.

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information
                - Output subtitle path

        Raises:
            ValueError: If transcription is requested but transcriber not configured.
            RuntimeError: If download or transcription fails.
            SubtitleProcessingError: If subtitle processing fails.
            AlignmentError: If alignment fails.

        Example:
            >>> from lattifai import {client_class}
            >>> from lattifai.config import TranscriptionConfig
            >>>
            >>> # With YouTube subtitles
            >>> client = {client_class}()
            >>> {await_keyword}alignments, path = {await_keyword}client.youtube(
            ...     url="https://youtube.com/watch?v=VIDEO_ID",
            ...     output_dir="./downloads"
            ... )
            >>>
            >>> # With Gemini transcription
            >>> config = TranscriptionConfig(gemini_api_key="YOUR_KEY")
            >>> client = {client_class}(transcription_config=config)
            >>> client.subtitler.config.use_transcription = True
            >>> {await_keyword}alignments, path = {await_keyword}client.youtube(
            ...     url="https://youtube.com/watch?v=VIDEO_ID"
            ... )
        """

    def _init_configs(
        self,
        client_config: Optional["ClientConfig"],
        alignment_config: Optional["AlignmentConfig"],
        subtitle_config: Optional["SubtitleConfig"],
        transcription_config: Optional["TranscriptionConfig"],
    ) -> tuple:
        """Initialize all configs with defaults if not provided."""
        from .config import AlignmentConfig, ClientConfig, SubtitleConfig

        if client_config is None:
            client_config = ClientConfig()
        if alignment_config is None:
            alignment_config = AlignmentConfig()
        if subtitle_config is None:
            subtitle_config = SubtitleConfig()

        return client_config, alignment_config, subtitle_config, transcription_config

    def _init_shared_components(
        self,
        subtitle_config: Optional["SubtitleConfig"],
        transcription_config: Optional["TranscriptionConfig"],
    ) -> None:
        """Initialize shared components used by both sync and async clients."""
        from .config import SubtitleConfig
        from .subtitle import Subtitler

        # subtitler IO
        subtitle_config = subtitle_config or SubtitleConfig()
        self.subtitler = Subtitler(config=subtitle_config)

        # transcriber (optional, lazy loaded when needed)
        self.transcription_config = transcription_config
        self._transcriber = None

        # downloader (lazy loaded when needed)
        self._downloader = None

    @property
    def transcriber(self):
        """Lazy load transcriber based on config."""
        if self._transcriber is None and self.transcription_config:
            from .transcription import create_transcriber

            self._transcriber = create_transcriber(transcription_config=self.transcription_config)
        return self._transcriber

    @property
    def downloader(self):
        """Lazy load YouTube downloader."""
        if self._downloader is None:
            from .workflow.youtube import YouTubeDownloader

            self._downloader = YouTubeDownloader()
        return self._downloader

    def _prepare_youtube_output_dir(self, output_dir: Optional["Pathlike"]) -> Path:
        """Prepare and return output directory for YouTube downloads."""
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "lattifai_youtube"
        else:
            output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _determine_media_format(self, media_format: Optional[str]) -> str:
        """Determine media format from parameter or config."""
        return media_format or "mp3"

    def _generate_output_subtitle_path(
        self, output_subtitle_path: Optional["Pathlike"], media_file: str, output_dir: Path
    ) -> Path:
        """Generate output subtitle path if not provided."""
        if not output_subtitle_path:
            media_name = Path(media_file).stem
            output_format = self.subtitler.config.output_format or "srt"
            output_subtitle_path = output_dir / f"{media_name}_LattifAI.{output_format}"
        return Path(output_subtitle_path)

    def _validate_transcription_setup(self) -> None:
        """Validate that transcription is properly configured if requested."""
        if self.subtitler.config.use_transcription and not self.transcriber:
            raise ValueError(
                "Transcription requested but transcriber not configured. "
                "Provide TranscriptionConfig with valid API key."
            )

    async def _download_media(
        self,
        url: str,
        output_dir: Path,
        media_format: str,
        force_overwrite: bool,
    ) -> str:
        """Download media from YouTube (async implementation)."""
        import colorful

        print(colorful.cyan("📥 Downloading media from YouTube..."))
        media_file = await self.downloader.download_media(
            url=url,
            output_dir=str(output_dir),
            media_format=media_format,
            force_overwrite=force_overwrite,
        )
        print(colorful.green(f"    ✓ Media downloaded: {media_file}"))
        return media_file

    def _download_media_sync(
        self,
        url: str,
        output_dir: Path,
        media_format: str,
        force_overwrite: bool,
    ) -> str:
        """Download media from YouTube (sync wrapper)."""
        import asyncio

        return asyncio.run(self._download_media(url, output_dir, media_format, force_overwrite))

    def _get_or_create_subtitles(
        self,
        url: str,
        output_dir: Path,
        media_file: Union[str, Path, AudioData],
        force_overwrite: bool,
        subtitle_lang: Optional[str],
        is_async: bool = False,
    ) -> Union[str, Awaitable[str]]:
        """
        Get subtitles by downloading or transcribing.

        Args:
            url: YouTube video URL
            output_dir: Output directory for subtitle file
            media_file: Media file path (used to generate subtitle filename)
            force_overwrite: Force overwrite existing files
            subtitle_lang: Subtitle language to download
            is_async: If True, returns coroutine; if False, runs synchronously

        Returns:
            Subtitle file path (str) or coroutine that returns str
        """
        import asyncio

        import colorful

        async def _async_impl():
            # First check if subtitle input_path is already provided
            if self.subtitler.config.input_path:
                subtitle_path = Path(self.subtitler.config.input_path)
                if subtitle_path.exists():
                    print(colorful.green(f"📄 Using provided subtitle file: {subtitle_path}"))
                    return str(subtitle_path)
                else:
                    raise FileNotFoundError(f"Provided subtitle path does not exist: {subtitle_path}")

            if self.subtitler.config.use_transcription:
                # Transcription mode: use Transcriber to transcribe
                self._validate_transcription_setup()

                # Generate transcript file path
                transcript_file = (
                    output_dir / f"{Path(str(media_file)).stem}_{self.transcriber.name}{self.transcriber.file_suffix}"
                )

                # Check if transcript file already exists
                if transcript_file.exists() and not force_overwrite:
                    from .workflow.file_manager import FileExistenceManager

                    choice = await asyncio.to_thread(
                        FileExistenceManager.prompt_file_selection,
                        file_type=f"{self.transcriber.name} transcript",
                        files=[str(transcript_file)],
                        operation="transcribe",
                        enable_gemini=False,
                    )

                    if choice == "cancel":
                        raise RuntimeError("Transcription cancelled by user")
                    elif choice == "use" or choice == str(transcript_file):
                        # User chose to use existing file (handles both "use" and file path)
                        return str(transcript_file)
                    # elif choice == "overwrite": continue to transcribe below

                print(colorful.cyan(f"🎤 Transcribing media with {self.transcriber.name}..."))
                if self.transcriber.supports_url:
                    transcription = await self.transcriber.transcribe(url)
                else:
                    transcription = await self.transcriber.transcribe_file(media_file)

                await asyncio.to_thread(self.transcriber.write, transcription, transcript_file, encoding="utf-8")
                subtitle_file = str(transcript_file)
                print(colorful.green(f"         ✓ Transcription completed: {subtitle_file}"))
            else:
                # Download YouTube subtitles
                subtitle_file = await self.downloader.download_subtitles(
                    url=url,
                    output_dir=str(output_dir),
                    force_overwrite=force_overwrite,
                    subtitle_lang=subtitle_lang,
                    enable_gemini_option=True,
                )
                if not subtitle_file:
                    raise RuntimeError("No subtitle file available. Either download subtitles or enable transcription.")

            return subtitle_file

        if is_async:
            return _async_impl()
        else:
            return asyncio.run(_async_impl())


class BaseAPIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(
        self,
        config: ClientConfig,
    ) -> None:
        if config.api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )

        self._api_key = config.api_key
        self._base_url = config.base_url
        self._timeout = config.timeout
        self._max_retries = config.max_retries

        headers = {
            "User-Agent": "LattifAI/Python",
            "Authorization": f"Bearer {self._api_key}",
        }
        if config.default_headers:
            headers.update(config.default_headers)
        self._default_headers = headers


class SyncAPIClient(BaseAPIClient):
    """Synchronous API client."""

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._default_headers,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request."""
        return self._client.request(method=method, url=url, json=json, **kwargs)

    def post(self, api_endpoint: str, *, json: Optional[Dict[str, Any]] = None, **kwargs) -> httpx.Response:
        """Make a POST request to the specified API endpoint."""
        return self._request("POST", api_endpoint, json=json, **kwargs)


class AsyncAPIClient(BaseAPIClient):
    """Asynchronous API client."""

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._default_headers,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request."""
        return await self._client.request(method=method, url=url, json=json, files=files, **kwargs)

    async def post(
        self,
        api_endpoint: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make a POST request to the specified API endpoint."""
        return await self._request("POST", api_endpoint, json=json, files=files, **kwargs)
