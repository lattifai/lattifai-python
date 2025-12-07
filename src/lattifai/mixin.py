"""Mixin class providing shared functionality for LattifAI clients."""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Optional, Union

import colorful
from lhotse.utils import Pathlike

from lattifai.audio2 import AudioData
from lattifai.caption import Caption
from lattifai.errors import CaptionProcessingError

if TYPE_CHECKING:
    from .config import AlignmentConfig, CaptionConfig, ClientConfig, DiarizationConfig, TranscriptionConfig


class LattifAIClientMixin:
    """
    Mixin class providing shared functionality for LattifAI clients.

    This mixin contains common logic for transcription and downloading that is
    used by both synchronous and asynchronous client implementations.
    """

    # Shared docstring templates for class, __init__, alignment, and youtube methods
    _CLASS_DOC = """
    {sync_or_async} LattifAI client for audio/video-caption alignment.

    This client provides {sync_or_async_lower} methods for aligning audio/video files with caption/transcript
    text using the Lattice-1 forced alignment model. It supports multiple caption formats
    (SRT, VTT, ASS, TXT) and provides word-level alignment with configurable sentence splitting.

    The client uses a config-driven architecture with four main configuration objects:
    - ClientConfig: API connection settings (API key, base URL, timeout, retries)
    - AlignmentConfig: Model and alignment behavior settings
    - CaptionConfig: Caption I/O format and processing settings
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
        ...     input_media="audio.wav",
        ...     input_caption="caption.srt",
        ...     output_caption_path="aligned.srt"
        ... )

    Attributes:
        aligner: Lattice1Aligner instance for performing forced alignment{async_note}
        captioner: Captioner instance for reading/writing caption files
        transcriber: Optional transcriber instance for audio transcription{transcriber_note}
    """

    _INIT_DOC = """
        Initialize {client_class} {sync_or_async_lower} client.

        Args:
            client_config: Client configuration for API connection settings. If None, uses defaults
                          (reads API key from LATTIFAI_API_KEY environment variable).
            alignment_config: Alignment {config_desc}
                            If None, uses {default_desc}.
            caption_config: Caption I/O configuration for format handling and processing.
                           If None, uses default settings{caption_note}.
            transcription_config: Transcription service configuration{transcription_note}.

        Raises:
            ConfigurationError: If API key is not provided {api_key_source}.
        """

    _ALIGNMENT_DOC = """
        Perform {async_prefix}forced alignment on audio and caption/text.

        This {async_word}method aligns caption text with audio by finding the precise timing of {timing_desc}
        and caption segment. {concurrency_note}

        The alignment process consists of five steps:
        1. Parse the input caption file into segments{async_suffix1}
        2. Generate a lattice graph from caption text{async_suffix2}
        3. Search the lattice using audio features{async_suffix3}
        4. Decode results to extract word-level timings{async_suffix4}
        5. Export aligned captions (if output path provided{async_suffix5})

        Args:
            input_media: Path to audio/video file (WAV, MP3, FLAC, MP4, etc.). Must be readable by ffmpeg.
            input_caption: Path to caption or plain text file to align with audio.
            input_caption_format: Input caption format ('srt', 'vtt', 'ass', 'txt'). If None, {format_default}
                   from file extension or uses config default.
            split_sentence: Enable automatic sentence re-splitting for better alignment accuracy.
                          If None, uses config default (typically False).
            output_caption_path: Optional path to write aligned caption file. If provided,
                                exports results{export_note}.

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information{timing_note}
                - Output caption path (same as input parameter, or None if not provided)

        Raises:
            CaptionProcessingError: If caption file cannot be parsed or output cannot be written.
            LatticeEncodingError: If lattice graph generation fails (invalid text format).
            AlignmentError: If audio alignment fails (audio processing or model inference error).
            LatticeDecodingError: If lattice decoding fails (invalid results from model).

        Example:
            >>> {example_imports}
            >>> {example_code}
        """

    _YOUTUBE_METHOD_DOC = """
        Download and align YouTube video with captions or transcription.

        This end-to-end method handles the complete YouTube alignment workflow:
        1. Downloads media from YouTube in specified format
        2. Downloads captions OR transcribes audio (based on config)
        3. Performs forced alignment with Lattice-1 model
        4. Exports aligned captions

        Args:
            url: YouTube video URL (e.g., https://youtube.com/watch?v=VIDEO_ID)
            output_dir: Directory for downloaded files. If None, uses temporary directory.
            media_format: Media format to download (mp3, mp4, wav, etc.). If None, uses config default.
            source_lang: Specific caption language to download (e.g., 'en', 'zh'). If None, downloads all.
            force_overwrite: Skip confirmation prompts and overwrite existing files.
            output_caption_path: Path for aligned caption output. If None, auto-generates.
            **alignment_kwargs: Additional arguments passed to alignment() method.

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information
                - Output caption path

        Raises:
            ValueError: If transcription is requested but transcriber not configured.
            RuntimeError: If download or transcription fails.
            CaptionProcessingError: If caption processing fails.
            AlignmentError: If alignment fails.

        Example:
            >>> from lattifai import {client_class}
            >>> from lattifai.config import TranscriptionConfig
            >>>
            >>> # With YouTube captions
            >>> client = {client_class}()
            >>> {await_keyword}alignments, path = {await_keyword}client.youtube(
            ...     url="https://youtube.com/watch?v=VIDEO_ID",
            ...     output_dir="./downloads"
            ... )
            >>>
            >>> # With Gemini transcription
            >>> config = TranscriptionConfig(gemini_api_key="YOUR_KEY")
            >>> client = {client_class}(transcription_config=config)
            >>> {await_keyword}alignments, path = {await_keyword}client.youtube(
            ...     url="https://youtube.com/watch?v=VIDEO_ID",
            ...     use_transcription=True
            ... )
        """

    def _init_configs(
        self,
        alignment_config: Optional["AlignmentConfig"],
        transcription_config: Optional["TranscriptionConfig"],
        diarization_config: Optional["DiarizationConfig"] = None,
    ) -> tuple:
        """Initialize all configs with defaults if not provided."""
        from .config import AlignmentConfig, DiarizationConfig, TranscriptionConfig

        if alignment_config is None:
            alignment_config = AlignmentConfig()
        if transcription_config is None:
            transcription_config = TranscriptionConfig()
        if diarization_config is None:
            diarization_config = DiarizationConfig()

        from lattifai.utils import _resolve_model_path

        if transcription_config is not None:
            transcription_config.lattice_model_path = _resolve_model_path(alignment_config.model_name)

        # Set client_wrapper for all configs
        alignment_config.client_wrapper = self
        transcription_config.client_wrapper = self
        diarization_config.client_wrapper = self

        return alignment_config, transcription_config, diarization_config

    def _init_shared_components(
        self,
        transcription_config: Optional["TranscriptionConfig"],
    ) -> None:
        """Initialize shared components (transcriber, downloader)."""
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

    def _generate_output_caption_path(
        self, output_caption_path: Optional["Pathlike"], media_file: str, output_dir: Path
    ) -> Path:
        """Generate output caption path if not provided."""
        if not output_caption_path:
            media_name = Path(media_file).stem
            output_format = self.caption_config.output_format or "srt"
            output_caption_path = output_dir / f"{media_name}_LattifAI.{output_format}"
        return Path(output_caption_path)

    def _validate_transcription_setup(self) -> None:
        """Validate that transcription is properly configured if requested."""
        if not self.transcriber:
            raise ValueError(
                "Transcription requested but transcriber not configured. "
                "Provide TranscriptionConfig with valid API key."
            )

    def _read_caption(
        self,
        input_caption: Union[Pathlike, Caption],
        input_caption_format: Optional[str] = None,
        normalize_text: Optional[bool] = None,
        verbose: bool = True,
    ) -> Caption:
        """
        Read caption file or return Caption object directly.

        Args:
            input_caption: Path to caption file or Caption object
            input_caption_format: Optional format hint for parsing

        Returns:
            Caption object

        Raises:
            CaptionProcessingError: If caption cannot be read
        """
        if isinstance(input_caption, Caption):
            return input_caption

        try:
            if verbose:
                print(colorful.cyan(f"📖 Step 1: Reading caption file from {input_caption}"))
            caption = Caption.read(
                input_caption,
                format=input_caption_format,
                normalize_text=normalize_text if normalize_text is not None else self.caption_config.normalize_text,
            )
            diarization_file = Path(str(input_caption)).with_suffix(".SpkDiar")
            if diarization_file.exists():
                if verbose:
                    print(colorful.cyan(f"📖 Step 1b: Reading speaker diarization from {diarization_file}"))
                caption.read_speaker_diarization(diarization_file)
            events_file = Path(str(input_caption)).with_suffix(".AED")
            if events_file.exists():
                if verbose:
                    print(colorful.cyan(f"📖 Step 1c: Reading audio events from {events_file}"))
                from tgt import read_textgrid

                caption.audio_events = read_textgrid(events_file)

            if verbose:
                print(colorful.green(f"         ✓ Parsed {len(caption)} caption segments"))
            return caption
        except Exception as e:
            raise CaptionProcessingError(
                f"Failed to parse caption file: {input_caption}",
                caption_path=str(input_caption),
                context={"original_error": str(e)},
            )

    def _write_caption(
        self,
        caption: Caption,
        output_caption_path: Pathlike,
    ) -> Pathlike:
        """
        Write caption to file.

        Args:
            caption: Caption object to write
            output_caption_path: Output file path

        Returns:
            Path to written file

        Raises:
            CaptionProcessingError: If caption cannot be written
        """
        try:
            result = caption.write(
                output_caption_path,
                include_speaker_in_text=self.caption_config.include_speaker_in_text,
            )
            diarization_file = Path(str(output_caption_path)).with_suffix(".SpkDiar")
            if not diarization_file.exists() and caption.speaker_diarization:
                print(colorful.green(f"    Writing speaker diarization to: {diarization_file}"))
                caption.write_speaker_diarization(diarization_file)

            print(colorful.green(f"🎉🎉🎉🎉🎉 Caption file written to: {output_caption_path}"))
            return result
        except Exception as e:
            raise CaptionProcessingError(
                f"Failed to write output file: {output_caption_path}",
                caption_path=str(output_caption_path),
                context={"original_error": str(e)},
            )

    async def _download_media(
        self,
        url: str,
        output_dir: Path,
        media_format: str,
        force_overwrite: bool,
    ) -> str:
        """Download media from YouTube (async implementation)."""
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

    def _transcribe(
        self,
        media_file: Union[str, Path, AudioData],
        source_lang: Optional[str],
        is_async: bool = False,
    ) -> Caption:
        """
        Get captions by downloading or transcribing.

        Args:
            url: YouTube video URL
            output_dir: Output directory for caption file
            media_file: Media file path (used to generate caption filename)
            force_overwrite: Force overwrite existing files
            source_lang: Caption language to download
            is_async: If True, returns coroutine; if False, runs synchronously

        Returns:
            Caption file path (str) or coroutine that returns str
        """
        import asyncio

        async def _async_impl():
            # Transcription mode: use Transcriber to transcribe
            self._validate_transcription_setup()

            print(colorful.cyan(f"🎤 Transcribing({self.transcriber.name}) media: {str(media_file)} ..."))
            transcription = await self.transcriber.transcribe_file(media_file, language=source_lang)
            print(colorful.green("         ✓ Transcription completed."))

            if "Gemini" in self.transcriber.name:
                # write to temp file and use Caption read
                with tempfile.NamedTemporaryFile(suffix=self.transcriber.file_suffix, delete=True) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    await asyncio.to_thread(
                        self.transcriber.write,
                        transcription,
                        tmp_path,
                        encoding="utf-8",
                    )
                    transcription = self._read_caption(
                        tmp_path, input_caption_format="gemini", normalize_text=False, verbose=False
                    )

            return transcription

        if is_async:
            return _async_impl()
        else:
            return asyncio.run(_async_impl())

    def _download_or_transcribe_caption(
        self,
        url: str,
        output_dir: Path,
        media_file: Union[str, Path, AudioData],
        force_overwrite: bool,
        source_lang: Optional[str],
        is_async: bool = False,
        use_transcription: bool = False,
    ) -> Union[Union[str, Caption], Awaitable[Union[str, Caption]]]:
        """
        Get captions by downloading or transcribing.
        Args:
            url: YouTube video URL
            output_dir: Output directory for caption file
            media_file: Media file path (used to generate caption filename)
            force_overwrite: Force overwrite existing files
            source_lang: Caption language to download
            is_async: If True, returns coroutine; if False, runs synchronously

        Returns:
            Caption file path (str) or coroutine that returns str
        """
        import asyncio

        from lattifai.workflow.youtube import TRANSCRIBE_CHOICE

        transcriber_name = self.transcriber.name

        async def _async_impl():
            # First check if caption input_path is already provided
            if self.caption_config.input_path:
                caption_path = Path(self.caption_config.input_path)
                if caption_path.exists():
                    print(colorful.green(f"📄 Using provided caption file: {caption_path}"))
                    return str(caption_path)
                else:
                    raise FileNotFoundError(f"Provided caption path does not exist: {caption_path}")

            # Generate transcript file path
            transcript_file = output_dir / f"{Path(str(media_file)).stem}_{self.transcriber.file_name}"

            if use_transcription:
                # Transcription mode: use Transcriber to transcribe
                self._validate_transcription_setup()

                # Check if transcript file already exists
                if transcript_file.exists() and not force_overwrite:
                    from .workflow.file_manager import FileExistenceManager

                    choice = await asyncio.to_thread(
                        FileExistenceManager.prompt_file_selection,
                        file_type=f"{transcriber_name} transcript",
                        files=[str(transcript_file)],
                        operation="transcribe",
                    )

                    if choice == "cancel":
                        raise RuntimeError("Transcription cancelled by user")
                    elif choice == "use" or choice == str(transcript_file):
                        # User chose to use existing file (handles both "use" and file path)
                        if "gemini" in transcriber_name.lower():
                            return str(transcript_file)

                        caption = self._read_caption(transcript_file, normalize_text=False)
                        caption.transcription = caption.supervisions
                        caption.supervisions = None
                        return caption

                    # elif choice == "overwrite": continue to transcribe below

                print(colorful.cyan(f"🎤 Transcribing media with {transcriber_name}..."))
                if self.transcriber.supports_url:
                    transcription = await self.transcriber.transcribe(url, language=source_lang)
                else:
                    transcription = await self.transcriber.transcribe_file(media_file, language=source_lang)

                await asyncio.to_thread(self.transcriber.write, transcription, transcript_file, encoding="utf-8")

                if isinstance(transcription, Caption):
                    caption_file = transcription
                else:
                    caption_file = str(transcript_file)
                print(colorful.green(f"         ✓ Transcription completed: {caption_file}"))
            else:
                # Download YouTube captions
                caption_file = await self.downloader.download_captions(
                    url=url,
                    output_dir=str(output_dir),
                    force_overwrite=force_overwrite,
                    source_lang=source_lang,
                    transcriber_name=transcriber_name,
                )

                if str(caption_file) == str(transcript_file):
                    # Transcription was used
                    caption = self._read_caption(transcript_file, normalize_text=False)
                    if transcriber_name and "gemini" not in transcriber_name.lower():
                        caption.transcription = caption.supervisions  # alignment will trust transcription's timestamps
                        caption.supervisions = None
                    else:
                        # Gemini transcription's timestamps are not accurate
                        pass

                    return caption

                if caption_file == TRANSCRIBE_CHOICE:
                    return await self._download_or_transcribe_caption(
                        url=url,
                        output_dir=output_dir,
                        media_file=media_file,
                        force_overwrite=force_overwrite,
                        source_lang=source_lang,
                        is_async=True,
                        use_transcription=True,
                    )
                elif not caption_file:
                    raise RuntimeError("No caption file available and transcription was declined by user.")

            return caption_file

        if is_async:
            return _async_impl()
        else:
            return asyncio.run(_async_impl())
