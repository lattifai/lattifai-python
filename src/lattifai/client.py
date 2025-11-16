"""LattifAI client implementation with config-driven architecture."""

import asyncio
from typing import List, Optional, Tuple

import colorful
from lhotse.utils import Pathlike

from lattifai.alignment import Lattice1Aligner
from lattifai.base_client import AsyncAPIClient, SyncAPIClient
from lattifai.config import AlignmentConfig, ClientConfig, SubtitleConfig
from lattifai.errors import (
    AlignmentError,
    LatticeDecodingError,
    LatticeEncodingError,
    SubtitleProcessingError,
)
from lattifai.subtitle import InputSubtitleFormat, Subtitler, Supervision


class LattifAI(SyncAPIClient):
    """
    Synchronous LattifAI client for audio/video-subtitle alignment.

    This client provides synchronous methods for aligning audio/video files with subtitle/transcript
    text using the Lattice-1 forced alignment model. It supports multiple subtitle formats
    (SRT, VTT, ASS, TXT) and provides word-level alignment with configurable sentence splitting.

    The client uses a config-driven architecture with three main configuration objects:
    - ClientConfig: API connection settings (API key, base URL, timeout, retries)
    - AlignmentConfig: Model and alignment behavior settings
    - SubtitleConfig: Subtitle I/O format and processing settings

    Example:
        >>> from lattifai import LattifAI, ClientConfig
        >>>
        >>> # Initialize with default settings
        >>> client = LattifAI()
        >>>
        >>> # Or with custom configuration
        >>> config = ClientConfig(api_key="your-api-key")
        >>> client = LattifAI(config=config)
        >>>
        >>> # Perform alignment
        >>> alignments, output_path = client.alignment(
        ...     input_media_path="audio.wav",
        ...     input_subtitle_path="subtitle.srt",
        ...     output_subtitle_path="aligned.srt"
        ... )

    Attributes:
        aligner: Lattice1Aligner instance for performing forced alignment
        subtitler: Subtitler instance for reading/writing subtitle files
    """

    def __init__(
        self,
        client_config: Optional[ClientConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
    ) -> None:
        """
        Initialize LattifAI synchronous client.

        Args:
            client_config: Client configuration for API connection settings. If None, uses defaults
                          (reads API key from LATTIFAI_API_KEY environment variable).
            alignment_config: Alignment model and behavior configuration. If None, uses default
                            settings (Lattice-1 model, auto device selection).
            subtitle_config: Subtitle I/O configuration for format handling and processing.
                           If None, uses default settings (auto-detect format).

        Raises:
            ConfigurationError: If API key is not provided and LATTIFAI_API_KEY env var is not set.
        """
        # Initialize configs with defaults if not provided
        if client_config is None:
            client_config = ClientConfig()

        if alignment_config is None:
            alignment_config = AlignmentConfig()

        if subtitle_config is None:
            subtitle_config = SubtitleConfig()

        # Initialize base API client
        super().__init__(config=client_config)

        # aligner
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # subtitler IO
        self.subtitler = Subtitler(config=subtitle_config)

    def alignment(
        self,
        input_media_path: Pathlike,
        input_subtitle_path: Pathlike,
        input_subtitle_format: Optional[InputSubtitleFormat] = None,
        split_sentence: Optional[bool] = None,
        output_subtitle_path: Optional[Pathlike] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        """
        Perform forced alignment on audio and subtitle/text.

        This method aligns subtitle text with audio by finding the precise timing of each word
        and subtitle segment. It uses the Lattice-1 forced alignment model to process the audio
        and match it against the provided subtitle text.

        The alignment process consists of five steps:
        1. Parse the input subtitle file into segments
        2. Generate a lattice graph from subtitle text
        3. Search the lattice using audio features
        4. Decode results to extract word-level timings
        5. Export aligned subtitles (if output path provided)

        Args:
            input_media_path: Path to audio/video file (WAV, MP3, FLAC, MP4, etc.). Must be readable by ffmpeg.
            input_subtitle_path: Path to subtitle or plain text file to align with audio.
            input_subtitle_format: Input subtitle format ('srt', 'vtt', 'ass', 'txt'). If None, auto-detects
                   from file extension or uses config default.
            split_sentence: Enable automatic sentence re-splitting for better alignment accuracy.
                          If None, uses config default (typically False).
            output_subtitle_path: Optional path to write aligned subtitle file. If provided,
                                exports results in the same format as input (or config default).

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information (start, duration, text)
                - Output subtitle path (same as input parameter, or None if not provided)

        Raises:
            SubtitleProcessingError: If subtitle file cannot be parsed or output cannot be written.
            LatticeEncodingError: If lattice graph generation fails (invalid text format).
            AlignmentError: If audio alignment fails (audio processing or model inference error).
            LatticeDecodingError: If lattice decoding fails (invalid results from model).

        Example:
            >>> client = LattifAI()
            >>> alignments, output_path = client.alignment(
            ...     input_media_path="speech.wav",
            ...     input_subtitle_path="transcript.srt",
            ...     output_subtitle_path="aligned.srt"
            ... )
            >>> for seg in alignments:
            ...     print(f"{seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
        """
        try:
            # Step 1: Parse subtitle file
            print(colorful.cyan(f"📖 Step 1: Reading subtitle file from {input_subtitle_path}"))
            try:
                supervisions = self.subtitler.read(input_path=input_subtitle_path, format=input_subtitle_format)
                print(colorful.green(f"         ✓ Parsed {len(supervisions)} subtitle segments"))
            except Exception as e:
                raise SubtitleProcessingError(
                    f"Failed to parse subtitle file: {input_subtitle_path}",
                    subtitle_path=str(input_subtitle_path),
                    context={"original_error": str(e)},
                )

            # Step 2-4: Align using Lattice1Aligner
            supervisions, alignments = self.aligner.alignment(
                input_media_path,
                supervisions,
                split_sentence=split_sentence or self.subtitler.config.split_sentence,
            )

            # Step 5: Export alignments
            if output_subtitle_path or self.subtitler.config.output_path:
                try:
                    _ = self.subtitler.write(alignments, output_path=output_subtitle_path)
                    print(colorful.green(f"🎉🎉🎉🎉🎉 Subtitle file written to: {output_subtitle_path}"))
                except Exception as e:
                    raise SubtitleProcessingError(
                        f"Failed to write output file: {output_subtitle_path}",
                        subtitle_path=str(output_subtitle_path),
                        context={"original_error": str(e)},
                    )

            return (alignments, output_subtitle_path)

        except (SubtitleProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                "Unexpected error during alignment process",
                media_path=str(input_media_path),
                subtitle_path=str(input_subtitle_path),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )


class AsyncLattifAI(AsyncAPIClient):
    """
    Asynchronous LattifAI client for audio/video-subtitle alignment.

    This client provides asynchronous methods for aligning audio/video files with subtitle/transcript
    text using the Lattice-1 forced alignment model. It supports concurrent processing and is
    ideal for batch alignment tasks or integration into async applications.

    The async client uses the same config-driven architecture as the synchronous client and
    supports multiple subtitle formats (SRT, VTT, ASS, TXT) with word-level alignment.

    Example:
        >>> import asyncio
        >>> from lattifai import AsyncLattifAI, ClientConfig
        >>>
        >>> async def align():
        ...     config = ClientConfig(api_key="your-api-key")
        ...     client = AsyncLattifAI(config=config)
        ...
        ...     alignments, output_path = await client.alignment(
        ...         input_media_path="audio.wav",
        ...         input_subtitle_path="subtitle.srt",
        ...         output_subtitle_path="aligned.srt"
        ...     )
        ...     return alignments
        >>>
        >>> asyncio.run(align())

    Attributes:
        aligner: Lattice1Aligner instance for performing forced alignment (wrapped in async)
        subtitler: Subtitler instance for reading/writing subtitle files
    """

    def __init__(
        self,
        client_config: Optional[ClientConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
    ) -> None:
        """
        Initialize asynchronous LattifAI client.

        Args:
            client_config: Client configuration for API connection settings. If None, uses defaults
                          (reads API key from LATTIFAI_API_KEY environment variable).
            alignment_config: Alignment configuration including API settings and model parameters.
                            If None, uses defaults. API key is required either via config or
                            LATTIFAI_API_KEY environment variable.
            subtitle_config: Subtitle I/O configuration for format handling and processing.
                           If None, uses default settings.

        Raises:
            ConfigurationError: If API key is not provided via alignment_config or environment variable.
        """
        # Initialize configs with defaults if not provided
        if client_config is None:
            client_config = ClientConfig()

        if alignment_config is None:
            alignment_config = AlignmentConfig()

        if subtitle_config is None:
            subtitle_config = SubtitleConfig()

        # Initialize base API client
        super().__init__(config=client_config)

        # aligner (will be async version in future)
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # subtitler IO
        self.subtitler = Subtitler(config=subtitle_config)

    async def alignment(
        self,
        input_media_path: Pathlike,
        input_subtitle_path: Pathlike,
        input_subtitle_format: Optional[InputSubtitleFormat] = None,
        split_sentence: Optional[bool] = None,
        output_subtitle_path: Optional[Pathlike] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        """
        Perform asynchronous forced alignment on audio and subtitle/text.

        This async method aligns subtitle text with audio by finding the precise timing of
        each word and subtitle segment. It supports concurrent processing and is ideal for
        batch alignment tasks.

        The alignment process consists of five steps:
        1. Parse the input subtitle file into segments
        2. Generate a lattice graph from subtitle text (async)
        3. Search the lattice using audio features (async)
        4. Decode results to extract word-level timings (async)
        5. Export aligned subtitles (if output path provided, async)

        Args:
            input_media_path: Path to audio/video file (WAV, MP3, FLAC, MP4, etc.). Must be readable by ffmpeg.
            input_subtitle_path: Path to subtitle or plain text file to align with audio.
            input_subtitle_format: Input subtitle format ('srt', 'vtt', 'ass', 'txt'). If None, uses config
                   default or auto-detects from file extension.
            split_sentence: Enable automatic sentence re-splitting for better alignment accuracy.
                          If None, uses config default (typically False).
            output_subtitle_path: Optional path to write aligned subtitle file. If provided,
                                exports results asynchronously.

        Returns:
            Tuple containing:
                - List of Supervision objects with aligned timing information
                - Output subtitle path (same as input parameter, or None if not provided)

        Raises:
            SubtitleProcessingError: If subtitle file cannot be parsed or output cannot be written.
            LatticeEncodingError: If lattice graph generation fails (invalid text format).
            AlignmentError: If audio alignment fails (audio processing or model inference error).
            LatticeDecodingError: If lattice decoding fails (invalid results from model).

        Example:
            >>> import asyncio
            >>> async def main():
            ...     client = AsyncLattifAI()
            ...     alignments, output_path = await client.alignment(
            ...         input_media_path="speech.wav",
            ...         input_subtitle_path="transcript.srt",
            ...         output_subtitle_path="aligned.srt"
            ...     )
            ...     for seg in alignments:
            ...         print(f"{seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
            >>> asyncio.run(main())
        """
        try:
            # Step 1: Parse subtitle file (async)
            print(colorful.cyan(f"📖 Step 1: Reading subtitle file from {input_subtitle_path}"))
            try:
                supervisions = await asyncio.to_thread(
                    self.subtitler.read, input_path=input_subtitle_path, format=input_subtitle_format
                )
                print(colorful.green(f"         ✓ Parsed {len(supervisions)} subtitle segments"))
            except Exception as e:
                raise SubtitleProcessingError(
                    f"Failed to parse subtitle file: {input_subtitle_path}",
                    subtitle_path=str(input_subtitle_path),
                    context={"original_error": str(e)},
                )

            # Step 2-4: Align using Lattice1Aligner (will be async in future)
            # For now, we wrap the sync aligner in asyncio.to_thread
            supervisions, alignments = await asyncio.to_thread(
                self.aligner.alignment,
                input_media_path,
                supervisions,
                split_sentence=split_sentence or self.subtitler.config.split_sentence,
            )

            # Step 5: Export alignments (async)
            if output_subtitle_path or self.subtitler.config.output_path:
                try:
                    await asyncio.to_thread(self.subtitler.write, alignments, output_path=output_subtitle_path)
                    print(colorful.green(f"🎉🎉🎉🎉🎉 Subtitle file written to: {output_subtitle_path}"))
                except Exception as e:
                    raise SubtitleProcessingError(
                        f"Failed to write output file: {output_subtitle_path}",
                        subtitle_path=str(output_subtitle_path),
                        context={"original_error": str(e)},
                    )

            return (alignments, output_subtitle_path)

        except (SubtitleProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                "Unexpected error during alignment process",
                media_path=str(input_media_path),
                subtitle_path=str(input_subtitle_path),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )


if __name__ == "__main__":
    client = LattifAI()
    import sys

    if len(sys.argv) == 5:
        audio, subtitle, output, split_sentence = sys.argv[1:]
        split_sentence = split_sentence.lower() in ("true", "1", "yes")
    else:
        audio = "tests/data/SA1.wav"
        subtitle = "tests/data/SA1.TXT"
        output = None
        split_sentence = False

    (alignments, output_subtitle_path) = client.alignment(
        input_media_path=audio,
        input_subtitle_path=subtitle,
        output_subtitle_path=output,
        split_sentence=split_sentence,
    )
