"""LattifAI client implementation with config-driven architecture."""

import asyncio
from typing import List, Optional, Tuple

import colorful
from lhotse.utils import Pathlike

from lattifai.alignment import Lattice1Aligner
from lattifai.audio2 import AudioLoader
from lattifai.base_client import AsyncAPIClient, LattifAIClientMixin, SyncAPIClient
from lattifai.config import AlignmentConfig, ClientConfig, SubtitleConfig, TranscriptionConfig
from lattifai.errors import (
    AlignmentError,
    LatticeDecodingError,
    LatticeEncodingError,
    SubtitleProcessingError,
)
from lattifai.subtitle import InputSubtitleFormat, Supervision


class LattifAI(LattifAIClientMixin, SyncAPIClient):
    __doc__ = LattifAIClientMixin._CLASS_DOC.format(
        sync_or_async="Synchronous",
        sync_or_async_lower="synchronous",
        client_class="LattifAI",
        await_keyword="",
        async_note="",
        transcriber_note=" (initialized if TranscriptionConfig provided)",
    )

    def __init__(
        self,
        client_config: Optional[ClientConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        __doc__ = LattifAIClientMixin._INIT_DOC.format(
            client_class="LattifAI",
            sync_or_async_lower="synchronous",
            config_desc="model and behavior configuration",
            default_desc="default settings (Lattice-1 model, auto device selection)",
            subtitle_note=" (auto-detect format)",
            transcription_note=". If provided with valid API key, enables transcription capabilities (e.g., Gemini for YouTube videos)",
            api_key_source="and LATTIFAI_API_KEY env var is not set",
        )
        # Initialize all configs with defaults
        client_config, alignment_config, subtitle_config, transcription_config = self._init_configs(
            client_config, alignment_config, subtitle_config, transcription_config
        )

        # Initialize base API client
        super().__init__(config=client_config)

        # audio loader
        self.audio_loader = AudioLoader(device=alignment_config.device)

        # aligner
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # Initialize shared components (subtitler, transcriber, downloader)
        self._init_shared_components(subtitle_config, transcription_config)

    def alignment(
        self,
        input_media_path: Pathlike,
        input_subtitle_path: Pathlike,
        output_subtitle_path: Optional[Pathlike] = None,
        input_subtitle_format: Optional[InputSubtitleFormat] = None,
        split_sentence: Optional[bool] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
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

            output_subtitle_path = output_subtitle_path or self.subtitler.config.output_path
            audio = self.audio_loader(
                input_media_path,
                channel_selector="average",
            )

            # Step 2-4: Align using Lattice1Aligner
            supervisions, alignments = self.aligner.alignment(
                audio,
                supervisions,
                split_sentence=split_sentence or self.subtitler.config.split_sentence,
                return_details=self.subtitler.config.word_level
                or (output_subtitle_path and str(output_subtitle_path).endswith(".TextGrid")),
            )

            # Step 5: Export alignments
            if output_subtitle_path:
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

    def youtube(
        self,
        url: str,
        output_dir: Optional[Pathlike] = None,
        media_format: Optional[str] = None,
        subtitle_lang: Optional[str] = None,
        force_overwrite: bool = False,
        output_subtitle_path: Optional[Pathlike] = None,
        **alignment_kwargs,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = self._download_media_sync(url, output_dir, media_format, force_overwrite)

        # Step 2: Get or create subtitles (download or transcribe)
        subtitle_file = self._get_or_create_subtitles(
            url, output_dir, media_file, force_overwrite, subtitle_lang, is_async=False
        )

        # Step 3: Generate output path if not provided
        output_subtitle_path = self._generate_output_subtitle_path(output_subtitle_path, media_file, output_dir)

        # Step 4: Perform alignment
        print(colorful.cyan("🔗 Performing forced alignment..."))
        return self.alignment(
            input_media_path=media_file,
            input_subtitle_path=subtitle_file,
            output_subtitle_path=output_subtitle_path,
            **alignment_kwargs,
        )


# Set docstrings for LattifAI methods
LattifAI.alignment.__doc__ = LattifAIClientMixin._ALIGNMENT_DOC.format(
    async_prefix="",
    async_word="",
    timing_desc="each word",
    concurrency_note="",
    async_suffix1="",
    async_suffix2="",
    async_suffix3="",
    async_suffix4="",
    async_suffix5="",
    format_default="auto-detects",
    export_note=" in the same format as input (or config default)",
    timing_note=" (start, duration, text)",
    example_imports="client = LattifAI()",
    example_code="""alignments, output_path = client.alignment(
        ...     input_media_path="speech.wav",
        ...     input_subtitle_path="transcript.srt",
        ...     output_subtitle_path="aligned.srt"
        ... )
        >>> for seg in alignments:
        ...     print(f"{seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")""",
)

LattifAI.youtube.__doc__ = LattifAIClientMixin._YOUTUBE_METHOD_DOC.format(client_class="LattifAI", await_keyword="")


class AsyncLattifAI(LattifAIClientMixin, AsyncAPIClient):
    __doc__ = LattifAIClientMixin._CLASS_DOC.format(
        sync_or_async="Asynchronous",
        sync_or_async_lower="asynchronous",
        client_class="AsyncLattifAI",
        await_keyword="await ",
        async_note=" (wrapped in async)",
        transcriber_note="",
    )

    def __init__(
        self,
        client_config: Optional[ClientConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        __doc__ = LattifAIClientMixin._INIT_DOC.format(
            client_class="AsyncLattifAI",
            sync_or_async_lower="asynchronous",
            config_desc="configuration including API settings and model parameters",
            default_desc="defaults. API key is required either via config or LATTIFAI_API_KEY environment variable",
            subtitle_note="",
            transcription_note=" for enabling transcription features",
            api_key_source="via alignment_config or environment variable",
        )
        # Initialize all configs with defaults
        client_config, alignment_config, subtitle_config, transcription_config = self._init_configs(
            client_config, alignment_config, subtitle_config, transcription_config
        )

        # Initialize base API client
        super().__init__(config=client_config)

        # aligner (will be async version in future)
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # audio loader
        self.audio_loader = AudioLoader(device=alignment_config.device)

        # Initialize shared components (subtitler, transcriber, downloader)
        self._init_shared_components(subtitle_config, transcription_config)

    async def alignment(
        self,
        input_media_path: Pathlike,
        input_subtitle_path: Pathlike,
        output_subtitle_path: Optional[Pathlike] = None,
        input_subtitle_format: Optional[InputSubtitleFormat] = None,
        split_sentence: Optional[bool] = None,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
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

            output_subtitle_path = output_subtitle_path or self.subtitler.config.output_path

            audio = await asyncio.to_thread(
                self.audio_loader,
                input_media_path,
                channel_selector="average",
            )

            # Step 2-4: Align using Lattice1Aligner (will be async in future)
            # For now, we wrap the sync aligner in asyncio.to_thread
            supervisions, alignments = await asyncio.to_thread(
                self.aligner.alignment,
                audio,
                supervisions,
                split_sentence=split_sentence or self.subtitler.config.split_sentence,
                return_details=self.subtitler.config.word_level
                or (output_subtitle_path and str(output_subtitle_path).endswith(".TextGrid")),
            )

            # Step 5: Export alignments (async)
            if output_subtitle_path:
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

    async def youtube(
        self,
        url: str,
        output_dir: Optional[Pathlike] = None,
        media_format: Optional[str] = None,
        subtitle_lang: Optional[str] = None,
        force_overwrite: bool = False,
        output_subtitle_path: Optional[Pathlike] = None,
        **alignment_kwargs,
    ) -> Tuple[List[Supervision], Optional[Pathlike]]:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = await self._download_media(url, output_dir, media_format, force_overwrite)

        # Step 2: Get or create subtitles (download or transcribe)
        subtitle_file = await self._get_or_create_subtitles(
            url, output_dir, media_file, force_overwrite, subtitle_lang, is_async=True
        )

        # Step 3: Generate output path if not provided
        output_subtitle_path = self._generate_output_subtitle_path(output_subtitle_path, media_file, output_dir)

        # Step 4: Perform alignment
        print(colorful.cyan("🔗 Performing forced alignment..."))
        return await self.alignment(
            input_media_path=media_file,
            input_subtitle_path=subtitle_file,
            output_subtitle_path=output_subtitle_path,
            **alignment_kwargs,
        )


# Set docstrings for AsyncLattifAI methods
AsyncLattifAI.alignment.__doc__ = LattifAIClientMixin._ALIGNMENT_DOC.format(
    async_prefix="asynchronous ",
    async_word="async ",
    timing_desc="each word",
    concurrency_note="It supports concurrent processing and is ideal for batch alignment tasks.",
    async_suffix1=" (async)",
    async_suffix2=" (async)",
    async_suffix3=" (async)",
    async_suffix4=" (async)",
    async_suffix5=", async",
    format_default="uses config default or auto-detects",
    export_note=" asynchronously",
    timing_note="",
    example_imports="import asyncio\n        >>> async def main():\n        ...     client = AsyncLattifAI()",
    example_code="""alignments, output_path = await client.alignment(
        ...         input_media_path="speech.wav",
        ...         input_subtitle_path="transcript.srt",
        ...         output_subtitle_path="aligned.srt"
        ...     )
        ...     for seg in alignments:
        ...         print(f"{seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
        >>> asyncio.run(main())""",
)

AsyncLattifAI.youtube.__doc__ = LattifAIClientMixin._YOUTUBE_METHOD_DOC.format(
    client_class="AsyncLattifAI", await_keyword="await "
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
