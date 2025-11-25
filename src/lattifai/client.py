"""LattifAI client implementation with config-driven architecture."""

import asyncio
from typing import Optional, Union

import colorful
from lhotse.utils import Pathlike

from lattifai.alignment import Lattice1Aligner, Segmenter
from lattifai.audio2 import AudioData, AudioLoader
from lattifai.base_client import AsyncAPIClient, LattifAIClientMixin, SyncAPIClient
from lattifai.caption import Caption, InputCaptionFormat
from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig, TranscriptionConfig
from lattifai.errors import (
    AlignmentError,
    CaptionProcessingError,
    LatticeDecodingError,
    LatticeEncodingError,
)


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
        caption_config: Optional[CaptionConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        __doc__ = LattifAIClientMixin._INIT_DOC.format(
            client_class="LattifAI",
            sync_or_async_lower="synchronous",
            config_desc="model and behavior configuration",
            default_desc="default settings (Lattice-1 model, auto device selection)",
            caption_note=" (auto-detect format)",
            transcription_note=". If provided with valid API key, enables transcription capabilities (e.g., Gemini for YouTube videos)",
            api_key_source="and LATTIFAI_API_KEY env var is not set",
        )
        # Initialize all configs with defaults
        client_config, alignment_config, caption_config, transcription_config = self._init_configs(
            client_config, alignment_config, caption_config, transcription_config
        )

        # Initialize base API client
        super().__init__(config=client_config)

        # Store caption config
        self.caption_config = caption_config

        # audio loader
        self.audio_loader = AudioLoader(device=alignment_config.device)

        # aligner
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # Initialize shared components (transcriber, downloader)
        self._init_shared_components(transcription_config)

    def alignment(
        self,
        input_media: Union[Pathlike, AudioData],
        input_caption: Union[Pathlike, Caption],
        output_caption_path: Optional[Pathlike] = None,
        input_caption_format: Optional[InputCaptionFormat] = None,
        split_sentence: Optional[bool] = None,
    ) -> Caption:
        try:
            # Step 1: Parse caption file
            caption = self._read_caption(input_caption, input_caption_format)

            output_caption_path = output_caption_path or self.caption_config.output_path
            if isinstance(input_media, AudioData):
                media_audio = input_media
            else:
                media_audio = self.audio_loader(
                    input_media,
                    channel_selector="average",
                )

            # Step 2: Check if segmented alignment is needed
            segment_strategy = self.caption_config.segment_strategy
            use_segmentation = segment_strategy != "none"

            if use_segmentation or caption.transcription:
                print(colorful.cyan(f"🔄 Using segmented alignment strategy: {segment_strategy}"))

                if caption.transcription:
                    segments = [(sup.start, sup.end, [sup], False) for sup in caption.transcription]
                elif self.caption_config.trust_input_timestamps:
                    # Create segmenter
                    segmenter = Segmenter(self.caption_config)
                    # Create segments from caption
                    segments = segmenter(caption)
                else:
                    raise NotImplementedError(
                        "Segmented alignment without trusting input timestamps is not yet implemented."
                    )

                # align each segment
                supervisions, alignments = [], []
                for i, (start, end, _supervisions, skipalign) in enumerate(segments, 1):
                    print(
                        colorful.cyan(f"  ⏩ aligning segment {i:04d}/{len(segments):04d}: {start:8.2f}s - {end:8.2f}s")
                    )
                    if skipalign:
                        supervisions.extend(_supervisions)
                        alignments.extend(_supervisions)  # may overlap with supervisions, but harmless
                        continue

                    offset = round(start, 4)
                    emission = self.aligner.emission(
                        media_audio.tensor[
                            :, int(start * media_audio.sampling_rate) : int(end * media_audio.sampling_rate)
                        ]
                    )

                    # Align segment
                    _supervisions, _alignments = self.aligner.alignment(
                        media_audio,
                        _supervisions,
                        split_sentence=split_sentence or self.caption_config.split_sentence,
                        return_details=self.caption_config.word_level
                        or (output_caption_path and str(output_caption_path).endswith(".TextGrid")),
                        emission=emission,
                        offset=offset,
                        verbose=False,
                    )

                    supervisions.extend(_supervisions)
                    alignments.extend(_alignments)
            else:
                # Step 2-4: Standard single-pass alignment
                supervisions, alignments = self.aligner.alignment(
                    media_audio,
                    caption.supervisions,
                    split_sentence=split_sentence or self.caption_config.split_sentence,
                    return_details=self.caption_config.word_level
                    or (output_caption_path and str(output_caption_path).endswith(".TextGrid")),
                )

            # Update caption with aligned results
            caption.supervisions = supervisions
            caption.alignments = alignments

            # Step 5: Export alignments
            if output_caption_path:
                self._write_caption(caption, output_caption_path)

            return caption

        except (CaptionProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                "Unexpected error during alignment process",
                media_path=str(input_media),
                caption_path=str(input_caption),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )

    def diarization(
        self,
        input_media: AudioData,
        caption: Caption,
        output_caption_path: Optional[Pathlike] = None,
    ) -> Caption:
        """
        Perform speaker diarization on aligned caption.

        Args:
            input_media: AudioData object
            caption: Caption object with aligned segments
            output_caption_path: Optional path to write diarized caption

        Returns:
            Caption object with speaker labels assigned

        Raises:
            ImportError: If lattifai_core diarization module is not available
            RuntimeError: If diarization fails
        """
        try:
            from lattifai_core.diarization import perform_diarization
        except ImportError:
            raise ImportError(
                "lattifai_core.diarization module not found. " "Please install lattifai-core with diarization support."
            )

        # Load audio if needed
        if isinstance(input_media, AudioData):
            media_audio = input_media
        else:
            media_audio = self.audio_loader(input_media, channel_selector="average")

        # Perform diarization
        diarized_supervisions = perform_diarization(media_audio, caption.supervisions)
        caption.supervisions = diarized_supervisions

        # Write output if requested
        if output_caption_path:
            self._write_caption(caption, output_caption_path)

        return caption

    def youtube(
        self,
        url: str,
        output_dir: Optional[Pathlike] = None,
        media_format: Optional[str] = None,
        caption_lang: Optional[str] = None,
        force_overwrite: bool = False,
        output_caption_path: Optional[Pathlike] = None,
        split_sentence: Optional[bool] = None,
    ) -> Caption:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = self._download_media_sync(url, output_dir, media_format, force_overwrite)

        media_audio = self.audio_loader(media_file, channel_selector="average")

        # Step 2: Get or create captions (download or transcribe)
        caption = self._download_or_transcribe_caption(
            url, output_dir, media_audio, force_overwrite, caption_lang, is_async=False
        )

        # Step 3: Generate output path if not provided
        output_caption_path = self._generate_output_caption_path(output_caption_path, media_file, output_dir)

        # Step 4: Perform alignment
        print(colorful.cyan("🔗 Performing forced alignment..."))

        caption: Caption = self.alignment(
            input_media=media_audio,
            input_caption=caption,
            output_caption_path=output_caption_path if not self.caption_config.speaker_diarization else None,
            split_sentence=split_sentence,
        )

        if self.caption_config.speaker_diarization:
            print(colorful.cyan("🗣️  Performing speaker diarization..."))
            caption = self.diarization(
                input_media=media_audio,
                caption=caption,
                output_caption_path=output_caption_path,
            )

        return caption


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
        ...     input_media="speech.wav",
        ...     input_caption="transcript.srt",
        ...     output_caption_path="aligned.srt"
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
        caption_config: Optional[CaptionConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
    ) -> None:
        __doc__ = LattifAIClientMixin._INIT_DOC.format(
            client_class="AsyncLattifAI",
            sync_or_async_lower="asynchronous",
            config_desc="configuration including API settings and model parameters",
            default_desc="defaults. API key is required either via config or LATTIFAI_API_KEY environment variable",
            caption_note="",
            transcription_note=" for enabling transcription features",
            api_key_source="via alignment_config or environment variable",
        )
        # Initialize all configs with defaults
        client_config, alignment_config, caption_config, transcription_config = self._init_configs(
            client_config, alignment_config, caption_config, transcription_config
        )

        # Initialize base API client
        super().__init__(config=client_config)

        # Store caption config
        self.caption_config = caption_config

        # aligner (will be async version in future)
        self.aligner = Lattice1Aligner(self, config=alignment_config)

        # audio loader
        self.audio_loader = AudioLoader(device=alignment_config.device)

        # Initialize shared components (transcriber, downloader)
        self._init_shared_components(transcription_config)

    async def alignment(
        self,
        input_media: Union[Pathlike, AudioData],
        input_caption: Union[Pathlike, Caption],
        output_caption_path: Optional[Pathlike] = None,
        input_caption_format: Optional[InputCaptionFormat] = None,
        split_sentence: Optional[bool] = None,
    ) -> Caption:
        try:
            # Step 1: Parse caption file (async)
            caption = await asyncio.to_thread(
                self._read_caption,
                input_caption,
                input_caption_format,
            )

            output_caption_path = output_caption_path or self.caption_config.output_path

            if isinstance(input_media, AudioData):
                media_audio = input_media
            else:
                media_audio = await asyncio.to_thread(
                    self.audio_loader,
                    input_media,
                    channel_selector="average",
                )

            # Step 2: Check if segmented alignment is needed
            segment_strategy = self.aligner.config.segment_strategy
            use_segmentation = segment_strategy != "none"

            if use_segmentation:
                if self.aligner.config.verbose:
                    print(colorful.cyan(f"🔄 Using segmented alignment strategy: {segment_strategy}"))

                # Create segmented aligner
                segmenter = Segmenter(self.aligner.config)

                # Create segments from caption (in thread to avoid blocking)
                segments = await asyncio.to_thread(
                    segmenter.create_segments,
                    caption,
                    media_audio,
                )

                if self.aligner.config.verbose:
                    segmenter.print_segment_info(segments, media_audio.duration)

                # Align each segment (async)
                aligned_segments = []
                for i, (start, end, segment_supervisions) in enumerate(segments, 1):
                    if self.aligner.config.verbose:
                        print(colorful.cyan(f"  ⏩ Aligning segment {i}/{len(segments)}: {start:.1f}s - {end:.1f}s"))

                    # Extract audio segment
                    segment_audio = await asyncio.to_thread(
                        media_audio.cut,
                        start=start,
                        end=end,
                    )

                    # Align segment (in thread)
                    aligned_supervisions, aligned_items = await asyncio.to_thread(
                        self.aligner.alignment,
                        segment_audio,
                        segment_supervisions,
                        split_sentence=split_sentence or self.caption_config.split_sentence,
                        return_details=self.caption_config.word_level
                        or (output_caption_path and str(output_caption_path).endswith(".TextGrid")),
                    )

                    aligned_segments.append((start, end, aligned_supervisions, aligned_items))

                # Merge aligned segments (in thread)
                caption.supervisions, caption.alignments = await asyncio.to_thread(
                    segmenter.merge_aligned_segments,
                    aligned_segments,
                )

            else:
                # Step 2-4: Standard single-pass alignment (async)
                supervisions, alignments = await asyncio.to_thread(
                    self.aligner.alignment,
                    media_audio,
                    caption.supervisions,
                    split_sentence=split_sentence or self.caption_config.split_sentence,
                    return_details=self.caption_config.word_level
                    or (output_caption_path and str(output_caption_path).endswith(".TextGrid")),
                )

                caption.supervisions = supervisions
                caption.alignments = alignments

            # Step 5: Export alignments (async)
            if output_caption_path:
                await asyncio.to_thread(
                    self._write_caption,
                    caption,
                    output_caption_path,
                )

            return caption

        except (CaptionProcessingError, LatticeEncodingError, AlignmentError, LatticeDecodingError):
            # Re-raise our specific errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                "Unexpected error during alignment process",
                media_path=str(input_media),
                caption_path=str(input_caption),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )

    async def diarization(
        self,
        input_media: AudioData,
        caption: Caption,
        output_caption_path: Optional[Pathlike] = None,
    ) -> Caption:
        """
        Perform speaker diarization on aligned caption (async).

        Args:
            input_media: AudioData object
            caption: Caption object with aligned segments
            output_caption_path: Optional path to write diarized caption

        Returns:
            Caption object with speaker labels assigned

        Raises:
            ImportError: If lattifai_core diarization module is not available
            RuntimeError: If diarization fails
        """
        try:
            from lattifai_core.diarization import perform_diarization
        except ImportError:
            raise ImportError(
                "lattifai_core.diarization module not found. " "Please install lattifai-core with diarization support."
            )

        # Load audio if needed
        if isinstance(input_media, AudioData):
            media_audio = input_media
        else:
            media_audio = await asyncio.to_thread(
                self.audio_loader,
                input_media,
                channel_selector="average",
            )

        # Perform diarization in thread pool
        diarized_supervisions = await asyncio.to_thread(
            perform_diarization,
            media_audio,
            caption.supervisions,
        )
        caption.supervisions = diarized_supervisions

        # Write output if requested
        if output_caption_path:
            await asyncio.to_thread(
                self._write_caption,
                caption,
                output_caption_path,
            )

        return caption

    async def youtube(
        self,
        url: str,
        output_dir: Optional[Pathlike] = None,
        media_format: Optional[str] = None,
        caption_lang: Optional[str] = None,
        force_overwrite: bool = False,
        output_caption_path: Optional[Pathlike] = None,
        split_sentence: Optional[bool] = None,
    ) -> Caption:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = await self._download_media(url, output_dir, media_format, force_overwrite)

        media_audio = await asyncio.to_thread(
            self.audio_loader,
            media_file,
            channel_selector="average",
        )

        # Step 2: Get or create captions (download or transcribe)
        caption = await self._download_or_transcribe_caption(
            url, output_dir, media_audio, force_overwrite, caption_lang, is_async=True
        )

        # Step 3: Generate output path if not provided
        output_caption_path = self._generate_output_caption_path(output_caption_path, media_file, output_dir)

        # Step 4: Perform alignment
        print(colorful.cyan("🔗 Performing forced alignment..."))
        caption: Caption = await self.alignment(
            input_media=media_audio,
            input_caption=caption,
            output_caption_path=output_caption_path if not self.caption_config.speaker_diarization else None,
            split_sentence=split_sentence,
        )
        if self.caption_config.speaker_diarization:
            print(colorful.cyan("🗣️  Performing speaker diarization..."))
            caption = await self.diarization(
                input_media=media_audio,
                caption=caption,
                output_caption_path=output_caption_path,
            )
        return caption


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
        ...         input_media="speech.wav",
        ...         input_caption="transcript.srt",
        ...         output_caption_path="aligned.srt"
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
        audio, caption, output, split_sentence = sys.argv[1:]
        split_sentence = split_sentence.lower() in ("true", "1", "yes")
    else:
        audio = "tests/data/SA1.wav"
        caption = "tests/data/SA1.TXT"
        output = None
        split_sentence = False

    (alignments, output_caption_path) = client.alignment(
        input_media=audio,
        input_caption=caption,
        output_caption_path=output,
        split_sentence=split_sentence,
    )
