"""LattifAI client implementation with config-driven architecture."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import colorful
from lattifai_core.client import SyncAPIClient
from lhotse.utils import Pathlike

from lattifai.alignment import Lattice1Aligner, Segmenter
from lattifai.audio2 import AudioData, AudioLoader
from lattifai.caption import Caption, InputCaptionFormat
from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig, DiarizationConfig, TranscriptionConfig
from lattifai.errors import (
    AlignmentError,
    CaptionProcessingError,
    LatticeDecodingError,
    LatticeEncodingError,
)
from lattifai.mixin import LattifAIClientMixin

if TYPE_CHECKING:
    from lattifai.diarization import LattifAIDiarizer  # noqa: F401


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
        diarization_config: Optional[DiarizationConfig] = None,
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
        if client_config is None:
            client_config = ClientConfig()

        # Initialize base API client
        super().__init__(config=client_config)

        # Initialize all configs with defaults
        alignment_config, transcription_config, diarization_config = self._init_configs(
            alignment_config, transcription_config, diarization_config
        )

        # Store configs
        if caption_config is None:
            caption_config = CaptionConfig()
        self.caption_config = caption_config

        # audio loader
        self.audio_loader = AudioLoader(device=alignment_config.device)

        # aligner
        self.aligner = Lattice1Aligner(config=alignment_config)

        # Initialize diarizer if enabled
        self.diarization_config = diarization_config
        self.diarizer: Optional["LattifAIDiarizer"] = None
        if self.diarization_config.enabled:
            from lattifai.diarization import LattifAIDiarizer  # noqa: F811

            self.diarizer = LattifAIDiarizer(config=self.diarization_config)

        # Initialize shared components (transcriber, downloader)
        self._init_shared_components(transcription_config)

    def alignment(
        self,
        input_media: Union[Pathlike, AudioData],
        input_caption: Optional[Union[Pathlike, Caption]] = None,
        output_caption_path: Optional[Pathlike] = None,
        input_caption_format: Optional[InputCaptionFormat] = None,
        split_sentence: Optional[bool] = None,
        channel_selector: Optional[str | int] = "average",
    ) -> Caption:
        try:
            # Step 1: Get caption
            if isinstance(input_media, AudioData):
                media_audio = input_media
            else:
                media_audio = self.audio_loader(
                    input_media,
                    channel_selector=channel_selector,
                )

            if not input_caption:
                caption = self._transcribe(media_audio, source_lang=self.caption_config.source_lang, is_async=False)
            else:
                caption = self._read_caption(input_caption, input_caption_format)

            output_caption_path = output_caption_path or self.caption_config.output_path

            # Step 2: Check if segmented alignment is needed
            alignment_strategy = self.aligner.config.strategy

            if alignment_strategy != "entire" or caption.transcription:
                print(colorful.cyan(f"🔄 Using segmented alignment strategy: {alignment_strategy}"))

                if caption.supervisions and alignment_strategy == "transcription":
                    # raise NotImplementedError("Transcription-based alignment is not yet implemented.")
                    assert (
                        "gemini" not in self.transcriber.name.lower()
                    ), "Transcription-based alignment is not supported with Gemini transcriber."
                    assert (
                        caption.supervisions
                    ), "Input caption should contain supervisions when using transcription-based alignment."
                    if not caption.transcription:
                        import asyncio

                        print(colorful.cyan("📝 Transcribing media for alignment..."))
                        if output_caption_path:
                            transcript_file = (
                                Path(str(output_caption_path)).parent
                                / f"{Path(str(media_audio)).stem}_{self.transcriber.file_name}"
                            )
                            if transcript_file.exists():
                                # print(colorful.cyan(f"Reading existing transcription from {transcript_file}"))
                                transcript = self._read_caption(transcript_file, verbose=False)
                                caption.transcription = transcript.supervisions
                                caption.audio_events = transcript.audio_events

                        if not caption.transcription:
                            transcript = asyncio.run(
                                self.transcriber.transcribe(media_audio, language=self.caption_config.source_lang)
                            )
                            caption.transcription = transcript.transcription
                            caption.audio_events = transcript.audio_events

                    # Align caption.supervisions with transcription to get segments
                    import regex
                    from error_align import ErrorAlign, error_align  # noqa: F401
                    from error_align.utils import DELIMITERS, NUMERIC_TOKEN, STANDARD_TOKEN, OpType

                    JOIN_TOKEN = "❄"
                    if JOIN_TOKEN not in DELIMITERS:
                        DELIMITERS.add(JOIN_TOKEN)

                    def custom_tokenizer(text: str) -> list:
                        """Default tokenizer that splits text into words based on whitespace.

                        Args:
                            text (str): The input text to tokenize.

                        Returns:
                            list: A list of tokens (words).

                        """
                        # Escape JOIN_TOKEN for use in regex pattern
                        escaped_join_token = regex.escape(JOIN_TOKEN)
                        return list(
                            regex.finditer(
                                rf"({NUMERIC_TOKEN})|({STANDARD_TOKEN}|{escaped_join_token})",
                                text,
                                regex.UNICODE | regex.VERBOSE,
                            )
                        )

                    alignments = error_align(
                        f"{JOIN_TOKEN}".join(sup.text for sup in caption.supervisions),
                        f"{JOIN_TOKEN}".join(sup.text for sup in caption.transcription),
                        tokenizer=custom_tokenizer,
                    )

                    for align in alignments:
                        if align.hyp == JOIN_TOKEN and align.op_type == OpType.MATCH:
                            pass

                        # if align.op_type == OpType.MATCH:
                        #     continue
                        # elif align.op_type in (OpType.INSERT, OpType.DELETE, OpType.SUBSTITUTE):
                        #     # print(colorful.yellow(f"⚠️ Alignment warning: {op}"))
                        #     pass

                    raise NotImplementedError("Transcription-based segmentation is not yet implemented.")
                else:
                    if caption.transcription:
                        if not caption.supervisions:  # youtube + transcription case
                            segments = [(sup.start, sup.end, [sup], not sup.text) for sup in caption.transcription]
                        else:
                            raise NotImplementedError(
                                f"Input caption with both supervisions and transcription(strategy={alignment_strategy}) is not supported."
                            )
                    elif self.aligner.config.trust_caption_timestamps:
                        # Create segmenter
                        segmenter = Segmenter(self.aligner.config)
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
                        colorful.green(
                            f"  ⏩ aligning segment {i:04d}/{len(segments):04d}: {start:8.2f}s - {end:8.2f}s"
                        )
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

            # Step 5: Speaker diarization
            if self.diarization_config.enabled and self.diarizer:
                print(colorful.cyan("🗣️  Performing speaker diarization..."))
                caption = self.speaker_diarization(
                    input_media=media_audio,
                    caption=caption,
                    output_caption_path=output_caption_path,
                )
            elif output_caption_path:
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

    def speaker_diarization(
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
            RuntimeError: If diarizer is not initialized or diarization fails
        """
        if not self.diarizer:
            raise RuntimeError("Diarizer not initialized. Set diarization_config.enabled=True")

        # Perform diarization and assign speaker labels to caption alignments
        if output_caption_path:
            diarization_file = Path(str(output_caption_path)).with_suffix(".SpkDiar")
            if diarization_file.exists():
                print(colorful.cyan(f"Reading existing speaker diarization from {diarization_file}"))
                caption.read_speaker_diarization(diarization_file)

        diarization, alignments = self.diarizer.diarize_with_alignments(
            input_media, caption.alignments, diarization=caption.speaker_diarization
        )
        caption.alignments = alignments
        caption.speaker_diarization = diarization

        # Write output if requested
        if output_caption_path:
            self._write_caption(caption, output_caption_path)

            if self.diarizer.config.debug:
                # debug
                from tgt import Interval, IntervalTier, TextGrid, write_to_file

                debug_tg = TextGrid()
                transcript_tier = IntervalTier(
                    start_time=0,
                    end_time=input_media.duration,
                    name="transcript",
                    objects=[Interval(sup.start, sup.end, sup.text) for sup in caption.alignments],
                )
                debug_tg.add_tier(transcript_tier)

                speaker_tier = IntervalTier(
                    start_time=0,
                    end_time=input_media.duration,
                    name="speaker",
                    objects=[Interval(sup.start, sup.end, sup.speaker) for sup in caption.alignments],
                )
                debug_tg.add_tier(speaker_tier)

                from collections import defaultdict

                spk2intervals = defaultdict(lambda: [])
                num_multispk = 0

                segments, skipks = [], []
                for k, supervision in enumerate(caption.alignments):  # TODO: alignments 本身存在 overlap, eg: [event]
                    # supervision = caption.alignments[k]
                    if supervision.custom.get("speaker", []):
                        num_multispk += 1
                    else:
                        continue

                    if k in skipks:
                        continue

                    for speaker in supervision.custom.get("speaker", []):
                        for name, start_time, end_time in speaker:
                            spk2intervals[name].append(Interval(start_time, end_time, name))

                    _segments = []
                    if k > 0:
                        _segments.append(caption.alignments[k - 1])
                    _segments.append(supervision)
                    while k + 1 < len(caption.alignments):
                        skipks.append(k + 1)
                        next_sup = caption.alignments[k + 1]
                        if not next_sup.custom.get("speaker", []):
                            k += 1
                            break
                        _segments.append(next_sup)
                        k += 1

                    if segments:
                        if _segments[0].start >= segments[-1][-1].end:
                            segments.append(_segments)
                        else:
                            if _segments[1:]:
                                segments.append(_segments[1:])
                            else:
                                pass
                    else:
                        segments.append(_segments)

                print(
                    f"Number of multi-speaker segments: {num_multispk}/{len(caption.alignments)} segments: {len(segments)}"
                )

                for speaker, intervals in sorted(spk2intervals.items(), key=lambda x: x[0]):
                    speaker_tier = IntervalTier(
                        start_time=0, end_time=input_media.duration, name=speaker, objects=intervals
                    )
                    debug_tg.add_tier(speaker_tier)

                for tier in caption.speaker_diarization.tiers:
                    tier.name = f"Diarization-{tier.name}"
                    debug_tg.add_tier(tier)

                tier = IntervalTier(
                    start_time=0,
                    end_time=input_media.duration,
                    name="resegment",
                    objects=[
                        Interval(round(sup.start, 2), round(sup.end, 2), sup.text)
                        for _segments in segments
                        for sup in _segments
                    ],
                )
                debug_tg.add_tier(tier)

                # if caption.audio_events:
                #     for tier in caption.audio_events.tiers:
                #         # tier.name = f"{tier.name}"
                #         debug_tg.add_tier(tier)

                debug_tgt_file = Path(str(output_caption_path)).with_suffix(".DiarizationDebug.TextGrid")
                write_to_file(debug_tg, debug_tgt_file, format="long")

        return caption

    def youtube(
        self,
        url: str,
        output_dir: Optional[Pathlike] = None,
        media_format: Optional[str] = None,
        source_lang: Optional[str] = None,
        force_overwrite: bool = False,
        output_caption_path: Optional[Pathlike] = None,
        split_sentence: Optional[bool] = None,
        use_transcription: bool = False,
        channel_selector: Optional[str | int] = "average",
    ) -> Caption:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = self._download_media_sync(url, output_dir, media_format, force_overwrite)

        media_audio = self.audio_loader(media_file, channel_selector=channel_selector)

        # Step 2: Get or create captions (download or transcribe)
        caption = self._download_or_transcribe_caption(
            url,
            output_dir,
            media_audio,
            force_overwrite,
            source_lang or self.caption_config.source_lang,
            is_async=False,
            use_transcription=use_transcription,
        )

        # Step 3: Generate output path if not provided
        output_caption_path = self._generate_output_caption_path(output_caption_path, media_file, output_dir)

        # Step 4: Perform alignment
        print(colorful.cyan("🔗 Performing forced alignment..."))

        caption: Caption = self.alignment(
            input_media=media_audio,
            input_caption=caption,
            output_caption_path=output_caption_path,
            split_sentence=split_sentence,
            channel_selector=channel_selector,
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
