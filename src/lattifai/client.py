"""LattifAI client implementation with config-driven architecture."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import colorful
from lattifai_core.client import SyncAPIClient

from lattifai.alignment import Lattice1Aligner, Segmenter
from lattifai.audio2 import AudioData, AudioLoader
from lattifai.caption import InputCaptionFormat
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
    PodcastConfig,
    TranscriptionConfig,
)
from lattifai.data import Caption
from lattifai.errors import (
    AlignmentError,
    CaptionProcessingError,
    LatticeDecodingError,
    LatticeEncodingError,
)
from lattifai.mixin import LattifAIClientMixin
from lattifai.types import Pathlike
from lattifai.utils import safe_print

if TYPE_CHECKING:
    from lattifai.diarization import LattifAIDiarizer  # noqa: F401
    from lattifai.event import LattifAIEventDetector  # noqa: F401
    from lattifai.podcast import PodcastLoader, SpeakerIdentifier  # noqa: F401
    from lattifai.podcast.types import EpisodeMetadata  # noqa: F401


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
        event_config: Optional[EventConfig] = None,
        podcast_config: Optional[PodcastConfig] = None,
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
        self.config = client_config

        # Initialize all configs with defaults
        alignment_config, transcription_config, diarization_config, event_config = self._init_configs(
            alignment_config, transcription_config, diarization_config, event_config
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

        # Initialize event detector if enabled
        self.event_config = event_config
        self.event_detector = None
        if self.event_config.enabled:
            self._ensure_event_detector()

        # Initialize shared components (transcriber, downloader)
        self._init_shared_components(transcription_config)

        # Initialize podcast config
        self.podcast_config = podcast_config or PodcastConfig(enabled=False)
        self._podcast_loader: Optional["PodcastLoader"] = None
        self._speaker_identifier: Optional["SpeakerIdentifier"] = None

    @property
    def podcast_loader(self) -> "PodcastLoader":
        """Lazy load podcast loader."""
        if self._podcast_loader is None:
            from lattifai.podcast import PodcastLoader

            self._podcast_loader = PodcastLoader()
        return self._podcast_loader

    @property
    def speaker_identifier(self) -> "SpeakerIdentifier":
        """Lazy load speaker identifier."""
        if self._speaker_identifier is None:
            from lattifai.podcast import SpeakerIdentifier

            gemini_key = (
                getattr(self.transcription_config, "gemini_api_key", None) if self.transcription_config else None
            )
            gemini_model = (
                getattr(self.transcription_config, "model_name", "gemini-2.5-flash")
                if self.transcription_config
                else "gemini-2.5-flash"
            )
            self._speaker_identifier = SpeakerIdentifier(
                gemini_api_key=gemini_key,
                gemini_model=gemini_model,
                intro_words=self.podcast_config.intro_words,
            )
        return self._speaker_identifier

    def podcast(
        self,
        url: Optional[str] = None,
        input_media: Optional[Union[Pathlike, "AudioData"]] = None,
        output_dir: Optional[Pathlike] = None,
        output_caption_path: Optional[Pathlike] = None,
        split_sentence: Optional[bool] = None,
        channel_selector: Optional[str | int] = "average",
        streaming_chunk_secs: Optional[float] = None,
        podcast_config: Optional[PodcastConfig] = None,
        use_transcription: bool = False,
    ) -> Caption:
        """Transcribe and align a podcast episode with speaker identification.

        This method adds a podcast-specific layer around the existing alignment()
        pipeline: metadata extraction, audio download, context injection for the
        transcription prompt, and post-alignment speaker identification.

        The heavy lifting (transcription, alignment, diarization, event detection)
        is fully delegated to alignment().

        Args:
            url: Podcast episode URL.
            input_media: Local audio/video file path or AudioData object.
            output_dir: Output directory for downloads and results.
            output_caption_path: Path for aligned caption output.
            split_sentence: Enable sentence splitting for alignment.
            channel_selector: Audio channel selection.
            streaming_chunk_secs: Chunk size for streaming long audio.
            podcast_config: Override podcast config for this call.

        Returns:
            Caption object with aligned, speaker-labeled segments.
        """
        from lattifai.podcast.types import PodcastPlatform

        pc = podcast_config or self.podcast_config
        output_dir_path = self._prepare_youtube_output_dir(output_dir)

        # --- Phase 1: Podcast-specific — resolve media + metadata ---
        episode = None
        if url:
            from lattifai.podcast.platforms import detect_platform

            platform = detect_platform(url)
            if platform == PodcastPlatform.YOUTUBE:
                safe_print(colorful.cyan("🎬 YouTube URL detected, delegating to youtube() workflow..."))
                return self.youtube(
                    url=url,
                    output_dir=output_dir,
                    output_caption_path=output_caption_path,
                    split_sentence=split_sentence,
                    channel_selector=channel_selector,
                    streaming_chunk_secs=streaming_chunk_secs,
                    use_transcription=True,
                )

            safe_print(colorful.cyan(f"🎙️ Starting podcast workflow for: {url}"))
            safe_print(colorful.cyan(f"    Platform: {platform.value}"))

            episode = self.podcast_loader.get_episode_metadata(url, rss_feed_url=pc.rss_feed_url)
            safe_print(colorful.green(f"    ✓ Episode: {episode.title}"))
            if episode.host_names:
                safe_print(colorful.green(f"    ✓ Host(s): {', '.join(episode.host_names)}"))
            if episode.guest_names:
                safe_print(colorful.green(f"    ✓ Guest(s): {', '.join(episode.guest_names)}"))
            if episode.transcript_url:
                safe_print(colorful.green(f"    ✓ Transcript: {episode.transcript_url}"))
                self._download_transcript(episode.transcript_url, output_dir_path)

            safe_print(colorful.cyan("📥 Downloading podcast audio..."))
            media_file = self.podcast_loader.download_audio(
                episode,
                output_dir=str(output_dir_path),
                source_url=url,
            )
            safe_print(colorful.green(f"    ✓ Audio downloaded: {media_file}"))
            input_media = media_file
        elif input_media:
            media_file = "audio_data" if isinstance(input_media, AudioData) else str(input_media)
            safe_print(colorful.cyan(f"🎙️ Starting podcast workflow for local audio: {media_file}"))
        else:
            raise ValueError("Either url or input_media must be provided.")

        # --- Phase 2: Podcast-specific — inject context into transcription prompt ---
        description_parts = self._build_podcast_description(pc, episode)
        original_tc = self.transcription_config
        original_transcriber = self._transcriber
        if description_parts and self.transcription_config:
            import copy

            self.transcription_config = copy.copy(self.transcription_config)
            self.transcription_config.description = "\n".join(description_parts)
            self.transcription_config.prompt = str(
                Path(__file__).parent / "transcription" / "prompts" / "gemini" / "podcast_transcription.txt"
            )
            self._transcriber = None  # force re-init with new prompt

        # --- Phase 3: Reuse alignment() — handles transcription + alignment + diarization ---
        # Propagate podcast num_speakers to diarization config if specified
        original_num_speakers = None
        if pc.num_speakers and self.diarization_config:
            original_num_speakers = self.diarization_config.num_speakers
            self.diarization_config.num_speakers = pc.num_speakers

        output_caption_path = output_caption_path or self._generate_output_caption_path(
            None, media_file, output_dir_path
        )
        try:
            caption = self.alignment(
                input_media=input_media,
                output_caption_path=output_caption_path,
                split_sentence=split_sentence,
                channel_selector=channel_selector,
                streaming_chunk_secs=streaming_chunk_secs,
                metadata={"podcast_url": url} if url else None,
            )
        finally:
            self.transcription_config = original_tc
            self._transcriber = original_transcriber
            if original_num_speakers is not None and self.diarization_config:
                self.diarization_config.num_speakers = original_num_speakers

        # --- Phase 4: Podcast-specific — speaker identification ---
        if pc.identify_speakers and caption.alignments:
            self._identify_and_apply_speakers(caption, episode, pc, output_caption_path)

        return caption

    def _download_transcript(self, transcript_url: str, output_dir: Path) -> Optional[Path]:
        """Download transcript from URL and save to output directory."""
        import requests

        try:
            resp = requests.get(transcript_url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            safe_print(colorful.yellow(f"    ⚠ Failed to download transcript: {e}"))
            return None

        # Derive filename from URL path
        from urllib.parse import urlparse

        slug = urlparse(transcript_url).path.strip("/").replace("/", "_") or "transcript"
        out_path = output_dir / f"{slug}.md"
        if out_path.exists():
            safe_print(colorful.green(f"    ✓ Transcript already exists: {out_path}"))
            return out_path

        # Convert HTML to plain text with basic structure
        html = resp.text
        # Extract <article> or <main> content if available
        import re as _re

        article_match = _re.search(r"<(?:article|main)[^>]*>(.*?)</(?:article|main)>", html, _re.DOTALL)
        content = article_match.group(1) if article_match else html

        # Simple HTML to markdown: preserve paragraphs and links
        content = _re.sub(r"<br\s*/?>", "\n", content)
        content = _re.sub(r"<p[^>]*>", "\n\n", content)
        content = _re.sub(r"</p>", "", content)
        content = _re.sub(r"<a\s+href=[\"']([^\"']+)[\"'][^>]*>([^<]*)</a>", r"[\2](\1)", content)
        content = _re.sub(r"<[^>]+>", "", content)
        from html import unescape

        content = unescape(content)
        content = _re.sub(r"\n{3,}", "\n\n", content).strip()

        out_path.write_text(content, encoding="utf-8")
        safe_print(colorful.green(f"    ✓ Transcript saved: {out_path}"))
        return out_path

    def _build_podcast_description(
        self,
        pc: PodcastConfig,
        episode: Optional["EpisodeMetadata"] = None,
    ) -> list:
        """Build description parts from podcast config and episode metadata."""
        parts = []
        if pc.show_notes:
            parts.append(pc.show_notes)
        elif episode and episode.show_notes:
            parts.append(episode.show_notes)
        if pc.host_names:
            parts.append(f"Host(s): {', '.join(pc.host_names)}")
        elif episode and episode.host_names:
            parts.append(f"Host(s): {', '.join(episode.host_names)}")
        if pc.guest_names:
            parts.append(f"Guest(s): {', '.join(pc.guest_names)}")
        elif episode and episode.guest_names:
            parts.append(f"Guest(s): {', '.join(episode.guest_names)}")
        return parts

    def _identify_and_apply_speakers(
        self,
        caption: Caption,
        episode,
        pc: PodcastConfig,
        output_caption_path: Optional[Pathlike],
    ) -> None:
        """Run speaker identification and apply name mapping to caption."""
        safe_print(colorful.cyan("🔍 Identifying speakers..."))
        transcript_text = "\n".join(f"{s.speaker or ''}: {s.text}" for s in caption.alignments if s.text)
        speakers = self.speaker_identifier.identify(
            transcript_text=transcript_text,
            episode=episode,
            method=pc.identification_method,
            host_names=pc.host_names or (episode.host_names if episode else None),
            guest_names=pc.guest_names or (episode.guest_names if episode else None),
            intro_words=pc.intro_words,
        )
        if not speakers:
            return

        tiers = sorted(set(s.speaker for s in caption.alignments if s.speaker))
        if not tiers:
            return

        tier_mapping = self.speaker_identifier.map_to_diarization_tiers(speakers, transcript_text, tiers)
        if not tier_mapping:
            return

        safe_print(colorful.green(f"    ✓ Speaker mapping: {tier_mapping}"))
        for sup_list in [caption.alignments, caption.supervisions or []]:
            for sup in sup_list:
                if sup.speaker and sup.speaker in tier_mapping:
                    if sup.custom is None:
                        sup.custom = {}
                    sup.custom["original_speaker"] = sup.speaker
                    sup.speaker = tier_mapping[sup.speaker]

        if output_caption_path:
            self._write_caption(caption, output_caption_path)

    def alignment(
        self,
        input_media: Union[Pathlike, AudioData],
        input_caption: Optional[Union[Pathlike, Caption]] = None,
        output_caption_path: Optional[Pathlike] = None,
        input_caption_format: Optional[InputCaptionFormat] = None,
        split_sentence: Optional[bool] = None,
        channel_selector: Optional[str | int] = "average",
        streaming_chunk_secs: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Caption:
        try:
            # Step 1: Get caption
            if isinstance(input_media, AudioData):
                media_audio = input_media
            else:
                media_audio = self.audio_loader(
                    input_media,
                    channel_selector=channel_selector,
                    streaming_chunk_secs=streaming_chunk_secs,
                )

            if not input_caption:
                output_dir = None
                if output_caption_path:
                    output_dir = Path(str(output_caption_path)).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                caption = self._transcribe(
                    media_audio, source_lang=self.caption_config.source_lang, is_async=False, output_dir=output_dir
                )
            else:
                caption = self._read_caption(input_caption, input_caption_format)

            output_caption_path = output_caption_path or self.caption_config.output_path

            # Step 2: Check if segmented alignment is needed
            alignment_strategy = self.aligner.config.strategy

            if alignment_strategy != "entire" or caption.transcription:
                safe_print(colorful.cyan(f"🔄   Using segmented alignment strategy: {alignment_strategy}"))

                if caption.supervisions and alignment_strategy == "transcription":
                    from lattifai.alignment.text_align import align_supervisions_and_transcription

                    if "gemini" in self.transcriber.name.lower():
                        raise ValueError(
                            f"Transcription-based alignment is not supported for {self.transcriber.name} "
                            "(Gemini's timestamp is not reliable)."
                        )
                    if not caption.transcription:
                        transcript = self._transcribe(
                            media_audio,
                            source_lang=self.caption_config.source_lang,
                            is_async=False,
                            output_dir=Path(str(output_caption_path)).parent if output_caption_path else None,
                        )
                        caption.transcription = transcript.supervisions or transcript.transcription
                        caption.event = transcript.event
                    if not caption.transcription:
                        raise ValueError("Transcription is empty after transcription step.")

                    if split_sentence or self.caption_config.split_sentence:
                        caption.supervisions = self.aligner.tokenizer.split_sentences(caption.supervisions)

                    matches = align_supervisions_and_transcription(
                        caption, max_duration=media_audio.duration, verbose=True
                    )

                    skipalign = False
                    matches = sorted(matches, key=lambda x: x[2].WER.WER)  # sort by WER
                    segments = [(m[3].start[1], m[3].end[1], m, skipalign) for m in matches]
                    for segment in segments:
                        # transcription segments -> sentence splitting
                        segment[2][1] = self.aligner.tokenizer.split_sentences(segment[2][1])
                else:
                    if caption.transcription:
                        if "gemini" in self.transcriber.name.lower():
                            raise ValueError(
                                f"Transcription-based alignment is not supported for {self.transcriber.name} "
                                "(Gemini's timestamp is not reliable)."
                            )
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
                sr = media_audio.sampling_rate
                supervisions, alignments = [], []
                for i, (start, end, _supervisions, skipalign) in enumerate(segments, 1):
                    safe_print(
                        colorful.green(
                            f"  ⏩ aligning segment {i:04d}/{len(segments):04d}: {start:8.2f}s - {end:8.2f}s"
                        )
                    )
                    if skipalign:
                        supervisions.extend(_supervisions)
                        alignments.extend(_supervisions)  # may overlap with supervisions, but harmless
                        continue

                    offset = round(start, 4)
                    # Extract audio slice
                    audio_slice = media_audio.ndarray[:, int(start * sr) : int(end * sr)]
                    emission = self.aligner.emission(audio_slice)

                    # Align segment
                    _supervisions, _alignments = self.aligner.alignment(
                        media_audio,
                        _supervisions,
                        split_sentence=split_sentence or self.caption_config.split_sentence,
                        return_details=True,
                        emission=emission,
                        offset=offset,
                        verbose=False,
                        metadata=metadata,
                    )

                    supervisions.extend(_supervisions)
                    alignments.extend(_alignments)

                # sort by start
                alignments = sorted(alignments, key=lambda x: x.start)
            else:
                # Step 2-4: Standard single-pass alignment
                supervisions, alignments = self.aligner.alignment(
                    media_audio,
                    caption.supervisions,
                    split_sentence=split_sentence or self.caption_config.split_sentence,
                    return_details=True,
                    metadata=metadata,
                )

            # Update caption with aligned results
            caption.supervisions = supervisions
            caption.alignments = alignments

            if output_caption_path:
                self._write_caption(caption, output_caption_path)

        except (CaptionProcessingError, LatticeEncodingError) as e:
            # Re-raise our specific errors as-is
            raise e
        except LatticeDecodingError as e:
            raise e
        except Exception as e:
            # Catch any unexpected errors and wrap them
            raise AlignmentError(
                message="Unexpected error during alignment process",
                media_path=str(input_media),
                caption_path=str(input_caption),
                context={"original_error": str(e), "error_type": e.__class__.__name__},
            )

        # Step 5: Speaker diarization
        if self.diarization_config.enabled and self.diarizer:
            safe_print(colorful.cyan("🗣️  Performing speaker diarization..."))
            caption = self.speaker_diarization(
                input_media=media_audio,
                caption=caption,
                output_caption_path=output_caption_path,
            )

        # Step 6: Event detection
        if self.event_config.enabled and self.event_detector:
            safe_print(colorful.cyan("🔊 Performing audio event detection..."))
            caption = self.event_detector.detect_and_update_caption(caption, media_audio)
            if output_caption_path:
                self._write_caption(caption, output_caption_path)

        # Step 7: Profile (all operations completed)
        if self.config.profile:
            self.aligner.profile()
            if self.event_detector:
                self.event_detector.profile()

        return caption

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
                safe_print(colorful.cyan(f"Reading existing speaker diarization from {diarization_file}"))
                caption.read_diarization(diarization_file)

        diarization, alignments = self.diarizer.diarize_with_alignments(
            input_media,
            caption.alignments,
            diarization=caption.diarization,
            alignment_fn=(self.aligner.alignment, self.aligner.emission),
            transcribe_fn=self.transcriber.transcribe_numpy if self.transcriber else None,
            separate_fn=self.aligner.separate if self.aligner.worker.separator_ort else None,
            output_path=output_caption_path,
        )
        caption.alignments = alignments
        caption.diarization = diarization

        # Write output if requested
        if output_caption_path:
            self._write_caption(caption, output_caption_path)

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
        streaming_chunk_secs: Optional[float] = None,
        audio_track_id: Optional[str] = "original",
        quality: str = "best",
    ) -> Caption:
        # Prepare output directory and media format
        output_dir = self._prepare_youtube_output_dir(output_dir)
        media_format = self._determine_media_format(media_format)

        safe_print(colorful.cyan(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = self._download_media_sync(url, output_dir, media_format, force_overwrite, audio_track_id, quality)

        media_audio = self.audio_loader(
            media_file, channel_selector=channel_selector, streaming_chunk_secs=streaming_chunk_secs
        )

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
        safe_print(colorful.cyan("🔗 Performing forced alignment..."))

        caption: Caption = self.alignment(
            input_media=media_audio,
            input_caption=caption,
            output_caption_path=output_caption_path,
            split_sentence=split_sentence,
            channel_selector=channel_selector,
            streaming_chunk_secs=None,
            metadata={"video_url": url},
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
