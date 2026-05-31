"""LattifAI client implementation with config-driven architecture."""

import functools
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from lattifai_core.client import SyncAPIClient

import lattifai._init  # noqa: F401 — suppress warnings and expose __version__
from lattifai.alignment import Lattice1Aligner, Segmenter
from lattifai.alignment._merge import SegmentResult, chained_merge_retry
from lattifai.audio2 import AudioData, AudioLoader
from lattifai.caption import AlignmentItem, InputCaptionFormat
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
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
from lattifai.theme import theme
from lattifai.types import Pathlike
from lattifai.utils import safe_print

if TYPE_CHECKING:
    from lattifai.diarization import LattifAIDiarizer  # noqa: F401
    from lattifai.event import LattifAIEventDetector  # noqa: F401

# Patterns for _extract_speaker_description (module-level for compile-once)
_SPEAKER_PATTERNS = re.compile(
    r"(?:host|guest|嘉宾|主播|主持|interviewer|panelist|speaker|featuring|"
    r"joined\s+by|co-host|with\s+\w+\s+(and|&)|【|嘉宾|主讲|对谈)",
    re.IGNORECASE,
)
_SKIP_LINE = re.compile(
    r"^(\d{1,2}:\d{2}(?::\d{2})?\s*[-–—]\s*|https?://|http://|Sign up|Subscribe|"
    r"Follow us|#\w|DISCLAIMER|CONTACT:|Email\s)",
    re.IGNORECASE,
)


def _build_speaker_context(metadata: dict) -> Optional[str]:
    """Build speaker context string from video metadata for LLM inference.

    If structured ``speakers`` are present (from enhanced meta.md), includes
    them directly. Otherwise falls back to extracting from title, channel,
    and description text.
    """
    parts = []

    # Structured speakers field (from enhanced meta.md download pipeline)
    speakers = metadata.get("speakers")
    if speakers and isinstance(speakers, list):
        host_names = [s["name"] for s in speakers if s.get("role") == "host" and s.get("name")]
        guest_names = [s["name"] for s in speakers if s.get("role") == "guest" and s.get("name")]
        if host_names:
            parts.append(f"Channel/Host: {', '.join(host_names)}")
        if guest_names:
            parts.append(f"Guests: {', '.join(guest_names)}")

    title = metadata.get("title")
    if title:
        parts.append(f"Title: {title}")

    if not speakers:
        # Fallback: extract from channel and description when no structured speakers
        uploader = metadata.get("uploader") or metadata.get("channel")
        if uploader:
            parts.append(f"Channel/Host: {uploader}")

    description = metadata.get("description", "")
    if description:
        parts.append(f"Description:\n{_extract_speaker_description(description)}")

    return "\n".join(parts) if parts else None


def _extract_speaker_description(description: str, budget: int = 1500) -> str:
    """Extract speaker-relevant lines from a video description.

    Keeps the intro paragraph and any lines containing speaker/host/guest
    signals, discarding timestamps-only blocks, links, and boilerplate.
    """
    paragraphs = description.split("\n\n")
    kept = []
    total = 0

    # Always keep first paragraph (intro)
    if paragraphs:
        intro = paragraphs[0].strip()
        if len(intro) > 600:
            intro = intro[:600] + "..."
        kept.append(intro)
        total += len(intro)

    # Scan remaining paragraphs for speaker signals
    for para in paragraphs[1:]:
        para = para.strip()
        if not para:
            continue
        # Keep paragraphs with speaker patterns
        if _SPEAKER_PATTERNS.search(para):
            # Filter out pure-link / pure-timestamp lines within the paragraph
            lines = [ln for ln in para.split("\n") if ln.strip() and not _SKIP_LINE.match(ln.strip())]
            if lines:
                block = "\n".join(lines)
                if total + len(block) > budget:
                    break
                kept.append(block)
                total += len(block)

    return "\n\n".join(kept)


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

    def alignment(
        self,
        input_media: Union[Pathlike, AudioData],
        input_caption: Optional[Union[Pathlike, Caption]] = None,
        output_caption_path: Optional[Pathlike] = None,
        input_caption_format: Optional[InputCaptionFormat] = None,
        split_sentence: Optional[bool] = None,
        word_level: Optional[bool] = None,
        channel_selector: Optional[str | int] = "average",
        streaming_chunk_secs: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Caption:
        original_word_level = self.caption_config.word_level
        try:
            if word_level is not None:
                self.caption_config.word_level = word_level
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
                    media_audio,
                    source_lang=self.caption_config.source_lang,
                    is_async=False,
                    output_dir=output_dir,
                )
            else:
                caption = self._read_caption(input_caption, input_caption_format)

            # External transcription (e.g. YT auto-caption VTT) replaces the
            # internal ASR for strategy='transcription'. Resegment only when the
            # source carries word-level alignment — otherwise split_sentences
            # falls back to character-ratio time estimation, which corrupts the
            # segment boundaries the downstream lattice aligner depends on.
            cfg = self.caption_config.input
            transcription_already_split = False
            if cfg.transcription_path and not caption.transcription:
                external = self._read_caption(cfg.transcription_path, cfg.transcription_format)
                caption.transcription = external.supervisions
                if split_sentence or cfg.split_sentence:
                    caption.transcription = self.aligner.tokenizer.split_sentences(
                        caption.transcription,
                        threshold=cfg.split_threshold,
                    )
                    transcription_already_split = True

            output_caption_path = output_caption_path or self.caption_config.output_path

            # Step 2: Check if segmented alignment is needed
            alignment_strategy = self.aligner.config.strategy

            if alignment_strategy != "entire" or caption.transcription:
                safe_print(theme.step(f"🔄   Using segmented alignment strategy: {alignment_strategy}"))

                # Set True when the caption-strategy path below pre-splits
                # sentences on the whole caption; tells the per-segment aligner
                # NOT to re-split (mirrors transcription_already_split above).
                caption_already_split = False

                if caption.supervisions and alignment_strategy == "transcription":
                    from lattifai.alignment.text_align import (
                        align_supervisions_and_transcription,
                    )

                    if not caption.transcription:
                        # Only need an internal ASR pass when transcription was not
                        # already preloaded (SDK preset OR caption.input.transcription_path).
                        # Gemini's per-word timing is unreliable, so refuse it here —
                        # the check is correctly scoped to the ASR call site, not the
                        # whole strategy (an externally-provided transcription has its
                        # own timing source that doesn't depend on the transcriber).
                        if "gemini" in self.transcriber.name.lower():
                            raise ValueError(
                                f"Transcription-based alignment is not supported for {self.transcriber.name} "
                                "(Gemini's timestamp is not reliable). "
                                "Provide an external transcription via "
                                "`caption.input.transcription_path=...` to skip the internal ASR."
                            )
                        transcript = self._transcribe(
                            media_audio,
                            source_lang=self.caption_config.source_lang,
                            is_async=False,
                            output_dir=(Path(str(output_caption_path)).parent if output_caption_path else None),
                        )
                        caption.transcription = transcript.supervisions or transcript.transcription
                        caption.event = transcript.event
                    if not caption.transcription:
                        raise ValueError("Transcription is empty after transcription step.")

                    if split_sentence or self.caption_config.split_sentence:
                        caption.supervisions = self.aligner.tokenizer.split_sentences(
                            caption.supervisions,
                            threshold=self.caption_config.input.split_threshold,
                        )

                    matches = align_supervisions_and_transcription(
                        caption, max_duration=media_audio.duration, verbose=True
                    )

                    skipalign = False
                    matches = sorted(matches, key=lambda x: x[2].WER.WER)  # sort by WER
                    segments = [(m[3].start[1], m[3].end[1], m, skipalign) for m in matches]
                    # Skip the per-match transcription split when the external
                    # source was already split at read time (transcription_already_split).
                    # Resplitting an already-split list is idempotent but wastes
                    # a wtpsplit call per match (N matches → N wasted GPU passes).
                    if not transcription_already_split:
                        for segment in segments:
                            # transcription segments -> sentence splitting
                            segment[2][1] = self.aligner.tokenizer.split_sentences(
                                segment[2][1],
                                threshold=self.caption_config.input.split_threshold,
                            )
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
                        # Pre-split sentences on the WHOLE caption before
                        # segmenting, so segment boundaries land on sentence
                        # boundaries instead of slicing mid-sentence (which would
                        # force a per-segment re-split and inflate the supervision
                        # count at every segment edge). Mirrors the transcription
                        # path's transcription_already_split optimisation.
                        #
                        # Data-driven gate: only safe when supervisions carry
                        # word-level alignment — without it split_sentences would
                        # estimate boundary timestamps by char-ratio and corrupt
                        # the timing the Segmenter relies on.
                        if (split_sentence or self.caption_config.split_sentence) and caption.supervisions:
                            has_word_align = all(
                                (getattr(s, "alignment", None) or {}).get("word") for s in caption.supervisions
                            )
                            if has_word_align:
                                caption.supervisions = self.aligner.tokenizer.split_sentences(
                                    caption.supervisions,
                                    threshold=self.caption_config.input.split_threshold,
                                )
                                caption_already_split = True
                        # Segment the (possibly pre-split) caption.
                        segments = Segmenter(self.aligner.config)(caption)
                    else:
                        raise NotImplementedError(
                            "Segmented alignment without trusting input timestamps is not yet implemented."
                        )

                # align each segment.
                #
                # Transcription strategy uses fail-and-merge retry: lattice
                # decode failures on a single segment used to crash the whole
                # run with zero output. Now failed segments are queued and
                # retried by expanding their window into successful neighbours
                # (see lattifai.alignment._merge). Other strategies retain the
                # original crash-on-failure behaviour.
                sr = media_audio.sampling_rate
                _segment_split = (
                    False
                    if alignment_strategy == "transcription" or caption_already_split
                    else (split_sentence or self.caption_config.split_sentence)
                )

                def _align_one(seg):
                    """Run lattice alignment on a single segment tuple.
                    Raised LatticeDecodingError propagates to the caller."""
                    s_start, s_end, s_sups, _ = seg
                    audio_slice = media_audio.ndarray[:, int(s_start * sr) : int(s_end * sr)]
                    emission = self.aligner.emission(audio_slice)
                    return self.aligner.alignment(
                        media_audio,
                        s_sups,
                        split_sentence=_segment_split,
                        return_details=True,
                        emission=emission,
                        offset=round(s_start, 4),
                        verbose=False,
                        metadata=metadata,
                    )

                # Phase 1: main pass. Segments are independent and each one's
                # tokenize/detokenize are network-bound, so a thread pool of
                # `batch_size` workers overlaps their round-trips. Ordering is
                # preserved regardless (ThreadPoolExecutor.map yields in submit
                # order); batch_size=1 keeps the original serial behaviour.
                def _run_segment(idx_seg):
                    i, seg = idx_seg
                    start, end, _supervisions, skipalign = seg
                    safe_print(
                        theme.ok(f"  ⏩ aligning segment {i:04d}/{len(segments):04d}: {start:8.2f}s - {end:8.2f}s")
                    )
                    if skipalign:
                        # skipalign segments bypass the aligner. A standalone
                        # event-marker sup with no word-level alignment (e.g. a
                        # "[laughter]" carved into its own sentence by the
                        # pre-split) would emit zero words and silently drop from
                        # the output. Synthesise one word spanning the sup so the
                        # marker survives — matches what the aligner yields for
                        # event markers on the non-skip path.
                        for _sup in _supervisions:
                            if not (getattr(_sup, "alignment", None) or {}).get("word"):
                                _sup.alignment = {
                                    **(_sup.alignment or {}),
                                    "word": [
                                        AlignmentItem(
                                            symbol=_sup.text.strip(),
                                            start=_sup.start,
                                            duration=_sup.duration,
                                            score=1.0,
                                        )
                                    ],
                                }
                        return SegmentResult(
                            idx=i - 1,
                            status="skip",
                            supervisions=list(_supervisions),
                            alignments=list(_supervisions),
                            exception=None,
                        )
                    try:
                        _sup_out, _ali_out = _align_one(seg)
                        return SegmentResult(
                            idx=i - 1,
                            status="ok",
                            supervisions=_sup_out,
                            alignments=_ali_out,
                            exception=None,
                        )
                    except LatticeDecodingError as e:
                        safe_print(theme.warn(f"  ⚠️  Segment {i} lattice decode failed; queueing for merge retry"))
                        return SegmentResult(
                            idx=i - 1,
                            status="fail",
                            supervisions=None,
                            alignments=None,
                            exception=e,
                        )

                _batch = self.aligner.config.batch_size
                if _batch == 1 or len(segments) <= 1:
                    seg_results = [_run_segment(x) for x in enumerate(segments, 1)]
                else:
                    with ThreadPoolExecutor(max_workers=_batch) as _ex:
                        seg_results = list(_ex.map(_run_segment, enumerate(segments, 1)))

                # Phase 2 (transcription only): chained merge-retry for failures.
                # Other strategies preserve the original behaviour — any failure
                # would have already propagated out of the main loop above for
                # them, since Phase 1 doesn't change LatticeDecodingError handling
                # for non-transcription paths in current code (segments shape
                # differs; merge helpers assume TextAlignResult tuples).
                if alignment_strategy == "transcription" and any(r.status == "fail" for r in seg_results):
                    seg_results = chained_merge_retry(
                        segments,
                        seg_results,
                        _align_one,
                        warn_fn=lambda msg: safe_print(theme.warn(msg)),
                    )

                # Phase 3: flatten in original index order.
                supervisions, alignments = [], []
                for r in seg_results:
                    if r.status == "fail":
                        # Last-ditch safety net (should not happen — Phase 2
                        # always resolves to 'ok' even via boundary fallback).
                        raise LatticeDecodingError(
                            lattice_id=str(r.idx),
                            message=f"Segment {r.idx + 1} alignment failed and merge retry produced no result",
                        )
                    supervisions.extend(r.supervisions)
                    alignments.extend(r.alignments)

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
        finally:
            self.caption_config.word_level = original_word_level

        # Step 5: Speaker diarization
        if self.diarization_config.enabled and self.diarizer:
            safe_print(theme.step("🗣️  Performing speaker diarization..."))
            caption = self.speaker_diarization(
                input_media=media_audio,
                caption=caption,
                output_caption_path=output_caption_path,
            )

        # Step 6: Event detection
        if self.event_config.enabled and self.event_detector:
            safe_print(theme.step("🔊 Performing audio event detection..."))
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
        speaker_context: Optional[str] = None,
    ) -> Caption:
        """
        Perform speaker diarization on aligned caption.

        Args:
            input_media: AudioData object
            caption: Caption object with aligned segments
            output_caption_path: Optional path to write diarized caption
            speaker_context: Per-call speaker context hint for LLM inference.
                When None and infer_speakers is enabled, auto-builds from
                caption.metadata if available.

        Returns:
            Caption object with speaker labels assigned

        Raises:
            RuntimeError: If diarizer is not initialized or diarization fails
        """
        if not self.diarizer:
            raise RuntimeError("Diarizer not initialized. Set diarization_config.enabled=True")

        # Merge per-call context with metadata auto-detect
        if self.diarization_config.infer_speakers and caption.metadata:
            meta_context = _build_speaker_context(caption.metadata)
            if meta_context:
                speaker_context = f"{speaker_context}\n{meta_context}" if speaker_context else meta_context

        # Perform diarization and assign speaker labels to caption alignments
        if output_caption_path:
            diarization_file = Path(str(output_caption_path)).with_suffix(".SpkDiar")
            if diarization_file.exists():
                safe_print(theme.step(f"Reading existing speaker diarization from {diarization_file}"))
                caption.read_diarization(diarization_file)

        diarization, alignments = self.diarizer.diarize_with_alignments(
            input_media,
            caption.alignments,
            diarization=caption.diarization,
            alignment_fn=(
                functools.partial(self.aligner.alignment, skip_duplicate_prompt=True),
                self.aligner.emission,
            ),
            transcribe_fn=(self.transcriber.transcribe_numpy if self.transcriber else None),
            separate_fn=(self.aligner.separate if self.aligner.worker.separator_ort else None),
            output_path=output_caption_path,
            speaker_context=speaker_context,
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

        safe_print(theme.step(f"🎬 Starting YouTube workflow for: {url}"))

        # Step 1: Download media
        media_file = self._download_media_sync(url, output_dir, media_format, force_overwrite, audio_track_id, quality)

        media_audio = self.audio_loader(
            media_file,
            channel_selector=channel_selector,
            streaming_chunk_secs=streaming_chunk_secs,
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
        # Metadata flows from frontmatter (read into caption.metadata) + video_url fallback
        safe_print(theme.step("🔗 Performing forced alignment..."))

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

    alignments, output_caption_path = client.alignment(
        input_media=audio,
        input_caption=caption,
        output_caption_path=output,
        split_sentence=split_sentence,
    )
