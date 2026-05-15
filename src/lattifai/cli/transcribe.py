"""Transcription CLI entry point with nemo_run."""

import tempfile
from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli.alignment import align as alignment_align
from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
    MediaConfig,
    TranscriptionConfig,
)
from lattifai.utils import _resolve_model_path

# Map a transcriber's native file_suffix to an explicit Caption format hint.
# Caption.read auto-detects most formats from the path suffix, but `.md` is
# ambiguous (could be summary markdown, podcast transcript markdown, etc.), so
# we pass the explicit "markdown" hint to make sure the right parser runs.
_TRANSCRIBER_SUFFIX_TO_FORMAT = {".md": "markdown"}


def _persist_transcript(transcript, output_path, transcriber):
    """Write transcript to disk, respecting the user-requested output suffix.

    Gemini and other LLM transcribers return raw markdown strings, but the user
    might ask for `.json` / `.srt` / `.vtt` output. ``transcriber.write`` on a
    raw string dumps the string verbatim regardless of the path suffix, which
    produces files with misleading extensions (e.g. markdown content saved
    under `output.json`). When the suffix mismatch is detected, round-trip
    through the Caption library so the on-disk format matches the path
    suffix.

    When the suffixes already match (or the transcript is already a structured
    ``Caption`` object), this delegates straight to ``transcriber.write`` so
    binary formats and custom render paths are preserved.
    """
    output_path = Path(output_path)
    src_suffix = (getattr(transcriber, "file_suffix", "") or "").lower()
    out_suffix = output_path.suffix.lower()

    if not isinstance(transcript, str) or not src_suffix or src_suffix == out_suffix:
        return transcriber.write(transcript, output_path, encoding="utf-8", cache_event=False)

    # Cross-format: parse the raw string via the transcriber's native format
    # and re-emit through Caption.write using the requested suffix.
    from lattifai.caption import Caption

    fmt_hint = _TRANSCRIBER_SUFFIX_TO_FORMAT.get(src_suffix)
    with tempfile.NamedTemporaryFile(suffix=src_suffix, mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(transcript)
        tmp_path = Path(tmp.name)
    try:
        cap = Caption.read(str(tmp_path), format=fmt_hint)
        cap.write(str(output_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


@run.cli.entrypoint(name="run", namespace="transcribe", entrypoint_cls=LattifAIEntrypoint)
def transcribe(
    input: Optional[str] = None,
    output_caption: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    event: Annotated[Optional[EventConfig], run.Config[EventConfig]] = None,
):
    """
    Transcribe audio/video file or YouTube URL to caption.

    This command performs automatic speech recognition (ASR) on audio/video files
    or YouTube videos, generating timestamped transcriptions in various caption formats.

    Shortcut: invoking ``lai-transcribe`` is equivalent to running ``lai transcribe run``.

    Args:
        input: Path to input audio/video file or YouTube URL (can be provided as positional argument)
        output_caption: Path for output caption file (can be provided as positional argument)
        media: Media configuration for input/output handling.
            Fields: input_path, output_dir, media_format, channel_selector, streaming_chunk_secs
        transcription: Transcription service configuration.
            Fields: model_name, device, language, gemini_api_key

    Examples:
        # Transcribe local file with positional arguments
        lai transcribe run audio.wav output.srt

        # Transcribe YouTube video
        lai transcribe run "https://www.youtube.com/watch?v=VIDEO_ID" ./output

        # Using specific transcription model
        lai transcribe run audio.mp4 output.ass \\
            transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

        # Using Gemini transcription (requires API key)
        lai transcribe run audio.wav output.srt \\
            transcription.model_name=gemini-2.5-pro \\
            transcription.gemini_api_key=YOUR_KEY

        # Specify language for transcription
        lai transcribe run audio.wav output.srt \\
            transcription.language=zh

        # With MediaConfig settings
        lai transcribe run audio.wav output.srt \\
            media.channel_selector=left \\
            media.streaming_chunk_secs=30.0

        # Full configuration with keyword arguments
        lai transcribe run \\
            input=audio.wav \\
            output_caption=output.srt \\
            transcription.device=cuda \\
            transcription.model_name=iic/SenseVoiceSmall
    """
    import asyncio
    from pathlib import Path

    from lattifai_core.client import SyncAPIClient

    from lattifai.audio2 import AudioLoader
    from lattifai.theme import theme
    from lattifai.transcription import create_transcriber
    from lattifai.utils import safe_print

    # Initialize configs with defaults
    client_config = client or ClientConfig()
    transcription_config = transcription or TranscriptionConfig()
    media_config = media or MediaConfig()

    # Initialize client wrapper to properly set client_wrapper
    client_wrapper = SyncAPIClient(config=client_config)
    transcription_config.client_wrapper = client_wrapper

    # Validate input is required
    if not input and not media_config.input_path:
        raise ValueError("Input is required. Provide input as positional argument or media.input_path.")

    # Assign input to media_config if provided
    if input:
        media_config.set_input_path(input)

    # Detect if input is a URL
    is_url = media_config.is_input_remote()

    # Prepare output paths
    output_dir = media_config.output_dir or Path(media_config.input_path).parent

    # Create transcriber
    if not transcription_config.lattice_model_path:
        transcription_config.lattice_model_path = _resolve_model_path(
            "LattifAI/Lattice-1",
            getattr(transcription_config, "model_hub", "huggingface"),
        )
    event_config = event or EventConfig()
    transcriber = create_transcriber(transcription_config=transcription_config, event_config=event_config)

    safe_print(theme.step(f"🎤 Starting transcription with {transcriber.name}..."))
    safe_print(theme.step(f"    Input: {media_config.input_path}"))

    # Perform transcription
    if is_url and transcriber.supports_url:
        # Check if transcriber supports URL directly
        safe_print(theme.step("    Transcribing from URL directly..."))
        transcript = asyncio.run(transcriber.transcribe(media_config.input_path))
    else:
        if is_url:
            # Download media first, then transcribe
            safe_print(theme.step("    Downloading media from URL..."))
            from lattifai.youtube import YouTubeDownloader

            downloader = YouTubeDownloader()
            input_path = asyncio.run(
                downloader.download_media(
                    url=media_config.input_path,
                    output_dir=str(output_dir),
                    media_format=media_config.normalize_format(),
                    force_overwrite=media_config.force_overwrite,
                )
            )
            safe_print(theme.step(f"    Media downloaded to: {input_path}"))
        else:
            input_path = Path(media_config.input_path)

        safe_print(theme.step("    Loading audio..."))
        # For files, load audio first
        audio_loader = AudioLoader(device=transcription_config.device)
        media_audio = audio_loader(
            input_path,
            channel_selector=media_config.channel_selector,
            streaming_chunk_secs=media_config.streaming_chunk_secs,
        )
        transcript = asyncio.run(transcriber.transcribe(media_audio))

    # Determine output caption path
    if output_caption:
        final_output = Path(str(output_caption))
        final_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        if is_url:
            # For URLs, generate output filename based on transcriber
            output_format = transcriber.file_suffix.lstrip(".")
            final_output = output_dir / f"youtube_LattifAI_{transcriber.name}.{output_format}"
        else:
            # For files, use input filename with suffix
            final_output = Path(media_config.input_path).with_suffix(".LattifAI.srt")

    safe_print(theme.step(f"   Output: {final_output}"))

    # Write output. ``_persist_transcript`` honours the user-requested suffix
    # (e.g. raw markdown → JSON when output_caption ends in `.json`); without
    # it, a Gemini transcript would be dumped verbatim into a `.json` file.
    _persist_transcript(transcript, final_output, transcriber)

    safe_print(theme.ok(f"🎉 Transcription completed: {final_output}"))

    return transcript


@run.cli.entrypoint(name="align", namespace="transcribe", entrypoint_cls=LattifAIEntrypoint)
def transcribe_align(
    input_media: Optional[str] = None,
    output_caption: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    return alignment_align(
        input_media=input_media,
        output_caption=output_caption,
        media=media,
        caption=caption,
        client=client,
        alignment=alignment,
        transcription=transcription,
        diarization=diarization,
    )


def main():
    """Entry point for lai-transcribe command."""
    run.cli.main(transcribe)


if __name__ == "__main__":
    main()
