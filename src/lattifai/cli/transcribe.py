"""Transcription CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli.alignment import align as alignment_align
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    MediaConfig,
    TranscriptionConfig,
)
from lattifai.utils import _resolve_model_path


@run.cli.entrypoint(name="run", namespace="transcribe")
def transcribe(
    input: Optional[str] = None,
    output_caption: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
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

    import colorful
    from lattifai_core.client import SyncAPIClient

    from lattifai.audio2 import AudioLoader
    from lattifai.transcription import create_transcriber
    from lattifai.utils import safe_print

    # Initialize configs with defaults
    client_config = client or ClientConfig()
    transcription_config = transcription or TranscriptionConfig()
    media_config = media or MediaConfig()

    # Initialize client wrapper to properly set client_wrapper
    client_wrapper = SyncAPIClient(config=client_config)
    transcription_config.client_wrapper = client_wrapper

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
    if is_url:
        # For URLs, use output_dir from media_config or current directory
        output_path = media_config.output_dir
    else:
        # For files, use input path directory
        output_path = Path(media_config.input_path).parent

    # Create transcriber
    if not transcription_config.lattice_model_path:
        transcription_config.lattice_model_path = _resolve_model_path("LattifAI/Lattice-1")
    transcriber = create_transcriber(transcription_config=transcription_config)

    safe_print(colorful.cyan(f"🎤 Starting transcription with {transcriber.name}..."))
    safe_print(colorful.cyan(f"    Input: {media_config.input_path}"))

    # Perform transcription
    if is_url and transcriber.supports_url:
        # Check if transcriber supports URL directly
        safe_print(colorful.cyan("    Transcribing from URL directly..."))
        transcript = asyncio.run(transcriber.transcribe(media_config.input_path))
    else:
        if is_url:
            # Download media first, then transcribe
            safe_print(colorful.cyan("    Downloading media from URL..."))
            from lattifai.workflow.youtube import YouTubeDownloader

            downloader = YouTubeDownloader()
            input_path = asyncio.run(
                downloader.download_media(
                    url=media_config.input_path,
                    output_dir=str(output_path),
                    media_format=media_config.normalize_format(),
                    force_overwrite=media_config.force_overwrite,
                )
            )
            safe_print(colorful.cyan(f"    Media downloaded to: {input_path}"))
        else:
            input_path = Path(media_config.input_path)

        safe_print(colorful.cyan("    Loading audio..."))
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
            final_output = output_path / f"youtube_LattifAI_{transcriber.name}.{output_format}"
        else:
            # For files, use input filename with suffix
            final_output = Path(media_config.input_path).with_suffix(".LattifAI.srt")

    safe_print(colorful.cyan(f"   Output: {final_output}"))

    # Write output
    transcriber.write(transcript, final_output, encoding="utf-8", cache_audio_events=False)

    safe_print(colorful.green(f"🎉 Transcription completed: {final_output}"))

    return transcript


@run.cli.entrypoint(name="align", namespace="transcribe")
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
