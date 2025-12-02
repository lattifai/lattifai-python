"""Transcription CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from lhotse.utils import Pathlike
from typing_extensions import Annotated

from lattifai.config import TranscriptionConfig
from lattifai.utils import _resolve_model_path


@run.cli.entrypoint(name="file", namespace="transcribe")
def transcribe_file(
    input_media: Optional[Pathlike] = None,
    output_caption: Optional[Pathlike] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
):
    """
    Transcribe audio/video file to caption.

    This command performs automatic speech recognition (ASR) on audio/video files,
    generating timestamped transcriptions in various caption formats.

    Shortcut: invoking ``lai-transcribe-file`` is equivalent to running ``lai transcribe file``.

    Args:
        input_media: Path to input audio/video file (can be provided as positional argument)
        output_caption: Path for output caption file (can be provided as positional argument)
        transcription: Transcription service configuration.
            Fields: model_name, device, language, gemini_api_key

    Examples:
        # Basic usage with positional arguments
        lai transcribe file audio.wav output.srt

        # Using specific transcription model
        lai transcribe file audio.mp4 output.ass \\
            transcription.model_name=nvidia/parakeet-tdt-0.6b-v3

        # Using Gemini transcription (requires API key)
        lai transcribe file audio.wav output.srt \\
            transcription.model_name=gemini-2.5-pro \\
            transcription.gemini_api_key=YOUR_KEY

        # Specify language for transcription
        lai transcribe file audio.wav output.srt \\
            transcription.language=zh

        # Full configuration with keyword arguments
        lai transcribe file \\
            input_media=audio.wav \\
            output_caption=output.srt \\
            transcription.device=cuda \\
            transcription.model_name=iic/SenseVoiceSmall
    """
    import asyncio
    from pathlib import Path

    import colorful

    from lattifai.transcription import create_transcriber

    # Initialize transcription config with defaults
    transcription_config = transcription or TranscriptionConfig()

    # Validate input_media is required
    if not input_media:
        raise ValueError("Input media is required. Provide input_media as positional argument.")

    input_path = Path(str(input_media))

    # Generate default output path if not provided
    if output_caption:
        output_path = Path(str(output_caption))
    else:
        output_path = input_path.with_suffix("LattifAI.srt")

    # Create transcriber
    if not transcription_config.lattice_model_path:
        transcription_config.lattice_model_path = _resolve_model_path("Lattifai/Lattice-1")
    transcriber = create_transcriber(transcription_config=transcription_config)

    print(colorful.cyan(f"🎤 Starting transcription with {transcriber.name}..."))
    print(colorful.cyan(f"    Input: {input_path}"))
    print(colorful.cyan(f"   Output: {output_path}"))

    from lattifai.audio2 import AudioLoader

    audio_loader = AudioLoader(device=transcription_config.device)
    media_audio = audio_loader(input_path, channel_selector="average")

    # Perform transcription
    transcript = asyncio.run(transcriber.transcribe_file(media_audio))

    # Write output
    transcriber.write(transcript, output_path, encoding="utf-8", cache_audio_events=False)

    print(colorful.green(f"🎉 Transcription completed: {output_path}"))

    return transcript


@run.cli.entrypoint(name="youtube", namespace="transcribe")
def transcribe_youtube(
    url: Optional[str] = None,
    output_caption: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    media_format: str = "mp3",
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
):
    """
    Download YouTube video and transcribe to caption.

    This command downloads media from YouTube and performs automatic speech recognition,
    generating timestamped transcriptions without alignment.

    Shortcut: invoking ``lai-transcribe-youtube`` is equivalent to running ``lai transcribe youtube``.

    Args:
        url: YouTube video URL (can be provided as positional argument)
        output_dir: Directory for output files (can be provided as positional argument)
        transcription: Transcription service configuration.
            Fields: model_name, device, language, gemini_api_key

    Examples:
        # Basic usage with positional arguments
        lai transcribe youtube "https://www.youtube.com/watch?v=VIDEO_ID" ./output

        # Using Gemini transcription (requires API key)
        lai transcribe youtube "https://www.youtube.com/watch?v=VIDEO_ID" ./output \\
            transcription.model_name=gemini-2.5-pro \\
            transcription.gemini_api_key=YOUR_KEY

        # Using local model with specific device
        lai transcribe youtube "https://www.youtube.com/watch?v=VIDEO_ID" ./output \\
            transcription.model_name=nvidia/parakeet-tdt-0.6b-v3 \\
            transcription.device=cuda

        # Using keyword argument (traditional syntax)
        lai transcribe youtube \\
            url="https://www.youtube.com/watch?v=VIDEO_ID" \\
            output_dir=./output \\
            transcription.device=mps
    """
    import asyncio
    import tempfile
    from pathlib import Path

    import colorful

    from lattifai.audio2 import AudioLoader
    from lattifai.transcription import create_transcriber
    from lattifai.workflow import FileExistenceManager
    from lattifai.workflow.youtube import YouTubeDownloader

    # Initialize transcription config with defaults
    transcription_config = transcription or TranscriptionConfig()

    # Validate URL input is required
    if not url:
        raise ValueError("YouTube URL is required. Provide url as positional argument.")

    # Prepare output directory
    if output_dir:
        output_path = Path(str(output_dir)).expanduser()
    else:
        output_path = Path(tempfile.gettempdir()) / "LattifAI_transcribe"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create transcriber
    transcriber = create_transcriber(transcription_config=transcription_config)

    # Create downloader
    downloader = YouTubeDownloader()

    print(colorful.cyan(f"🎬 Starting YouTube transcription workflow for: {url}"))

    # Step 1: Download media
    async def download_media():
        print(colorful.cyan("📥 Downloading media from YouTube..."))
        media_file = await downloader.download_media(
            url=url,
            output_dir=str(output_path),
            media_format=media_format,
            force_overwrite=False,
        )
        print(colorful.green(f"    ✓ Media downloaded: {media_file}"))
        return media_file

    media_file = asyncio.run(download_media())

    # Step 2: Transcribe
    print(colorful.cyan(f"🎤 Transcribing with {transcriber.name}..."))

    if transcriber.supports_url:
        # Use URL directly if supported
        transcript = asyncio.run(transcriber.transcribe_url(url))
    else:
        # Load audio and transcribe file
        if not transcriber.config.lattice_model_path:
            transcriber.config.lattice_model_path = _resolve_model_path("Lattifai/Lattice-1")
        audio_loader = AudioLoader(device=transcriber.config.device)
        media_audio = audio_loader(media_file, channel_selector="average")
        transcript = asyncio.run(transcriber.transcribe_file(media_audio))

    # Step 3: Write output
    if not output_caption:
        output_format = transcriber.file_suffix.lstrip(".")
        output_caption = output_path / f"{Path(media_file).stem}_LattifAI_{transcriber.name}.{output_format}"
    transcriber.write(transcript, output_caption, encoding="utf-8", cache_audio_events=False)

    print(colorful.green(f"🎉 Transcription completed: {output_caption}"))

    return transcript


def main_file():
    """Entry point for lai-transcribe-file command."""
    run.cli.main(transcribe_file)


def main_youtube():
    """Entry point for lai-transcribe-youtube command."""
    run.cli.main(transcribe_youtube)


if __name__ == "__main__":
    main_file()
