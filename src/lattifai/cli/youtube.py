"""YouTube workflow CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import LattifAI
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    MediaConfig,
    TranscriptionConfig,
)


@run.cli.entrypoint(name="youtube", namespace="alignment")
def youtube(
    yt_url: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """
    Download media from YouTube (when needed) and align captions.

    This command provides a convenient workflow for aligning captions with YouTube videos.
    It can automatically download media from YouTube URLs and optionally transcribe audio
    using Gemini or download available captions from YouTube.

    When a YouTube URL is provided:
    1. Downloads media in the specified format (audio or video)
    2. Optionally transcribes audio with Gemini OR downloads YouTube captions
    3. Performs forced alignment with the provided or generated captions

    Shortcut: invoking ``lai-youtube`` is equivalent to running ``lai alignment youtube``.

    Args:
        yt_url: YouTube video URL (can be provided as positional argument)
        media: Media configuration for controlling formats and output directories.
            Fields: input_path (YouTube URL), output_dir, output_format, force_overwrite
        client: API client configuration.
            Fields: api_key, timeout, max_retries
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name, device, batch_size
        caption: Caption configuration for reading/writing caption files.
            Fields: output_format, output_path, normalize_text,
                    split_sentence, word_level, encoding
        transcription: Transcription service configuration (enables Gemini transcription).
            Fields: gemini_api_key, model_name, language, device
        diarization: Speaker diarization configuration.
            Fields: enabled, num_speakers, min_speakers, max_speakers, device

    Examples:
        # Download from YouTube and align (positional argument)
        lai alignment youtube "https://www.youtube.com/watch?v=VIDEO_ID"

        # With custom output directory and format
        lai alignment youtube "https://www.youtube.com/watch?v=VIDEO_ID" \\
            media.output_dir=/tmp/youtube \\
            media.output_format=mp3

        # Full configuration with smart splitting and word-level alignment
        lai alignment youtube "https://www.youtube.com/watch?v=VIDEO_ID" \\
            caption.output_path=aligned.srt \\
            caption.split_sentence=true \\
            caption.word_level=true \\
            alignment.device=cuda

        # Use Gemini transcription (requires API key)
        lai alignment youtube "https://www.youtube.com/watch?v=VIDEO_ID" \\
            transcription.gemini_api_key=YOUR_KEY \\
            transcription.model_name=gemini-2.0-flash

        # Using keyword argument (traditional syntax)
        lai alignment youtube \\
            yt_url="https://www.youtube.com/watch?v=VIDEO_ID" \\
            alignment.device=mps
    """
    # Initialize configs with defaults
    media_config = media or MediaConfig()
    caption_config = caption or CaptionConfig()

    # Validate URL input: require exactly one of yt_url or media.input_path
    if yt_url and media_config.input_path:
        raise ValueError(
            "Cannot specify both positional yt_url and media.input_path. "
            "Use either positional argument or config, not both."
        )

    if not yt_url and not media_config.input_path:
        raise ValueError("YouTube URL is required. Provide either positional yt_url or media.input_path parameter.")

    # Assign yt_url to media_config.input_path if provided
    if yt_url:
        media_config.set_input_path(yt_url)

    # Create LattifAI client with all configurations
    lattifai_client = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        transcription_config=transcription,
        diarization_config=diarization,
    )
    # Call the client's youtube method
    return lattifai_client.youtube(
        url=media_config.input_path,
        output_dir=media_config.output_dir,
        output_caption_path=caption_config.output_path,
        media_format=media_config.normalize_format() if media_config.output_format else None,
        force_overwrite=media_config.force_overwrite,
        split_sentence=caption_config.split_sentence,
        channel_selector=media_config.channel_selector,
    )


def main():
    run.cli.main(youtube)


if __name__ == "__main__":
    main()
