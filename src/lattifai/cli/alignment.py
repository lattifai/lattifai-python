"""Alignment CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from lhotse.utils import Pathlike
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

__all__ = ["align"]


@run.cli.entrypoint(name="align", namespace="alignment")
def align(
    input_media: Optional[str] = None,
    input_caption: Optional[str] = None,
    output_caption: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """
    Align audio/video with caption file.

    This command performs forced alignment between audio/video media and caption text,
    generating accurate timestamps for each caption segment and optionally word-level
    timestamps. The alignment engine uses advanced speech recognition models to ensure
    precise synchronization between audio and text.

    Shortcut: invoking ``lai-align`` is equivalent to running ``lai alignment align``.

    Args:
        media: Media configuration for audio/video input and output handling.
            Fields: input_path, media_format, sample_rate, channels, output_dir,
                    output_path, output_format, prefer_audio, default_audio_format,
                    default_video_format, force_overwrite
        client: API client configuration.
            Fields: api_key, timeout, max_retries, default_headers
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name, device, batch_size
        caption: Caption I/O configuration (file reading/writing and formatting).
            Fields: input_format, input_path, output_format, output_path,
                    normalize_text, split_sentence, word_level,
                    include_speaker_in_text, encoding

    Examples:
        # Basic usage with positional arguments
        lai alignment align audio.wav caption.srt output.srt

        # Mixing positional and keyword arguments
        lai alignment align audio.mp4 caption.srt output.json \\
            alignment.device=cuda \\
            caption.word_level=true

        # Smart sentence splitting with custom output format
        lai alignment align audio.wav caption.srt output.vtt \\
            caption.split_sentence=true

        # Using keyword arguments (traditional syntax)
        lai alignment align \\
            input_media=audio.wav \\
            input_caption=caption.srt \\
            output_caption=output.srt

        # Full configuration with nested config objects
        lai alignment align audio.wav caption.srt aligned.json \\
            media.output_dir=/tmp/output \\
            caption.split_sentence=true \\
            caption.word_level=true \\
            caption.normalize_text=true \\
            alignment.device=mps \\
            alignment.model_name=Lattifai/Lattice-1-Alpha
    """
    media_config = media or MediaConfig()

    # Validate that input_media and media_config.input_path are not both provided
    if input_media and media_config.input_path:
        raise ValueError(
            "Cannot specify both positional input_media and media.input_path. "
            "Use either positional argument or config, not both."
        )

    # Assign input_media to media_config.input_path if provided
    if input_media:
        media_config.set_input_path(input_media)

    if not media_config.input_path:
        raise ValueError("Input media path must be specified via positional argument input_media= or media.input_path=")

    caption_config = caption or CaptionConfig()

    # Validate that output_caption_path and caption_config.output_path are not both provided
    if output_caption and caption_config.output_path:
        raise ValueError(
            "Cannot specify both positional output_caption and caption.output_path. "
            "Use either positional argument or config, not both."
        )

    # Assign paths to caption_config if provided
    if input_caption:
        caption_config.set_input_path(input_caption)

    if output_caption:
        caption_config.set_output_path(output_caption)

    client = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        transcription_config=transcription,
        diarization_config=diarization,
    )

    is_url = media_config.input_path.startswith(("http://", "https://"))
    if is_url:
        # Call the client's youtube method
        return client.youtube(
            url=media_config.input_path,
            output_dir=media_config.output_dir,
            output_caption_path=caption_config.output_path,
            media_format=media_config.normalize_format() if media_config.output_format else None,
            force_overwrite=media_config.force_overwrite,
            split_sentence=caption_config.split_sentence,
            channel_selector=media_config.channel_selector,
        )

    return client.alignment(
        input_media=media_config.input_path,
        input_caption=caption_config.input_path,
        output_caption_path=caption_config.output_path,
        split_sentence=caption_config.split_sentence,
        channel_selector=media_config.channel_selector,
    )


def main():
    run.cli.main(align)


if __name__ == "__main__":
    main()
