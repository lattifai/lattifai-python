"""Alignment CLI entry point with nemo_run."""

from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, ClientConfig, MediaConfig, SubtitleConfig

__all__ = ["align"]


@run.cli.entrypoint(name="align", namespace="alignment")
def align(
    input_media_path: Optional[Path] = None,
    input_subtitle_path: Optional[Path] = None,
    output_subtitle_path: Optional[Path] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Align audio/video with subtitle file.

    This command performs forced alignment between audio/video media and subtitle text,
    generating accurate timestamps for each subtitle segment and optionally word-level
    timestamps. The alignment engine uses advanced speech recognition models to ensure
    precise synchronization between audio and text.

    Shortcut: invoking ``lai-align`` is equivalent to running ``lai alignment align``.

    Args:
        media: Media configuration for audio/video input and output handling.
            Fields: input_path, media_format, sample_rate, channels, output_dir,
                    output_path, output_format, prefer_audio, default_audio_format,
                    default_video_format, force_overwrite
        client: API client configuration.
            Fields: api_key, base_url, timeout, max_retries, default_headers
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name_or_path, device, batch_size
        subtitle: Subtitle I/O configuration (file reading/writing and formatting).
            Fields: input_format, input_path, output_format, output_path,
                    normalize_text, split_sentence, word_level,
                    include_speaker_in_text, encoding

    Examples:
        # Basic usage with positional arguments
        lai alignment align audio.wav subtitle.srt output.srt

        # Mixing positional and keyword arguments
        lai alignment align audio.mp4 subtitle.srt output.json \\
            alignment.device=cuda \\
            subtitle.word_level=true

        # Smart sentence splitting with custom output format
        lai alignment align audio.wav subtitle.srt output.vtt \\
            subtitle.split_sentence=true

        # Using keyword arguments (traditional syntax)
        lai alignment align \\
            input_media_path=audio.wav \\
            input_subtitle_path=subtitle.srt \\
            output_subtitle_path=output.srt

        # Full configuration with nested config objects
        lai alignment align audio.wav subtitle.srt aligned.json \\
            media.output_dir=/tmp/output \\
            subtitle.split_sentence=true \\
            subtitle.word_level=true \\
            subtitle.normalize_text=true \\
            alignment.device=mps \\
            alignment.model_name_or_path=Lattifai/Lattice-1-Alpha
    """
    media_config = media or MediaConfig()
    if not output_subtitle_path and not media_config.input_path:
        raise ValueError("Provide an input media path via argument or media.input-path configuration.")

    subtitle_config = subtitle or SubtitleConfig()

    client = LattifAI(client_config=client, alignment_config=alignment, subtitle_config=subtitle_config)

    return client.alignment(
        input_media_path=input_media_path or media_config.input_path,
        input_subtitle_path=input_subtitle_path or subtitle_config.input_path,
        output_subtitle_path=output_subtitle_path or subtitle_config.output_path,
    )


def main():
    run.cli.main(align)


if __name__ == "__main__":
    main()
