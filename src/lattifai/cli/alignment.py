"""Alignment CLI entry point with nemo_run."""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, ClientConfig, MediaConfig, SubtitleConfig

__all__ = ["align"]


@run.cli.entrypoint(name="align", namespace="alignment")
def align(
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Align audio/video with subtitle file.
    Shortcut: invoking ``lairun align`` is equivalent to running ``lairun lai align``.

    Args:
        alignment: Alignment configuration (includes API settings)
        subtitle: Subtitle configuration

    Examples:
        # Basic usage
        lairun align audio.wav subtitle.srt output.srt

        # With config overrides
        lairun align audio.wav subtitle.srt output.srt \
            --alignment.device=cuda \
            --alignment.word-level=true

        # With custom model and API settings
        lairun align audio.wav subtitle.srt output.srt \
            --alignment.model-name-or-path="Lattifai/Lattice-1" \
            --alignment.api-key="your-key"

        # Using media config for URL input
        lairun align --media.input-path="https://example.com/audio.mp3" subtitle.srt \
            --media.output-dir="/tmp/alignment"
    """
    media_config = media or MediaConfig()
    if not media_config.input_path:
        raise ValueError("Provide an input media path via argument or media.input-path configuration.")

    subtitle_config = subtitle or SubtitleConfig()

    client = LattifAI(client_config=client, alignment_config=alignment, subtitle_config=subtitle_config)

    return client.alignment(
        input_media_path=media_config.input_path,
        input_subtitle_path=subtitle_config.input_path,
        output_subtitle_path=subtitle_config.output_path,
    )


def main():
    run.cli.main(align)


if __name__ == "__main__":
    main()
