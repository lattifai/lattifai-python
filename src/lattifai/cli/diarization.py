"""Speaker diarization CLI entry point with nemo_run."""

from pathlib import Path
from typing import Optional

import colorful
import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig, DiarizationConfig, MediaConfig
from lattifai.utils import safe_print

__all__ = ["diarize"]


@run.cli.entrypoint(name="run", namespace="diarization")
def diarize(
    input_media: Optional[str] = None,
    input_caption: Optional[str] = None,
    output_caption: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """Run speaker diarization on aligned captions and audio."""

    media_config = media or MediaConfig()
    caption_config = caption or CaptionConfig()
    diarization_config = diarization or DiarizationConfig()

    if input_media and media_config.input_path:
        raise ValueError("Cannot specify both positional input_media and media.input_path.")
    if input_media:
        media_config.set_input_path(input_media)
    if not media_config.input_path:
        raise ValueError("Input media path must be provided via positional input_media or media.input_path.")

    if input_caption and caption_config.input_path:
        raise ValueError("Cannot specify both positional input_caption and caption.input_path.")
    if input_caption:
        caption_config.set_input_path(input_caption)
    if not caption_config.input_path:
        raise ValueError("Input caption path must be provided via positional input_caption or caption.input_path.")

    if output_caption and caption_config.output_path:
        raise ValueError("Cannot specify both positional output_caption and caption.output_path.")
    if output_caption:
        caption_config.set_output_path(output_caption)

    diarization_config.enabled = True

    client_instance = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        diarization_config=diarization_config,
    )

    safe_print(colorful.cyan("🎧 Loading media for diarization..."))
    media_audio = client_instance.audio_loader(
        media_config.input_path,
        channel_selector=media_config.channel_selector,
        streaming_chunk_secs=media_config.streaming_chunk_secs,
    )

    safe_print(colorful.cyan("📖 Loading caption segments..."))
    caption_obj = client_instance._read_caption(
        caption_config.input_path,
        input_caption_format=None if caption_config.input_format == "auto" else caption_config.input_format,
        verbose=False,
    )

    if not caption_obj.alignments:
        caption_obj.alignments = caption_obj.supervisions

    if not caption_obj.alignments:
        raise ValueError("Caption does not contain segments for diarization.")

    if caption_config.output_path:
        output_path = caption_config.output_path
    else:
        from datetime import datetime

        input_caption_path = Path(caption_config.input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        default_output = (
            input_caption_path.parent / f"{input_caption_path.stem}.diarized.{timestamp}.{caption_config.output_format}"
        )
        caption_config.set_output_path(default_output)
        output_path = caption_config.output_path

    safe_print(colorful.cyan("🗣️ Performing speaker diarization..."))
    diarized_caption = client_instance.speaker_diarization(
        input_media=media_audio,
        caption=caption_obj,
        output_caption_path=output_path,
    )

    return diarized_caption


def main():
    run.cli.main(diarize)


if __name__ == "__main__":
    main()
