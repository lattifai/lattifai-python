"""Speaker diarization CLI entry point with nemo_run."""

from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli._shared import build_lattifai_client, resolve_caption_paths, resolve_media_input
from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig, DiarizationConfig, MediaConfig
from lattifai.theme import theme
from lattifai.utils import safe_print

__all__ = ["diarize"]


@run.cli.entrypoint(name="run", namespace="diarization")
def diarize(
    input_media: Optional[str] = None,
    input_caption: Optional[str] = None,
    output_caption: Optional[str] = None,
    infer_speakers: bool = False,
    speaker_context: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """Run speaker diarization on aligned captions and audio."""

    media_config = resolve_media_input(
        media,
        input_media,
        positional_name="input_media",
        required_message="Input media path must be provided via positional input_media or media.input_path.",
    )
    caption_config = resolve_caption_paths(
        caption,
        input_path=input_caption,
        output_path=output_caption,
        require_input=True,
        input_required_message=(
            "Input caption path must be provided via positional input_caption or caption.input_path."
        ),
    )
    diarization_config = diarization or DiarizationConfig()

    diarization_config.enabled = True
    if infer_speakers:
        diarization_config.infer_speakers = True

    client_instance = build_lattifai_client(
        client=client,
        alignment=alignment,
        caption=caption_config,
        diarization=diarization_config,
    )

    safe_print(theme.step("🎧 Loading media for diarization..."))
    media_audio = client_instance.audio_loader(
        media_config.input_path,
        channel_selector=media_config.channel_selector,
        streaming_chunk_secs=media_config.streaming_chunk_secs,
    )

    safe_print(theme.step("📖 Loading caption segments..."))
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

    safe_print(theme.step("🗣️ Performing speaker diarization..."))
    diarized_caption = client_instance.speaker_diarization(
        input_media=media_audio,
        caption=caption_obj,
        output_caption_path=output_path,
        speaker_context=speaker_context,
    )

    return diarized_caption


def main():
    run.cli.main(diarize)


if __name__ == "__main__":
    main()
