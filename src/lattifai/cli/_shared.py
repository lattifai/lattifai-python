"""Shared helpers for CLI workflow commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lattifai.client import LattifAI
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
    MediaConfig,
    TranscriptionConfig,
)


def resolve_media_input(
    media: Optional[MediaConfig],
    positional_input: Optional[str],
    *,
    positional_name: str,
    required_message: str,
) -> MediaConfig:
    """Resolve media input from positional argument or MediaConfig."""
    media_config = media or MediaConfig()
    if positional_input and media_config.input_path:
        raise ValueError(f"Cannot specify both positional {positional_name} and media.input_path.")
    if positional_input:
        media_config.set_input_path(positional_input)
    if not media_config.input_path:
        raise ValueError(required_message)
    return media_config


def resolve_caption_paths(
    caption: Optional[CaptionConfig],
    *,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_name: str = "input_caption",
    output_name: str = "output_caption",
    require_input: bool = False,
    input_required_message: Optional[str] = None,
) -> CaptionConfig:
    """Resolve caption input/output paths from positional arguments or CaptionConfig."""
    caption_config = caption or CaptionConfig()

    if input_path and caption_config.input_path:
        raise ValueError(f"Cannot specify both positional {input_name} and caption.input.path.")
    if input_path:
        caption_config.set_input_path(input_path)
    if require_input and not caption_config.input_path:
        raise ValueError(input_required_message or "Input caption path is required.")

    if output_path and caption_config.output_path:
        raise ValueError(f"Cannot specify both positional {output_name} and caption.output.path.")
    if output_path:
        caption_config.set_output_path(output_path)

    return caption_config


def build_lattifai_client(
    *,
    client: Optional[ClientConfig] = None,
    alignment: Optional[AlignmentConfig] = None,
    caption: Optional[CaptionConfig] = None,
    transcription: Optional[TranscriptionConfig] = None,
    diarization: Optional[DiarizationConfig] = None,
    event: Optional[EventConfig] = None,
) -> LattifAI:
    """Build a LattifAI client from CLI config objects."""
    return LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption,
        transcription_config=transcription,
        diarization_config=diarization,
        event_config=event,
    )


def run_youtube_workflow(
    *,
    media: MediaConfig,
    caption: CaptionConfig,
    client: Optional[ClientConfig] = None,
    alignment: Optional[AlignmentConfig] = None,
    transcription: Optional[TranscriptionConfig] = None,
    diarization: Optional[DiarizationConfig] = None,
    event: Optional[EventConfig] = None,
    use_transcription: bool = False,
):
    """Run the shared YouTube workflow used by multiple CLI commands."""
    lattifai_client = build_lattifai_client(
        client=client,
        alignment=alignment,
        caption=caption,
        transcription=transcription,
        diarization=diarization,
        event=event,
    )
    return lattifai_client.youtube(
        url=media.input_path,
        output_dir=media.output_dir,
        output_caption_path=caption.output_path,
        media_format=media.normalize_format() if media.output_format else None,
        force_overwrite=media.force_overwrite,
        split_sentence=caption.split_sentence,
        channel_selector=media.channel_selector,
        streaming_chunk_secs=media.streaming_chunk_secs,
        use_transcription=use_transcription,
        audio_track_id=media.audio_track_id,
        quality=media.quality,
    )


def ensure_parent_dir(path: Path) -> Path:
    """Create the parent directory for a file path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
