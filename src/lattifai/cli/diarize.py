"""Speaker diarization CLI entry point with nemo_run."""

import os
from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli._shared import build_lattifai_client, resolve_caption_paths, resolve_media_input
from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import AlignmentConfig, CaptionConfig, ClientConfig, DiarizationConfig, MediaConfig
from lattifai.theme import theme
from lattifai.utils import safe_print

__all__ = ["diarize"]


def _resolve_context(context: Optional[str]) -> Optional[str]:
    """Resolve context parameter: file path → parsed metadata string, or pass through.

    Supports .meta.md files with YAML frontmatter (title, channel, speakers, description).
    """
    if not context:
        return None

    if not os.path.isfile(context):
        return context  # Inline string — use as-is

    # Parse YAML frontmatter from .meta.md file
    try:
        text = Path(context).read_text(encoding="utf-8")
    except OSError:
        return None

    # Extract YAML frontmatter between --- delimiters
    if not text.startswith("---"):
        return text.strip() or None

    end = text.find("\n---", 3)
    if end < 0:
        return None

    frontmatter = text[3:end].strip()
    body = text[end + 4 :].strip()

    try:
        import yaml

        meta = yaml.safe_load(frontmatter)
    except Exception:
        return frontmatter  # Fall back to raw frontmatter text

    if not isinstance(meta, dict):
        return frontmatter

    # Build speaker context from structured metadata (reuse _build_speaker_context format)
    parts = []

    # Structured speakers
    speakers = meta.get("speakers")
    if speakers and isinstance(speakers, list):
        host_names = [s["name"] for s in speakers if s.get("role") == "host" and s.get("name")]
        guest_names = [s["name"] for s in speakers if s.get("role") == "guest" and s.get("name")]
        if host_names:
            parts.append(f"Channel/Host: {', '.join(host_names)}")
        if guest_names:
            parts.append(f"Guests: {', '.join(guest_names)}")

    title = meta.get("title")
    if title:
        parts.append(f"Title: {title}")

    channel = meta.get("channel") or meta.get("uploader")
    if channel and not any("Channel/Host:" in p for p in parts):
        parts.append(f"Channel/Host: {channel}")

    # Description (first few meaningful lines)
    desc = body or meta.get("description") or ""
    if desc:
        lines = [ln.strip() for ln in desc.split("\n") if ln.strip()]
        # Skip URLs, timestamps, sponsor blocks
        meaningful = []
        for ln in lines:
            if ln.startswith(("http", "#", "0:", "00:", "𝐒𝐏𝐎𝐍", "𝐓𝐈𝐌𝐄", "𝐄𝐏𝐈𝐒")):
                continue
            meaningful.append(ln)
            if len(meaningful) >= 3:
                break
        if meaningful:
            parts.append("Description:\n" + "\n".join(meaningful))

    return "\n".join(parts) if parts else None


@run.cli.entrypoint(name="run", namespace="diarize", entrypoint_cls=LattifAIEntrypoint)
def diarize(
    input_media: Optional[str] = None,
    input_caption: Optional[str] = None,
    output_caption: Optional[str] = None,
    context: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """Run speaker diarization on aligned captions and audio.

    Args:
        context: File path (.meta.md) or inline string providing speaker/media context.
            If the value is an existing file, its YAML frontmatter is parsed to
            extract title, channel, speakers, and description. Otherwise it is
            passed as a plain text hint to the speaker inference LLM.

    Speaker inference can be enabled via config.toml [diarization].infer_speakers
    or CLI: diarization.infer_speakers=true
    """

    # Resolve context: file path → parsed metadata, string → pass through
    speaker_context = _resolve_context(context)

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
            "Input caption path must be provided via positional input_caption or caption.input.path."
        ),
    )
    diarization_config = diarization or DiarizationConfig()

    diarization_config.enabled = True

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
