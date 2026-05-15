"""Speaker diarization CLI entry point with nemo_run."""

import os
from pathlib import Path
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.cli._shared import (
    build_lattifai_client,
    resolve_caption_paths,
    resolve_media_input,
)
from lattifai.cli.entrypoint import LattifAIEntrypoint
from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    MediaConfig,
)
from lattifai.theme import theme
from lattifai.utils import safe_print

__all__ = ["diarize", "naming"]


_DESCRIPTION_CHAR_CAP = 800
_DESCRIPTION_SKIP_PREFIXES = ("http", "#", "0:", "00:", "𝐒𝐏𝐎𝐍", "𝐓𝐈𝐌𝐄", "𝐄𝐏𝐈𝐒")

# Hard caps on per-field LLM prompt inflation. A malformed or oversized meta.md
# must not be able to push the prompt past these limits.
_MAX_ALIASES_PER_SPEAKER = 8
_MAX_ALIAS_CHARS = 40
_MAX_BIO_CHARS = 300
_MAX_TOPICS = 20
_MAX_TOPIC_CHARS = 50
_MAX_PRIOR_EPISODES = 5
_MAX_PRIOR_EPISODE_CHARS = 200


def _filter_description_lines(text: str) -> str:
    """Drop URL/timestamp/sponsor noise lines, cap at _DESCRIPTION_CHAR_CAP chars.

    The cap is inclusive of any trailing ellipsis — a truncated output is
    guaranteed to be no longer than ``_DESCRIPTION_CHAR_CAP`` characters.
    """
    meaningful: list = []
    used = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(_DESCRIPTION_SKIP_PREFIXES):
            continue
        if used + len(stripped) + 1 > _DESCRIPTION_CHAR_CAP:
            # Reserve one char for the ellipsis so the total stays ≤ cap.
            remaining = _DESCRIPTION_CHAR_CAP - used - 1
            if remaining > 20:
                meaningful.append(stripped[:remaining].rstrip() + "…")
            break
        meaningful.append(stripped)
        used += len(stripped) + 1
    return "\n".join(meaningful)


def _format_speakers_block(speakers: list) -> str:
    """Render structured speakers as the 'Speakers:' block consumed by
    `extract_candidate_names._parse_speakers_block`.

    Layout per entry::

        - <name> (<role>) @ <affiliation>
          aliases: <alias>, <alias>
          bio: <bio text>
    """
    lines = ["Speakers:"]
    for sp in speakers:
        if not isinstance(sp, dict):
            continue
        name = str(sp.get("name") or "").strip()
        if not name:
            continue
        role = str(sp.get("role") or "").strip().lower()
        affiliation = str(sp.get("affiliation") or "").strip()
        header = f"- {name}"
        if role:
            header += f" ({role})"
        if affiliation:
            header += f" @ {affiliation}"
        lines.append(header)
        aliases = sp.get("aliases")
        if aliases:
            if isinstance(aliases, str):
                alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
            elif isinstance(aliases, list):
                alias_list = [str(a).strip() for a in aliases if str(a).strip()]
            else:
                alias_list = []
            if alias_list:
                # Cap count and per-item length to bound prompt size.
                bounded = [a[:_MAX_ALIAS_CHARS] for a in alias_list[:_MAX_ALIASES_PER_SPEAKER]]
                lines.append(f"  aliases: {', '.join(bounded)}")
        bio = str(sp.get("bio") or "").strip()
        if bio:
            # Single-line bio so the line-based parser doesn't split it across entries.
            bio_single = " ".join(bio.split())
            lines.append(f"  bio: {bio_single[:_MAX_BIO_CHARS]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _resolve_context(context: Optional[str]) -> Optional[str]:
    """Resolve context parameter: file path → parsed metadata string, or pass through.

    Supports .meta.md files with YAML frontmatter. Recognised keys:
        title, channel / uploader, speakers (list of {name, role, affiliation,
        aliases, bio}), topics (list), prior_episodes (list), description.
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

    # Build speaker context from structured metadata. Legacy lines
    # (Channel/Host, Title, Guests) are retained verbatim so the regex paths in
    # `extract_candidate_names` keep working; the new `Speakers:` block carries
    # the rich per-speaker signals (affiliation / aliases / bio).
    parts: list = []

    speakers = meta.get("speakers")
    structured_speakers: list = []
    if speakers and isinstance(speakers, list):
        host_names = [s["name"] for s in speakers if isinstance(s, dict) and s.get("role") == "host" and s.get("name")]
        guest_names = [
            s["name"] for s in speakers if isinstance(s, dict) and s.get("role") == "guest" and s.get("name")
        ]
        if host_names:
            parts.append(f"Channel/Host: {', '.join(host_names)}")
        if guest_names:
            parts.append(f"Guests: {', '.join(guest_names)}")
        structured_speakers = [s for s in speakers if isinstance(s, dict) and s.get("name")]

    title = meta.get("title")
    if title:
        parts.append(f"Title: {title}")

    channel = meta.get("channel") or meta.get("uploader")
    if channel and not any("Channel/Host:" in p for p in parts):
        parts.append(f"Channel/Host: {channel}")

    if structured_speakers:
        speakers_block = _format_speakers_block(structured_speakers)
        if speakers_block:
            parts.append(speakers_block)

    topics = meta.get("topics")
    if topics and isinstance(topics, list):
        topic_strs = [str(t).strip()[:_MAX_TOPIC_CHARS] for t in topics if str(t).strip()]
        if topic_strs:
            parts.append(f"Topics: {', '.join(topic_strs[:_MAX_TOPICS])}")

    prior_episodes = meta.get("prior_episodes")
    if prior_episodes and isinstance(prior_episodes, list):
        prior_lines = [f"- {str(ep).strip()[:_MAX_PRIOR_EPISODE_CHARS]}" for ep in prior_episodes if str(ep).strip()][
            :_MAX_PRIOR_EPISODES
        ]
        if prior_lines:
            parts.append("Prior episodes:\n" + "\n".join(prior_lines))

    desc = body or meta.get("description") or ""
    if desc:
        filtered = _filter_description_lines(desc)
        if filtered:
            parts.append("Description:\n" + filtered)

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


@run.cli.entrypoint(name="naming", namespace="diarize", entrypoint_cls=LattifAIEntrypoint)
def naming(
    input_caption: Optional[str] = None,
    output_caption: Optional[str] = None,
    context: Optional[str] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
):
    """Infer real speaker names from a diarized caption file using LLM.

    Reads a caption file with speaker labels (e.g. SPEAKER_00, SPEAKER_01),
    then uses an LLM to identify each speaker's real name from conversation
    content and optional metadata context.

    No audio input required — operates purely on caption text.

    Args:
        input_caption: Path to diarized caption file (SRT, JSON, VTT, etc.).
        output_caption: Optional output path. If omitted, prints the mapping
            and rewrites in-place.
        context: File path (.meta.md) or inline string providing speaker/media
            context (title, channel, speakers, description).

    Examples:
        lai diarize naming episode.diarized.json
        lai diarize naming episode.srt --context="Lex Fridman interviews Sam Altman"
        lai diarize naming episode.json episode.named.json context=metadata.meta.md
    """
    from lattifai.caption import Caption
    from lattifai.diarization.speaker import infer_speaker_names

    caption_config = resolve_caption_paths(
        caption,
        input_path=input_caption,
        output_path=output_caption,
        require_input=True,
        input_required_message="Input caption path required: lai diarize naming <caption_file>",
    )
    diarization_config = diarization or DiarizationConfig()

    # Resolve context: file path → parsed metadata, string → pass through
    speaker_context = _resolve_context(context)

    safe_print(theme.step(f"Loading caption: {caption_config.input_path}"))
    cap = Caption.read(str(caption_config.input_path))

    if not cap.supervisions:
        raise ValueError("Caption file contains no segments.")

    # Check speakers exist
    speaker_labels = {sup.speaker for sup in cap.supervisions if sup.speaker}
    if len(speaker_labels) < 2:
        safe_print(theme.warn(f"Only {len(speaker_labels)} speaker(s) found — naming requires 2+ speakers."))
        return cap

    safe_print(theme.step(f"Found {len(speaker_labels)} speakers: {', '.join(sorted(speaker_labels))}"))

    # Build LLM client from diarization config
    if diarization_config.llm is None:
        from lattifai.config.diarization import DiarizationLLMConfig

        diarization_config.llm = DiarizationLLMConfig()
    llm_client = diarization_config.llm.create_client()

    safe_print(theme.step(f"Inferring speaker names via LLM [{diarization_config.llm.model_name}]..."))
    name_map = infer_speaker_names(
        supervisions=cap.supervisions,
        context=speaker_context,
        llm_client=llm_client,
    )

    if not name_map:
        safe_print(theme.warn("Could not infer any speaker names."))
        return cap

    # Display results
    safe_print(theme.ok("\nSpeaker name mapping:"))
    for label, name in sorted(name_map.items()):
        safe_print(f"  {label} → {name}")

    # Apply mapping to supervisions
    for sup in cap.supervisions:
        if sup.speaker in name_map:
            sup.speaker = name_map[sup.speaker]

    # Write output. Caption.write infers the output format from the path
    # suffix (.json, .srt, .vtt, …) — passing an `output_format` kwarg crashes
    # because Caption.write does not accept it.
    output_path = caption_config.output_path or caption_config.input_path
    cap.write(str(output_path))
    safe_print(theme.ok(f"Saved: {output_path}"))

    return cap


def main():
    run.cli.main(diarize)


if __name__ == "__main__":
    main()
