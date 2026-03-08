"""Podcast transcription CLI entry point with nemo_run.

Thin wrapper around the existing alignment pipeline. The podcast-specific logic
(metadata extraction, audio download, speaker identification) lives in
client.podcast(); this CLI merely wires up the nemo_run config objects and
delegates.  For pure transcription without alignment, use ``lai transcribe run``
with ``transcription.prompt=<podcast_prompt>`` and ``transcription.description=<show_notes>``.
"""

from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.config import (
    AlignmentConfig,
    CaptionConfig,
    ClientConfig,
    DiarizationConfig,
    EventConfig,
    MediaConfig,
    PodcastConfig,
    TranscriptionConfig,
)


@run.cli.entrypoint(name="transcribe", namespace="podcast")
def podcast_transcribe(
    url: Optional[str] = None,
    audio: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
    event: Annotated[Optional[EventConfig], run.Config[EventConfig]] = None,
    podcast: Annotated[Optional[PodcastConfig], run.Config[PodcastConfig]] = None,
):
    """
    Transcribe a podcast episode with speaker identification.

    Wraps the standard alignment pipeline with podcast-specific preprocessing
    (metadata extraction, audio download) and post-processing (speaker naming).

    Supports: Apple Podcasts, RSS feeds, Xiaoyuzhou, local audio files.
    YouTube URLs are auto-delegated to the youtube workflow.

    For transcription-only (no alignment), use ``lai transcribe run`` with::

        transcription.prompt=<path/to/podcast_transcription.txt>
        transcription.description="Host: Alice\\nGuest: Bob\\n..."

    Examples:
        lai podcast transcribe "https://podcasts.apple.com/us/podcast/..."

        lai podcast transcribe "https://feeds.example.com/podcast.xml"

        lai podcast transcribe audio=episode.mp3 \\
            podcast.show_notes="Host Alice interviews Bob about AI"

        lai podcast transcribe "https://..." \\
            podcast.host_names='["Alice"]' podcast.guest_names='["Bob"]'

        lai podcast transcribe "https://..." \\
            diarization.enabled=true podcast.identify_speakers=true
    """
    from lattifai.client import LattifAI

    media_config = media or MediaConfig()
    caption_config = caption or CaptionConfig()
    podcast_config = podcast or PodcastConfig()

    if url and audio:
        raise ValueError("Cannot specify both url and audio. Use one or the other.")
    if not url and not audio and not media_config.input_path:
        raise ValueError("Provide a podcast URL, audio file path, or media.input_path.")

    lattifai_client = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        transcription_config=transcription,
        diarization_config=diarization,
        event_config=event,
        podcast_config=podcast_config,
    )

    return lattifai_client.podcast(
        url=url,
        input_media=audio or media_config.input_path or None,
        output_dir=media_config.output_dir,
        output_caption_path=caption_config.output_path,
        split_sentence=caption_config.split_sentence,
        channel_selector=media_config.channel_selector,
        streaming_chunk_secs=media_config.streaming_chunk_secs,
    )


def main():
    run.cli.main(podcast_transcribe)


if __name__ == "__main__":
    main()
