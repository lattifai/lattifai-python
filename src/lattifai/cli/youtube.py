"""YouTube workflow CLI entry point with nemo_run."""

from __future__ import annotations

import asyncio
from typing import Optional

import nemo_run as run
from typing_extensions import Annotated

from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, ClientConfig, MediaConfig, SubtitleConfig
from lattifai.workflow.youtube import YouTubeDownloader


def _is_youtube_url(value: str) -> bool:
    """Return True when the provided URL looks like a YouTube link."""
    lowered = value.lower()
    return any(host in lowered for host in ("youtube.com", "youtu.be"))


async def _download_media_with_config(url: str, config: MediaConfig) -> str:
    """Download media from YouTube based on media configuration."""
    downloader = YouTubeDownloader()
    media_format = config.normalize_format()
    return await downloader.download_media(
        url=url,
        output_dir=str(config.output_dir),
        media_format=media_format,
        force_overwrite=config.force_overwrite,
    )


@run.cli.entrypoint(name="youtube", namespace="alignment")
def youtube(
    yt_url: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Download media from YouTube (when needed) and align subtitles.

    Args:
        yt_url: Local path or URL to audio/video content.
        media: Media configuration for controlling formats and output directories.
        client: API client configuration.
        alignment: Alignment configuration (includes model and API settings).
        subtitle: Subtitle configuration used for reading/writing subtitle files.

    Examples:
        # Download from YouTube and align with existing subtitle
        lai youtube --media.input-path="https://youtu.be/VIDEO"

        # Use a pre-downloaded file
        lai youtube /path/to/audio.mp3

        # Override download format
        lai youtube --media.input-path="https://youtu.be/VIDEO"
    """
    media_config = media or MediaConfig()

    if yt_url is not None:
        media_config.set_input_path(yt_url)
    elif not media_config.input_path:
        raise ValueError("Provide an input media path via argument or media.input-path configuration.")

    subtitle_config = subtitle or SubtitleConfig()

    source = media_config.input_path
    if not source:
        raise ValueError("Unable to determine media source. Ensure input path or media config is provided.")

    if media_config.is_input_remote() and _is_youtube_url(source):
        downloaded_path = asyncio.run(_download_media_with_config(source, media_config))
        media_config.set_input_path(downloaded_path)

    client = LattifAI(client_config=client, alignment_config=alignment, subtitle_config=subtitle_config)

    return client.alignment(
        yt_url=media_config.input_path,
        input_subtitle_path=subtitle_config.input_path,
        output_subtitle_path=subtitle_config.output_path,
    )


def main():
    run.cli.main(youtube)


if __name__ == "__main__":
    main()
