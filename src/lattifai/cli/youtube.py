"""YouTube workflow CLI entry point with nemo_run."""

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


async def _download_media_and_subtitles(url: str, config: MediaConfig, subtitle_config: SubtitleConfig) -> str:
    """
    Download media and optionally subtitles from YouTube.

    Args:
        url: YouTube URL to download from
        config: Media configuration for download settings
        subtitle_config: Subtitle configuration for subtitle download settings

    Returns:
        Path to the downloaded media file
    """
    downloader = YouTubeDownloader()
    media_format = config.normalize_format()

    # Download subtitles if not already provided in subtitle_config
    if not subtitle_config.input_path:
        subtitle_file = await downloader.download_subtitles(
            url=url,
            output_dir=str(config.output_dir),
            force_overwrite=config.force_overwrite,
            subtitle_lang=None,  # Download all available subtitles
            enable_gemini_option=False,
        )
        if subtitle_file and subtitle_file != "gemini":
            subtitle_config.input_path = subtitle_file

    # Download media
    media_file = await downloader.download_media(
        url=url,
        output_dir=str(config.output_dir),
        media_format=media_format,
        force_overwrite=config.force_overwrite,
    )

    return media_file


@run.cli.entrypoint(name="youtube", namespace="alignment")
def youtube(
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    subtitle: Annotated[Optional[SubtitleConfig], run.Config[SubtitleConfig]] = None,
):
    """
    Download media from YouTube (when needed) and align subtitles.

    This command provides a convenient workflow for aligning subtitles with YouTube videos.
    It can automatically download media from YouTube URLs, or work with pre-downloaded files.
    The command intelligently detects whether the input is a YouTube URL or a local file path.

    When a YouTube URL is provided:
    1. Downloads media in the specified format (audio or video)
    2. Optionally downloads available subtitles from YouTube
    3. Performs forced alignment with the provided or downloaded subtitles

    Shortcut: invoking ``lai-youtube`` is equivalent to running ``lai alignment youtube``.

    Args:
        media: Media configuration for controlling formats and output directories.
            Fields: input_path (YouTube URL or local file path), media_format,
                    sample_rate, channels, output_dir, output_path, output_format,
                    prefer_audio, default_audio_format, default_video_format,
                    force_overwrite
        client: API client configuration.
            Fields: api_key, base_url, timeout, max_retries, default_headers
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name_or_path, device, batch_size
        subtitle: Subtitle configuration for reading/writing subtitle files.
            Fields: input_format, input_path, output_format, output_path,
                    normalize_text, split_sentence, word_level,
                    include_speaker_in_text, encoding

    Examples:
        # Download from YouTube and align with existing subtitle
        lai alignment youtube https://youtu.be/VIDEO --subtitle.input-path=sub.srt

        # Use a pre-downloaded file
        lai alignment youtube /path/to/audio.mp3 --subtitle.input-path=sub.srt

        # Download as audio and enable word-level alignment
        lai alignment youtube https://youtu.be/VIDEO \\
            --media.prefer-audio=true \\
            --subtitle.input-path=sub.srt \\
            --subtitle.word-level=true

        # Override download format and use GPU acceleration
        lai alignment youtube https://youtu.be/VIDEO \\
            --media.output-format=mp3 \\
            --subtitle.input-path=sub.srt \\
            --subtitle.word-level=true \\
            --alignment.device=cuda

        # Full configuration example with custom output
        lai alignment youtube https://youtu.be/VIDEO \\
            media.output_dir=/tmp/youtube \\
            --media.output-format=wav \\
            --subtitle.input-path=subtitle.srt \\
            --subtitle.output-path=aligned.srt \\
            subtitle.split_sentence=true \\
            --subtitle.word-level=true \\
            --alignment.device=mps \\
            --alignment.model-name-or-path=Lattifai/Lattice-1-Alpha
    """
    media_config = media or MediaConfig()

    if not media_config.input_path:
        raise ValueError("Provide media.input_path=youtube_url to specify the YouTube video URL.")

    subtitle_config = subtitle or SubtitleConfig()

    source = media_config.input_path
    if not source:
        raise ValueError("Unable to determine media source. Ensure input path or media config is provided.")

    if media_config.is_input_remote() and _is_youtube_url(source):
        downloaded_path = asyncio.run(_download_media_and_subtitles(source, media_config, subtitle_config))
        media_config.set_input_path(downloaded_path)

    if not subtitle_config.output_path:
        from pathlib import Path

        media_name = Path(media_config.input_path).stem
        subtitle_config.output_path = (
            media_config.output_dir / f"{media_name}_LattifAI.{subtitle_config.output_format or 'srt'}"
        )

    client = LattifAI(client_config=client, alignment_config=alignment, subtitle_config=subtitle_config)

    return client.alignment(
        input_media_path=media_config.input_path,
        input_subtitle_path=subtitle_config.input_path,
        output_subtitle_path=subtitle_config.output_path,
    )


def main():
    run.cli.main(youtube)


if __name__ == "__main__":
    main()
