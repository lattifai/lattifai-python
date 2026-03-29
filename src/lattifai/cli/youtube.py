"""YouTube workflow CLI entry point with nemo_run."""

from typing import Literal, Optional

import nemo_run as run
from typing_extensions import Annotated

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


@run.cli.entrypoint(name="alignment", namespace="youtube")
def youtube(
    yt_url: Optional[str] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    client: Annotated[Optional[ClientConfig], run.Config[ClientConfig]] = None,
    alignment: Annotated[Optional[AlignmentConfig], run.Config[AlignmentConfig]] = None,
    caption: Annotated[Optional[CaptionConfig], run.Config[CaptionConfig]] = None,
    transcription: Annotated[Optional[TranscriptionConfig], run.Config[TranscriptionConfig]] = None,
    diarization: Annotated[Optional[DiarizationConfig], run.Config[DiarizationConfig]] = None,
    event: Annotated[Optional[EventConfig], run.Config[EventConfig]] = None,
    use_transcription: bool = False,
):
    """
    Download media from YouTube (when needed) and align captions.

    This command provides a convenient workflow for aligning captions with YouTube videos.
    It can automatically download media from YouTube URLs and optionally transcribe audio
    using Gemini or download available captions from YouTube.

    When a YouTube URL is provided:
    1. Downloads media in the specified format (audio or video)
    2. Optionally transcribes audio with Gemini OR downloads YouTube captions
    3. Performs forced alignment with the provided or generated captions

    Shortcut: invoking ``lai-youtube`` is equivalent to running ``lai youtube alignment``.

    Args:
        yt_url: YouTube video URL (can be provided as positional argument)
        media: Media configuration for controlling formats and output directories.
            Fields: input_path (YouTube URL), output_dir, output_format, force_overwrite,
                    audio_track_id (default: "original"), quality (default: "best")
        client: API client configuration.
            Fields: api_key, timeout, max_retries
        alignment: Alignment configuration (model selection and inference settings).
            Fields: model_name, device, batch_size
        caption: Caption configuration for reading/writing caption files.
            Fields: output_format, output_path, normalize_text,
                    split_sentence, word_level, encoding
        transcription: Transcription service configuration (enables Gemini transcription).
            Fields: gemini_api_key, model_name, language, device
        diarization: Speaker diarization configuration.
            Fields: enabled, num_speakers, min_speakers, max_speakers, device
        use_transcription: If True, skip YouTube caption download and directly use
            transcription.model_name to transcribe. If False (default), first try to
            download YouTube captions; if download fails (no captions available or
            errors like HTTP 429), automatically fallback to transcription if
            transcription.model_name is configured.

    Examples:
        # Download from YouTube and align (positional argument)
        lai youtube alignment "https://www.youtube.com/watch?v=VIDEO_ID"

        # With custom output directory and format
        lai youtube alignment "https://www.youtube.com/watch?v=VIDEO_ID" \\
            media.output_dir=/tmp/youtube \\
            media.output_format=mp3

        # Full configuration with smart splitting and word-level alignment
        lai youtube alignment "https://www.youtube.com/watch?v=VIDEO_ID" \\
            caption.output_path=aligned.srt \\
            caption.split_sentence=true \\
            caption.word_level=true \\
            alignment.device=cuda

        # Use Gemini transcription (requires API key)
        lai youtube alignment "https://www.youtube.com/watch?v=VIDEO_ID" \\
            transcription.gemini_api_key=YOUR_KEY \\
            transcription.model_name=gemini-2.0-flash

        # Using keyword argument (traditional syntax)
        lai alignment youtube \\
            yt_url="https://www.youtube.com/watch?v=VIDEO_ID" \\
            alignment.device=mps
    """
    # Initialize configs with defaults
    media_config = media or MediaConfig()
    caption_config = caption or CaptionConfig()

    # Validate URL input: require exactly one of yt_url or media.input_path
    if yt_url and media_config.input_path:
        raise ValueError(
            "Cannot specify both positional yt_url and media.input_path. "
            "Use either positional argument or config, not both."
        )

    if not yt_url and not media_config.input_path:
        raise ValueError("YouTube URL is required. Provide either positional yt_url or media.input_path parameter.")

    # Assign yt_url to media_config.input_path if provided
    if yt_url:
        media_config.set_input_path(yt_url)

    # Create LattifAI client with all configurations
    lattifai_client = LattifAI(
        client_config=client,
        alignment_config=alignment,
        caption_config=caption_config,
        transcription_config=transcription,
        diarization_config=diarization,
        event_config=event,
    )

    # Call the client's youtube method
    # If use_transcription=True, skip YouTube caption download and use transcription directly.
    # If use_transcription=False (default), try YouTube captions first; on failure,
    # automatically fallback to transcription if transcription.model_name is configured.
    return lattifai_client.youtube(
        url=media_config.input_path,
        output_dir=media_config.output_dir,
        output_caption_path=caption_config.output_path,
        media_format=media_config.normalize_format() if media_config.output_format else None,
        force_overwrite=media_config.force_overwrite,
        split_sentence=caption_config.split_sentence,
        channel_selector=media_config.channel_selector,
        streaming_chunk_secs=media_config.streaming_chunk_secs,
        use_transcription=use_transcription,
        audio_track_id=media_config.audio_track_id,
        quality=media_config.quality,
    )


@run.cli.entrypoint(name="download", namespace="youtube")
def youtube_download(
    yt_url: Optional[str] = None,
    only: Optional[Literal["media", "caption", "transcript", "meta"]] = None,
    media: Annotated[Optional[MediaConfig], run.Config[MediaConfig]] = None,
    source_lang: Optional[str] = None,
):
    """
    Download media, captions, and metadata from YouTube (no alignment).

    Downloads:
    1. Audio/video file
    2. YouTube captions (.vtt/.srt)
    3. External transcript (.transcript.md) if URL found in description
    4. Video metadata embedded in transcript frontmatter

    All files are saved to the output directory. Use ``lai youtube alignment``
    to also perform forced alignment after downloading.

    Args:
        yt_url: YouTube video URL
        only: Download only a specific part: "media", "caption", "transcript", or "meta".
            If None (default), download all.
        media: Media configuration (output_dir, output_format, quality)
        source_lang: Caption language to download (e.g., 'zh-Hans', 'en', 'zh-CN').
            If None (default), auto-detects from video metadata.

    Examples:
        lai youtube download "https://www.youtube.com/watch?v=VIDEO_ID"

        lai youtube download "https://www.youtube.com/watch?v=VIDEO_ID" \\
            media.output_dir=./downloads media.output_format=mp3
    """
    import asyncio

    import colorful

    from lattifai.utils import safe_print
    from lattifai.youtube.client import YouTubeDownloader

    media_config = media or MediaConfig()

    if yt_url and media_config.input_path:
        raise ValueError("Cannot specify both positional yt_url and media.input_path.")
    if not yt_url and not media_config.input_path:
        raise ValueError("YouTube URL is required.")
    url = yt_url or media_config.input_path

    output_dir = media_config.output_dir or "."
    downloader = YouTubeDownloader()
    video_id = downloader.extract_video_id(url)

    if only and only not in ("media", "caption", "transcript", "meta"):
        raise ValueError(f"Invalid only={only!r}. Must be 'media', 'caption', 'transcript', or 'meta'.")

    safe_print(colorful.cyan(f"📥 Downloading YouTube video: {video_id}"))

    # Fetch video info once (used by transcript download and metadata save)
    info = asyncio.get_event_loop().run_until_complete(downloader.get_video_info(url))

    media_file = None
    caption_file = None
    transcript_file = None

    # 1. Download media
    if not only or only == "media":
        safe_print(colorful.cyan("🎵 Downloading media..."))
        media_format = media_config.normalize_format() if media_config.output_format else None
        media_file = asyncio.get_event_loop().run_until_complete(
            downloader.download_media(
                url,
                output_dir=output_dir,
                media_format=media_format,
                quality=media_config.quality,
                audio_track_id=media_config.audio_track_id,
            )
        )
        if media_file:
            safe_print(colorful.green(f"  ✅ Media: {media_file}"))

    # 2. Download captions (includes external transcript internally)
    if not only or only == "caption":
        safe_print(colorful.cyan("📝 Downloading captions..."))
        caption_file = asyncio.get_event_loop().run_until_complete(
            downloader.download_captions(
                url,
                output_dir=output_dir,
                force_overwrite=media_config.force_overwrite,
                source_lang=source_lang,
            )
        )
        if caption_file:
            safe_print(colorful.green(f"  ✅ Caption: {caption_file}"))

    # 3. Download only external transcript from video description
    if only == "transcript":
        safe_print(colorful.cyan("📄 Downloading external transcript..."))
        description = info.get("description", "")
        transcript_url = downloader._extract_transcript_url_from_description(description)
        if transcript_url:
            safe_print(colorful.cyan(f"  🔗 Found: {transcript_url}"))
            transcript_file = asyncio.get_event_loop().run_until_complete(
                downloader._download_external_transcript(
                    transcript_url,
                    output_dir,
                    video_id,
                    youtube_url=url,
                    video_info=info,
                    force_overwrite=media_config.force_overwrite,
                )
            )
            if transcript_file:
                safe_print(colorful.green(f"  ✅ Transcript: {transcript_file}"))
        else:
            # Fallback: try podscripts.co
            safe_print(colorful.cyan("  🔍 No transcript in description, trying podscripts.co..."))
            podscripts_url = asyncio.get_event_loop().run_until_complete(
                downloader._find_podscripts_url(info.get("title", ""), info.get("uploader", ""))
            )
            if podscripts_url:
                safe_print(colorful.cyan(f"  🔗 Found: {podscripts_url}"))
                transcript_file = asyncio.get_event_loop().run_until_complete(
                    downloader._download_external_transcript(
                        podscripts_url,
                        output_dir,
                        video_id,
                        youtube_url=url,
                        video_info=info,
                        force_overwrite=media_config.force_overwrite,
                    )
                )
                if transcript_file:
                    safe_print(colorful.green(f"  ✅ Transcript: {transcript_file}"))
            else:
                safe_print(colorful.yellow(f"  ⚠️ No transcript found for: {url}"))

    # 4. Save video metadata as YAML frontmatter markdown
    if not only or only == "meta":
        from pathlib import Path

        meta_path = Path(output_dir) / f"{video_id}.meta.md"

        # Format duration as HH:MM:SS
        duration = info.get("duration", 0)
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

        meta_lines = ["---"]
        meta_lines.append(f"title: \"{info.get('title', '')}\"")
        meta_lines.append(f"channel: \"{info.get('uploader', '')}\"")
        meta_lines.append(f"url: \"{info.get('webpage_url', '')}\"")
        meta_lines.append(f'duration: "{duration_str}"')
        meta_lines.append(f"upload_date: \"{info.get('upload_date', '')}\"")
        meta_lines.append(f"view_count: {info.get('view_count', 0)}")
        meta_lines.append(f"thumbnail: \"{info.get('thumbnail', '')}\"")
        meta_lines.append("---")
        meta_lines.append("")

        description = info.get("description", "")
        if description:
            meta_lines.append(description)
            meta_lines.append("")

        meta_path.write_text("\n".join(meta_lines), encoding="utf-8")
        safe_print(colorful.green(f"  ✅ Metadata: {meta_path}"))

    safe_print(colorful.green(f"\n✅ All files saved to: {output_dir}"))
    return media_file or caption_file or transcript_file


def main():
    run.cli.main(youtube)


if __name__ == "__main__":
    main()
