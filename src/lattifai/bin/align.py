import asyncio
import os

import click
import colorful
from lhotse.utils import Pathlike

from lattifai.bin.cli_base import cli
from lattifai.client import AsyncLattifAI, LattifAI
from lattifai.io import INPUT_SUBTITLE_FORMATS, OUTPUT_SUBTITLE_FORMATS


@cli.command()
@click.option(
    "-F",
    "--input_format",
    "--input-format",
    type=click.Choice(INPUT_SUBTITLE_FORMATS, case_sensitive=False),
    default="auto",
    help="Input subtitle format.",
)
@click.option(
    "-S",
    "--split-sentence",
    "--split_sentence",
    is_flag=True,
    default=False,
    help="Re-segment subtitles by semantics.",
)
@click.option(
    "-W",
    "--word-level",
    "--word_level",
    is_flag=True,
    default=False,
    help="Include word-level alignment timestamps in output (for JSON, TextGrid, and subtitle formats).",
)
@click.option(
    "-D",
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    default="cpu",
    help="Device to use for inference.",
)
@click.option(
    "-M",
    "--model-name-or-path",
    "--model_name_or_path",
    type=str,
    default="Lattifai/Lattice-1-Alpha",
    help="Model name or path for alignment.",
)
@click.option(
    "-K",
    "-L",
    "--api-key",
    "--api_key",
    type=str,
    default=None,
    help="API key for LattifAI.",
)
@click.argument(
    "input_media_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "input_subtitle_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "output_subtitle_path",
    type=click.Path(allow_dash=True),
)
def align(
    input_media_path: Pathlike,
    input_subtitle_path: Pathlike,
    output_subtitle_path: Pathlike,
    input_format: str = "auto",
    split_sentence: bool = False,
    word_level: bool = False,
    device: str = "cpu",
    model_name_or_path: str = "Lattifai/Lattice-1-Alpha",
    api_key: str = None,
):
    """
    Command used to align media(audio/video) with subtitles
    """
    try:
        client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)
        client.alignment(
            input_media_path,
            input_subtitle_path,
            format=input_format.lower(),
            split_sentence=split_sentence,
            return_details=word_level,
            output_subtitle_path=output_subtitle_path,
        )
        click.echo(colorful.green(f"✅ Alignment completed successfully: {output_subtitle_path}"))
    except Exception as e:
        from lattifai.errors import LattifAIError

        # Display error message
        if isinstance(e, LattifAIError):
            click.echo(colorful.red("❌ Alignment failed:"))
            click.echo(e.get_message())
            # Show support info
            click.echo(e.get_support_info())
        else:
            click.echo(colorful.red(f"❌ Alignment failed: {str(e)}"))

        raise click.ClickException("Alignment failed")


@cli.command()
@click.option(
    "-M",
    "--media-format",
    "--media_format",
    type=click.Choice(
        [
            # Audio formats
            "mp3",
            "wav",
            "m4a",
            "aac",
            "flac",
            "ogg",
            "opus",
            "aiff",
            # Video formats
            "mp4",
            "webm",
            "mkv",
            "avi",
            "mov",
        ],
        case_sensitive=False,
    ),
    default="mp3",
    help="Media format for YouTube download (audio or video).",
)
@click.option(
    "-S",
    "--split-sentence",
    "--split_sentence",
    is_flag=True,
    default=False,
    help="Re-segment subtitles by semantics.",
)
@click.option(
    "-W",
    "--word-level",
    "--word_level",
    is_flag=True,
    default=False,
    help="Include word-level alignment timestamps in output (for JSON, TextGrid, and subtitle formats).",
)
@click.option(
    "-O",
    "--output-dir",
    "--output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=".",
    help="Output directory (default: current directory).",
)
@click.option(
    "-D",
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    default="cpu",
    help="Device to use for inference.",
)
@click.option(
    "-M",
    "--model-name-or-path",
    "--model_name_or_path",
    type=str,
    default="Lattifai/Lattice-1-Alpha",
    help="Model name or path for alignment.",
)
@click.option(
    "-K",
    "-L",
    "--api-key",
    "--api_key",
    type=str,
    default=None,
    help="API key for LattifAI.",
)
@click.option(
    "-G",
    "--gemini-api-key",
    "--gemini_api_key",
    type=str,
    default=None,
    help="Gemini API key for transcription fallback when subtitles are unavailable.",
)
@click.option(
    "-F",
    "--output-format",
    "--output_format",
    type=click.Choice(OUTPUT_SUBTITLE_FORMATS, case_sensitive=False),
    default="vtt",
    help="Subtitle output format.",
)
@click.argument(
    "yt_url",
    type=str,
)
def youtube(
    yt_url: str,
    media_format: str = "mp3",
    split_sentence: bool = False,
    word_level: bool = False,
    output_dir: str = ".",
    device: str = "cpu",
    model_name_or_path: str = "Lattifai/Lattice-1-Alpha",
    api_key: str = None,
    gemini_api_key: str = None,
    output_format: str = "vtt",
):
    """
    Download media and subtitles from YouTube for further alignment.
    """
    from lattifai.workflows.gemini import GeminiTranscriber
    from lattifai.workflows.youtube import YouTubeDownloader, YouTubeSubtitleAgent

    # Get Gemini API key
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    async def _process():
        # Initialize components with their configuration (only config, not runtime params)
        downloader = YouTubeDownloader()
        transcriber = GeminiTranscriber(api_key=gemini_key)
        aligner = AsyncLattifAI(api_key=api_key, model_name_or_path=model_name_or_path, device=device)

        # Create agent with initialized components
        agent = YouTubeSubtitleAgent(
            downloader=downloader,
            transcriber=transcriber,
            aligner=aligner,
            max_retries=0,
        )

        result = await agent.process_youtube_url(
            url=yt_url,
            output_dir=output_dir,
            media_format=media_format,
            force_overwrite=False,
            output_format=output_format,
            split_sentence=split_sentence,
            word_level=word_level,
        )
        return result

    try:
        result = asyncio.run(_process())

        # Display results
        click.echo(colorful.green("✅ Processing completed!"))
        click.echo()

        # Show metadata
        metadata = result.get("metadata", {})
        if metadata:
            click.echo(f'🎬    Title: {metadata.get("title", "Unknown")}')
            click.echo(f'⏱️  Duration: {metadata.get("duration", 0)} seconds')
            click.echo()

        # Show exported files
        exported_files = result.get("exported_files", {})
        if exported_files:
            click.echo(colorful.green("📄 Generated subtitle files:"))
            for format_name, file_path in exported_files.items():
                click.echo(f"  {format_name}: {file_path}")
            click.echo()

        # Show subtitle count
        subtitle_count = result.get("subtitle_count", 0)
        click.echo(f"📝 Generated {subtitle_count} subtitle segments")

    except Exception as e:
        from lattifai.errors import LattifAIError

        # Extract error message without support info (to avoid duplication)
        if isinstance(e, LattifAIError):
            # Use the get_message() method which includes proper formatting
            click.echo(colorful.red("❌ Failed to process YouTube URL:"))
            click.echo(e.get_message())
            # Show support info once at the end
            click.echo(e.get_support_info())
        else:
            click.echo(colorful.red(f"❌ Failed to process YouTube URL: {str(e)}"))

        raise click.ClickException("Processing failed")
