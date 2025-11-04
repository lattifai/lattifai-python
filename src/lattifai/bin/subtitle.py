import asyncio
from pathlib import Path

import click
from lhotse.utils import Pathlike

from lattifai.bin.cli_base import cli
from lattifai.io import SUBTITLE_FORMATS


@cli.group()
def subtitle():
    """Commands for subtitle format conversion and management."""
    pass


@subtitle.command()
@click.argument(
    "input_subtitle_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "output_subtitle_path",
    type=click.Path(allow_dash=True),
)
def convert(
    input_subtitle_path: Pathlike,
    output_subtitle_path: Pathlike,
):
    """
    Convert subtitle file to another format.
    """
    if str(output_subtitle_path).lower().endswith(".TextGrid".lower()):
        from lattifai.io import SubtitleIO

        alignments = SubtitleIO.read(input_subtitle_path)
        SubtitleIO.write(alignments, output_subtitle_path)
    else:
        import pysubs2

        subtitle = pysubs2.load(input_subtitle_path)

        subtitle.save(output_subtitle_path)


@subtitle.command()
@click.argument("url", type=str, required=True)
@click.option(
    "--output-dir",
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default=".",
    help="Output directory for downloaded subtitle files (default: current directory).",
)
@click.option(
    "--output-format",
    "--output_format",
    "-f",
    type=click.Choice(SUBTITLE_FORMATS + ["best"], case_sensitive=False),
    default="best",
    help="Preferred subtitle format to download (default: best available).",
)
@click.option("--force-overwrite", "-F", is_flag=True, help="Overwrite existing files without prompting.")
@click.option(
    "--lang",
    "-l",
    "-L",
    "--subtitle-lang",
    "--subtitle_lang",
    type=str,
    help='Specific subtitle language/track to download (e.g., "en").',
)
def download(
    url: str,
    output_dir: str,
    output_format: str,
    force_overwrite: bool,
    lang: str,
):
    """
    Download subtitles from YouTube URL using yt-dlp.

    URL should be a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID).
    """
    # Import here to avoid circular imports and keep startup fast
    from lattifai.workflows.youtube import YouTubeDownloader

    # Validate URL format
    if not _is_valid_youtube_url(url):
        click.echo(f"Error: Invalid YouTube URL format: {url}", err=True)
        click.echo("Please provide a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)", err=True)
        raise click.Abort()

    # Convert relative path to absolute
    output_path = Path(output_dir).resolve()

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading subtitles from: {url}")
    click.echo(f"          Output directory: {output_path}")
    click.echo(f"         Preferred format: {output_format}")
    if lang:
        click.echo(f"       Subtitle language: {lang}")
    else:
        click.echo("       Subtitle language: All available")

    # Initialize downloader and download
    downloader = YouTubeDownloader()

    async def download_subtitles():
        try:
            result = await downloader.download_subtitles(
                url=url,
                output_dir=str(output_path),
                force_overwrite=force_overwrite,
                subtitle_lang=lang,
            )

            if result:
                click.echo("✅ Subtitles downloaded successfully!")
                return result
            else:
                click.echo("⚠️  No subtitles available for this video")
                return None

        except Exception as e:
            click.echo(f"❌ Error downloading subtitles: {str(e)}", err=True)
            raise click.Abort()

    # Run the async function
    result = asyncio.run(download_subtitles())

    if result:
        if result == "gemini":
            click.echo("✨ Gemini transcription selected (use the agent command to transcribe)")
        else:
            click.echo(f"📄 Subtitle file saved to: {result}")


@subtitle.command()
@click.argument("url", type=str, required=True)
def list_subs(url: str):
    """
    List available subtitle tracks for a YouTube video.

    URL should be a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
    """
    # Import here to avoid circular imports and keep startup fast
    from lattifai.workflows.youtube import YouTubeDownloader

    # Validate URL format
    if not _is_valid_youtube_url(url):
        click.echo(f"Error: Invalid YouTube URL format: {url}", err=True)
        click.echo("Please provide a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)", err=True)
        raise click.Abort()

    click.echo(f"Listing available subtitles for: {url}")

    # Initialize downloader
    downloader = YouTubeDownloader()

    async def list_available_subtitles():
        try:
            result = await downloader.list_available_subtitles(url)

            if result:
                click.echo("📋 Available subtitle tracks:")
                for subtitle_info in result:
                    click.echo(f'  🎬 Language: {subtitle_info["language"]} - {subtitle_info["name"]}')
                    click.echo(f'     📄 Formats: {", ".join(subtitle_info["formats"])}')
                    click.echo()

                click.echo("💡 To download a specific track, use:")
                click.echo(f'   lattifai subtitle download "{url}" --lang <language_code>')
                click.echo('   Example: lattifai subtitle download "{}" --lang en-JkeT_87f4cc'.format(url))
            else:
                click.echo("⚠️  No subtitles available for this video")

        except Exception as e:
            click.echo(f"❌ Error listing subtitles: {str(e)}", err=True)
            raise click.Abort()

    # Run the async function
    asyncio.run(list_available_subtitles())


def _is_valid_youtube_url(url: str) -> bool:
    """
    Validate if the URL is a valid YouTube URL format.

    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    """
    import re

    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        if re.search(pattern, url):
            return True
    return False
