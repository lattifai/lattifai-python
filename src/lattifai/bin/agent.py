"""
Agent command for YouTube workflow
"""

import asyncio
import os
import sys
from typing import Optional

import click
import colorful

from lattifai.bin.cli_base import cli
from lattifai.io import OUTPUT_SUBTITLE_FORMATS


@cli.command()
@click.option("--youtube", "--yt", is_flag=True, help="Process YouTube URL through agentic workflow.")
@click.option(
    "-K",
    "-L",
    "--api-key",
    "--api_key",
    type=str,
    help="LattifAI API key for alignment (overrides LATTIFAI_API_KEY env var).",
)
@click.option(
    "-G",
    "--gemini-api-key",
    "--gemini_api_key",
    type=str,
    help="Gemini API key for transcription (overrides GEMINI_API_KEY env var).",
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
    "--media-format",
    "--media_format",
    type=click.Choice(
        ["mp3", "wav", "m4a", "aac", "opus", "mp4", "webm", "mkv", "avi", "mov", "flv", "wmv", "mpeg", "mpg", "3gp"],
        case_sensitive=False,
    ),
    default="mp4",
    help="Media format for YouTube download (audio or video).",
)
@click.option(
    "--output-format",
    "--output_format",
    type=click.Choice(OUTPUT_SUBTITLE_FORMATS, case_sensitive=False),
    default="srt",
    help="Subtitle output format.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Output directory for generated files (default: current directory).",
)
@click.option(
    "--max-retries",
    "--max_retries",
    type=int,
    default=0,
    help="Maximum number of retries for failed steps.",
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
    "--word-level",
    "--word_level",
    is_flag=True,
    default=False,
    help="Include word-level alignment timestamps in output (for JSON, TextGrid, and subtitle formats).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option("--force", "-f", is_flag=True, help="Force overwrite existing files without confirmation.")
@click.argument("url", type=str, required=True)
def agent(
    youtube: bool,
    url: str,
    api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    device: str = "cpu",
    model_name_or_path: str = "Lattifai/Lattice-1-Alpha",
    media_format: str = "mp4",
    output_format: str = "srt",
    output_dir: Optional[str] = None,
    max_retries: int = 0,
    split_sentence: bool = False,
    word_level: bool = False,
    verbose: bool = False,
    force: bool = False,
):
    """
    LattifAI Agentic Workflow Agent

    Process multimedia content through intelligent agent-based pipelines.

    Example:
        lattifai agent --youtube https://www.youtube.com/watch?v=example
    """

    if not youtube:
        click.echo(colorful.red("❌ Please specify a workflow type. Use --youtube for YouTube processing."))
        return

    # Setup logging
    import logging

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set default output directory
    if not output_dir:
        output_dir = os.getcwd()

    # Get API keys
    lattifai_api_key = api_key or os.getenv("LATTIFAI_API_KEY")
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    try:
        # Run the YouTube workflow
        asyncio.run(
            _run_youtube_workflow(
                url=url,
                lattifai_api_key=lattifai_api_key,
                gemini_api_key=gemini_key,
                device=device,
                model_name_or_path=model_name_or_path,
                media_format=media_format,
                output_format=output_format,
                output_dir=output_dir,
                max_retries=max_retries,
                split_sentence=split_sentence,
                word_level=word_level,
                force_overwrite=force,
            )
        )

    except KeyboardInterrupt:
        click.echo(colorful.yellow("\n⚠️ Process interrupted by user"))
        sys.exit(1)
    except Exception as e:
        from lattifai.errors import LattifAIError

        # Extract error message without support info (to avoid duplication)
        if isinstance(e, LattifAIError):
            # Use the get_message() method which includes proper formatting
            click.echo(colorful.red("❌ Workflow failed:"))
            click.echo(e.get_message())
            # Show support info once at the end
            click.echo(e.get_support_info())
        else:
            click.echo(colorful.red(f"❌ Workflow failed: {str(e)}"))

        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def _run_youtube_workflow(
    url: str,
    lattifai_api_key: Optional[str],
    gemini_api_key: str,
    device: str,
    model_name_or_path: str,
    media_format: str,
    output_format: str,
    output_dir: str,
    max_retries: int,
    split_sentence: bool = False,
    word_level: bool = False,
    force_overwrite: bool = False,
):
    """Run the YouTube processing workflow"""

    click.echo(colorful.cyan("🚀 LattifAI Agentic Workflow - YouTube Processing"))
    click.echo(f"📺      YouTube URL: {url}")
    click.echo(f"🎬     Media format: {media_format}")
    click.echo(f"📝    Output format: {output_format}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"🔄      Max retries: {max_retries}")
    click.echo()

    # Import workflow components
    from lattifai.client import AsyncLattifAI
    from lattifai.workflows import YouTubeSubtitleAgent
    from lattifai.workflows.gemini import GeminiTranscriber
    from lattifai.workflows.youtube import YouTubeDownloader

    # Initialize components with their configuration (only persistent config, not runtime params)
    downloader = YouTubeDownloader()
    transcriber = GeminiTranscriber(api_key=gemini_api_key)
    aligner = AsyncLattifAI(model_name_or_path=model_name_or_path, device=device, api_key=lattifai_api_key)

    # Initialize agent with components
    agent = YouTubeSubtitleAgent(
        downloader=downloader,
        transcriber=transcriber,
        aligner=aligner,
        max_retries=max_retries,
    )

    # Process the URL
    result = await agent.process_youtube_url(
        url=url,
        output_dir=output_dir,
        media_format=media_format,
        force_overwrite=force_overwrite,
        output_format=output_format,
        split_sentence=split_sentence,
        word_level=word_level,
    )

    # Display results
    click.echo(colorful.bold_white_on_green("🎉 Workflow completed successfully!"))
    click.echo()
    click.echo(colorful.bold_white_on_green("📊 Results:"))

    # Show metadata
    metadata = result.get("metadata", {})
    if metadata:
        click.echo(f'🎬    Title: {metadata.get("title", "Unknown")}')
        click.echo(f'👤 Uploader: {metadata.get("uploader", "Unknown").strip()}')
        click.echo(f'⏱️  Duration: {metadata.get("duration", 0)} seconds')
        click.echo()

    # Show exported files
    exported_files = result.get("exported_files", {})
    if exported_files:
        click.echo(colorful.bold_white_on_green("📄 Generated subtitle files:"))
        for format_name, file_path in exported_files.items():
            click.echo(f"  {format_name.upper()}: {file_path}")
        click.echo()

    # Show subtitle count
    subtitle_count = result.get("subtitle_count", 0)
    click.echo(f"📝 Generated {subtitle_count} subtitle segments")

    click.echo(colorful.bold_white_on_green("✨ All done! Your aligned subtitles are ready."))


# Add dependencies check
def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []

    try:
        from google import genai  # noqa: F401
    except ImportError:
        missing_deps.append("google-genai")

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing_deps.append("yt-dlp")

    try:
        from dotenv import load_dotenv  # noqa: F401
    except ImportError:
        missing_deps.append("python-dotenv")

    if missing_deps:
        click.echo(colorful.red("❌ Missing required dependencies:"))
        for dep in missing_deps:
            click.echo(f"  - {dep}")
        click.echo()
        click.echo("Install them with:")
        click.echo(f'  pip install {" ".join(missing_deps)}')
        return False

    return True


# Check dependencies when module is imported
if not check_dependencies():
    pass  # Don't exit on import, let the command handle it


if __name__ == "__main__":
    import os

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    asyncio.run(
        _run_youtube_workflow(
            # url='https://www.youtube.com/watch?v=7nv1snJRCEI',
            url="https://www.youtube.com/watch?v=DQacCB9tDaw",
            lattifai_api_key=os.getenv("LATTIFAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            device="mps",
            model_name_or_path="Lattifai/Lattice-1-Alpha",
            media_format="mp3",
            output_format="TextGrid",
            output_dir="~/Downloads/lattifai_youtube",
            max_retries=0,
            split_sentence=True,
            word_level=False,
            force_overwrite=False,
        )
    )
