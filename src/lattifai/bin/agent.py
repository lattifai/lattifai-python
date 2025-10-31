"""
Agent command for YouTube workflow
"""

import asyncio
import os
import sys
from typing import List, Optional

import click
import colorful
from lhotse.utils import Pathlike

from lattifai.bin.cli_base import cli


@cli.command()
@click.option('--youtube', '--yt', is_flag=True, help='Process YouTube URL through agentic workflow.')
@click.option(
    '--gemini-api-key',
    '--gemini_api_key',
    type=str,
    help='Gemini API key for transcription (overrides GEMINI_API_KEY env var).',
)
@click.option(
    '--media-format',
    '--media_format',
    type=click.Choice(
        ['mp3', 'wav', 'm4a', 'aac', 'opus', 'mp4', 'webm', 'mkv', 'avi', 'mov', 'flv', 'wmv', 'mpeg', 'mpg', '3gp'],
        case_sensitive=False,
    ),
    default='mp4',
    help='Media format for YouTube download (audio or video).',
)
@click.option(
    '--output-format',
    '--output_format',
    type=click.Choice(['srt', 'vtt', 'ass', 'ssa', 'sub', 'sbv', 'txt'], case_sensitive=False),
    default='srt',
    help='Subtitle output format.',
)
@click.option(
    '--output-dir',
    '--output_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help='Output directory for generated files (default: current directory).',
)
@click.option(
    '--max-retries',
    '--max_retries',
    type=int,
    default=0,
    help='Maximum number of retries for failed steps.',
)
@click.option(
    '--split-sentence',
    '--split_sentence',
    is_flag=True,
    default=False,
    help='Re-segment subtitles by semantics.',
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
@click.option('--force', '-f', is_flag=True, help='Force overwrite existing files without confirmation.')
@click.argument('url', type=str, required=True)
def agent(
    youtube: bool,
    url: str,
    gemini_api_key: Optional[str] = None,
    media_format: str = 'mp4',
    output_format: str = 'srt',
    output_dir: Optional[str] = None,
    max_retries: int = 0,
    split_sentence: bool = False,
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
        click.echo(colorful.red('❌ Please specify a workflow type. Use --youtube for YouTube processing.'))
        return

    # Setup logging
    import logging

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set default output directory
    if not output_dir:
        output_dir = os.getcwd()

    # Get Gemini API key
    api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        click.echo(
            colorful.red(
                '❌ Gemini API key is required. Set GEMINI_API_KEY environment variable or use --gemini-api-key option.'
            )
        )
        return

    try:
        # Run the YouTube workflow
        asyncio.run(
            _run_youtube_workflow(
                url=url,
                api_key=api_key,
                media_format=media_format,
                output_format=output_format,
                output_dir=output_dir,
                max_retries=max_retries,
                split_sentence=split_sentence,
                force_overwrite=force,
            )
        )

    except KeyboardInterrupt:
        click.echo(colorful.yellow('\n⚠️ Process interrupted by user'))
        sys.exit(1)
    except Exception as e:
        click.echo(colorful.red(f'❌ Workflow failed: {str(e)}'))
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def _run_youtube_workflow(
    url: str,
    api_key: str,
    media_format: str,
    output_format: str,
    output_dir: str,
    max_retries: int,
    split_sentence: bool = False,
    force_overwrite: bool = False,
):
    """Run the YouTube processing workflow"""

    # Determine if format is audio or video
    is_audio_format = media_format.lower() in ['mp3', 'wav', 'm4a', 'aac', 'opus']
    format_type = 'Audio' if is_audio_format else 'Video'

    click.echo(colorful.cyan('🚀 LattifAI Agentic Workflow - YouTube Processing'))
    click.echo(f'📺      YouTube URL: {url}')
    click.echo(f'🎬     Media format: {media_format} ({format_type})')
    click.echo(f'📝    Output format: {output_format}')
    click.echo(f'📁 Output directory: {output_dir}')
    click.echo(f'🔄      Max retries: {max_retries}')
    click.echo()

    # Import the workflow agent
    from lattifai.workflows import YouTubeSubtitleAgent

    # Initialize agent
    agent = YouTubeSubtitleAgent(
        gemini_api_key=api_key,
        video_format=media_format,  # YouTubeSubtitleAgent still uses video_format parameter
        output_format=output_format,
        max_retries=max_retries,
        split_sentence=split_sentence,
        force_overwrite=force_overwrite,
    )

    # Process the URL
    result = await agent.process_youtube_url(url=url, output_dir=output_dir, output_format=output_format)

    # Display results
    click.echo(colorful.bold_white_on_green('🎉 Workflow completed successfully!'))
    click.echo()
    click.echo(colorful.bold_white_on_green('📊 Results:'))

    # Show metadata
    metadata = result.get('metadata', {})
    if metadata:
        click.echo(f'🎬    Title: {metadata.get("title", "Unknown")}')
        click.echo(f'👤 Uploader: {metadata.get("uploader", "Unknown").strip()}')
        click.echo(f'⏱️  Duration: {metadata.get("duration", 0)} seconds')
        click.echo()

    # Show exported files
    exported_files = result.get('exported_files', {})
    if exported_files:
        click.echo(colorful.bold_white_on_green('📄 Generated subtitle files:'))
        for format_name, file_path in exported_files.items():
            click.echo(f'  {format_name.upper()}: {file_path}')
        click.echo()

    # Show subtitle count
    subtitle_count = result.get('subtitle_count', 0)
    click.echo(f'📝 Generated {subtitle_count} subtitle segments')

    click.echo(colorful.bold_white_on_green('✨ All done! Your aligned subtitles are ready.'))


# Add dependencies check
def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []

    try:
        from google import genai
    except ImportError:
        missing_deps.append('google-genai')

    try:
        import yt_dlp
    except ImportError:
        missing_deps.append('yt-dlp')

    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_deps.append('python-dotenv')

    if missing_deps:
        click.echo(colorful.red('❌ Missing required dependencies:'))
        for dep in missing_deps:
            click.echo(f'  - {dep}')
        click.echo()
        click.echo('Install them with:')
        click.echo(f'  pip install {" ".join(missing_deps)}')
        return False

    return True


# Check dependencies when module is imported
if not check_dependencies():
    pass  # Don't exit on import, let the command handle it


if __name__ == '__main__':
    asyncio.run(
        _run_youtube_workflow(
            url='https://www.youtube.com/watch?v=DQacCB9tDaw',
            api_key=None,
            media_format='mp4',
            output_format='TextGrid',
            output_dir='~/Downloads/lattifai_openai4o_debug',
            max_retries=0,
            split_sentence=True,
            force_overwrite=False,
        )
    )
