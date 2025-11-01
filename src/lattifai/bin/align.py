import asyncio
import os
from pathlib import Path

import click
import colorful
from lhotse.utils import Pathlike

from lattifai import LattifAI
from lattifai.bin.cli_base import cli
from lattifai.io import INPUT_SUBTITLE_FORMATS, OUTPUT_SUBTITLE_FORMATS, SUBTITLE_FORMATS, SubtitleIO
from lattifai.workflows.file_manager import FileExistenceManager
from lattifai.workflows.gemini import GeminiTranscriber
from lattifai.workflows.youtube import YouTubeDownloader


@cli.command()
@click.option(
    '-F',
    '--subtitle_format',
    '--subtitle-format',
    type=click.Choice(INPUT_SUBTITLE_FORMATS, case_sensitive=False),
    default='auto',
    help='Input subtitle format.',
)
@click.option(
    '-S',
    '--split-sentence',
    '--split_sentence',
    is_flag=True,
    default=False,
    help='Re-segment subtitles by semantics.',
)
@click.option(
    '-W',
    '--word-level',
    '--word_level',
    is_flag=True,
    default=False,
    help='Include word-level alignment timestamps in output (for JSON, TextGrid, and subtitle formats).',
)
@click.option(
    '-D',
    '--device',
    type=click.Choice(['cpu', 'cuda', 'mps'], case_sensitive=False),
    default='cpu',
    help='Device to use for inference.',
)
@click.option(
    '-M',
    '--model-name-or-path',
    '--model_name_or_path',
    type=str,
    default='Lattifai/Lattice-1-Alpha',
    help='Model name or path for alignment.',
)
@click.option(
    '--api-key',
    '--api_key',
    type=str,
    default=None,
    help='API key for LattifAI.',
)
@click.argument(
    'input_audio_path',
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    'input_subtitle_path',
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    'output_subtitle_path',
    type=click.Path(allow_dash=True),
)
def align(
    input_media_path: Pathlike,
    input_subtitle_path: Pathlike,
    output_subtitle_path: Pathlike,
    input_format: str = 'auto',
    split_sentence: bool = False,
    word_level: bool = False,
    device: str = 'cpu',
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    api_key: str = None,
):
    """
    Command used to align media(audio/video) with subtitles
    """
    client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)
    client.alignment(
        input_media_path,
        input_subtitle_path,
        format=input_format.lower(),
        split_sentence=split_sentence,
        return_details=word_level,
        output_subtitle_path=output_subtitle_path,
    )


@cli.command()
@click.option(
    '-M',
    '--media-format',
    '--media_format',
    type=click.Choice(
        [
            # Audio formats
            'mp3',
            'wav',
            'm4a',
            'aac',
            'flac',
            'ogg',
            'opus',
            'aiff',
            # Video formats
            'mp4',
            'webm',
            'mkv',
            'avi',
            'mov',
        ],
        case_sensitive=False,
    ),
    default='mp3',
    help='Media format for YouTube download (audio or video).',
)
@click.option(
    '-S',
    '--split-sentence',
    '--split_sentence',
    is_flag=True,
    default=False,
    help='Re-segment subtitles by semantics.',
)
@click.option(
    '-W',
    '--word-level',
    '--word_level',
    is_flag=True,
    default=False,
    help='Include word-level alignment timestamps in output (for JSON, TextGrid, and subtitle formats).',
)
@click.option(
    '-O',
    '--output-dir',
    '--output_dir',
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default='.',
    help='Output directory (default: current directory).',
)
@click.option(
    '-D',
    '--device',
    type=click.Choice(['cpu', 'cuda', 'mps'], case_sensitive=False),
    default='cpu',
    help='Device to use for inference.',
)
@click.option(
    '-M',
    '--model-name-or-path',
    '--model_name_or_path',
    type=str,
    default='Lattifai/Lattice-1-Alpha',
    help='Model name or path for alignment.',
)
@click.option(
    '--api-key',
    '--api_key',
    type=str,
    default=None,
    help='API key for LattifAI.',
)
@click.option(
    '--gemini-api-key',
    '--gemini_api_key',
    type=str,
    default=None,
    help='Gemini API key for transcription fallback when subtitles are unavailable.',
)
@click.option(
    '-F',
    '--output-format',
    '--output_format',
    type=click.Choice(OUTPUT_SUBTITLE_FORMATS, case_sensitive=False),
    default='vtt',
    help='Subtitle output format.',
)
@click.argument(
    'yt_url',
    type=str,
)
def youtube(
    yt_url: str,
    media_format: str = 'mp3',
    split_sentence: bool = False,
    word_level: bool = False,
    output_dir: str = '.',
    device: str = 'cpu',
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    api_key: str = None,
    gemini_api_key: str = None,
    output_format: str = 'vtt',
):
    """
    Download media and subtitles from YouTube for further alignment.
    """

    async def _download():
        # Pass gemini_api_key to downloader so it can offer Gemini option
        gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        downloader = YouTubeDownloader(media_format=media_format, gemini_api_key=gemini_key)
        media_path = await downloader.download_media(yt_url, output_dir=output_dir, media_format=media_format)
        subtitle_path = await downloader.download_subtitles(yt_url, output_dir=output_dir)
        return media_path, subtitle_path

    media_path, subtitle_path = asyncio.run(_download())

    # Robustly extract video_id
    from lattifai.workflows.youtube import YouTubeDownloader as YTDL

    video_id = YTDL.extract_video_id(yt_url)
    client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)

    # Handle subtitle_path which can be a string, list, or None
    # Special handling for 'gemini' return value from download_subtitles
    user_chose_gemini = False
    if subtitle_path == 'gemini':
        click.echo(colorful.magenta('✨ User selected Gemini transcription'))
        subtitle_path = None  # Treat as no subtitle to trigger transcription below
        user_chose_gemini = True  # Skip checking for existing files

    if not subtitle_path:
        # Only check for existing files if user didn't explicitly choose Gemini
        if not user_chose_gemini:
            click.echo(colorful.yellow('⚠️  No subtitles found from YouTube download.'))

            # Check for existing subtitle files (including Gemini transcripts)
            click.echo(colorful.cyan('🔍 Checking for existing subtitle files...'))
            existing_files = FileExistenceManager.check_existing_files(
                video_id, output_dir, subtitle_formats=SUBTITLE_FORMATS + ['md']
            )

            # If existing subtitle files found, prompt user to select
            if existing_files['subtitle']:
                # Enable Gemini option if API key is available
                has_gemini_key = bool(gemini_api_key or os.getenv('GEMINI_API_KEY'))
                subtitle_choice = FileExistenceManager.prompt_file_selection(
                    file_type='subtitle',
                    files=existing_files['subtitle'],
                    operation='transcribe',
                    enable_gemini=has_gemini_key,
                )

                if subtitle_choice == 'cancel':
                    click.echo(colorful.red('❌ Operation cancelled by user'))
                    raise click.ClickException('Operation cancelled by user')
                elif subtitle_choice in ('overwrite', 'gemini'):
                    # User wants to transcribe, continue to transcription below
                    if subtitle_choice == 'overwrite':
                        click.echo(colorful.yellow('🔄 Will re-download or transcribe...'))
                    # For 'gemini', the message is already printed by FileExistenceManager
                elif subtitle_choice and subtitle_choice != 'proceed':
                    # User selected a specific file
                    subtitle_path = subtitle_choice
                    click.echo(colorful.green(f'✅ Using existing subtitle: {subtitle_path}'))
                else:
                    # proceed means no files found, continue to transcription
                    pass

        # If subtitle_path is still not set, we need to transcribe
        if not subtitle_path:
            click.echo(colorful.magenta('✨ Attempting transcription with Gemini...'))

            # Get Gemini API key from parameter or environment variable
            gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            if not gemini_key:
                click.echo(
                    colorful.red(
                        '❌ Gemini API key is required for transcription. '
                        'Set GEMINI_API_KEY environment variable or use --gemini-api-key option.'
                    )
                )
                raise click.ClickException('Missing Gemini API key')

            try:
                # Use GeminiTranscriber to transcribe the media file
                transcriber = GeminiTranscriber(api_key=gemini_key)

                async def _transcribe():
                    return await transcriber.transcribe_url(yt_url)

                transcript = asyncio.run(_transcribe())

                # Save transcript as Gemini.md file (following the agent pattern)
                gemini_subtitle_path = Path(output_dir) / f'{video_id}_Gemini.md'
                with open(gemini_subtitle_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)

                subtitle_path = str(gemini_subtitle_path)
                click.echo(colorful.green(f'✅ Transcription completed: {subtitle_path}'))

            except Exception as e:
                click.echo(colorful.red(f'❌ Transcription failed: {str(e)}'))
                raise click.ClickException(f'Transcription failed: {str(e)}')

    assert not isinstance(subtitle_path, list)

    output_subtitle_path = f'{output_dir}/{video_id}.{output_format}'
    client.alignment(
        media_path,
        subtitle_path,
        format='auto',  # Auto-detect input subtitle format
        split_sentence=split_sentence,
        return_details=word_level,
        output_subtitle_path=output_subtitle_path,
    )
