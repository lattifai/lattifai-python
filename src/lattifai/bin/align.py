import asyncio

import click
import colorful
from lhotse.utils import Pathlike

from lattifai import LattifAI
from lattifai.bin.cli_base import cli
from lattifai.io import INPUT_SUBTITLE_FORMATS, OUTPUT_SUBTITLE_FORMATS, SubtitleIO
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
    output_format: str = 'vtt',
):
    """
    Download media and subtitles from YouTube for further alignment.
    """

    async def _download():
        downloader = YouTubeDownloader(media_format=media_format)
        media_path = await downloader.download_media(yt_url, output_dir=output_dir, media_format=media_format)
        subtitle_path = await downloader.download_subtitles(yt_url, output_dir=output_dir)
        return media_path, subtitle_path

    media_path, subtitle_path = asyncio.run(_download())

    # Robustly extract video_id
    from lattifai.workflows.youtube import YouTubeDownloader as YTDL

    video_id = YTDL.extract_video_id(yt_url)
    client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)

    # Handle subtitle_path which can be a string, list, or None
    if not subtitle_path:
        raise RuntimeError(
            'No subtitle file was downloaded. Media file transcription support is coming soon. '
            'Please use the `lattifai agent --youtube` command for automatic transcription workflow.'
        )

    # If subtitle_path is a list, use the first one
    if isinstance(subtitle_path, list):
        if len(subtitle_path) == 0:
            raise RuntimeError(
                'No subtitle file was downloaded. Media file transcription support is coming soon. '
                'Please use the `lattifai agent --youtube` command for automatic transcription workflow.'
            )
        subtitle_path = subtitle_path[0]

    output_subtitle_path = f'{output_dir}/{video_id}.{output_format}'
    client.alignment(
        media_path,
        subtitle_path,
        format='auto',  # Auto-detect input subtitle format
        split_sentence=split_sentence,
        return_details=word_level,
        output_subtitle_path=output_subtitle_path,
    )
