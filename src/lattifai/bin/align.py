import asyncio

import click
import colorful
from lhotse.utils import Pathlike

from lattifai import LattifAI
from lattifai.bin.cli_base import cli
from lattifai.io import SubtitleIO
from lattifai.workflows.youtube import YouTubeDownloader


@cli.command()
@click.option(
    '-F',
    '--input-format',
    '--input_format',
    type=click.Choice(['srt', 'vtt', 'ass', 'txt', 'auto', 'gemini'], case_sensitive=False),
    default='auto',
    help='Input Subtitle format.',
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
    input_audio_path: Pathlike,
    input_subtitle_path: Pathlike,
    output_subtitle_path: Pathlike,
    input_format: str = 'auto',
    device: str = 'cpu',
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    split_sentence: bool = False,
    api_key: str = None,
):
    """
    Command used to align audio with subtitles
    """
    client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)
    client.alignment(
        input_audio_path,
        input_subtitle_path,
        format=input_format.lower(),
        split_sentence=split_sentence,
        output_subtitle_path=output_subtitle_path,
    )


@cli.command()
@click.option(
    '-A',
    '--audio-format',
    '--audio_format',
    type=str,
    default='mp3',
    help='Audio format (e.g. mp3, wav, m4a, etc.)',
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
    '-O',
    '--output-dir',
    '--output_dir',
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default='.',
    help='Output directory, default is current folder',
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
    help='Lattifai model name or path',
)
@click.option(
    '--api-key',
    '--api_key',
    type=str,
    default=None,
    help='API key for LattifAI.',
)
@click.option(
    '--output-formats',
    '--output_formats',
    multiple=True,
    type=click.Choice(['srt', 'vtt', 'ass', 'txt'], case_sensitive=False),
    default=['vtt'],
    help='Subtitle output formats (can specify multiple, e.g. --output-formats srt --output-formats vtt)',
)
@click.argument(
    'yt_url',
    type=str,
)
def youtube(
    yt_url: str,
    audio_format: str = 'mp3',
    split_sentence: bool = False,
    output_dir: str = '.',
    device: str = 'cpu',
    model_name_or_path: str = 'Lattifai/Lattice-1-Alpha',
    api_key: str = None,
    output_formats: tuple = ('vtt',),
):
    """
    Download audio and subtitles from YouTube for further alignment.
    """

    async def _download():
        downloader = YouTubeDownloader(audio_format=audio_format)
        audio_path = await downloader.download_audio(yt_url, output_dir=output_dir, audio_format=audio_format)
        subtitle_paths = await downloader.download_subtitles(yt_url, output_dir=output_dir)
        return audio_path, subtitle_paths

    audio_path, subtitle_paths = asyncio.run(_download())

    # Robustly extract video_id
    from lattifai.workflows.youtube import YouTubeDownloader as YTDL

    video_id = YTDL.extract_video_id(yt_url)
    client = LattifAI(model_name_or_path=model_name_or_path, device=device, api_key=api_key)
    # Use the first subtitle file, raise error if none found
    if not subtitle_paths or len(subtitle_paths) == 0:
        raise RuntimeError('No subtitle file was downloaded')

    output_subtitle_path = f'{output_dir}/{video_id}.{output_formats[0]}'
    (alignments, _) = client.alignment(
        audio_path,
        subtitle_paths[0],
        format=output_formats[0],
        split_sentence=split_sentence,
        output_subtitle_path=output_subtitle_path,
    )

    for fmt in output_formats[1:]:
        output_subtitle_path = f'{output_dir}/{video_id}.{fmt}'
        SubtitleIO.write(alignments, output_path=output_subtitle_path)
