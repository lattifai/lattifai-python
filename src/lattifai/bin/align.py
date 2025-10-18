import click
import colorful
from lhotse.utils import Pathlike

from lattifai.bin.cli_base import cli


@cli.command()
@click.option(
    '-F',
    '--input_format',
    type=click.Choice(['srt', 'vtt', 'ass', 'txt', 'auto'], case_sensitive=False),
    default='auto',
    help='Input Subtitle format.',
)
@click.option(
    '-D',
    '--device',
    type=click.Choice(['cpu', 'cuda', 'mps'], case_sensitive=False),
    default='cpu',
    help='Device to use for inference.',
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
):
    """
    Command used to align audio with subtitles
    """
    from lattifai import LattifAI

    client = LattifAI(device=device)
    client.alignment(
        input_audio_path, input_subtitle_path, format=input_format.lower(), output_subtitle_path=output_subtitle_path
    )
