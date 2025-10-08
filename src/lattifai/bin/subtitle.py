import click
from lhotse.utils import Pathlike

from lattifai.bin.cli_base import cli


@cli.group()
def subtitle():
    """Group of commands used to convert subtitle format."""
    pass


@subtitle.command()
@click.argument(
    'input_subtitle_path',
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    'output_subtitle_path',
    type=click.Path(allow_dash=True),
)
def convert(
    input_subtitle_path: Pathlike,
    output_subtitle_path: Pathlike,
):
    """
    Convert subtitle file to another format.
    """
    import pysubs2

    subtitle = pysubs2.load(input_subtitle_path)
    subtitle.save(output_subtitle_path)
