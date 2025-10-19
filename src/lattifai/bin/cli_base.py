import logging

import click


@click.group()
def cli():
    """
    The shell entry point to Lattifai, a tool for audio data manipulation.
    """
    # Load environment variables from .env file
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO,
    )
