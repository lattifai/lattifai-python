import logging

import click


@click.group()
def cli():
    """
    The shell entry point to Lattifai, a tool for audio data manipulation.
    """
    # Load environment variables from .env file
    from dotenv import find_dotenv, load_dotenv

    # Try to find and load .env file from current directory or parent directories
    load_dotenv(find_dotenv(usecwd=True))

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
