"""CLI module for LattifAI with nemo_run entry points."""

import nemo_run as run  # noqa: F401

# Import and re-export entrypoints at package level so NeMo Run can find them
from lattifai.cli.agent import agent
from lattifai.cli.align import align
from lattifai.cli.subtitle import convert
from lattifai.cli.youtube import youtube

__all__ = [
    "agent",
    "align",
    "convert",
    "youtube",
]
