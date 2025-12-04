"""CLI module for LattifAI with nemo_run entry points."""

import nemo_run as run  # noqa: F401

# Import and re-export entrypoints at package level so NeMo Run can find them
from lattifai.cli.alignment import align
from lattifai.cli.caption import convert
from lattifai.cli.transcribe import transcribe, transcribe_align
from lattifai.cli.youtube import youtube

__all__ = [
    "align",
    "convert",
    "transcribe",
    "transcribe_align",
    "youtube",
]
