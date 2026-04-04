"""CLI module for LattifAI with nemo_run entry points."""

import lattifai._init  # noqa: F401 # isort: skip  # Suppress warnings early

import nemo_run as run  # noqa: F401

# Import and re-export entrypoints at package level so NeMo Run can find them
from lattifai.cli.alignment import align
from lattifai.cli.caption import convert, diff
from lattifai.cli.diarize import diarize, naming
from lattifai.cli.serve import serve
from lattifai.cli.summarize import summarize_caption
from lattifai.cli.transcribe import transcribe, transcribe_align
from lattifai.cli.translate import translate, translate_youtube
from lattifai.cli.youtube import youtube, youtube_download

# doctor and update are registered as direct Typer commands via _main.py,
# not through nemo_run's namespace system, so they don't need re-export here.

__all__ = [
    "align",
    "convert",
    "diff",
    "diarize",
    "naming",
    "serve",
    "summarize_caption",
    "transcribe",
    "transcribe_align",
    "translate",
    "translate_youtube",
    "youtube",
    "youtube_download",
]
