"""Custom nemo_run Entrypoint that auto-applies config.toml defaults.

Injects ~/.lattifai/config.toml values into CLI Config objects **before**
fdl.build(), using nemo_run's ``__arguments__`` dict to precisely
distinguish "user explicitly set this" from "left at default".

Priority: CLI arg > config.toml > dataclass default.

Usage:
    Config classes just need a ``_toml_section`` class attribute::

        @dataclass
        class DiarizationConfig:
            _toml_section = "diarization"
            infer_speakers: bool = False  # resolved from [diarization] if unset

    CLI entrypoints use ``entrypoint_cls``::

        @run.cli.entrypoint(name="run", namespace="diarize",
                            entrypoint_cls=LattifAIEntrypoint)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List

import fiddle as fdl
from nemo_run.cli.api import Entrypoint
from nemo_run.cli.cli_parser import parse_cli_args
from nemo_run.config import Partial
from rich.console import Console

from lattifai.config.toml_mixin import resolve_toml_raw_value

logger = logging.getLogger(__name__)


def _apply_toml_defaults(buildable: fdl.Config | fdl.Partial) -> None:
    """Walk a Fiddle Config/Partial tree and inject config.toml defaults.

    For each Config whose target class declares ``_toml_section``, every
    dataclass field NOT present in ``__arguments__`` (i.e. not explicitly
    set via CLI) is filled from the matching TOML section.

    Nested TOML tables (dicts) and complex sub-configs are skipped —
    they are handled by their own ``__post_init__`` resolution
    (e.g. ``DiarizationLLMConfig`` reads from ``[diarization.llm]``).
    """
    if not hasattr(buildable, "__arguments__"):
        return

    # Recurse into nested Config/Partial arguments first
    for value in buildable.__arguments__.values():
        if isinstance(value, (fdl.Config, fdl.Partial)):
            _apply_toml_defaults(value)

    # Resolve the target class
    target = buildable.__fn_or_cls__
    toml_section = getattr(target, "_toml_section", "")
    if not toml_section:
        return
    if not dataclasses.is_dataclass(target):
        return

    explicitly_set = set(buildable.__arguments__.keys())

    for f in dataclasses.fields(target):
        if f.name in explicitly_set:
            continue  # User explicitly set this via CLI — never override
        if f.name.startswith("_"):
            continue

        raw = resolve_toml_raw_value(toml_section, f.name)
        if raw is None or isinstance(raw, dict):
            continue  # Missing or nested table — skip

        logger.debug("config.toml [%s].%s = %r (injected)", toml_section, f.name, raw)
        setattr(buildable, f.name, raw)


class LattifAIEntrypoint(Entrypoint):
    """Entrypoint that injects ``~/.lattifai/config.toml`` defaults before build.

    Drop-in replacement for ``nemo_run.cli.api.Entrypoint``. Pass via
    ``entrypoint_cls=LattifAIEntrypoint`` in ``@run.cli.entrypoint()``.
    """

    def _execute_simple(self, args: List[str], console: Console):
        config = parse_cli_args(self.fn, args, Partial)
        _apply_toml_defaults(config)
        fn = fdl.build(config)
        fn.func.__io__ = config
        fn()
