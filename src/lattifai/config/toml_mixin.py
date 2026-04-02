"""Mixin for resolving dataclass field defaults from config.toml.

Provides a generic mechanism for any config dataclass to fall back to
~/.lattifai/config.toml when a field is left at its declared default.

Priority order: explicit constructor arg > config.toml > dataclass default.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_toml_raw_value(section: str, key: str) -> Any:
    """Read a raw (non-stringified) value from config.toml ``[section].key``.

    Unlike :func:`~lattifai.config.llm.resolve_toml_value` (which returns
    ``str``), this preserves the native Python type produced by the TOML
    parser — ``bool``, ``int``, ``float``, ``str``, ``list``, etc.

    Supports dotted sections for nested TOML tables::

        resolve_toml_raw_value("diarization.llm", "model_name")
        # reads config["diarization"]["llm"]["model_name"]

    Returns ``None`` if the key is not present or the config system is
    unavailable.
    """
    try:
        from lattifai.cli.config import _load_config, _normalize_config

        config = _normalize_config(_load_config())
        # Walk through nested sections (e.g., "diarization.llm")
        data = config
        for part in section.split("."):
            if isinstance(data, dict):
                data = data.get(part, {})
            else:
                return None
        if isinstance(data, dict):
            return data.get(key)
        return None
    except (ImportError, OSError):
        return None


class ConfigTomlMixin:
    """Mixin that auto-resolves unset dataclass fields from config.toml.

    Only ``_toml_section`` is required. All dataclass fields with simple
    defaults (not ``default_factory``, not ``MISSING``) are automatically
    checked against the corresponding TOML section.

    Usage::

        @dataclass
        class MyConfig(ConfigTomlMixin):
            _toml_section = "my_section"

            flag: bool = False      # auto-resolved from [my_section].flag
            name: str = "default"   # auto-resolved from [my_section].name

            def __post_init__(self):
                self._resolve_from_toml()

    A field is considered "unset" when its current value equals the declared
    dataclass default.  This works seamlessly with ``nemo_run`` CLI:

    * User passes ``--flag`` → value differs from default → preserved.
    * User omits the flag → value equals default → config.toml override applies.

    Fields are **skipped** when:

    * They use ``default_factory`` (complex objects like nested configs).
    * They have no default (required fields).
    * The TOML value is a ``dict`` (nested table — handled by nested configs).
    * They are listed in ``_toml_exclude``.
    """

    # Class-level attributes (unannotated so dataclass ignores them)
    _toml_section = ""
    _toml_exclude = ()

    def _resolve_from_toml(self) -> None:
        """Fill unset fields from ``config.toml [_toml_section]``."""
        if not self._toml_section:
            return

        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if f.name.startswith("_") or f.name in self._toml_exclude:
                continue

            # Skip factory defaults (complex types like nested configs, lists)
            if f.default_factory is not dataclasses.MISSING:
                continue

            # Skip required fields (no default to compare against)
            if f.default is dataclasses.MISSING:
                continue

            current = getattr(self, f.name)
            if current != f.default:
                continue  # Explicitly set by caller — preserve

            raw = resolve_toml_raw_value(self._toml_section, f.name)
            if raw is None or isinstance(raw, dict):
                continue  # Skip missing values and nested tables

            logger.debug("config.toml [%s].%s = %r", self._toml_section, f.name, raw)
            setattr(self, f.name, raw)
