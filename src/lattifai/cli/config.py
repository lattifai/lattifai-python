"""Implementation of 'lai config' for managing LattifAI CLI configuration."""

import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from lattifai.theme import _Theme as T

console = Console()

CONFIG_DIR = Path.home() / ".lattifai"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Top-level keys: user-facing name (uppercase) -> environment variable name
KEY_MAP = {
    "LATTIFAI_API_KEY": "LATTIFAI_API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "OPENAI_API_BASE_URL": "OPENAI_API_BASE_URL",
    "DEFAULT_AUDIO_FORMAT": "LATTIFAI_DEFAULT_AUDIO_FORMAT",
    "DEFAULT_VIDEO_FORMAT": "LATTIFAI_DEFAULT_VIDEO_FORMAT",
}

# LLMConfig internal fields — not user-settable via config.toml
_INTERNAL_FIELDS = frozenset({"section", "fallback_model"})


def _discover_section_keys() -> dict[str, None]:
    """Auto-discover valid section keys from Config dataclasses.

    Scans all Config classes with ``_toml_section`` and LLMConfig subclasses
    to build the set of valid ``section.field`` keys for config.toml.
    """
    from lattifai.config import (
        AlignmentConfig,
        CaptionConfig,
        ClientConfig,
        DiarizationConfig,
        EventConfig,
        MediaConfig,
        SummarizationConfig,
        TranscriptionConfig,
        TranslationConfig,
    )
    from lattifai.config.diarization import DiarizationLLMConfig
    from lattifai.config.translation import TranslationLLMConfig

    result: dict[str, None] = {}

    def _extract(cls: type, section: str) -> None:
        if not dataclasses.is_dataclass(cls):
            return
        for f in dataclasses.fields(cls):
            if f.name.startswith("_"):
                continue
            if f.repr is False:
                continue
            if f.name in _INTERNAL_FIELDS:
                continue
            # Skip nested Config / dataclass fields (they become sub-sections)
            if "Config" in str(f.type):
                continue
            result[f"{section}.{f.name}"] = None

    # Config classes with _toml_section
    for cls in (
        AlignmentConfig,
        CaptionConfig,
        ClientConfig,
        DiarizationConfig,
        EventConfig,
        MediaConfig,
        SummarizationConfig,
        TranscriptionConfig,
        TranslationConfig,
    ):
        section = getattr(cls, "_toml_section", "")
        if section:
            _extract(cls, section)

    # LLMConfig subclasses — section is a dataclass field, not a class attribute
    for cls in (DiarizationLLMConfig, TranslationLLMConfig):
        section = ""
        for f in dataclasses.fields(cls):
            if f.name == "section" and f.default is not dataclasses.MISSING:
                section = f.default
                break
        if section:
            _extract(cls, section)

    return result


# Section-scoped keys: auto-discovered from Config dataclasses.
# Value is the env var name (or None if no env mapping).
SECTION_KEYS = _discover_section_keys()

# All valid keys for CLI validation
ALL_KEYS = set(KEY_MAP) | set(SECTION_KEYS)

SECTION_KEY_MAP = {
    "auth": {
        "LATTIFAI_API_KEY",
        "API_KEY_ID",
        "USER_EMAIL",
        "KEY_NAME",
        "LOGGED_IN_AT",
        "IS_TRIAL",
        "EXPIRES_AT",
        "CREDITS",
    },
    "api": {"GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE_URL"},
    "defaults": {"DEFAULT_AUDIO_FORMAT", "DEFAULT_VIDEO_FORMAT"},
}

SECTION_ORDER = [
    "auth",
    "api",
    "defaults",
    "alignment",
    "caption",
    "client",
    "diarization",
    "event",
    "media",
    "summarization",
    "transcription",
    "translation",
]

# Keys that should be masked in display — top-level + any key ending with "api_key"
SECRET_KEYS = {"LATTIFAI_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"} | {
    k for k in SECTION_KEYS if k.endswith(".api_key")
}


def _mask_value(value: str) -> str:
    """Mask a secret value for display."""
    if len(value) > 12:
        return value[:4] + "..." + value[-4:]
    return "****"


def _load_config() -> dict:
    """Load config from ~/.lattifai/config.toml."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def _normalize_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat config into a section-aware structure."""
    normalized: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, dict):
            normalized[key] = dict(value)
            continue

        section = _get_section_name(key)
        if section:
            normalized.setdefault(section, {})
            normalized[section][key] = value
        else:
            normalized[key] = value

    return normalized


def _get_section_name(key: str) -> Optional[str]:
    """Return the TOML section name for a known config key."""
    for section, keys in SECTION_KEY_MAP.items():
        if key in keys:
            return section
    return None


def _parse_dotted_key(key: str) -> tuple[Optional[str], str]:
    """Parse 'section.key' into (section, key). Returns (None, key) for top-level.

    For multi-level keys like 'translation.llm.model_name', splits on the
    **last** dot: section='translation.llm', subkey='model_name'.
    """
    if "." in key:
        section, _, subkey = key.rpartition(".")
        return section, subkey
    return None, key


def _walk_nested(config: dict, dotted_section: str) -> dict:
    """Walk into nested dicts following a dotted section path.

    For 'translation.llm', returns config['translation']['llm'] (or {}).
    """
    data = config
    for part in dotted_section.split("."):
        if isinstance(data, dict):
            data = data.get(part, {})
        else:
            return {}
    return data if isinstance(data, dict) else {}


def get_config_value(key: str) -> Optional[str]:
    """Get a value from ~/.lattifai/config.toml by key name.

    Supports both top-level keys ('GEMINI_API_KEY') and dotted keys
    ('transcription.model_name', 'translation.llm.model_name').
    Returns None if the key is not set in the config file.
    """
    config = _normalize_config(_load_config())

    # Try dotted key first
    section, subkey = _parse_dotted_key(key)
    if section:
        section_data = _walk_nested(config, section)
        if subkey in section_data:
            return str(section_data[subkey])
        return None

    # Top-level or legacy section lookup
    if key in config:
        return str(config[key])

    legacy_section = _get_section_name(key)
    if legacy_section:
        section_data = config.get(legacy_section, {})
        if key in section_data:
            return str(section_data[key])
    return None


def _format_toml_value(value: Any) -> str:
    """Serialize a Python value into a TOML literal.

    Uses json.dumps for string escaping to correctly handle control characters
    (newlines, tabs, etc.) which are compatible with basic TOML string syntax.
    """
    import json

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _write_table(lines: list[str], prefix: str, data: dict[str, Any]) -> None:
    """Write a TOML table recursively."""
    scalar_items = []
    nested_items = []

    for key, value in data.items():
        if isinstance(value, dict):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if prefix:
        lines.append(f"[{prefix}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")
    if prefix and (scalar_items or nested_items):
        lines.append("")

    for key, value in nested_items:
        child_prefix = f"{prefix}.{key}" if prefix else key
        _write_table(lines, child_prefix, value)


def _ensure_config_permissions() -> None:
    """Apply best-effort permissions for config dir and file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass

    if CONFIG_FILE.exists():
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass


def _save_config(data: dict[str, Any]) -> None:
    """Save config to ~/.lattifai/config.toml."""
    normalized = _normalize_config(data)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass

    lines: list[str] = []
    root_scalars = {key: value for key, value in normalized.items() if not isinstance(value, dict)}
    for key, value in sorted(root_scalars.items()):
        lines.append(f"{key} = {_format_toml_value(value)}")

    if root_scalars:
        lines.append("")

    for section in SECTION_ORDER:
        section_data = normalized.get(section)
        if isinstance(section_data, dict) and section_data:
            _write_table(lines, section, section_data)

    other_sections = {
        key: value for key, value in normalized.items() if isinstance(value, dict) and key not in SECTION_ORDER
    }
    for section, value in sorted(other_sections.items()):
        _write_table(lines, section, value)

    content = "\n".join(line for line in lines).strip()
    file_content = (content + "\n") if content else ""

    # Fix TOCTOU: set umask before writing to ensure file is created with 600
    old_umask = os.umask(0o077)
    try:
        CONFIG_FILE.write_text(file_content)
    finally:
        os.umask(old_umask)

    # Belt-and-suspenders chmod for pre-existing files
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def get_auth_value(key: str) -> Optional[str]:
    """Read a value from the [auth] section."""
    config = _normalize_config(_load_config())
    auth = config.get("auth", {})
    if key in auth:
        return str(auth[key])
    return None


def set_auth_value(key: str, value: Any) -> None:
    """Write a value into the [auth] section."""
    config = _normalize_config(_load_config())
    auth = config.setdefault("auth", {})
    auth[key] = value
    _save_config(config)


def clear_auth() -> None:
    """Remove the entire [auth] section."""
    config = _normalize_config(_load_config())
    config.pop("auth", None)
    _save_config(config)


def _normalize_key(key: str) -> str:
    """Normalize a user-provided key for lookup.

    Top-level keys -> UPPERCASE. Dotted keys -> all parts lowercase.
    Supports multi-level: 'Translation.LLM.Model_Name' -> 'translation.llm.model_name'.
    """
    if "." in key:
        return key.lower()
    return key.upper()


def _resolve_value(key: str) -> tuple[Optional[str], str]:
    """Resolve a config value and its source. Returns (value, source).

    Supports top-level keys and dotted section keys (e.g. 'transcription.model').
    """
    section, subkey = _parse_dotted_key(key)

    if section:
        # Dotted key: check env var if mapped, then config file
        env_name = SECTION_KEYS.get(key)
        if env_name:
            env_val = os.environ.get(env_name)
            if env_val:
                return env_val, "environment"

        config = _normalize_config(_load_config())
        section_data = _walk_nested(config, section)
        if subkey in section_data:
            return str(section_data[subkey]), "config file"

        return None, "not set"

    # Top-level key
    env_name = KEY_MAP.get(key, key.upper())

    env_val = os.environ.get(env_name)
    if env_val:
        return env_val, "environment"

    config = _normalize_config(_load_config())
    if key in config:
        return str(config[key]), "config file"

    legacy_section = _get_section_name(key)
    if legacy_section:
        section_data = config.get(legacy_section, {})
        if key in section_data:
            return str(section_data[key]), "config file"

    # Check .env via dotenv
    try:
        from dotenv import dotenv_values

        dotenv = dotenv_values()
        if env_name in dotenv:
            return dotenv[env_name], ".env file"
    except ImportError:
        pass

    return None, "not set"


app = typer.Typer(help="Manage LattifAI CLI configuration.")


@app.command("show")
def show():
    """Show all configuration values and their sources.

    Top-level keys (API keys, defaults) are always shown.
    Section keys are only shown when they have a value set.
    """
    table = Table(show_header=True, header_style=T.RICH_LABEL, show_lines=False, pad_edge=False)
    table.add_column("Key", min_width=24)
    table.add_column("Value", min_width=30)
    table.add_column("Source", min_width=12)

    # Top-level keys — always shown
    for key in KEY_MAP:
        value, source = _resolve_value(key)
        if value is None:
            display_val = f"[{T.RICH_DIM}]not set[/{T.RICH_DIM}]"
            display_src = f"[{T.RICH_DIM}]-[/{T.RICH_DIM}]"
        else:
            display_value = _mask_value(value) if key in SECRET_KEYS else value
            display_val = f"[{T.RICH_OK}]{display_value}[/{T.RICH_OK}]"
            display_src = source
        table.add_row(key, display_val, display_src)

    # Section-scoped keys — only show keys that have a value
    section_rows = []
    for key in sorted(SECTION_KEYS):
        value, source = _resolve_value(key)
        if value is not None:
            display_value = _mask_value(value) if key in SECRET_KEYS else value
            section_rows.append((key, f"[{T.RICH_OK}]{display_value}[/{T.RICH_OK}]", source))

    if section_rows:
        table.add_row("", "", "")  # separator
        for key, display_val, display_src in section_rows:
            table.add_row(key, display_val, display_src)

    console.print()
    console.print(f"[{T.RICH_HEADER}]LattifAI Configuration[/{T.RICH_HEADER}]")
    console.print(f"[{T.RICH_DIM}]Config file: {CONFIG_FILE}[/{T.RICH_DIM}]")
    console.print()
    console.print(table)
    console.print()


@app.command("set")
def set_value(key: str, value: Optional[str] = typer.Argument(None)):
    """Set a configuration value.

    Accepts both `KEY VALUE` and `KEY=VALUE` syntax.
    Use dotted keys for section values: transcription.model_name, translation.llm.model_name
    """
    # Support KEY=VALUE syntax
    if value is None and "=" in key:
        key, value = key.split("=", 1)
    if value is None:
        console.print(
            f"[{T.RICH_ERR}]Missing value. Usage: lai config set KEY VALUE  or  lai config set KEY=VALUE[/{T.RICH_ERR}]"
        )
        raise typer.Exit(1)

    normalized = _normalize_key(key)
    if normalized not in ALL_KEYS:
        valid_top = ", ".join(KEY_MAP.keys())
        valid_section = ", ".join(SECTION_KEYS.keys())
        console.print(f"[{T.RICH_ERR}]Unknown key: {key}[/{T.RICH_ERR}]")
        console.print(f"Top-level keys: {valid_top}")
        console.print(f"Section keys:   {valid_section}")
        raise typer.Exit(1)

    # Validate format keys before persisting
    if normalized in ("DEFAULT_AUDIO_FORMAT", "DEFAULT_VIDEO_FORMAT"):
        from lattifai.config.media import AUDIO_FORMATS, VIDEO_FORMATS

        fmt = value.strip().lower()
        valid_formats = AUDIO_FORMATS if normalized == "DEFAULT_AUDIO_FORMAT" else VIDEO_FORMATS
        if fmt not in valid_formats:
            console.print(f"[{T.RICH_ERR}]Unsupported format: {value}[/{T.RICH_ERR}]")
            console.print(f"Supported: {', '.join(valid_formats)}")
            raise typer.Exit(1)
        value = fmt

    config = _normalize_config(_load_config())
    section, subkey = _parse_dotted_key(normalized)

    if section:
        # Walk into nested sections, creating dicts as needed
        parts = section.split(".")
        node = config
        for part in parts:
            node = node.setdefault(part, {})
        node[subkey] = value
    else:
        legacy_section = _get_section_name(normalized)
        if legacy_section:
            config.setdefault(legacy_section, {})
            config[legacy_section][normalized] = value
        else:
            config[normalized] = value
    _save_config(config)

    display = _mask_value(value) if normalized in SECRET_KEYS else value
    console.print(f"[{T.RICH_OK}]Set {normalized} = {display}[/{T.RICH_OK}]")


@app.command("get")
def get_value(key: str):
    """Get a configuration value.

    Use dotted keys for section values: transcription.model_name, translation.llm.model_name
    """
    normalized = _normalize_key(key)
    if normalized not in ALL_KEYS:
        valid_top = ", ".join(KEY_MAP.keys())
        valid_section = ", ".join(SECTION_KEYS.keys())
        console.print(f"[{T.RICH_ERR}]Unknown key: {key}[/{T.RICH_ERR}]")
        console.print(f"Top-level keys: {valid_top}")
        console.print(f"Section keys:   {valid_section}")
        raise typer.Exit(1)

    value, source = _resolve_value(normalized)
    if value is None:
        console.print(f"[{T.RICH_WARN}]{normalized} is not set[/{T.RICH_WARN}]")
    else:
        display = _mask_value(value) if normalized in SECRET_KEYS else value
        console.print(f"{normalized} = [{T.RICH_OK}]{display}[/{T.RICH_OK}] ({source})")
