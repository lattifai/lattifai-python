"""Implementation of 'lai config' for managing LattifAI CLI configuration."""

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

# Mapping: user-facing key name -> environment variable name
KEY_MAP = {
    "lattifai_api_key": "LATTIFAI_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_api_base_url": "OPENAI_API_BASE_URL",
    "default_audio_format": "LATTIFAI_DEFAULT_AUDIO_FORMAT",
    "default_video_format": "LATTIFAI_DEFAULT_VIDEO_FORMAT",
}

SECTION_KEY_MAP = {
    "auth": {"lattifai_api_key", "api_key_id", "user_email", "key_name", "logged_in_at"},
    "api": {"gemini_api_key", "openai_api_key", "openai_api_base_url"},
    "defaults": {"default_audio_format", "default_video_format"},
}

SECTION_ORDER = ["auth", "api", "defaults"]

# Keys that should be masked in display
SECRET_KEYS = {"lattifai_api_key", "gemini_api_key", "openai_api_key"}


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


def get_config_value(key: str) -> Optional[str]:
    """Get a value from ~/.lattifai/config.toml by user-facing key name.

    This is the public API for other config modules to read persisted values.
    Returns None if the key is not set in the config file.
    """
    config = _normalize_config(_load_config())
    if key in config:
        return str(config[key])

    section = _get_section_name(key)
    if section:
        section_data = config.get(section, {})
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


def _resolve_value(key: str) -> tuple[Optional[str], str]:
    """Resolve a config value and its source. Returns (value, source)."""
    env_name = KEY_MAP.get(key, key.upper())

    # Check environment variable first
    env_val = os.environ.get(env_name)
    if env_val:
        return env_val, "environment"

    # Check config file
    config = _normalize_config(_load_config())
    if key in config:
        return str(config[key]), "config file"

    section = _get_section_name(key)
    if section:
        section_data = config.get(section, {})
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
    """Show all configuration values and their sources."""
    table = Table(show_header=True, header_style=T.RICH_LABEL, show_lines=False, pad_edge=False)
    table.add_column("Key", min_width=22)
    table.add_column("Value", min_width=30)
    table.add_column("Source", min_width=12)

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

    console.print()
    console.print(f"[{T.RICH_HEADER}]LattifAI Configuration[/{T.RICH_HEADER}]")
    console.print(f"[{T.RICH_DIM}]Config file: {CONFIG_FILE}[/{T.RICH_DIM}]")
    console.print()
    console.print(table)
    console.print()


@app.command("set")
def set_value(key: str, value: str):
    """Set a configuration value."""
    if key not in KEY_MAP:
        valid = ", ".join(KEY_MAP.keys())
        console.print(f"[{T.RICH_ERR}]Unknown key: {key}[/{T.RICH_ERR}]")
        console.print(f"Valid keys: {valid}")
        raise typer.Exit(1)

    # Validate format keys before persisting
    if key in ("default_audio_format", "default_video_format"):
        from lattifai.config.media import AUDIO_FORMATS, VIDEO_FORMATS

        normalized = value.strip().lower()
        valid_formats = AUDIO_FORMATS if key == "default_audio_format" else VIDEO_FORMATS
        if normalized not in valid_formats:
            console.print(f"[{T.RICH_ERR}]Unsupported format: {value}[/{T.RICH_ERR}]")
            console.print(f"Supported: {', '.join(valid_formats)}")
            raise typer.Exit(1)
        value = normalized

    config = _normalize_config(_load_config())
    section = _get_section_name(key)
    if section:
        config.setdefault(section, {})
        config[section][key] = value
    else:
        config[key] = value
    _save_config(config)

    display = _mask_value(value) if key in SECRET_KEYS else value
    console.print(f"[{T.RICH_OK}]Set {key} = {display}[/{T.RICH_OK}]")


@app.command("get")
def get_value(key: str):
    """Get a configuration value."""
    if key not in KEY_MAP:
        valid = ", ".join(KEY_MAP.keys())
        console.print(f"[{T.RICH_ERR}]Unknown key: {key}[/{T.RICH_ERR}]")
        console.print(f"Valid keys: {valid}")
        raise typer.Exit(1)

    value, source = _resolve_value(key)
    if value is None:
        console.print(f"[{T.RICH_WARN}]{key} is not set[/{T.RICH_WARN}]")
    else:
        display = _mask_value(value) if key in SECRET_KEYS else value
        console.print(f"{key} = [{T.RICH_OK}]{display}[/{T.RICH_OK}] ({source})")
