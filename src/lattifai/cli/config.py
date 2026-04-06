"""Implementation of 'lai config' for managing LattifAI CLI configuration."""

import os
from pathlib import Path
from typing import Optional

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


def get_config_value(key: str) -> Optional[str]:
    """Get a value from ~/.lattifai/config.toml by user-facing key name.

    This is the public API for other config modules to read persisted values.
    Returns None if the key is not set in the config file.
    """
    config = _load_config()
    if key in config:
        return str(config[key])
    return None


def _save_config(data: dict) -> None:
    """Save config to ~/.lattifai/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in sorted(data.items()):
        # Quote string values for TOML
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def _resolve_value(key: str) -> tuple[Optional[str], str]:
    """Resolve a config value and its source. Returns (value, source)."""
    env_name = KEY_MAP.get(key, key.upper())

    # Check environment variable first
    env_val = os.environ.get(env_name)
    if env_val:
        return env_val, "environment"

    # Check config file
    config = _load_config()
    if key in config:
        return str(config[key]), "config file"

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

    config = _load_config()
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
