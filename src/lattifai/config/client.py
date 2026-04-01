"""LattifAI Client configuration."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _deobfuscate_stored_key(raw: Optional[str]) -> Optional[str]:
    """Deobfuscate a stored API key from config.toml.

    Independent from cli/auth.py to avoid circular imports.
    Returns plaintext keys unchanged. Raises RuntimeError for
    unrecoverable v1: ciphertext (wrong device or corrupted).
    """
    if not raw or not raw.startswith("v1:"):
        return raw
    from lattifai_auth import deobfuscate_key

    try:
        return deobfuscate_key(raw)
    except RuntimeError:
        raise RuntimeError("Stored LattifAI API key is bound to a different device. " "Run: lai auth login")
    except ValueError:
        raise RuntimeError("Stored LattifAI API key is malformed or corrupted. " "Run: lai auth login")


@dataclass
class ClientConfig:
    """
    Core alignment configuration.

    Defines model selection, decoding behavior, and API settings for forced alignment.
    """

    # API configuration
    api_key: Optional[str] = field(default=None)
    """LattifAI API key. If None, reads from LATTIFAI_API_KEY environment variable."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    max_retries: int = 2
    """Maximum number of retry attempts for failed requests."""

    default_headers: Optional[Dict[str, str]] = field(default=None)
    """Optional static headers to include in all requests."""

    profile: bool = False
    """Enable profiling of client operations tasks.
    When True, prints detailed timing information for various stages of the process.
    """

    # Client identification for usage tracking
    client_name: Optional[str] = field(default="python-sdk")
    """Client identifier for usage tracking (e.g., 'python-sdk', 'claude-plugin')."""

    client_version: Optional[str] = field(default=None)
    """Client version for usage tracking. If None, uses lattifai package version."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""

        # Auto-load API key: CLI arg > env var > config.toml [auth] > .env
        if self.api_key is None:
            env_val = os.environ.get("LATTIFAI_API_KEY")

            if not env_val:
                try:
                    from lattifai.cli.config import get_auth_value

                    raw = get_auth_value("lattifai_api_key")
                    env_val = _deobfuscate_stored_key(raw)
                except ImportError:
                    pass

            if not env_val:
                try:
                    from dotenv import dotenv_values, find_dotenv

                    dotenv_path = find_dotenv(usecwd=True)
                    if dotenv_path:
                        dotenv_value = dotenv_values(dotenv_path).get("LATTIFAI_API_KEY")
                        env_val = str(dotenv_value) if dotenv_value else None
                except ImportError:
                    pass

            object.__setattr__(self, "api_key", env_val)

        # Auto-load client version from package if not provided
        if self.client_version is None:
            try:
                from importlib.metadata import version

                object.__setattr__(self, "client_version", version("lattifai"))
            except Exception:
                object.__setattr__(self, "client_version", "unknown")

        # Validate API parameters
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
