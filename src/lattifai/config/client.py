"""LattifAI Client configuration."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ClientConfig:
    """
    Core alignment configuration.

    Defines model selection, decoding behavior, and API settings for forced alignment.
    """

    _toml_section = "client"

    # API configuration
    api_key: Optional[str] = field(default=None)
    """LattifAI API key. If None, resolved via lattifai.auth.resolve_api_key()."""

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

        # Auto-load API key via unified resolver
        if self.api_key is None:
            try:
                from lattifai.auth import resolve_api_key

                object.__setattr__(self, "api_key", resolve_api_key())
            except ImportError:
                object.__setattr__(self, "api_key", os.environ.get("LATTIFAI_API_KEY"))

        # Auto-load client version from package if not provided
        if self.client_version is None:
            try:
                from importlib.metadata import version

                object.__setattr__(self, "client_version", version("lattifai"))
            except Exception:
                object.__setattr__(self, "client_version", "unknown")

        # Inject X-Device-Auth into default_headers for device-bound keys
        if self.api_key:
            try:
                from lattifai_auth import generate_auth_payload

                payload = generate_auth_payload(self.api_key)
                if self.default_headers is None:
                    object.__setattr__(self, "default_headers", {"X-Device-Auth": payload})
                else:
                    self.default_headers["X-Device-Auth"] = payload
            except (ImportError, RuntimeError, ValueError):
                pass

        # Validate API parameters
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
