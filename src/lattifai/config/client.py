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

        # Load environment variables from .env file
        from dotenv import find_dotenv, load_dotenv

        # Try to find and load .env file from current directory or parent directories
        load_dotenv(find_dotenv(usecwd=True))

        # Auto-load API key from environment if not provided
        if self.api_key is None:
            object.__setattr__(self, "api_key", os.environ.get("LATTIFAI_API_KEY"))

        # Auto-load client version from package if not provided
        if self.client_version is None:
            try:
                from lattifai import __version__

                object.__setattr__(self, "client_version", __version__)
            except ImportError:
                object.__setattr__(self, "client_version", "unknown")

        # Validate API parameters
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
