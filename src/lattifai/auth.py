"""LattifAI authentication utilities.

Provides API key management, request signing, and URL resolution.
Used by both the CLI commands and the Python SDK.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from lattifai_auth import deobfuscate_key, generate_auth_payload, obfuscate_key

DEFAULT_SITE_URL = "https://lattifai.com"
DEFAULT_API_URL = "https://api.lattifai.com/v1"


def obfuscate(key: str) -> str:
    """Obfuscate an API key for device-bound local storage.

    Returns empty/None keys unchanged.
    """
    if not key:
        return key
    return obfuscate_key(key)


def deobfuscate(raw: Optional[str]) -> Optional[str]:
    """Deobfuscate a stored API key.

    Returns plaintext keys unchanged. Raises RuntimeError when the key
    cannot be recovered (wrong device or corrupted).
    """
    if not raw:
        return raw
    if not raw.startswith("v1:"):
        return raw
    try:
        return deobfuscate_key(raw)
    except RuntimeError:
        raise RuntimeError("Stored API key is bound to a different device.\nRun:  lai auth login")
    except ValueError:
        raise RuntimeError("Stored API key is malformed or corrupted.\nRun:  lai auth login")


def _resolve_env(key: str, default: str = "") -> str:
    """Resolve a value: os.environ > .env file > default."""
    return os.environ.get(key) or load_dotenv_value(key) or default


def resolve_site_url(site_url: Optional[str] = None) -> str:
    """Resolve the web site URL (authorization page + code exchange)."""
    return (site_url or _resolve_env("LATTIFAI_SITE_URL", DEFAULT_SITE_URL)).rstrip("/")


def resolve_api_url(api_url: Optional[str] = None) -> str:
    """Resolve the backend API URL (whoami + session revoke).

    Strips trailing /v1 if present — endpoint paths already include it.
    """
    url = (api_url or _resolve_env("LATTIFAI_BASE_URL", DEFAULT_API_URL)).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def load_dotenv_value(key: str) -> Optional[str]:
    """Read a value from the nearest .env file without mutating the environment."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:
        return None
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return None
    value = dotenv_values(dotenv_path).get(key)
    return str(value) if value else None


def resolve_api_key() -> Optional[str]:
    """Resolve API key: env var > config.toml [auth] > .env fallback.

    Deobfuscates stored keys automatically.
    """
    # 1. env var or .env
    if key := _resolve_env("LATTIFAI_API_KEY"):
        return key

    # 2. config.toml [auth] session (obfuscated)
    try:
        from lattifai.cli.config import get_auth_value

        raw = get_auth_value("LATTIFAI_API_KEY")
        if raw:
            return deobfuscate(raw)
    except ImportError:
        pass

    return None


def auth_headers(api_key: str) -> dict[str, str]:
    """Build authorization headers with X-Device-Auth HMAC signature."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        headers["X-Device-Auth"] = generate_auth_payload(api_key)
    except (RuntimeError, ValueError):
        pass
    return headers


def request_whoami(api_key: str, api_url: Optional[str] = None) -> dict[str, Any]:
    """Fetch current auth metadata from the backend API."""
    url = resolve_api_url(api_url)
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{url}/v1/auth/whoami", headers=auth_headers(api_key))
        response.raise_for_status()
        return response.json()


def revoke_session(api_key: str, api_url: Optional[str] = None) -> dict[str, Any]:
    """Revoke the current API key session via the backend API."""
    url = resolve_api_url(api_url)
    with httpx.Client(timeout=15.0) as client:
        response = client.delete(f"{url}/v1/auth/session", headers=auth_headers(api_key))
        response.raise_for_status()
        return response.json()
