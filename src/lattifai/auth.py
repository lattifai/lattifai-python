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


# ---------------------------------------------------------------------------
# API key obfuscation
# ---------------------------------------------------------------------------


def obfuscate(key: str) -> str:
    """Obfuscate an API key for device-bound local storage."""
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
        return raw  # plaintext — pass through
    try:
        return deobfuscate_key(raw)
    except RuntimeError:
        raise RuntimeError("Stored API key is bound to a different device.\nRun:  lai auth login")
    except ValueError:
        raise RuntimeError("Stored API key is malformed or corrupted.\nRun:  lai auth login")


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def resolve_site_url(site_url: Optional[str] = None) -> str:
    """Resolve the web site URL (authorization page + code exchange)."""
    return (site_url or os.environ.get("LATTIFAI_SITE_URL") or DEFAULT_SITE_URL).rstrip("/")


def resolve_api_url(api_url: Optional[str] = None) -> str:
    """Resolve the backend API URL (whoami + session revoke).

    Strips trailing /v1 if present — endpoint paths already include it.
    """
    url = (api_url or os.environ.get("LATTIFAI_BASE_URL") or DEFAULT_API_URL).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def resolve_api_key() -> Optional[str]:
    """Resolve API key: env var > config.toml [auth] session > .env fallback.

    Deobfuscates stored keys automatically.
    """
    if key := os.environ.get("LATTIFAI_API_KEY"):
        return key

    # config.toml [auth] session
    try:
        from lattifai.cli.config import get_auth_value

        raw = get_auth_value("LATTIFAI_API_KEY")
        if raw:
            return deobfuscate(raw)
    except ImportError:
        pass

    # .env fallback
    try:
        from dotenv import dotenv_values, find_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            val = dotenv_values(dotenv_path).get("LATTIFAI_API_KEY")
            if val:
                return str(val)
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Request signing
# ---------------------------------------------------------------------------


def auth_headers(api_key: str) -> dict[str, str]:
    """Build authorization headers with X-Device-Auth HMAC signature."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        headers["X-Device-Auth"] = generate_auth_payload(api_key)
    except (RuntimeError, ValueError):
        pass  # non-fatal: request proceeds without device auth
    return headers


# ---------------------------------------------------------------------------
# Backend API calls
# ---------------------------------------------------------------------------


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
