"""Base client classes for LattifAI SDK."""

import os
from abc import ABC
from typing import Any, Awaitable, Callable, Dict, Optional, Union  # noqa: F401

import httpx

from .config import ClientConfig

# Import from errors module for consistency
from .errors import APIError, ConfigurationError


class BaseAPIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(
        self,
        config: ClientConfig,
    ) -> None:
        if config.api_key is None:
            raise ConfigurationError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LATTIFAI_API_KEY environment variable"
            )

        self._api_key = config.api_key
        self._base_url = config.base_url
        self._timeout = config.timeout
        self._max_retries = config.max_retries

        headers = {
            "User-Agent": "LattifAI/Python",
            "Authorization": f"Bearer {self._api_key}",
        }
        if config.default_headers:
            headers.update(config.default_headers)
        self._default_headers = headers


class SyncAPIClient(BaseAPIClient):
    """Synchronous API client."""

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._default_headers,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request."""
        return self._client.request(method=method, url=url, json=json, **kwargs)

    def post(self, api_endpoint: str, *, json: Optional[Dict[str, Any]] = None, **kwargs) -> httpx.Response:
        """Make a POST request to the specified API endpoint."""
        return self._request("POST", api_endpoint, json=json, **kwargs)


class AsyncAPIClient(BaseAPIClient):
    """Asynchronous API client."""

    def __init__(self, config: ClientConfig) -> None:
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._default_headers,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an HTTP request."""
        return await self._client.request(method=method, url=url, json=json, files=files, **kwargs)

    async def post(
        self,
        api_endpoint: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make a POST request to the specified API endpoint."""
        return await self._request("POST", api_endpoint, json=json, files=files, **kwargs)
