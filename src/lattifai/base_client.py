"""Base client classes for LattifAI SDK."""

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional

import colorful
import httpx

from .config import ClientConfig
from .errors import ConfigurationError

if TYPE_CHECKING:
    pass


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
        """Make an HTTP request with retry logic."""
        import time

        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(method=method, url=url, json=json, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                # Retry on server errors (5xx)
                if 500 <= e.response.status_code < 600:
                    last_exception = e
                    if attempt < self._max_retries:
                        sleep_time = 0.5 * (2**attempt)  # Exponential backoff
                        print(
                            colorful.yellow(
                                f"    ⚠️ Request failed with {e.response.status_code}. Retrying in {sleep_time:.1f}s ({attempt + 1}/{self._max_retries})..."  # noqa: E501
                            )
                        )
                        time.sleep(sleep_time)
                        continue
                raise e
            except httpx.RequestError as e:
                # Retry on connection errors
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = 0.5 * (2**attempt)
                    print(
                        colorful.yellow(
                            f"    ⚠️ Connection error: {e}. Retrying in {sleep_time:.1f}s ({attempt + 1}/{self._max_retries})..."  # noqa: E501
                        )
                    )
                    time.sleep(sleep_time)
                    continue
                raise e

        if last_exception:
            raise last_exception
        return None  # Should not be reached

    def post(self, api_endpoint: str, *, json: Optional[Dict[str, Any]] = None, **kwargs) -> httpx.Response:
        """Make a POST request to the specified API endpoint."""
        return self._request("POST", api_endpoint, json=json, **kwargs)
