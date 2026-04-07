"""Shared pytest fixtures for lattifai tests."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a dummy API key and bypass server-side license verification.

    Use this fixture for unit tests that construct a LattifAI client
    but do not need a real API connection.
    """
    monkeypatch.setenv("LATTIFAI_API_KEY", "test_key")
    with patch("lattifai_core.client.core.SyncAPIClient.verify_license"):
        yield
