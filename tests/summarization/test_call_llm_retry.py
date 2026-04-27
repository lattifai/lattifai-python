"""Tests for ContentSummarizer._call_llm JSON-decode retry behaviour.

Gemini occasionally returns malformed JSON ("Expecting ',' delimiter") that
even ``json-repair`` cannot recover. The summariser should retry once
transparently before surfacing the error to the caller.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from lattifai.config.summarization import SummarizationConfig
from lattifai.summarization.summarizer import ContentSummarizer


@pytest.fixture
def cfg() -> SummarizationConfig:
    return SummarizationConfig()


def _make_summariser(cfg: SummarizationConfig, client: object) -> ContentSummarizer:
    return ContentSummarizer(cfg, client)  # type: ignore[arg-type]


def test_call_llm_retries_once_on_json_decode_error(cfg):
    """First attempt raises JSONDecodeError; second returns a valid dict."""
    valid = {"title": "ok", "summary": "fine"}
    client = type("StubClient", (), {})()
    client.generate_json = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[
            json.JSONDecodeError("Expecting ',' delimiter", "doc", 0),
            valid,
        ]
    )

    s = _make_summariser(cfg, client)
    result = asyncio.run(s._call_llm("hello"))

    assert result == valid
    assert client.generate_json.await_count == 2  # type: ignore[attr-defined]


def test_call_llm_propagates_after_two_failures(cfg):
    """If both attempts fail with JSONDecodeError, the second is re-raised."""
    client = type("StubClient", (), {})()
    client.generate_json = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[
            json.JSONDecodeError("first failure", "doc", 0),
            json.JSONDecodeError("second failure", "doc", 0),
        ]
    )

    s = _make_summariser(cfg, client)
    with pytest.raises(json.JSONDecodeError, match="second failure"):
        asyncio.run(s._call_llm("hello"))

    assert client.generate_json.await_count == 2  # type: ignore[attr-defined]


def test_call_llm_does_not_retry_on_runtime_error(cfg):
    """Non-JSON errors (e.g. wrong return type) propagate without retry."""
    client = type("StubClient", (), {})()
    client.generate_json = AsyncMock(return_value=["not", "a", "dict"])  # type: ignore[attr-defined]

    s = _make_summariser(cfg, client)
    with pytest.raises(RuntimeError, match="Expected dict"):
        asyncio.run(s._call_llm("hello"))

    assert client.generate_json.await_count == 1  # type: ignore[attr-defined]
