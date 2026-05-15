"""Unit tests for lattifai.llm.gemini.GeminiClient and timeout helpers.

Pinned regressions / contracts:
- ``GeminiClient.get_client(http_timeout_ms=N)`` constructs a fresh
  google-genai ``Client`` with ``http_options=HttpOptions(timeout=N)``,
  bypassing the cached default. Required because google-genai 2.x defaults
  ``HttpOptions.timeout`` to ``None`` (infinite). With no timeout, a stalled
  server response keeps the TCP connection open until Linux's TCP keepalive
  (default 7200s) tears it down — observed as a 2h26m Release Tests hang on
  ``gemini-3.1-pro-preview`` in CI run 25904188835.
- ``resolve_http_timeout_ms()`` scales by audio duration at a 1h-audio :
  10min-timeout ratio (i.e. ``audio_sec / 6``), floored at 30s and capped
  at 30min.
- An explicit ``override_ms`` short-circuits the scaling so tests can pin
  a tiny value (e.g. ``100``) and exercise the timeout path quickly.
"""

from unittest.mock import patch

import pytest

from lattifai.transcription.base import resolve_http_timeout_ms


def _import_gemini_client():
    from lattifai.llm.gemini import GeminiClient

    return GeminiClient


class TestGeminiClientGetClient:
    def test_default_get_client_does_not_pin_timeout(self):
        """The cached default client is built without http_options.

        Callers that need a timeout-bounded request pass http_timeout_ms
        explicitly. The cached path stays unchanged so short LLM calls
        (e.g. JSON generation) don't pay per-call construction cost.
        """
        GeminiClient = _import_gemini_client()
        with patch("google.genai.Client") as mock_client_cls:
            client = GeminiClient(api_key="test-key")
            client._get_client()
        assert mock_client_cls.called
        kwargs = mock_client_cls.call_args.kwargs
        assert "http_options" not in kwargs

    def test_get_client_with_timeout_passes_http_options(self):
        """An explicit http_timeout_ms is forwarded as HttpOptions(timeout=N)."""
        GeminiClient = _import_gemini_client()
        with patch("google.genai.Client") as mock_client_cls:
            client = GeminiClient(api_key="test-key")
            client.get_client(http_timeout_ms=12_345)
        kwargs = mock_client_cls.call_args.kwargs
        assert "http_options" in kwargs
        assert kwargs["http_options"].timeout == 12_345

    def test_get_client_with_timeout_is_not_cached(self):
        """Per-call timeout requests return fresh (un-cached) clients.

        Otherwise a small test timeout would leak into subsequent production
        calls. Each timeout-scoped call must build its own Client.
        """
        GeminiClient = _import_gemini_client()
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.side_effect = lambda **kw: object()
            client = GeminiClient(api_key="test-key")
            a = client.get_client(http_timeout_ms=100)
            b = client.get_client(http_timeout_ms=100)
        assert a is not b
        assert mock_client_cls.call_count == 2

    def test_missing_api_key_raises(self):
        GeminiClient = _import_gemini_client()
        client = GeminiClient(api_key=None)
        with pytest.raises(ValueError, match="Gemini API key is required"):
            client._get_client()


class TestResolveHttpTimeoutMs:
    """The 1h-audio : 10min-timeout ratio, with floor/cap clamps."""

    def test_one_hour_audio_maps_to_ten_minutes(self):
        # 3600s audio / 6 = 600s = 600_000 ms = 10 minutes.
        assert resolve_http_timeout_ms(3600.0) == 600_000

    def test_thirty_minute_audio_maps_to_five_minutes(self):
        assert resolve_http_timeout_ms(1800.0) == 300_000

    def test_short_audio_clamps_to_floor(self):
        # Anything that scales below 30s gets the 30s floor.
        # 60s audio / 6 = 10s scaled, but floor=30s.
        assert resolve_http_timeout_ms(60.0) == 30_000
        assert resolve_http_timeout_ms(1.0) == 30_000

    def test_huge_audio_clamps_to_cap(self):
        # 12h audio would scale to 2h timeout; clamp at 30 minutes.
        assert resolve_http_timeout_ms(12 * 3600.0) == 1_800_000

    def test_override_short_circuits_scaling(self):
        # Tests pin small overrides; scaling rules are ignored.
        assert resolve_http_timeout_ms(3600.0, override_ms=100) == 100
        # Override also bypasses the floor — tests intentionally pick tiny values.
        assert resolve_http_timeout_ms(0.0, override_ms=1) == 1

    def test_override_zero_is_honoured_not_treated_as_none(self):
        # 0 is a legal explicit override (immediate timeout); only None auto-scales.
        assert resolve_http_timeout_ms(60.0, override_ms=0) == 0
