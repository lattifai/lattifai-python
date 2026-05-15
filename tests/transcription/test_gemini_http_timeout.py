"""Integration: GeminiTranscriber must scale Gemini HTTP timeout by audio duration.

Pinned regression: with google-genai 2.x's default ``HttpOptions.timeout=None``,
``transcribe_numpy`` on ``gemini-3.1-pro-preview`` hung Release Tests for 2h26m
in CI run 25904188835 (until Linux TCP keepalive tore the socket down).
"""

from unittest.mock import MagicMock, patch

import numpy as np

from lattifai.config import TranscriptionConfig
from lattifai.transcription.gemini import GeminiTranscriber


def _make_fake_client_factory(call_log):
    """Build a fake google.genai.Client factory that records http_options.timeout."""

    def factory(**kwargs):
        http_opts = kwargs.get("http_options")
        call_log.append(http_opts.timeout if http_opts is not None else None)
        client = MagicMock()
        uploaded = MagicMock(uri="gs://fake/file", mime_type="audio/wav")
        client.files.upload.return_value = uploaded
        response = MagicMock(text="hello world")
        client.models.generate_content.return_value = response
        return client

    return factory


def _make_audio(seconds: int, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype="float32")


class TestTranscribeNumpyTimeoutPropagation:
    def test_short_audio_uses_floor_for_both_upload_and_generate(self):
        """60s audio falls below the scaled budget → 30s floor applied to both calls."""
        call_log: list = []
        config = TranscriptionConfig(model_name="gemini-3.1-pro-preview", gemini_api_key="test-key")
        with patch("google.genai.Client", side_effect=_make_fake_client_factory(call_log)):
            transcriber = GeminiTranscriber(config)
            sup = transcriber.transcribe_numpy(_make_audio(60), language="en")

        assert sup.text == "hello world"
        # Two timeout-scoped Client() calls — one for upload, one for generate_content.
        assert call_log == [
            30_000,
            30_000,
        ], f"Expected both upload and generate_content to use 30s floor; got {call_log}"

    def test_one_hour_audio_uses_ten_minute_budget(self):
        """3600s audio scales to 600_000 ms (10 minutes) by the 1:6 ratio."""
        call_log: list = []
        config = TranscriptionConfig(model_name="gemini-3.1-pro-preview", gemini_api_key="test-key")
        with patch("google.genai.Client", side_effect=_make_fake_client_factory(call_log)):
            transcriber = GeminiTranscriber(config)
            transcriber.transcribe_numpy(_make_audio(3600), language="en")

        assert call_log == [600_000, 600_000]

    def test_explicit_override_wins_over_duration_scaling(self):
        """``config.http_timeout_ms`` short-circuits the auto-scaler — tests pin small values."""
        call_log: list = []
        config = TranscriptionConfig(
            model_name="gemini-3.1-pro-preview",
            gemini_api_key="test-key",
            http_timeout_ms=500,  # tiny pin
        )
        with patch("google.genai.Client", side_effect=_make_fake_client_factory(call_log)):
            transcriber = GeminiTranscriber(config)
            # Even a long audio must respect the user-pinned override.
            transcriber.transcribe_numpy(_make_audio(3600), language="en")

        assert call_log == [500, 500]
