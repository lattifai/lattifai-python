"""Unit tests for LatticeTokenizer.split_sentences threshold passthrough.

Verifies that the wtpsplit threshold flows correctly from
``CaptionInputConfig.split_threshold`` through ``LatticeTokenizer`` down to
the underlying ``SentenceSplitter`` instance from lattifai-captions.

These are pure passthrough tests; the real splitter model is mocked out so
the test does not download a wtpsplit model.
"""

from unittest.mock import MagicMock

import pytest

from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision


@pytest.fixture
def tokenizer_with_mock_splitter():
    """LatticeTokenizer with sentence_splitter pre-stubbed (no model load)."""
    client = MagicMock()
    tok = LatticeTokenizer(client_wrapper=client)

    # Bypass init_sentence_splitter() so no wtpsplit/onnx model is fetched.
    mock_splitter = MagicMock()
    mock_splitter.split_sentences = MagicMock(return_value=[])
    tok.sentence_splitter = mock_splitter
    tok.init_sentence_splitter = MagicMock()  # no-op; splitter already set

    return tok, mock_splitter


def _supervisions():
    return [Supervision(text="hello world", start=0.0, duration=1.0)]


class TestSplitSentencesThresholdPassthrough:
    """Threshold flows from caller into the underlying SentenceSplitter."""

    def test_default_threshold_is_0_35(self, tokenizer_with_mock_splitter):
        """When no threshold is given, tokenizer uses 0.35 (library default)."""
        tok, mock_splitter = tokenizer_with_mock_splitter
        tok.split_sentences(_supervisions())

        mock_splitter.split_sentences.assert_called_once()
        kwargs = mock_splitter.split_sentences.call_args.kwargs
        assert kwargs["threshold"] == 0.35

    def test_custom_threshold_is_passed_through(self, tokenizer_with_mock_splitter):
        """Explicit threshold value reaches SentenceSplitter unchanged."""
        tok, mock_splitter = tokenizer_with_mock_splitter
        tok.split_sentences(_supervisions(), threshold=0.10)

        kwargs = mock_splitter.split_sentences.call_args.kwargs
        assert kwargs["threshold"] == 0.10

    def test_aggressive_threshold_0_20_is_supported(self, tokenizer_with_mock_splitter):
        """More aggressive 0.20 must remain selectable for dense monologues."""
        tok, mock_splitter = tokenizer_with_mock_splitter
        tok.split_sentences(_supervisions(), threshold=0.20)

        kwargs = mock_splitter.split_sentences.call_args.kwargs
        assert kwargs["threshold"] == 0.20

    def test_strip_whitespace_still_passed(self, tokenizer_with_mock_splitter):
        """Existing strip_whitespace kwarg must still flow through."""
        tok, mock_splitter = tokenizer_with_mock_splitter
        tok.split_sentences(_supervisions(), strip_whitespace=False, threshold=0.15)

        kwargs = mock_splitter.split_sentences.call_args.kwargs
        assert kwargs["strip_whitespace"] is False
        assert kwargs["threshold"] == 0.15
