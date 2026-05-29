"""Unit tests for `LatticeTokenizer.tokenize` split_sentence semantics on the
difftokenize branch (TextAlignResult input).

Historically `split_sentence=True` was a silent no-op when the input was a
TextAlignResult (sub_align + asr_align list), while it actively resegmented
inputs in the Supervision-list branch — an interface inconsistency that hid
bugs. These tests verify the parameter now applies symmetrically:

  * split_sentence=False: zero splitter calls (default, fast path)
  * split_sentence=True:  splitter invoked once for sub_align AND once for
    asr_align, and the in-place mutation reaches the returned supervisions
    so downstream detokenize sees the resegmented inputs.

The HTTP client and split_sentences are mocked — no model load, no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision


@pytest.fixture
def tokenizer_with_mocks():
    """LatticeTokenizer with sentence_splitter + HTTP client mocked."""
    client = MagicMock()

    # difftokenize endpoint returns a well-formed lattice response.
    http_resp = MagicMock()
    http_resp.status_code = 200
    http_resp.json.return_value = {
        "id": "lattice-test-id",
        "lattice_graph": "graph-bytes",
        "final_state": 42,
        "acoustic_scale": 1.0,
        "diff_detokenize": True,
    }
    client.post.return_value = http_resp

    tok = LatticeTokenizer(client_wrapper=client)

    # Stub out the splitter so no wtpsplit/onnx model is fetched.
    # _sentinel is a deterministic substitute for "the splitter ran on input X".
    mock_splitter = MagicMock()
    tok.sentence_splitter = mock_splitter
    tok.init_sentence_splitter = MagicMock()

    # Stub prenormalize → identity (no pronunciation dict lookup over network).
    tok.prenormalize = MagicMock(return_value={})

    return tok, mock_splitter, client


def _sup(text: str, start: float, end: float) -> Supervision:
    return Supervision(text=text, start=start, duration=end - start)


def _make_match() -> list:
    """TextAlignResult-shaped input: [sub_align, asr_align, aligned, ts, chunk].

    Only [0] and [1] (the supervision lists) matter for tokenize semantics.
    """
    sub_align = [_sup("Hello world. How are you?", 0.0, 3.0)]
    asr_align = [_sup("hello world how are you", 0.0, 3.0)]
    return [sub_align, asr_align, None, None, None]


class TestDifftokenizeSplitSentence:
    """split_sentence applied symmetrically on the TextAlignResult branch."""

    def test_split_false_skips_splitter(self, tokenizer_with_mocks):
        """Default path — splitter never invoked, supervisions unchanged."""
        tok, mock_splitter, _ = tokenizer_with_mocks
        match = _make_match()
        original_sub = match[0]
        original_asr = match[1]

        result_sups, _lattice_id, _lattice, _diff = tok.tokenize(
            match,
            split_sentence=False,
            boost=5.0,
        )

        mock_splitter.split_sentences.assert_not_called()
        assert result_sups[0] is original_sub
        assert result_sups[1] is original_asr

    def test_split_true_invokes_splitter_on_both_sides(self, tokenizer_with_mocks):
        """split_sentence=True triggers splitter for sub_align AND asr_align."""
        tok, mock_splitter, _ = tokenizer_with_mocks

        # Splitter returns a marker list so we can verify each side was processed
        # independently (i.e. two distinct calls, not one shared invocation).
        split_sub = [_sup("Hello world.", 0.0, 1.5), _sup("How are you?", 1.5, 3.0)]
        split_asr = [_sup("hello world", 0.0, 1.5), _sup("how are you", 1.5, 3.0)]
        mock_splitter.split_sentences.side_effect = [split_sub, split_asr]

        match = _make_match()
        result_sups, _lid, _lattice, _diff = tok.tokenize(
            match,
            split_sentence=True,
            boost=5.0,
        )

        # Two distinct splitter calls — one per side.
        assert mock_splitter.split_sentences.call_count == 2

        # Mutation visible on the returned supervisions reference.
        assert result_sups[0] is split_sub
        assert result_sups[1] is split_asr

    def test_split_true_resegmented_lists_reach_backend_request(self, tokenizer_with_mocks):
        """The split outputs (not the originals) are what get serialized to
        the difftokenize HTTP request body."""
        tok, mock_splitter, client = tokenizer_with_mocks

        split_sub = [_sup("Hello world.", 0.0, 1.5), _sup("How are you?", 1.5, 3.0)]
        split_asr = [_sup("hello world", 0.0, 1.5), _sup("how are you", 1.5, 3.0)]
        mock_splitter.split_sentences.side_effect = [split_sub, split_asr]

        match = _make_match()
        tok.tokenize(match, split_sentence=True, boost=5.0)

        # The request body should reflect the split versions (2 sups each), not the
        # original 1-sup per side.
        post_kwargs = client.post.call_args.kwargs
        body = post_kwargs["json"]
        assert len(body["supervisions"]) == 2
        assert len(body["transcription"]) == 2
        assert body["supervisions"][0]["text"] == "Hello world."
        assert body["transcription"][1]["text"] == "how are you"

    def test_split_false_sends_originals_to_backend(self, tokenizer_with_mocks):
        """Sanity: with split_sentence=False, the original sub_align / asr_align
        reach the backend untouched. Guards against accidental mutation."""
        tok, _, client = tokenizer_with_mocks
        match = _make_match()
        tok.tokenize(match, split_sentence=False, boost=5.0)

        body = client.post.call_args.kwargs["json"]
        assert len(body["supervisions"]) == 1
        assert len(body["transcription"]) == 1
        assert body["supervisions"][0]["text"] == "Hello world. How are you?"
        assert body["transcription"][0]["text"] == "hello world how are you"
