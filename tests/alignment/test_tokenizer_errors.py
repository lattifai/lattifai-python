"""Unit tests for LatticeTokenizer error handling paths.

Tests HTTP status code handling in tokenize() and detokenize() methods.
"""

from unittest.mock import MagicMock

import pytest

from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision
from lattifai.errors import AuthenticationError, LatticeDecodingError, QuotaExceededError


@pytest.fixture
def tokenizer():
    """Create a LatticeTokenizer with mocked client_wrapper and prenormalize."""
    client = MagicMock()
    tok = LatticeTokenizer(client_wrapper=client)
    tok.model_name = "LattifAI/Lattice-1"
    tok.words = ["hello", "world"]
    # Mock prenormalize to skip G2P model dependency
    tok.prenormalize = MagicMock(return_value={})
    return tok


def _make_response(status_code, text="error", json_data=None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


def _supervisions():
    """Create minimal supervisions for tokenize()."""
    return [Supervision(text="hello world", start=0.0, duration=1.0)]


def _detokenize_args():
    """Create minimal arguments for detokenize()."""
    import numpy as np

    lattice_id = "test-lattice-id"
    emission_stats = {"max_probs": [], "aligned_probs": []}
    results = [[MagicMock()]]
    labels = [["h", "e", "l", "l", "o"]]
    frame_shift = 0.01
    offset = 0.0
    channel = 0
    lattice_results = (emission_stats, results, labels, frame_shift, offset, channel)
    return lattice_id, lattice_results


# ===========================================================================
# tokenize() error paths
# ===========================================================================


class TestTokenize401:
    """Test 401 Unauthorized handling in tokenize()."""

    def test_raises_authentication_error(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(401, "Unauthorized: invalid API key")
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            tokenizer.tokenize(_supervisions())

    def test_includes_server_response(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(401, "trial key not permitted")
        with pytest.raises(AuthenticationError, match="trial key not permitted"):
            tokenizer.tokenize(_supervisions())

    def test_includes_login_hint(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(401, "expired")
        with pytest.raises(AuthenticationError, match="lai auth login"):
            tokenizer.tokenize(_supervisions())


class TestTokenize402:
    """Test 402 Quota Exceeded handling in tokenize()."""

    def test_raises_quota_error(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(402, json_data={"detail": "Monthly quota exceeded"})
        with pytest.raises(QuotaExceededError, match="Monthly quota exceeded"):
            tokenizer.tokenize(_supervisions())

    def test_default_message(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(402, json_data={})
        with pytest.raises(QuotaExceededError, match="Quota exceeded"):
            tokenizer.tokenize(_supervisions())


class TestTokenizeGenericError:
    """Test generic non-200 handling in tokenize()."""

    def test_500_raises_exception(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(500, "Internal Server Error")
        with pytest.raises(Exception, match="Failed to tokenize texts"):
            tokenizer.tokenize(_supervisions())

    def test_503_raises_exception(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(503, "Service Unavailable")
        with pytest.raises(Exception, match="Failed to tokenize texts"):
            tokenizer.tokenize(_supervisions())


class TestTokenizeSuccess:
    """Test tokenize() success path returns expected structure."""

    def test_returns_tuple(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(
            200,
            json_data={
                "id": "lattice-123",
                "lattice_graph": [[1, 2]],
                "final_state": 42,
                "acoustic_scale": 1.0,
            },
        )
        result = tokenizer.tokenize(_supervisions())
        supervisions, lattice_id, graph_data, diff_detok = result
        assert lattice_id == "lattice-123"
        assert graph_data[0] == [[1, 2]]
        assert graph_data[1] == 42


# ===========================================================================
# detokenize() error paths
# ===========================================================================


class TestDetokenize400:
    """Test 400 Bad Request handling in detokenize()."""

    def test_raises_lattice_decoding_error(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(400, "bad lattice")
        lattice_id, lattice_results = _detokenize_args()
        with pytest.raises(LatticeDecodingError):
            tokenizer.detokenize(lattice_id, lattice_results, _supervisions())


class TestDetokenize401:
    """Test 401 Unauthorized handling in detokenize()."""

    def test_raises_authentication_error(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(401, "Unauthorized")
        lattice_id, lattice_results = _detokenize_args()
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            tokenizer.detokenize(lattice_id, lattice_results, _supervisions())

    def test_includes_server_response(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(401, "token expired at 2026-04-01")
        lattice_id, lattice_results = _detokenize_args()
        with pytest.raises(AuthenticationError, match="token expired at 2026-04-01"):
            tokenizer.detokenize(lattice_id, lattice_results, _supervisions())


class TestDetokenize402:
    """Test 402 Quota Exceeded handling in detokenize()."""

    def test_raises_quota_error(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(402, json_data={"detail": "Daily limit reached"})
        lattice_id, lattice_results = _detokenize_args()
        with pytest.raises(QuotaExceededError, match="Daily limit reached"):
            tokenizer.detokenize(lattice_id, lattice_results, _supervisions())


class TestDetokenizeGenericError:
    """Test generic non-200 handling in detokenize()."""

    def test_500_raises_exception(self, tokenizer):
        tokenizer.client_wrapper.post.return_value = _make_response(500, "Internal Server Error")
        lattice_id, lattice_results = _detokenize_args()
        with pytest.raises(Exception, match="Failed to detokenize lattice"):
            tokenizer.detokenize(lattice_id, lattice_results, _supervisions())
