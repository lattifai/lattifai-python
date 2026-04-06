"""Test Qwen3-ASR accepts numpy array input (reproduces VAD slice bug).

The qwen_asr library's prepare_audio() accepts:
  - str (file path)
  - tuple(np.ndarray, int) (audio + sample_rate)

But NOT bare np.ndarray. When VAD slices audio via _slice_audio_by_segments(),
the result is List[np.ndarray] — which qwen_asr rejects with:
  TypeError: Unsupported audio input type: <class 'numpy.ndarray'>

This test verifies that LattifAITranscriber wraps numpy arrays into the
(array, sample_rate) tuple format that qwen_asr expects.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig

_ASR_SAMPLE_RATE = 16000


class MockClientWrapper:
    def check_permission(self, permission: str):
        pass


@pytest.fixture
def qwen_transcriber():
    """Create a Qwen3-ASR transcriber with mocked model."""
    config = TranscriptionConfig(model_name="Qwen/Qwen3-ASR-0.6B", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()

    from lattifai.transcription.lattifai import LattifAITranscriber

    transcriber = LattifAITranscriber(transcription_config=config)
    return transcriber


@pytest.fixture
def mock_qwen_model():
    """Create a mock Qwen3ASRModel that validates input types."""
    model = MagicMock()

    def mock_transcribe(audio, language=None):
        results = []
        for a in audio:
            # qwen_asr expects str or tuple(ndarray, int) — NOT bare ndarray
            if isinstance(a, np.ndarray):
                raise TypeError(f"Unsupported audio input type: {type(a)}")
            if isinstance(a, tuple) and len(a) == 2:
                arr, sr = a
                assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
                assert isinstance(sr, int), f"Expected int sample_rate, got {type(sr)}"
            elif isinstance(a, str):
                pass  # file path, OK
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")

            result = MagicMock()
            result.text = "hello world"
            result.language = "en"
            results.append(result)
        return results

    model.transcribe = mock_transcribe
    return model


def test_qwen_asr_rejects_bare_numpy(qwen_transcriber, mock_qwen_model):
    """Verify that wrapping numpy arrays as (array, sr) tuples fixes the bug.

    Before fix: bare np.ndarray passed to qwen_asr → TypeError
    After fix: (np.ndarray, 16000) tuple passed → success
    """
    qwen_transcriber._asr_model = mock_qwen_model

    # Simulate VAD-sliced audio: list of 1D numpy arrays
    audio_segments = [
        np.random.randn(16000).astype(np.float32),
        np.random.randn(32000).astype(np.float32),
    ]

    # This should NOT raise TypeError after the fix
    supervisions, _ = qwen_transcriber._transcribe_impl(
        audio=audio_segments,
        language="zh",
        return_hypotheses=True,
        progress_bar=False,
    )

    assert len(supervisions) == 2
    for sup in supervisions:
        assert isinstance(sup, Supervision)
        assert sup.text == "hello world"


def test_qwen_asr_single_numpy(qwen_transcriber, mock_qwen_model):
    """Verify single numpy array is also wrapped correctly."""
    qwen_transcriber._asr_model = mock_qwen_model

    single_audio = np.random.randn(16000).astype(np.float32)

    supervisions, _ = qwen_transcriber._transcribe_impl(
        audio=single_audio,
        language="en",
        return_hypotheses=True,
        progress_bar=False,
    )

    assert len(supervisions) == 1
    assert supervisions[0].text == "hello world"
    assert supervisions[0].duration == pytest.approx(1.0, rel=0.01)
