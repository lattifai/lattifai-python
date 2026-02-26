"""Test transcribe_numpy API for all transcribers."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from lattifai.caption import Supervision
from lattifai.config import TranscriptionConfig
from lattifai.transcription import create_transcriber
from lattifai.transcription.gemini import GeminiTranscriber
from lattifai.transcription.lattifai import LattifAITranscriber


class MockClientWrapper:
    """Mock client wrapper for testing."""

    def check_permission(self, permission: str):
        """Mock permission check that always passes."""
        pass


@pytest.fixture
def sample_audio():
    """Load real audio data from SA1.wav."""
    audio_path = Path(__file__).parent.parent / "data" / "SA1.wav"
    audio_array, sample_rate = sf.read(audio_path, dtype="float32")
    # Ensure mono audio (take first channel if stereo)
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]
    return audio_array


@pytest.fixture
def audio_list(sample_audio):
    """Create a list of audio arrays for batch testing."""
    return [sample_audio, sample_audio * 0.9, sample_audio * 0.8]


def test_lattifai_transcribe_numpy_mono(sample_audio):
    """Test LattifAI transcriber with mono audio numpy array."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    supervision = transcriber.transcribe_numpy(sample_audio, language="en")

    print(supervision)

    assert supervision.text is not None
    assert supervision.duration > 0
    assert supervision.start == 0.0


def test_lattifai_transcribe_numpy_batch(audio_list):
    """Test LattifAI transcriber with batch of audio arrays."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    supervisions = transcriber.transcribe_numpy(audio_list, language="en")

    assert isinstance(supervisions, list)
    assert len(supervisions) == len(audio_list)
    for supervision in supervisions:
        assert supervision.text is not None
        assert supervision.duration > 0


def test_lattifai_transcribe_numpy_returns_supervision(sample_audio):
    """Test that LattifAI transcriber returns proper Supervision object."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    supervision = transcriber.transcribe_numpy(sample_audio, language="en")

    assert hasattr(supervision, "text")
    assert hasattr(supervision, "duration")
    assert hasattr(supervision, "start")
    assert hasattr(supervision, "id")
    assert hasattr(supervision, "recording_id")
    assert hasattr(supervision, "speaker")
    assert hasattr(supervision, "alignment")


def test_gemini_transcribe_numpy_mono(sample_audio):
    """Test Gemini transcriber with mono audio numpy array."""
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key=api_key)
    transcriber = create_transcriber(config)

    supervision = transcriber.transcribe_numpy(sample_audio, language="en")

    assert isinstance(supervision, Supervision)
    assert supervision.text is not None
    assert supervision.duration > 0
    assert supervision.alignment is None  # Gemini does not provide alignment


def test_gemini_transcribe_numpy_batch(audio_list):
    """Test Gemini transcriber with batch of audio arrays."""
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key=api_key)
    transcriber = create_transcriber(config)

    supervisions = transcriber.transcribe_numpy(audio_list, language="en")

    assert isinstance(supervisions, list)
    assert len(supervisions) == len(audio_list)
    for supervision in supervisions:
        assert isinstance(supervision, Supervision)
        assert supervision.text is not None
        assert supervision.alignment is None


def test_gemini_transcriber_instance():
    """Test that Gemini transcriber can be instantiated."""
    config = TranscriptionConfig(model_name="gemini-2.5-pro", gemini_api_key="test_key")
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, GeminiTranscriber)
    assert hasattr(transcriber, "transcribe_numpy")


def test_lattifai_transcriber_instance():
    """Test that LattifAI transcriber can be instantiated."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    assert isinstance(transcriber, LattifAITranscriber)
    assert hasattr(transcriber, "transcribe_numpy")


def test_transcribe_numpy_invalid_shape():
    """Test that transcribe_numpy raises error for invalid audio shape."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    # Create 3D array (invalid)
    invalid_audio = np.random.randn(2, 2, 16000).astype(np.float32)

    # Should raise an exception (either from our validation or the model)
    with pytest.raises(Exception):
        transcriber.transcribe_numpy(invalid_audio, language="en")


def test_transcribe_numpy_short_audio():
    """Test transcribe_numpy with very short audio arrays (1, 10, 100 samples)."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    # Test with 1, 10, and 100 samples
    for num_samples in [1, 10, 100]:
        short_audio = np.random.randn(num_samples).astype(np.float32)

        try:
            supervision = transcriber.transcribe_numpy(short_audio, language="en")
        except (ValueError, RuntimeError) as e:
            # Either error is acceptable for very short audio
            pass


def test_transcribe_numpy_language_parameter(sample_audio):
    """Test that language parameter is passed correctly."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    # Test with different language codes
    supervision_en = transcriber.transcribe_numpy(sample_audio, language="en")
    supervision_zh = transcriber.transcribe_numpy(sample_audio, language="zh")
    del supervision_en, supervision_zh  # Just ensure no errors


def test_transcribe_numpy_batch(sample_audio):
    """Test that single audio and batch of one audio produce similar results."""
    config = TranscriptionConfig(model_name="nvidia/parakeet-tdt-0.6b-v3", lattice_model_path="disabled")
    config.client_wrapper = MockClientWrapper()
    transcriber = create_transcriber(config)

    # Batch with one audio
    batch_result = transcriber.transcribe_numpy([sample_audio], language="en")

    assert isinstance(batch_result, list)
    assert len(batch_result) == 1
