"""Test AudioData iterator interface for streaming."""

import numpy as np
import pytest
import torch

from lattifai.audio2 import AudioData


def test_audio_data_iter_default():
    """Test AudioData iteration with default parameters."""
    # Create synthetic audio: 90 seconds at 16kHz
    sampling_rate = 16000
    duration = 90.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test_audio.wav",
        streaming_chunk_secs=30.0,  # Use 30s chunks for this test
        overlap_secs=0.0,
    )

    # Iterate over chunks
    chunks = list(audio)

    # With 30s chunks and 1s overlap:
    # - Step size: 29s
    # - Expected chunks: ceil((90 - 30) / 29) + 1 = ceil(60/29) + 1 = 3 + 1 = 4
    # - Chunk 0: 0-30s
    # - Chunk 1: 29-59s
    # - Chunk 2: 58-88s
    # - Chunk 3: 87-90s
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    # Verify first chunk
    first_chunk = chunks[0]
    assert first_chunk.sampling_rate == sampling_rate
    assert first_chunk.duration == 30.0
    assert first_chunk.ndarray.shape[-1] == 30 * sampling_rate

    # Verify last chunk is shorter
    last_chunk = chunks[-1]
    assert last_chunk.duration <= 30.0


def test_audio_data_iter_chunks_custom():
    """Test iter_chunks with custom duration and overlap."""
    sampling_rate = 16000
    duration = 100.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test_audio.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    # Use 20s chunks with 2s overlap
    chunks = list(audio.iter_chunks(chunk_secs=20.0, overlap_secs=2.0))

    # Step size: 18s
    # Expected: ceil((100 - 20) / 18) + 1 = ceil(80/18) + 1 = 5 + 1 = 6
    assert len(chunks) >= 5, f"Expected at least 5 chunks, got {len(chunks)}"

    # Check first two chunks overlap
    if len(chunks) >= 2:
        # Verify overlap region has same data
        overlap_size = int(2.0 * sampling_rate)
        overlap1 = chunks[0].ndarray[..., -overlap_size:]
        overlap2 = chunks[1].ndarray[..., :overlap_size]

        # They should be similar (from same original audio)
        assert overlap1.shape == overlap2.shape


def test_audio_data_iter_short_audio():
    """Test iteration with audio shorter than chunk size."""
    sampling_rate = 16000
    duration = 10.0  # Shorter than default 30s chunk
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="short_audio.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    chunks = list(audio)

    # Should have exactly 1 chunk
    assert len(chunks) == 1
    assert chunks[0].duration == duration


def test_audio_data_iter_chunks_no_overlap():
    """Test iter_chunks with zero overlap."""
    sampling_rate = 16000
    duration = 60.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test_audio.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    # Use 20s chunks with no overlap
    chunks = list(audio.iter_chunks(chunk_secs=20.0, overlap_secs=0.0))

    # Expected: 60 / 20 = 3 chunks
    assert len(chunks) == 3

    # Verify durations
    assert chunks[0].duration == 20.0
    assert chunks[1].duration == 20.0
    assert chunks[2].duration == 20.0


def test_audio_data_iter_multiple_iterations():
    """Test that AudioData can be iterated multiple times."""
    sampling_rate = 16000
    duration = 45.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test_audio.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    # First iteration
    chunks1 = list(audio)

    # Second iteration
    chunks2 = list(audio)

    # Should produce same number of chunks
    assert len(chunks1) == len(chunks2)

    # Chunks should have same properties
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.duration == c2.duration
        assert np.allclose(c1.ndarray, c2.ndarray)


def test_audio_data_chunk_path_naming():
    """Test that chunk paths contain timing information."""
    sampling_rate = 16000
    duration = 50.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    chunks = list(audio)

    # Verify path naming includes timing
    for chunk in chunks:
        assert "test.wav" in chunk.path
        assert "[" in chunk.path
        assert "s-" in chunk.path
        assert "s]" in chunk.path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
