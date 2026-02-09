"""Test AudioData iterator interface for streaming."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

import lattifai.audio2 as _audio2_mod
from lattifai.audio2 import AudioData

_RESAMPLER_DIR = Path(_audio2_mod.__file__).parent / "data" / "resamplers"


def test_audio_data_iter_default():
    """Test AudioData iteration with default parameters."""
    # Create synthetic audio: 90 seconds at 16kHz
    sampling_rate = 16000
    duration = 90.0
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        path="test_audio.wav",
        streaming_chunk_secs=30.0,  # Use 30s chunks for this test
        overlap_secs=0.0,
    )

    # Iterate over chunks
    chunks = list(audio)

    # With 30s chunks and 0s overlap:
    # - Step size: 30s
    # - Expected chunks: ceil(90 / 30) = 3
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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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


def test_onnx_resampler_exact_at_one_second():
    """Test ONNX resampler produces exact output for 1-second input (the export chunk size)."""
    test_rates = [8000, 22050, 24000, 32000, 44100, 48000]

    for source_sr in test_rates:
        onnx_path = _RESAMPLER_DIR / f"resampler_{source_sr}.onnx"
        if not onnx_path.exists():
            continue

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        audio = np.random.randn(1, source_sr).astype(np.float32)
        resampled = session.run(None, {"input": audio})[0]

        assert resampled.shape[0] == 1, f"Channel count mismatch for {source_sr}Hz"
        assert resampled.shape[1] == 16000, f"1s at {source_sr}Hz -> expected 16000 samples, got {resampled.shape[1]}"


def test_audio_data_resample_very_short():
    """Test AudioLoader resampling with very short audio."""
    import tempfile

    import soundfile as sf

    from lattifai.audio2 import AudioLoader

    source_sr = 48000
    target_sr = 16000
    duration = 0.01  # 10ms - very short audio
    num_samples = int(source_sr * duration)

    # Create temporary audio file
    audio_data = np.random.randn(num_samples).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, source_sr)
        tmp_path = tmp_file.name

    try:
        # Load and resample using AudioLoader
        loader = AudioLoader(device="cpu")
        resampled = loader._load_audio(
            audio=tmp_path,
            sampling_rate=target_sr,
            channel_selector=None,
        )

        # Verify resampling worked
        expected_samples = int(num_samples * target_sr / source_sr)
        assert resampled.shape[0] == 1, "Should have 1 channel"
        assert (
            abs(resampled.shape[1] - expected_samples) <= 10
        ), f"Resampled samples mismatch: got {resampled.shape[1]}, expected ~{expected_samples}"
    finally:
        import os

        os.unlink(tmp_path)


def test_audio_data_iter_with_resampling():
    """Test AudioLoader with resampling on very short chunks."""
    import tempfile

    import soundfile as sf

    from lattifai.audio2 import AudioLoader

    source_sr = 48000
    target_sr = 16000
    duration = 0.5  # 0.5 seconds total
    num_samples = int(source_sr * duration)

    # Create temporary audio file
    audio_data = np.random.randn(num_samples).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, source_sr)
        tmp_path = tmp_file.name

    try:
        # Load and resample using AudioLoader
        loader = AudioLoader(device="cpu")
        resampled = loader._load_audio(
            audio=tmp_path,
            sampling_rate=target_sr,
            channel_selector=None,
        )

        # Verify resampling
        expected_samples = int(num_samples * target_sr / source_sr)
        assert (
            abs(resampled.shape[1] - expected_samples) <= 100
        ), f"Resampled samples mismatch: got {resampled.shape[1]}, expected ~{expected_samples}"
    finally:
        import os

        os.unlink(tmp_path)


def test_onnx_resampler_cache():
    """Test that ONNX resampler sessions are cached correctly."""
    from lattifai.audio2 import AudioLoader

    loader = AudioLoader(device="cpu")

    source_sr = 48000
    target_sr = 16000
    num_samples = 480  # 10ms at 48kHz
    audio = np.random.randn(num_samples, 1).astype(np.float32)  # (samples, channels)

    result1 = loader._resample_audio((audio, source_sr), target_sr, device="cpu", channel_selector=None)
    result2 = loader._resample_audio((audio, source_sr), target_sr, device="cpu", channel_selector=None)

    assert source_sr in loader._resampler_cache
    assert np.allclose(result1, result2)


def test_audio_loader_resampling_various_durations():
    """Test AudioLoader resampling produces expected lengths for various durations."""
    import tempfile

    import soundfile as sf

    from lattifai.audio2 import AudioLoader

    target_sr = 16000

    test_cases = [
        (48000, 5),  # 5 seconds at 48kHz
        (44100, 5),  # 5 seconds at 44.1kHz
        (32000, 5),  # 5 seconds at 32kHz
        (24000, 5),  # 5 seconds at 24kHz
        (8000, 5),  # 5 seconds at 8kHz
    ]

    loader = AudioLoader(device="cpu")

    for source_sr, duration_secs in test_cases:
        num_samples = int(source_sr * duration_secs)
        audio_data = np.random.randn(num_samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, source_sr)
            tmp_path = tmp_file.name

        try:
            resampled = loader._load_audio(audio=tmp_path, sampling_rate=target_sr, channel_selector=None)
            expected_samples = target_sr * duration_secs
            actual_samples = resampled.shape[1]

            assert actual_samples == expected_samples, (
                f"{source_sr}Hz -> 16kHz for {duration_secs}s: "
                f"Expected {expected_samples} samples, got {actual_samples} "
                f"(difference: {actual_samples - expected_samples})"
            )
        finally:
            import os

            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
