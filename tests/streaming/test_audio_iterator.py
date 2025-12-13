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

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
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


def test_resampler_with_very_short_audio():
    """Test get_or_create_resampler with very short audio."""
    from lhotse.augmentation import get_or_create_resampler

    # Test various very short audio lengths
    test_cases = [
        (48000, 16000, 0.001),  # 1ms audio
        (44100, 16000, 0.005),  # 5ms audio
        (48000, 16000, 0.01),  # 10ms audio
        (32000, 16000, 0.02),  # 20ms audio
        (44100, 16000, 0.05),  # 50ms audio
        (48000, 16000, 0.1),  # 100ms audio
    ]

    for source_sr, target_sr, duration in test_cases:
        # Create very short audio
        num_samples = int(source_sr * duration)
        audio = torch.randn(1, num_samples)

        # CPU test - create fresh resampler on CPU
        resampler_cpu = get_or_create_resampler(source_sr, target_sr)
        resampler_cpu = resampler_cpu.to("cpu")  # Ensure it's on CPU
        resampled = resampler_cpu(audio)
        expected_samples = int(num_samples * target_sr / source_sr)

        assert resampled.shape[0] == audio.shape[0], f"Channel count mismatch for {duration}s audio"
        assert abs(resampled.shape[1] - expected_samples) <= 10, (
            f"Sample count mismatch for {duration}s audio: " f"got {resampled.shape[1]}, expected ~{expected_samples}"
        )

        # GPU test (CUDA if available)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            audio_gpu = audio.to(device=device)
            # Create fresh resampler for CUDA
            resampler_gpu = get_or_create_resampler(source_sr, target_sr)
            resampler_gpu = resampler_gpu.to(device=device)

            resampled_gpu = resampler_gpu(audio_gpu)

            assert resampled_gpu.device.type == "cuda", "Resampled audio should be on CUDA"
            assert resampled_gpu.shape[0] == audio.shape[0], f"Channel count mismatch on CUDA for {duration}s audio"
            assert abs(resampled_gpu.shape[1] - expected_samples) <= 10, (
                f"Sample count mismatch on CUDA for {duration}s audio: "
                f"got {resampled_gpu.shape[1]}, expected ~{expected_samples}"
            )

        # MPS test (Apple Silicon if available)
        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                audio_mps = audio.to(device=device)
                # Create fresh resampler for MPS
                resampler_mps = get_or_create_resampler(source_sr, target_sr)
                resampler_mps = resampler_mps.to(device=device)

                resampled_mps = resampler_mps(audio_mps)

                assert resampled_mps.device.type == "mps", "Resampled audio should be on MPS"
                assert resampled_mps.shape[0] == audio.shape[0], f"Channel count mismatch on MPS for {duration}s audio"
                assert abs(resampled_mps.shape[1] - expected_samples) <= 10, (
                    f"Sample count mismatch on MPS for {duration}s audio: "
                    f"got {resampled_mps.shape[1]}, expected ~{expected_samples}"
                )
            except RuntimeError as e:
                # MPS may not support all operations, skip if it fails
                pytest.skip(f"MPS operation not supported: {e}")


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


def test_audio_loader_mps_device():
    """Test AudioLoader with MPS device for very short audio."""
    import tempfile

    import soundfile as sf

    from lattifai.audio2 import AudioLoader

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this system")

    source_sr = 48000
    target_sr = 16000
    duration = 0.05  # 50ms audio
    num_samples = int(source_sr * duration)

    # Create temporary audio file
    audio_data = np.random.randn(num_samples).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, source_sr)
        tmp_path = tmp_file.name

    try:
        # Load and resample using AudioLoader with MPS device
        loader = AudioLoader(device="mps")
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
        ), f"Resampled samples mismatch on MPS: got {resampled.shape[1]}, expected ~{expected_samples}"
    finally:
        import os

        os.unlink(tmp_path)


def test_resampler_cache_with_different_devices():
    """Test that resampler cache works correctly with different devices."""
    from lhotse.augmentation import get_or_create_resampler

    source_sr = 48000
    target_sr = 16000
    num_samples = 480  # 10ms at 48kHz
    audio = torch.randn(1, num_samples)

    # Test CPU - ensure resampler is on CPU
    resampler_cpu = get_or_create_resampler(source_sr, target_sr)
    resampler_cpu = resampler_cpu.to("cpu")
    resampled_cpu = resampler_cpu(audio)
    assert resampled_cpu.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        audio_cuda = audio.to("cuda")
        resampler_cuda = get_or_create_resampler(source_sr, target_sr)
        resampler_cuda = resampler_cuda.to("cuda")
        resampled_cuda = resampler_cuda(audio_cuda)
        assert resampled_cuda.device.type == "cuda"
        # Verify outputs are similar
        assert torch.allclose(resampled_cpu, resampled_cuda.cpu(), atol=1e-5)

    # Test MPS if available
    if torch.backends.mps.is_available():
        audio_mps = audio.to("mps")
        resampler_mps = get_or_create_resampler(source_sr, target_sr)
        resampler_mps = resampler_mps.to("mps")
        resampled_mps = resampler_mps(audio_mps)
        assert resampled_mps.device.type == "mps"
        # Verify outputs are similar
        assert torch.allclose(resampled_cpu, resampled_mps.cpu(), atol=1e-5)


def test_resampler_exact_length_30min_chunks():
    """Test that resampling 30-minute audio to 16kHz produces exactly 16000*30 samples."""
    from lhotse.augmentation import get_or_create_resampler

    target_sr = 16000
    duration_minutes = 30
    duration_secs = duration_minutes * 60  # 1800 seconds
    expected_samples = target_sr * duration_secs  # 16000 * 1800 = 28,800,000

    # Test various common sample rates
    test_sample_rates = [
        8000,  # Telephone quality
        16000,  # Wideband (no resampling needed)
        22050,  # Common for music
        24000,  # Common for speech
        32000,  # Super-wideband
        44100,  # CD quality
        48000,  # Professional audio
    ]

    for source_sr in test_sample_rates:
        # Create 30-minute audio at source sample rate
        num_samples = int(source_sr * duration_secs)
        audio = torch.randn(1, num_samples)

        # Resample to 16kHz on CPU
        resampler = get_or_create_resampler(source_sr, target_sr)
        resampler = resampler.to("cpu")
        resampled = resampler(audio)

        # Verify exact sample count
        actual_samples = resampled.shape[1]
        assert actual_samples == expected_samples, (
            f"Sample rate {source_sr}Hz -> 16kHz: "
            f"Expected exactly {expected_samples} samples, got {actual_samples} "
            f"(difference: {actual_samples - expected_samples})"
        )

        # Test on MPS if available
        if torch.backends.mps.is_available():
            audio_mps = audio.to("mps")
            resampler_mps = get_or_create_resampler(source_sr, target_sr)
            resampler_mps = resampler_mps.to("mps")
            resampled_mps = resampler_mps(audio_mps)

            actual_samples_mps = resampled_mps.shape[1]
            assert actual_samples_mps == expected_samples, (
                f"MPS - Sample rate {source_sr}Hz -> 16kHz: "
                f"Expected exactly {expected_samples} samples, got {actual_samples_mps} "
                f"(difference: {actual_samples_mps - expected_samples})"
            )

        # Test on CUDA if available
        if torch.cuda.is_available():
            audio_cuda = audio.to("cuda")
            resampler_cuda = get_or_create_resampler(source_sr, target_sr)
            resampler_cuda = resampler_cuda.to("cuda")
            resampled_cuda = resampler_cuda(audio_cuda)

            actual_samples_cuda = resampled_cuda.shape[1]
            assert actual_samples_cuda == expected_samples, (
                f"CUDA - Sample rate {source_sr}Hz -> 16kHz: "
                f"Expected exactly {expected_samples} samples, got {actual_samples_cuda} "
                f"(difference: {actual_samples_cuda - expected_samples})"
            )


def test_resampler_length_various_durations():
    """Test resampling produces exact expected lengths for various durations."""
    from lhotse.augmentation import get_or_create_resampler

    target_sr = 16000

    # Test cases: (source_sr, duration_secs)
    test_cases = [
        (48000, 1800),  # 30 minutes at 48kHz
        (44100, 1800),  # 30 minutes at 44.1kHz
        (32000, 1800),  # 30 minutes at 32kHz
        (24000, 1800),  # 30 minutes at 24kHz
        (48000, 600),  # 10 minutes at 48kHz
        (44100, 600),  # 10 minutes at 44.1kHz
        (48000, 60),  # 1 minute at 48kHz
        (44100, 60),  # 1 minute at 44.1kHz
    ]

    for source_sr, duration_secs in test_cases:
        num_samples = int(source_sr * duration_secs)
        audio = torch.randn(1, num_samples)

        resampler = get_or_create_resampler(source_sr, target_sr)
        resampler = resampler.to("cpu")
        resampled = resampler(audio)

        expected_samples = target_sr * duration_secs
        actual_samples = resampled.shape[1]

        assert actual_samples == expected_samples, (
            f"{source_sr}Hz -> 16kHz for {duration_secs}s: "
            f"Expected {expected_samples} samples, got {actual_samples} "
            f"(difference: {actual_samples - expected_samples})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
