"""Test loading long audio files with various formats."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from lattifai.audio2 import AudioLoader

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(
    not (TEST_DATA_DIR / "Sh-YrLYC7p8.opus").exists(), reason="Test file Sh-YrLYC7p8.opus not found - local test only"
)
def test_load_long_opus_file_1():
    """Test loading long OPUS audio file: Sh-YrLYC7p8.opus with all validations."""
    audio_path = TEST_DATA_DIR / "Sh-YrLYC7p8.opus"

    # Test with CPU
    loader = AudioLoader(device="cpu")

    start_time = time.time()
    audio = loader(
        audio=audio_path,
        sampling_rate=16000,
        channel_selector="average",
    )
    load_time = time.time() - start_time

    # Basic validation
    assert audio.sampling_rate == 16000
    assert audio.ndarray.shape[0] == 1, "Should have 1 channel (averaged)"
    assert audio.ndarray.shape[1] > 0, "Should have samples"
    assert audio.ndarray.dtype == np.float32

    # Verify duration is reasonable
    duration = audio.duration
    assert duration > 0, f"Duration should be positive, got {duration}"

    # Duration: 01:59:59.39, start: 0.007500, bitrate: 137 kb/s
    target_duration = 1 * 3600 + 59 * 60 + 59.39
    assert abs(duration - target_duration) < 1, f"Duration mismatch: expected {target_duration}, got {duration}"

    # Memory efficiency validation
    expected_samples = int(duration * 16000)
    actual_samples = audio.ndarray.shape[1]
    assert (
        abs(actual_samples - expected_samples) <= 100
    ), f"Sample count mismatch: expected ~{expected_samples}, got {actual_samples}"

    memory_mb = audio.ndarray.nbytes / (1024 * 1024)

    # Print summary
    print(f"✓ Loaded {audio_path.name}:")
    print(f"  Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    print(f"  Samples: {audio.ndarray.shape[1]:,}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Load time: {load_time:.2f}s ({duration/load_time:.2f}x realtime)")


@pytest.mark.skipif(
    not (TEST_DATA_DIR / "AG3fwNC5ltw.opus").exists(), reason="Test file AG3fwNC5ltw.opus not found - local test only"
)
def test_load_long_opus_file_2():
    """Test loading long OPUS audio file: AG3fwNC5ltw.opus with all validations."""
    audio_path = TEST_DATA_DIR / "AG3fwNC5ltw.opus"

    # Test with CPU
    loader = AudioLoader(device="cpu")

    start_time = time.time()
    audio = loader(
        audio=audio_path,
        sampling_rate=16000,
        channel_selector="average",
    )
    load_time = time.time() - start_time

    # Basic validation
    assert audio.sampling_rate == 16000
    assert audio.ndarray.shape[0] == 1, "Should have 1 channel (averaged)"
    assert audio.ndarray.shape[1] > 0, "Should have samples"
    assert audio.ndarray.dtype == np.float32

    # Verify duration is reasonable
    duration = audio.duration
    assert duration > 0, f"Duration should be positive, got {duration}"

    # Duration: 18:37:27.46, start: 0.023021, bitrate: 128 kb/s
    target_duration = 18 * 3600 + 37 * 60 + 27.46
    assert abs(duration - target_duration) < 1, f"Duration mismatch: expected {target_duration}, got {duration}"

    # Memory efficiency validation
    expected_samples = int(duration * 16000)
    actual_samples = audio.ndarray.shape[1]
    assert (
        abs(actual_samples - expected_samples) <= 100
    ), f"Sample count mismatch: expected ~{expected_samples}, got {actual_samples}"

    memory_mb = audio.ndarray.nbytes / (1024 * 1024)

    # Print summary
    print(f"✓ Loaded {audio_path.name}:")
    print(f"  Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    print(f"  Samples: {audio.ndarray.shape[1]:,}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Load time: {load_time:.2f}s ({duration/load_time:.2f}x realtime)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
