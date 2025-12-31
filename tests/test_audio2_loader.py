"""Test audio2.py AudioLoader functionality."""

from pathlib import Path

import numpy as np
import pytest

from lattifai.audio2 import AudioLoader


@pytest.fixture
def audio_loader():
    """Create an AudioLoader instance for testing."""
    return AudioLoader(device="cpu")


@pytest.fixture
def test_data_dir():
    """Get the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sa1_audio_files(test_data_dir):
    """Get all SA1.* audio files for testing."""
    patterns = ["SA1.mp3", "SA1.m4a", "SA1.opus", "SA1.wav", "SA1.flac", "SA1.ogg", "SA1.aac", "SA1.webm"]
    files = []
    for pattern in patterns:
        file_path = test_data_dir / pattern
        if file_path.exists():
            files.append(file_path)
    return files


def test_audio_value_range_mp3(audio_loader, test_data_dir):
    """Test that loaded SA1.mp3 audio values are in [-1.0, 1.0] range."""
    audio_file = test_data_dir / "SA1.mp3"

    if not audio_file.exists():
        pytest.skip("SA1.mp3 not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 for {audio_file.name}"
    assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 for {audio_file.name}"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"


def test_audio_value_range_m4a(audio_loader, test_data_dir):
    """Test that loaded SA1.m4a audio values are in [-1.0, 1.0] range."""
    audio_file = test_data_dir / "SA1.m4a"

    if not audio_file.exists():
        pytest.skip("SA1.m4a not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 for {audio_file.name}"
    assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 for {audio_file.name}"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"


def test_audio_value_range_opus(audio_loader, test_data_dir):
    """Test that loaded SA1.opus audio values are in [-1.0, 1.0] range."""
    audio_file = test_data_dir / "SA1.opus"

    if not audio_file.exists():
        pytest.skip("SA1.opus not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 for {audio_file.name}"
    assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 for {audio_file.name}"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"


def test_audio_value_range_all_formats(audio_loader, sa1_audio_files):
    """Test that all SA1.* audio formats maintain proper value range."""
    if not sa1_audio_files:
        pytest.skip("No SA1.* audio files found in test data directory")

    tested_count = 0
    for audio_file in sa1_audio_files:
        audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

        # Check value range
        min_val = np.min(audio_data.ndarray)
        max_val = np.max(audio_data.ndarray)

        assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 for {audio_file.name}"
        assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 for {audio_file.name}"

        # Verify basic properties
        assert audio_data.sampling_rate == 16000
        assert audio_data.ndarray.ndim == 2, "Audio should be 2D (channels, samples)"
        assert audio_data.ndarray.shape[0] >= 1, "At least one channel expected"

        tested_count += 1

    assert tested_count > 0, f"Successfully tested {tested_count} SA1.* audio files"


def test_audio_value_range_with_different_sample_rates(audio_loader, test_data_dir):
    """Test that value range is maintained across different target sample rates."""
    audio_file = test_data_dir / "SA1.mp3"

    if not audio_file.exists():
        pytest.skip("SA1.mp3 not found in test data directory")

    sample_rates = [8000, 16000, 22050, 44100]

    for sr in sample_rates:
        audio_data = audio_loader(audio_file, sampling_rate=sr, channel_selector="average")

        min_val = np.min(audio_data.ndarray)
        max_val = np.max(audio_data.ndarray)

        assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 at {sr}Hz for {audio_file.name}"
        assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 at {sr}Hz for {audio_file.name}"
        assert audio_data.sampling_rate == sr


def test_audio_value_range_with_channel_selection(audio_loader, test_data_dir):
    """Test that value range is maintained with different channel selection modes."""
    audio_file = test_data_dir / "SA1.wav"

    if not audio_file.exists():
        pytest.skip("SA1.wav not found in test data directory")

    channel_modes = ["average", 0]

    for channel_selector in channel_modes:
        audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector=channel_selector)

        min_val = np.min(audio_data.ndarray)
        max_val = np.max(audio_data.ndarray)

        assert min_val >= -1.0, (
            f"Audio min value {min_val} is below -1.0 with channel={channel_selector} " f"for {audio_file.name}"
        )
        assert max_val <= 1.0, (
            f"Audio max value {max_val} is above 1.0 with channel={channel_selector} " f"for {audio_file.name}"
        )


def test_audio_data_statistics(audio_loader, test_data_dir):
    """Test basic statistics of loaded audio to ensure it looks reasonable."""
    audio_file = test_data_dir / "SA1.mp3"

    if not audio_file.exists():
        pytest.skip("SA1.mp3 not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range
    assert np.min(audio_data.ndarray) >= -1.0
    assert np.max(audio_data.ndarray) <= 1.0

    # Check that standard deviation is reasonable (not too small, not too large)
    std = np.std(audio_data.ndarray)
    assert 0.0 < std < 1.0, f"Standard deviation {std} seems unusual"

    # Check mean is near zero (for typical audio)
    mean = np.mean(audio_data.ndarray)
    assert abs(mean) < 0.5, f"Mean {mean} seems too far from zero"


def test_audio_value_range_streaming_chunks(audio_loader, test_data_dir):
    """Test that audio chunks from streaming also maintain proper value range."""
    audio_file = test_data_dir / "SA1.wav"

    if not audio_file.exists():
        pytest.skip("SA1.wav not found in test data directory")

    # Load with streaming enabled
    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average", streaming_chunk_secs=30.0)

    # Check full audio first
    assert np.min(audio_data.ndarray) >= -1.0
    assert np.max(audio_data.ndarray) <= 1.0

    # Check chunks if streaming mode is enabled
    if audio_data.streaming_mode:
        for idx, chunk in enumerate(audio_data):
            min_val = np.min(chunk.ndarray)
            max_val = np.max(chunk.ndarray)

            assert min_val >= -1.0, f"Chunk {idx} min value {min_val} is below -1.0 for {audio_file.name}"
            assert max_val <= 1.0, f"Chunk {idx} max value {max_val} is above 1.0 for {audio_file.name}"

            # Only test first few chunks to avoid excessive test time
            if idx >= 2:
                break


def test_audio_dtype(audio_loader, test_data_dir):
    """Test that loaded audio has correct dtype (float32)."""
    audio_file = test_data_dir / "SA1.wav"

    if not audio_file.exists():
        pytest.skip("SA1.wav not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check dtype is float32
    assert audio_data.ndarray.dtype == np.float32, f"Expected float32, got {audio_data.ndarray.dtype}"

    # Verify value range for float32
    assert np.min(audio_data.ndarray) >= -1.0
    assert np.max(audio_data.ndarray) <= 1.0


def test_audio_value_range_large_opus_file(audio_loader, test_data_dir):
    """Test that loaded Sh-YrLYC7p8.opus audio values are in [-1.0, 1.0] range.

    Note: This test allows a small tolerance (0.02) for edge cases where audio
    processing may cause slight overshoot beyond [-1.0, 1.0] range.
    """
    audio_file = test_data_dir / "Sh-YrLYC7p8.opus"

    if not audio_file.exists():
        pytest.skip("Sh-YrLYC7p8.opus not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range with tolerance
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    tolerance = 0.02  # Allow 2% overshoot for edge cases

    assert min_val >= -1.0 - tolerance, (
        f"Audio min value {min_val} is below -1.0 - tolerance ({-1.0 - tolerance}) " f"for {audio_file.name}"
    )
    assert max_val <= 1.0 + tolerance, (
        f"Audio max value {max_val} is above 1.0 + tolerance ({1.0 + tolerance}) " f"for {audio_file.name}"
    )

    # Count how many samples are outside strict [-1.0, 1.0] range
    outside_range = np.sum((audio_data.ndarray < -1.0) | (audio_data.ndarray > 1.0))
    total_samples = audio_data.ndarray.size
    outside_percentage = (outside_range / total_samples) * 100

    # Report if any samples are outside range
    if outside_range > 0:
        print(f"\nWarning: {outside_range} samples ({outside_percentage:.6f}%) are outside [-1.0, 1.0] range")
        print(f"Min: {min_val}, Max: {max_val}")

    # Ensure less than 0.001% of samples are outside range
    assert outside_percentage < 0.001, f"Too many samples ({outside_percentage:.6f}%) outside [-1.0, 1.0] range"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"

    # Check dtype
    assert audio_data.ndarray.dtype == np.float32

    # Check basic properties
    assert audio_data.sampling_rate == 16000
    assert audio_data.ndarray.ndim == 2
    assert audio_data.ndarray.shape[0] >= 1

    # Verify reasonable statistics for this large file
    std = np.std(audio_data.ndarray)
    assert 0.0 < std < 1.0, f"Standard deviation {std} seems unusual"

    mean = np.mean(audio_data.ndarray)
    assert abs(mean) < 0.5, f"Mean {mean} seems too far from zero"


def test_audio_value_range_large_mp3_file(audio_loader, test_data_dir):
    """Test that loaded Sh-YrLYC7p8.mp3 audio values are in [-1.0, 1.0] range.

    This tests a large MP3 file to ensure proper value range handling.
    """
    audio_file = test_data_dir / "Sh-YrLYC7p8.mp3"

    if not audio_file.exists():
        pytest.skip("Sh-YrLYC7p8.mp3 not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    assert min_val >= -1.0, f"Audio min value {min_val} is below -1.0 for {audio_file.name}"
    assert max_val <= 1.0, f"Audio max value {max_val} is above 1.0 for {audio_file.name}"

    # Count how many samples are outside strict [-1.0, 1.0] range (should be 0)
    outside_range = np.sum((audio_data.ndarray < -1.0) | (audio_data.ndarray > 1.0))
    total_samples = audio_data.ndarray.size
    outside_percentage = (outside_range / total_samples) * 100

    # Report if any samples are outside range
    if outside_range > 0:
        print(f"\nWarning: {outside_range} samples ({outside_percentage:.6f}%) are outside [-1.0, 1.0] range")
        print(f"Min: {min_val}, Max: {max_val}")

    assert outside_range == 0, f"{outside_range} samples ({outside_percentage:.6f}%) are outside [-1.0, 1.0] range"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"

    # Check dtype
    assert audio_data.ndarray.dtype == np.float32

    # Check basic properties
    assert audio_data.sampling_rate == 16000
    assert audio_data.ndarray.ndim == 2
    assert audio_data.ndarray.shape[0] >= 1

    # Verify reasonable statistics for this large file
    std = np.std(audio_data.ndarray)
    assert 0.0 < std < 1.0, f"Standard deviation {std} seems unusual"

    mean = np.mean(audio_data.ndarray)
    assert abs(mean) < 0.5, f"Mean {mean} seems too far from zero"


def test_audio_value_range_very_large_opus_file(audio_loader, test_data_dir):
    """Test that loaded AG3fwNC5ltw.opus audio values are mostly in [-1.0, 1.0] range.

    Note: This is a very large file (~1GB, 18+ hours). The test allows tolerance
    for samples outside [-1.0, 1.0] range but requires less than 0.1% overshoot.
    """
    audio_file = test_data_dir / "AG3fwNC5ltw.opus"

    if not audio_file.exists():
        pytest.skip("AG3fwNC5ltw.opus not found in test data directory")

    audio_data = audio_loader(audio_file, sampling_rate=16000, channel_selector="average")

    # Check value range with tolerance
    min_val = np.min(audio_data.ndarray)
    max_val = np.max(audio_data.ndarray)

    tolerance = 0.15  # Allow 15% overshoot for very large files

    assert min_val >= -1.0 - tolerance, (
        f"Audio min value {min_val} is below -1.0 - tolerance ({-1.0 - tolerance}) " f"for {audio_file.name}"
    )
    assert max_val <= 1.0 + tolerance, (
        f"Audio max value {max_val} is above 1.0 + tolerance ({1.0 + tolerance}) " f"for {audio_file.name}"
    )

    # Count how many samples are outside strict [-1.0, 1.0] range
    outside_range = np.sum((audio_data.ndarray < -1.0) | (audio_data.ndarray > 1.0))
    total_samples = audio_data.ndarray.size
    outside_percentage = (outside_range / total_samples) * 100

    # Always report statistics for this large file
    print(f"\nAG3fwNC5ltw.opus statistics:")
    print(f"  Duration: {audio_data.duration:.2f}s ({audio_data.duration / 3600:.2f} hours)")
    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
    print(f"  Samples outside [-1.0, 1.0]: {outside_range} ({outside_percentage:.4f}%)")
    print(f"  Mean: {np.mean(audio_data.ndarray):.8f}, Std: {np.std(audio_data.ndarray):.6f}")

    # Ensure less than 0.1% of samples are outside range
    assert outside_percentage < 0.1, f"Too many samples ({outside_percentage:.4f}%) outside [-1.0, 1.0] range"

    # Check that audio is not all zeros
    assert not np.allclose(audio_data.ndarray, 0.0), f"Audio appears to be silent for {audio_file.name}"

    # Check dtype
    assert audio_data.ndarray.dtype == np.float32

    # Check basic properties
    assert audio_data.sampling_rate == 16000
    assert audio_data.ndarray.ndim == 2
    assert audio_data.ndarray.shape[0] >= 1

    # Verify reasonable statistics for this very large file
    std = np.std(audio_data.ndarray)
    assert 0.0 < std < 1.0, f"Standard deviation {std} seems unusual"

    mean = np.mean(audio_data.ndarray)
    assert abs(mean) < 0.5, f"Mean {mean} seems too far from zero"
