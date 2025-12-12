"""Test MediaConfig streaming parameters."""

import pytest

from lattifai.config import MediaConfig


def test_media_config_streaming_disabled_by_default():
    """Test that streaming is enabled by default with 600.0 seconds."""
    config = MediaConfig()
    assert config.streaming_chunk_secs == 600.0


def test_media_config_streaming_enabled():
    """Test enabling streaming mode."""
    config = MediaConfig(streaming_chunk_secs=30.0)
    assert config.streaming_chunk_secs == 30.0


def test_media_config_streaming_custom_parameters():
    """Test custom streaming parameters."""
    config = MediaConfig(streaming_chunk_secs=60.0)
    assert config.streaming_chunk_secs == 60.0


def test_media_config_streaming_chunk_secs_too_small():
    """Test validation for chunk duration too small."""
    with pytest.raises(ValueError, match="streaming_chunk_secs must be between 1 and 1800 seconds"):
        MediaConfig(streaming_chunk_secs=0.5)


def test_media_config_streaming_chunk_secs_too_large():
    """Test validation for chunk duration too large."""
    with pytest.raises(ValueError, match="streaming_chunk_secs must be between 1 and 1800 seconds"):
        MediaConfig(streaming_chunk_secs=2000.0)


def test_media_config_streaming_valid_edge_cases():
    """Test valid edge cases for streaming parameters."""
    # Minimum chunk
    config1 = MediaConfig(streaming_chunk_secs=1.0)  # Minimum allowed
    assert config1.streaming_chunk_secs == 1.0

    # Maximum chunk
    config2 = MediaConfig(streaming_chunk_secs=1800.0)  # Maximum allowed
    assert config2.streaming_chunk_secs == 1800.0


def test_media_config_streaming_disabled():
    """Test that streaming can be disabled explicitly."""
    config = MediaConfig(streaming_chunk_secs=None)
    assert config.streaming_chunk_secs is None


def test_media_config_streaming_with_other_parameters():
    """Test streaming config combined with other media parameters."""
    config = MediaConfig(
        media_format="wav",
        sample_rate=16000,
        streaming_chunk_secs=45.0,
    )
    assert config.streaming_chunk_secs == 45.0
    assert config.sample_rate == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
