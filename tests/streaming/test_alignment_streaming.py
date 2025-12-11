"""Test AlignmentConfig streaming parameters."""

import pytest

from lattifai.config import AlignmentConfig


def test_alignment_config_streaming_disabled_by_default():
    """Test that streaming is disabled by default."""
    config = AlignmentConfig()
    assert config.enable_streaming is False
    assert config.streaming_chunk_duration == 30.0


def test_alignment_config_streaming_enabled():
    """Test enabling streaming mode."""
    config = AlignmentConfig(enable_streaming=True)
    assert config.enable_streaming is True


def test_alignment_config_streaming_custom_parameters():
    """Test custom streaming parameters."""
    config = AlignmentConfig(
        enable_streaming=True,
        streaming_chunk_duration=60.0,
    )
    assert config.streaming_chunk_duration == 60.0


def test_alignment_config_streaming_chunk_duration_too_small():
    """Test validation for chunk duration too small."""
    with pytest.raises(ValueError, match="streaming_chunk_duration must be between 10 and 120 seconds"):
        AlignmentConfig(
            enable_streaming=True,
            streaming_chunk_duration=5.0,
        )


def test_alignment_config_streaming_chunk_duration_too_large():
    """Test validation for chunk duration too large."""
    with pytest.raises(ValueError, match="streaming_chunk_duration must be between 10 and 120 seconds"):
        AlignmentConfig(
            enable_streaming=True,
            streaming_chunk_duration=150.0,
        )


def test_alignment_config_streaming_valid_edge_cases():
    """Test valid edge cases for streaming parameters."""
    # Minimum chunk
    config1 = AlignmentConfig(
        enable_streaming=True,
        streaming_chunk_duration=10.0,  # Minimum allowed
    )
    assert config1.streaming_chunk_duration == 10.0

    # Maximum chunk
    config2 = AlignmentConfig(
        enable_streaming=True,
        streaming_chunk_duration=120.0,  # Maximum allowed
    )
    assert config2.streaming_chunk_duration == 120.0


def test_alignment_config_streaming_validation_only_when_enabled():
    """Test that streaming validation only applies when streaming is enabled."""
    # Should not raise error when streaming is disabled
    config = AlignmentConfig(
        enable_streaming=False,
        streaming_chunk_duration=200.0,  # Would be invalid if streaming was enabled
    )
    assert config.enable_streaming is False


def test_alignment_config_streaming_with_other_parameters():
    """Test streaming config combined with other alignment parameters."""
    config = AlignmentConfig(
        model_name="LattifAI/Lattice-1",
        device="cpu",
        batch_size=2,
        strategy="entire",
        enable_streaming=True,
        streaming_chunk_duration=45.0,
    )
    assert config.enable_streaming is True
    assert config.streaming_chunk_duration == 45.0
    assert config.model_name == "LattifAI/Lattice-1"
    assert config.device == "cpu"
    assert config.batch_size == 2
    assert config.strategy == "entire"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
