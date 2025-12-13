"""Test streaming alignment functionality."""

import random
from pathlib import Path

import numpy as np
import torch

from lattifai import LattifAI
from lattifai.audio2 import AudioData
from lattifai.config import AlignmentConfig


def test_audio_data_chunking():
    """Test AudioData chunking with memory usage comparison."""
    sampling_rate = 16000
    duration = 300.0  # 5 minutes
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32) * 0.1

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        path="test_long_audio.wav",
        streaming_chunk_secs=None,
        overlap_secs=0.0,
    )

    # Test chunking with 30s chunks and 1s overlap
    chunks = list(audio.iter_chunks(30.0, 1.0))

    assert len(chunks) > 0, "No chunks produced"
    assert chunks[0].duration == 30.0, "First chunk should be 30 seconds"

    chunk_memory = chunks[0].ndarray.nbytes / 1024 / 1024
    full_memory = audio.ndarray.nbytes / 1024 / 1024

    # Verify memory savings
    assert chunk_memory < full_memory, "Chunk should use less memory than full audio"
    print(f"✓ Chunking test passed: {len(chunks)} chunks, {chunk_memory:.2f} MB per chunk vs {full_memory:.2f} MB full")


def test_chunking_with_tail_frames():
    """Test that chunking handles various audio lengths correctly with and without overlap."""
    sampling_rate = 16000
    chunk_duration = 30.0

    for overlap_duration in [0.0, 1.0]:
        chunk_samples = int(chunk_duration * sampling_rate)
        overlap_samples = int(overlap_duration * sampling_rate)
        step_samples = chunk_samples - overlap_samples

        # Test a few different tail sizes
        for r in range(1, 4):
            tail_samples = r * 160 + random.randint(0, 160)
            total_samples = 3 * step_samples + tail_samples

            audio = AudioData(
                sampling_rate=sampling_rate,
                ndarray=np.random.randn(1, total_samples).astype(np.float32) * 0.1,
                path=f"test_tail_{r}.wav",
                streaming_chunk_secs=None,
                overlap_secs=0.0,
            )

            chunks = list(audio.iter_chunks(chunk_duration, overlap_duration))
            assert len(chunks) >= 1, "No chunks produced"

            # Verify last chunk covers the tail
            last_chunk_start = (len(chunks) - 1) * step_samples
            last_chunk_end = last_chunk_start + chunks[-1].ndarray.shape[-1]
            assert last_chunk_end >= total_samples, "Last chunk should cover all samples"

    print("✓ Tail frame chunking tests passed")


def test_streaming_alignment_with_real_data():
    """Test streaming alignment with real data if available."""
    data_dir = Path(__file__).parent.parent / "data"
    media_path = data_dir / "SA1.mp4"
    caption_path = data_dir / "SA1.TXT"

    if not media_path.exists():
        print(f"Skipping test: {media_path} not found")
        return

    client = LattifAI(alignment_config=AlignmentConfig(device="cpu"))

    # Test with a few different chunk durations
    for chunk_duration in [3.0, 4.0]:
        result_caption = client.alignment(
            input_media=str(media_path),
            input_caption=str(caption_path),
            streaming_chunk_secs=chunk_duration,
        )
        assert result_caption.alignments, "No alignments returned"

    print(f"✓ Streaming alignment with {media_path.name}: {len(result_caption.alignments)} alignments")


if __name__ == "__main__":
    test_audio_data_chunking()
    test_chunking_with_tail_frames()
    test_streaming_alignment_with_real_data()
