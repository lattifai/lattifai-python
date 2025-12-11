"""Test streaming alignment functionality."""

import random
from pathlib import Path

import numpy as np
import torch

from lattifai import LattifAI
from lattifai.audio2 import AudioData
from lattifai.config import AlignmentConfig


def test_streaming_alignment():
    """Test streaming alignment with synthetic audio."""
    print("=== Testing Streaming Alignment ===\n")

    # Create synthetic 5-minute audio
    sampling_rate = 16000
    duration = 300.0  # 5 minutes
    num_samples = int(duration * sampling_rate)

    ndarray = np.random.randn(1, num_samples).astype(np.float32) * 0.1
    tensor = torch.from_numpy(ndarray)

    audio = AudioData(
        sampling_rate=sampling_rate,
        ndarray=ndarray,
        tensor=tensor,
        device="cpu",
        path="test_long_audio.wav",
    )

    print(f"Test audio: {audio.duration/60:.1f} minutes ({audio.duration:.1f}s)")
    print(f"Memory size: {audio.ndarray.nbytes / 1024 / 1024:.2f} MB\n")

    # Test with streaming disabled (baseline)
    print("--- Test 1: Streaming Disabled (Baseline) ---")
    config_no_streaming = AlignmentConfig(
        enable_streaming=False,
        device="cpu",
    )

    client_no_streaming = LattifAI(alignment_config=config_no_streaming)
    print(f"Streaming: {client_no_streaming.aligner.config.enable_streaming}")
    print("Note: For full test, provide actual caption file\n")

    # Test with streaming enabled
    print("--- Test 2: Streaming Enabled ---")
    config_streaming = AlignmentConfig(
        enable_streaming=True,
        streaming_chunk_duration=30.0,
        device="cpu",
    )

    client_streaming = LattifAI(alignment_config=config_streaming)
    print(f"Streaming: {client_streaming.aligner.config.enable_streaming}")
    print(f"Chunk duration: {client_streaming.aligner.config.streaming_chunk_duration}s")
    print()

    # Test AudioData iteration
    print("--- Test 3: AudioData Iteration ---")
    chunk_count = 0
    total_chunk_memory = 0

    for i, chunk in enumerate(audio.iter_chunks(30.0, 1.0)):
        chunk_count += 1
        chunk_memory = chunk.ndarray.nbytes / 1024 / 1024
        total_chunk_memory += chunk_memory

        if i < 3:  # Show first 3 chunks
            print(f"Chunk {i+1}:")
            print(f"  Duration: {chunk.duration:.2f}s")
            print(f"  Memory: {chunk_memory:.2f} MB")

    print(f"\nTotal chunks: {chunk_count}")
    print(f"Peak memory per chunk: ~{chunk_memory:.2f} MB")
    print(f"vs Full audio: {audio.ndarray.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory savings: {(1 - chunk_memory / (audio.ndarray.nbytes / 1024 / 1024)) * 100:.1f}%")
    print()

    # Test with longer audio
    print("--- Test 4: Simulated 20-Hour Audio ---")
    duration_20h = 20 * 60 * 60  # 20 hours in seconds
    full_size_20h = duration_20h * sampling_rate * 4 / 1024 / 1024  # 4 bytes per float32
    chunk_30s_size = 30 * sampling_rate * 4 / 1024 / 1024

    print(f"20-hour audio file:")
    print(f"  Full size in memory: {full_size_20h:.2f} MB (~{full_size_20h/1024:.2f} GB)")
    print(f"  With streaming (30s chunks): {chunk_30s_size:.2f} MB peak")
    print(f"  Memory reduction: {(1 - chunk_30s_size / full_size_20h) * 100:.2f}%")
    print()

    print("✓ Streaming alignment infrastructure ready!")
    print("\nTo test with real audio:")
    print("  alignments = client_streaming.alignment(")
    print("      input_media='long_audio.wav',")
    print("      input_caption='transcript.txt',")
    print("  )")


def test_streaming_small_tail_frames():
    """Test that chunking handles various audio lengths correctly with and without overlap.

    This test verifies that iter_chunks correctly splits audio into chunks
    and that all chunks are produced without errors, even for small tail chunks.
    Tests both with overlap (overlap_duration > 0) and without overlap (overlap_duration = 0).
    """
    sampling_rate = 16000
    chunk_duration = 30.0

    # Test both with and without overlap
    for overlap_duration in [0.0, 1.0]:
        chunk_samples = int(chunk_duration * sampling_rate)
        overlap_samples = int(overlap_duration * sampling_rate)
        step_samples = chunk_samples - overlap_samples

        k = 3  # number of step intervals

        for r in range(1, 10):
            # Create audio with k step intervals plus a small tail
            tail_samples = r * 160 + random.randint(0, 160)
            total_samples = k * step_samples + tail_samples

            ndarray = np.random.randn(1, total_samples).astype(np.float32) * 0.1
            tensor = torch.from_numpy(ndarray)

            audio = AudioData(
                sampling_rate=sampling_rate,
                ndarray=ndarray,
                tensor=tensor,
                device="cpu",
                path=f"test_tail_{r}.wav",
            )

            chunks = list(audio.iter_chunks(chunk_duration, overlap_duration))
            assert len(chunks) >= 1, "No chunks produced"

            # Verify chunk generation logic
            # iter_chunks moves by step_samples and yields chunks of up to chunk_samples
            # Last chunk may be shorter if we reach the end
            for i, chunk in enumerate(chunks):
                chunk_start = i * step_samples
                chunk_end = min(chunk_start + chunk_samples, total_samples)
                expected_len = chunk_end - chunk_start
                actual_len = chunk.ndarray.shape[-1]

                assert actual_len == expected_len, (
                    f"Chunk {i} has size {actual_len}, expected {expected_len} "
                    f"(overlap={overlap_duration}, chunk_start={chunk_start}, chunk_end={chunk_end}, total={total_samples})"
                )

            # Verify coverage: last chunk should reach or exceed total_samples
            last_chunk_start = (len(chunks) - 1) * step_samples
            last_chunk_end = last_chunk_start + chunks[-1].ndarray.shape[-1]
            assert (
                last_chunk_end >= total_samples
            ), f"Last chunk ends at {last_chunk_end}, but audio has {total_samples} samples"

            # Verify no excessive chunks: if we have n chunks, the (n-1)th chunk start should be < total_samples
            if len(chunks) > 1:
                second_last_start = (len(chunks) - 2) * step_samples
                assert (
                    second_last_start < total_samples
                ), f"Second-to-last chunk starts at {second_last_start}, but audio only has {total_samples} samples"

    print("✓ Small-tail chunking tests passed for various tail sizes with and without overlap")


def test_simulated_client_alignment_small_tail():
    """Test streaming alignment with real data (SA1)."""

    # Build a client with streaming enabled
    config_streaming = AlignmentConfig(
        enable_streaming=True,
        streaming_chunk_duration=30.0,
        device="cpu",
    )

    client = LattifAI(alignment_config=config_streaming)

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    media_path = data_dir / "SA1.mp4"
    caption_path = data_dir / "SA1.TXT"

    if not media_path.exists():
        print(f"Skipping test: {media_path} not found")
        return

    print(f"Testing with media: {media_path}")

    # Call the real client.alignment
    # Duration: 00:00:03.97, start: 0.000000, bitrate: 70 kb/s
    for chunk_duration in [1, 2, 3, 3.8, 3.9, 3.95, 3.96, 3.97, 4.0, 5.0]:
        print(f"Testing chunk duration: {chunk_duration} seconds")

        client.aligner.config.streaming_chunk_duration = chunk_duration
        result_caption = client.alignment(
            input_media=str(media_path),
            input_caption=str(caption_path),
        )

    # Validate results
    assert result_caption.alignments, "No alignments returned"
    print(f"✓ Streaming alignment successful for {media_path.name}")
    print(f"  Generated {len(result_caption.alignments)} alignments")


if __name__ == "__main__":
    test_simulated_client_alignment_small_tail()
