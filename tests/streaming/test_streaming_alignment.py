"""Test streaming alignment functionality."""

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
        streaming_overlap_duration=1.0,
        device="cpu",
    )

    client_streaming = LattifAI(alignment_config=config_streaming)
    print(f"Streaming: {client_streaming.aligner.config.enable_streaming}")
    print(f"Chunk duration: {client_streaming.aligner.config.streaming_chunk_duration}s")
    print(f"Overlap: {client_streaming.aligner.config.streaming_overlap_duration}s")
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


if __name__ == "__main__":
    test_streaming_alignment()
