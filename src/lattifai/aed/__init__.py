"""Audio Event Detection (AED) module for LattifAI.

This module provides audio event detection capabilities, it can identify various
audio events including speech, music, singing, and demographic characteristics
(male, female, child voices).

Key Components:
    LattifAIEventDetector: Main LED class that wraps lattifai_core's
        LattifAIEventDetector for seamless integration with LattifAI workflows.

Features:
    - Multi-class audio event detection (30+ reduced classes or 400+ full classes)
    - Voice Activity Detection (VAD) for speech segmentation
    - Gender/age classification for speech segments
    - Configurable detection thresholds and top-k filtering
    - Support for both bundled and custom pretrained models

Detected Event Types:
    - Speech: General speech activity
    - Male/Female/Child: Speaker demographic classification
    - Music: Musical content detection
    - Singing: Vocal music detection
    - Synthetic: Synthetic/electronic sounds

Configuration:
    Use AEDConfig to control:
    - enabled: Whether to run audio event detection
    - device: GPU/CPU device selection
    - dtype: Model precision (float32, float16, bfloat16)
    - reduced: Use reduced label set (33 vs 400+ classes)
    - top_k: Number of top event classes to detect
    - vad_chunk_size/vad_max_gap: VAD segmentation parameters

Example:
    >>> from lattifai.aed import LattifAIEventDetector
    >>> from lattifai.config import AEDConfig
    >>> from lattifai.audio2 import AudioLoader
    >>>
    >>> config = AEDConfig(enabled=True, device="cuda")
    >>> detector = LattifAIEventDetector(config)
    >>>
    >>> audio = AudioLoader.load("speech.wav")
    >>> result = detector.detect(audio)
    >>>
    >>> # Access VAD segments directly
    >>> for start, end in result.vad_segments:
    ...     print(f"Speech: {start:.2f} - {end:.2f}")
    >>>
    >>> # Or access the full TextGrid
    >>> print(result.audio_events)

Performance Notes:
    - GPU acceleration provides significant speedup (10x+ over CPU)
    - Use dtype="float16" for faster inference with minimal accuracy loss
    - fast_mode=True reduces computation by only detecting top_k classes
    - Long audio files are automatically chunked to manage memory

See Also:
    - lattifai.config.AEDConfig: Configuration options
    - lattifai_core.event: Core AED implementation
"""

from .lattifai import LattifAIEventDetector

__all__ = ["LattifAIEventDetector"]
