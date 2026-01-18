"""Speaker diarization module for LattifAI.

This module provides multi-speaker identification and labeling capabilities
using pyannote.audio-based diarization models. It can identify who spoke
when in an audio file and optionally match detected speakers with existing
speaker labels from input captions.

Key Components:
    LattifAIDiarizer: Main diarization class that wraps pyannote.audio
        pipelines for speaker segmentation and clustering.

Features:
    - Automatic speaker detection with configurable min/max speaker counts
    - Speaker label preservation from input captions (e.g., "Alice:", ">> Bob:")
    - Integration with alignment results to assign speakers to words/segments
    - Support for pre-computed diarization results (avoid reprocessing)

Configuration:
    Use DiarizationConfig to control:
    - enabled: Whether to run diarization
    - min_speakers/max_speakers: Constrain speaker count detection
    - device: GPU/CPU device selection
    - debug: Enable verbose output

Example:
    >>> from lattifai import LattifAI
    >>> from lattifai.config import DiarizationConfig
    >>> client = LattifAI(diarization_config=DiarizationConfig(enabled=True))
    >>> caption = client.alignment(audio="speech.wav", input_caption="transcript.srt")
    >>> for seg in caption.supervisions:
    ...     print(f"{seg.speaker}: {seg.text}")

Performance Notes:
    - Diarization adds ~10-30% processing time to alignment
    - GPU acceleration recommended for longer audio files
    - Results are cached when output_path is provided

See Also:
    - lattifai.config.DiarizationConfig: Configuration options
    - lattifai.client.LattifAI.speaker_diarization: Direct diarization method
"""

from .lattifai import LattifAIDiarizer

__all__ = ["LattifAIDiarizer"]
