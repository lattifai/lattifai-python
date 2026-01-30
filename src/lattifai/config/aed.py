"""Audio Event Detection (AED) configuration for LattifAI."""

from dataclasses import dataclass
from typing import Literal

from ..utils import _select_device


@dataclass
class AEDConfig:
    """
    Audio Event Detection configuration.

    Settings for detecting audio events (Speech, Music, Male, Female, Child,
    Singing, Synthetic) in audio files using the ATST-SED model.
    """

    enabled: bool = False
    """Enable audio event detection."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device for AED models."""

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Model dtype for inference. Use float16 for faster inference on GPU."""

    batch_size: int = 64
    """Batch size for inference."""

    top_k: int = 10
    """Number of top event classes to detect."""

    vad_chunk_size: float = 30.0
    """VAD chunk size in seconds for speech segmentation."""

    vad_max_gap: float = 2.0
    """Maximum gap in seconds between VAD segments to merge."""

    fast_mode: bool = True
    """Enable fast mode (only detect top_k classes, skip others)."""

    model_path: str = ""
    """Path to pretrained model. If empty, uses default bundled model."""

    verbose: bool = False
    """Enable verbose logging for AED operations."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""
        # Validate device
        if self.device not in ("cpu", "cuda", "mps", "auto") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got '{self.device}'")

        if self.device == "auto":
            self.device = _select_device(self.device)

        # Validate dtype
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dtype must be one of ('float32', 'float16', 'bfloat16'), got '{self.dtype}'")

        # Validate batch_size
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Validate top_k
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Validate vad_chunk_size
        if self.vad_chunk_size < 0:
            raise ValueError("vad_chunk_size must be non-negative")

        # Validate vad_max_gap
        if self.vad_max_gap < 0:
            raise ValueError("vad_max_gap must be non-negative")
