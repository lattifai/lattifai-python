"""Alignment configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from ..utils import _select_device


@dataclass
class AlignmentConfig:
    """
    Core alignment configuration.

    Defines model selection, decoding behavior, and API settings for forced alignment.
    """

    # Alignment configuration
    model_name_or_path: str = "Lattifai/Lattice-1-Alpha"
    """Model identifier or path to local model directory (e.g., 'Lattifai/Lattice-1-Alpha')."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device: 'cpu' for CPU, 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon."""

    batch_size: int = 1
    """Batch size for inference (number of samples processed simultaneously)."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""
        # Validate alignment parameters
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.device not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got {self.device}")

        if self.device == "auto":
            self.device = _select_device(self.device)
