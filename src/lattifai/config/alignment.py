"""Alignment configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Literal, Optional

from ..utils import _select_device

if TYPE_CHECKING:
    from ..base_client import SyncAPIClient


@dataclass
class AlignmentConfig:
    """
    Core alignment configuration.

    Defines model selection, decoding behavior, and API settings for forced alignment.
    """

    # Alignment configuration
    model_name: str = "LattifAI/Lattice-1"
    """Model identifier or path to local model directory (e.g., 'LattifAI/Lattice-1')."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device: 'cpu' for CPU, 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon."""

    batch_size: int = 1
    """Batch size for inference (number of samples processed simultaneously)."""

    # Segmented Alignment for Long Audio
    trust_caption_timestamps: bool = False
    """When True, use original caption timestamps as strong reference constraints during alignment.
    The alignment process will still adjust timestamps but stay close to the input timing.
    Use this when you want to re-segment caption sentence boundaries (caption.split_sentence=True)
    while preserving the approximate timing from the original captions.
    When False (default), performs unconstrained forced alignment based purely on media-caption matching.
    """

    strategy: Literal["caption", "transcription", "entire"] = "entire"
    """Alignment strategy for long audio alignment:
    - 'entire': Process entire audio as single alignment (default, suitable for <30 min)
    - 'caption': Split based on existing caption boundaries and gaps (segment_max_gap)
        work with `alignment.trust_caption_timestamps=true`
    - 'transcription': Align media with transcription first, then segment based on transcription

    Use segmentation for long audio (>30 min) to reduce memory usage and improve performance.
    """

    segment_duration: float = 300.0
    """Target duration (in seconds) for each alignment segment when using 'caption' strategy.
    Default: 300.0 (5 minutes). Typical range: 30-600 seconds (30s-10min).
    Shorter segments = lower memory, longer segments = better context for alignment.
    """

    segment_max_gap: float = 4.0
    """Maximum gap (in seconds) between captions to consider them part of the same segment.
    Used by 'caption' and 'adaptive' strategies. Gaps larger than this trigger segment splitting.
    Default: 4.0 seconds. Useful for detecting scene changes or natural breaks in content.
    """

    # Beam search parameters for forced alignment
    search_beam: int = 200
    """Search beam size for beam search decoding. Larger values explore more hypotheses but are slower.
    Default: 200. Typical range: 20-500.
    """

    output_beam: int = 80
    """Output beam size for keeping top hypotheses. Should be smaller than search_beam.
    Default: 80. Typical range: 10-200.
    """

    min_active_states: int = 400
    """Minimum number of active states during decoding. Controls memory and search space.
    Default: 400. Typical range: 30-1000.
    """

    max_active_states: int = 10000
    """Maximum number of active states during decoding. Prevents excessive memory usage.
    Default: 10000. Typical range: 1000-20000.
    """

    client_wrapper: Optional["SyncAPIClient"] = field(default=None, repr=False)
    """Reference to the SyncAPIClient instance. Auto-set during client initialization."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""
        # Validate alignment parameters
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.device not in ("cpu", "cuda", "mps", "auto") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got {self.device}")

        if self.device == "auto":
            self.device = _select_device(self.device)
