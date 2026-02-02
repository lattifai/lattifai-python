"""Audio Event Detection configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from ..utils import _select_device

if TYPE_CHECKING:
    from ..client import SyncAPIClient


@dataclass
class EventConfig:
    """
    Audio Event Detection configuration.

    Settings for detecting audio events (Speech, Music, Male, Female...) in audio files using the AED model.

    Event Matching:
        When event_matching is enabled, the AED system will:
        1. Parse [Event] markers from input captions (e.g., [Music], [Applause])
        2. Match caption events to AED labels using semantic matching
        3. Force detection of matched labels even if not in top_k
        4. Update caption timestamps based on AED detection results

        Event matching logic is implemented in lattifai_core.event.EventMatcher.
    """

    enabled: bool = False
    """Enable audio event detection."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device for AED models."""

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

    event_matching: bool = True
    """Whether update events in the alignment"""

    extra_labels: List[str] = field(default_factory=list)
    """Additional AED labels to always detect, even if not in top_k.
    Example: ["Applause", "Laughter", "Music"]
    """

    event_aliases: Dict[str, List[str]] = field(default_factory=dict)
    """Custom aliases for event matching.
    Example: {"Laughs": ["Laughter"], "Heavy breathing": ["Breathing", "Pant"]}
    """

    event_timestamp_tolerance: float = 2.0
    """Maximum time difference (seconds) for matching caption events to AED detections."""

    update_event_timestamps: bool = True
    """Whether to update caption event timestamps based on AED detections."""

    client_wrapper: Optional["SyncAPIClient"] = field(default=None, repr=False)
    """Reference to the SyncAPIClient instance. Auto-set during client initialization."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""
        # Validate device
        if self.device not in ("cpu", "cuda", "mps", "auto") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got '{self.device}'")

        if self.device == "auto":
            self.device = _select_device(self.device)

        # Validate top_k
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Validate vad_chunk_size
        if self.vad_chunk_size < 0:
            raise ValueError("vad_chunk_size must be non-negative")

        # Validate vad_max_gap
        if self.vad_max_gap < 0:
            raise ValueError("vad_max_gap must be non-negative")

        # Validate event_timestamp_tolerance
        if self.event_timestamp_tolerance < 0:
            raise ValueError("event_timestamp_tolerance must be non-negative")
