"""Speaker diarization configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from ..utils import _select_device

if TYPE_CHECKING:
    from ..base_client import SyncAPIClient


@dataclass
class DiarizationConfig:
    """
    Speaker diarization configuration.

    Settings for speaker diarization operations.
    """

    enabled: bool = False
    """Enable speaker diarization."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    """Computation device for diarization models."""

    num_speakers: Optional[int] = None
    """Number of speakers, when known. If not set, diarization will attempt to infer the number of speakers."""

    min_speakers: Optional[int] = None
    """Minimum number of speakers. Has no effect when `num_speakers` is provided."""

    max_speakers: Optional[int] = None
    """Maximum number of speakers. Has no effect when `num_speakers` is provided."""

    model_name: str = "pyannote/speaker-diarization-community-1"
    """Model name for speaker diarization."""

    verbose: bool = False
    """Enable debug logging for diarization operations."""

    debug: bool = False
    """Enable debug mode for diarization operations."""

    client_wrapper: Optional["SyncAPIClient"] = field(default=None, repr=False)
    """Reference to the SyncAPIClient instance. Auto-set during client initialization."""

    def __post_init__(self):
        """Validate and auto-populate configuration after initialization."""
        # Validate device
        if self.device not in ("cpu", "cuda", "mps", "auto") and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of ('cpu', 'cuda', 'mps', 'auto'), got '{self.device}'")

        if self.device == "auto":
            self.device = _select_device(self.device)

        # Validate speaker counts
        if self.num_speakers is not None and self.num_speakers < 1:
            raise ValueError("num_speakers must be at least 1")

        if self.min_speakers is not None and self.min_speakers < 1:
            raise ValueError("min_speakers must be at least 1")

        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError("max_speakers must be at least 1")

        if self.min_speakers is not None and self.max_speakers is not None and self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers")
