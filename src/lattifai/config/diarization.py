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

    min_claim_duration: float = 10.0
    """Confidence gate for speaker-tier mapping: minimum total overlap in seconds.
    置信度门控：speaker 被标注的片段与 diarization tier 重叠的最小总时长（秒）。

    When input captions carry speaker labels — from Gemini transcription, original
    subtitle metadata, or any other source — we check how much of each labeled
    speaker's audio overlaps with each diarization tier (SPEAKER_00, SPEAKER_01, ...).
    If the total overlap with the best-matching tier is below this threshold, the
    evidence is too weak to rename the tier — it stays as-is (e.g. "SPEAKER_01").

    Example: a 0.8s labeled segment is not enough to confidently claim a tier
    that covers 60s of audio. Set higher for stricter mapping, 0 to disable.

    Tip: to keep raw diarization names (SPEAKER_00, SPEAKER_01, ...) and skip
    all speaker renaming, set both this and min_claim_count to very large values
    (e.g. min_claim_duration=999999, min_claim_count=9999).
    如果希望保留 SPEAKER_XX 原始格式、不做任何重命名，将这两个值设得足够大即可。
    """

    min_claim_count: int = 2
    """Confidence gate for speaker-tier mapping: minimum number of labeled segments.
    置信度门控：speaker 被标注的片段与 diarization tier 匹配的最小次数。

    Works together with min_claim_duration as a dual safeguard. A speaker must have
    at least this many labeled segments overlapping with its dominant tier for the
    mapping to be accepted. Both thresholds must pass simultaneously.

    Example: with min_claim_count=2, a single labeled line (even if long) won't
    rename a tier — the same speaker must appear in at least two labeled segments.

    See min_claim_duration tip for how to disable renaming entirely.
    """

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

        if self.min_claim_duration < 0:
            raise ValueError("min_claim_duration must be non-negative")

        if self.min_claim_count < 1:
            raise ValueError("min_claim_count must be at least 1")
