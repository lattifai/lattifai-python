"""Composed workflow configurations for LattifAI."""

from dataclasses import dataclass, field
from pathlib import Path

import nemo_run as run

from .alignment import AlignmentConfig
from .subtitle import SubtitleConfig
from .transcription import TranscriptionConfig


@dataclass
class YouTubeWorkflowConfig:
    """
    Complete YouTube workflow configuration.

    Composes all configs needed for YouTube video processing:
    download, transcription, alignment, and subtitle generation.
    """

    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    """Alignment configuration (includes API settings)."""

    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    """Subtitle I/O configuration."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    """Transcription service configuration."""

    output_dir: Path = field(default_factory=lambda: Path("."))
    """Base output directory for all workflow outputs."""

    def __post_init__(self):
        """Validate and synchronize configuration after initialization."""
        # Ensure output_dir is a Path object
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Synchronize subtitle output_dir with workflow output_dir
        object.__setattr__(self.subtitle, "output_dir", self.output_dir)
