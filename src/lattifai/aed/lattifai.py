"""LattifAI Audio Event Detection implementation."""

import logging
from typing import TYPE_CHECKING, Optional, Union

from lattifai.audio2 import AudioData
from lattifai.config.aed import AEDConfig
from lattifai.logging import get_logger

if TYPE_CHECKING:
    from lattifai_core.event import LEDOutput

    from lattifai.data import Caption


formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


class LattifAIEventDetector:
    """
    LattifAI Audio Event Detector.

    This class provides a high-level interface for audio event detection,
    wrapping the core LattifAIEventDetector from lattifai_core.

    Attributes:
        config: AED configuration object.

    Example:
        >>> from lattifai.aed import LattifAIEventDetector
        >>> from lattifai.config import AEDConfig
        >>>
        >>> config = AEDConfig(enabled=True, device="cuda")
        >>> detector = LattifAIEventDetector(config)
        >>>
        >>> # Detect events from audio data
        >>> result = detector.detect(audio_data)
        >>>
        >>> # Access VAD segments directly
        >>> for start, end in result.vad_segments:
        ...     print(f"Speech: {start:.2f} - {end:.2f}")
        >>>
        >>> # Or access the full TextGrid
        >>> for tier in result.audio_events.tiers:
        ...     print(f"Event type: {tier.name}")
    """

    def __init__(self, config: Optional[AEDConfig] = None):
        """
        Initialize LattifAI Audio Event Detector.

        Args:
            config: AED configuration. If None, uses default configuration.
        """
        if config is None:
            config = AEDConfig()

        self.config = config
        self.logger = get_logger("aed")

        self._detector = None

    @property
    def name(self) -> str:
        """Human-readable name of the detector."""
        return "LattifAI_AED"

    @property
    def detector(self):
        """Lazy-load and return the audio event detector."""
        if self._detector is None:
            from lattifai_core.event import LattifAIEventDetector as CoreEventDetector

            # Load from pretrained model file
            self._detector = CoreEventDetector.from_pretrained(
                model_path=self.config.model_path,
                device=self.config.device,
                dtype=self.config.dtype,
                password=True,
            )

        return self._detector

    def detect(
        self,
        input_media: AudioData,
        vad_chunk_size: Optional[float] = None,
        vad_max_gap: Optional[float] = None,
        fast_mode: Optional[bool] = None,
    ) -> "LEDOutput":
        """
        Detect audio events in the input audio.

        Args:
            input_media: Audio data to analyze.
            vad_chunk_size: Override config vad_chunk_size.
            vad_max_gap: Override config vad_max_gap.
            fast_mode: Override config fast_mode.

        Returns:
            LEDOutput containing audio_events, event_names, vad_segments.
        """
        return self.detector(
            audio=input_media,
            VAD_CHUNK_SIZE=vad_chunk_size or self.config.vad_chunk_size,
            VAD_MAX_GAP=vad_max_gap or self.config.vad_max_gap,
            fast_mode=fast_mode if fast_mode is not None else self.config.fast_mode,
        )

    def profiling(self, reset: bool = False) -> str:
        """Get profiling information for the detector."""
        if self._detector is None:
            return ""
        return self.detector.profiling(reset=reset, logger=self.logger)

    def update(
        self,
        audio: AudioData,
        caption: "Caption",
        # led_output: Optional["LEDOutput"] = None
    ) -> "Caption":
        """
        Run AED detection and update caption with audio events.

        This is the main entry point for integrating AED with alignment.

        Args:
            audio: AudioData to analyze
            caption: Caption to update with AED results

        Returns:
            Updated Caption with audio_events field populated
        """
        # Run AED detection
        led_output, supervisions = self.detect.update(audio, caption.alignments or caption.supervisions)

        # Store audio_events in caption
        caption.audio_events = led_output.audio_events
        if caption.alignments:
            caption.alignments = supervisions
        else:
            caption.supervisions = supervisions

        return caption
