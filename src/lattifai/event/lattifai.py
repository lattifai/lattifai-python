"""LattifAI Audio Event Detection implementation."""

import logging
from typing import TYPE_CHECKING, Optional

from lattifai.audio2 import AudioData
from lattifai.config.event import EventConfig
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
        config: EventConfig configuration object.

    Example:
        >>> from lattifai.event import LattifAIEventDetector
        >>> from lattifai.config import EventConfig
        >>>
        >>> config = EventConfig(enabled=True, device="cuda")
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

    def __init__(self, config: EventConfig):
        """
        Initialize LattifAI Audio Event Detector.

        Args:
            config: EventConfig configuration.
        """
        self.config = config
        self.logger = get_logger("event")
        self._detector = None

    @property
    def name(self) -> str:
        """Human-readable name of the detector."""
        return "LattifAI_EventDetector"

    @property
    def detector(self):
        """Lazy-load and return the audio event detector."""
        if self._detector is None:
            from lattifai_core.event import LattifAIEventDetector as CoreEventDetector

            self._detector = CoreEventDetector.from_pretrained(
                model_path=self.config.model_path,
                device=self.config.device,
                client_wrapper=self.config.client_wrapper,
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
            vad_chunk_size=vad_chunk_size or self.config.vad_chunk_size,
            vad_max_gap=vad_max_gap or self.config.vad_max_gap,
            fast_mode=fast_mode if fast_mode is not None else self.config.fast_mode,
            custom_aliases=self.config.event_aliases or {},
        )

    def profiling(self, reset: bool = False) -> str:
        """Get profiling information for the detector."""
        if self._detector is None:
            return ""
        return self.detector.profiling(reset=reset, logger=self.logger)

    def detect_and_update_caption(
        self,
        caption: "Caption",
        input_media: AudioData,
        vad_chunk_size: Optional[float] = None,
        vad_max_gap: Optional[float] = None,
        fast_mode: Optional[bool] = None,
    ) -> "Caption":
        """
        Run event detection and update caption with audio events.

        This is the main entry point for integrating event detection with alignment.
        When event_matching is enabled, it also updates caption timestamps for [Event] markers.

        Args:
            audio: AudioData to analyze
            caption: Caption to update with event detection results

        Returns:
            Updated Caption with event field populated
        """
        # Event matching: update caption timestamps based on detected events
        if self.config.event_matching:
            # Get supervisions to process
            supervisions = caption.alignments or caption.supervisions

            led_output, supervisions = self.detector.detect_and_update_supervisions(
                supervisions=supervisions,
                audio=input_media,
                vad_chunk_size=vad_chunk_size or self.config.vad_chunk_size,
                vad_max_gap=vad_max_gap or self.config.vad_max_gap,
                fast_mode=fast_mode if fast_mode is not None else self.config.fast_mode,
                custom_aliases=self.config.event_aliases or {},
                extra_events=self.config.extra_events or None,
                timestamp_tolerance=self.config.event_timestamp_tolerance,
                update_timestamps=self.config.update_event_timestamps,
            )
            # Store LEDOutput in caption
            caption.event = led_output

            if caption.alignments:
                caption.alignments = supervisions
            else:
                caption.supervisions = supervisions
        else:
            # Simple detection without event matching
            led_output = self.detect(
                input_media=input_media,
                vad_chunk_size=vad_chunk_size,
                vad_max_gap=vad_max_gap,
                fast_mode=fast_mode,
            )
            caption.event = led_output

        return caption
