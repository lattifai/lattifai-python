"""LattifAI Audio Event Detection implementation."""

import logging
from typing import TYPE_CHECKING, Optional, Union

from tgt import TextGrid

from lattifai.audio2 import AudioData
from lattifai.config.aed import AEDConfig
from lattifai.logging import get_logger

if TYPE_CHECKING:
    from lattifai_core.event import LEDOutput

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)


class LattifAIEventDetector:
    """
    LattifAI Audio Event Detector using ATST-SED.

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
            )

        return self._detector

    def detect(
        self,
        input_media: AudioData,
        vad_chunk_size: Optional[float] = None,
        vad_max_gap: Optional[float] = None,
        fast_mode: Optional[bool] = None,
    ):
        """
        Detect audio events in the input audio.

        Args:
            input_media: Audio data to analyze. Can be AudioData namedtuple
                with (sampling_rate, ndarray) or path that will be loaded.
            vad_chunk_size: Override config vad_chunk_size. Size in seconds
                for VAD chunking. Use 0.0 to disable chunking.
            vad_max_gap: Override config vad_max_gap. Maximum gap in seconds
                between VAD segments before they are merged.
            fast_mode: Override config fast_mode. If True, only detect
                top_k classes for faster inference.

        Returns:
            LEDOutput containing:
            - audio_events: TextGrid with detected events (Speech, Music, etc.)
            - event_names: List of detected event type names
            - vad_segments: List of (start, end) tuples for VAD segments

        Example:
            >>> detector = LattifAIEventDetector()
            >>> result = detector.detect(audio_data)
            >>>
            >>> # Get VAD segments for alignment chunking
            >>> for start, end in result.vad_segments:
            ...     print(f"Speech segment: {start:.2f} - {end:.2f}")
            >>>
            >>> # Get speech segments from TextGrid
            >>> if result.audio_events.has_tier("Speech"):
            ...     for interval in result.audio_events.get_tier_by_name("Speech").intervals:
            ...         print(f"{interval.start_time:.2f}-{interval.end_time:.2f}")
        """
        if vad_chunk_size is None:
            vad_chunk_size = self.config.vad_chunk_size
        if vad_max_gap is None:
            vad_max_gap = self.config.vad_max_gap
        if fast_mode is None:
            fast_mode = self.config.fast_mode

        # Convert to AudioData namedtuple if needed
        from lattifai_core.event import AudioData as CoreAudioData

        if isinstance(input_media, AudioData):
            core_audio = CoreAudioData(
                sampling_rate=input_media.sampling_rate,
                ndarray=input_media.ndarray,
            )
        else:
            core_audio = input_media

        return self.detector(
            audio=core_audio,
            VAD_CHUNK_SIZE=vad_chunk_size,
            VAD_MAX_GAP=vad_max_gap,
            fast_mode=fast_mode,
        )

    def profiling(self, reset: bool = False) -> str:
        """
        Get profiling information for the detector.

        Args:
            reset: If True, reset timing counters after reporting.

        Returns:
            Formatted string with timing statistics for each processing stage.
        """
        if self._detector is None:
            return ""
        return self.detector.profiling(reset=reset, logger=self.logger)

    @staticmethod
    def get_gender(led_output: Union["LEDOutput", TextGrid]) -> TextGrid:
        """
        Extract gender/age classification tiers from LED results.

        Args:
            led_output: LEDOutput instance or TextGrid with LED detection results.

        Returns:
            TextGrid containing only Male, Female, and Child tiers.
        """
        from lattifai_core.event import LattifAIEventDetector as CoreEventDetector

        return CoreEventDetector.get_gender(led_output)
