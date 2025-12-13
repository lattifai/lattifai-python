"""Audio loading and resampling utilities."""

from collections import namedtuple
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from lhotse.augmentation import get_or_create_resampler
from lhotse.utils import Pathlike

from lattifai.errors import AudioLoadError

# ChannelSelectorType = Union[int, Iterable[int], str]
ChannelSelectorType = Union[int, str]


class AudioData(namedtuple("AudioData", ["sampling_rate", "ndarray", "path", "streaming_chunk_secs", "overlap_secs"])):
    """Audio data container with sampling rate and numpy array.

    Supports iteration to stream audio chunks for processing long audio files.
    The streaming_chunk_secs field indicates whether streaming mode should be used downstream.
    The overlap_secs field specifies the overlap duration between consecutive chunks.
    Note: tensor field removed to reduce memory usage. Convert ndarray to tensor on-demand.
    """

    def __str__(self) -> str:
        return self.path

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return self.ndarray.shape[-1] / self.sampling_rate

    @property
    def streaming_mode(self) -> bool:
        """Indicates whether streaming mode is enabled based on streaming_chunk_secs."""
        if self.streaming_chunk_secs is not None:
            return self.duration > self.streaming_chunk_secs * 1.1
        return False

    def __iter__(self):
        """Initialize iterator for chunk-based audio streaming.

        Returns an iterator that yields audio chunks as AudioData instances.
        Uses streaming_chunk_secs and overlap_secs from the instance.
        """
        return self.iter_chunks()

    def iter_chunks(
        self,
        chunk_secs: Optional[float] = None,
        overlap_secs: Optional[float] = None,
    ):
        """Iterate over audio chunks with configurable duration and overlap.

        Args:
            chunk_secs: Duration of each chunk in seconds (default: uses streaming_chunk_secs or 600.0).
            overlap_secs: Overlap between consecutive chunks in seconds (default: uses overlap_secs or 0.0).

        Yields:
            AudioData: Chunks of audio data.

        Example:
            >>> audio = loader("long_audio.wav")
            >>> for chunk in audio.iter_chunks(chunk_secs=60.0, overlap_secs=2.0):
            ...     process(chunk)
        """
        chunk_duration = chunk_secs or self.streaming_chunk_secs or 600.0
        overlap_duration = overlap_secs or self.overlap_secs or 0.0

        chunk_size = int(chunk_duration * self.sampling_rate)
        overlap_size = int(overlap_duration * self.sampling_rate)
        step_size = chunk_size - overlap_size
        total_samples = self.ndarray.shape[-1]

        current_offset = 0
        while current_offset < total_samples:
            start = current_offset
            end = min(start + chunk_size, total_samples)

            # Extract chunk from ndarray only
            chunk_ndarray = self.ndarray[..., start:end]

            yield AudioData(
                sampling_rate=self.sampling_rate,
                ndarray=chunk_ndarray,
                path=f"{self.path}[{start/self.sampling_rate:.2f}s-{end/self.sampling_rate:.2f}s]",
                streaming_chunk_secs=None,
                overlap_secs=None,
            )

            current_offset += step_size


class AudioLoader:
    """Load and preprocess audio files into AudioData format."""

    def __init__(
        self,
        device: str = "cpu",
    ):
        """Initialize AudioLoader.

        Args:
            device: Device to load audio tensors on (default: "cpu").
        """
        self.device = device
        self._resampler_cache = {}

    def _resample_audio(
        self,
        audio_sr: Tuple[np.ndarray, int],
        sampling_rate: int,
        device: Optional[str],
        channel_selector: Optional[ChannelSelectorType],
    ) -> np.ndarray:
        """Resample audio to target sampling rate with channel selection.

        Args:
            audio_sr: Tuple of (audio, original_sample_rate).
            sampling_rate: Target sampling rate.
            device: Device to perform resampling on.
            channel_selector: How to select channels.

        Returns:
            Resampled audio array of shape (1, T) or (C, T).
        """
        audio, sr = audio_sr

        if channel_selector is None:
            # keep the original multi-channel signal
            tensor = audio.T
            del audio  # Free original audio memory
        elif isinstance(channel_selector, int):
            assert audio.shape[1] >= channel_selector, f"Invalid channel: {channel_selector}"
            tensor = audio[:, channel_selector : channel_selector + 1].T.copy()
            del audio
        elif isinstance(channel_selector, str):
            assert channel_selector == "average"
            tensor = np.mean(audio, axis=1, keepdims=True).T
            del audio
        else:
            raise ValueError(f"Unsupported channel_selector: {channel_selector}")

        # tensor: np.ndarray (channels, samples)
        if sr != sampling_rate:
            cache_key = (sr, sampling_rate, device)
            if cache_key not in self._resampler_cache:
                self._resampler_cache[cache_key] = get_or_create_resampler(sr, sampling_rate).to(device=device)
            resampler = self._resampler_cache[cache_key]

            tensor = resampler(torch.from_numpy(tensor).to(device=device))
            tensor = tensor.cpu().numpy()

        return tensor

    def _load_audio(
        self,
        audio: Union[Pathlike, BinaryIO],
        sampling_rate: int,
        channel_selector: Optional[ChannelSelectorType],
    ) -> np.ndarray:
        """Load audio from file or binary stream and resample to target rate.

        Args:
            audio: Path to audio file or binary stream.
            sampling_rate: Target sampling rate.
            channel_selector: How to select channels.

        Returns:
            Resampled audio as a NumPy array of shape (channels, samples).

        Raises:
            ImportError: If PyAV is needed but not installed.
            ValueError: If no audio stream found.
            RuntimeError: If audio loading fails.
        """
        if isinstance(audio, Pathlike):
            audio = str(Path(str(audio)).expanduser())

        # load audio in chunks to reduce memory footprint for long files
        try:
            # First check file duration to decide loading strategy
            info = sf.info(audio)
            duration = info.duration

            # For very long audio (>60 minutes), use chunk-based loading
            if duration > 3600:  # 60 minutes
                with sf.SoundFile(audio, "r") as f:
                    sample_rate = f.samplerate
                    total_frames = f.frames

                    # Pre-calculate output size to avoid list accumulation
                    num_channels = 1 if channel_selector else f.channels
                    expected_output_samples = int(total_frames * sampling_rate / sample_rate)

                    # Pre-allocate output array
                    waveform = np.zeros((num_channels, expected_output_samples), dtype=np.float32)

                    # Use source sample rate for reading, not target
                    chunk_frames = int(sample_rate * 1800)  # 30-minute chunks at source rate
                    output_offset = 0

                    while True:
                        chunk = f.read(frames=chunk_frames, dtype="float32", always_2d=True)
                        if chunk.size == 0:
                            break

                        # Resample chunk -> (channels, samples)
                        resampled_chunk = self._resample_audio(
                            (chunk, sample_rate),
                            sampling_rate,
                            device=self.device,
                            channel_selector=channel_selector,
                        )

                        # Write directly to pre-allocated array
                        chunk_length = resampled_chunk.shape[-1]
                        waveform[..., output_offset : output_offset + chunk_length] = resampled_chunk
                        output_offset += chunk_length

                        # Clean up immediately
                        del chunk, resampled_chunk

                    # Trim to actual size if needed (due to rounding in resampling)
                    if output_offset < expected_output_samples:
                        waveform = waveform[..., :output_offset]

                return waveform
            else:
                # For shorter audio, use standard loading
                waveform, sample_rate = sf.read(audio, always_2d=True, dtype="float32")
                # Resample and return directly to avoid double processing
                result = self._resample_audio(
                    (waveform, sample_rate),
                    sampling_rate,
                    device=self.device,
                    channel_selector=channel_selector,
                )
                del waveform
                return result
        except Exception as primary_error:
            print(f"Primary error with soundfile: {primary_error}")
            # Fallback to PyAV for formats not supported by soundfile
            try:
                import av
            except ImportError:
                raise AudioLoadError(
                    "PyAV (av) is required for loading certain audio formats. "
                    f"Install it with: pip install av\n"
                    f"Primary error was: {primary_error}"
                )

            try:
                container = av.open(audio)
                audio_stream = next((s for s in container.streams if s.type == "audio"), None)

                if audio_stream is None:
                    raise ValueError(f"No audio stream found in file: {audio}")

                audio_stream.codec_context.format = av.AudioFormat("flt")  # 32-bit float
                sample_rate = audio_stream.codec_context.sample_rate

                # Estimate duration to decide processing strategy
                duration_estimate = None
                if audio_stream.duration and audio_stream.time_base:
                    duration_estimate = float(audio_stream.duration * audio_stream.time_base)
                else:
                    print(f"WARNING: Failed to estimate duration for audio: {audio}")

                # For very long audio (>30 minutes), process and resample in chunks
                if duration_estimate and duration_estimate > 1800:
                    # Estimate output size and pre-allocate with buffer
                    num_channels = 1 if channel_selector else audio_stream.codec_context.channels
                    estimated_samples = int(duration_estimate * sampling_rate * 1.1)  # 10% buffer
                    waveform = np.zeros((num_channels, estimated_samples), dtype=np.float32)

                    frames = []
                    accumulated_samples = 0
                    output_offset = 0
                    chunk_sample_target = int(sample_rate * 600)  # 10 minutes at original rate

                    for frame in container.decode(audio_stream):
                        array = frame.to_ndarray()

                        # Ensure shape is (samples, channels)
                        if array.ndim == 1:
                            array = array.reshape(-1, 1)
                        elif array.ndim == 2 and array.shape[0] < array.shape[1]:
                            array = array.T

                        frames.append(array)
                        accumulated_samples += array.shape[0]

                        # Process chunk when accumulated enough samples
                        if accumulated_samples >= chunk_sample_target:
                            chunk = np.concatenate(frames, axis=0).astype(np.float32)
                            del frames  # Free frames list before resampling
                            # Resample chunk -> (channels, samples)
                            resampled_chunk = self._resample_audio(
                                (chunk, sample_rate),
                                sampling_rate,
                                device=self.device,
                                channel_selector=channel_selector,
                            )

                            chunk_length = resampled_chunk.shape[-1]
                            if output_offset + chunk_length > waveform.shape[-1]:
                                print(
                                    f"WARNING: Trimming resampled chunk from {chunk_length} to {waveform.shape[-1] - output_offset} samples to fit waveform buffer for audio: {audio}"  # noqa: E501
                                )
                                resampled_chunk = resampled_chunk[:, : waveform.shape[-1] - output_offset]

                            # Write directly to array
                            waveform[..., output_offset : output_offset + chunk_length] = resampled_chunk
                            output_offset += chunk_length

                            # Clean up immediately
                            del chunk, resampled_chunk
                            frames = []  # Create new list
                            accumulated_samples = 0

                    # Process remaining frames
                    if frames:
                        chunk = np.concatenate(frames, axis=0).astype(np.float32)
                        del frames
                        resampled_chunk = self._resample_audio(
                            (chunk, sample_rate),
                            sampling_rate,
                            device=self.device,
                            channel_selector=channel_selector,
                        )

                        chunk_length = resampled_chunk.shape[-1]
                        if output_offset + chunk_length > waveform.shape[-1]:
                            print(
                                f"WARNING: Trimming resampled chunk from {chunk_length} to {waveform.shape[-1] - output_offset} samples to fit waveform buffer for audio: {audio}"  # noqa: E501
                            )
                            resampled_chunk = resampled_chunk[:, : waveform.shape[-1] - output_offset]

                        waveform[..., output_offset : output_offset + chunk_length] = resampled_chunk
                        output_offset += chunk_length
                        del chunk, resampled_chunk

                    container.close()

                    if output_offset == 0:
                        raise ValueError(f"No audio data found in file: {audio}")

                    # Trim to actual size
                    waveform = waveform[..., :output_offset]
                    return waveform
                else:
                    # For shorter audio, process in batches to reduce memory
                    frames = []
                    for frame in container.decode(audio_stream):
                        array = frame.to_ndarray()
                        # Ensure shape is (channels, samples)
                        if array.ndim == 1:
                            array = array.reshape(-1, 1)
                        elif array.ndim == 2 and array.shape[0] < array.shape[1]:
                            array = array.T
                        frames.append(array)
                    container.close()

                    if not frames:
                        raise ValueError(f"No audio data found in file: {audio}")

                    # Concatenate remaining frames
                    waveform = np.concatenate(frames, axis=0).astype(np.float32)
                    del frames
                    # Resample and return directly
                    result = self._resample_audio(
                        (waveform, sample_rate),
                        sampling_rate,
                        device=self.device,
                        channel_selector=channel_selector,
                    )
                    del waveform
                    return result
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file {audio}: {e}")

    def __call__(
        self,
        audio: Union[Pathlike, BinaryIO],
        sampling_rate: int = 16000,
        channel_selector: Optional[ChannelSelectorType] = "average",
        streaming_chunk_secs: Optional[float] = None,
    ) -> AudioData:
        """
        Args:
            audio: Path to audio file or binary stream.
            channel_selector: How to select channels (default: "average").
            sampling_rate: Target sampling rate (default: use instance sampling_rate).
            streaming_chunk_secs: Duration in seconds for streaming chunks (default: None, disabled).

        Returns:
            AudioData namedtuple with sampling_rate, ndarray, and streaming_chunk_secs fields.
        """
        ndarray = self._load_audio(audio, sampling_rate, channel_selector)
        return AudioData(
            sampling_rate=sampling_rate,
            ndarray=ndarray,
            path=str(audio) if isinstance(audio, Pathlike) else "<BinaryIO>",
            streaming_chunk_secs=streaming_chunk_secs,
            overlap_secs=0.0,
        )
