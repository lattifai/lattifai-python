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
        audio_sr: Tuple[torch.Tensor, int],
        sampling_rate: int,
        device: Optional[str],
        channel_selector: Optional[ChannelSelectorType],
    ) -> torch.Tensor:
        """Resample audio to target sampling rate with channel selection.

        Args:
            audio_sr: Tuple of (audio_tensor, original_sample_rate).
            sampling_rate: Target sampling rate.
            device: Device to perform resampling on.
            channel_selector: How to select channels.

        Returns:
            Resampled audio tensor of shape (1, T) or (C, T).
        """
        audio, sr = audio_sr

        if channel_selector is None:
            # keep the original multi-channel signal
            tensor = audio
        elif isinstance(channel_selector, int):
            assert audio.shape[0] >= channel_selector, f"Invalid channel: {channel_selector}"
            tensor = audio[channel_selector : channel_selector + 1].clone()
            del audio
        elif isinstance(channel_selector, str):
            assert channel_selector == "average"
            tensor = torch.mean(audio.to(device), dim=0, keepdim=True)
            del audio
        else:
            raise ValueError(f"Unsupported channel_selector: {channel_selector}")
            # assert isinstance(channel_selector, Iterable)
            # num_channels = audio.shape[0]
            # print(f"Selecting channels {channel_selector} from the signal with {num_channels} channels.")
            # if max(channel_selector) >= num_channels:
            #     raise ValueError(
            #         f"Cannot select channel subset {channel_selector} from a signal with {num_channels} channels."
            #     )
            # tensor = audio[channel_selector]

        tensor = tensor.to(device)
        if sr != sampling_rate:
            cache_key = (sr, sampling_rate, device)
            if cache_key not in self._resampler_cache:
                self._resampler_cache[cache_key] = get_or_create_resampler(sr, sampling_rate).to(device=device)
            resampler = self._resampler_cache[cache_key]

            length = tensor.size(-1)
            chunk_size = sampling_rate * 3600
            if length > chunk_size:
                resampled_chunks = []
                for i in range(0, length, chunk_size):
                    resampled_chunks.append(resampler(tensor[..., i : i + chunk_size]))
                tensor = torch.cat(resampled_chunks, dim=-1)
            else:
                tensor = resampler(tensor)

        return tensor

    def _load_audio(
        self,
        audio: Union[Pathlike, BinaryIO],
        sampling_rate: int,
        channel_selector: Optional[ChannelSelectorType],
    ) -> torch.Tensor:
        """Load audio from file or binary stream and resample to target rate.

        Args:
            audio: Path to audio file or binary stream.
            sampling_rate: Target sampling rate.
            channel_selector: How to select channels.

        Returns:
            Resampled audio tensor.

        Raises:
            ImportError: If PyAV is needed but not installed.
            ValueError: If no audio stream found.
            RuntimeError: If audio loading fails.
        """
        if isinstance(audio, Pathlike):
            audio = str(Path(str(audio)).expanduser())

        # load audio
        try:
            waveform, sample_rate = sf.read(audio, always_2d=True, dtype="float32")  # numpy array
            waveform = waveform.T  # (channels, samples)
        except Exception as primary_error:
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

                # Resample to target sample rate during decoding
                audio_stream.codec_context.format = av.AudioFormat("flt")  # 32-bit float

                frames = []
                for frame in container.decode(audio_stream):
                    # Convert frame to numpy array
                    array = frame.to_ndarray()
                    # Ensure shape is (channels, samples)
                    if array.ndim == 1:
                        array = array.reshape(1, -1)
                    elif array.ndim == 2 and array.shape[0] > array.shape[1]:
                        array = array.T
                    frames.append(array)

                container.close()

                if not frames:
                    raise ValueError(f"No audio data found in file: {audio}")

                # Concatenate all frames
                waveform = np.concatenate(frames, axis=1).astype(np.float32)  # (channels, samples)
                sample_rate = audio_stream.codec_context.sample_rate
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file {audio}: {e}")

        return self._resample_audio(
            (torch.from_numpy(waveform), sample_rate),
            sampling_rate,
            device=self.device,
            channel_selector=channel_selector,
        )

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
        tensor = self._load_audio(audio, sampling_rate, channel_selector)

        # tensor is (1, T) or (C, T), convert to numpy and free tensor memory
        ndarray = tensor.cpu().numpy()
        del tensor

        return AudioData(
            sampling_rate=sampling_rate,
            ndarray=ndarray,
            path=str(audio) if isinstance(audio, Pathlike) else "<BinaryIO>",
            streaming_chunk_secs=streaming_chunk_secs,
            overlap_secs=0.0,
        )
