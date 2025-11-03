import json
import time
from collections import defaultdict
from typing import Any, BinaryIO, Dict, Tuple, Union

import numpy as np
import onnxruntime as ort
import resampy
import soundfile as sf
import torch
from lhotse import FbankConfig
from lhotse.audio import read_audio
from lhotse.features.kaldi.layers import Wav2LogFilterBank
from lhotse.utils import Pathlike

from lattifai.errors import AlignmentError, AudioFormatError, AudioLoadError, DependencyError, ModelLoadError


class Lattice1AlphaWorker:
    """Worker for processing audio with LatticeGraph."""

    def __init__(self, model_path: Pathlike, device: str = 'cpu', num_threads: int = 8) -> None:
        try:
            self.config = json.load(open(f'{model_path}/config.json'))
        except Exception as e:
            raise ModelLoadError(f'config from {model_path}', original_error=e)

        # SessionOptions
        sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads  # CPU cores
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.add_session_config_entry('session.intra_op.allow_spinning', '0')

        providers = []
        if device.startswith('cuda') and ort.get_all_providers().count('CUDAExecutionProvider') > 0:
            providers.append('CUDAExecutionProvider')
        elif device.startswith('mps') and ort.get_all_providers().count('MPSExecutionProvider') > 0:
            providers.append('MPSExecutionProvider')

        try:
            self.acoustic_ort = ort.InferenceSession(
                f'{model_path}/acoustic_opt.onnx',
                sess_options,
                providers=providers + ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
            )
        except Exception as e:
            raise ModelLoadError(f'acoustic model from {model_path}', original_error=e)

        try:
            config = FbankConfig(num_mel_bins=80, device=device, snip_edges=False)
            config_dict = config.to_dict()
            config_dict.pop('device')
            self.extractor = Wav2LogFilterBank(**config_dict).to(device).eval()
        except Exception as e:
            raise ModelLoadError(f'feature extractor for device {device}', original_error=e)

        self.device = torch.device(device)
        self.timings = defaultdict(lambda: 0.0)

    @torch.inference_mode()
    def emission(self, audio: torch.Tensor) -> torch.Tensor:
        _start = time.time()
        # audio -> features -> emission
        features = self.extractor(audio)  # (1, T, D)
        if features.shape[1] > 6000:
            features_list = torch.split(features, 6000, dim=1)
            emissions = []
            for features in features_list:
                ort_inputs = {
                    'features': features.cpu().numpy(),
                    'feature_lengths': np.array([features.size(1)], dtype=np.int64),
                }
                emission = self.acoustic_ort.run(None, ort_inputs)[0]  # (1, T, vocab_size) numpy
                emissions.append(emission)
            emission = torch.cat(
                [torch.from_numpy(emission).to(self.device) for emission in emissions], dim=1
            )  # (1, T, vocab_size)
        else:
            ort_inputs = {
                'features': features.cpu().numpy(),
                'feature_lengths': np.array([features.size(1)], dtype=np.int64),
            }
            emission = self.acoustic_ort.run(None, ort_inputs)[0]  # (1, T, vocab_size) numpy
            emission = torch.from_numpy(emission).to(self.device)

        self.timings['emission'] += time.time() - _start
        return emission  # (1, T, vocab_size) torch

    def load_audio(self, audio: Union[Pathlike, BinaryIO]) -> Tuple[torch.Tensor, int]:
        # load audio
        try:
            waveform, sample_rate = read_audio(audio)  # numpy array
            if len(waveform.shape) == 1:
                waveform = waveform.reshape([1, -1])  # (1, L)
            else:  # make sure channel first
                if waveform.shape[0] > waveform.shape[1]:
                    waveform = waveform.transpose(0, 1)
                # average multiple channels
                waveform = np.mean(waveform, axis=0, keepdims=True)  # (1, L)
        except Exception as primary_error:
            # Fallback to PyAV for formats not supported by soundfile
            try:
                import av
            except ImportError:
                raise DependencyError(
                    'av (PyAV)', install_command='pip install av', context={'primary_error': str(primary_error)}
                )

            try:
                container = av.open(audio)
                audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

                if audio_stream is None:
                    raise AudioFormatError(str(audio), 'No audio stream found in file')

                # Resample to target sample rate during decoding
                audio_stream.codec_context.format = av.AudioFormat('flt')  # 32-bit float

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
                    raise AudioFormatError(str(audio), 'No audio data found in file')

                # Concatenate all frames
                waveform = np.concatenate(frames, axis=1)
                # Average multiple channels to mono
                if waveform.shape[0] > 1:
                    waveform = np.mean(waveform, axis=0, keepdims=True)

                sample_rate = audio_stream.codec_context.sample_rate
            except Exception as e:
                raise AudioLoadError(str(audio), original_error=e)

        try:
            if sample_rate != self.config['sample_rate']:
                waveform = resampy.resample(waveform, sample_rate, self.config['sample_rate'], axis=1)
        except Exception:
            raise AudioFormatError(
                str(audio), f'Failed to resample from {sample_rate}Hz to {self.config["sample_rate"]}Hz'
            )

        return torch.from_numpy(waveform).to(self.device)  # (1, L)

    def alignment(
        self, audio: Union[Union[Pathlike, BinaryIO], torch.tensor], lattice_graph: Tuple[str, int, float]
    ) -> Dict[str, Any]:
        """Process audio with LatticeGraph.

        Args:
            audio: Audio file path or binary data
            lattice_graph: LatticeGraph data

        Returns:
            Processed LatticeGraph

        Raises:
            AudioLoadError: If audio cannot be loaded
            DependencyError: If required dependencies are missing
            AlignmentError: If alignment process fails
        """
        # load audio
        if isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = self.load_audio(audio)  # (1, L)

        _start = time.time()
        try:
            emission = self.emission(waveform.to(self.device))  # (1, T, vocab_size)
        except Exception as e:
            raise AlignmentError(
                'Failed to compute acoustic features from audio',
                audio_path=str(audio) if not isinstance(audio, torch.Tensor) else 'tensor',
                context={'original_error': str(e)},
            )
        self.timings['emission'] += time.time() - _start

        try:
            import k2
        except ImportError:
            raise DependencyError('k2', install_command='pip install install-k2 && python -m install_k2')

        try:
            from lattifai_core.lattice.decode import align_segments
        except ImportError:
            raise DependencyError('lattifai_core', install_command='Contact support for lattifai_core installation')

        lattice_graph_str, final_state, acoustic_scale = lattice_graph

        _start = time.time()
        try:
            # graph
            decoding_graph = k2.Fsa.from_str(lattice_graph_str, acceptor=False)
            decoding_graph.requires_grad_(False)
            decoding_graph = k2.arc_sort(decoding_graph)
            decoding_graph.skip_id = int(final_state)
            decoding_graph.return_id = int(final_state + 1)
        except Exception as e:
            raise AlignmentError(
                'Failed to create decoding graph from lattice',
                context={'original_error': str(e), 'lattice_graph_length': len(lattice_graph_str)},
            )
        self.timings['decoding_graph'] += time.time() - _start

        _start = time.time()
        if self.device.type == 'mps':
            device = 'cpu'  # k2 does not support mps yet
        else:
            device = self.device

        try:
            results, labels = align_segments(
                emission.to(device) * acoustic_scale,
                decoding_graph.to(device),
                torch.tensor([emission.shape[1]], dtype=torch.int32),
                search_beam=200,
                output_beam=80,
                min_active_states=400,
                max_active_states=10000,
                subsampling_factor=1,
                reject_low_confidence=False,
            )
        except Exception as e:
            raise AlignmentError(
                'Failed to perform forced alignment',
                audio_path=str(audio) if not isinstance(audio, torch.Tensor) else 'tensor',
                context={'original_error': str(e), 'emission_shape': list(emission.shape), 'device': str(device)},
            )
        self.timings['align_segments'] += time.time() - _start

        channel = 0
        return emission, results, labels, 0.02, 0.0, channel  # frame_shift=20ms, offset=0.0s
