import json
import time
from collections import defaultdict
from typing import Any, BinaryIO, Dict, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from lhotse import FbankConfig
from lhotse.features.kaldi.layers import Wav2LogFilterBank
from lhotse.utils import Pathlike


class Lattice1AlphaWorker:
    """Worker for processing audio with LatticeGraph."""

    def __init__(self, model_path: Pathlike, device: str = 'cpu', num_threads: int = 8) -> None:
        if device != 'cpu':
            raise NotImplementedError(f'Only cpu is supported for now, got device={device}.')
        self.config = json.load(open(f'{model_path}/config.json'))

        # SessionOptions
        sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads  # CPU cores
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.add_session_config_entry('session.intra_op.allow_spinning', '0')

        providers = []
        if device.startswith('cuda') or ort.get_all_providers().count('CUDAExecutionProvider') > 0:
            providers.append('CUDAExecutionProvider')
        self.acoustic_ort = ort.InferenceSession(
            f'{model_path}/acoustic_opt.onnx',
            sess_options,
            providers=providers + ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        )
        config = FbankConfig(num_mel_bins=80, device=device, snip_edges=False)
        config_dict = config.to_dict()
        config_dict.pop('device')
        self.extractor = Wav2LogFilterBank(**config_dict).to(device).eval()

        self.device = torch.device(device)
        self.timings = defaultdict(lambda: 0.0)

    @torch.inference_mode()
    def emission(self, audio: torch.Tensor) -> torch.Tensor:
        _start = time.time()
        # audio -> features -> emission
        features = self.extractor(audio)  # (1, T, D)
        ort_inputs = {
            'features': features.cpu().numpy(),
            'feature_lengths': np.array([features.size(1)], dtype=np.int64),
        }
        emission = self.acoustic_ort.run(None, ort_inputs)[0]  # (1, T, vocab_size) numpy
        self.timings['emission'] += time.time() - _start
        return torch.from_numpy(emission).to(self.device)  # (1, T, vocab_size) torch

    def load_audio(self, audio: Union[Pathlike, BinaryIO]) -> Tuple[torch.Tensor, int]:
        # load audio
        waveform, sample_rate = torchaudio.load(audio, channels_first=True)
        if waveform.size(0) > 1:  # TODO: support choose channel
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.config['sample_rate']:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config['sample_rate'])
        return waveform

    def alignment(
        self, audio: Union[Union[Pathlike, BinaryIO], torch.tensor], lattice_graph: Tuple[str, int, float]
    ) -> Dict[str, Any]:
        """Process audio with LatticeGraph.

        Args:
            audio: Audio file path or binary data
            lattice_graph: LatticeGraph data

        Returns:
            Processed LatticeGraph
        """
        # load audio
        if isinstance(audio, torch.Tensor):
            waveform = audio
        else:
            waveform = self.load_audio(audio)  # (1, L)

        _start = time.time()
        emission = self.emission(waveform.to(self.device))  # (1, T, vocab_size)
        self.timings['emission'] += time.time() - _start

        import k2
        from lattifai_core.lattice.decode import align_segments

        lattice_graph_str, final_state, acoustic_scale = lattice_graph

        _start = time.time()
        # graph
        decoding_graph = k2.Fsa.from_str(lattice_graph_str, acceptor=False)
        decoding_graph.requires_grad_(False)
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph.skip_id = int(final_state)
        decoding_graph.return_id = int(final_state + 1)
        self.timings['decoding_graph'] += time.time() - _start

        _start = time.time()
        results, labels = align_segments(
            emission.to(self.device) * acoustic_scale,
            decoding_graph.to(self.device),
            torch.tensor([emission.shape[1]], dtype=torch.int32),
            search_beam=100,
            output_beam=40,
            min_active_states=200,
            max_active_states=10000,
            subsampling_factor=1,
            reject_low_confidence=False,
        )
        self.timings['align_segments'] += time.time() - _start

        channel = 0
        return emission, results, labels, 0.02, 0.0, channel  # frame_shift=20ms, offset=0.0s
