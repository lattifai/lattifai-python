import json
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from lhotse import FbankConfig
from lhotse.features.kaldi.layers import Wav2LogFilterBank
from lhotse.utils import Pathlike
from tqdm import tqdm

from lattifai.audio2 import AudioData
from lattifai.errors import AlignmentError, DependencyError, ModelLoadError


class Lattice1Worker:
    """Worker for processing audio with LatticeGraph."""

    def __init__(
        self, model_path: Pathlike, device: str = "cpu", num_threads: int = 8, config: Optional[Any] = None
    ) -> None:
        try:
            self.model_config = json.load(open(f"{model_path}/config.json"))
        except Exception as e:
            raise ModelLoadError(f"config from {model_path}", original_error=e)

        # Store alignment config with beam search parameters
        self.alignment_config = config

        # SessionOptions
        sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads  # CPU cores
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        acoustic_model_path = f"{model_path}/acoustic_opt.onnx"

        providers = []
        all_providers = ort.get_all_providers()
        if device.startswith("cuda") and all_providers.count("CUDAExecutionProvider") > 0:
            providers.append("CUDAExecutionProvider")
        if "MPSExecutionProvider" in all_providers:
            providers.append("MPSExecutionProvider")
        if "CoreMLExecutionProvider" in all_providers:
            if "quant" in acoustic_model_path:
                # NOTE: CPUExecutionProvider is faster for quantized models
                pass
            else:
                providers.append("CoreMLExecutionProvider")

        try:
            self.acoustic_ort = ort.InferenceSession(
                acoustic_model_path,
                sess_options,
                providers=providers + ["CPUExecutionProvider"],
            )
        except Exception as e:
            raise ModelLoadError(f"acoustic model from {model_path}", original_error=e)

        # get input_names
        input_names = [inp.name for inp in self.acoustic_ort.get_inputs()]
        if "audios" not in input_names:
            try:
                config = FbankConfig(num_mel_bins=80, device=device, snip_edges=False)
                config_dict = config.to_dict()
                config_dict.pop("device")
                self.extractor = Wav2LogFilterBank(**config_dict).to(device).eval()
            except Exception as e:
                raise ModelLoadError(f"feature extractor for device {device}", original_error=e)
        else:
            self.extractor = None  # ONNX model includes feature extractor

        self.device = torch.device(device)
        self.timings = defaultdict(lambda: 0.0)

    @property
    def frame_shift(self) -> float:
        return 0.02  # 20 ms

    @torch.inference_mode()
    def emission(self, ndarray: np.ndarray, acoustic_scale: float = 1.0, device: Optional[str] = None) -> torch.Tensor:
        """Generate emission probabilities from audio ndarray.

        Args:
            ndarray: Audio data as numpy array of shape (1, T) or (C, T)

        Returns:
            Emission tensor of shape (1, T, vocab_size)
        """
        _start = time.time()
        if self.extractor is not None:
            # audio -> features -> emission
            audio = torch.from_numpy(ndarray).to(self.device)
            if audio.shape[1] < 160:
                audio = torch.nn.functional.pad(audio, (0, 320 - audio.shape[1]))
            features = self.extractor(audio)  # (1, T, D)
            if features.shape[1] > 6000:
                emissions = []
                for start in range(0, features.size(1), 6000):
                    _features = features[:, start : start + 6000, :]
                    ort_inputs = {
                        "features": _features.cpu().numpy(),
                        "feature_lengths": np.array([_features.size(1)], dtype=np.int64),
                    }
                    emission = self.acoustic_ort.run(None, ort_inputs)[0]  # (1, T, vocab_size) numpy
                    emissions.append(emission)
                emission = torch.cat(
                    [torch.from_numpy(emission).to(device or self.device) for emission in emissions], dim=1
                )  # (1, T, vocab_size)
                del emissions
            else:
                ort_inputs = {
                    "features": features.cpu().numpy(),
                    "feature_lengths": np.array([features.size(1)], dtype=np.int64),
                }
                emission = self.acoustic_ort.run(None, ort_inputs)[0]  # (1, T, vocab_size) numpy
                emission = torch.from_numpy(emission).to(device or self.device)
        else:
            if ndarray.shape[1] < 160:
                ndarray = np.pad(ndarray, ((0, 0), (0, 320 - ndarray.shape[1])), mode="constant")

            CHUNK_SIZE = 60 * 16000  # 60 seconds
            if ndarray.shape[1] > CHUNK_SIZE:
                emissions = []
                for start in range(0, ndarray.shape[1], CHUNK_SIZE):
                    emission = self.acoustic_ort.run(
                        None,
                        {
                            "audios": ndarray[:, start : start + CHUNK_SIZE],
                        },
                    )  # (1, T, vocab_size) numpy
                    emissions.append(emission[0])

                emission = torch.cat(
                    [torch.from_numpy(emission).to(device or self.device) for emission in emissions], dim=1
                )  # (1, T, vocab_size)
                del emissions
            else:
                emission = self.acoustic_ort.run(
                    None,
                    {
                        "audios": ndarray,
                    },
                )  # (1, T, vocab_size) numpy
                emission = torch.from_numpy(emission[0]).to(device or self.device)

        if acoustic_scale != 1.0:
            emission = emission.mul_(acoustic_scale)

        self.timings["emission"] += time.time() - _start
        return emission  # (1, T, vocab_size) torch

    def alignment(
        self,
        audio: AudioData,
        lattice_graph: Tuple[str, int, float],
        emission: Optional[torch.Tensor] = None,
        offset: float = 0.0,
    ) -> Dict[str, Any]:
        """Process audio with LatticeGraph.

        Args:
            audio: AudioData object
            lattice_graph: LatticeGraph data
            emission: Pre-computed emission tensor (ignored if streaming=True)
            offset: Time offset for the audio
            streaming: If True, use streaming mode for memory-efficient processing

        Returns:
            Processed LatticeGraph

        Raises:
            AudioLoadError: If audio cannot be loaded
            DependencyError: If required dependencies are missing
            AlignmentError: If alignment process fails
        """
        try:
            import k2
        except ImportError:
            raise DependencyError("k2", install_command="pip install install-k2 && python -m install_k2")

        try:
            from lattifai_core.lattice.decode import align_segments
        except ImportError:
            raise DependencyError("lattifai_core", install_command="Contact support for lattifai_core installation")

        lattice_graph_str, final_state, acoustic_scale = lattice_graph

        _start = time.time()
        try:
            # Create decoding graph
            decoding_graph = k2.Fsa.from_str(lattice_graph_str, acceptor=False)
            decoding_graph.requires_grad_(False)
            decoding_graph = k2.arc_sort(decoding_graph)
            decoding_graph.skip_id = int(final_state)
            decoding_graph.return_id = int(final_state + 1)
        except Exception as e:
            raise AlignmentError(
                "Failed to create decoding graph from lattice",
                context={"original_error": str(e), "lattice_graph_length": len(lattice_graph_str)},
            )
        self.timings["decoding_graph"] += time.time() - _start

        if self.device.type == "mps":
            device = "cpu"  # k2 does not support mps yet
        else:
            device = self.device

        _start = time.time()

        # Get beam search parameters from config or use defaults
        search_beam = self.alignment_config.search_beam or 200
        output_beam = self.alignment_config.output_beam or 80
        min_active_states = self.alignment_config.min_active_states or 400
        max_active_states = self.alignment_config.max_active_states or 10000

        if emission is None and audio.streaming_mode:
            # Streaming mode: pass emission iterator to align_segments
            # The align_segments function will automatically detect the iterator
            # and use k2.OnlineDenseIntersecter for memory-efficient processing

            def emission_iterator():
                """Generate emissions for each audio chunk with progress tracking."""
                total_duration = audio.duration
                processed_duration = 0.0
                total_minutes = int(total_duration / 60.0)

                with tqdm(
                    total=total_minutes,
                    desc=f"Processing audio ({total_minutes} min)",
                    unit="min",
                    unit_scale=False,
                    unit_divisor=1,
                ) as pbar:
                    for chunk in audio.iter_chunks():
                        chunk_emission = self.emission(chunk.ndarray, acoustic_scale=acoustic_scale, device=device)

                        # Update progress based on chunk duration in minutes
                        chunk_duration = int(chunk.duration / 60.0)
                        pbar.update(chunk_duration)
                        processed_duration += chunk_duration

                        yield chunk_emission

            # Calculate total frames for supervision_segments
            total_frames = int(audio.duration / self.frame_shift)

            results, labels = align_segments(
                emission_iterator(),  # Pass iterator for streaming
                decoding_graph.to(device),
                torch.tensor([total_frames], dtype=torch.int32),
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
                subsampling_factor=1,
                reject_low_confidence=False,
            )

            # For streaming, don't return emission tensor to save memory
            emission_result = None
        else:
            # Batch mode: compute full emission tensor and pass to align_segments
            if emission is None:
                emission = self.emission(
                    audio.ndarray, acoustic_scale=acoustic_scale, device=device
                )  # (1, T, vocab_size)
            else:
                emission = emission.to(device) * acoustic_scale

            results, labels = align_segments(
                emission,
                decoding_graph.to(device),
                torch.tensor([emission.shape[1]], dtype=torch.int32),
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
                subsampling_factor=1,
                reject_low_confidence=False,
            )

            emission_result = emission

        self.timings["align_segments"] += time.time() - _start

        channel = 0
        return emission_result, results, labels, self.frame_shift, offset, channel  # frame_shift=20ms


def _load_worker(model_path: str, device: str, config: Optional[Any] = None) -> Lattice1Worker:
    """Instantiate lattice worker with consistent error handling."""
    try:
        return Lattice1Worker(model_path, device=device, num_threads=8, config=config)
    except Exception as e:
        raise ModelLoadError(f"worker from {model_path}", original_error=e)
