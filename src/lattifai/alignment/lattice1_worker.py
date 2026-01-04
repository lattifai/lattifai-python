import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import colorful
import numpy as np
import onnxruntime as ort
from lhotse.utils import Pathlike
from tqdm import tqdm

from lattifai.audio2 import AudioData
from lattifai.errors import AlignmentError, DependencyError, ModelLoadError
from lattifai.utils import safe_print


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

        # Get vocab_size from model output
        self.vocab_size = self.acoustic_ort.get_outputs()[0].shape[-1]

        # get input_names
        input_names = [inp.name for inp in self.acoustic_ort.get_inputs()]
        assert "audios" in input_names, f"Input name audios not found in {input_names}"

        # Initialize separator if available
        separator_model_path = Path(model_path) / "separator.onnx"
        if separator_model_path.exists():
            try:
                self.separator_ort = ort.InferenceSession(
                    str(separator_model_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            except Exception as e:
                raise ModelLoadError(f"separator model from {model_path}", original_error=e)
        else:
            self.separator_ort = None

        self.timings = defaultdict(lambda: 0.0)

    @property
    def frame_shift(self) -> float:
        return 0.02  # 20 ms

    def emission(self, ndarray: np.ndarray, acoustic_scale: float = 1.0) -> np.ndarray:
        """Generate emission probabilities from audio ndarray.

        Args:
            ndarray: Audio data as numpy array of shape (1, T) or (C, T)

        Returns:
            Emission numpy array of shape (1, T, vocab_size)
        """
        _start = time.time()

        if ndarray.shape[1] < 160:
            ndarray = np.pad(ndarray, ((0, 0), (0, 320 - ndarray.shape[1])), mode="constant")

        CHUNK_SIZE = 60 * 16000  # 60 seconds
        total_samples = ndarray.shape[1]

        if total_samples > CHUNK_SIZE:
            frame_samples = int(16000 * self.frame_shift)
            emissions = np.empty((1, total_samples // frame_samples + 1, self.vocab_size), dtype=np.float32)
            for start in range(0, total_samples, CHUNK_SIZE):
                chunk = ndarray[:, start : start + CHUNK_SIZE]
                if chunk.shape[1] < 160:
                    chunk = np.pad(chunk, ((0, 0), (0, 320 - chunk.shape[1])), mode="constant")

                emission_out = self.acoustic_ort.run(None, {"audios": chunk})[0]
                if acoustic_scale != 1.0:
                    emission_out *= acoustic_scale
                sf = start // frame_samples  # start frame
                lf = sf + emission_out.shape[1]  # last frame
                emissions[0, sf:lf, :] = emission_out
            emissions[:, lf:, :] = 0.0
        else:
            emission_out = self.acoustic_ort.run(
                None,
                {
                    "audios": ndarray,
                },
            )  # (1, T, vocab_size) numpy
            emissions = emission_out[0]

            if acoustic_scale != 1.0:
                emissions *= acoustic_scale

        self.timings["emission"] += time.time() - _start
        return emissions  # (1, T, vocab_size) numpy

    def alignment(
        self,
        audio: AudioData,
        lattice_graph: Tuple[str, int, float],
        emission: Optional[np.ndarray] = None,
        offset: float = 0.0,
    ) -> Dict[str, Any]:
        """Process audio with LatticeGraph.

        Args:
            audio: AudioData object
            lattice_graph: LatticeGraph data
            emission: Pre-computed emission numpy array (ignored if streaming=True)
            offset: Time offset for the audio
            streaming: If True, use streaming mode for memory-efficient processing

        Returns:
            Processed LatticeGraph

        Raises:
            AudioLoadError: If audio cannot be loaded
            DependencyError: If required dependencies are missing
            AlignmentError: If alignment process fails
        """
        import k2py as k2

        lattice_graph_str, final_state, acoustic_scale = lattice_graph

        _start = time.time()
        try:
            # Create decoding graph using k2py
            graph_dict = k2.CreateFsaVecFromStr(lattice_graph_str, int(final_state), False)
            decoding_graph = graph_dict["fsa"]
            aux_labels = graph_dict["aux_labels"]
        except Exception as e:
            raise AlignmentError(
                "Failed to create decoding graph from lattice",
                context={"original_error": str(e), "lattice_graph_length": len(lattice_graph_str)},
            )
        self.timings["decoding_graph"] += time.time() - _start

        _start = time.time()

        # Get beam search parameters from config or use defaults
        search_beam = self.alignment_config.search_beam or 200
        output_beam = self.alignment_config.output_beam or 80
        min_active_states = self.alignment_config.min_active_states or 400
        max_active_states = self.alignment_config.max_active_states or 10000

        if emission is None and audio.streaming_mode:
            # Initialize OnlineDenseIntersecter for streaming
            intersecter = k2.OnlineDenseIntersecter(
                decoding_graph,
                aux_labels,
                float(search_beam),
                float(output_beam),
                int(min_active_states),
                int(max_active_states),
            )

            # Streaming mode
            total_duration = audio.duration
            total_minutes = int(total_duration / 60.0)

            with tqdm(
                total=total_minutes,
                desc=f"Processing audio ({total_minutes} min)",
                unit="min",
                unit_scale=False,
                unit_divisor=1,
            ) as pbar:
                for chunk in audio.iter_chunks():
                    chunk_emission = self.emission(chunk.ndarray, acoustic_scale=acoustic_scale)
                    intersecter.decode(chunk_emission[0])

                    # Update progress
                    chunk_duration = int(chunk.duration / 60.0)
                    pbar.update(chunk_duration)

            emission_result = None
            # Get results from intersecter
            results, labels = intersecter.finish()
        else:
            # Batch mode
            if emission is None:
                emission = self.emission(audio.ndarray, acoustic_scale=acoustic_scale)  # (1, T, vocab_size)
            else:
                if acoustic_scale != 1.0:
                    emission *= acoustic_scale
            # Use AlignSegments directly
            results, labels = k2.AlignSegments(
                graph_dict,
                emission[0],  # Pass the prepared scores
                float(search_beam),
                float(output_beam),
                int(min_active_states),
                int(max_active_states),
            )
            emission_result = emission

        self.timings["align_segments"] += time.time() - _start

        channel = 0
        return emission_result, results, labels, self.frame_shift, offset, channel  # frame_shift=20ms

    def profile(self) -> None:
        """Print formatted profiling statistics."""
        if not self.timings:
            return

        safe_print(colorful.bold(colorful.cyan("\n⏱️  Alignment Profiling")))
        safe_print(colorful.gray("─" * 44))
        safe_print(
            f"{colorful.bold('Phase'.ljust(21))} "
            f"{colorful.bold('Time'.ljust(12))} "
            f"{colorful.bold('Percent'.rjust(8))}"
        )
        safe_print(colorful.gray("─" * 44))

        total_time = sum(self.timings.values())

        # Sort by duration descending
        sorted_stats = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for name, duration in sorted_stats:
            percentage = (duration / total_time * 100) if total_time > 0 else 0.0
            # Name: Cyan, Time: Yellow, Percent: Gray
            safe_print(
                f"{name:<20} "
                f"{colorful.yellow(f'{duration:7.4f}s'.ljust(12))} "
                f"{colorful.gray(f'{percentage:.2f}%'.rjust(8))}"
            )

        safe_print(colorful.gray("─" * 44))
        # Pad "Total Time" before coloring to ensure correct alignment (ANSI codes don't count for width)
        safe_print(
            f"{colorful.bold('Total Time'.ljust(20))} "
            f"{colorful.bold(colorful.yellow(f'{total_time:7.4f}s'.ljust(12)))}\n"
        )


def _load_worker(model_path: str, device: str, config: Optional[Any] = None) -> Lattice1Worker:
    """Instantiate lattice worker with consistent error handling."""
    try:
        return Lattice1Worker(model_path, device=device, num_threads=8, config=config)
    except Exception as e:
        raise ModelLoadError(f"worker from {model_path}", original_error=e)
