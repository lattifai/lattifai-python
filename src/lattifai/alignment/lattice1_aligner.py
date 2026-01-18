"""Lattice-1 Aligner implementation."""

from typing import Any, List, Optional, Tuple, Union

import colorful
import numpy as np

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import AlignmentConfig
from lattifai.errors import (
    AlignmentError,
    LatticeDecodingError,
    LatticeEncodingError,
)
from lattifai.utils import _resolve_model_path, safe_print

from .lattice1_worker import _load_worker
from .text_align import TextAlignResult
from .tokenizer import _load_tokenizer

ClientType = Any


def _extract_text_for_error(supervisions: Union[list, tuple]) -> str:
    """Extract text from supervisions for error messages."""
    if not supervisions:
        return ""
    # TextAlignResult is a tuple: (caption_sups, transcript_sups, ...)
    if isinstance(supervisions, tuple):
        supervisions = supervisions[0] or supervisions[1] or []
    return " ".join(s.text for s in supervisions if s and s.text)


class Lattice1Aligner(object):
    """Synchronous LattifAI client with config-driven architecture."""

    def __init__(
        self,
        config: AlignmentConfig,
    ) -> None:
        self.config = config

        if config.client_wrapper is None:
            raise ValueError("AlignmentConfig.client_wrapper is not set. It must be initialized by the client.")

        client_wrapper = config.client_wrapper
        # Resolve model path using configured model hub
        model_path = _resolve_model_path(config.model_name, getattr(config, "model_hub", "huggingface"))

        self.tokenizer = _load_tokenizer(
            client_wrapper, model_path, config.model_name, config.device, model_hub=config.model_hub
        )
        self.worker = _load_worker(model_path, config.device, config)

        self.frame_shift = self.worker.frame_shift

    def emission(self, ndarray: np.ndarray) -> np.ndarray:
        """Generate emission probabilities from audio ndarray.

        Args:
            ndarray: Audio data as numpy array of shape (1, T) or (C, T)

        Returns:
            Emission numpy array of shape (1, T, vocab_size)
        """
        return self.worker.emission(ndarray)

    def separate(self, audio: np.ndarray) -> np.ndarray:
        """Separate audio using separator model.

        Args:
            audio: np.ndarray object containing the audio to separate, shape (1, T)

        Returns:
            Separated audio as numpy array

        Raises:
            RuntimeError: If separator model is not available
        """
        if self.worker.separator_ort is None:
            raise RuntimeError("Separator model not available. separator.onnx not found in model path.")
        # Run separator model
        separator_output = self.worker.separator_ort.run(
            None,
            {"audios": audio},
        )
        return separator_output[0]

    def alignment(
        self,
        audio: AudioData,
        supervisions: Union[List[Supervision], TextAlignResult],
        split_sentence: Optional[bool] = False,
        return_details: Optional[bool] = False,
        emission: Optional[np.ndarray] = None,
        offset: float = 0.0,
        verbose: bool = True,
    ) -> Tuple[List[Supervision], List[Supervision]]:
        """
        Perform alignment on audio and supervisions.

        Args:
            audio: Audio file path
            supervisions: List of supervision segments to align
            split_sentence: Enable sentence re-splitting

        Returns:
            Tuple of (supervisions, alignments)

        Raises:
            LatticeEncodingError: If lattice graph generation fails
            AlignmentError: If audio alignment fails
            LatticeDecodingError: If lattice decoding fails
        """
        # Step 2: Create lattice graph
        if verbose:
            safe_print(colorful.cyan("🔗 Step 2: Creating lattice graph from segments"))
        try:
            supervisions, lattice_id, lattice_graph = self.tokenizer.tokenize(
                supervisions, split_sentence=split_sentence, boost=self.config.boost
            )
            if verbose:
                safe_print(colorful.green(f"         ✓ Generated lattice graph with ID: {lattice_id}"))
        except Exception as e:
            text_content = _extract_text_for_error(supervisions)
            raise LatticeEncodingError(text_content, original_error=e)

        # Step 3: Search lattice graph
        if verbose:
            safe_print(colorful.cyan(f"🔍 Step 3: Searching lattice graph with media: {audio}"))
            if audio.streaming_mode:
                safe_print(
                    colorful.yellow(
                        f"         ⚡Using streaming mode with {audio.streaming_chunk_secs}s (chunk duration)"
                    )
                )
        try:
            lattice_results = self.worker.alignment(
                audio,
                lattice_graph,
                emission=emission,
                offset=offset,
            )
            if verbose:
                safe_print(colorful.green("         ✓ Lattice search completed"))
        except Exception as e:
            raise AlignmentError(
                f"Audio alignment failed for {audio}",
                media_path=str(audio),
                context={"original_error": str(e)},
            )

        # Step 4: Decode lattice results
        if verbose:
            safe_print(colorful.cyan("🎯 Step 4: Decoding lattice results to aligned segments"))
        try:
            alignments = self.tokenizer.detokenize(
                lattice_id,
                lattice_results,
                supervisions=supervisions,
                return_details=return_details,
                start_margin=self.config.start_margin,
                end_margin=self.config.end_margin,
            )
            if verbose:
                safe_print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
        except LatticeDecodingError:
            safe_print(colorful.red("         x Failed to decode lattice alignment results"))
            raise
        except Exception as e:
            safe_print(colorful.red("         x Failed to decode lattice alignment results"))
            raise LatticeDecodingError(lattice_id, original_error=e)

        return (supervisions, alignments)

    def profile(self) -> None:
        """Print profiling statistics."""
        self.worker.profile()
