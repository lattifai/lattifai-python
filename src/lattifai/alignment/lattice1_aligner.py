"""Lattice-1 Aligner implementation."""

from typing import Any, List, Optional, Tuple

import colorful
import numpy as np
import torch

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
from .tokenizer import _load_tokenizer

ClientType = Any


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
        model_path = _resolve_model_path(config.model_name)

        self.tokenizer = _load_tokenizer(client_wrapper, model_path, config.model_name, config.device)
        self.worker = _load_worker(model_path, config.device, config)

        self.frame_shift = self.worker.frame_shift

    def emission(self, ndarray: np.ndarray) -> torch.Tensor:
        """Generate emission probabilities from audio ndarray.

        Args:
            ndarray: Audio data as numpy array of shape (1, T) or (C, T)

        Returns:
            Emission tensor of shape (1, T, vocab_size)
        """
        return self.worker.emission(ndarray)

    def alignment(
        self,
        audio: AudioData,
        supervisions: List[Supervision],
        split_sentence: Optional[bool] = False,
        return_details: Optional[bool] = False,
        emission: Optional[torch.Tensor] = None,
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
        try:
            if verbose:
                safe_print(colorful.cyan("🔗 Step 2: Creating lattice graph from segments"))
            try:
                supervisions, lattice_id, lattice_graph = self.tokenizer.tokenize(
                    supervisions, split_sentence=split_sentence
                )
                if verbose:
                    safe_print(colorful.green(f"         ✓ Generated lattice graph with ID: {lattice_id}"))
            except Exception as e:
                text_content = " ".join([sup.text for sup in supervisions]) if supervisions else ""
                raise LatticeEncodingError(text_content, original_error=e)

            if verbose:
                safe_print(colorful.cyan(f"🔍 Step 3: Searching lattice graph with media: {audio}"))
                if audio.streaming_chunk_secs:
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

            if verbose:
                safe_print(colorful.cyan("🎯 Step 4: Decoding lattice results to aligned segments"))
            try:
                alignments = self.tokenizer.detokenize(
                    lattice_id, lattice_results, supervisions=supervisions, return_details=return_details
                )
                if verbose:
                    safe_print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
            except LatticeDecodingError as e:
                safe_print(colorful.red("         x Failed to decode lattice alignment results"))
                raise e
            except Exception as e:
                safe_print(colorful.red("         x Failed to decode lattice alignment results"))
                raise LatticeDecodingError(lattice_id, original_error=e)

            return (supervisions, alignments)

        except (LatticeEncodingError, AlignmentError, LatticeDecodingError):
            raise
        except Exception as e:
            raise e
