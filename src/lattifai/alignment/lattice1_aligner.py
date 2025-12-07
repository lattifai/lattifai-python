"""Lattice-1 Aligner implementation."""

from typing import Any, List, Optional, Tuple

import colorful
import torch

from lattifai.audio2 import AudioData
from lattifai.caption import Supervision
from lattifai.config import AlignmentConfig
from lattifai.errors import (
    AlignmentError,
    LatticeDecodingError,
    LatticeEncodingError,
)
from lattifai.utils import _resolve_model_path

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
        self.worker = _load_worker(model_path, config.device)

        self.frame_shift = self.worker.frame_shift

    def emission(self, audio: torch.Tensor) -> torch.Tensor:
        return self.worker.emission(audio.to(self.worker.device))

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
                print(colorful.cyan("🔗 Step 2: Creating lattice graph from segments"))
            try:
                supervisions, lattice_id, lattice_graph = self.tokenizer.tokenize(
                    supervisions, split_sentence=split_sentence
                )
                if verbose:
                    print(colorful.green(f"         ✓ Generated lattice graph with ID: {lattice_id}"))
            except Exception as e:
                text_content = " ".join([sup.text for sup in supervisions]) if supervisions else ""
                raise LatticeEncodingError(text_content, original_error=e)

            if verbose:
                print(colorful.cyan(f"🔍 Step 3: Searching lattice graph with media: {audio}"))
            try:
                lattice_results = self.worker.alignment(audio, lattice_graph, emission=emission, offset=offset)
                if verbose:
                    print(colorful.green("         ✓ Lattice search completed"))
            except Exception as e:
                raise AlignmentError(
                    f"Audio alignment failed for {audio}",
                    media_path=str(audio),
                    context={"original_error": str(e)},
                )

            if verbose:
                print(colorful.cyan("🎯 Step 4: Decoding lattice results to aligned segments"))
            try:
                alignments = self.tokenizer.detokenize(
                    lattice_id, lattice_results, supervisions=supervisions, return_details=return_details
                )
                if verbose:
                    print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
            except LatticeDecodingError as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise e
            except Exception as e:
                print(colorful.red("         x Failed to decode lattice alignment results"))
                raise LatticeDecodingError(lattice_id, original_error=e)

            return (supervisions, alignments)

        except (LatticeEncodingError, AlignmentError, LatticeDecodingError):
            raise
        except Exception as e:
            raise e
