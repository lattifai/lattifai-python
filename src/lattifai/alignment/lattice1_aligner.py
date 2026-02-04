"""Lattice-1 Aligner implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union

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
                supervisions,
                split_sentence=split_sentence,
                boost=self.config.boost,
                transition_penalty=self.config.transition_penalty,
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
                check_sanity=self.config.check_sanity,
            )
            if verbose:
                safe_print(colorful.green(f"         ✓ Successfully aligned {len(alignments)} segments"))
            if not self.config.check_sanity:
                # Find and report low-score segments
                low_score_segments = _find_low_score_segments(alignments)
                if low_score_segments:
                    safe_print(colorful.yellow(_format_low_score_warning(low_score_segments)))
        except LatticeDecodingError as e:
            safe_print(colorful.red("         x Failed to decode lattice alignment results"))
            _alignments = self.tokenizer.detokenize(
                lattice_id,
                lattice_results,
                supervisions=supervisions,
                return_details=return_details,
                start_margin=self.config.start_margin,
                end_margin=self.config.end_margin,
                check_sanity=False,
            )
            # Find low-score segments to provide helpful error context
            low_score_segments = _find_low_score_segments(_alignments)
            del _alignments
            if low_score_segments:
                warning_str = _format_low_score_warning(low_score_segments)
                raise LatticeDecodingError(
                    lattice_id,
                    message=colorful.yellow("Media-text mismatch detected:\n" + warning_str),
                    skip_help=True,
                )
            else:
                raise e
        except Exception as e:
            safe_print(colorful.red("         x Failed to decode lattice alignment results"))
            raise LatticeDecodingError(lattice_id, original_error=e)

        return (supervisions, alignments)

    def profile(self) -> None:
        """Print profiling statistics."""
        self.worker.profile()


def _is_event_segment(text: str) -> bool:
    """Check if text is an event marker like [MUSIC], [Applause], [Writes equation]."""
    text = text.strip()
    return text.startswith("[") and text.endswith("]")


def _detect_score_anomalies(
    alignments: List[Supervision],
    drop_threshold: float = 0.08,
    window_size: int = 5,
) -> Optional[Dict[str, Any]]:
    """Detect score anomalies indicating alignment mismatch.

    Compares average of window_size segments before vs after each position.
    When the drop is significant, it indicates the audio doesn't match
    the text starting at that position.

    Event segments like [MUSIC], [Applause] are excluded from scoring as they
    naturally have low alignment scores.

    Args:
        alignments: List of aligned supervisions with scores
        drop_threshold: Minimum drop between before/after averages to trigger
        window_size: Number of segments to average on each side

    Returns:
        Dict with anomaly info if found, None otherwise
    """
    # Build (original_index, score) pairs, excluding events and None scores
    indexed_scores = [
        (i, s.score) for i, s in enumerate(alignments) if s.score is not None and not _is_event_segment(s.text)
    ]
    if len(indexed_scores) < window_size * 2:
        return None

    scores = [score for _, score in indexed_scores]
    orig_indices = [idx for idx, _ in indexed_scores]

    for i in range(window_size, len(scores) - window_size):
        before_avg = np.mean(scores[i - window_size : i])
        after_avg = np.mean(scores[i : i + window_size])
        drop = before_avg - after_avg

        # Trigger: significant drop between before and after windows
        if drop > drop_threshold:
            # Find the exact mutation point (largest single-step drop)
            max_drop = 0
            filtered_mutation_idx = i
            for j in range(i - 1, min(i + window_size, len(scores) - 1)):
                single_drop = scores[j] - scores[j + 1]
                if single_drop > max_drop:
                    max_drop = single_drop
                    filtered_mutation_idx = j + 1

            # Map back to original alignments index
            mutation_idx = orig_indices[filtered_mutation_idx]

            # Segments: last normal + anomaly segments
            last_normal = alignments[mutation_idx - 1] if mutation_idx > 0 else None
            anomaly_segments = [
                alignments[j] for j in range(mutation_idx, min(mutation_idx + window_size, len(alignments)))
            ]

            return {
                "mutation_index": mutation_idx,
                "before_avg": round(before_avg, 4),
                "after_avg": round(after_avg, 4),
                "window_drop": round(drop, 4),
                "mutation_drop": round(max_drop, 4),
                "last_normal": last_normal,
                "segments": anomaly_segments,
            }

    return None


def _format_anomaly_warning(anomaly: Dict[str, Any]) -> str:
    """Format anomaly detection result as warning message."""
    lines = [
        f"⚠️  Score anomaly detected at segment #{anomaly['mutation_index']}",
        f"    Window avg: {anomaly['before_avg']:.4f} → {anomaly['after_avg']:.4f} (drop: {anomaly['window_drop']:.4f})",  # noqa: E501
        f"    Mutation drop: {anomaly['mutation_drop']:.4f}",
        "",
    ]

    # Show last normal segment
    if anomaly.get("last_normal"):
        seg = anomaly["last_normal"]
        text_preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        lines.append(f'    [{seg.start:.2f}s-{seg.end:.2f}s] score={seg.score:.4f} "{text_preview}"')

    # Separator - mutation point
    lines.append("    " + "─" * 60)
    lines.append(f"    ⬇️  MUTATION: The following {len(anomaly['segments'])}+ segments don't match audio")
    lines.append("    " + "─" * 60)

    # Show anomaly segments
    for seg in anomaly["segments"]:
        text_preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        lines.append(f'    [{seg.start:.2f}s-{seg.end:.2f}s] score={seg.score:.4f} "{text_preview}"')

    lines.append("")
    lines.append("    Possible causes: Transcription error, missing content, or wrong audio region")
    return "\n".join(lines)


def _find_low_score_segments(
    alignments: List[Supervision],
    threshold: float = 0.7,
) -> List[Tuple[int, Supervision]]:
    """Find segments with scores below threshold, excluding event markers.

    Args:
        alignments: List of aligned supervisions with scores
        threshold: Score threshold (segments below this are considered low)

    Returns:
        List of (index, supervision) tuples for low-score segments
    """
    return [
        (i, s)
        for i, s in enumerate(alignments)
        if s.score is not None and s.score < threshold and not _is_event_segment(s.text)
    ]


def _format_low_score_warning(low_score_segments: List[Tuple[int, Supervision]]) -> str:
    """Format low-score segments as warning message."""
    lines = [
        f"⚠️  Found {len(low_score_segments)} low-score segments (potential mismatches):",
        "",
    ]
    for idx, seg in low_score_segments:
        text_preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        lines.append(f'    #{idx} [{seg.start:.2f}s-{seg.end:.2f}s] score={seg.score:.4f} "{text_preview}"')
    return "\n".join(lines)
