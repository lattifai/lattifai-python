"""Helpers for chained merge-retry of failed lattice alignment segments.

Used by the transcription-strategy main loop in ``LattifAI.alignment``: when
a single segment's lattice decode fails (HTTP 500 retry exhausted, malformed
acoustic match, …), the segment is queued and the main loop continues. Once
the main loop finishes, this module's ``chained_merge_retry`` runs through
the failures, expanding each one's window into successful neighbours and
re-aligning the merged span. The merged alignment then covers the original
failure(s) plus the neighbour(s) that got pulled in.

Only the transcription-strategy path is supported — segment 2-tuple-of-tuple
shape is ``(start, end, match_tuple, skipalign)`` where ``match_tuple`` is
``[sub_align, asr_align, aligned, ts, chunk]`` (see ``text_align.py``).
Other strategies have different segment shapes and are routed around.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional, Tuple

from lattifai.caption import Supervision
from lattifai.errors import LatticeDecodingError

# Segment shape: (start, end, _supervisions, skipalign). Re-stating it here as
# a type alias keeps callers and helpers in sync without importing the full
# TextAlignResult definition (which lives in lattifai.alignment.text_align).
Segment = Tuple[float, float, list, bool]


class SegmentResult(NamedTuple):
    """One main-loop result. ``status`` is 'ok' | 'fail' | 'skip'."""

    idx: int
    status: str
    supervisions: Optional[List[Supervision]]
    alignments: Optional[List[Supervision]]
    exception: Optional[Exception]


class MergeResult(NamedTuple):
    """One successful merge-retry result. ``covered_indices`` enumerates the
    segments (failed + neighbours pulled in) whose original results are now
    replaced by this merged alignment."""

    covered_indices: List[int]
    supervisions: List[Supervision]
    alignments: List[Supervision]


def merge_transcription_segments(segs: List[Segment]) -> Segment:
    """Combine N consecutive transcription-strategy segments into one.

    ``sub_align`` and ``asr_align`` concat; quality / timestamp / chunk
    fields are dropped because the merged segment will be re-aligned and
    detokenize doesn't read them on the diff path.
    """
    if not segs:
        raise ValueError("merge_transcription_segments requires at least 1 segment")
    sub_align = []
    asr_align = []
    for s in segs:
        sub_align.extend(s[2][0])
        asr_align.extend(s[2][1])
    return (
        min(s[0] for s in segs),
        max(s[1] for s in segs),
        [sub_align, asr_align, None, None, None],
        False,
    )


# Single-segment alignment callback signature: takes a Segment and runs the
# full per-segment alignment (audio slice + emission + aligner.alignment + …),
# returning (supervisions, alignments) or raising LatticeDecodingError.
AlignFn = Callable[[Segment], Tuple[List[Supervision], List[Supervision]]]


def chained_merge_retry(
    segments: List[Segment],
    results: List[SegmentResult],
    align_fn: AlignFn,
    *,
    warn_fn: Callable[[str], None] = lambda msg: None,
) -> List[SegmentResult]:
    """In-place rewrite of ``results`` so every failed segment is covered.

    Strategy (per user spec):
      1. For each failed segment, prefer expanding into the NEXT neighbour first.
         If that fails (or hits the episode end), try PREV.
      2. Chained: when the immediate neighbour itself fails, pull in one more
         segment in the same direction and retry. No upper bound — recurses to
         the episode boundary if needed.
      3. Fallback: if both directions exhaust to boundaries, mark the failed
         segment as 'ok' but populate from the original sub_align supervisions
         (raw caption timing) so downstream code still has data.

    Returns the same list (mutated in place); also returns it for callers that
    prefer a functional style.
    """
    n = len(segments)
    for fail_idx in range(n):
        if results[fail_idx].status != "fail":
            continue

        merged: Optional[MergeResult] = None
        for direction in ("next", "prev"):
            merged = _try_chained_direction(
                fail_idx,
                direction,
                segments,
                results,
                align_fn,
                warn_fn,
            )
            if merged is not None:
                break

        if merged is not None:
            primary = merged.covered_indices[0]
            for covered_idx in merged.covered_indices:
                # First covered index carries the new alignment; the rest are
                # marked 'ok' with empty lists so the final flatten skips them
                # (deduplication on the output side).
                if covered_idx == primary:
                    results[covered_idx] = SegmentResult(
                        idx=covered_idx,
                        status="ok",
                        supervisions=merged.supervisions,
                        alignments=merged.alignments,
                        exception=None,
                    )
                else:
                    results[covered_idx] = SegmentResult(
                        idx=covered_idx,
                        status="ok",
                        supervisions=[],
                        alignments=[],
                        exception=None,
                    )
        else:
            # Boundary-exhausted fallback: use the original caption sub_align
            # (raw VTT/MD timing). Better than dropping the segment entirely.
            warn_fn(
                f"  ⚠️  Segment {fail_idx + 1} unrecoverable after merge retry to boundary; "
                f"using caption.supervisions raw timing as fallback"
            )
            fallback = segments[fail_idx][2][0]  # sub_align
            results[fail_idx] = SegmentResult(
                idx=fail_idx,
                status="ok",
                supervisions=list(fallback),
                alignments=list(fallback),
                exception=None,
            )

    return results


def _try_chained_direction(
    fail_idx: int,
    direction: str,
    segments: List[Segment],
    results: List[SegmentResult],
    align_fn: AlignFn,
    warn_fn: Callable[[str], None],
) -> Optional[MergeResult]:
    """Expand the failed segment's window in one direction until success or
    boundary. Returns ``None`` if we hit the episode boundary without success.
    """
    n = len(segments)
    covered: List[int] = [fail_idx]
    cursor = fail_idx
    step = 1 if direction == "next" else -1

    while True:
        nb_idx = cursor + step
        if not (0 <= nb_idx < n):
            return None  # boundary

        # Skip neighbours that are themselves 'skip' (skipalign=True) — they
        # don't go through lattice alignment and shouldn't be pulled into a
        # merge window.
        if results[nb_idx].status == "skip":
            cursor = nb_idx
            continue

        covered.append(nb_idx)
        sorted_covered = sorted(covered)
        merged_seg = merge_transcription_segments([segments[i] for i in sorted_covered])

        try:
            sup_out, ali_out = align_fn(merged_seg)
            warn_fn(
                f"  ↻ Segment {fail_idx + 1} recovered by merging "
                f"{len(sorted_covered)} segments (idx {sorted_covered[0] + 1}-{sorted_covered[-1] + 1})"
            )
            return MergeResult(
                covered_indices=sorted_covered,
                supervisions=sup_out,
                alignments=ali_out,
            )
        except LatticeDecodingError:
            cursor = nb_idx
            continue
