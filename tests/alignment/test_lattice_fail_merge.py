"""Unit tests for `lattifai.alignment._merge`.

Covers the fail-and-merge retry helpers used by `LattifAI.alignment` on the
transcription strategy path. The `align_fn` callable is faked so no audio /
HTTP / lattice backend is touched — the tests focus on the orchestration
logic: which segments get pulled into a merge, when the next/prev direction
fires, when boundary fallback kicks in, and that the output array stays
1-to-1 with the input segments (no duplicates, no lost data).
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from lattifai.alignment._merge import (
    SegmentResult,
    chained_merge_retry,
    merge_transcription_segments,
)
from lattifai.caption import Supervision
from lattifai.errors import LatticeDecodingError


def _sup(text: str, start: float, end: float) -> Supervision:
    return Supervision(text=text, start=start, duration=end - start)


def _seg(start: float, end: float, sub_texts: List[str], asr_texts: List[str]) -> tuple:
    """Build a transcription-strategy segment tuple.

    Shape: (start, end, match_tuple, skipalign) where match_tuple is
    [sub_align, asr_align, aligned, ts, chunk].
    """
    sub = [_sup(t, start, end) for t in sub_texts]
    asr = [_sup(t, start, end) for t in asr_texts]
    return (start, end, [sub, asr, None, None, None], False)


# ---------------------------------------------------------------------------
# merge_transcription_segments
# ---------------------------------------------------------------------------


class TestMergeTranscriptionSegments:
    """Pure list-concat semantics. Times take min/max."""

    def test_concat_sub_and_asr_separately(self):
        s1 = _seg(0.0, 5.0, ["sub1"], ["asr1"])
        s2 = _seg(5.0, 10.0, ["sub2"], ["asr2a", "asr2b"])
        merged = merge_transcription_segments([s1, s2])

        assert merged[0] == 0.0
        assert merged[1] == 10.0
        sub_align, asr_align, *_ = merged[2]
        assert [s.text for s in sub_align] == ["sub1", "sub2"]
        assert [s.text for s in asr_align] == ["asr1", "asr2a", "asr2b"]
        # skipalign forced to False — merged span must always be re-aligned
        assert merged[3] is False

    def test_quality_fields_dropped(self):
        s1 = _seg(0.0, 5.0, ["a"], ["a"])
        merged = merge_transcription_segments([s1])
        # aligned / timestamp / chunk slots cleared — re-alignment overwrites
        assert merged[2][2] is None
        assert merged[2][3] is None
        assert merged[2][4] is None

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least 1 segment"):
            merge_transcription_segments([])


# ---------------------------------------------------------------------------
# chained_merge_retry — orchestration
# ---------------------------------------------------------------------------


def _make_segments(n: int) -> list:
    """n consecutive segments, 10s each, one sub_align / asr_align per side."""
    return [_seg(i * 10.0, (i + 1) * 10.0, [f"sub{i}"], [f"asr{i}"]) for i in range(n)]


def _make_results(n: int, fail_indices: set) -> list:
    """Initial Phase-1 results: failures at the given indices, others 'ok'."""
    out = []
    for i in range(n):
        if i in fail_indices:
            out.append(
                SegmentResult(
                    idx=i,
                    status="fail",
                    supervisions=None,
                    alignments=None,
                    exception=LatticeDecodingError(lattice_id=str(i)),
                )
            )
        else:
            sup = [_sup(f"sub{i}", i * 10.0, (i + 1) * 10.0)]
            out.append(
                SegmentResult(
                    idx=i,
                    status="ok",
                    supervisions=sup,
                    alignments=sup,
                    exception=None,
                )
            )
    return out


class TestChainedMergeRetry:
    """End-to-end behaviour of the retry pass."""

    def test_no_failures_is_noop(self):
        """All-ok results pass through unchanged — no align_fn calls."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices=set())
        align_fn = MagicMock()

        out = chained_merge_retry(segments, results, align_fn)

        align_fn.assert_not_called()
        assert all(r.status == "ok" for r in out)
        assert [r.idx for r in out] == [0, 1, 2]

    def test_single_failure_recovers_by_merging_next(self):
        """fail at idx=1, next neighbour pulled in. align_fn called once."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices={1})

        # align_fn succeeds on first call (merged [1, 2])
        merged_sup = [_sup("merged", 10.0, 30.0)]
        align_fn = MagicMock(return_value=(merged_sup, merged_sup))

        chained_merge_retry(segments, results, align_fn)

        align_fn.assert_called_once()
        # Inspect the merged segment passed in: should span 10s-30s
        called_seg = align_fn.call_args.args[0]
        assert called_seg[0] == 10.0
        assert called_seg[1] == 30.0

        # idx=1 carries the new alignment; idx=2 was covered, now empty
        # (deduplication happens on output flatten).
        assert results[1].status == "ok"
        assert results[1].supervisions == merged_sup
        assert results[2].status == "ok"
        assert results[2].supervisions == []  # covered, payload moved to idx=1

    def test_next_fails_then_prev_succeeds(self):
        """fail at idx=1; merge with next fails (next is also the last seg
        AND align_fn raises); falls back to prev direction successfully."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices={1})

        # First call: [1,2] merge fails. Second call: [0,1] merge succeeds.
        merged_sup = [_sup("merged", 0.0, 20.0)]
        align_fn = MagicMock(
            side_effect=[
                LatticeDecodingError(lattice_id="merge-next"),
                (merged_sup, merged_sup),
            ]
        )

        chained_merge_retry(segments, results, align_fn)

        assert align_fn.call_count == 2
        # First call covered [1, 2] (next direction)
        first_seg = align_fn.call_args_list[0].args[0]
        assert first_seg[0] == 10.0 and first_seg[1] == 30.0
        # Second call covered [0, 1] (prev direction)
        second_seg = align_fn.call_args_list[1].args[0]
        assert second_seg[0] == 0.0 and second_seg[1] == 20.0

        # Primary covered index for prev-direction merge is the lower (0).
        assert results[0].status == "ok"
        assert results[0].supervisions == merged_sup
        assert results[1].status == "ok"
        assert results[1].supervisions == []  # covered, moved to idx=0

    def test_chained_next_expands_when_immediate_neighbour_also_fails(self):
        """fail at idx=1; merging [1,2] fails too; chain extends to [1,2,3]."""
        segments = _make_segments(5)
        results = _make_results(5, fail_indices={1})

        merged_sup = [_sup("merged", 10.0, 40.0)]
        align_fn = MagicMock(
            side_effect=[
                LatticeDecodingError(lattice_id="merge-next-1"),
                (merged_sup, merged_sup),
            ]
        )

        chained_merge_retry(segments, results, align_fn)

        assert align_fn.call_count == 2
        # Second (successful) merge spans idx 1..3 → 10s-40s
        winning_seg = align_fn.call_args_list[1].args[0]
        assert winning_seg[0] == 10.0
        assert winning_seg[1] == 40.0

        assert results[1].status == "ok"
        assert results[1].supervisions == merged_sup
        assert results[2].status == "ok" and results[2].supervisions == []
        assert results[3].status == "ok" and results[3].supervisions == []

    def test_boundary_fallback_when_both_directions_exhaust(self):
        """3 segments, all 3 fail → next direction hits end, prev direction
        hits start, both exhaust → fallback to caption.sub_align timing."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices={0, 1, 2})

        # align_fn always raises — never recovers via merging
        align_fn = MagicMock(side_effect=LatticeDecodingError(lattice_id="always-fail"))
        warn_log: List[str] = []

        chained_merge_retry(
            segments,
            results,
            align_fn,
            warn_fn=warn_log.append,
        )

        # All three end up 'ok' via fallback path
        assert all(r.status == "ok" for r in results)
        # Fallback uses sub_align — sup0 / sup1 / sup2 from the original segments
        assert results[0].supervisions[0].text == "sub0"
        assert results[1].supervisions[0].text == "sub1"
        assert results[2].supervisions[0].text == "sub2"
        # Warned three times about unrecoverable fallback
        unrecoverable_warns = [w for w in warn_log if "unrecoverable" in w]
        assert len(unrecoverable_warns) == 3

    def test_failure_at_first_segment_skips_prev_direction(self):
        """fail at idx=0; next direction succeeds without needing prev."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices={0})

        merged_sup = [_sup("merged", 0.0, 20.0)]
        align_fn = MagicMock(return_value=(merged_sup, merged_sup))

        chained_merge_retry(segments, results, align_fn)

        # Only one call: merge [0, 1] via next
        align_fn.assert_called_once()
        first_seg = align_fn.call_args.args[0]
        assert first_seg[0] == 0.0 and first_seg[1] == 20.0

    def test_failure_at_last_segment_falls_to_prev(self):
        """fail at idx=2 (last); next direction hits boundary; prev wins."""
        segments = _make_segments(3)
        results = _make_results(3, fail_indices={2})

        merged_sup = [_sup("merged", 10.0, 30.0)]
        # Next direction hits boundary immediately (no call), then prev fires.
        align_fn = MagicMock(return_value=(merged_sup, merged_sup))

        chained_merge_retry(segments, results, align_fn)

        # Single call: merge [1, 2] via prev
        align_fn.assert_called_once()
        called_seg = align_fn.call_args.args[0]
        assert called_seg[0] == 10.0 and called_seg[1] == 30.0
        # Primary covered index is the lower (1)
        assert results[1].status == "ok"
        assert results[1].supervisions == merged_sup
        assert results[2].supervisions == []  # covered, moved to idx=1
