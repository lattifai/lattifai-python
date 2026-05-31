"""Unit tests for ``_widen_transcript_window`` (zero model / error_align deps).

The transcription-strategy segment window is taken from the transcription
(e.g. YouTube VTT) timestamps, whose line-level boundaries routinely clip the
first/last word of a turn. ``_widen_transcript_window`` extends each segment
window halfway into the inter-sentence gap — to the midpoint between the
previous sentence's ``end`` and the current ``start`` (and symmetrically on
the right) — giving the lattice aligner some acoustic context to recover the
clipped words without reaching the neighbour's own (possibly loose) boundary,
capped at 2 s per side so a large neighbour gap (ad / music residue) cannot
drag the window into unrelated speech.
"""

from lattifai.alignment.text_align import _widen_transcript_window
from lattifai.caption import Supervision


def S(start: float, end: float) -> Supervision:
    """Build a Supervision spanning [start, end] (Supervision stores duration)."""
    return Supervision(text="x", start=start, duration=end - start)


def test_widen_normal_extends_to_gap_midpoints():
    # transcription [0-1] [2-3] [4-5]; group = middle sentence idx [1:2)
    tr = [S(0, 1), S(2, 3), S(4, 5)]
    lo, hi = _widen_transcript_window(tr, 1, 2)
    assert lo == 1.5  # midpoint of (prev.end=1, cur.start=2)
    assert hi == 3.5  # midpoint of (cur.end=3, next.start=4)


def test_widen_caps_at_2s_each_side():
    # huge gaps before/after the group -> cap to +/-2 s, not the gap midpoint
    tr = [S(0, 1), S(10, 11), S(20, 21)]
    lo, hi = _widen_transcript_window(tr, 1, 2)
    assert lo == 8.0  # cur_lo(10) - 2.0, not midpoint(5.5)
    assert hi == 13.0  # cur_hi(11) + 2.0, not midpoint(15.5)


def test_widen_first_group_no_left_extend():
    tr = [S(0, 1), S(2, 3)]
    lo, hi = _widen_transcript_window(tr, 0, 1)
    assert lo == 0.0  # no previous sentence -> stays at cur_lo
    assert hi == 1.5  # midpoint of (cur.end=1, next.start=2)


def test_widen_last_group_no_right_extend():
    tr = [S(0, 1), S(2, 3)]
    lo, hi = _widen_transcript_window(tr, 1, 2)
    assert lo == 1.5  # midpoint of (prev.end=1, cur.start=2)
    assert hi == 3.0  # no next sentence -> stays at cur_hi


def test_widen_guards_out_of_order_vtt():
    # prev.end (2.5) > cur.start (2.0): only-extend never shrinks the window
    tr = [S(0, 2.5), S(2, 3), S(4, 5)]
    lo, hi = _widen_transcript_window(tr, 1, 2)
    assert lo == 2.0  # midpoint(2.25) > cur_lo(2.0) -> clamped, window not shrunk
    assert hi == 3.5  # midpoint of (cur.end=3, next.start=4)


def test_widen_multi_sentence_group():
    # group spans two sentences idx [1:3); window = gap midpoints on each side
    tr = [S(0, 1), S(2, 3), S(3.5, 4.5), S(6, 7)]
    lo, hi = _widen_transcript_window(tr, 1, 3)
    assert lo == 1.5  # midpoint of (tr[0].end=1, cur.start=2)
    assert hi == 5.25  # midpoint of (cur.end=4.5, tr[3].start=6)
