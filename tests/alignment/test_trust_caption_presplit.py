"""Trust-caption-timestamps path: pre-split sentences BEFORE segmenting.

When ``strategy='caption'`` + ``trust_caption_timestamps=True`` and the input
captions carry word-level alignment, sentence splitting must happen ONCE on the
whole caption *before* the Segmenter slices it — not per-segment inside each
``aligner.alignment`` call. Per-segment splitting re-splits at every segment
edge and inflates the supervision count (the +7-sup drift observed in the a16z
``entire`` vs ``caption`` A/B). Mirrors the transcription path's
``transcription_already_split`` optimisation.

The gate is data-driven: it keys off whether ``caption.supervisions`` carry
word-level alignment at runtime — NOT off the input source (VTT/SRT/MD).

These tests fake the aligner (``__new__`` + MagicMock) so no model / audio /
HTTP is touched; they pin the orchestration contract:
  - word-level alignment present → ``split_sentences`` called exactly once,
    and every per-segment ``aligner.alignment`` gets ``split_sentence=False``.
  - word-level alignment absent  → no pre-split, each segment keeps
    ``split_sentence=True``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from lattifai.audio2 import AudioData
from lattifai.caption import AlignmentItem, Supervision
from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, CaptionConfig
from lattifai.data import Caption


def _word_aligned_sup(sup_id: str, text: str, start: float, dur: float) -> Supervision:
    # A single AlignmentItem suffices — has_word_align only checks truthiness.
    item = AlignmentItem(symbol=text.split()[0], start=start, duration=dur)
    return Supervision(id=sup_id, text=text, start=start, duration=dur, alignment={"word": [item]})


def _plain_sup(sup_id: str, text: str, start: float, dur: float) -> Supervision:
    return Supervision(id=sup_id, text=text, start=start, duration=dur)


def _two_segment_caption(sup_factory) -> Caption:
    # gap 4.0s → 10.0s (> segment_max_gap=4.0) forces a 2-segment split.
    return Caption(
        supervisions=[
            sup_factory("s0", "Hello world. How are you.", 0.0, 2.0),
            sup_factory("s1", "I am fine.", 2.0, 2.0),
            sup_factory("s2", "Second segment here.", 10.0, 2.0),
            sup_factory("s3", "Another one.", 12.0, 2.0),
        ]
    )


def _make_client(caption: Caption):
    """Minimal LattifAI with a faked aligner — skips model/HTTP via __new__."""
    client = LattifAI.__new__(LattifAI)
    client.caption_config = CaptionConfig()
    client.aligner = MagicMock()
    client.aligner.config = AlignmentConfig(strategy="caption", trust_caption_timestamps=True)
    # split_sentences passthrough spy (returns the list unchanged).
    client.aligner.tokenizer.split_sentences = MagicMock(side_effect=lambda sups, **kw: list(sups))
    client.aligner.emission = MagicMock(return_value=np.zeros(10, dtype=np.float32))

    split_flags: list = []

    def _fake_alignment(media, sups, split_sentence=None, **kw):
        split_flags.append(split_sentence)
        return list(sups), list(sups)

    client.aligner.alignment = MagicMock(side_effect=_fake_alignment)
    # transcriber / audio_loader are lazy properties never reached on this path
    # (caption strategy, no transcription; media is already an AudioData).
    client._read_caption = MagicMock(return_value=caption)
    # align() also runs diarization/event/profile after alignment — disable them
    # so the test isolates the segmentation + split-sentence orchestration.
    client.diarization_config = MagicMock(enabled=False)
    client.event_config = MagicMock(enabled=False)
    client.config = MagicMock(profile=False)

    media = MagicMock(spec=AudioData)
    media.ndarray = np.zeros((1, 16000 * 15), dtype=np.float32)
    media.sampling_rate = 16000
    media.duration = 15.0
    return client, media, split_flags


def test_presplit_when_word_alignment_present():
    """Word-level alignment present → split once on whole caption, segments
    aligned with split_sentence=False (no per-segment re-split)."""
    cap = _two_segment_caption(_word_aligned_sup)
    client, media, split_flags = _make_client(cap)

    client.alignment(input_media=media, input_caption="dummy.vtt", split_sentence=True)

    assert (
        client.aligner.tokenizer.split_sentences.call_count == 1
    ), "expected ONE pre-split on the whole caption before segmenting"
    assert len(split_flags) == 2, "Segmenter should yield 2 segments for a >gap split"
    assert all(flag is False for flag in split_flags), split_flags


def test_no_presplit_when_word_alignment_absent():
    """No word-level alignment → data-driven gate skips pre-split; each segment
    keeps its own split_sentence=True intent."""
    cap = _two_segment_caption(_plain_sup)
    client, media, split_flags = _make_client(cap)

    client.alignment(input_media=media, input_caption="dummy.vtt", split_sentence=True)

    assert (
        client.aligner.tokenizer.split_sentences.call_count == 0
    ), "must NOT pre-split when supervisions lack word-level alignment"
    assert len(split_flags) == 2
    assert all(flag is True for flag in split_flags), split_flags


def test_skipalign_event_marker_keeps_word():
    """A standalone ``[event]`` sup is skipalign'd by the Segmenter (bypasses
    the aligner). It must still emit a word entry — otherwise the marker
    silently drops from the output, which is exactly the 3 lost
    ``[laughter]`` / ``[clears throat]`` / ``[snorts]`` words seen in the a16z
    run after pre-split carved them into standalone event sups."""
    cap = Caption(
        supervisions=[
            _word_aligned_sup("s0", "Hello world.", 0.0, 2.0),
            # standalone event marker w/o word-level alignment; the >4s gap
            # forces the Segmenter to cut it into its own skipalign segment.
            Supervision(id="ev", text="[laughter]", start=10.0, duration=1.0),
        ]
    )
    client, media, _ = _make_client(cap)

    result = client.alignment(input_media=media, input_caption="x.vtt", split_sentence=True)

    ev = [s for s in result.supervisions if s.text.strip() == "[laughter]"]
    assert ev, "event-marker sup vanished from output"
    assert (getattr(ev[0], "alignment", None) or {}).get(
        "word"
    ), "skipalign event marker emitted no word → would drop downstream"
