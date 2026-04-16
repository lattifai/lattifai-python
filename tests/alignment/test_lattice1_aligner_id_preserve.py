"""Unit tests for Lattice1Aligner id preservation behavior.

When split_sentence=False the alignment pipeline must preserve the input
Supervision.id on the returned alignments (1:1 mapping). The backend's
``detokenize`` endpoint rebuilds Supervision objects and drops the id, so
Lattice1Aligner.alignment() must re-populate ids from the source list.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from lattifai.alignment.lattice1_aligner import Lattice1Aligner
from lattifai.caption import Supervision


def _make_aligner():
    """Build a Lattice1Aligner without running __init__ (skips model/HTTP)."""
    aligner = Lattice1Aligner.__new__(Lattice1Aligner)
    aligner.config = MagicMock()
    aligner.config.boost = 0.0
    aligner.config.transition_penalty = 0.0
    aligner.config.start_margin = 0.1
    aligner.config.end_margin = 0.1
    aligner.config.check_sanity = False
    aligner.tokenizer = MagicMock()
    aligner.worker = MagicMock()
    aligner.frame_shift = 0.02
    return aligner


def _input_supervisions():
    return [
        Supervision(id="sup_001", text="hello", start=0.0, duration=1.0),
        Supervision(id="sup_002", text="world", start=1.0, duration=1.0),
    ]


def _backend_alignments():
    """Simulate detokenize() output: ids are dropped by the backend."""
    return [
        Supervision(id="", text="hello", start=0.0, duration=0.9),
        Supervision(id="", text="world", start=1.0, duration=0.95),
    ]


def _configure(aligner, in_sups, out_alignments):
    # tokenize() passes supervisions through untouched when split_sentence=False
    aligner.tokenizer.tokenize.return_value = (
        in_sups,
        "lattice-id-x",
        ("graph", "final_state", 1.0),
        False,
    )
    aligner.worker.alignment.return_value = (
        {"max_probs": np.zeros(10), "aligned_probs": np.zeros(10)},
        [[]],
        [[]],
        0.02,
        0.0,
        0,
    )
    aligner.tokenizer.detokenize.return_value = out_alignments


def test_alignment_preserves_input_ids_when_split_sentence_false():
    aligner = _make_aligner()
    in_sups = _input_supervisions()
    out_alignments = _backend_alignments()
    _configure(aligner, in_sups, out_alignments)

    audio = MagicMock()
    audio.streaming_mode = False

    supervisions, alignments = aligner.alignment(
        audio,
        in_sups,
        split_sentence=False,
        verbose=False,
    )

    assert [a.id for a in alignments] == [
        "sup_001",
        "sup_002",
    ], "Expected alignments to inherit ids from input supervisions"
    # The returned 'supervisions' handle remains the original list, unchanged.
    assert supervisions is in_sups


def test_alignment_does_not_fabricate_ids_when_split_sentence_true():
    """When split_sentence=True, 1:1 mapping is not guaranteed; do not copy."""
    aligner = _make_aligner()
    in_sups = _input_supervisions()
    # Simulate tokenizer.split_sentences producing a differently-sized list.
    split_sups = [
        Supervision(id="", text="hello", start=0.0, duration=0.5),
        Supervision(id="", text="world", start=0.5, duration=0.5),
        Supervision(id="", text="!", start=1.0, duration=0.5),
    ]
    out_alignments = [Supervision(id="", text=s.text, start=s.start, duration=s.duration) for s in split_sups]
    aligner.tokenizer.tokenize.return_value = (
        split_sups,
        "lattice-id-y",
        ("graph", "final_state", 1.0),
        False,
    )
    aligner.worker.alignment.return_value = (
        {"max_probs": np.zeros(10), "aligned_probs": np.zeros(10)},
        [[]],
        [[]],
        0.02,
        0.0,
        0,
    )
    aligner.tokenizer.detokenize.return_value = out_alignments

    audio = MagicMock()
    audio.streaming_mode = False

    _, alignments = aligner.alignment(audio, in_sups, split_sentence=True, verbose=False)

    assert all(a.id == "" for a in alignments), "Split path must not copy ids across a size mismatch"


def test_alignment_skips_copy_when_lengths_mismatch():
    """Defensive: if alignments length unexpectedly differs, leave ids untouched."""
    aligner = _make_aligner()
    in_sups = _input_supervisions()
    out_alignments = _backend_alignments()[:1]  # len 1 vs input len 2
    _configure(aligner, in_sups, out_alignments)

    audio = MagicMock()
    audio.streaming_mode = False

    _, alignments = aligner.alignment(audio, in_sups, split_sentence=False, verbose=False)
    assert [a.id for a in alignments] == [""]
