"""Release test: verify word-level alignment margin invariants.

When word_level=True, the returned supervision.alignment['word'] list must
satisfy timing constraints relative to the supervision boundaries:
  - supervision.start  < word[0].start      (start margin)
  - supervision.end    > word[-1].end        (end margin)

This test is included in the release flow because the invariant depends on
both client and backend working together correctly.
"""

import os

import pytest

from lattifai.caption import Caption, Supervision

requires_api_key = pytest.mark.skipif(
    not os.environ.get("LATTIFAI_API_KEY"),
    reason="Requires LATTIFAI_API_KEY",
)


def _check_word_margin_invariant(supervisions, start_margin=0.10, end_margin=0.10, tolerance=0.01):
    """Assert word-level margin invariants on a list of supervisions.

    For each supervision that has word-level alignment:
      - supervision.start must be < first word's start
      - supervision.end must be > last word's end

    Args:
        supervisions: List of Supervision objects with alignment data.
        start_margin: Expected minimum gap between supervision.start and word[0].start.
        end_margin: Expected minimum gap between last word end and supervision.end.
        tolerance: Allowed floating-point tolerance.
    """
    violations = []
    for i, sup in enumerate(supervisions):
        if not getattr(sup, "alignment", None):
            continue
        words = sup.alignment.get("word", [])
        if not words:
            continue

        first_word_start = words[0].start
        last_word_end = words[-1].start + words[-1].duration
        sup_end = sup.start + sup.duration

        if sup.start >= first_word_start - tolerance:
            violations.append(
                f"  sup[{i}]: supervision.start ({sup.start:.4f}) >= word[0].start ({first_word_start:.4f})"
                f' text="{sup.text[:40]}..."'
            )

        if sup_end <= last_word_end + tolerance:
            violations.append(
                f"  sup[{i}]: supervision.end ({sup_end:.4f}) <= last_word.end ({last_word_end:.4f})"
                f' text="{sup.text[:40]}..."'
            )

    if violations:
        msg = f"Word-level margin violations ({len(violations)}):\n" + "\n".join(violations)
        pytest.fail(msg)


class TestWordLevelMarginLocal:
    """Test word-level margin with local model (no API key needed for model, but needs LATTIFAI_API_KEY for backend)."""

    @requires_api_key
    def test_standard_alignment_word_margin(self):
        """Standard alignment: supervision.start < word[0].start."""
        from lattifai.client import LattifAI

        client = LattifAI()
        caption = client.alignment(
            input_media="tests/data/SA1.mp3",
            input_caption="tests/data/SA1.TXT",
            word_level=True,
        )
        assert isinstance(caption, Caption)
        assert len(caption.supervisions) > 0
        _check_word_margin_invariant(caption.supervisions)
        del client

    @requires_api_key
    def test_split_sentence_word_margin(self):
        """Split-sentence alignment: margin invariant still holds after sentence splitting."""
        from lattifai.client import LattifAI

        client = LattifAI()
        caption = client.alignment(
            input_media="tests/data/SA1.mp3",
            input_caption="tests/data/SA1.srt",
            word_level=True,
            split_sentence=True,
        )
        assert isinstance(caption, Caption)
        _check_word_margin_invariant(caption.supervisions)
        del client

    @requires_api_key
    def test_markdown_transcript_word_margin(self):
        """Markdown transcript (triggers difftokenize path): margin invariant must hold."""
        from lattifai.client import LattifAI

        client = LattifAI()
        caption = client.alignment(
            input_media="tests/data/DQacCB9tDaw_16K_First05Mins.mp3",
            input_caption="tests/data/DQacCB9tDaw_16K_First05Mins.md",
            word_level=True,
            split_sentence=True,
        )
        assert isinstance(caption, Caption)
        assert len(caption.supervisions) > 0
        _check_word_margin_invariant(caption.supervisions)
        del client


class TestWordLevelMarginUnit:
    """Unit tests for margin invariant checking (no API key needed)."""

    def test_valid_margins_pass(self):
        """Supervisions with proper margins should pass."""
        from lattifai.caption.supervision import AlignmentItem

        sup = Supervision(
            text="hello world",
            start=0.92,
            duration=1.28,
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.0, duration=0.5, score=0.9),
                    AlignmentItem(symbol="world", start=1.5, duration=0.5, score=0.9),
                ]
            },
        )
        # Should not raise
        _check_word_margin_invariant([sup])

    def test_zero_start_margin_fails(self):
        """Supervision.start == word[0].start should fail the check."""
        from lattifai.caption.supervision import AlignmentItem

        sup = Supervision(
            text="hello world",
            start=1.0,  # same as word[0].start — no margin!
            duration=1.2,
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.0, duration=0.5, score=0.9),
                    AlignmentItem(symbol="world", start=1.5, duration=0.5, score=0.9),
                ]
            },
        )
        with pytest.raises(pytest.fail.Exception, match="margin violations"):
            _check_word_margin_invariant([sup])

    def test_zero_end_margin_fails(self):
        """Supervision.end == last_word.end should fail the check."""
        from lattifai.caption.supervision import AlignmentItem

        sup = Supervision(
            text="hello world",
            start=0.92,
            duration=1.08,  # end = 2.0, same as last word end
            alignment={
                "word": [
                    AlignmentItem(symbol="hello", start=1.0, duration=0.5, score=0.9),
                    AlignmentItem(symbol="world", start=1.5, duration=0.5, score=0.9),
                ]
            },
        )
        with pytest.raises(pytest.fail.Exception, match="margin violations"):
            _check_word_margin_invariant([sup])

    def test_no_alignment_skipped(self):
        """Supervisions without alignment data should be skipped (no error)."""
        sup = Supervision(text="hello", start=0.0, duration=1.0)
        _check_word_margin_invariant([sup])
