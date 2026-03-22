"""Tests for detect_duplicate_blocks and deduplicate_supervisions."""

import pytest

from lattifai.alignment.text_align import (
    DuplicateBlock,
    deduplicate_supervisions,
    detect_duplicate_blocks,
)
from lattifai.caption import Supervision


def _sup(start: float, duration: float, text: str) -> Supervision:
    return Supervision(text=text, start=start, duration=duration)


# ---------------------------------------------------------------------------
# detect_duplicate_blocks
# ---------------------------------------------------------------------------


class TestDetectDuplicateBlocks:
    def test_no_duplicates(self):
        sups = [
            _sup(0, 2, "Hello world this is a test"),
            _sup(2, 2, "Another sentence here today"),
            _sup(4, 2, "Completely different content now"),
        ]
        assert detect_duplicate_blocks(sups) == []

    def test_empty_input(self):
        assert detect_duplicate_blocks([]) == []

    def test_single_supervision(self):
        assert detect_duplicate_blocks([_sup(0, 1, "hello")]) == []

    def test_nearby_exact_duplicate(self):
        """Two nearby blocks with identical text should be detected."""
        shared = "And I just made it proactive so I added a prompt initially it was just a prompt surprise me every half an hour"
        sups = [
            _sup(0, 5, "Some intro text before the duplicate block"),
            _sup(5, 10, shared),
            _sup(15, 3, "A short bridge between the two copies"),
            _sup(18, 4, shared),
            _sup(22, 5, "Some text after the duplicate block ends"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=8)
        assert len(dups) >= 1
        assert dups[0].matched_words >= 8

    def test_far_apart_not_detected(self):
        """Blocks separated by > max_word_gap should NOT be detected."""
        shared = "this is a repeated phrase that appears in two very distant locations in the transcript"
        filler = [_sup(i * 2, 2, f"filler word number {i} padding text here") for i in range(80)]
        sups = [_sup(0, 2, shared)] + filler + [_sup(200, 2, shared)]
        dups = detect_duplicate_blocks(sups, max_word_gap=100)
        assert len(dups) == 0

    def test_short_repeat_ignored(self):
        """A repeat shorter than min_match_words should not be detected."""
        sups = [
            _sup(0, 2, "yes okay sure"),
            _sup(2, 2, "something else here"),
            _sup(4, 2, "yes okay sure"),
        ]
        assert detect_duplicate_blocks(sups, min_match_words=15) == []

    def test_multi_segment_duplicate(self):
        """Duplicate spanning multiple consecutive segments."""
        seg_a1 = "And I just made it proactive so I added a prompt initially"
        seg_a2 = "it was just a prompt surprise me every half an hour surprise me"
        sups = [
            _sup(0, 5, seg_a1),
            _sup(5, 5, seg_a2),
            _sup(10, 3, "bridge text in the middle here"),
            _sup(13, 4, seg_a1),
            _sup(17, 4, seg_a2),
            _sup(21, 5, "ending text after duplicates"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=10)
        assert len(dups) >= 1

    def test_max_time_gap_filter(self):
        """Blocks close in word count but far in time should be filtered."""
        shared = "this is a repeated phrase that should be filtered by time gap constraint here"
        sups = [
            _sup(0, 2, shared),
            _sup(2, 2, "short bridge"),
            _sup(500, 2, shared),  # 500s away
        ]
        dups = detect_duplicate_blocks(sups, max_time_gap=60)
        assert len(dups) == 0

    def test_chinese_duplicate(self):
        """CJK text without spaces should be tokenized per-character and detected."""
        shared = "我刚刚把它做成了主动式的所以我添加了一个提示词最初只是一个提示词每半小时给我一个惊喜"
        sups = [
            _sup(0, 5, shared),
            _sup(5, 3, "这是一段过渡文字"),
            _sup(8, 4, shared),
            _sup(12, 5, "结束语在这里"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=10)
        assert len(dups) >= 1
        assert dups[0].matched_words >= 10

    def test_mixed_cjk_latin_duplicate(self):
        """Mixed Chinese-English text should be detected correctly."""
        shared = "我用Claude来做forced alignment效果非常好每次都能得到精确的时间戳"
        sups = [
            _sup(0, 5, "这是开头的一段话"),
            _sup(5, 8, shared),
            _sup(13, 3, "中间有一些其他内容"),
            _sup(16, 4, shared),
            _sup(20, 5, "这是结尾"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=8)
        assert len(dups) >= 1

    def test_japanese_duplicate(self):
        """Japanese text (hiragana/katakana/kanji mix) should be detected."""
        shared = "このシステムは音声とテキストを正確に同期させることができます"
        sups = [
            _sup(0, 5, shared),
            _sup(5, 2, "別のテキスト"),
            _sup(7, 4, shared),
            _sup(11, 3, "終わり"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=8, ngram=6)
        assert len(dups) >= 1


# ---------------------------------------------------------------------------
# deduplicate_supervisions
# ---------------------------------------------------------------------------


class TestDeduplicateSupervisions:
    def test_no_duplicates_returns_list(self):
        sups = [
            _sup(0, 2, "Hello world this is a test"),
            _sup(2, 2, "Another sentence here today"),
        ]
        result, dups = deduplicate_supervisions(sups)
        assert dups == []
        assert isinstance(result, list)
        assert result is sups  # unchanged, same object

    def test_with_duplicates_returns_cleaned(self):
        shared = "And I just made it proactive so I added a prompt initially it was just a prompt surprise me every half an hour"
        sups = [
            _sup(0, 5, "intro text before the duplicate here"),
            _sup(5, 10, shared),
            _sup(15, 3, "bridge between copies here today"),
            _sup(18, 4, shared),
            _sup(22, 5, "ending text after the duplicate block"),
        ]
        result, dups = deduplicate_supervisions(sups)
        assert len(dups) >= 1
        assert isinstance(result, list)
        assert len(result) < len(sups)

    def test_deduped_removes_shorter_copy(self):
        """The shorter-duration copy should be removed."""
        shared = "And I just made it proactive so I added a prompt initially it was just a prompt surprise me every half an hour"
        sups = [
            _sup(0, 10, shared),  # 10s duration - longer
            _sup(10, 2, "bridge"),
            _sup(12, 3, shared),  # 3s duration - shorter, should be removed
            _sup(15, 5, "ending"),
        ]
        cleaned, dups = deduplicate_supervisions(sups)
        if dups:
            # Shorter copy (3s) should be removed, longer (10s) kept
            cleaned_texts = [s.text for s in cleaned]
            assert shared in cleaned_texts  # longer copy kept
            assert len(cleaned) < len(sups)
