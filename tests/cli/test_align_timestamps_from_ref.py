"""Tests for align_timestamps_from_ref function."""

from dataclasses import dataclass
from typing import Optional

import pytest

from lattifai.cli.caption import align_timestamps_from_ref


@dataclass
class FakeSupervision:
    """Minimal supervision stub for testing."""

    text: str
    start: float = 0.0
    duration: float = 0.0
    speaker: Optional[str] = None


def _starts(results):
    """Extract start times for easy assertion."""
    return [round(r.start, 4) for r in results]


def _durations(results):
    """Extract durations for easy assertion."""
    return [round(r.duration, 4) for r in results]


# ---------------------------------------------------------------------------
# Basic Latin matching
# ---------------------------------------------------------------------------
class TestLatinMatching:
    def test_exact_match(self):
        """Input text matches reference exactly."""
        ref = [
            FakeSupervision("Hello world, how are you doing today?", start=1.0, duration=3.0),
            FakeSupervision("This is the second segment here.", start=5.0, duration=2.0),
            FakeSupervision("And the third segment is here.", start=8.0, duration=2.0),
        ]
        inp = [
            FakeSupervision("Hello world, how are you doing today?", start=0.0, duration=10.0),
            FakeSupervision("This is the second segment here.", start=10.0, duration=10.0),
            FakeSupervision("And the third segment is here.", start=20.0, duration=10.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [1.0, 5.0, 8.0]
        assert len(results) == 3

    def test_partial_match_preserves_text(self):
        """Original text and speaker labels are preserved; only timestamps change."""
        ref = [FakeSupervision("The quick brown fox jumps over the lazy dog", start=2.5, duration=4.0)]
        inp = [FakeSupervision("The quick brown fox jumps over the lazy dog", start=0.0, duration=1.0, speaker="Alice")]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 2.5
        assert results[0].text == "The quick brown fox jumps over the lazy dog"
        assert results[0].speaker == "Alice"


# ---------------------------------------------------------------------------
# CJK matching (the main fix)
# ---------------------------------------------------------------------------
class TestCJKMatching:
    def test_chinese_segments(self):
        """Chinese text must match via character-based keys."""
        ref = [
            FakeSupervision("今天天气非常好我们一起出去玩吧", start=0.0, duration=3.0),
            FakeSupervision("下午我们去公园散步怎么样", start=4.0, duration=2.5),
            FakeSupervision("晚上一起吃饭看电影好不好", start=7.0, duration=3.0),
        ]
        inp = [
            FakeSupervision("今天天气非常好我们一起出去玩吧", start=0.0),
            FakeSupervision("下午我们去公园散步怎么样", start=0.0),
            FakeSupervision("晚上一起吃饭看电影好不好", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [0.0, 4.0, 7.0]

    def test_japanese_segments(self):
        """Japanese text matches via character-based keys."""
        ref = [
            FakeSupervision("今日はとても良い天気ですね散歩に行きましょう", start=1.0, duration=4.0),
            FakeSupervision("午後は公園でピクニックをしましょう", start=6.0, duration=3.0),
        ]
        inp = [
            FakeSupervision("今日はとても良い天気ですね散歩に行きましょう", start=0.0),
            FakeSupervision("午後は公園でピクニックをしましょう", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [1.0, 6.0]

    def test_short_chinese_segments(self):
        """Short CJK segments (6+ chars) should still match."""
        ref = [
            FakeSupervision("你好世界再见朋友", start=2.0, duration=1.5),
            FakeSupervision("明天见面聊天吧", start=5.0, duration=2.0),
        ]
        inp = [
            FakeSupervision("你好世界再见朋友", start=0.0),
            FakeSupervision("明天见面聊天吧", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [2.0, 5.0]

    def test_very_short_cjk_no_match(self):
        """CJK text shorter than 6 chars should NOT match (too ambiguous)."""
        ref = [FakeSupervision("你好世界", start=3.0, duration=1.0)]
        inp = [FakeSupervision("你好世界", start=0.0)]
        results = align_timestamps_from_ref(inp, ref)

        # 4 chars < minimum key length 6, should fall back to original
        assert results[0].start == 0.0


# ---------------------------------------------------------------------------
# Mixed CJK + Latin
# ---------------------------------------------------------------------------
class TestMixedLanguage:
    def test_mixed_chinese_english(self):
        """Segments mixing Chinese and English should match."""
        ref = [
            FakeSupervision("这个Python程序运行很快速度非常好", start=1.0, duration=3.0),
            FakeSupervision("We need to fix the API before release", start=5.0, duration=2.0),
        ]
        inp = [
            FakeSupervision("这个Python程序运行很快速度非常好", start=0.0),
            FakeSupervision("We need to fix the API before release", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [1.0, 5.0]


# ---------------------------------------------------------------------------
# Duplicate / repeated text
# ---------------------------------------------------------------------------
class TestDuplicateText:
    def test_repeated_text_sequential_match(self):
        """Same text appearing twice should match to different positions sequentially."""
        ref = [
            FakeSupervision("This is the chorus of the song yeah", start=10.0, duration=5.0),
            FakeSupervision("Some other verse with different words here", start=20.0, duration=5.0),
            FakeSupervision("This is the chorus of the song yeah", start=30.0, duration=5.0),
        ]
        inp = [
            FakeSupervision("This is the chorus of the song yeah", start=0.0),
            FakeSupervision("Some other verse with different words here", start=0.0),
            FakeSupervision("This is the chorus of the song yeah", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        # First chorus → 10.0, verse → 20.0, second chorus → 30.0
        assert _starts(results) == [10.0, 20.0, 30.0]

    def test_repeated_cjk_text_sequential(self):
        """Repeated CJK text should match sequentially."""
        ref = [
            FakeSupervision("这是副歌部分大家一起唱吧", start=5.0, duration=3.0),
            FakeSupervision("这是主歌部分独唱一段旋律", start=10.0, duration=3.0),
            FakeSupervision("这是副歌部分大家一起唱吧", start=15.0, duration=3.0),
        ]
        inp = [
            FakeSupervision("这是副歌部分大家一起唱吧", start=0.0),
            FakeSupervision("这是主歌部分独唱一段旋律", start=0.0),
            FakeSupervision("这是副歌部分大家一起唱吧", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert _starts(results) == [5.0, 10.0, 15.0]


# ---------------------------------------------------------------------------
# Unmatched segments
# ---------------------------------------------------------------------------
class TestUnmatched:
    def test_unmatched_keeps_original_timestamp(self):
        """Segments with no match in reference keep their original timestamps."""
        ref = [
            FakeSupervision("The quick brown fox jumps over the lazy dog", start=2.0, duration=3.0),
        ]
        inp = [
            FakeSupervision("Completely different text that has no overlap", start=5.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        # No match → keep original 5.0
        assert results[0].start == 5.0

    def test_unmatched_monotonicity(self):
        """Unmatched segments must not go before the previous segment's start."""
        ref = [
            FakeSupervision("First segment has some interesting words here today", start=10.0, duration=3.0),
        ]
        inp = [
            FakeSupervision("First segment has some interesting words here today", start=0.0),
            FakeSupervision("No match at all for this text", start=3.0),  # original 3.0 < prev 10.0
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 10.0
        assert results[1].start == 10.0  # bumped up to prev_start

    def test_mixed_matched_unmatched(self):
        """Mix of matched and unmatched segments."""
        ref = [
            FakeSupervision("Alpha segment with enough words to match here", start=1.0, duration=2.0),
            FakeSupervision("Gamma segment with enough words to match here", start=6.0, duration=2.0),
        ]
        inp = [
            FakeSupervision("Alpha segment with enough words to match here", start=0.0),
            FakeSupervision("Beta no match possible", start=3.0),
            FakeSupervision("Gamma segment with enough words to match here", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 1.0  # matched
        assert results[1].start == 3.0  # unmatched, keeps 3.0 (> prev 1.0)
        assert results[2].start == 6.0  # matched


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------
class TestEmptyInputs:
    def test_empty_supervisions(self):
        ref = [FakeSupervision("hello world", start=1.0, duration=1.0)]
        assert align_timestamps_from_ref([], ref) == []

    def test_empty_ref(self):
        inp = [FakeSupervision("hello", start=1.0)]
        results = align_timestamps_from_ref(inp, [])
        assert results[0].start == 1.0

    def test_both_empty(self):
        assert align_timestamps_from_ref([], []) == []

    def test_ref_only_punctuation(self):
        """Reference with only punctuation normalizes to empty → treated as empty ref."""
        ref = [FakeSupervision("...", start=1.0, duration=1.0)]
        inp = [FakeSupervision("hello world test", start=2.0)]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 2.0  # no match, keeps original


# ---------------------------------------------------------------------------
# Last segment duration
# ---------------------------------------------------------------------------
class TestLastSegmentDuration:
    def test_last_duration_from_ref_end(self):
        """Last segment duration should be derived from reference end time, not hardcoded."""
        ref = [
            FakeSupervision("First segment with several words for matching", start=0.0, duration=5.0),
            FakeSupervision("Second segment also has several words for matching", start=10.0, duration=8.0),
        ]
        inp = [
            FakeSupervision("First segment with several words for matching", start=0.0),
            FakeSupervision("Second segment also has several words for matching", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        # ref_end_time = max(0+5, 10+8) = 18.0
        # last duration = 18.0 - 10.0 = 8.0
        assert results[-1].duration == 8.0

    def test_intermediate_durations_from_gaps(self):
        """Intermediate durations are computed from next segment's start."""
        ref = [
            FakeSupervision("Alpha bravo charlie delta echo foxtrot golf", start=1.0, duration=2.0),
            FakeSupervision("Hotel india juliet kilo lima mike november", start=5.0, duration=3.0),
            FakeSupervision("Oscar papa quebec romeo sierra tango uniform", start=10.0, duration=4.0),
        ]
        inp = [
            FakeSupervision("Alpha bravo charlie delta echo foxtrot golf", start=0.0),
            FakeSupervision("Hotel india juliet kilo lima mike november", start=0.0),
            FakeSupervision("Oscar papa quebec romeo sierra tango uniform", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].duration == 4.0  # 5.0 - 1.0
        assert results[1].duration == 5.0  # 10.0 - 5.0
        assert results[2].duration == 4.0  # ref_end = 14.0, 14.0 - 10.0


# ---------------------------------------------------------------------------
# Monotonicity guarantee
# ---------------------------------------------------------------------------
class TestMonotonicity:
    def test_timestamps_never_decrease(self):
        """Output timestamps must be non-decreasing regardless of match quality."""
        ref = [
            FakeSupervision("The very first line of text in this document here", start=0.5, duration=2.0),
            FakeSupervision("The very last line of text in this document here", start=9.0, duration=1.0),
        ]
        inp = [
            FakeSupervision("The very first line of text in this document here", start=0.0),
            FakeSupervision("No match here", start=0.0),
            FakeSupervision("Still no match", start=0.0),
            FakeSupervision("The very last line of text in this document here", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        starts = _starts(results)
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], f"Timestamp decreased at index {i}: {starts}"


# ---------------------------------------------------------------------------
# Markdown link stripping
# ---------------------------------------------------------------------------
class TestNormalization:
    def test_markdown_links_stripped(self):
        """Markdown links [text](url) should be stripped before matching."""
        ref = [FakeSupervision("Check out this amazing project for developers", start=3.0, duration=2.0)]
        inp = [FakeSupervision("Check out [this amazing](http://example.com) project for developers", start=0.0)]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 3.0

    def test_punctuation_ignored(self):
        """Punctuation differences should not prevent matching."""
        ref = [FakeSupervision("Hello, world! How are you doing today?", start=2.0, duration=1.5)]
        inp = [FakeSupervision("Hello world How are you doing today", start=0.0)]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 2.0


# ---------------------------------------------------------------------------
# Duration edge cases
# ---------------------------------------------------------------------------
class TestDurationEdgeCases:
    def test_all_durations_non_negative(self):
        """No duration should be negative."""
        ref = [
            FakeSupervision("Segment one with enough words to match easily", start=5.0, duration=1.0),
            FakeSupervision("Segment two with enough words to match easily", start=3.0, duration=1.0),
        ]
        inp = [
            FakeSupervision("Segment one with enough words to match easily", start=0.0),
            FakeSupervision("Segment two with enough words to match easily", start=0.0),
        ]
        results = align_timestamps_from_ref(inp, ref)

        for r in results:
            assert r.duration >= 0.0, f"Negative duration: {r.duration}"

    def test_single_segment(self):
        """Single segment should work without errors."""
        ref = [FakeSupervision("Only one segment but it has enough words for matching", start=4.0, duration=6.0)]
        inp = [FakeSupervision("Only one segment but it has enough words for matching", start=0.0)]
        results = align_timestamps_from_ref(inp, ref)

        assert results[0].start == 4.0
        assert results[0].duration == 6.0  # ref_end=10.0, 10.0-4.0=6.0
