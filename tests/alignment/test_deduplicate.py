"""Tests for detect_duplicate_blocks and deduplicate_supervisions."""

import pytest

from lattifai.alignment.text_align import (
    DuplicateBlock,
    deduplicate_supervisions,
    detect_duplicate_blocks,
    is_lyrics_supervisions,
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

    def test_spelled_out_word_not_detected_as_duplicate(self):
        """Spelled-out words like C-L-A-U-D-E should not trigger duplicate detection.

        Hyphens inflate token count (11 tokens) past min_match_words=10, but
        total character length is only 11, well below min_match_chars=20.
        """
        sups = [
            _sup(0, 5, "Some context before the spelling"),
            _sup(5, 5, "Claude spelled with a W-C-L-A-U-D-E."),
            _sup(10, 3, "A bridge between the two mentions"),
            _sup(13, 7, "Versus C-L-A-U-D-E from Anthropic."),
            _sup(20, 5, "Some context after the spelling"),
        ]
        dups = detect_duplicate_blocks(sups)
        assert len(dups) == 0, f"Spelled-out word should not be flagged as duplicate, got {dups}"

    def test_intra_segment_parallel_structure_not_detected(self):
        """Parallel structure within a single supervision should NOT be flagged.

        Example: "接受文本、音频和图像的任意组合作为输入，并生成文本、音频和图像的任意组合的输出"
        contains "文本、音频和图像的任意组合" twice but it's rhetorical parallelism, not a duplicate block.
        """
        sups = [
            _sup(0, 5, "这是开头的一段话用来填充"),
            _sup(
                5,
                10,
                "这使得GPT-4o能够接受文本、音频和图像的任意组合作为输入，并生成文本、音频和图像的任意组合的输出。",
            ),
            _sup(15, 5, "是兼具了听觉视觉的多模态模型"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=10)
        assert len(dups) == 0

    def test_intra_segment_parallel_at_different_timestamps(self):
        """Same parallel structure in two separate supervisions at different times should NOT be flagged
        when both are intra-segment matches and the time gap exceeds max_time_gap."""
        sups = [
            _sup(0, 5, "这是开头"),
            _sup(
                5,
                20,
                "这使得GPT-4o能够接受文本、音频和图像的任意组合作为输入，并生成文本、音频和图像的任意组合的输出。",
            ),
            _sup(25, 5, "过渡内容在这里"),
            # Same sentence repeated by speaker referencing earlier content, 12 minutes later
            _sup(
                750,
                15,
                "我们视频之前也有说到，这使得GPT-4o能够接受文本、音频和图像的任意组合作为输入，并生成文本、音频和图像的任意组合的输出。",
            ),
            _sup(765, 5, "这是结尾"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=10, max_time_gap=300)
        assert len(dups) == 0

    def test_true_duplicate_still_detected_with_parallel_structure(self):
        """A genuine nearby duplicate should still be detected even if it contains parallel structure."""
        text = "这使得GPT-4o能够接受文本、音频和图像的任意组合作为输入，并生成文本、音频和图像的任意组合的输出。"
        sups = [
            _sup(0, 5, "这是开头"),
            _sup(5, 10, text),
            _sup(15, 3, "过渡文字"),
            _sup(18, 10, text),  # genuine duplicate nearby
            _sup(28, 5, "这是结尾"),
        ]
        dups = detect_duplicate_blocks(sups, min_match_words=10)
        assert len(dups) >= 1
        # The detected block should span different supervisions
        assert dups[0].first != dups[0].second


# ---------------------------------------------------------------------------
# is_lyrics_supervisions
# ---------------------------------------------------------------------------


class TestIsLyricsSupervisions:
    def test_no_markers(self):
        sups = [_sup(0, 2, "Hello world this is a test"), _sup(2, 2, "Another regular sentence")]
        assert is_lyrics_supervisions(sups) is False

    def test_empty_input(self):
        assert is_lyrics_supervisions([]) is False

    def test_verse_and_chorus(self):
        sups = [
            _sup(0, 2, "[Verse 1]"),
            _sup(2, 2, "Some lyric line one"),
            _sup(4, 2, "[Chorus]"),
            _sup(6, 2, "Some lyric line two"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_intro_outro(self):
        sups = [
            _sup(0, 2, "[Intro]"),
            _sup(2, 2, "Opening lyric"),
            _sup(4, 2, "[Outro]"),
            _sup(6, 2, "Closing lyric"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_pre_chorus_and_chorus(self):
        sups = [
            _sup(0, 2, "[Pre-Chorus]"),
            _sup(2, 2, "Build-up lyric"),
            _sup(4, 2, "[Chorus]"),
            _sup(6, 2, "Hook lyric"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_only_chorus_repeated_below_threshold(self):
        """Only one section type — should not pass the default threshold of 2."""
        sups = [
            _sup(0, 2, "[Chorus]"),
            _sup(2, 2, "Same lyric"),
            _sup(4, 2, "[Chorus]"),
            _sup(6, 2, "Same lyric again"),
        ]
        assert is_lyrics_supervisions(sups) is False

    def test_min_distinct_one(self):
        """With min_distinct_sections=1 a single Chorus is enough."""
        sups = [_sup(0, 2, "[Chorus]"), _sup(2, 2, "Lyric")]
        assert is_lyrics_supervisions(sups, min_distinct_sections=1) is True

    def test_marker_inline_with_text(self):
        sups = [
            _sup(0, 2, "[Verse 1] First line of the verse"),
            _sup(2, 2, "[Chorus] First line of the chorus"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_instrumental_break_with_description(self):
        sups = [
            _sup(0, 2, "[Instrumental Break - Chiptune Synth Melody]"),
            _sup(2, 2, "[Bridge]"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_section_with_colon_descriptor(self):
        """Real-world markers like ``[Intro: Female Vocal]`` and ``[Outro: ...]``."""
        sups = [
            _sup(0, 2, "[Intro: Female Vocal]"),
            _sup(2, 2, "Some opening line"),
            _sup(4, 2, "[Outro: Female Vocal]"),
            _sup(6, 2, "Closing line"),
        ]
        assert is_lyrics_supervisions(sups) is True

    def test_word_boundary_introduction_not_intro(self):
        """``[Introduction]`` must not be mistakenly matched as ``intro``."""
        sups = [
            _sup(0, 2, "[Introduction]"),
            _sup(2, 2, "[Conclusion]"),
        ]
        assert is_lyrics_supervisions(sups) is False

    def test_non_section_brackets_ignored(self):
        """Performance directives like [whispered] should not be treated as song sections."""
        sups = [
            _sup(0, 2, "[whispered] something quiet"),
            _sup(2, 2, "[laughter] continues"),
            _sup(4, 2, "[applause] more text"),
        ]
        assert is_lyrics_supervisions(sups) is False

    def test_chinese_lyrics_with_english_markers(self):
        """Section markers in English are common even when lyrics body is Chinese."""
        sups = [
            _sup(0, 2, "[Verse 1]"),
            _sup(2, 2, "窗外的雨滴落在玻璃上"),
            _sup(4, 2, "[Chorus]"),
            _sup(6, 2, "如果时光能够倒流"),
        ]
        assert is_lyrics_supervisions(sups) is True


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
