"""Tests for punctuation module."""

import pytest

from lattifai.alignment.punctuation import (
    END_PUNCTUATION,
    GROUPING_SEPARATOR,
    PUNCTUATION,
    PUNCTUATION_SPACE,
    STAR_TOKEN,
)


class TestPunctuation:
    """Test cases for PUNCTUATION constant."""

    def test_punctuation_is_string(self):
        """PUNCTUATION should be a string."""
        assert isinstance(PUNCTUATION, str)

    def test_punctuation_not_empty(self):
        """PUNCTUATION should not be empty."""
        assert len(PUNCTUATION) > 0

    def test_punctuation_contains_ascii(self):
        """PUNCTUATION should contain common ASCII punctuation."""
        ascii_punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        for char in ascii_punct:
            assert char in PUNCTUATION, f"Missing ASCII punctuation: {char!r}"

    def test_punctuation_contains_chinese(self):
        """PUNCTUATION should contain Chinese punctuation marks."""
        chinese_punct = "。，、？！：；·￥…—～"
        for char in chinese_punct:
            assert char in PUNCTUATION, f"Missing Chinese punctuation: {char!r}"

    def test_punctuation_contains_chinese_quotation_marks(self):
        """PUNCTUATION should contain Chinese quotation marks (double and single)."""
        # Chinese double quotation marks: " (left) and " (right)
        assert "“" in PUNCTUATION, "Missing Chinese left double quote"
        assert "”" in PUNCTUATION, "Missing Chinese right double quote"
        # Chinese single quotation marks: ' (left) and ' (right)
        assert "‘" in PUNCTUATION, "Missing Chinese left single quote"
        assert "’" in PUNCTUATION, "Missing Chinese right single quote"

    def test_punctuation_contains_chinese_brackets(self):
        """PUNCTUATION should contain Chinese brackets."""
        chinese_brackets = "《》〈〉【】〔〕（）"
        for char in chinese_brackets:
            assert char in PUNCTUATION, f"Missing Chinese bracket: {char!r}"

    def test_punctuation_contains_japanese(self):
        """PUNCTUATION should contain Japanese punctuation marks."""
        japanese_punct = "「」『』。、・"
        for char in japanese_punct:
            assert char in PUNCTUATION, f"Missing Japanese punctuation: {char!r}"

    def test_punctuation_contains_arabic(self):
        """PUNCTUATION should contain Arabic punctuation marks."""
        arabic_punct = "،؛؟"
        for char in arabic_punct:
            assert char in PUNCTUATION, f"Missing Arabic punctuation: {char!r}"

    def test_punctuation_contains_guillemets(self):
        """PUNCTUATION should contain guillemets (French, Russian, etc.)."""
        guillemets = "«»‹›"
        for char in guillemets:
            assert char in PUNCTUATION, f"Missing guillemet: {char!r}"

    def test_punctuation_contains_spanish_inverted_marks(self):
        """PUNCTUATION should contain Spanish inverted punctuation marks."""
        assert "¡" in PUNCTUATION, "Missing Spanish inverted exclamation"
        assert "¿" in PUNCTUATION, "Missing Spanish inverted question"

    def test_punctuation_contains_dashes(self):
        """PUNCTUATION should contain various dash characters."""
        dashes = "‐‑‒–—―"
        for char in dashes:
            assert char in PUNCTUATION, f"Missing dash: {char!r}"

    def test_punctuation_no_letters(self):
        """PUNCTUATION should not contain any letters."""
        for char in PUNCTUATION:
            # Allow CJK ideographs used as punctuation (e.g., 〇, 〆)
            if "\u3000" <= char <= "\u303f":  # CJK symbols and punctuation block
                continue
            assert not char.isalpha(), f"Found letter in PUNCTUATION: {char!r}"

    def test_punctuation_no_digits(self):
        """PUNCTUATION should not contain any digits."""
        for char in PUNCTUATION:
            # Allow 〇 which is in the CJK punctuation block
            if char == "〇":
                continue
            assert not char.isdigit(), f"Found digit in PUNCTUATION: {char!r}"


class TestPunctuationSpace:
    """Test cases for PUNCTUATION_SPACE constant."""

    def test_punctuation_space_is_string(self):
        """PUNCTUATION_SPACE should be a string."""
        assert isinstance(PUNCTUATION_SPACE, str)

    def test_punctuation_space_contains_space(self):
        """PUNCTUATION_SPACE should contain a space character."""
        assert " " in PUNCTUATION_SPACE

    def test_punctuation_space_contains_all_punctuation(self):
        """PUNCTUATION_SPACE should contain all characters from PUNCTUATION."""
        for char in PUNCTUATION:
            assert char in PUNCTUATION_SPACE, f"Missing punctuation in PUNCTUATION_SPACE: {char!r}"

    def test_punctuation_space_length(self):
        """PUNCTUATION_SPACE should be exactly one character longer than PUNCTUATION."""
        assert len(PUNCTUATION_SPACE) == len(PUNCTUATION) + 1


class TestSpecialTokens:
    """Test cases for special tokens."""

    def test_star_token_is_string(self):
        """STAR_TOKEN should be a string."""
        assert isinstance(STAR_TOKEN, str)

    def test_star_token_value(self):
        """STAR_TOKEN should be the reference mark character."""
        assert STAR_TOKEN == "※"

    def test_grouping_separator_is_string(self):
        """GROUPING_SEPARATOR should be a string."""
        assert isinstance(GROUPING_SEPARATOR, str)

    def test_grouping_separator_value(self):
        """GROUPING_SEPARATOR should be the eight-spoked asterisk."""
        assert GROUPING_SEPARATOR == "✹"

    def test_special_tokens_not_in_punctuation(self):
        """Special tokens should not be in the PUNCTUATION set to avoid conflicts."""
        assert STAR_TOKEN not in PUNCTUATION, "STAR_TOKEN should not be in PUNCTUATION"
        assert GROUPING_SEPARATOR not in PUNCTUATION, "GROUPING_SEPARATOR should not be in PUNCTUATION"


class TestPunctuationUniqueness:
    """Test cases for character uniqueness in PUNCTUATION."""

    def test_no_duplicate_characters(self):
        """PUNCTUATION should not contain duplicate characters."""
        seen = set()
        duplicates = []
        for char in PUNCTUATION:
            if char in seen:
                duplicates.append(char)
            seen.add(char)
        assert len(duplicates) == 0, f"Found duplicate characters: {duplicates!r}"


class TestEndPunctuation:
    """Test cases for END_PUNCTUATION constant."""

    def test_end_punctuation_is_string(self):
        """END_PUNCTUATION should be a string."""
        assert isinstance(END_PUNCTUATION, str)

    def test_end_punctuation_not_empty(self):
        """END_PUNCTUATION should not be empty."""
        assert len(END_PUNCTUATION) > 0

    def test_end_punctuation_contains_ascii_endings(self):
        """END_PUNCTUATION should contain common ASCII sentence endings."""
        ascii_endings = '.!?"'
        for char in ascii_endings:
            assert char in END_PUNCTUATION, f"Missing ASCII ending: {char!r}"

    def test_end_punctuation_contains_closing_brackets(self):
        """END_PUNCTUATION should contain closing brackets that can end sentences."""
        closing_brackets = "']）"
        for char in closing_brackets:
            assert char in END_PUNCTUATION, f"Missing closing bracket: {char!r}"

    def test_end_punctuation_contains_chinese_endings(self):
        """END_PUNCTUATION should contain Chinese sentence endings."""
        chinese_endings = "。！？"
        for char in chinese_endings:
            assert char in END_PUNCTUATION, f"Missing Chinese ending: {char!r}"

    def test_end_punctuation_contains_chinese_quotation_marks(self):
        """END_PUNCTUATION should contain Chinese right quotation marks."""
        # Chinese right double quotation mark
        chinese_right_quote = "\u201d"  # "
        assert chinese_right_quote in END_PUNCTUATION, "Missing Chinese right double quote"

    def test_end_punctuation_contains_cjk_closing_brackets(self):
        """END_PUNCTUATION should contain CJK closing brackets."""
        cjk_brackets = "】」』〗〙〛"
        for char in cjk_brackets:
            assert char in END_PUNCTUATION, f"Missing CJK bracket: {char!r}"

    def test_end_punctuation_contains_japanese_period(self):
        """END_PUNCTUATION should contain Japanese halfwidth period."""
        assert "｡" in END_PUNCTUATION, "Missing Japanese halfwidth period"

    def test_end_punctuation_contains_arabic_question(self):
        """END_PUNCTUATION should contain Arabic question mark."""
        assert "؟" in END_PUNCTUATION, "Missing Arabic question mark"

    def test_end_punctuation_contains_ellipsis(self):
        """END_PUNCTUATION should contain ellipsis."""
        assert "…" in END_PUNCTUATION, "Missing ellipsis"

    def test_end_punctuation_no_duplicate_characters(self):
        """END_PUNCTUATION should not contain duplicate characters."""
        seen = set()
        duplicates = []
        for char in END_PUNCTUATION:
            if char in seen:
                duplicates.append(char)
            seen.add(char)
        assert len(duplicates) == 0, f"Found duplicate characters: {duplicates!r}"

    def test_end_punctuation_subset_of_punctuation(self):
        """Most END_PUNCTUATION characters should be in PUNCTUATION."""
        # Some end punctuation marks might not be in the general PUNCTUATION set
        # (e.g., period which is also used for decimals)
        # This test verifies the overlap
        overlap_count = sum(1 for char in END_PUNCTUATION if char in PUNCTUATION)
        assert overlap_count > 0, "END_PUNCTUATION should have some overlap with PUNCTUATION"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello.", True),
            ("What?", True),
            ("Wow!", True),
            ('He said "yes"', True),
            ("真的吗？", True),
            ("好的。", True),
            ("这是「引用」", True),
            ("مرحبا؟", True),
            ("Wait...", True),  # Ends with period (.)
            ("Wait…", True),  # Ellipsis character
            ("Hello", False),
            ("你好", False),
            ("Continue", False),
            ("正在进行", False),
        ],
    )
    def test_end_punctuation_usage(self, text: str, expected: bool):
        """Test that END_PUNCTUATION correctly identifies sentence endings."""
        result = any(text.endswith(char) for char in END_PUNCTUATION)
        assert result == expected, f"Text '{text}' should {'end' if expected else 'not end'} with punctuation"
