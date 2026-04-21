"""Currency-aware tokenization tests for ``tokenize_multilingual_text``.

Regression tests for a real-world bug where a Jensen Huang / Dwarkesh Patel
podcast transcript produced ``['$', '30']`` as two separate word tokens
during Lattice-1 alignment, which in turn caused the backend to rebuild
``supervision.text`` as ``"up to$30"`` (missing space before the ``$``).

The fix keeps ``$30`` / ``$6.3`` / ``$1,000`` / ``€42`` etc. as a single
token so both the lattice word list and the reassembled text stay correct.

Mirror of ``lattifai-core/tests/backend/test_text_parser.py`` — the regex
lives in both repos and must stay in sync.
"""

from lattifai.alignment.tokenizer import tokenize_multilingual_text


class TestCurrencyTokenization:
    """Currency prefix + amount must tokenize as a single unit."""

    def test_dollar_integer(self):
        assert tokenize_multilingual_text("up to $30 billion", keep_spaces=False) == [
            "up",
            "to",
            "$30",
            "billion",
        ]

    def test_dollar_decimal(self):
        assert tokenize_multilingual_text("CoreWeave up to $6.3 billion", keep_spaces=False) == [
            "CoreWeave",
            "up",
            "to",
            "$6.3",
            "billion",
        ]

    def test_dollar_single_digit(self):
        assert tokenize_multilingual_text("invested $2 billion.", keep_spaces=False) == [
            "invested",
            "$2",
            "billion",
            ".",
        ]

    def test_thousands_separator(self):
        assert tokenize_multilingual_text("price is $1,000 exactly", keep_spaces=False) == [
            "price",
            "is",
            "$1,000",
            "exactly",
        ]
        assert tokenize_multilingual_text("$1,234,567.89", keep_spaces=False) == ["$1,234,567.89"]

    def test_multi_symbol(self):
        assert tokenize_multilingual_text("€42 £100 ¥500 ￥800 ＄9", keep_spaces=False) == [
            "€42",
            "£100",
            "¥500",
            "￥800",
            "＄9",
        ]

    def test_bare_symbol_without_amount(self):
        """A lone `$` must remain standalone — fallback to single-char token."""
        assert tokenize_multilingual_text("just a $ sign alone", keep_spaces=False) == [
            "just",
            "a",
            "$",
            "sign",
            "alone",
        ]

    def test_decimal_only(self):
        assert tokenize_multilingual_text("$30.50 each", keep_spaces=False) == [
            "$30.50",
            "each",
        ]

    def test_adjacent_punctuation(self):
        assert tokenize_multilingual_text("It cost $30, right?", keep_spaces=False) == [
            "It",
            "cost",
            "$30",
            ",",
            "right",
            "?",
        ]

    def test_attach_punctuation(self):
        """`$30` is multi-char so it is NOT absorbed by the preceding word."""
        assert tokenize_multilingual_text("up to $30.", keep_spaces=False, attach_punctuation=True) == [
            "up",
            "to",
            "$30.",
        ]

    def test_reassembly_preserves_spaces(self):
        """Joining tokens with single spaces reproduces the original phrase.

        This is the core promise that keeps the backend-rebuilt
        ``supervision.text`` correct: when ``$30`` is a single token, the
        word boundary before it carries its space, so the reassembled text
        reads ``"up to $30 billion"`` and not ``"up to$30 billion"``.
        """
        tokens = tokenize_multilingual_text("up to $30 billion", keep_spaces=False)
        assert " ".join(tokens) == "up to $30 billion"

    def test_real_podcast_sentence(self):
        """End-to-end: the exact sentence that first revealed the bug."""
        text = "It's been reported that you've done up to $30 billion in OpenAI " "and $10 billion in Anthropic."
        tokens = tokenize_multilingual_text(text, keep_spaces=False)
        # Key claims: no bare '$' tokens, every currency attached to its amount.
        assert "$" not in tokens
        assert "$30" in tokens
        assert "$10" in tokens
        # Reassembly is lossless.
        assert " ".join(tokens).replace(" ,", ",").replace(" .", ".") == text


class TestExtendedCurrencySymbols:
    """Locale coverage beyond $/€/£/¥ — INR/KRW/RUB/THB/ILS/…"""

    def test_extended_symbols(self):
        result = tokenize_multilingual_text("₹300 ₩1,000 ₽50 ฿42 ₪25", keep_spaces=False)
        assert result == ["₹300", "₩1,000", "₽50", "฿42", "₪25"]

    def test_all_supported_symbols(self):
        for sym in "$€£¥￥＄₹₩₽฿₪₺₴₦₫₱₡﷼":
            tokens = tokenize_multilingual_text(f"cost {sym}42 today", keep_spaces=False)
            assert tokens == ["cost", f"{sym}42", "today"], f"symbol {sym!r} not merged"


class TestMagnitudeSuffix:
    """K/M/B/T suffix after currency — `$30B`, `$1.5T`, `$5M`, `$100K`."""

    def test_upper_case(self):
        assert tokenize_multilingual_text("$30B ARR", keep_spaces=False) == [
            "$30B",
            "ARR",
        ]
        assert tokenize_multilingual_text("$1.5T market cap", keep_spaces=False) == ["$1.5T", "market", "cap"]
        assert tokenize_multilingual_text("$5M raise", keep_spaces=False) == [
            "$5M",
            "raise",
        ]
        assert tokenize_multilingual_text("$100K salary", keep_spaces=False) == [
            "$100K",
            "salary",
        ]

    def test_lower_case(self):
        assert tokenize_multilingual_text("spend $30b", keep_spaces=False) == [
            "spend",
            "$30b",
        ]

    def test_word_boundary_guard(self):
        """`$30Kruger` must NOT swallow the `K` — only a true suffix merges."""
        assert tokenize_multilingual_text("$30Kruger", keep_spaces=False) == [
            "$30",
            "Kruger",
        ]


class TestPercentTokenization:
    """Percent amount — `25%`, `70%`, `100%`, `1,234.5%`."""

    def test_integer(self):
        assert tokenize_multilingual_text("revenue up 25% YoY", keep_spaces=False) == ["revenue", "up", "25%", "YoY"]
        assert tokenize_multilingual_text("70% accuracy, 100% done", keep_spaces=False) == [
            "70%",
            "accuracy",
            ",",
            "100%",
            "done",
        ]

    def test_decimal(self):
        assert tokenize_multilingual_text("25.5% growth", keep_spaces=False) == ["25.5%", "growth"]

    def test_thousands(self):
        assert tokenize_multilingual_text("1,234% joke", keep_spaces=False) == ["1,234%", "joke"]


class TestStandaloneThousands:
    """Numbers with thousands separators but no currency symbol."""

    def test_standalone(self):
        assert tokenize_multilingual_text("100,000 companies", keep_spaces=False) == ["100,000", "companies"]
        assert tokenize_multilingual_text("20,000 GPU hours", keep_spaces=False) == ["20,000", "GPU", "hours"]
        assert tokenize_multilingual_text("1,234,567 items", keep_spaces=False) == ["1,234,567", "items"]

    def test_decimal(self):
        assert tokenize_multilingual_text("1,234.56 exactly", keep_spaces=False) == ["1,234.56", "exactly"]

    def test_does_not_break_comma_lists(self):
        """`2, 3, 4` stays separate — thousands rule needs `\\d{3}` right after `,`."""
        assert tokenize_multilingual_text("2, 3, 4", keep_spaces=False) == [
            "2",
            ",",
            "3",
            ",",
            "4",
        ]
