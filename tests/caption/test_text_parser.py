import pytest

from lattifai.caption.text_parser import normalize_text, parse_speaker_text, parse_timestamp_text

# Test cases for parse_speaker_text
# Each tuple contains: (input_line, expected_speaker, expected_text)
SPEAKER_TEXT_CASES = [
    # Test cases for SPEAKER_PATTERN (>>, &gt;&gt;, etc.)
    (">> SPEAKER:", ">> SPEAKER:", ""),
    (">> SPEAKER: Hello world", ">> SPEAKER:", "Hello world"),
    (">> SPEAKER : Hello world", ">> SPEAKER :", "Hello world"),
    (">> SPEAKER：Hello world", ">> SPEAKER：", "Hello world"),
    ("> SPEAKER: Hello", "> SPEAKER:", "Hello"),
    ("&gt;&gt; SPEAKER: Hello", ">> SPEAKER:", "Hello"),
    ("&gt; SPEAKER: Hello", "> SPEAKER:", "Hello"),
    # ('  >> SPEAKER:  Hello  ', '>> SPEAKER:', 'Hello'),
    (">> Speaker Name: Some text here.", ">> Speaker Name:", "Some text here."),
    (">> 中文说话人：你好", ">> 中文说话人：", "你好"),
    # Test with \N replacement
    (">> SPEAKER: Hello\\NWorld", ">> SPEAKER:", "Hello World"),
    # Test cases for SPEAKER_LATTIFAI ([SPEAKER_XX])
    ("[SPEAKER_01]: Text here", "[SPEAKER_01]:", "Text here"),
    ("[SPEAKER_123]: More text", "[SPEAKER_123]:", "More text"),
    ("[SPEAKER_ABC]: With letters", "[SPEAKER_ABC]:", "With letters"),
    # ('[SPEAKER_01] : Text with space', '[SPEAKER_01] :', 'Text with space'),
    ("[SPEAKER_01]：Text with Chinese colon", "[SPEAKER_01]：", "Text with Chinese colon"),
    # Test cases for SPEAKER_PATTERN2 (ALL CAPS NAME:)
    ("NAME: Some text", "NAME:", "Some text"),
    ("FIRST LAST: Some text", "FIRST LAST:", "Some text"),
    ("SINGLE: text", "SINGLE:", "text"),
    ("A B: text", "A B:", "text"),
    # ('NAME : With space', 'NAME :', 'With space'),
    ("NAME：With Chinese colon", "NAME：", "With Chinese colon"),
    # Test cases with no speaker
    ("This is a normal line of text.", None, "This is a normal line of text."),
    ("A line with a colon: but no speaker.", None, "A line with a colon: but no speaker."),
    (">> No colon here", None, ">> No colon here"),
    ("[SPEAKER_01] No colon", None, "[SPEAKER_01] No colon"),
    ("lowercase: not a speaker", None, "lowercase: not a speaker"),
    ("A LONGER NAME THAN FIFTEEN: not a speaker", None, "A LONGER NAME THAN FIFTEEN: not a speaker"),
    ("A LONGER FIRST AND LAST NAME: not a speaker", None, "A LONGER FIRST AND LAST NAME: not a speaker"),
    # # Edge cases
    # (':: text', '::', 'text'),  # Matches SPEAKER_PATTERN
    # (': text', None, ': text'),
    # ('  ', None, '  '),
    # ('', None, ''),
]


@pytest.mark.parametrize("input_line, expected_speaker, expected_text", SPEAKER_TEXT_CASES)
def test_parse_speaker_text(input_line, expected_speaker, expected_text):
    """
    Tests the parse_speaker_text function with various inputs.
    """
    speaker, text = parse_speaker_text(normalize_text(input_line))
    assert speaker == expected_speaker, f"Failed for input: '{input_line}'"
    assert text == expected_text


# html_entities
HTML_ENTITIES_CASES = [
    ("Hello &amp; welcome", "Hello & welcome"),
    ("5 &lt; 10 &gt; 2", "5 < 10 > 2"),
    ("She said &quot;Hello&quot;", 'She said "Hello"'),
    ("It&#39;s a test", "It's a test"),
    ("Non-breaking&nbsp;space", "Non-breaking space"),
    ("Line with \\N new line", "Line with new line"),
    ("Multiple   spaces", "Multiple spaces"),
    ("Curly apostrophe don’t", "Curly apostrophe don't"),
    ("Curly apostrophe 5’s", "Curly apostrophe 5's"),
    ("Ellipsis… here", "Ellipsis here"),  # Note: ellipsis replaced with space
]


@pytest.mark.parametrize("input_text, expected_output", HTML_ENTITIES_CASES)
def test_normalize_text(input_text, expected_output):
    """
    Tests the decode_html_entities function with various inputs.
    """
    output = normalize_text(input_text)
    assert output == expected_output, f"Failed for input: '{input_text}'"


# Test cases for parse_timestamp_text
# Each tuple contains: (input_line, expected_start, expected_end, expected_text)
TIMESTAMP_TEXT_CASES = [
    # Valid timestamp formats
    ("[1.00-2.00] Hello world", 1.0, 2.0, "Hello world"),
    ("[0.50-1.50] Test text", 0.5, 1.5, "Test text"),
    ("[10.25-15.75] Multiple words here", 10.25, 15.75, "Multiple words here"),
    ("[0.00-100.99] Long duration", 0.0, 100.99, "Long duration"),
    ("[1.23-4.56] Text with numbers 123", 1.23, 4.56, "Text with numbers 123"),
    # With extra spaces
    ("[1.00-2.00]   Text with spaces  ", 1.0, 2.0, "Text with spaces"),
    # No text after timestamp
    ("[1.00-2.00]", 1.0, 2.0, ""),
    ("[5.00-6.00] ", 5.0, 6.0, ""),
    # Invalid formats - should return None, None, original_text
    ("No timestamp here", None, None, "No timestamp here"),
    ("1.00-2.00 Missing brackets", None, None, "1.00-2.00 Missing brackets"),
    ("[1.00] Only one number", None, None, "[1.00] Only one number"),
    ("[1.00-] Missing end", None, None, "[1.00-] Missing end"),
    ("[-2.00] Missing start", None, None, "[-2.00] Missing start"),
    ("[abc-def] Invalid numbers", None, None, "[abc-def] Invalid numbers"),
    ("", None, None, ""),
    # With speaker pattern after timestamp
    ("[1.00-2.00] [SPEAKER_01]: Hello", 1.0, 2.0, "[SPEAKER_01]: Hello"),
    ("[3.00-5.00] >> ALICE: How are you?", 3.0, 5.0, ">> ALICE: How are you?"),
    ("[10.00-12.00] BOB: I'm fine", 10.0, 12.0, "BOB: I'm fine"),
]


@pytest.mark.parametrize("input_line, expected_start, expected_end, expected_text", TIMESTAMP_TEXT_CASES)
def test_parse_timestamp_text(input_line, expected_start, expected_end, expected_text):
    """
    Tests the parse_timestamp_text function with various inputs.
    """
    start, end, text = parse_timestamp_text(input_line)
    assert start == expected_start, f"Failed start for input: '{input_line}'"
    assert end == expected_end, f"Failed end for input: '{input_line}'"
    assert text == expected_text, f"Failed text for input: '{input_line}'"
