import pytest

from lattifai.subtitle.text_parser import normalize_text, parse_speaker_text

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
