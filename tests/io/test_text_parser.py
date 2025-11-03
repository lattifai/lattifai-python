import sys
from pathlib import Path

import pytest

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lattifai.io.text_parser import parse_speaker_text

# Test cases for parse_speaker_text
# Each tuple contains: (input_line, expected_speaker, expected_text)
test_cases = [
    # Test cases for SPEAKER_PATTERN (>>, &gt;&gt;, etc.)
    ('>> SPEAKER:', '>> SPEAKER:', ''),
    ('>> SPEAKER: Hello world', '>> SPEAKER:', 'Hello world'),
    ('>> SPEAKER : Hello world', '>> SPEAKER :', 'Hello world'),
    ('>> SPEAKER：Hello world', '>> SPEAKER：', 'Hello world'),
    ('> SPEAKER: Hello', '> SPEAKER:', 'Hello'),
    ('&gt;&gt; SPEAKER: Hello', '&gt;&gt; SPEAKER:', 'Hello'),
    ('&gt; SPEAKER: Hello', '&gt; SPEAKER:', 'Hello'),
    # ('  >> SPEAKER:  Hello  ', '>> SPEAKER:', 'Hello'),
    ('>> Speaker Name: Some text here.', '>> Speaker Name:', 'Some text here.'),
    ('>> 中文说话人：你好', '>> 中文说话人：', '你好'),
    # Test with \N replacement
    ('>> SPEAKER: Hello\\NWorld', '>> SPEAKER:', 'Hello World'),
    # Test cases for SPEAKER_LATTIFAI ([SPEAKER_XX])
    ('[SPEAKER_01]: Text here', '[SPEAKER_01]:', 'Text here'),
    ('[SPEAKER_123]: More text', '[SPEAKER_123]:', 'More text'),
    ('[SPEAKER_ABC]: With letters', '[SPEAKER_ABC]:', 'With letters'),
    # ('[SPEAKER_01] : Text with space', '[SPEAKER_01] :', 'Text with space'),
    ('[SPEAKER_01]：Text with Chinese colon', '[SPEAKER_01]：', 'Text with Chinese colon'),
    # Test cases for SPEAKER_PATTERN2 (ALL CAPS NAME:)
    ('NAME: Some text', 'NAME:', 'Some text'),
    ('FIRST LAST: Some text', 'FIRST LAST:', 'Some text'),
    ('SINGLE: text', 'SINGLE:', 'text'),
    ('A B: text', 'A B:', 'text'),
    # ('NAME : With space', 'NAME :', 'With space'),
    ('NAME：With Chinese colon', 'NAME：', 'With Chinese colon'),
    # Test cases with no speaker
    ('This is a normal line of text.', None, 'This is a normal line of text.'),
    ('A line with a colon: but no speaker.', None, 'A line with a colon: but no speaker.'),
    ('>> No colon here', None, '>> No colon here'),
    ('[SPEAKER_01] No colon', None, '[SPEAKER_01] No colon'),
    ('lowercase: not a speaker', None, 'lowercase: not a speaker'),
    ('A LONGER NAME THAN FIFTEEN: not a speaker', None, 'A LONGER NAME THAN FIFTEEN: not a speaker'),
    ('A LONGER FIRST AND LAST NAME: not a speaker', None, 'A LONGER FIRST AND LAST NAME: not a speaker'),
    # # Edge cases
    # (':: text', '::', 'text'),  # Matches SPEAKER_PATTERN
    # (': text', None, ': text'),
    # ('  ', None, '  '),
    # ('', None, ''),
]


@pytest.mark.parametrize('input_line, expected_speaker, expected_text', test_cases)
def test_parse_speaker_text(input_line, expected_speaker, expected_text):
    """
    Tests the parse_speaker_text function with various inputs.
    """
    speaker, text = parse_speaker_text(input_line)
    assert speaker == expected_speaker, f"Failed for input: '{input_line}'"
    assert text == expected_text
