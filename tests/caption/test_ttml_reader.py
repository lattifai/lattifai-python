"""Tests for TTML reader functionality."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption.formats.ttml import TTMLFormat
from lattifai.caption.supervision import Supervision


class TestTTMLReader:
    """Test TTML reading capabilities."""

    def test_parse_time_formats(self):
        """Test parsing of different time formats."""
        # Clock time
        assert TTMLFormat._parse_ttml_time("00:00:10.500") == 10.5
        assert TTMLFormat._parse_ttml_time("01:00:00.000") == 3600.0
        # HH:MM:SS:FF (assuming 30fps fallback)
        assert abs(TTMLFormat._parse_ttml_time("00:00:01:15") - 1.5) < 0.001

        # Offset time
        assert TTMLFormat._parse_ttml_time("10.5s") == 10.5
        assert TTMLFormat._parse_ttml_time("500ms") == 0.5
        # Frames (assuming 30fps fallback)
        assert abs(TTMLFormat._parse_ttml_time("30f") - 1.0) < 0.001

        # Simple float
        assert TTMLFormat._parse_ttml_time("10.5") == 10.5

    def test_read_simple_ttml(self):
        """Test reading simple TTML content."""
        content = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xml:lang="en">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:02.500">Hello world</p>
      <p begin="00:00:03.000" dur="2s">Second line</p>
    </div>
  </body>
</tt>"""

        sups = TTMLFormat.read(content)
        assert len(sups) == 2

        assert sups[0].text == "Hello world"
        assert sups[0].start == 1.0
        assert sups[0].duration == 1.5

        assert sups[1].text == "Second line"
        assert sups[1].start == 3.0
        assert sups[1].duration == 2.0

    def test_read_word_level_spans(self):
        """Test reading word-level spans."""
        content = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xml:lang="en">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:02.000">
        <span begin="00:00:01.000" end="00:00:01.500">Hello</span>
        <span begin="00:00:01.500" end="00:00:02.000">Word</span>
      </p>
    </div>
  </body>
</tt>"""

        sups = TTMLFormat.read(content)
        assert len(sups) == 1

        sup = sups[0]
        assert sup.text == "Hello Word"
        assert sup.start == 1.0
        assert sup.duration == 1.0

        assert sup.alignment is not None
        assert "word" in sup.alignment
        words = sup.alignment["word"]
        assert len(words) == 2

        assert words[0].symbol == "Hello"
        assert words[0].start == 1.0
        assert words[0].duration == 0.5

        assert words[1].symbol == "Word"
        assert words[1].start == 1.5
        assert words[1].duration == 0.5

    def test_read_nested_spans_and_metadata(self):
        """Test reading agents/metadata and handling untimed spans."""
        content = """<?xml version="1.0" encoding="utf-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:ttp="http://www.w3.org/ns/ttml#parameter" xml:lang="en">
  <body>
    <div>
      <p begin="10s" end="20s" ttp:agent="Speaker 1">
        <span>Just some text</span>
      </p>
    </div>
  </body>
</tt>"""

        sups = TTMLFormat.read(content)
        assert len(sups) == 1
        assert sups[0].text == "Just some text"
        assert sups[0].start == 10.0
        assert sups[0].duration == 10.0
        assert sups[0].speaker == "Speaker 1"
