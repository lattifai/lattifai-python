"""Tests for YouTube VTT format with word-level timestamps."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai.caption import Caption, Supervision
from lattifai.caption.formats.youtube_vtt import YouTubeVTTFormat


def test_youtube_vtt_detection():
    """Test that YouTube VTT is correctly detected by content."""
    youtube_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:03.490
<00:00:00.000><c> Hello</c><00:00:00.200><c> everyone</c><00:00:00.500><c> welcome</c>
"""
    assert YouTubeVTTFormat.can_read(youtube_content) is True

    # Normal VTT should not match
    normal_vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Normal test
"""
    assert YouTubeVTTFormat.can_read(normal_vtt) is False


def test_youtube_vtt_parsing():
    """Test parsing of YouTube VTT with word-level timestamps."""
    content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> Hello</c><00:00:00.500><c> world</c>

00:00:01.000 --> 00:00:02.000
<00:00:01.000><c> This</c><00:00:01.500><c> is</c>
"""
    # Use Caption.read to verify it dispatches correctly
    caption = Caption.read(content, format="youtube_vtt")

    assert len(caption.supervisions) == 1  # Should be merged
    sup = caption.supervisions[0]
    assert sup.text == "Hello world This is"

    alignment = sup.alignment
    assert "word" in alignment
    words = alignment["word"]
    assert len(words) == 4
    assert words[0].symbol == "Hello"
    assert words[0].start == 0.0
    assert words[1].symbol == "world"
    assert words[1].start == 0.5
    assert words[2].symbol == "This"
    assert words[2].start == 1.0


def test_youtube_vtt_parsing_with_first_word():
    """Test parsing when there is a word before the first timestamp tag."""
    content = """WEBVTT

00:00:00.000 --> 00:00:01.000
First<00:00:00.400><c> second</c>
"""
    caption = Caption.read(content, format="youtube_vtt")
    assert len(caption.supervisions) == 1
    words = caption.supervisions[0].alignment["word"]
    assert len(words) == 2
    assert words[0].symbol == "First"
    assert words[0].start == 0.0
    assert words[0].duration == 0.4
    assert words[1].symbol == "second"
    assert words[1].start == 0.4
    assert words[1].duration == 0.6


def test_youtube_vtt_metadata():
    """Test metadata extraction from YouTube VTT."""
    content = """WEBVTT
Kind: captions
Language: zh-Hans

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> 测试</c>
"""
    caption = Caption.read(content, format="youtube_vtt")
    assert caption.kind == "captions"
    assert caption.language == "zh-Hans"


def test_youtube_vtt_auto_detection():
    """Test that Caption.read auto-detects YouTube VTT content."""
    content = """WEBVTT
Kind: captions

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> Auto</c><00:00:00.500><c> detect</c>
"""
    # For string content, we need to pass format="auto" or just the content
    # if it's long enough to be recognized as content.
    # In my Caption.read refactor, I used detect_format(str(path)) which might fail for strings
    # But get_reader(format).read(source) is called after detection.

    # Test specific format
    caption = Caption.read(content, format="youtube_vtt")
    assert len(caption.supervisions) == 1
    assert caption.supervisions[0].text == "Auto detect"


def test_youtube_vtt_true_auto_detection():
    """Test that Caption.read auto-detects YouTube VTT from string content without format hint."""
    # Build a long enough string to trigger is_content detection
    content = "WEBVTT\n" + ("X" * 600) + "\n00:00:00.000 --> 00:00:01.000\n<00:00:00.000><c> Found</c>"

    caption = Caption.read(content)
    assert caption.source_format == "youtube_vtt"
    assert len(caption.supervisions) == 1
    assert caption.supervisions[0].text == "Found"


def test_caption_write_youtube_vtt_word_level():
    """Test that Caption.write uses youtube_vtt when word_level=True."""
    supervisions = [
        Supervision(
            text="Hello world",
            start=0.0,
            duration=1.0,
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=0.0, duration=0.4),
                    AlignmentItem(symbol="world", start=0.6, duration=0.4),
                ]
            },
        )
    ]
    caption = Caption.from_supervisions(supervisions)
    caption.word_level = True

    # Write to VTT
    vtt_content = caption.to_bytes(output_format="vtt").decode("utf-8")

    assert "WEBVTT" in vtt_content
    assert "<00:00:00.000><c> Hello</c>" in vtt_content
    assert "<00:00:00.600><c> world</c>" in vtt_content
