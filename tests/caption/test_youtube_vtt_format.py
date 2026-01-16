"""Tests for YouTube VTT format with word-level timestamps."""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai import Caption
from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision
from lattifai.caption.formats.youtube_vtt import YouTubeVTTFormat
from lattifai.config.caption import KaraokeConfig


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

    assert len(caption.supervisions) == 2

    # First supervision
    sup1 = caption.supervisions[0]
    assert sup1.text == "Hello world"
    alignment1 = sup1.alignment
    assert "word" in alignment1
    words1 = alignment1["word"]
    assert len(words1) == 2
    assert words1[0].symbol == "Hello"
    assert words1[0].start == 0.0
    assert words1[1].symbol == "world"
    assert words1[1].start == 0.5

    # Second supervision
    sup2 = caption.supervisions[1]
    assert sup2.text == "This is"
    alignment2 = sup2.alignment
    assert "word" in alignment2
    words2 = alignment2["word"]
    assert len(words2) == 2
    assert words2[0].symbol == "This"
    assert words2[0].start == 1.0


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

    # Write to VTT with word_level=True and karaoke_config (YouTube VTT karaoke format)
    karaoke_config = KaraokeConfig(enabled=True)
    vtt_content = caption.to_bytes(output_format="vtt", word_level=True, karaoke_config=karaoke_config).decode("utf-8")

    assert "WEBVTT" in vtt_content
    assert "<00:00:00.000><c> Hello</c>" in vtt_content
    assert "<00:00:00.600><c> world</c>" in vtt_content


def test_youtube_vtt_format_parsing():
    """Test parsing YouTube VTT format and verify content integrity.

    This test verifies that YouTube VTT files are correctly parsed, including:
    1. Filtering out 0.010-second duration cues followed by matching start times
    2. Preserving text content and timing information
    3. Maintaining word-level alignment data

    Note: Raw file has word-level timestamps and will only extract words with timestamps.
    Target file has plain text for all segments.
    """
    raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"
    target_file = "tests/data/captions/DoesFastChargingHurttheBatteryTarget.vtt"

    # Parse raw file (with 0.010 duration cues and word-level timestamps)
    raw_caption = Caption.read(raw_file)

    # Parse target file (cleaned, without 0.010 duration cues, plain text)
    target_caption = Caption.read(target_file)

    # Verify basic caption properties
    assert raw_caption is not None, "Raw caption should be successfully parsed"
    assert len(raw_caption.supervisions) > 0, "Raw caption should contain supervisions"
    assert target_caption is not None, "Target caption should be successfully parsed"
    assert len(target_caption.supervisions) > 0, "Target caption should contain supervisions"

    # Verify that raw and target have the same number of supervisions after filtering
    assert len(raw_caption.supervisions) == len(
        target_caption.supervisions
    ), f"Raw and target should have same number of supervisions after filtering: raw={len(raw_caption.supervisions)}, target={len(target_caption.supervisions)}"

    # Verify that raw file has word-level alignment
    has_alignment = sum(1 for sup in raw_caption.supervisions if sup.alignment and sup.alignment.get("word"))
    assert has_alignment > 0, "Raw caption should have word-level alignment"

    # Verify that no raw supervision has exactly 0.010 second duration
    short_durations = [sup for sup in raw_caption.supervisions if abs(sup.duration - 0.010) < 0.001]
    assert (
        len(short_durations) == 0
    ), f"Raw caption should not have 0.010-second durations, found {len(short_durations)}"

    # Verify timing, text, and speaker consistency between raw and target
    for i, (raw_sup, target_sup) in enumerate(zip(raw_caption.supervisions, target_caption.supervisions)):
        # Check start time
        assert (
            abs(raw_sup.start - target_sup.start) < 0.02
        ), f"Supervision {i}: start times should match (raw={raw_sup.start:.3f}, target={target_sup.start:.3f})"

        # Check duration
        assert (
            abs(raw_sup.duration - target_sup.duration) < 0.02
        ), f"Supervision {i}: durations should match (raw={raw_sup.duration:.3f}, target={target_sup.duration:.3f})"

        # Check text content
        assert (
            raw_sup.text == target_sup.text
        ), f"Supervision {i}: text should match (raw='{raw_sup.text}', target='{target_sup.text}')"

        # Check speaker
        assert (
            raw_sup.speaker == target_sup.speaker
        ), f"Supervision {i}: speaker should match (raw={raw_sup.speaker}, target={target_sup.speaker})"

    # Verify supervision structure for raw
    for sup in raw_caption.supervisions:
        assert sup.id is not None, "Each supervision should have an ID"
        assert sup.start >= 0, "Start time should be non-negative"
        assert sup.duration > 0, "Duration should be positive"
        assert sup.text is not None, "Each supervision should have text"
        assert sup.recording_id is not None, "Each supervision should have a recording_id"

    # Verify text integrity with sentence splitting (raw only since it has word timestamps)
    tokenizer = LatticeTokenizer(client_wrapper=None)
    tokenizer.init_sentence_splitter()
    splits = tokenizer.split_sentences(raw_caption.supervisions)

    origin_text = "".join([(sup.speaker or "").strip() + sup.text for sup in raw_caption.supervisions]).replace(" ", "")
    split_text = "".join([(sup.speaker or "").strip() + sup.text for sup in splits]).replace(" ", "")

    assert origin_text == split_text, "Text integrity should be preserved after sentence splitting"


def test_youtube_vtt_010_duration_filter():
    """Test that 0.010-second duration cues without word timestamps are filtered.

    YouTube VTT files often contain redundant cues with 0.010 second duration
    that are followed by another cue starting at the same time. These should
    be automatically filtered during parsing.
    """
    raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"

    caption = Caption.read(raw_file)

    # Check that no supervision has exactly 0.010 second duration
    for sup in caption.supervisions:
        assert (
            abs(sup.duration - 0.010) >= 0.001
        ), f"Supervision at {sup.start:.3f} should not have 0.010s duration, got {sup.duration:.3f}"


def test_youtube_vtt_word_level_alignment():
    """Test that word-level alignment is correctly extracted from YouTube VTT.

    YouTube VTT files contain word-level timestamps in the format:
    Word1<00:00:10.559><c> Word2</c><00:00:11.000><c> Word3</c>
    """
    raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"

    caption = Caption.read(raw_file)

    # Verify all supervisions have word-level alignment
    for sup in caption.supervisions:
        assert sup.alignment is not None, f"Supervision at {sup.start:.3f} should have alignment data"
        assert "word" in sup.alignment, f"Supervision at {sup.start:.3f} should have word alignment"
        word_alignments = sup.alignment["word"]
        assert len(word_alignments) > 0, f"Supervision at {sup.start:.3f} should have word alignments"

        # Verify word alignment structure
        for word in word_alignments:
            assert word.symbol is not None and word.symbol.strip(), "Word should have non-empty symbol"
            assert word.start >= 0, "Word start time should be non-negative"
            assert word.duration > 0, "Word duration should be positive"

        # Verify text starts with word symbols (may contain additional text from merged cues)
        reconstructed_text = " ".join(word.symbol for word in word_alignments)
        assert sup.text.startswith(
            reconstructed_text
        ), f"Supervision text should start with reconstructed text from words"


if __name__ == "__main__":
    test_youtube_vtt_detection()
    test_youtube_vtt_parsing()
    test_youtube_vtt_parsing_with_first_word()
    test_youtube_vtt_metadata()
    test_youtube_vtt_auto_detection()
    test_youtube_vtt_true_auto_detection()
    test_caption_write_youtube_vtt_word_level()
    test_youtube_vtt_format_parsing()
    test_youtube_vtt_010_duration_filter()
    test_youtube_vtt_word_level_alignment()
