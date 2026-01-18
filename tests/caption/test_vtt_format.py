"""Tests for VTT format handler.

VTTFormat supports both standard WebVTT and YouTube VTT (with word-level timestamps).
- Reading: Auto-detects YouTube VTT format and extracts word-level alignment
- Writing: Standard VTT by default, YouTube VTT when karaoke_config.enabled=True
"""

import pytest
from lhotse.supervision import AlignmentItem

from lattifai import Caption
from lattifai.alignment.tokenizer import LatticeTokenizer
from lattifai.caption import Supervision
from lattifai.caption.formats.vtt import VTTFormat
from lattifai.config.caption import KaraokeConfig

# =============================================================================
# Standard VTT Tests
# =============================================================================


class TestStandardVTT:
    """Tests for standard WebVTT format parsing and writing."""

    def test_read_standard_vtt(self):
        """Test parsing standard VTT without word-level timestamps."""
        content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world

00:00:02.000 --> 00:00:04.000
Standard VTT
"""
        caption = Caption.read(content, format="vtt")
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Hello world"
        assert caption.supervisions[1].text == "Standard VTT"
        # Standard VTT should not have word-level alignment
        assert caption.supervisions[0].alignment is None

    def test_read_vtt_with_metadata(self):
        """Test metadata extraction from VTT header."""
        content = """WEBVTT
Kind: captions
Language: en-US

00:00:00.000 --> 00:00:02.000
Test content
"""
        caption = Caption.read(content, format="vtt")
        assert caption.kind == "captions"
        assert caption.language == "en-US"

    def test_write_standard_vtt(self):
        """Test writing standard VTT format."""
        supervisions = [
            Supervision(text="Hello world", start=0.0, duration=2.0),
            Supervision(text="Test line", start=2.0, duration=2.0),
        ]
        caption = Caption.from_supervisions(supervisions)
        vtt_content = caption.to_bytes(output_format="vtt").decode("utf-8")

        assert "WEBVTT" in vtt_content
        assert "Hello world" in vtt_content
        assert "Test line" in vtt_content
        assert "00:00:00.000 --> 00:00:02.000" in vtt_content

    def test_write_vtt_with_metadata(self):
        """Test writing VTT with metadata header."""
        supervisions = [Supervision(text="Test", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(supervisions)
        vtt_content = caption.to_bytes(
            output_format="vtt", metadata={"kind": "subtitles", "language": "zh-Hans"}
        ).decode("utf-8")

        assert "WEBVTT" in vtt_content
        assert "Kind: subtitles" in vtt_content
        assert "Language: zh-Hans" in vtt_content

    def test_is_not_youtube_vtt(self):
        """Test that standard VTT is not detected as YouTube VTT."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Normal test
"""
        assert VTTFormat._is_youtube_vtt(content) is False


# =============================================================================
# YouTube VTT Tests (Word-level timestamps)
# =============================================================================


class TestYouTubeVTT:
    """Tests for YouTube VTT format with word-level timestamps."""

    def test_detect_youtube_vtt(self):
        """Test detection of YouTube VTT format."""
        content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:03.490
<00:00:00.000><c> Hello</c><00:00:00.200><c> everyone</c><00:00:00.500><c> welcome</c>
"""
        assert VTTFormat._is_youtube_vtt(content) is True

    def test_read_youtube_vtt(self):
        """Test parsing YouTube VTT with word-level timestamps."""
        content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> Hello</c><00:00:00.500><c> world</c>

00:00:01.000 --> 00:00:02.000
<00:00:01.000><c> This</c><00:00:01.500><c> is</c>
"""
        caption = Caption.read(content, format="vtt")

        assert len(caption.supervisions) == 2

        # First supervision
        sup1 = caption.supervisions[0]
        assert sup1.text == "Hello world"
        assert sup1.alignment is not None
        assert "word" in sup1.alignment
        words1 = sup1.alignment["word"]
        assert len(words1) == 2
        assert words1[0].symbol == "Hello"
        assert words1[0].start == 0.0
        assert words1[1].symbol == "world"
        assert words1[1].start == 0.5

        # Second supervision
        sup2 = caption.supervisions[1]
        assert sup2.text == "This is"
        assert "word" in sup2.alignment
        words2 = sup2.alignment["word"]
        assert len(words2) == 2
        assert words2[0].symbol == "This"
        assert words2[0].start == 1.0

    def test_read_youtube_vtt_first_word_before_tag(self):
        """Test parsing when first word appears before timestamp tag."""
        content = """WEBVTT

00:00:00.000 --> 00:00:01.000
First<00:00:00.400><c> second</c>
"""
        caption = Caption.read(content, format="vtt")
        assert len(caption.supervisions) == 1
        words = caption.supervisions[0].alignment["word"]
        assert len(words) == 2
        assert words[0].symbol == "First"
        assert words[0].start == 0.0
        assert words[0].duration == 0.4
        assert words[1].symbol == "second"
        assert words[1].start == 0.4
        assert words[1].duration == 0.6

    def test_read_youtube_vtt_metadata(self):
        """Test metadata extraction from YouTube VTT."""
        content = """WEBVTT
Kind: captions
Language: zh-Hans

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> 测试</c>
"""
        caption = Caption.read(content, format="vtt")
        assert caption.kind == "captions"
        assert caption.language == "zh-Hans"

    def test_youtube_vtt_auto_detection(self):
        """Test that VTTFormat auto-detects YouTube VTT and extracts word alignment."""
        content = """WEBVTT
Kind: captions

00:00:00.000 --> 00:00:01.000
<00:00:00.000><c> Auto</c><00:00:00.500><c> detect</c>
"""
        caption = Caption.read(content, format="vtt")
        assert len(caption.supervisions) == 1
        assert caption.supervisions[0].text == "Auto detect"
        # Should have word-level alignment
        assert caption.supervisions[0].alignment is not None
        assert "word" in caption.supervisions[0].alignment

    def test_write_youtube_vtt_karaoke(self):
        """Test writing YouTube VTT style with karaoke enabled."""
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
        karaoke_config = KaraokeConfig(enabled=True)
        vtt_content = caption.to_bytes(output_format="vtt", word_level=True, karaoke_config=karaoke_config).decode(
            "utf-8"
        )

        assert "WEBVTT" in vtt_content
        assert "<00:00:00.000><c> Hello</c>" in vtt_content
        assert "<00:00:00.600><c> world</c>" in vtt_content

    def test_filter_010_duration_cues(self):
        """Test that 0.010-second duration cues are filtered.

        YouTube VTT files often contain redundant cues with 0.010 second duration
        that are followed by another cue starting at the same time.
        """
        raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"
        caption = Caption.read(raw_file)

        for sup in caption.supervisions:
            assert abs(sup.duration - 0.010) >= 0.001, f"Supervision at {sup.start:.3f} should not have 0.010s duration"


# =============================================================================
# Integration Tests
# =============================================================================


class TestVTTIntegration:
    """Integration tests for VTT format with real files."""

    def test_youtube_vtt_file_parsing(self):
        """Test parsing real YouTube VTT file and verify content integrity."""
        raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"
        target_file = "tests/data/captions/DoesFastChargingHurttheBatteryTarget.vtt"

        raw_caption = Caption.read(raw_file, format="vtt")
        target_caption = Caption.read(target_file)

        # Verify basic properties
        assert raw_caption is not None
        assert len(raw_caption.supervisions) > 0
        assert target_caption is not None
        assert len(target_caption.supervisions) > 0

        # Verify supervision counts match after filtering
        assert len(raw_caption.supervisions) == len(
            target_caption.supervisions
        ), f"Raw ({len(raw_caption.supervisions)}) and target ({len(target_caption.supervisions)}) counts should match"

        # Verify raw has word-level alignment
        has_alignment = sum(1 for sup in raw_caption.supervisions if sup.alignment and sup.alignment.get("word"))
        assert has_alignment > 0, "Raw caption should have word-level alignment"

        # Verify timing and text consistency
        for i, (raw_sup, target_sup) in enumerate(zip(raw_caption.supervisions, target_caption.supervisions)):
            assert abs(raw_sup.start - target_sup.start) < 0.02, f"Supervision {i}: start time mismatch"
            assert abs(raw_sup.duration - target_sup.duration) < 0.02, f"Supervision {i}: duration mismatch"
            assert raw_sup.text == target_sup.text, f"Supervision {i}: text mismatch"

    def test_youtube_vtt_word_alignment_structure(self):
        """Test word-level alignment structure from YouTube VTT."""
        raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"
        caption = Caption.read(raw_file)

        for sup in caption.supervisions:
            assert sup.alignment is not None
            assert "word" in sup.alignment
            word_alignments = sup.alignment["word"]
            assert len(word_alignments) > 0

            for word in word_alignments:
                assert word.symbol is not None and word.symbol.strip()
                assert word.start >= 0
                assert word.duration > 0

            # Text should start with reconstructed words
            reconstructed = " ".join(word.symbol for word in word_alignments)
            assert sup.text.startswith(reconstructed)

    def test_sentence_splitting_preserves_text(self):
        """Test that sentence splitting preserves text integrity."""
        raw_file = "tests/data/captions/DoesFastChargingHurttheBatteryRaw.vtt"
        caption = Caption.read(raw_file)

        tokenizer = LatticeTokenizer(client_wrapper=None)
        tokenizer.init_sentence_splitter()
        splits = tokenizer.split_sentences(caption.supervisions)

        origin_text = "".join([(sup.speaker or "").strip() + sup.text for sup in caption.supervisions]).replace(" ", "")
        split_text = "".join([(sup.speaker or "").strip() + sup.text for sup in splits]).replace(" ", "")

        assert origin_text == split_text


# =============================================================================
# Main entry point for direct execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
