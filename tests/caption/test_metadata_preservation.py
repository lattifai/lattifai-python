"""Tests for caption metadata preservation during format conversions."""

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.formats.lrc import LRCFormat
from lattifai.caption.formats.pysubs2 import ASSFormat, SRTFormat, SSAFormat, VTTFormat
from lattifai.caption.formats.textgrid import TextGridFormat
from lattifai.caption.formats.ttml import TTMLFormat


class TestASSMetadataPreservation:
    """Test ASS format metadata preservation."""

    ASS_CONTENT = """[Script Info]
Title: Test Subtitle
PlayResX: 1920
PlayResY: 1080
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Custom,Impact,56,&H00FF0000,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,3,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello World
Dialogue: 1,0:00:03.50,0:00:06.00,Custom,,10,10,20,Karaoke,Styled text
"""

    def test_ass_read_preserves_global_metadata(self):
        """Test that ASS reader extracts Script Info and Styles."""
        metadata = ASSFormat.extract_metadata(self.ASS_CONTENT)

        assert "ass_info" in metadata
        assert metadata["ass_info"]["Title"] == "Test Subtitle"
        assert metadata["ass_info"]["PlayResX"] == "1920"
        assert metadata["ass_info"]["PlayResY"] == "1080"

        assert "ass_styles" in metadata
        assert "Default" in metadata["ass_styles"]
        assert "Custom" in metadata["ass_styles"]
        assert metadata["ass_styles"]["Default"]["fontname"] == "Arial"
        assert metadata["ass_styles"]["Custom"]["fontname"] == "Impact"

    def test_ass_read_preserves_event_custom(self):
        """Test that ASS reader preserves event attributes in Supervision.custom."""
        supervisions = ASSFormat.read(self.ASS_CONTENT)

        assert len(supervisions) == 2

        # First event: Default style, layer 0
        assert supervisions[0].custom["ass_style"] == "Default"
        assert supervisions[0].custom["ass_layer"] == 0
        assert supervisions[0].custom["ass_effect"] == ""

        # Second event: Custom style, layer 1, margins, effect
        assert supervisions[1].custom["ass_style"] == "Custom"
        assert supervisions[1].custom["ass_layer"] == 1
        assert supervisions[1].custom["ass_margin_l"] == 10
        assert supervisions[1].custom["ass_margin_r"] == 10
        assert supervisions[1].custom["ass_margin_v"] == 20
        assert supervisions[1].custom["ass_effect"] == "Karaoke"

    def test_ass_round_trip_preserves_metadata(self):
        """Test that ASS read->write round-trip preserves metadata."""
        # Read
        caption = Caption.from_string(self.ASS_CONTENT, format="ass")

        # Verify metadata was loaded
        assert "ass_info" in caption.metadata
        assert "ass_styles" in caption.metadata
        assert caption.metadata["ass_info"]["Title"] == "Test Subtitle"

        # Write back
        output = caption.to_string(format="ass")

        # Verify metadata in output
        assert "Title: Test Subtitle" in output
        assert "PlayResX: 1920" in output
        assert "Style: Default,Arial,48" in output
        assert "Style: Custom,Impact,56" in output
        # Verify event attributes preserved
        assert "Dialogue: 1," in output  # Layer 1 preserved

    def test_ass_write_with_external_metadata(self):
        """Test that Caption.write() accepts external metadata parameter."""
        caption = Caption(
            supervisions=[
                Supervision(text="Hello", start=0, duration=2),
            ]
        )

        external_metadata = {
            "ass_info": {"Title": "External Title", "PlayResX": "1280"},
            "ass_styles": {
                "MyStyle": {
                    "fontname": "Verdana",
                    "fontsize": 32.0,
                    "primarycolor": "&H00FFFF00",
                    "alignment": 5,
                }
            },
        }

        output = caption.write(output_format="ass", metadata=external_metadata)

        assert b"Title: External Title" in output
        assert b"PlayResX: 1280" in output
        assert b"Style: MyStyle,Verdana,32" in output


class TestSSAMetadataPreservation:
    """Test SSA format (inherits from ASS)."""

    def test_ssa_inherits_ass_metadata_handling(self):
        """Test that SSA format uses ASS's metadata methods."""
        ssa_content = """[Script Info]
Title: SSA Test

[V4 Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20

[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: Marked=0,0:00:01.00,0:00:03.00,Default,NTP,0,0,0,,Hello
"""
        supervisions = SSAFormat.read(ssa_content)
        metadata = SSAFormat.extract_metadata(ssa_content)

        assert len(supervisions) == 1
        assert "ass_info" in metadata or "ass_styles" in metadata


class TestVTTMetadataPreservation:
    """Test VTT format metadata preservation."""

    VTT_CONTENT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
Hello World

00:00:03.500 --> 00:00:06.000
This is a test
"""

    def test_vtt_read_extracts_metadata(self):
        """Test that VTT reader extracts Kind and Language."""
        caption = Caption.from_string(self.VTT_CONTENT, format="vtt")

        assert caption.metadata.get("kind") == "captions"
        assert caption.metadata.get("language") == "en"

    def test_vtt_round_trip_preserves_metadata(self):
        """Test that VTT metadata is preserved in round-trip."""
        caption = Caption.from_string(self.VTT_CONTENT, format="vtt")
        output = caption.to_string(format="vtt")

        assert "Kind: captions" in output
        assert "Language: en" in output


class TestSRTMetadataPreservation:
    """Test SRT format metadata (BOM) preservation."""

    def test_srt_bom_metadata(self):
        """Test that SRT with BOM encoding flag is preserved in write."""
        # Create supervisions directly
        supervisions = [Supervision(text="Hello", start=1.0, duration=2.0)]

        # Test that writing with encoding=utf-8-sig metadata adds BOM
        metadata = {"encoding": "utf-8-sig"}
        output = SRTFormat.to_bytes(supervisions, metadata=metadata)
        assert output.startswith(b"\xef\xbb\xbf")

        # Without the encoding metadata, should not have BOM
        output_no_bom = SRTFormat.to_bytes(supervisions, metadata={})
        assert not output_no_bom.startswith(b"\xef\xbb\xbf")


class TestLRCMetadataPreservation:
    """Test LRC format metadata preservation."""

    LRC_CONTENT = """[ar:Test Artist]
[ti:Test Song]
[al:Test Album]
[offset:100]
[by:Creator]

[00:01.00]Hello world
[00:03.00]This is a test
"""

    def test_lrc_read_extracts_metadata(self):
        """Test that LRC reader extracts metadata tags."""
        metadata = LRCFormat.extract_metadata(self.LRC_CONTENT)

        assert metadata["lrc_ar"] == "Test Artist"
        assert metadata["lrc_ti"] == "Test Song"
        assert metadata["lrc_al"] == "Test Album"
        assert metadata["lrc_offset"] == "100"
        assert metadata["lrc_by"] == "Creator"

    def test_lrc_round_trip_preserves_metadata(self):
        """Test that LRC metadata is preserved in round-trip."""
        caption = Caption.from_string(self.LRC_CONTENT, format="lrc")

        assert "lrc_ar" in caption.metadata
        assert caption.metadata["lrc_ar"] == "Test Artist"

        output = caption.to_string(format="lrc")

        assert "[ar:Test Artist]" in output
        assert "[ti:Test Song]" in output
        assert "[al:Test Album]" in output
        assert "[offset:100]" in output


class TestTTMLMetadataPreservation:
    """Test TTML format metadata preservation."""

    TTML_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:03.000">Hello World</p>
    </div>
  </body>
</tt>
"""

    def test_ttml_read_extracts_language(self):
        """Test that TTML reader extracts language."""
        metadata = TTMLFormat.extract_metadata(self.TTML_CONTENT)

        assert metadata.get("ttml_language") == "en"


class TestTextGridMetadataPreservation:
    """Test TextGrid format metadata preservation."""

    def test_textgrid_read_preserves_tier_info(self):
        """Test that TextGrid reader preserves tier info in custom."""
        # Create a simple TextGrid content
        textgrid_content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 5
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "utterances"
        xmin = 0
        xmax = 5
        intervals: size = 1
            intervals [1]:
                xmin = 1
                xmax = 3
                text = "Hello"
"""
        supervisions = TextGridFormat.read(textgrid_content)

        assert len(supervisions) == 1
        assert supervisions[0].custom["textgrid_tier"] == "utterances"
        assert supervisions[0].custom["textgrid_tier_index"] == 0

    def test_textgrid_extract_metadata(self):
        """Test that TextGrid extracts xmin/xmax and tier names."""
        textgrid_content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 10.5
        intervals: size = 0
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 10.5
        intervals: size = 0
"""
        metadata = TextGridFormat.extract_metadata(textgrid_content)

        assert metadata["textgrid_xmin"] == 0.0
        assert metadata["textgrid_xmax"] == 10.5
        assert "words" in metadata["textgrid_tiers"]
        assert "phones" in metadata["textgrid_tiers"]


class TestSentenceSplitCustomInheritance:
    """Test that sentence splitting preserves Supervision.custom."""

    def test_split_inherits_first_sup_custom(self):
        """Test that sentence splitting inherits custom from first_sup."""
        from lattifai.alignment.sentence_splitter import SentenceSplitter

        supervisions = [
            Supervision(text="Hello world.", start=0, duration=2, custom={"ass_style": "Style1", "ass_layer": 0}),
            Supervision(text="How are you?", start=2, duration=2, custom={"ass_style": "Style1", "ass_layer": 0}),
        ]

        result = SentenceSplitter._distribute_time_info(supervisions, ["Hello world.", "How are you?"])

        assert len(result) == 2
        assert result[0].custom["ass_style"] == "Style1"
        assert result[1].custom["ass_style"] == "Style1"

    def test_split_marks_conflict_when_crossing_different_customs(self):
        """Test that splitting marks conflict when crossing supervisions with different custom."""
        from lattifai.alignment.sentence_splitter import SentenceSplitter

        supervisions = [
            Supervision(text="Hello", start=0, duration=1, custom={"ass_style": "Style1"}),
            Supervision(text="world.", start=1, duration=1, custom={"ass_style": "Style2"}),
        ]

        # Split text that crosses both supervisions
        result = SentenceSplitter._distribute_time_info(supervisions, ["Hello world."])

        assert len(result) == 1
        # Should inherit from first but mark conflict
        assert result[0].custom["ass_style"] == "Style1"
        assert result[0].custom.get("_split_from_multiple") is True
        assert result[0].custom.get("_source_count") == 2


class TestCaptionWriteMetadataParameter:
    """Test Caption.write() metadata parameter behavior."""

    def test_write_merges_external_metadata(self):
        """Test that external metadata is merged with self.metadata."""
        caption = Caption(
            supervisions=[Supervision(text="Hello", start=0, duration=2)],
            metadata={"kind": "captions", "language": "en"},
        )

        # External metadata should override/supplement
        output = caption.write(output_format="vtt", metadata={"language": "de", "new_key": "value"})

        # External 'language' should override
        assert b"Language: de" in output
        # Original 'kind' should be preserved
        assert b"Kind: captions" in output

    def test_write_without_metadata_uses_self_metadata(self):
        """Test that write() uses self.metadata when no external metadata."""
        caption = Caption(
            supervisions=[Supervision(text="Hello", start=0, duration=2)],
            metadata={"kind": "subtitles", "language": "fr"},
        )

        output = caption.write(output_format="vtt")

        assert b"Kind: subtitles" in output
        assert b"Language: fr" in output
