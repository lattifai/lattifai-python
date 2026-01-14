"""Test professional NLE format writers.

Tests for:
- Avid DS format
- FCPXML (Final Cut Pro)
- Premiere Pro XML
- Adobe Audition CSV markers
- EdiMarker CSV (Pro Tools)
- TTML/IMSC1/EBU-TT-D
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import (
    AuditionCSVConfig,
    AuditionCSVWriter,
    AvidDSConfig,
    AvidDSWriter,
    Caption,
    EdiMarkerConfig,
    EdiMarkerWriter,
    FCPXMLConfig,
    FCPXMLWriter,
    PremiereXMLConfig,
    PremiereXMLWriter,
    Supervision,
    TTMLConfig,
    TTMLWriter,
)
from lattifai.caption.utils import (
    CollisionMode,
    TimecodeOffset,
    apply_timecode_offset,
    detect_overlaps,
    format_srt_timestamp,
    generate_srt_content,
    resolve_overlaps,
    split_long_lines,
)


# Test fixtures
@pytest.fixture
def sample_supervisions():
    """Create sample supervisions for testing."""
    return [
        Supervision(text="Hello, this is the first caption.", start=0.0, duration=2.0, speaker="Speaker1"),
        Supervision(text="And this is the second one.", start=2.5, duration=2.0, speaker="Speaker2"),
        Supervision(text="Finally, the third caption.", start=5.0, duration=2.0, speaker="Speaker1"),
    ]


@pytest.fixture
def overlapping_supervisions():
    """Create overlapping supervisions for testing collision resolution."""
    return [
        Supervision(text="First line", start=0.0, duration=3.0, speaker="A"),
        Supervision(text="Overlapping line", start=2.0, duration=2.0, speaker="B"),
        Supervision(text="Third line", start=5.0, duration=2.0, speaker="A"),
    ]


class TestAvidDSWriter:
    """Tests for Avid DS format writer."""

    def test_seconds_to_timecode_25fps(self):
        """Test timecode conversion at 25fps."""
        tc = AvidDSWriter.seconds_to_timecode(3661.5, fps=25.0)
        # 3661.5 seconds = 1 hour, 1 minute, 1 second, 12.5 frames (rounds to 13)
        assert tc == "01:01:01:13"

    def test_seconds_to_timecode_29_97fps_ndf(self):
        """Test non-drop-frame timecode at 29.97fps."""
        tc = AvidDSWriter.seconds_to_timecode(60.0, fps=29.97, drop_frame=False)
        # At 29.97fps, 60 seconds = ~1798 frames, which is 59 seconds + 29 frames
        assert "00:59" in tc or "01:00" in tc  # May vary slightly due to frame rate

    def test_wrap_text(self):
        """Test text wrapping for broadcast safety."""
        long_text = "This is a very long line that exceeds the forty character limit for broadcast"
        wrapped = AvidDSWriter.wrap_text(long_text, max_length=40)
        assert all(len(line) <= 40 for line in wrapped)
        assert len(wrapped) >= 2

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to bytes."""
        config = AvidDSConfig(fps=25.0)
        content = AvidDSWriter.to_bytes(sample_supervisions, config)

        assert b"@ This file written with the Avid Caption plugin" in content
        assert b"Hello" in content

    def test_write_to_file(self, sample_supervisions, tmp_path):
        """Test writing supervisions to file."""
        output_path = tmp_path / "output_avid.txt"
        config = AvidDSConfig(fps=25.0)
        result = AvidDSWriter.write(sample_supervisions, output_path, config)

        assert result.exists()
        content = result.read_text()
        assert "@ This file written with the Avid Caption plugin" in content


class TestFCPXMLWriter:
    """Tests for FCPXML format writer."""

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to FCPXML bytes."""
        config = FCPXMLConfig()
        content = FCPXMLWriter.to_bytes(sample_supervisions, config)

        assert b'<fcpxml version="1.10">' in content
        assert b"<caption" in content

    def test_speaker_to_role_mapping(self, sample_supervisions):
        """Test speaker-to-role mapping."""
        config = FCPXMLConfig(map_speakers_to_roles=True)
        content = FCPXMLWriter.to_bytes(sample_supervisions, config)

        assert b"Speaker1" in content or b"Dialogue" in content

    def test_write_bundle(self, sample_supervisions, tmp_path):
        """Test writing FCPXML bundle (.fcpxmld)."""
        output_path = tmp_path / "output.fcpxmld"
        config = FCPXMLConfig(use_bundle=True)
        result = FCPXMLWriter.write(sample_supervisions, output_path, config)

        assert result.is_dir()
        assert (result / "Info.fcpxml").exists()


class TestPremiereXMLWriter:
    """Tests for Premiere Pro XML format writer."""

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to Premiere XML bytes."""
        config = PremiereXMLConfig()
        content = PremiereXMLWriter.to_bytes(sample_supervisions, config)

        assert b'<xmeml version="4">' in content
        assert b"<sequence>" in content
        assert b"<clipitem" in content

    def test_separate_speaker_tracks(self, sample_supervisions):
        """Test speaker separation to different tracks."""
        config = PremiereXMLConfig(separate_speaker_tracks=True)
        content = PremiereXMLWriter.to_bytes(sample_supervisions, config)

        # Should have multiple tracks for different speakers
        assert content.count(b"<track>") >= 2

    def test_write_to_file(self, sample_supervisions, tmp_path):
        """Test writing to file."""
        output_path = tmp_path / "output.xml"
        config = PremiereXMLConfig()
        result = PremiereXMLWriter.write(sample_supervisions, output_path, config)

        assert result.exists()
        assert result.suffix == ".xml"


class TestAuditionCSVWriter:
    """Tests for Adobe Audition CSV format writer."""

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to Audition CSV bytes."""
        config = AuditionCSVConfig()
        content = AuditionCSVWriter.to_bytes(sample_supervisions, config)

        assert b"Name,Start,Duration,Time Format,Type,Description" in content
        assert b"Cue" in content

    def test_decimal_time_format(self, sample_supervisions):
        """Test decimal time format."""
        config = AuditionCSVConfig(time_format="decimal")
        content = AuditionCSVWriter.to_bytes(sample_supervisions, config)

        assert b"decimal" in content

    def test_include_speaker_in_name(self, sample_supervisions):
        """Test speaker name inclusion."""
        config = AuditionCSVConfig(include_speaker_in_name=True)
        content = AuditionCSVWriter.to_bytes(sample_supervisions, config)

        assert b"Speaker1" in content


class TestEdiMarkerWriter:
    """Tests for EdiMarker CSV format writer."""

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to EdiMarker CSV bytes."""
        config = EdiMarkerConfig()
        content = EdiMarkerWriter.to_bytes(sample_supervisions, config)

        assert b"Name,Start,End,Text" in content

    def test_timecode_format(self, sample_supervisions):
        """Test timecode format in output."""
        config = EdiMarkerConfig()
        content = EdiMarkerWriter.to_bytes(sample_supervisions, config, fps=24.0)

        # Should have timecode format
        assert b":" in content


class TestTTMLWriter:
    """Tests for TTML format writer."""

    def test_write_to_bytes(self, sample_supervisions):
        """Test writing supervisions to TTML bytes."""
        config = TTMLConfig()
        content = TTMLWriter.to_bytes(sample_supervisions, config)

        assert b"<tt" in content
        assert b"<body>" in content
        assert b"<p " in content

    def test_imsc1_profile(self, sample_supervisions):
        """Test IMSC1 profile output."""
        config = TTMLConfig(profile="imsc1")
        content = TTMLWriter.to_bytes(sample_supervisions, config)

        assert b"imsc1" in content or b"profile" in content

    def test_ebu_tt_d_profile(self, sample_supervisions):
        """Test EBU-TT-D profile output."""
        config = TTMLConfig(profile="ebu-tt-d")
        content = TTMLWriter.to_bytes(sample_supervisions, config)

        assert b"ebu" in content or b"profile" in content

    def test_write_imsc1_shorthand(self, sample_supervisions, tmp_path):
        """Test write_imsc1 convenience method."""
        output_path = tmp_path / "output.ttml"
        result = TTMLWriter.write_imsc1(sample_supervisions, output_path)

        assert result.exists()

    def test_speaker_format_in_ttml(self, sample_supervisions, tmp_path):
        """Test that speaker format uses 'speaker ' instead of 'speaker: '."""
        config = TTMLConfig(profile="imsc1")
        content = TTMLWriter.to_bytes(sample_supervisions, config)

        # New format uses "speaker " not "speaker: "
        assert b"Speaker1 " in content
        assert b"Speaker1: " not in content

    def test_speaker_span_element(self, sample_supervisions):
        """Test speaker is rendered in bold span element."""
        config = TTMLConfig(profile="imsc1")
        content = TTMLWriter.to_bytes(sample_supervisions, config)

        # Should contain span elements with bold font weight
        assert b"<" in content  # XML structure
        assert b"fontWeight" in content
        assert b"bold" in content

    def test_ttml_no_speaker_inclusion(self, tmp_path):
        """Test TTML output without speaker inclusion."""
        supervisions = [
            Supervision(text="Test text", start=0.0, duration=2.0, speaker="Alice"),
        ]

        config = TTMLConfig(profile="imsc1")
        content = TTMLWriter.to_bytes(supervisions, include_speaker=False, config=config)

        # Should not include speaker name in output
        assert b"Alice " not in content
        assert b"Test text" in content


class TestTimecodeOffset:
    """Tests for timecode offset functionality."""

    def test_total_seconds(self):
        """Test total seconds calculation."""
        offset = TimecodeOffset(hours=1, minutes=30, seconds=15.5, fps=25.0)
        assert offset.total_seconds == pytest.approx(5415.5, rel=0.01)

    def test_from_timecode(self):
        """Test creating offset from timecode string."""
        offset = TimecodeOffset.from_timecode("01:30:15:12", fps=25.0)
        assert offset.hours == 1
        assert offset.minutes == 30
        assert offset.frames == 12

    def test_broadcast_start(self):
        """Test broadcast start offset (01:00:00:00)."""
        offset = TimecodeOffset.broadcast_start()
        assert offset.hours == 1
        assert offset.total_seconds == 3600

    def test_apply_offset(self, sample_supervisions):
        """Test applying offset to supervisions."""
        offset = TimecodeOffset(hours=1)
        result = apply_timecode_offset(sample_supervisions, offset)

        assert result[0].start == pytest.approx(3600.0, rel=0.01)
        assert result[1].start == pytest.approx(3602.5, rel=0.01)


class TestOverlapResolution:
    """Tests for overlap resolution functionality."""

    def test_detect_overlaps(self, overlapping_supervisions):
        """Test overlap detection."""
        overlaps = detect_overlaps(overlapping_supervisions)
        assert len(overlaps) == 1
        assert overlaps[0] == (0, 1)

    def test_resolve_overlaps_merge(self, overlapping_supervisions):
        """Test merge mode for overlap resolution."""
        result = resolve_overlaps(overlapping_supervisions, mode=CollisionMode.MERGE)

        # First two should be merged
        assert len(result) == 2

    def test_resolve_overlaps_trim(self, overlapping_supervisions):
        """Test trim mode for overlap resolution."""
        result = resolve_overlaps(overlapping_supervisions, mode=CollisionMode.TRIM)

        # All should remain but first is trimmed
        assert len(result) == 3
        assert result[0].end <= result[1].start + 0.1  # Small threshold

    def test_resolve_overlaps_keep(self, overlapping_supervisions):
        """Test keep mode (no resolution)."""
        result = resolve_overlaps(overlapping_supervisions, mode=CollisionMode.KEEP)

        assert len(result) == 3


class TestSRTUtils:
    """Tests for SRT format utilities."""

    def test_format_srt_timestamp(self):
        """Test SRT timestamp formatting."""
        ts = format_srt_timestamp(3661.5)
        assert ts == "01:01:01,500"

    def test_format_srt_timestamp_comma_separator(self):
        """Test that SRT uses comma for milliseconds."""
        ts = format_srt_timestamp(1.234)
        assert "," in ts
        assert ts == "00:00:01,234"

    def test_generate_srt_with_bom(self, sample_supervisions):
        """Test SRT generation with BOM."""
        content = generate_srt_content(sample_supervisions, use_bom=True)

        # UTF-8 BOM bytes
        assert content.startswith(b"\xef\xbb\xbf")

    def test_generate_srt_without_bom(self, sample_supervisions):
        """Test SRT generation without BOM."""
        content = generate_srt_content(sample_supervisions, use_bom=False)

        assert not content.startswith(b"\xef\xbb\xbf")


class TestSplitLongLines:
    """Tests for line splitting functionality."""

    def test_split_long_text(self):
        """Test splitting long text."""
        sups = [
            Supervision(
                text="This is a very long caption text that needs to be split into multiple parts",
                start=0.0,
                duration=5.0,
            )
        ]
        result = split_long_lines(sups, max_chars_per_line=40, max_lines=2)

        # Should be split into multiple supervisions
        assert len(result) >= 1
        for sup in result:
            lines = sup.text.split("\n")
            assert all(len(line) <= 40 for line in lines)


class TestCaptionIntegration:
    """Integration tests for Caption class with professional formats."""

    def test_write_avid_ds_via_caption(self, sample_supervisions, tmp_path):
        """Test writing Avid DS format via Caption class using explicit format."""
        caption = Caption(supervisions=sample_supervisions)
        output_path = tmp_path / "output.avid_ds"

        # Use explicit output_format parameter
        caption.write(output_path, output_format="avid_ds")

        # Check bytes were written correctly
        content = output_path.read_bytes()
        assert b"@ This file written with the Avid Caption plugin" in content

    def test_write_fcpxml_via_caption(self, sample_supervisions, tmp_path):
        """Test writing FCPXML via Caption class."""
        caption = Caption(supervisions=sample_supervisions)

        # Write with explicit format to ensure FCPXML is used
        content = caption.to_bytes(output_format="fcpxml")
        assert b"fcpxml" in content

    def test_to_bytes_avid_ds(self, sample_supervisions):
        """Test converting to Avid DS bytes via Caption class."""
        caption = Caption(supervisions=sample_supervisions)
        content = caption.to_bytes(output_format="avid_ds")

        assert b"@ This file written with the Avid Caption plugin" in content

    def test_to_bytes_fcpxml(self, sample_supervisions):
        """Test converting to FCPXML bytes via Caption class."""
        caption = Caption(supervisions=sample_supervisions)
        content = caption.to_bytes(output_format="fcpxml")

        assert b"fcpxml" in content

    def test_to_bytes_premiere_xml(self, sample_supervisions):
        """Test converting to Premiere XML bytes via Caption class."""
        caption = Caption(supervisions=sample_supervisions)
        content = caption.to_bytes(output_format="premiere_xml")

        assert b"xmeml" in content

    def test_to_bytes_audition_csv(self, sample_supervisions):
        """Test converting to Audition CSV bytes via Caption class."""
        caption = Caption(supervisions=sample_supervisions)
        content = caption.to_bytes(output_format="audition_csv")

        assert b"Name,Start,Duration" in content

    def test_to_bytes_imsc1(self, sample_supervisions):
        """Test converting to IMSC1 TTML bytes via Caption class."""
        caption = Caption(supervisions=sample_supervisions)
        content = caption.to_bytes(output_format="imsc1")

        assert b"<tt" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
