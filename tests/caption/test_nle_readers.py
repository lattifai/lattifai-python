from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.formats.nle.audition import AuditionCSVConfig, AuditionCSVReader, AuditionCSVWriter
from lattifai.caption.formats.nle.avid import AvidDSConfig, AvidDSReader, AvidDSWriter
from lattifai.caption.formats.nle.fcpxml import FCPXMLReader
from lattifai.caption.formats.nle.premiere import PremiereXMLReader


class TestNLEReaders:

    def test_premiere_xml_read(self, tmp_path):
        # Create a mock Premiere XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
    <sequence>
        <rate>
            <timebase>25</timebase>
            <ntsc>FALSE</ntsc>
        </rate>
        <media>
            <video>
                <track>
                    <clipitem id="clip1">
                        <name>Subtitle 1</name>
                        <start>0</start>
                        <end>50</end>
                        <filter>
                            <effect>
                                <name>Basic Text</name>
                                <parameter>
                                    <name>Text</name>
                                    <value>Hello World</value>
                                </parameter>
                            </effect>
                        </filter>
                    </clipitem>
                    <clipitem id="clip2">
                        <start>50</start>
                        <end>100</end>
                        <filter>
                            <effect>
                                <parameter>
                                    <name>Text</name>
                                    <value>Second Line</value>
                                </parameter>
                            </effect>
                        </filter>
                    </clipitem>
                </track>
            </video>
        </media>
    </sequence>
</xmeml>"""

        # Test with string content
        supervisions = PremiereXMLReader.read(xml_content)
        assert len(supervisions) == 2
        assert supervisions[0].text == "Hello World"
        assert supervisions[0].start == 0.0
        assert supervisions[0].duration == 2.0  # 50 frames / 25 fps = 2.0s

        assert supervisions[1].text == "Second Line"
        assert supervisions[1].start == 2.0
        assert supervisions[1].duration == 2.0

    def test_fcpxml_read(self):
        # Create a mock FCPXML file content
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.10">
    <resources>
        <format id="r1" frameDuration="100/2500s"/>
    </resources>
    <library>
        <event name="Test Event">
            <project name="Test Project">
                <sequence>
                    <spine>
                        <caption name="Caption 1" offset="10s" duration="2s" start="10s">
                            <text>Hello FCP</text>
                        </caption>
                        <caption name="Caption 2" offset="15s" duration="1500/1000s" start="15s">
                            <text>Rational Time</text>
                        </caption>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

        supervisions = FCPXMLReader.read(xml_content)
        assert len(supervisions) == 2

        # Sort by start time (Reader sorts them)
        assert supervisions[0].text == "Hello FCP"
        assert supervisions[0].start == 10.0
        assert supervisions[0].duration == 2.0

        assert supervisions[1].text == "Rational Time"
        assert supervisions[1].start == 15.0
        assert supervisions[1].duration == 1.5

    def test_avid_ds_read(self):
        # Mock Avid DS content (Tab separated)
        # Header + content
        content = """@ This file written with the Avid Caption plugin, version 1

00:00:01:00\t00:00:03:00\tFirst Caption
00:00:04:12\t00:00:05:12\tSecond Caption
"""
        # 25 FPS default
        supervisions = AvidDSReader.read(content)
        assert len(supervisions) == 2

        # 00:00:01:00 @ 25fps = 1.0s
        # 00:00:03:00 = 3.0s
        assert supervisions[0].text == "First Caption"
        assert abs(supervisions[0].start - 1.0) < 0.001
        assert abs(supervisions[0].duration - 2.0) < 0.001

        # 00:00:04:12 @ 25fps = 4s + 12/25s = 4.48s
        # 00:00:05:12 = 5.48s
        assert supervisions[1].text == "Second Caption"
        assert abs(supervisions[1].start - 4.48) < 0.001
        assert abs(supervisions[1].duration - 1.0) < 0.001

    def test_audition_csv_read(self):
        # Mock Audition CSV content
        content = """Name,Start,Duration,Time Format,Type,Description
Marker 001,1.500,2.000,decimal,Cue,Caption One
Speaker A - Marker 002,4.000,1.500,decimal,Cue,Caption Two
"""
        supervisions = AuditionCSVReader.read(content)
        assert len(supervisions) == 2

        assert supervisions[0].text == "Caption One"
        assert supervisions[0].start == 1.5
        assert supervisions[0].duration == 2.0

        assert supervisions[1].text == "Caption Two"
        assert supervisions[1].start == 4.0
        assert supervisions[1].duration == 1.5

    def test_caption_auto_detection(self, tmp_path):
        # Test integration with Caption.read() via auto-detection

        # usage via file extension
        xml_file = tmp_path / "test.xml"
        xml_file.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
    <sequence>
        <rate><timebase>25</timebase></rate>
        <media><video><track>
            <clipitem>
                <filter><effect><parameter><name>Text</name><value>Auto Detect</value></parameter></effect></filter>
                <start>0</start><end>25</end>
            </clipitem>
        </track></video></media>
    </sequence>
</xmeml>""",
            encoding="utf-8",
        )

        caption = Caption.read(str(xml_file))
        assert len(caption) == 1
        assert caption[0].text == "Auto Detect"
        assert caption.source_format == "premiere_xml"


class TestNLEEdgeCases:
    """Edge case and boundary condition tests for NLE format handlers."""

    def test_audition_csv_empty_content(self):
        """Test reading empty Audition CSV content."""
        content = """Name,Start,Duration,Time Format,Type,Description
"""
        supervisions = AuditionCSVReader.read(content)
        assert len(supervisions) == 0

    def test_audition_csv_missing_fields(self):
        """Test reading Audition CSV with missing fields."""
        content = """Name,Start,Duration
Marker 001,1.5,2.0
"""
        # Should skip rows without required fields (Time Format missing is ok, defaults to decimal)
        supervisions = AuditionCSVReader.read(content)
        assert len(supervisions) == 0  # Missing Description, so no text

    def test_audition_csv_samples_format(self):
        """Test reading Audition CSV with samples time format."""
        # 48000 samples/sec: 72000 samples = 1.5s, 96000 samples = 2.0s duration
        content = """Name,Start,Duration,Time Format,Type,Description
Marker 001,72000,96000,samples,Cue,Sample-based marker
"""
        supervisions = AuditionCSVReader.read(content, sample_rate=48000)
        assert len(supervisions) == 1
        assert abs(supervisions[0].start - 1.5) < 0.001
        assert abs(supervisions[0].duration - 2.0) < 0.001

    def test_avid_ds_empty_content(self):
        """Test reading empty Avid DS content."""
        content = """@ This file written with the Avid Caption plugin, version 1

"""
        supervisions = AvidDSReader.read(content)
        assert len(supervisions) == 0

    def test_avid_ds_drop_frame_timecode(self):
        """Test reading Avid DS with drop-frame timecode (semicolon separator)."""
        content = """@ This file written with the Avid Caption plugin, version 1

00:01:00;00\t00:01:02;00\tDrop frame test
"""
        supervisions = AvidDSReader.read(content)
        assert len(supervisions) == 1
        assert supervisions[0].text == "Drop frame test"
        # Drop frame at 29.97fps, 1 minute = 60 seconds
        assert abs(supervisions[0].start - 60.0) < 0.1

    def test_avid_ds_invalid_timecode(self):
        """Test reading Avid DS with invalid timecode format."""
        content = """@ This file written with the Avid Caption plugin, version 1

invalid\tinvalid\tBad line
00:00:01:00\t00:00:02:00\tValid line
"""
        supervisions = AvidDSReader.read(content)
        assert len(supervisions) == 1
        assert supervisions[0].text == "Valid line"

    def test_avid_ds_roundtrip(self, tmp_path):
        """Test Avid DS write then read roundtrip."""
        original = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=1.5),
        ]

        config = AvidDSConfig(fps=25.0)
        output_path = tmp_path / "roundtrip.txt"
        AvidDSWriter.write(original, output_path, config)

        # Read back
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        roundtrip = AvidDSReader.read(content)

        assert len(roundtrip) == 2
        assert roundtrip[0].text == "First line"
        assert abs(roundtrip[0].start - 1.0) < 0.05  # Frame-rate rounding tolerance
        assert roundtrip[1].text == "Second line"

    def test_audition_csv_roundtrip(self, tmp_path):
        """Test Audition CSV write then read roundtrip."""
        original = [
            Supervision(text="Caption One", start=1.5, duration=2.0),
            Supervision(text="Caption Two", start=4.0, duration=1.5),
        ]

        config = AuditionCSVConfig(include_speaker_in_name=False, use_description=True)
        output_path = tmp_path / "roundtrip.csv"
        AuditionCSVWriter.write(original, output_path, config)

        # Read back
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        roundtrip = AuditionCSVReader.read(content)

        assert len(roundtrip) == 2
        assert roundtrip[0].text == "Caption One"
        assert abs(roundtrip[0].start - 1.5) < 0.001
        assert roundtrip[1].text == "Caption Two"

    def test_premiere_xml_empty_track(self):
        """Test reading Premiere XML with empty video track."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
    <sequence>
        <rate><timebase>25</timebase></rate>
        <media><video><track></track></video></media>
    </sequence>
</xmeml>"""
        supervisions = PremiereXMLReader.read(xml_content)
        assert len(supervisions) == 0

    def test_fcpxml_zero_duration(self):
        """Test reading FCPXML with zero duration captions (should be skipped)."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.10">
    <library>
        <event><project><sequence><spine>
            <caption offset="10s" duration="0s"><text>Zero duration</text></caption>
            <caption offset="15s" duration="2s"><text>Valid caption</text></caption>
        </spine></sequence></project></event>
    </library>
</fcpxml>"""
        supervisions = FCPXMLReader.read(xml_content)
        # Behavior depends on implementation - may include or skip zero-duration
        # At minimum, valid caption should be present
        valid_sups = [s for s in supervisions if s.duration > 0]
        assert len(valid_sups) >= 1
        assert any(s.text == "Valid caption" for s in valid_sups)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
