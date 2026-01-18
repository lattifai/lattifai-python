#!/usr/bin/env python3
"""
Test suite for speaker label roundtrip consistency
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision


class TestSpeakerRoundtrip:
    """Test speaker label preservation across read/write operations."""

    def test_srt_speaker_roundtrip_lattifai_format(self, tmp_path):
        """Test [SPEAKER]: format roundtrip in SRT."""
        # Create supervisions with speaker labels
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="[SPEAKER_01]:"),
            Supervision(text="How are you", start=4.0, duration=2.0, speaker="ALICE:"),
            Supervision(text="I'm fine", start=7.0, duration=2.0, speaker="BOB:"),
        ]

        # Write to SRT file
        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file, include_speaker_in_text=True)

        # Read back
        caption_read = Caption.read(srt_file)

        # Verify speaker labels are preserved
        assert len(caption_read.supervisions) == 3
        # Note: parse_speaker_text returns the full prefix including colon
        assert caption_read.supervisions[0].speaker == "[SPEAKER_01]:"
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[1].speaker == "ALICE:"
        assert caption_read.supervisions[1].text == "How are you"
        assert caption_read.supervisions[2].speaker == "BOB:"
        assert caption_read.supervisions[2].text == "I'm fine"

        print("✓ SRT [SPEAKER]: format roundtrip successful")

    def test_vtt_speaker_roundtrip(self, tmp_path):
        """Test speaker roundtrip in VTT format."""
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="[SPEAKER_00]:"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="[SPEAKER_01]:"),
        ]

        # Write to VTT file
        vtt_file = tmp_path / "test.vtt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(vtt_file, include_speaker_in_text=True)

        # Read back
        caption_read = Caption.read(vtt_file)

        # Verify
        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].speaker == "[SPEAKER_00]:"
        assert caption_read.supervisions[0].text == "First line"
        assert caption_read.supervisions[1].speaker == "[SPEAKER_01]:"
        assert caption_read.supervisions[1].text == "Second line"

        print("✓ VTT speaker roundtrip successful")

    def test_txt_speaker_roundtrip(self, tmp_path):
        """Test speaker roundtrip in TXT format.

        Note: New format uses "speaker text" instead of "[speaker]: text".
        """
        supervisions = [
            Supervision(text="Line one", start=1.0, duration=2.0, speaker="[NARRATOR]:"),
            Supervision(text="Line two", start=4.0, duration=2.0, speaker="[ALICE]:"),
        ]

        # Write to TXT file
        txt_file = tmp_path / "test.txt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(txt_file, include_speaker_in_text=True)

        # Read and check content format - new format: "speaker text"
        content = txt_file.read_text()
        assert "[NARRATOR]: Line one" in content
        assert "[ALICE]: Line two" in content

        print("✓ TXT speaker format correct")

    def test_speaker_format_parsing(self, tmp_path):
        """Test that different speaker formats are correctly parsed."""
        # Create SRT with different speaker formats
        srt_content = """1
00:00:01,000 --> 00:00:03,000
[SPEAKER_01]: First speaker format

2
00:00:04,000 --> 00:00:06,000
>> ALICE: Second speaker format

3
00:00:07,000 --> 00:00:09,000
BOB: Third speaker format
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        # Read and verify all formats are parsed
        caption = Caption.read(srt_file)

        assert len(caption.supervisions) == 3

        # [SPEAKER_01]: format
        assert caption.supervisions[0].speaker == "[SPEAKER_01]:"
        assert caption.supervisions[0].text == "First speaker format"

        # >> ALICE: format
        assert caption.supervisions[1].speaker == ">> ALICE:"
        assert caption.supervisions[1].text == "Second speaker format"

        # BOB: format (Gemini style)
        assert caption.supervisions[2].speaker == "BOB:"
        assert caption.supervisions[2].text == "Third speaker format"

        print("✓ Multiple speaker formats parsed correctly")

    def test_write_without_speaker(self, tmp_path):
        """Test writing without speaker labels."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="[ALICE]:"),
            Supervision(text="World", start=4.0, duration=2.0, speaker="[BOB]:"),
        ]

        # Write without speaker
        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file, include_speaker_in_text=False)

        # Read back
        caption_read = Caption.read(srt_file)

        # Speaker info should still be in event.name but not in text
        assert caption_read.supervisions[0].text == "Hello"
        assert caption_read.supervisions[1].text == "World"
        # pysubs2 stores speaker in event.name, which gets assigned to supervision.speaker
        # When include_speaker_in_text=False, only event.name is used (no colon format)
        assert not caption_read.supervisions[0].speaker
        assert not caption_read.supervisions[1].speaker

        print("✓ Write without speaker in text successful")


def run_tests():
    """Run all tests."""
    print("🧪 Running Speaker Roundtrip Tests\n")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_suite = TestSpeakerRoundtrip()

        print("\n📄 Testing speaker roundtrip...")
        test_suite.test_srt_speaker_roundtrip_lattifai_format(tmp_path)
        test_suite.test_vtt_speaker_roundtrip(tmp_path)
        test_suite.test_txt_speaker_roundtrip(tmp_path)

        print("\n📄 Testing speaker format parsing...")
        test_suite.test_speaker_format_parsing(tmp_path)
        test_suite.test_write_without_speaker(tmp_path)

    print("\n" + "=" * 60)
    print("✅ All speaker roundtrip tests passed!")
    print("\n📝 Summary:")
    print("   • [SPEAKER]: format used for writing")
    print("   • All formats (>>, [SPEAKER]:, NAME:) correctly parsed")
    print("   • Speaker labels preserved in roundtrip")
    print("   • Supports writing with/without speaker in text")


if __name__ == "__main__":
    import sys

    try:
        run_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
