#!/usr/bin/env python3
"""
Test suite for TXT format with timestamp parsing.
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision


class TestTxtTimestampFormat:
    """Test TXT format with [start-end] timestamp parsing."""

    def test_txt_with_timestamps(self, tmp_path):
        """Test reading TXT file with timestamp format."""
        txt_content = """[1.00-3.00] Hello world
[4.50-6.75] How are you today?
[8.00-10.50] I'm doing great, thanks!
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)

        assert len(caption.supervisions) == 3

        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].end == 3.0
        assert caption.supervisions[0].text == "Hello world"

        assert caption.supervisions[1].start == 4.5
        assert caption.supervisions[1].end == 6.75
        assert caption.supervisions[1].text == "How are you today?"

        assert caption.supervisions[2].start == 8.0
        assert caption.supervisions[2].end == 10.5
        assert caption.supervisions[2].text == "I'm doing great, thanks!"

    def test_txt_with_timestamps_and_speakers(self, tmp_path):
        """Test TXT format with both timestamps and speaker labels."""
        txt_content = """[1.00-3.00] [SPEAKER_01]: Hello world
[4.50-6.75] >> ALICE: How are you?
[8.00-10.50] BOB: I'm fine, thanks!
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)

        assert len(caption.supervisions) == 3

        # [SPEAKER_01]: format
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].end == 3.0
        assert caption.supervisions[0].speaker == "[SPEAKER_01]:"
        assert caption.supervisions[0].text == "Hello world"

        # >> SPEAKER: format
        assert caption.supervisions[1].start == 4.5
        assert caption.supervisions[1].end == 6.75
        assert caption.supervisions[1].speaker == ">> ALICE:"
        assert caption.supervisions[1].text == "How are you?"

        # NAME: format
        assert caption.supervisions[2].start == 8.0
        assert caption.supervisions[2].end == 10.5
        assert caption.supervisions[2].speaker == "BOB:"
        assert caption.supervisions[2].text == "I'm fine, thanks!"

    def test_txt_without_timestamps(self, tmp_path):
        """Test TXT format without timestamps (plain text)."""
        txt_content = """Hello world
How are you?
I'm fine!
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)

        assert len(caption.supervisions) == 3

        # Should have default timing (0.0) when no timestamp
        assert caption.supervisions[0].start == 0.0
        assert caption.supervisions[0].duration == 0.0
        assert caption.supervisions[0].text == "Hello world"

        assert caption.supervisions[1].text == "How are you?"
        assert caption.supervisions[2].text == "I'm fine!"

    def test_txt_mixed_with_and_without_timestamps(self, tmp_path):
        """Test TXT format with mixed lines (some with timestamps, some without)."""
        txt_content = """[1.00-3.00] First line with timestamp
Second line without timestamp
[5.00-7.00] Third line with timestamp
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)

        assert len(caption.supervisions) == 3

        # First line has timestamp
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].end == 3.0
        assert caption.supervisions[0].text == "First line with timestamp"

        # Second line has no timestamp (defaults to 0.0)
        assert caption.supervisions[1].start == 0.0
        assert caption.supervisions[1].text == "Second line without timestamp"

        # Third line has timestamp
        assert caption.supervisions[2].start == 5.0
        assert caption.supervisions[2].end == 7.0
        assert caption.supervisions[2].text == "Third line with timestamp"

    def test_txt_write_with_timestamps(self, tmp_path):
        """Test writing TXT format preserves timestamp format."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="ALICE"),
            Supervision(text="World", start=4.0, duration=1.5, speaker="BOB"),
            Supervision(text="Test", start=6.0, duration=2.5),
        ]

        txt_file = tmp_path / "output.txt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(txt_file, include_speaker_in_text=True)

        # Read and verify content
        content = txt_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3
        assert "[1.00-3.00]" in lines[0]
        assert "ALICE" in lines[0]
        assert "Hello" in lines[0]

        assert "[4.00-5.50]" in lines[1]
        assert "BOB" in lines[1]
        assert "World" in lines[1]

        assert "[6.00-8.50]" in lines[2]
        assert "Test" in lines[2]

    def test_txt_roundtrip_with_timestamps_and_speakers(self, tmp_path):
        """Test full roundtrip: write TXT with timestamps/speakers, then read back."""
        original_supervisions = [
            Supervision(text="First", start=1.0, duration=2.0, speaker="SPEAKER_01"),
            Supervision(text="Second", start=5.0, duration=3.0, speaker="SPEAKER_02"),
        ]

        # Write
        txt_file = tmp_path / "roundtrip.txt"
        caption = Caption.from_supervisions(original_supervisions)
        caption.write(txt_file, include_speaker_in_text=True)

        # Read back
        caption_read = Caption.read(txt_file)

        assert len(caption_read.supervisions) == 2

        # Verify first supervision
        assert caption_read.supervisions[0].start == 1.0
        assert caption_read.supervisions[0].end == 3.0
        # Note: speaker is included in text by write(), then extracted by parse_speaker_text()
        assert caption_read.supervisions[0].text == "First"
        # The speaker field will contain the format used (not just the name)
        assert caption_read.supervisions[0].speaker is not None

        # Verify second supervision
        assert caption_read.supervisions[1].start == 5.0
        assert caption_read.supervisions[1].end == 8.0
        assert caption_read.supervisions[1].text == "Second"
        assert caption_read.supervisions[1].speaker is not None

    def test_txt_empty_lines_ignored(self, tmp_path):
        """Test that empty lines are ignored."""
        txt_content = """[1.00-2.00] First

[3.00-4.00] Second


[5.00-6.00] Third
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)

        # Should only have 3 supervisions, empty lines ignored
        assert len(caption.supervisions) == 3
        assert caption.supervisions[0].text == "First"
        assert caption.supervisions[1].text == "Second"
        assert caption.supervisions[2].text == "Third"


def run_tests():
    """Run all tests."""
    print("🧪 Running TXT Timestamp Format Tests\n")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_suite = TestTxtTimestampFormat()

        print("\n📄 Testing timestamp parsing...")
        test_suite.test_txt_with_timestamps(tmp_path)
        print("  ✓ Basic timestamp parsing works")

        test_suite.test_txt_with_timestamps_and_speakers(tmp_path)
        print("  ✓ Timestamp + speaker parsing works")

        test_suite.test_txt_without_timestamps(tmp_path)
        print("  ✓ Plain text without timestamps works")

        test_suite.test_txt_mixed_with_and_without_timestamps(tmp_path)
        print("  ✓ Mixed format works")

        print("\n📄 Testing write operations...")
        test_suite.test_txt_write_with_timestamps(tmp_path)
        print("  ✓ Writing with timestamps works")

        test_suite.test_txt_roundtrip_with_timestamps_and_speakers(tmp_path)
        print("  ✓ Roundtrip with timestamps and speakers works")

        test_suite.test_txt_empty_lines_ignored(tmp_path)
        print("  ✓ Empty lines correctly ignored")

    print("\n" + "=" * 60)
    print("✅ All TXT timestamp format tests passed!")
    print("\n📝 Summary:")
    print("   • [start-end] timestamp format supported")
    print("   • Works with speaker labels")
    print("   • Backward compatible with plain text")
    print("   • Full roundtrip support")


if __name__ == "__main__":
    import sys

    try:
        run_tests()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
