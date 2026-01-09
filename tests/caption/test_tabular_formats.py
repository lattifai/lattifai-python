#!/usr/bin/env python3
"""
Test suite for TSV, CSV, and AUD format reading and writing
"""

import tempfile
from pathlib import Path

from lattifai.caption import Caption, Supervision


class TestTabularFormats:
    """Test TSV, CSV, and AUD format support."""

    def test_tsv_read_with_speaker(self, tmp_path):
        """Test reading TSV format with speaker column."""
        # Create TSV file with speaker
        tsv_content = """speaker\tstart\tend\ttext
Speaker1\t1000\t3000\tFirst caption
Speaker2\t4000\t6000\tSecond caption
"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read the file
        caption = Caption.read(tsv_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2

        # Check first supervision
        assert caption.supervisions[0].text == "First caption"
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].end == 3.0
        assert caption.supervisions[0].speaker == "Speaker1"

        # Check second supervision
        assert caption.supervisions[1].text == "Second caption"
        assert caption.supervisions[1].start == 4.0
        assert caption.supervisions[1].end == 6.0
        assert caption.supervisions[1].speaker == "Speaker2"

        print(f"✓ TSV with speaker read correctly, parsed {len(caption.supervisions)} segments")

    def test_tsv_read_without_speaker(self, tmp_path):
        """Test reading TSV format without speaker column."""
        # Create TSV file without speaker
        tsv_content = """start\tend\ttext
1000\t3000\tFirst caption
4000\t6000\tSecond caption
"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read the file
        caption = Caption.read(tsv_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2

        # Check supervisions
        assert caption.supervisions[0].text == "First caption"
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].speaker is None

        assert caption.supervisions[1].text == "Second caption"
        assert caption.supervisions[1].start == 4.0
        assert caption.supervisions[1].speaker is None

        print(f"✓ TSV without speaker read correctly")

    def test_csv_read(self, tmp_path):
        """Test reading CSV format."""
        # Create CSV file
        csv_content = """start,end,text
1000,3000,First caption
4000,6000,Second caption
"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        # Read the file
        caption = Caption.read(csv_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2

        assert caption.supervisions[0].text == "First caption"
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[1].text == "Second caption"

        print(f"✓ CSV read correctly")

    def test_aud_read_with_speaker(self, tmp_path):
        """Test reading AUD format with speaker labels."""
        # Create AUD file with speaker
        aud_content = """1.0\t3.0\t[[Speaker1]]First caption
4.0\t6.0\t[[Speaker2]]Second caption
"""
        aud_file = tmp_path / "test.aud"
        aud_file.write_text(aud_content)

        # Read the file
        caption = Caption.read(aud_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2

        # Check speaker extraction from [[speaker]] format
        assert caption.supervisions[0].text == "First caption"
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[0].speaker == "Speaker1"

        assert caption.supervisions[1].text == "Second caption"
        assert caption.supervisions[1].start == 4.0
        assert caption.supervisions[1].speaker == "Speaker2"

        print(f"✓ AUD with speaker read correctly")

    def test_aud_read_without_speaker(self, tmp_path):
        """Test reading AUD format without speaker labels."""
        # Create AUD file without speaker
        aud_content = """1.0\t3.0\tFirst caption
4.0\t6.0\tSecond caption
"""
        aud_file = tmp_path / "test.aud"
        aud_file.write_text(aud_content)

        # Read the file
        caption = Caption.read(aud_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2

        assert caption.supervisions[0].text == "First caption"
        assert caption.supervisions[0].speaker is None
        assert caption.supervisions[1].text == "Second caption"
        assert caption.supervisions[1].speaker is None

        print(f"✓ AUD without speaker read correctly")

    def test_tsv_write_with_speaker(self, tmp_path):
        """Test writing TSV format with speaker."""
        # Create supervisions with speaker
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="Speaker1"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="Speaker2"),
        ]

        # Write to TSV file
        output_file = tmp_path / "output.tsv"
        caption = Caption.from_supervisions(supervisions)
        result_path = caption.write(output_file, include_speaker_in_text=True)

        assert output_file.exists()
        assert result_path == output_file

        # Read back and verify
        content = output_file.read_text()
        assert "speaker" in content.lower()
        assert "Speaker1" in content
        assert "First line" in content
        assert "Speaker2" in content
        assert "Second line" in content

        print(f"✓ TSV with speaker written correctly")

    def test_csv_write(self, tmp_path):
        """Test writing CSV format."""
        # Create supervisions
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]

        # Write to CSV file
        output_file = tmp_path / "output.csv"
        caption = Caption.from_supervisions(supervisions)
        result_path = caption.write(output_file, include_speaker_in_text=False)

        assert output_file.exists()

        # Read back and verify
        content = output_file.read_text()
        assert "First line" in content
        assert "Second line" in content
        assert "start" in content.lower()

        print(f"✓ CSV written correctly")

    def test_aud_write_with_speaker(self, tmp_path):
        """Test writing AUD format with speaker."""
        # Create supervisions with speaker
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="Speaker1"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="Speaker2"),
        ]

        # Write to AUD file
        output_file = tmp_path / "output.aud"
        caption = Caption.from_supervisions(supervisions)
        result_path = caption.write(output_file)

        assert output_file.exists()

        # Read back and verify
        content = output_file.read_text()
        assert "[[Speaker1]]" in content
        assert "[[Speaker2]]" in content
        assert "First line" in content
        assert "Second line" in content

        print(f"✓ AUD with speaker written correctly")

    def test_aud_roundtrip(self, tmp_path):
        """Test reading and writing AUD format preserves data."""
        # Create supervisions
        supervisions = [
            Supervision(text="First line", start=1.5, duration=2.5, speaker="Alice"),
            Supervision(text="Second line", start=5.0, duration=3.0, speaker="Bob"),
        ]

        # Write to AUD file
        aud_file = tmp_path / "roundtrip.aud"
        caption = Caption.from_supervisions(supervisions)
        caption.write(aud_file)

        # Read back
        caption_read = Caption.read(aud_file)

        assert len(caption_read.supervisions) == 2

        # Check first supervision
        assert caption_read.supervisions[0].text == "First line"
        assert caption_read.supervisions[0].start == 1.5
        assert abs(caption_read.supervisions[0].duration - 2.5) < 0.01
        assert caption_read.supervisions[0].speaker == "Alice"

        # Check second supervision
        assert caption_read.supervisions[1].text == "Second line"
        assert caption_read.supervisions[1].start == 5.0
        assert abs(caption_read.supervisions[1].duration - 3.0) < 0.01
        assert caption_read.supervisions[1].speaker == "Bob"

        print(f"✓ AUD roundtrip successful")

    def test_tsv_roundtrip(self, tmp_path):
        """Test reading and writing TSV format preserves data."""
        # Create supervisions
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="Alice"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="Bob"),
        ]

        # Write to TSV file
        tsv_file = tmp_path / "roundtrip.tsv"
        caption = Caption.from_supervisions(supervisions)
        caption.write(tsv_file, include_speaker_in_text=True)

        # Read back
        caption_read = Caption.read(tsv_file)

        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].text == "First line"
        assert caption_read.supervisions[0].start == 1.0
        assert caption_read.supervisions[1].text == "Second line"

        print(f"✓ TSV roundtrip successful")

    def test_normalize_text(self, tmp_path):
        """Test normalize_text option with TSV format."""
        # Create TSV with HTML entities
        tsv_content = """start\tend\ttext
1000\t3000\tFirst &amp; caption
4000\t6000\tSecond &lt;caption&gt;
"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read with normalization
        caption = Caption.read(tsv_file, normalize_text=True)

        assert "First & caption" in caption.supervisions[0].text
        assert "Second <caption>" in caption.supervisions[1].text

        print(f"✓ Text normalization works with TSV")

    def test_malformed_lines_skipped(self, tmp_path):
        """Test that malformed lines are skipped gracefully."""
        # Create TSV with some malformed lines
        tsv_content = """start\tend\ttext
1000\t3000\tValid caption
invalid\tdata\there
4000\t6000\tAnother valid caption
\t\t
"""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        # Read the file
        caption = Caption.read(tsv_file)

        # Should only get the valid captions
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Valid caption"
        assert caption.supervisions[1].text == "Another valid caption"

        print(f"✓ Malformed lines skipped correctly")


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
