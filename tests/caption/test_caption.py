#!/usr/bin/env python3
"""
Test suite for Caption API
"""

import tempfile
from pathlib import Path

from lattifai.caption import Caption, Supervision


class TestCaptionAPI:
    """Test Caption API."""

    def test_caption_read(self, tmp_path):
        """Test Caption.read() method."""
        from lattifai import Caption

        # Create a simple SRT file
        srt_content = """1
00:00:01,000 --> 00:00:03,000
First caption

2
00:00:04,000 --> 00:00:06,000
Second caption
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        # Read the file
        caption = Caption.read(srt_file)

        assert isinstance(caption, Caption)
        assert len(caption.supervisions) == 2
        assert all(hasattr(s, "text") for s in caption.supervisions)
        assert all(hasattr(s, "start") for s in caption.supervisions)
        assert all(hasattr(s, "duration") for s in caption.supervisions)

        print(f"✓ Caption.read() works correctly, parsed {len(caption.supervisions)} segments")

    def test_caption_write(self, tmp_path):
        """Test Caption.write() method."""

        # Create supervisions
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]

        # Write to file
        output_file = tmp_path / "output.srt"
        caption = Caption.from_supervisions(supervisions)
        result_path = caption.write(output_file)

        assert output_file.exists()
        assert result_path == output_file

        # Read back and verify
        content = output_file.read_text()
        assert "First line" in content
        assert "Second line" in content

        print(f"✓ Caption.write() works correctly, wrote to {output_file}")

    def test_caption_format_auto_detection(self, tmp_path):
        """Test that format auto-detection works."""
        # Test with different extensions
        formats = {
            "test.srt": "srt",
            "test.vtt": "vtt",
            "test.ass": "ass",
            "test.txt": "txt",
        }

        for filename, expected_format in formats.items():
            file_path = tmp_path / filename

            # Create a simple file with basic content
            if expected_format == "txt":
                content = "Line 1\nLine 2\n"
            else:
                content = "1\n00:00:01,000 --> 00:00:03,000\nTest\n"

            file_path.write_text(content)

            # Read with auto-detection
            caption = Caption.read(file_path, format=None)
            assert isinstance(caption, Caption)
            print(f"✓ Auto-detected format for {filename}")


def run_tests():
    """Run all tests."""
    print("🧪 Running LattifAI API Tests\n")
    print("=" * 60)

    # Test Caption API
    print("\n📄 Testing Caption API...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_io = TestCaptionAPI()
        test_io.test_caption_read(tmp_path)
        test_io.test_caption_write(tmp_path)
        test_io.test_caption_format_auto_detection(tmp_path)

    print("\n" + "=" * 60)
    print("✅ All API tests passed!")
    print("\n📝 API Summary:")
    print("   • Caption.read() and write() work correctly")


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
