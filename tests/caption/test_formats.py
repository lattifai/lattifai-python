#!/usr/bin/env python3
"""
Comprehensive test suite for all caption formats
"""

from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision


class TestCaptionFormats:
    """Test all supported caption formats."""

    @pytest.mark.parametrize(
        "format_ext,content_template",
        [
            ("srt", "1\n00:00:01,000 --> 00:00:03,000\n{text}\n"),
            ("vtt", "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n{text}\n"),
            ("ass", "[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{text}\n"),
            ("ssa", "[Events]\nDialogue: Marked=0,0:00:01.00,0:00:03.00,Default,NTP,0,0,0,,{text}\n"),
        ],
    )
    def test_pysubs2_format_read(self, tmp_path, format_ext, content_template):
        """Test reading various pysubs2-supported formats."""
        test_text = "Test subtitle text"
        content = content_template.format(text=test_text)

        file_path = tmp_path / f"test.{format_ext}"
        file_path.write_text(content)

        caption = Caption.read(file_path)
        assert isinstance(caption, Caption)
        assert len(caption.supervisions) > 0
        assert test_text in caption.supervisions[0].text
        print(f"✓ Read {format_ext.upper()} format successfully")

    @pytest.mark.parametrize(
        "format_ext",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "json", "TextGrid"],
    )
    def test_format_write(self, tmp_path, format_ext):
        """Test writing various caption formats."""
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / f"output.{format_ext}"

        result_path = caption.write(output_file)
        assert output_file.exists()
        assert result_path == output_file
        print(f"✓ Write {format_ext.upper()} format successfully")

    def test_sbv_format_complete(self, tmp_path):
        """Test SBV format with comprehensive scenarios."""
        # Test with multiline text
        sbv_content = """0:00:01.000,0:00:03.500
First line
Second line of same subtitle

0:00:04.000,0:00:06.500
Single line subtitle

0:00:07.000,0:00:09.000
SPEAKER ONE: Dialogue with speaker
"""
        sbv_file = tmp_path / "test.sbv"
        sbv_file.write_text(sbv_content)

        caption = Caption.read(sbv_file)
        assert len(caption.supervisions) == 3
        assert "First line Second line" in caption.supervisions[0].text
        assert caption.supervisions[2].speaker == "SPEAKER ONE:"
        print("✓ SBV multiline and speaker handling works")

    def test_txt_format_with_timestamps(self, tmp_path):
        """Test TXT format with timestamp markers."""
        txt_content = """[1.0-3.0] First line with timestamp
[4.0-6.0] SPEAKER: Second line with speaker
Plain line without timestamp
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)
        assert len(caption.supervisions) == 3
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[1].speaker == "SPEAKER:"
        print("✓ TXT format with timestamps works")

    def test_format_round_trip(self, tmp_path):
        """Test write and read back maintains data integrity."""
        original_supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="ALICE"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="BOB"),
        ]

        # JSON format uses custom structure, not pysubs2 compatible for reading
        formats_to_test = ["srt", "vtt", "sbv"]

        for fmt in formats_to_test:
            caption = Caption.from_supervisions(original_supervisions)
            output_file = tmp_path / f"test.{fmt}"

            # Write
            caption.write(output_file)

            # Read back
            caption_readback = Caption.read(output_file)

            # Verify
            assert len(caption_readback.supervisions) == len(original_supervisions)
            assert abs(caption_readback.supervisions[0].start - 1.0) < 0.1
            assert abs(caption_readback.supervisions[1].start - 4.0) < 0.1

            print(f"✓ Round-trip for {fmt.upper()} format successful")

    def test_textgrid_format(self, tmp_path):
        """Test TextGrid format writing."""
        supervisions = [
            Supervision(text="First utterance", start=0.0, duration=2.0, speaker="SPEAKER_01"),
            Supervision(text="Second utterance", start=2.5, duration=1.5, speaker="SPEAKER_02"),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.TextGrid"

        caption.write(output_file)
        assert output_file.exists()

        # Read back and verify
        content = output_file.read_text()
        assert "utterances" in content
        assert "First utterance" in content
        assert "SPEAKER_01" in content or "SPEAKER_02" in content
        print("✓ TextGrid format works correctly")

    @pytest.mark.parametrize(
        "special_chars",
        [
            "Text with 'quotes'",
            'Text with "double quotes"',
            "Text with <tags>",
            "Text with & ampersand",
            "Text with émojis 😀",
        ],
    )
    def test_special_characters_handling(self, tmp_path, special_chars):
        """Test handling of special characters in various formats."""
        supervisions = [Supervision(text=special_chars, start=1.0, duration=2.0)]
        caption = Caption.from_supervisions(supervisions)

        # Test formats that support round-trip (exclude json as it uses custom format)
        for fmt in ["srt", "vtt", "sbv"]:
            output_file = tmp_path / f"special_{fmt}.{fmt}"
            caption.write(output_file)

            # Read back
            caption_readback = Caption.read(output_file)
            # Basic check - should not crash
            assert len(caption_readback.supervisions) > 0

        print(f"✓ Special characters '{special_chars[:20]}...' handled correctly")


class TestFormatCoverage:
    """Test format coverage completeness."""

    def test_all_input_formats_defined(self):
        """Verify all input formats are properly defined."""
        from lattifai.config.caption import INPUT_CAPTION_FORMATS

        expected_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "ttml", "sami", "smi", "auto", "gemini"]

        for fmt in expected_formats:
            assert fmt in INPUT_CAPTION_FORMATS, f"Format {fmt} missing from INPUT_CAPTION_FORMATS"

        print(f"✓ All {len(expected_formats)} input formats are defined")

    def test_all_output_formats_defined(self):
        """Verify all output formats are properly defined."""
        from lattifai.config.caption import OUTPUT_CAPTION_FORMATS

        expected_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "ttml", "sami", "smi", "TextGrid", "json"]

        for fmt in expected_formats:
            assert fmt in OUTPUT_CAPTION_FORMATS, f"Format {fmt} missing from OUTPUT_CAPTION_FORMATS"

        print(f"✓ All {len(expected_formats)} output formats are defined")

    def test_format_detection_coverage(self):
        """Test that format detection works for all common extensions."""
        from lattifai.config.caption import ALL_CAPTION_FORMATS

        common_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "TextGrid", "json", "gemini"]

        for fmt in common_formats:
            assert fmt in ALL_CAPTION_FORMATS, f"Format {fmt} not in ALL_CAPTION_FORMATS"

        print(f"✓ All {len(common_formats)} common formats are detected")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
