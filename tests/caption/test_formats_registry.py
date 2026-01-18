#!/usr/bin/env python3
"""Test format registry and module structure."""

from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.formats import (
    detect_format,
    get_reader,
    get_writer,
    list_readers,
    list_writers,
)


class TestFormatRegistry:
    """Test format registration system."""

    def test_list_readers(self):
        """Test listing all registered readers."""
        readers = list_readers()
        assert isinstance(readers, list)
        assert len(readers) > 0
        assert "srt" in readers
        assert "vtt" in readers
        assert "gemini" in readers

    def test_list_writers(self):
        """Test listing all registered writers."""
        writers = list_writers()
        assert isinstance(writers, list)
        assert len(writers) > 0
        # Standard formats
        assert "srt" in writers
        assert "vtt" in writers
        # Professional formats
        assert "avid_ds" in writers
        assert "fcpxml" in writers
        assert "premiere_xml" in writers
        assert "ttml" in writers

    def test_get_reader(self):
        """Test getting reader by format ID."""
        srt_reader = get_reader("srt")
        assert srt_reader is not None
        assert hasattr(srt_reader, "read")

    def test_get_writer(self):
        """Test getting writer by format ID."""
        srt_writer = get_writer("srt")
        assert srt_writer is not None
        assert hasattr(srt_writer, "write")

    def test_detect_format(self):
        """Test format detection from file path."""
        assert detect_format("test.srt") == "srt"
        assert detect_format("test.vtt") == "vtt"
        assert detect_format("test.SRT") == "srt"  # Case insensitive
        assert detect_format("gemini.md") == "gemini"


class TestFormatHandlers:
    """Test format handler implementations."""

    def test_srt_roundtrip(self, tmp_path):
        """Test SRT format roundtrip."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]
        caption = Caption.from_supervisions(supervisions)

        # Write and read back
        output_path = tmp_path / "test.srt"
        caption.write(output_path)

        caption2 = Caption.read(output_path)
        assert len(caption2.supervisions) == 2
        assert "Hello world" in caption2.supervisions[0].text

    def test_gemini_format(self, tmp_path):
        """Test Gemini format handling."""
        gemini_reader = get_reader("gemini")
        gemini_writer = get_writer("gemini")

        assert gemini_reader is not None
        assert gemini_writer is not None

    def test_professional_formats_available(self):
        """Test professional NLE formats are available."""
        professional_formats = ["avid_ds", "fcpxml", "premiere_xml", "audition_csv", "ttml"]
        writers = list_writers()

        for fmt in professional_formats:
            assert fmt in writers, f"Professional format '{fmt}' not registered"


class TestFormatIntegration:
    """Integration tests for format system."""

    def test_caption_write_with_format_detection(self, tmp_path):
        """Test Caption.write uses format detection."""
        supervisions = [Supervision(text="Test", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(supervisions)

        # Should detect format from extension
        srt_path = tmp_path / "test.srt"
        caption.write(srt_path)
        assert srt_path.exists()

    def test_to_bytes_all_formats(self):
        """Test to_bytes works for common formats."""
        supervisions = [Supervision(text="Test", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(supervisions)

        formats_to_test = ["srt", "vtt", "ass"]
        for fmt in formats_to_test:
            result = caption.to_bytes(output_format=fmt)
            assert isinstance(result, bytes)
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
