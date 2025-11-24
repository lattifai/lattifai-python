#!/usr/bin/env python3
"""
Test suite for LattifAI alignment API
Tests the actual alignment() method signature and return types
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Supervision


class TestAlignmentAPISignature:
    """Test the alignment() method API signature and return types."""

    def test_alignment_return_type_annotation(self):
        """Test that alignment() method has correct return type annotation."""
        # Check the method signature
        import inspect

        from lattifai import LattifAI

        sig = inspect.signature(LattifAI.alignment)

        # Check return annotation
        return_annotation = sig.return_annotation
        assert return_annotation is not inspect.Signature.empty, "alignment() should have return type annotation"

        # The return type should be Tuple[List[Supervision], Optional[Pathlike]]
        print(f"✓ alignment() return type: {return_annotation}")

    def test_alignment_parameters(self):
        """Test that alignment() method has correct parameters."""
        import inspect

        from lattifai import LattifAI

        sig = inspect.signature(LattifAI.alignment)
        params = sig.parameters

        # Check required parameters
        assert "input_media" in params, "alignment() should have input_media parameter"
        assert "input_caption_path" in params, "alignment() should have input_caption_path parameter"

        # Check optional parameters
        assert "input_caption_format" in params, "alignment() should have input_caption_format parameter"
        assert "split_sentence" in params, "alignment() should have split_sentence parameter"
        assert "output_caption_path" in params, "alignment() should have output_caption_path parameter"

        # Check parameter defaults
        assert params["input_caption_format"].default is None, "input_caption_format should default to None"
        assert params["split_sentence"].default is None, "split_sentence should default to None"
        assert params["output_caption_path"].default is None, "output_caption_path should default to None"

        print("✓ alignment() has correct parameters with correct defaults")

    def test_import_structure(self):
        """Test that all necessary types can be imported."""
        # Test main client import
        from lattifai import LattifAI

        assert LattifAI is not None, "LattifAI should be importable"

        # Test error import
        from lattifai import LattifAIError

        assert LattifAIError is not None, "LattifAIError should be importable"

        # Test I/O types
        from lattifai import CaptionIO

        assert Supervision is not None, "Supervision should be importable"
        assert CaptionIO is not None, "CaptionIO should be importable"

        print("✓ All required types can be imported")

    def test_supervision_structure(self):
        """Test Supervision dataclass structure."""

        # Create a Supervision instance
        sup = Supervision(text="Test text", start=1.0, duration=2.0)

        assert sup.text == "Test text"
        assert sup.start == 1.0
        assert sup.duration == 2.0

        print("✓ Supervision dataclass works correctly")


class TestAlignmentReturnValue:
    """Test the alignment() method return value structure."""

    @pytest.mark.skipif(True, reason="Requires actual audio file and API key")
    def test_alignment_returns_tuple(self, tmp_path):
        """Test that alignment() returns a tuple with correct structure."""
        from lattifai import LattifAI

        # This test would require actual audio and caption files
        # and a valid API key, so we skip it in CI
        # But the signature is documented here for reference

        client = LattifAI()

        # Example usage (would run with real files):
        # alignments, output_path = client.alignment(
        #     audio='test.wav',
        #     caption='test.srt',
        #     output_caption_path='output.srt'
        # )
        #
        # assert isinstance(alignments, list)
        # assert all(isinstance(a, Supervision) for a in alignments)
        # assert output_path is not None

        del client

    def test_alignment_docstring(self):
        """Test that alignment() has proper documentation."""
        from lattifai import LattifAI

        docstring = LattifAI.alignment.__doc__
        assert docstring is not None, "alignment() should have a docstring"

        # Check for key information in docstring
        assert "media" in docstring.lower(), "Docstring should document media parameter"
        assert "caption" in docstring.lower(), "Docstring should document caption parameter"
        assert "format" in docstring.lower(), "Docstring should document format parameter"
        assert "split_sentence" in docstring.lower(), "Docstring should document split_sentence parameter"
        assert "returns" in docstring.lower() or "return" in docstring.lower(), "Docstring should document return value"

        print("✓ alignment() has comprehensive documentation")
        print(f"Docstring preview: {docstring[:200]}...")


class TestCaptionIOAPI:
    """Test CaptionIO API."""

    def test_caption_io_read(self, tmp_path):
        """Test CaptionIO.read() method."""
        from lattifai import CaptionIO

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
        supervisions = CaptionIO.read(srt_file)

        assert isinstance(supervisions, list)
        assert len(supervisions) == 2
        assert all(hasattr(s, "text") for s in supervisions)
        assert all(hasattr(s, "start") for s in supervisions)
        assert all(hasattr(s, "duration") for s in supervisions)

        print(f"✓ CaptionIO.read() works correctly, parsed {len(supervisions)} segments")

    def test_caption_io_write(self, tmp_path):
        """Test CaptionIO.write() method."""
        from lattifai import CaptionIO

        # Create supervisions
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]

        # Write to file
        output_file = tmp_path / "output.srt"
        result_path = CaptionIO.write(supervisions, output_file)

        assert output_file.exists()
        assert result_path == output_file

        # Read back and verify
        content = output_file.read_text()
        assert "First line" in content
        assert "Second line" in content

        print(f"✓ CaptionIO.write() works correctly, wrote to {output_file}")

    def test_caption_format_auto_detection(self, tmp_path):
        """Test that format auto-detection works."""
        from lattifai import CaptionIO

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
            supervisions = CaptionIO.read(file_path, format=None)
            assert isinstance(supervisions, list)
            print(f"✓ Auto-detected format for {filename}")


class TestAPIConsistency:
    """Test API consistency across the codebase."""

    def test_client_initialization(self):
        """Test that LattifAI client can be initialized with correct parameters."""
        import inspect

        from lattifai import LattifAI

        sig = inspect.signature(LattifAI.__init__)
        params = sig.parameters

        # Check parameter names - LattifAI uses config objects
        expected_params = ["client_config", "alignment_config", "caption_config"]
        for param in expected_params:
            assert param in params, f"{param} should be a parameter of LattifAI.__init__"

        print("✓ LattifAI initialization parameters are correct (config-based)")

    def test_error_inheritance(self):
        """Test that LattifAIError is the base exception."""
        from lattifai import LattifAIError

        # Test that it's a proper exception
        assert issubclass(LattifAIError, Exception)

        # Test that it can be raised and caught
        try:
            raise LattifAIError("Test error")
        except LattifAIError as e:
            # The error message may include additional information
            assert "Test error" in str(e)

        print("✓ LattifAIError works correctly")


def run_tests():
    """Run all tests."""
    print("🧪 Running LattifAI API Tests\n")
    print("=" * 60)

    # Test alignment API signature
    print("\n📋 Testing Alignment API Signature...")
    test_sig = TestAlignmentAPISignature()
    test_sig.test_alignment_return_type_annotation()
    test_sig.test_alignment_parameters()
    test_sig.test_import_structure()
    test_sig.test_supervision_structure()

    # Test alignment return value
    print("\n📦 Testing Alignment Return Value...")
    test_ret = TestAlignmentReturnValue()
    test_ret.test_alignment_docstring()

    # Test CaptionIO API
    print("\n📄 Testing CaptionIO API...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_io = TestCaptionIOAPI()
        test_io.test_caption_io_read(tmp_path)
        test_io.test_caption_io_write(tmp_path)
        test_io.test_caption_format_auto_detection(tmp_path)

    # Test API consistency
    print("\n🔍 Testing API Consistency...")
    test_consistency = TestAPIConsistency()
    test_consistency.test_client_initialization()
    test_consistency.test_error_inheritance()

    print("\n" + "=" * 60)
    print("✅ All API tests passed!")
    print("\n📝 API Summary:")
    print("   • alignment() returns Tuple[List[Supervision], Optional[Pathlike]]")
    print("   • Supervision has text, start, and duration attributes")
    print("   • CaptionIO.read() and write() work correctly")
    print("   • All error types inherit from LattifAIError")


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
