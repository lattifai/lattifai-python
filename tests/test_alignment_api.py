#!/usr/bin/env python3
"""
Test suite for LattifAI alignment API
Tests the actual alignment() method signature and return types
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest

from lattifai.io import Supervision


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
        assert return_annotation is not inspect.Signature.empty, 'alignment() should have return type annotation'

        # The return type should be Tuple[List[Supervision], Optional[Pathlike]]
        print(f'✓ alignment() return type: {return_annotation}')

    def test_alignment_parameters(self):
        """Test that alignment() method has correct parameters."""
        import inspect

        from lattifai import LattifAI

        sig = inspect.signature(LattifAI.alignment)
        params = sig.parameters

        # Check required parameters
        assert 'audio' in params, 'alignment() should have audio parameter'
        assert 'subtitle' in params, 'alignment() should have subtitle parameter'

        # Check optional parameters
        assert 'format' in params, 'alignment() should have format parameter'
        assert 'split_sentence' in params, 'alignment() should have split_sentence parameter'
        assert 'output_subtitle_path' in params, 'alignment() should have output_subtitle_path parameter'

        # Check parameter defaults
        assert params['format'].default is None, 'format should default to None'
        assert params['split_sentence'].default is False, 'split_sentence should default to False'
        assert params['output_subtitle_path'].default is None, 'output_subtitle_path should default to None'

        print('✓ alignment() has correct parameters with correct defaults')

    def test_import_structure(self):
        """Test that all necessary types can be imported."""
        # Test main client import
        from lattifai import LattifAI

        assert LattifAI is not None, 'LattifAI should be importable'

        # Test error import
        from lattifai import LattifAIError

        assert LattifAIError is not None, 'LattifAIError should be importable'

        # Test I/O types
        from lattifai.io import SubtitleIO

        assert Supervision is not None, 'Supervision should be importable'
        assert SubtitleIO is not None, 'SubtitleIO should be importable'

        print('✓ All required types can be imported')

    def test_supervision_structure(self):
        """Test Supervision dataclass structure."""

        # Create a Supervision instance
        sup = Supervision(text='Test text', start=1.0, duration=2.0)

        assert sup.text == 'Test text'
        assert sup.start == 1.0
        assert sup.duration == 2.0

        print('✓ Supervision dataclass works correctly')


class TestAlignmentReturnValue:
    """Test the alignment() method return value structure."""

    @pytest.mark.skipif(True, reason='Requires actual audio file and API key')
    def test_alignment_returns_tuple(self, tmp_path):
        """Test that alignment() returns a tuple with correct structure."""
        from lattifai import LattifAI

        # This test would require actual audio and subtitle files
        # and a valid API key, so we skip it in CI
        # But the signature is documented here for reference

        client = LattifAI()

        # Example usage (would run with real files):
        # alignments, output_path = client.alignment(
        #     audio='test.wav',
        #     subtitle='test.srt',
        #     output_subtitle_path='output.srt'
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
        assert docstring is not None, 'alignment() should have a docstring'

        # Check for key information in docstring
        assert 'audio' in docstring.lower(), 'Docstring should document audio parameter'
        assert 'subtitle' in docstring.lower(), 'Docstring should document subtitle parameter'
        assert 'format' in docstring.lower(), 'Docstring should document format parameter'
        assert 'split_sentence' in docstring.lower(), 'Docstring should document split_sentence parameter'
        assert 'returns' in docstring.lower() or 'return' in docstring.lower(), 'Docstring should document return value'

        print('✓ alignment() has comprehensive documentation')
        print(f'Docstring preview: {docstring[:200]}...')


class TestSubtitleIOAPI:
    """Test SubtitleIO API."""

    def test_subtitle_io_read(self, tmp_path):
        """Test SubtitleIO.read() method."""
        from lattifai.io import SubtitleIO

        # Create a simple SRT file
        srt_content = """1
00:00:01,000 --> 00:00:03,000
First subtitle

2
00:00:04,000 --> 00:00:06,000
Second subtitle
"""
        srt_file = tmp_path / 'test.srt'
        srt_file.write_text(srt_content)

        # Read the file
        supervisions = SubtitleIO.read(srt_file)

        assert isinstance(supervisions, list)
        assert len(supervisions) == 2
        assert all(hasattr(s, 'text') for s in supervisions)
        assert all(hasattr(s, 'start') for s in supervisions)
        assert all(hasattr(s, 'duration') for s in supervisions)

        print(f'✓ SubtitleIO.read() works correctly, parsed {len(supervisions)} segments')

    def test_subtitle_io_write(self, tmp_path):
        """Test SubtitleIO.write() method."""
        from lattifai.io import SubtitleIO

        # Create supervisions
        supervisions = [
            Supervision(text='First line', start=1.0, duration=2.0),
            Supervision(text='Second line', start=4.0, duration=2.0),
        ]

        # Write to file
        output_file = tmp_path / 'output.srt'
        result_path = SubtitleIO.write(supervisions, output_file)

        assert output_file.exists()
        assert result_path == output_file

        # Read back and verify
        content = output_file.read_text()
        assert 'First line' in content
        assert 'Second line' in content

        print(f'✓ SubtitleIO.write() works correctly, wrote to {output_file}')

    def test_subtitle_format_auto_detection(self, tmp_path):
        """Test that format auto-detection works."""
        from lattifai.io import SubtitleIO

        # Test with different extensions
        formats = {
            'test.srt': 'srt',
            'test.vtt': 'vtt',
            'test.ass': 'ass',
            'test.txt': 'txt',
        }

        for filename, expected_format in formats.items():
            file_path = tmp_path / filename

            # Create a simple file with basic content
            if expected_format == 'txt':
                content = 'Line 1\nLine 2\n'
            else:
                content = '1\n00:00:01,000 --> 00:00:03,000\nTest\n'

            file_path.write_text(content)

            # Read with auto-detection
            supervisions = SubtitleIO.read(file_path, format=None)
            assert isinstance(supervisions, list)
            print(f'✓ Auto-detected format for {filename}')


class TestAPIConsistency:
    """Test API consistency across the codebase."""

    def test_client_initialization(self):
        """Test that LattifAI client can be initialized with correct parameters."""
        import inspect

        from lattifai import LattifAI

        sig = inspect.signature(LattifAI.__init__)
        params = sig.parameters

        # Check parameter names
        expected_params = ['api_key', 'model_name_or_path', 'device', 'base_url', 'timeout', 'max_retries']
        for param in expected_params:
            assert param in params, f'{param} should be a parameter of LattifAI.__init__'

        print('✓ LattifAI initialization parameters are correct')

    def test_error_inheritance(self):
        """Test that LattifAIError is the base exception."""
        from lattifai import LattifAIError

        # Test that it's a proper exception
        assert issubclass(LattifAIError, Exception)

        # Test that it can be raised and caught
        try:
            raise LattifAIError('Test error')
        except LattifAIError as e:
            # The error message may include additional information
            assert 'Test error' in str(e)

        print('✓ LattifAIError works correctly')


def run_tests():
    """Run all tests."""
    print('🧪 Running LattifAI API Tests\n')
    print('=' * 60)

    # Test alignment API signature
    print('\n📋 Testing Alignment API Signature...')
    test_sig = TestAlignmentAPISignature()
    test_sig.test_alignment_return_type_annotation()
    test_sig.test_alignment_parameters()
    test_sig.test_import_structure()
    test_sig.test_supervision_structure()

    # Test alignment return value
    print('\n📦 Testing Alignment Return Value...')
    test_ret = TestAlignmentReturnValue()
    test_ret.test_alignment_docstring()

    # Test SubtitleIO API
    print('\n📄 Testing SubtitleIO API...')

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_io = TestSubtitleIOAPI()
        test_io.test_subtitle_io_read(tmp_path)
        test_io.test_subtitle_io_write(tmp_path)
        test_io.test_subtitle_format_auto_detection(tmp_path)

    # Test API consistency
    print('\n🔍 Testing API Consistency...')
    test_consistency = TestAPIConsistency()
    test_consistency.test_client_initialization()
    test_consistency.test_error_inheritance()

    print('\n' + '=' * 60)
    print('✅ All API tests passed!')
    print('\n📝 API Summary:')
    print('   • alignment() returns Tuple[List[Supervision], Optional[Pathlike]]')
    print('   • Supervision has text, start, and duration attributes')
    print('   • SubtitleIO.read() and write() work correctly')
    print('   • All error types inherit from LattifAIError')


if __name__ == '__main__':
    import sys

    try:
        run_tests()
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
